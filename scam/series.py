from dataclasses import dataclass, field
from enum import Enum
from .schema import (
    schema_fields, schema_to_json, schema_from_json, validate_schema_match,
    make_trace_type, _to_hdf5_value, _from_hdf5_value, SchemaError,
)
import h5py
import numpy as np
import json
import os
import uuid

# Global session UUID - generated once per Python session
_SESSION_UUID = str(uuid.uuid4())

# Shared file handle registry: absolute path -> (h5py.File, ref_count)
_open_files: dict[str, tuple] = {}


def _acquire_file(filename, h5mode):
    """Get or create a shared h5py.File handle. Ref-counted."""
    key = os.path.abspath(filename)
    if key in _open_files:
        fh, count = _open_files[key]
        _open_files[key] = (fh, count + 1)
        return fh
    fh = h5py.File(filename, h5mode, libver='latest')
    _open_files[key] = (fh, 1)
    return fh


def _release_file(filename):
    """Decrement ref count; close file when last user releases."""
    key = os.path.abspath(filename)
    if key not in _open_files:
        return
    fh, count = _open_files[key]
    if count <= 1:
        fh.close()
        del _open_files[key]
    else:
        _open_files[key] = (fh, count - 1)


def get_session_uuid():
    """Get the current session UUID."""
    return _SESSION_UUID


def new_session_uuid():
    """Generate a new session UUID and set it as current."""
    global _SESSION_UUID
    _SESSION_UUID = str(uuid.uuid4())
    return _SESSION_UUID


class SeriesMode(Enum):
    MEMORY = "memory"
    WRITING = "writing"
    READING = "reading"


@dataclass
class Series:
    name: str
    traces: list = field(default_factory=list)
    metadata: dict[str, any] = field(default_factory=dict)
    trace_type: type = field(default=None)

    # Internal state
    _mode: SeriesMode = field(default=SeriesMode.MEMORY, init=False)
    _h5file: any = field(default=None, init=False)
    _h5group: any = field(default=None, init=False)
    _source: tuple = field(default=None, init=False)
    _sample_shape: tuple = field(default=None, init=False)
    _field_shapes: dict = field(default=None, init=False)
    _schema_fields: list = field(default=None, init=False)
    _trace_count: int = field(default=0, init=False)

    def __post_init__(self):
        if self.trace_type is not None:
            self._schema_fields = schema_fields(self.trace_type)
        else:
            self._schema_fields = None  # deferred until open_for_reading

        if self.traces:
            if self.trace_type is None:
                raise TypeError("trace_type required when constructing Series with traces")
            first = self.traces[0]
            if type(first) is not self.trace_type:
                raise TypeError(f"Expected {self.trace_type.__name__}, got {type(first).__name__}")
            self._field_shapes = self._extract_shapes(first)
            for i, t in enumerate(self.traces[1:], 1):
                if type(t) is not self.trace_type:
                    raise TypeError(f"Trace {i}: expected {self.trace_type.__name__}, got {type(t).__name__}")
                self._validate_shapes(t)
            self._sample_shape = self._get_first_array_shape()

    def _extract_shapes(self, trace):
        """Extract shapes of all Array and bytes fields from a trace instance."""
        shapes = {}
        for name, kind, _, _ in self._schema_fields:
            if kind == 'array':
                val = getattr(trace, name)
                shapes[name] = np.asarray(val).shape
            elif kind == 'bytes':
                val = getattr(trace, name)
                shapes[name] = (len(bytes(val)),)
        return shapes

    def _validate_shapes(self, trace):
        """Validate that a trace's field shapes match the locked shapes."""
        if self._field_shapes is None:
            return
        for name, kind, _, _ in self._schema_fields:
            if kind == 'array':
                val = getattr(trace, name)
                actual = np.asarray(val).shape
            elif kind == 'bytes':
                val = getattr(trace, name)
                actual = (len(bytes(val)),)
            else:
                continue
            expected = self._field_shapes[name]
            if actual != expected:
                raise ValueError(
                    f"Field '{name}' shape {actual} != locked shape {expected}"
                )

    def _get_first_array_shape(self):
        """Get the shape of the first Array field from locked shapes."""
        if self._field_shapes is None:
            return None
        for name, kind, _, _ in self._schema_fields:
            if kind == 'array':
                return self._field_shapes.get(name)
        return None

    # ============ WRITE Mode (Always Streaming) ============

    def open_for_writing(self, filename, experiment_name, mode='auto', chunk_size=100,
                         experiment_metadata=None, measurement_id=None, confirm_append=True):
        """Open series for writing - ALWAYS streams to HDF5."""
        if self._mode != SeriesMode.MEMORY:
            raise RuntimeError(f"Cannot open for writing in {self._mode} mode")

        if self.trace_type is None:
            raise TypeError("trace_type is required for writing")
        if self._schema_fields is None:
            self._schema_fields = schema_fields(self.trace_type)

        if measurement_id is None:
            measurement_id = _SESSION_UUID

        if os.path.exists(filename):
            if mode == 'w':
                raise ValueError(
                    f"File '{filename}' already exists. Cannot use mode='w' as it would overwrite all data. "
                    f"Use mode='auto' or mode='a' to append, or delete the file first."
                )
            elif mode not in ['auto', 'a']:
                raise ValueError(f"Invalid mode '{mode}'. Use 'auto', 'a', or 'w'.")

            self._h5file = _acquire_file(filename, 'a')

            if experiment_name in self._h5file:
                exp_group = self._h5file[experiment_name]
                if self.name in exp_group:
                    self._h5group = exp_group[self.name]

                    # Recover trace count from dataset shape
                    existing_count = self._dataset_trace_count()

                    if existing_count > 0 and confirm_append:
                        self._verify_append_safety(filename, experiment_name, existing_count, measurement_id)

                    self._trace_count = existing_count

                    # Recover schema and shapes from existing data
                    if '_scam_schema' in self._h5group.attrs:
                        validate_schema_match(self.trace_type, self._h5group.attrs['_scam_schema'])
                    if '_scam_shapes' in self._h5group.attrs:
                        self._field_shapes = json.loads(self._h5group.attrs['_scam_shapes'])
                        self._field_shapes = {k: tuple(v) for k, v in self._field_shapes.items()}
                        self._sample_shape = self._get_first_array_shape()
                else:
                    self._h5group = exp_group.create_group(self.name)
            else:
                exp_group = self._h5file.create_group(experiment_name)
                if experiment_metadata:
                    for key, value in experiment_metadata.items():
                        if isinstance(value, (str, int, float)):
                            exp_group.attrs[key] = value
                self._h5group = exp_group.create_group(self.name)
        else:
            if mode == 'a':
                raise ValueError(f"File '{filename}' doesn't exist. Cannot use mode='a' on non-existent file.")
            self._h5file = _acquire_file(filename, 'w')
            exp_group = self._h5file.create_group(experiment_name)
            if experiment_metadata:
                for key, value in experiment_metadata.items():
                    if isinstance(value, (str, int, float)):
                        exp_group.attrs[key] = value
            self._h5group = exp_group.create_group(self.name)

        # Store series metadata as attributes (before SWMR)
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float)):
                self._h5group.attrs[key] = value

        if measurement_id:
            self._h5group.attrs['measurement_id'] = measurement_id

        self._h5group.attrs['_scam_schema'] = schema_to_json(self.trace_type)

        # Persist existing in-memory traces
        for i, trace in enumerate(self.traces):
            if self._field_shapes is None:
                self._field_shapes = self._extract_shapes(trace)
                self._sample_shape = self._get_first_array_shape()
            if not self._datasets_exist():
                self._create_datasets(chunk_size)
            self._write_trace(trace, self._trace_count + i)

        if self.traces:
            self._trace_count += len(self.traces)

        self._mode = SeriesMode.WRITING
        self._source = (filename, experiment_name)
        return self

    def add_trace(self, trace):
        """Add trace to series. Auto-persists in WRITING mode."""
        if self._mode == SeriesMode.READING:
            raise RuntimeError("Cannot add traces in READING mode")

        if self.trace_type is None:
            self.trace_type = type(trace)
            self._schema_fields = schema_fields(self.trace_type)
        elif type(trace) is not self.trace_type:
            raise TypeError(f"Expected {self.trace_type.__name__}, got {type(trace).__name__}")

        # Lock shapes on first trace
        if self._field_shapes is None:
            self._field_shapes = self._extract_shapes(trace)
            self._sample_shape = self._get_first_array_shape()
        else:
            self._validate_shapes(trace)

        if self._mode == SeriesMode.MEMORY:
            self.traces.append(trace)
            return

        if self._mode == SeriesMode.WRITING:
            if not self._datasets_exist():
                self._create_datasets(100)

            self._write_trace(trace, self._trace_count)
            self._trace_count += 1

            # Resize datasets to exact trace count (makes shape = count)
            self._trim_datasets(self._trace_count)

            if self._trace_count % 50 == 0:
                self._h5file.flush()

    def enable_swmr(self):
        """Enable SWMR mode for concurrent reading.

        Call after ALL series sharing this file have created their datasets
        (i.e., after each has written at least one trace). Once enabled:
        - Readers can open the file concurrently with swmr=True
        - No new groups, datasets, or attributes can be created
        - Only dataset writes, resizes, and flushes are allowed
        """
        if self._mode != SeriesMode.WRITING:
            raise RuntimeError("Can only enable SWMR in WRITING mode")
        if self._h5file.swmr_mode:
            return

        # Store shapes before SWMR (no attribute writes after)
        if self._field_shapes:
            shapes_json = json.dumps({k: list(v) for k, v in self._field_shapes.items()})
            self._h5group.attrs['_scam_shapes'] = shapes_json

        self._h5file.swmr_mode = True

    def close_writing(self):
        """Close write mode and finalize datasets."""
        if self._mode == SeriesMode.WRITING:
            if self._trace_count > 0 and self._datasets_exist():
                # Store shapes (only possible if SWMR is not active)
                if self._field_shapes and not self._h5file.swmr_mode:
                    shapes_json = json.dumps({k: list(v) for k, v in self._field_shapes.items()})
                    self._h5group.attrs['_scam_shapes'] = shapes_json

                # Trim datasets to exact count (should already be exact, but ensure)
                self._trim_datasets(self._trace_count)

            self._h5file.flush()
            _release_file(self._source[0])
            self._h5file = None
            self._h5group = None
            self._mode = SeriesMode.MEMORY
        return self

    # ============ READ Mode (Always Lazy) ============

    def open_for_reading(self, filename=None, experiment_name=None, swmr=False):
        """Open series for lazy reading from HDF5.

        Args:
            swmr: If True, open in SWMR mode for concurrent reading while
                  another process writes. Use refresh() to see new data.
        """
        if self._mode != SeriesMode.MEMORY:
            raise RuntimeError(f"Cannot open for reading in {self._mode} mode")

        if filename is None and self._source:
            filename, experiment_name = self._source
        elif filename is None:
            raise ValueError("No source available")

        if swmr:
            self._h5file = h5py.File(filename, 'r', libver='latest', swmr=True)
        else:
            self._h5file = h5py.File(filename, 'r')
        self._h5group = self._h5file[experiment_name][self.name]
        self._mode = SeriesMode.READING
        self._source = (filename, experiment_name)

        self.traces = []

        # Recover schema fields for reading
        if '_scam_schema' in self._h5group.attrs:
            stored_json = self._h5group.attrs['_scam_schema']
            if self.trace_type is not None:
                validate_schema_match(self.trace_type, stored_json)
            else:
                self.trace_type = make_trace_type(stored_json)
            self._schema_fields = schema_fields(self.trace_type)
        else:
            if self.trace_type is None:
                raise SchemaError("No _scam_schema in file and no trace_type provided")

        # Recover shapes
        if '_scam_shapes' in self._h5group.attrs:
            self._field_shapes = json.loads(self._h5group.attrs['_scam_shapes'])
            self._field_shapes = {k: tuple(v) for k, v in self._field_shapes.items()}
        self._sample_shape = self._get_first_array_shape()

        return self

    def close_reading(self):
        """Close read mode."""
        if self._mode == SeriesMode.READING:
            self._h5file.close()
            self._h5file = None
            self._h5group = None
            self._mode = SeriesMode.MEMORY
        return self

    # ============ Unified Interface ============

    def __iter__(self):
        if self._mode == SeriesMode.READING:
            count = len(self)
            for i in range(count):
                yield self._read_trace(i)
        elif self._source and len(self.traces) == 0:
            raise RuntimeError(
                f"Series '{self.name}' was loaded from HDF5 but not opened for reading. "
                f"Call series.open_for_reading() first to access trace data."
            )
        else:
            yield from self.traces

    def refresh(self):
        """Refresh datasets to see latest data from SWMR writer.

        Only effective when opened with swmr=True.
        """
        if self._mode != SeriesMode.READING:
            return
        for name, _, _, _ in self._schema_fields:
            if name in self._h5group:
                self._h5group[name].id.refresh()

    def __len__(self):
        if self._mode == SeriesMode.READING:
            return self._dataset_trace_count()
        if self._mode == SeriesMode.WRITING:
            return self._trace_count
        return len(self.traces)

    def __getitem__(self, index):
        if self._mode == SeriesMode.READING:
            if isinstance(index, slice):
                start, stop, step = index.indices(len(self))
                return [self._read_trace(i) for i in range(start, stop, step)]
            return self._read_trace(index)
        elif self._source and len(self.traces) == 0:
            raise RuntimeError(
                f"Series '{self.name}' was loaded from HDF5 but not opened for reading. "
                f"Call series.open_for_reading() first to access trace data."
            )
        return self.traces[index]

    def to_matrix(self, dtype=None, field_name=None):
        """Build matrix of all samples from the first (or named) Array/bytes field."""
        target = None
        for name, kind, np_dtype, _ in self._schema_fields:
            if kind in ('array', 'bytes'):
                if field_name is None or name == field_name:
                    target = (name, np_dtype)
                    break
        if target is None:
            raise ValueError(f"No Array/bytes field found" + (f" named '{field_name}'" if field_name else ""))

        ds_name, default_dtype = target
        if dtype is None:
            dtype = default_dtype

        if self._mode == SeriesMode.READING:
            count = len(self)
            if count == 0:
                return np.array([])
            shape = self._h5group[ds_name].shape[1:]
            matrix = np.empty((count,) + shape, dtype=dtype)
            for i in range(0, count, 1000):
                end = min(i + 1000, count)
                matrix[i:end] = self._h5group[ds_name][i:end]
            return matrix
        else:
            if not self.traces:
                return np.array([])
            return np.array(
                [np.asarray(getattr(t, ds_name)) for t in self.traces], dtype=dtype
            )

    def remove_trace(self, index):
        """Remove trace (only in MEMORY mode)."""
        if self._mode != SeriesMode.MEMORY:
            raise RuntimeError(f"Cannot remove traces in {self._mode} mode")
        if 0 <= index < len(self.traces):
            return self.traces.pop(index)
        raise IndexError(f"Index {index} out of range")

    # ============ Internal Helpers ============

    def _dataset_trace_count(self):
        """Derive trace count from the first dataset's shape[0]."""
        for name, _, _, _ in self._schema_fields:
            if name in self._h5group:
                return self._h5group[name].shape[0]
        return 0

    def _datasets_exist(self):
        """Check if HDF5 datasets have been created for this series."""
        if self._h5group is None:
            return False
        return self._schema_fields[0][0] in self._h5group

    def _create_datasets(self, chunk_size):
        """Create HDF5 datasets based on schema fields. Start at size 0."""
        for name, kind, np_dtype, py_type in self._schema_fields:
            if kind == 'array':
                shape = self._field_shapes[name]
                self._h5group.create_dataset(
                    name,
                    shape=(0,) + shape,
                    maxshape=(None,) + shape,
                    chunks=(chunk_size,) + shape,
                    dtype=np_dtype,
                )
            elif kind == 'bytes':
                shape = self._field_shapes[name]
                self._h5group.create_dataset(
                    name,
                    shape=(0,) + shape,
                    maxshape=(None,) + shape,
                    chunks=(chunk_size,) + shape,
                    dtype=np.uint8,
                )
            elif kind == 'scalar':
                if np_dtype is None:
                    dt = h5py.string_dtype()
                else:
                    dt = np_dtype
                self._h5group.create_dataset(
                    name,
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(chunk_size,),
                    dtype=dt,
                )

    def _write_trace(self, trace, index):
        """Write single trace to HDF5. Resizes datasets to fit."""
        first_ds_name = self._schema_fields[0][0]
        current_size = self._h5group[first_ds_name].shape[0]
        if index >= current_size:
            new_size = index + 1
            for name, kind, _, _ in self._schema_fields:
                ds = self._h5group[name]
                if kind in ('array', 'bytes'):
                    shape = self._field_shapes[name]
                    ds.resize((new_size,) + shape)
                else:
                    ds.resize((new_size,))

        for fd in self._schema_fields:
            name = fd[0]
            value = getattr(trace, name)
            hdf5_val = _to_hdf5_value(fd, value)
            self._h5group[name][index] = hdf5_val

    def _trim_datasets(self, count):
        """Resize all datasets to exactly count traces."""
        for name, kind, _, _ in self._schema_fields:
            if name not in self._h5group:
                continue
            ds = self._h5group[name]
            if kind in ('array', 'bytes'):
                shape = self._field_shapes[name]
                target = (count,) + shape
            else:
                target = (count,)
            if ds.shape != target:
                ds.resize(target)

    def _read_trace(self, index):
        """Read single trace from HDF5 and construct a trace_type instance."""
        kwargs = {}
        for fd in self._schema_fields:
            name = fd[0]
            hdf5_val = self._h5group[name][index]
            kwargs[name] = _from_hdf5_value(fd, hdf5_val)

        return self.trace_type(**kwargs)

    def _verify_append_safety(self, filename, experiment_name, existing_count, measurement_id):
        """Verify that appending to existing series is intentional."""
        existing_measurement_id = self._h5group.attrs.get('measurement_id', None)
        if existing_measurement_id:
            existing_measurement_id = (
                existing_measurement_id.decode()
                if isinstance(existing_measurement_id, bytes)
                else str(existing_measurement_id)
            )

        if measurement_id and existing_measurement_id and measurement_id == existing_measurement_id:
            return

        if measurement_id and existing_measurement_id and measurement_id != existing_measurement_id:
            raise ValueError(
                f"Measurement ID mismatch: existing='{existing_measurement_id}', specified='{measurement_id}'. "
                f"Use confirm_append=False to bypass."
            )

        print(f"Experiment: existent, Series: existent ({existing_count} traces), continue? [y/N]: ", end="")
        response = input().strip().lower()
        if response not in ['y', 'yes']:
            raise RuntimeError("Measurement cancelled by user.")
