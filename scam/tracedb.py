from dataclasses import dataclass, field
from .experiment import Experiment
from .series import Series
from .schema import (
    schema_fields, schema_to_json, schema_from_json,
    _to_hdf5_value, _from_hdf5_value,
)
import h5py
import numpy as np
import json
import warnings


@dataclass
class TraceDB:
    experiments: dict[str, Experiment] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.experiments.values())

    def __getitem__(self, key):
        return self.experiments[key]

    def __len__(self):
        return len(self.experiments)

    def add_experiment(self, experiment):
        if experiment.name in self.experiments:
            raise ValueError(f"Experiment '{experiment.name}' already exists")
        self.experiments[experiment.name] = experiment

    def remove_experiment(self, name):
        if name in self.experiments:
            return self.experiments.pop(name)
        raise KeyError(f"Experiment '{name}' not found")

    def save_hdf5(self, filename, mode='update', overwrite_ok=False):
        """Save database to HDF5 file."""
        import os

        if mode not in ['update', 'overwrite']:
            raise ValueError("mode must be 'update' or 'overwrite'")

        file_exists = os.path.exists(filename)

        if mode == 'overwrite' and file_exists and not overwrite_ok:
            raise ValueError(
                f"File '{filename}' already exists. Set overwrite_ok=True to overwrite or use mode='update' to merge."
            )

        if mode == 'update' and file_exists:
            existing_db = TraceDB.load_hdf5(filename)

            for exp in existing_db.experiments.values():
                for series in exp.series:
                    if series._source and not series.traces:
                        series.open_for_reading()
                        series.traces = list(series)
                        series.close_reading()
                        series._source = None

            for exp_name, experiment in self.experiments.items():
                if exp_name in existing_db.experiments:
                    existing_exp = existing_db.experiments[exp_name]
                    existing_series_names = {s.name for s in existing_exp.series}

                    for series in experiment.series:
                        if series.name in existing_series_names:
                            warnings.warn(
                                f"Series '{series.name}' in experiment '{exp_name}' already exists. "
                                f"Skipping to avoid data loss.",
                                UserWarning,
                            )
                        else:
                            existing_exp.series.append(series)
                else:
                    existing_db.experiments[exp_name] = experiment

            self._write_hdf5(filename, existing_db.experiments)
        else:
            self._write_hdf5(filename, self.experiments)

    def _write_hdf5(self, filename, experiments_dict):
        """Schema-driven HDF5 write."""
        with h5py.File(filename, 'w') as f:
            for exp_name, experiment in experiments_dict.items():
                exp_group = f.create_group(exp_name)

                for key, value in experiment.metadata.items():
                    if isinstance(value, (str, int, float)):
                        exp_group.attrs[key] = value

                for series in experiment.series:
                    series_group = exp_group.create_group(series.name)

                    for key, value in series.metadata.items():
                        if isinstance(value, (str, int, float)):
                            series_group.attrs[key] = value

                    sf = schema_fields(series.trace_type)
                    series_group.attrs['_scam_schema'] = schema_to_json(series.trace_type)

                    if not series.traces:
                        continue

                    # Collect field shapes from first trace
                    field_shapes = {}
                    for name, kind, _, _ in sf:
                        if kind == 'array':
                            val = getattr(series.traces[0], name)
                            field_shapes[name] = np.asarray(val).shape
                        elif kind == 'bytes':
                            val = getattr(series.traces[0], name)
                            field_shapes[name] = (len(bytes(val)),)

                    series_group.attrs['_scam_shapes'] = json.dumps(
                        {k: list(v) for k, v in field_shapes.items()}
                    )

                    # Build and write datasets
                    for fd in sf:
                        name, kind, np_dtype, py_type = fd
                        if kind == 'array':
                            data = np.array(
                                [_to_hdf5_value(fd, getattr(t, name)) for t in series.traces]
                            )
                            series_group.create_dataset(name, data=data)
                        elif kind == 'bytes':
                            data = np.array(
                                [_to_hdf5_value(fd, getattr(t, name)) for t in series.traces]
                            )
                            series_group.create_dataset(name, data=data)
                        elif kind == 'scalar':
                            if np_dtype is None:
                                # String type
                                vals = [_to_hdf5_value(fd, getattr(t, name)) for t in series.traces]
                                series_group.create_dataset(
                                    name, data=[v.encode('utf-8') if isinstance(v, str) else b'' for v in vals]
                                )
                            else:
                                vals = [_to_hdf5_value(fd, getattr(t, name)) for t in series.traces]
                                series_group.create_dataset(name, data=np.array(vals, dtype=np_dtype))

    @classmethod
    def load_hdf5(cls, filename, trace_types=None):
        """
        Load database from HDF5 file - ALWAYS lazy.

        Args:
            filename: HDF5 file path
            trace_types: Optional dict {series_name: dataclass_type} to specify
                         trace types for specific series. If None, uses stored schema.
        """
        if trace_types is None:
            trace_types = {}

        db = cls()

        with h5py.File(filename, 'r') as f:
            for exp_name in f.keys():
                exp_group = f[exp_name]

                experiment = Experiment(name=exp_name, series=[])
                for attr_name in exp_group.attrs:
                    experiment.metadata[attr_name] = exp_group.attrs[attr_name]

                for series_name in exp_group.keys():
                    series_group = exp_group[series_name]

                    # Determine trace type
                    tt = trace_types.get(series_name, None)

                    series = Series(name=series_name, traces=[], trace_type=tt)
                    series._source = (filename, exp_name)

                    for attr_name in series_group.attrs:
                        if attr_name not in ['measurement_id', '_scam_schema', '_scam_shapes']:
                            series.metadata[attr_name] = series_group.attrs[attr_name]

                    experiment.series.append(series)

                db.experiments[exp_name] = experiment

        return db

    def get_or_create_experiment(self, name, metadata=None):
        if name not in self.experiments:
            exp = Experiment(name=name, series=[])
            if metadata:
                exp.metadata.update(metadata)
            self.experiments[name] = exp
        else:
            warnings.warn(f"Experiment '{name}' already exists, returning existing experiment", UserWarning)
        return self.experiments[name]
