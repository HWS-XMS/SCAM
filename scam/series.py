from dataclasses import dataclass, field
from enum import Enum
from .trace import Trace
import h5py
import numpy as np
from datetime import datetime
import uuid

# Global session UUID - generated once per Python session
_SESSION_UUID = str(uuid.uuid4())


def get_session_uuid():
    """Get the current session UUID."""
    return _SESSION_UUID


def new_session_uuid():
    """Generate a new session UUID and set it as current."""
    global _SESSION_UUID
    _SESSION_UUID = str(uuid.uuid4())
    return _SESSION_UUID


class SeriesMode(Enum):
    MEMORY = "memory"      # In-memory only
    WRITING = "writing"    # Streaming to HDF5
    READING = "reading"    # Lazy reading from HDF5


@dataclass
class Series:
    name: str
    traces: list[Trace] = field(default_factory=list)
    metadata: dict[str, any] = field(default_factory=dict)
    
    # Internal state
    _mode: SeriesMode = field(default=SeriesMode.MEMORY, init=False)
    _h5file: any = field(default=None, init=False)
    _h5group: any = field(default=None, init=False)
    _source: tuple = field(default=None, init=False)  # (filename, exp_name)
    _sample_shape: tuple = field(default=None, init=False)
    
    def __post_init__(self):
        """Validate initial traces if provided."""
        if self.traces:
            expected_shape = self.traces[0].samples.shape
            for i, trace in enumerate(self.traces[1:], 1):
                if trace.samples.shape != expected_shape:
                    raise ValueError(f"Trace {i} has shape {trace.samples.shape}, expected {expected_shape}")
            self._sample_shape = expected_shape
    
    # ============ WRITE Mode (Always Streaming) ============
    
    def open_for_writing(self, filename, experiment_name, mode='w', chunk_size=100, experiment_metadata=None, 
                         measurement_id=None, confirm_append=True):
        """
        Open series for writing - ALWAYS streams to HDF5.
        
        Args:
            filename: HDF5 file to write to
            experiment_name: Name of experiment group
            mode: 'w' for new file, 'a' for append
            chunk_size: HDF5 chunk size for efficiency
            experiment_metadata: Optional experiment metadata
            measurement_id: Optional measurement session identifier. If None, uses session UUID
            confirm_append: If True, prompt for confirmation when appending to existing series
        """
        if self._mode != SeriesMode.MEMORY:
            raise RuntimeError(f"Cannot open for writing in {self._mode} mode")
        
        # Use session UUID if no measurement_id provided
        if measurement_id is None:
            measurement_id = _SESSION_UUID
        
        import os
        
        # Handle append mode
        if mode == 'a' and os.path.exists(filename):
            self._h5file = h5py.File(filename, 'a')
            
            if experiment_name in self._h5file:
                exp_group = self._h5file[experiment_name]
                if self.name in exp_group:
                    # Reopen existing series for appending - SAFETY CHECKS
                    self._h5group = exp_group[self.name]
                    existing_count = self._h5group.attrs.get('trace_count', 0)
                    
                    # Safety check: verify this is intentional
                    if existing_count > 0 and confirm_append:
                        self._verify_append_safety(filename, experiment_name, existing_count, measurement_id)
                    
                    if 'samples' in self._h5group and len(self._h5group['samples']) > 0:
                        self._sample_shape = self._h5group['samples'].shape[1:]
                else:
                    # New series in existing experiment
                    self._h5group = exp_group.create_group(self.name)
            else:
                # New experiment
                exp_group = self._h5file.create_group(experiment_name)
                if experiment_metadata:
                    for key, value in experiment_metadata.items():
                        if isinstance(value, (str, int, float)):
                            exp_group.attrs[key] = value
                self._h5group = exp_group.create_group(self.name)
        else:
            # Create new file
            self._h5file = h5py.File(filename, 'w')
            exp_group = self._h5file.create_group(experiment_name)
            if experiment_metadata:
                for key, value in experiment_metadata.items():
                    if isinstance(value, (str, int, float)):
                        exp_group.attrs[key] = value
            self._h5group = exp_group.create_group(self.name)
        
        # Store series metadata
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float)):
                self._h5group.attrs[key] = value
        
        # Store measurement session info for safety
        if measurement_id:
            self._h5group.attrs['measurement_id'] = measurement_id
        
        # Initialize datasets if we know the shape and they don't exist yet
        if self._sample_shape and 'samples' not in self._h5group:
            self._create_datasets(chunk_size)
        
        # Persist existing traces if any
        current_count = self._h5group.attrs.get('trace_count', 0)
        for i, trace in enumerate(self.traces):
            self._write_trace(trace, current_count + i)
        
        if self.traces:
            self._h5group.attrs['trace_count'] = current_count + len(self.traces)
        
        self._mode = SeriesMode.WRITING
        self._source = (filename, experiment_name)
        return self
    
    def add_trace(self, trace):
        """Add trace to series. Auto-persists in WRITING mode."""
        if self._mode == SeriesMode.READING:
            raise RuntimeError("Cannot add traces in READING mode")
        
        # Validate shape
        if self.traces and trace.samples.shape != self.traces[0].samples.shape:
            raise ValueError(f"Shape mismatch: {trace.samples.shape} vs {self.traces[0].samples.shape}")
        
        if not self._sample_shape:
            self._sample_shape = trace.samples.shape
        
        # Add to memory
        self.traces.append(trace)
        
        # Auto-persist if writing
        if self._mode == SeriesMode.WRITING:
            # Create datasets on first trace
            if len(self.traces) == 1 and 'samples' not in self._h5group:
                self._create_datasets(100)
            
            # Write to disk
            index = self._h5group.attrs.get('trace_count', 0)
            self._write_trace(trace, index)
            self._h5group.attrs['trace_count'] = index + 1
            
            # Auto-flush every 50 traces
            if (index + 1) % 50 == 0:
                self._h5file.flush()
    
    def close_writing(self):
        """Close write mode and finalize datasets."""
        if self._mode == SeriesMode.WRITING:
            count = self._h5group.attrs.get('trace_count', 0)
            if count > 0 and 'samples' in self._h5group:
                # Trim to actual size
                self._h5group['samples'].resize((count,) + self._sample_shape)
                for name in ['timestamps', 'stimuli', 'responses', 'keys']:
                    self._h5group[name].resize((count,))
            
            self._h5file.close()
            self._h5file = None
            self._h5group = None
            self._mode = SeriesMode.MEMORY
        return self
    
    # ============ READ Mode (Always Lazy) ============
    
    def open_for_reading(self, filename=None, experiment_name=None):
        """Open series for lazy reading from HDF5."""
        if self._mode != SeriesMode.MEMORY:
            raise RuntimeError(f"Cannot open for reading in {self._mode} mode")
        
        if filename is None and self._source:
            filename, experiment_name = self._source
        elif filename is None:
            raise ValueError("No source available")
        
        self._h5file = h5py.File(filename, 'r')
        self._h5group = self._h5file[experiment_name][self.name]
        self._mode = SeriesMode.READING
        self._source = (filename, experiment_name)
        
        # Clear memory traces for lazy-only access
        self.traces = []
        
        if 'samples' in self._h5group and len(self._h5group['samples']) > 0:
            self._sample_shape = self._h5group['samples'].shape[1:]
        
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
        """Iterate over traces - lazy in READ mode."""
        if self._mode == SeriesMode.READING:
            count = len(self)  # Use __len__ which handles all cases
            for i in range(count):
                yield self._read_trace(i)
        else:
            yield from self.traces
    
    def __len__(self):
        """Get number of traces."""
        if self._mode == SeriesMode.READING:
            # Try trace_count first, then check if samples exists
            if 'trace_count' in self._h5group.attrs:
                return self._h5group.attrs['trace_count']
            elif 'samples' in self._h5group:
                return len(self._h5group['samples'])
            else:
                return 0
        return len(self.traces)
    
    def __getitem__(self, index):
        """Get trace by index - lazy in READ mode."""
        if self._mode == SeriesMode.READING:
            if isinstance(index, slice):
                start, stop, step = index.indices(len(self))
                return [self._read_trace(i) for i in range(start, stop, step)]
            return self._read_trace(index)
        return self.traces[index]
    
    def to_matrix(self, dtype=np.float32):
        """Build matrix of all samples - efficient chunked loading in READ mode."""
        if self._mode == SeriesMode.READING:
            count = len(self)
            if count == 0:
                return np.array([])
            
            matrix = np.empty((count,) + self._sample_shape, dtype=dtype)
            
            # Load in 1000-trace chunks
            for i in range(0, count, 1000):
                end = min(i + 1000, count)
                matrix[i:end] = self._h5group['samples'][i:end]
            return matrix
        else:
            if not self.traces:
                return np.array([])
            return np.array([t.samples for t in self.traces], dtype=dtype)
    
    def remove_trace(self, index):
        """Remove trace (only in MEMORY mode)."""
        if self._mode != SeriesMode.MEMORY:
            raise RuntimeError(f"Cannot remove traces in {self._mode} mode")
        
        if 0 <= index < len(self.traces):
            return self.traces.pop(index)
        raise IndexError(f"Index {index} out of range")
    
    # ============ Internal Helpers ============
    
    def _create_datasets(self, chunk_size):
        """Create HDF5 datasets for streaming."""
        initial_size = 1000
        
        self._h5group.create_dataset(
            'samples',
            shape=(initial_size,) + self._sample_shape,
            maxshape=(None,) + self._sample_shape,
            chunks=(chunk_size,) + self._sample_shape,
            dtype=np.float32
        )
        
        string_dt = h5py.string_dtype()
        for name in ['timestamps', 'stimuli', 'responses', 'keys']:
            self._h5group.create_dataset(
                name,
                shape=(initial_size,),
                maxshape=(None,),
                chunks=(chunk_size,),
                dtype=string_dt
            )
        
        self._h5group.attrs['trace_count'] = 0
    
    def _write_trace(self, trace, index):
        """Write single trace to HDF5."""
        # Resize if needed
        if index >= self._h5group['samples'].shape[0]:
            new_size = index + 1000
            self._h5group['samples'].resize((new_size,) + self._sample_shape)
            for name in ['timestamps', 'stimuli', 'responses', 'keys']:
                self._h5group[name].resize((new_size,))
        
        self._h5group['samples'][index] = trace.samples
        self._h5group['timestamps'][index] = trace.timestamp.isoformat()
        self._h5group['stimuli'][index] = str(trace.stimulus) if trace.stimulus else ""
        self._h5group['responses'][index] = str(trace.response) if trace.response else ""
        self._h5group['keys'][index] = str(trace.key) if trace.key else ""
    
    def _read_trace(self, index):
        """Read single trace from HDF5."""
        return Trace(
            samples=self._h5group['samples'][index],
            timestamp=datetime.fromisoformat(self._h5group['timestamps'][index].decode()),
            stimulus=self._h5group['stimuli'][index].decode() or None,
            response=self._h5group['responses'][index].decode() or None,
            key=self._h5group['keys'][index].decode() or None
        )
    
    def _verify_append_safety(self, filename, experiment_name, existing_count, measurement_id):
        """Verify that appending to existing series is intentional."""
        # Check measurement ID match if provided
        existing_measurement_id = self._h5group.attrs.get('measurement_id', None)
        if existing_measurement_id:
            existing_measurement_id = existing_measurement_id.decode() if isinstance(existing_measurement_id, bytes) else str(existing_measurement_id)
        
        # If measurement IDs match, allow seamless continuation
        if measurement_id and existing_measurement_id and measurement_id == existing_measurement_id:
            return  # Same session - no confirmation needed
        
        # If measurement IDs don't match, raise error
        if measurement_id and existing_measurement_id and measurement_id != existing_measurement_id:
            raise ValueError(
                f"Measurement ID mismatch: existing='{existing_measurement_id}', specified='{measurement_id}'. "
                f"Use confirm_append=False to bypass."
            )
        
        # If no measurement ID comparison possible, prompt for confirmation
        print(f"Experiment: existent, Series: existent ({existing_count} traces), continue? [y/N]: ", end="")
        response = input().strip().lower()
        if response not in ['y', 'yes']:
            raise RuntimeError("Measurement cancelled by user.")
    
