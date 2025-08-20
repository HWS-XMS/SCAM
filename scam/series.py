from dataclasses import dataclass, field
from .trace import Trace
import h5py
import numpy as np
import warnings


@dataclass
class Series:
    name: str
    traces: list[Trace]
    metadata: dict[str, any] = field(default_factory=dict)
    _hdf5_file: any = field(default=None, init=False)
    _hdf5_group: any = field(default=None, init=False)
    _auto_flush_interval: int = field(default=50, init=False)
    _traces_since_flush: int = field(default=0, init=False)
    
    def __post_init__(self):
        if self.traces:
            expected_shape = self.traces[0].samples.shape
            for i, trace in enumerate(self.traces[1:], 1):
                if trace.samples.shape != expected_shape:
                    raise ValueError(f"Trace {i} has shape {trace.samples.shape}, expected {expected_shape}")
    
    def __iter__(self):
        return iter(self.traces)
    
    def __getitem__(self, index):
        return self.traces[index]
    
    def __len__(self):
        return len(self.traces)
    
    def add_trace(self, trace, persist=False, flush=None):
        """
        Add a trace to the series.
        
        Args:
            trace: The Trace object to add
            persist: If True and streaming is active, immediately persist to HDF5
            flush: If True, force flush to disk. If None, auto-flush based on interval
        """
        # Validate trace shape
        if self.traces and trace.samples.shape != self.traces[0].samples.shape:
            raise ValueError(f"Trace has shape {trace.samples.shape}, expected {self.traces[0].samples.shape}")
        
        # Add to in-memory list
        self.traces.append(trace)
        
        # Handle streaming persistence if enabled
        if persist and self.is_streaming():
            self._persist_trace_to_stream(trace)
            
            # Handle flushing
            self._traces_since_flush += 1
            if flush is True or (flush is None and self._traces_since_flush >= self._auto_flush_interval):
                self.flush_stream()
                self._traces_since_flush = 0
        elif persist and not self.is_streaming():
            warnings.warn("persist=True but no stream is open. Call enable_streaming() first.", UserWarning)
    
    def remove_trace(self, index):
        if 0 <= index < len(self.traces):
            return self.traces.pop(index)
        raise IndexError(f"Trace index {index} out of range")
    
    def is_streaming(self):
        """Check if streaming to HDF5 is currently active."""
        return self._hdf5_file is not None
    
    def enable_streaming(self, filename, experiment_name, mode='w', auto_flush_interval=50, initial_size=1000, chunk_size=100, sample_shape=None, experiment_metadata=None):
        """
        Enable streaming mode for automatic HDF5 persistence.
        
        Args:
            filename: HDF5 file to stream to
            experiment_name: Name of the experiment group
            mode: 'w' to create new file, 'a' to append to existing file
            auto_flush_interval: Number of traces between auto-flushes (default 50)
            initial_size: Initial dataset size (default 1000)
            chunk_size: HDF5 chunk size (default 100)
            sample_shape: Shape of trace samples (inferred from first trace if None)
            experiment_metadata: Metadata for the experiment
        """
        self._auto_flush_interval = auto_flush_interval
        self._traces_since_flush = 0
        return self._open_hdf5_stream(filename, experiment_name, mode, initial_size, chunk_size, sample_shape, experiment_metadata)
    
    def open_hdf5_stream(self, filename, experiment_name, initial_size=1000, chunk_size=100, sample_shape=None, experiment_metadata=None):
        """Legacy method - use enable_streaming() instead."""
        warnings.warn("open_hdf5_stream() is deprecated. Use enable_streaming() instead.", DeprecationWarning)
        return self.enable_streaming(filename, experiment_name, 'w', 50, initial_size, chunk_size, sample_shape, experiment_metadata)
    
    def _open_hdf5_stream(self, filename, experiment_name, mode='w', initial_size=1000, chunk_size=100, sample_shape=None, experiment_metadata=None):
        import os
        
        # Handle append mode
        if mode == 'a' and os.path.exists(filename):
            self._hdf5_file = h5py.File(filename, 'a')
            
            # Check if experiment exists
            if experiment_name in self._hdf5_file:
                exp_group = self._hdf5_file[experiment_name]
                
                # Check if series exists
                if self.name in exp_group:
                    # Reopen existing series for appending
                    self._hdf5_group = exp_group[self.name]
                    
                    # Get the current trace count to continue from there
                    existing_count = self._hdf5_group.attrs.get('trace_count', 0)
                    
                    # Load sample shape from existing data
                    if 'samples' in self._hdf5_group and len(self._hdf5_group['samples']) > 0:
                        self._sample_shape = self._hdf5_group['samples'].shape[1:]
                    else:
                        self._sample_shape = sample_shape
                    
                    return self
                else:
                    # Create new series in existing experiment
                    self._hdf5_group = exp_group.create_group(self.name)
            else:
                # Create new experiment
                exp_group = self._hdf5_file.create_group(experiment_name)
                if experiment_metadata:
                    for key, value in experiment_metadata.items():
                        if isinstance(value, (str, int, float)):
                            exp_group.attrs[key] = value
                self._hdf5_group = exp_group.create_group(self.name)
        else:
            # Create new file (mode='w' or file doesn't exist)
            self._hdf5_file = h5py.File(filename, 'w')
            exp_group = self._hdf5_file.create_group(experiment_name)
            
            # Store experiment metadata
            if experiment_metadata:
                for key, value in experiment_metadata.items():
                    if isinstance(value, (str, int, float)):
                        exp_group.attrs[key] = value
            
            self._hdf5_group = exp_group.create_group(self.name)
        
        # Store series metadata
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float)):
                self._hdf5_group.attrs[key] = value
        
        # Determine sample shape
        if self.traces:
            self._sample_shape = self.traces[0].samples.shape
        elif sample_shape:
            self._sample_shape = sample_shape
        else:
            self._sample_shape = None  # Will be set from first trace
        
        # Only create datasets if we know the shape
        if self._sample_shape:
            self._hdf5_group.create_dataset(
                'samples', 
                shape=(initial_size,) + self._sample_shape,
                maxshape=(None,) + self._sample_shape,
                chunks=(chunk_size,) + self._sample_shape,
                dtype=np.float32
            )
        
        # Create string datasets for metadata
        string_dt = h5py.string_dtype()
        for name in ['timestamps', 'stimuli', 'responses', 'keys']:
            self._hdf5_group.create_dataset(
                name,
                shape=(initial_size,),
                maxshape=(None,),
                chunks=(chunk_size,),
                dtype=string_dt
            )
        
        self._hdf5_group.attrs['trace_count'] = 0
        return self
    
    def append_trace_to_stream(self, trace):
        """Legacy method - use add_trace(trace, persist=True) instead."""
        warnings.warn("append_trace_to_stream() is deprecated. Use add_trace(trace, persist=True) instead.", DeprecationWarning)
        if not self._hdf5_file:
            raise RuntimeError("HDF5 stream not open. Call enable_streaming() first")
        # Use the new unified method without adding to list again
        self._persist_trace_to_stream(trace)
        self.traces.append(trace)
        return self
    
    def _persist_trace_to_stream(self, trace):
        """Internal method to persist a single trace to the HDF5 stream."""
        if not self._hdf5_file:
            raise RuntimeError("HDF5 stream not open. Call enable_streaming() first")
        
        # If this is the first trace and we didn't know the shape, create datasets now
        if self._sample_shape is None:
            self._sample_shape = trace.samples.shape
            self._hdf5_group.create_dataset(
                'samples', 
                shape=(1000,) + self._sample_shape,
                maxshape=(None,) + self._sample_shape,
                chunks=(100,) + self._sample_shape,
                dtype=np.float32
            )
        
        # Get current count
        count = self._hdf5_group.attrs['trace_count']
        
        # Resize datasets if needed
        if count >= self._hdf5_group['samples'].shape[0]:
            new_size = count + 1000  # grow by 1000
            self._hdf5_group['samples'].resize((new_size,) + trace.samples.shape)
            for name in ['timestamps', 'stimuli', 'responses', 'keys']:
                self._hdf5_group[name].resize((new_size,))
        
        # Write data
        self._hdf5_group['samples'][count] = trace.samples
        self._hdf5_group['timestamps'][count] = trace.timestamp.isoformat()
        self._hdf5_group['stimuli'][count] = str(trace.stimulus) if trace.stimulus else ""
        self._hdf5_group['responses'][count] = str(trace.response) if trace.response else ""
        self._hdf5_group['keys'][count] = str(trace.key) if trace.key else ""
        
        # Update count
        self._hdf5_group.attrs['trace_count'] = count + 1
    
    def flush_stream(self):
        if self._hdf5_file:
            self._hdf5_file.flush()
        return self
    
    def disable_streaming(self):
        """Disable streaming mode and close HDF5 file."""
        return self.close_stream()
    
    def close_stream(self):
        """Close the HDF5 stream and finalize datasets."""
        if self._hdf5_file:
            # Trim datasets to actual size
            count = self._hdf5_group.attrs['trace_count']
            if count > 0:
                self._hdf5_group['samples'].resize((count,) + self._hdf5_group['samples'].shape[1:])
                for name in ['timestamps', 'stimuli', 'responses', 'keys']:
                    self._hdf5_group[name].resize((count,))
            
            self._hdf5_file.close()
            self._hdf5_file = None
            self._hdf5_group = None
            self._traces_since_flush = 0
        return self
    
    def __enter__(self):
        """Context manager support for streaming."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure stream is closed when exiting context."""
        self.close_stream()
        return False