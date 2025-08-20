from dataclasses import dataclass, field
from .trace import Trace
import h5py
import numpy as np


@dataclass
class Series:
    name: str
    traces: list[Trace]
    metadata: dict[str, any] = field(default_factory=dict)
    _hdf5_file: any = field(default=None, init=False)
    _hdf5_group: any = field(default=None, init=False)
    
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
    
    def add_trace(self, trace):
        if self.traces and trace.samples.shape != self.traces[0].samples.shape:
            raise ValueError(f"Trace has shape {trace.samples.shape}, expected {self.traces[0].samples.shape}")
        self.traces.append(trace)
    
    def remove_trace(self, index):
        if 0 <= index < len(self.traces):
            return self.traces.pop(index)
        raise IndexError(f"Trace index {index} out of range")
    
    def open_hdf5_stream(self, filename, experiment_name, initial_size=1000, chunk_size=100, sample_shape=None, experiment_metadata=None):
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
        if not self._hdf5_file:
            raise RuntimeError("HDF5 stream not open. Call open_hdf5_stream() first")
        
        # Add to in-memory list
        self.add_trace(trace)
        
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
        
        return self
    
    def flush_stream(self):
        if self._hdf5_file:
            self._hdf5_file.flush()
        return self
    
    def close_stream(self):
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
        return self