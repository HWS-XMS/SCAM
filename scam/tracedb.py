from dataclasses import dataclass, field
from .experiment import Experiment
from .series import Series
from .trace import Trace
import h5py
import numpy as np
import warnings
from datetime import datetime


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
        """
        Save database to HDF5 file.
        
        Args:
            filename: Path to save the HDF5 file
            mode: 'update' to merge with existing file, 'overwrite' to replace completely
            overwrite_ok: If False, will raise error when overwriting existing data
        """
        import os
        
        if mode not in ['update', 'overwrite']:
            raise ValueError("mode must be 'update' or 'overwrite'")
        
        file_exists = os.path.exists(filename)
        
        if mode == 'overwrite' and file_exists and not overwrite_ok:
            raise ValueError(f"File '{filename}' already exists. Set overwrite_ok=True to overwrite or use mode='update' to merge.")
        
        if mode == 'update' and file_exists:
            # Load existing data first
            existing_db = TraceDB.load_hdf5(filename)
            
            # Merge current data into existing
            for exp_name, experiment in self.experiments.items():
                if exp_name in existing_db.experiments:
                    # Merge series within existing experiment
                    existing_exp = existing_db.experiments[exp_name]
                    existing_series_names = {s.name for s in existing_exp.series}
                    
                    for series in experiment.series:
                        if series.name in existing_series_names:
                            warnings.warn(f"Series '{series.name}' in experiment '{exp_name}' already exists. Skipping to avoid data loss. Use unique series names or overwrite_ok=True.", UserWarning)
                        else:
                            existing_exp.series.append(series)
                else:
                    # Add new experiment
                    existing_db.experiments[exp_name] = experiment
            
            # Now save the merged database
            self._write_hdf5(filename, existing_db.experiments)
        else:
            # Write new file or overwrite
            self._write_hdf5(filename, self.experiments)
    
    def _write_hdf5(self, filename, experiments_dict):
        """Internal method to write experiments to HDF5."""
        with h5py.File(filename, 'w') as f:
            for exp_name, experiment in experiments_dict.items():
                exp_group = f.create_group(exp_name)
                
                for key, value in experiment.metadata.items():
                    if isinstance(value, str):
                        exp_group.attrs[key] = value
                    elif isinstance(value, (int, float)):
                        exp_group.attrs[key] = value
                
                for series in experiment.series:
                    series_group = exp_group.create_group(series.name)
                    
                    for key, value in series.metadata.items():
                        if isinstance(value, str):
                            series_group.attrs[key] = value
                        elif isinstance(value, (int, float)):
                            series_group.attrs[key] = value
                    
                    traces_data = []
                    timestamps = []
                    stimuli = []
                    responses = []
                    keys = []
                    
                    for trace in series.traces:
                        traces_data.append(trace.samples)
                        timestamps.append(trace.timestamp.isoformat())
                        stimuli.append(str(trace.stimulus) if trace.stimulus is not None else "")
                        responses.append(str(trace.response) if trace.response is not None else "")
                        keys.append(str(trace.key) if trace.key is not None else "")
                    
                    if traces_data:
                        series_group.create_dataset('samples', data=np.array(traces_data))
                        series_group.create_dataset('timestamps', data=[t.encode('utf-8') for t in timestamps])
                        series_group.create_dataset('stimuli', data=[s.encode('utf-8') for s in stimuli])
                        series_group.create_dataset('responses', data=[r.encode('utf-8') for r in responses])
                        series_group.create_dataset('keys', data=[k.encode('utf-8') for k in keys])
    
    @classmethod
    def load_hdf5(cls, filename):
        db = cls()
        
        with h5py.File(filename, 'r') as f:
            for exp_name in f.keys():
                exp_group = f[exp_name]
                
                experiment = Experiment(name=exp_name, series=[])
                for attr_name in exp_group.attrs:
                    experiment.metadata[attr_name] = exp_group.attrs[attr_name]
                
                for series_name in exp_group.keys():
                    series_group = exp_group[series_name]
                    
                    series = Series(name=series_name, traces=[])
                    for attr_name in series_group.attrs:
                        series.metadata[attr_name] = series_group.attrs[attr_name]
                    
                    if 'samples' in series_group:
                        samples_data = series_group['samples'][:]
                        timestamps_data = [t.decode('utf-8') for t in series_group['timestamps'][:]]
                        stimuli_data = [s.decode('utf-8') if s.decode('utf-8') else None for s in series_group['stimuli'][:]]
                        responses_data = [r.decode('utf-8') if r.decode('utf-8') else None for r in series_group['responses'][:]]
                        keys_data = [k.decode('utf-8') if k.decode('utf-8') else None for k in series_group['keys'][:]]
                        
                        for i in range(len(samples_data)):
                            trace = Trace(
                                samples=samples_data[i],
                                timestamp=datetime.fromisoformat(timestamps_data[i]),
                                stimulus=stimuli_data[i],
                                response=responses_data[i],
                                key=keys_data[i]
                            )
                            series.traces.append(trace)
                    
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