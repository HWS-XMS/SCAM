from dataclasses import dataclass, field
from .series import Series
import warnings


@dataclass
class Experiment:
    name: str
    series: list[Series]
    metadata: dict[str, any] = field(default_factory=dict)
    
    def __iter__(self):
        return iter(self.series)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            for series in self.series:
                if series.name == key:
                    return series
            raise KeyError(f"Series '{key}' not found")
        return self.series[key]
    
    def __len__(self):
        return len(self.series)
    
    def add_series(self, series):
        for existing in self.series:
            if existing.name == series.name:
                raise ValueError(f"Series '{series.name}' already exists in experiment")
        self.series.append(series)
    
    def remove_series(self, name):
        for i, series in enumerate(self.series):
            if series.name == name:
                return self.series.pop(i)
        raise KeyError(f"Series '{name}' not found")
    
    def get_or_create_series(self, name, metadata=None):
        try:
            existing_series = self[name]
            warnings.warn(f"Series '{name}' already exists in experiment '{self.name}', returning existing series", UserWarning)
            return existing_series
        except KeyError:
            series = Series(name=name, traces=[])
            if metadata:
                series.metadata.update(metadata)
            self.series.append(series)
            return series