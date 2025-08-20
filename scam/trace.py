from dataclasses import dataclass
import numpy as np
from datetime import datetime


@dataclass
class Trace:
    samples: np.ndarray
    stimulus: any = None
    response: any = None
    key: any = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()