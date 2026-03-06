from dataclasses import dataclass
import numpy as np
from .schema import Array, Scalar


@dataclass
class Trace:
    """Default trace schema for simple SCA measurements."""
    samples:   Array[np.float32]
    stimulus:  Scalar[str] = None
    response:  Scalar[str] = None
    key:       Scalar[str] = None
    timestamp: Scalar[str] = None
