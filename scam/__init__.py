"""
SCAM - Side Channel Analysis Measurements

A Python package for managing Side Channel Analysis data with HDF5 persistence.
Provides hierarchical organization: TraceDB -> Experiment -> Series -> Trace
"""

from .schema import Array, Scalar, SchemaError, make_trace_type
from .trace import Trace
from .series import Series, get_session_uuid, new_session_uuid
from .experiment import Experiment
from .tracedb import TraceDB

__version__ = "2.0.0"
__author__ = "SCAM Contributors"
__email__ = "info@example.com"

__all__ = [
    'Array', 'Scalar', 'SchemaError', 'make_trace_type',
    'Trace', 'Series', 'Experiment', 'TraceDB',
    'get_session_uuid', 'new_session_uuid',
]
