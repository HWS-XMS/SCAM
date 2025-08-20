"""
SCAM - Side Channel Analysis Measurements

A Python package for managing Side Channel Analysis data with HDF5 persistence.
Provides hierarchical organization: TraceDB -> Experiment -> Series -> Trace
"""

from .trace import Trace
from .series import Series
from .experiment import Experiment
from .tracedb import TraceDB

__version__ = "0.2.0"
__author__ = "SCAM Contributors" 
__email__ = "info@example.com"

__all__ = ['Trace', 'Series', 'Experiment', 'TraceDB']