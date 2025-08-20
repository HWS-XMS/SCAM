# SCAM - Side Channel Analysis Measurements

A Python package for organizing and managing Side Channel Analysis data with HDF5 persistence.

## Features

- **Hierarchical Organization**: TraceDB → Experiment → Series → Trace
- **HDF5 Persistence**: Efficient storage with metadata preservation
- **Data Safety**: Automatic merging prevents accidental data loss (v0.2.0+)
- **Crash-Safe Streaming**: Real-time data capture with flush protection
- **Convenient APIs**: Get-or-create methods with helpful warnings
- **Shape Validation**: Automatic trace compatibility checking
- **Flexible Metadata**: Store arbitrary information at any level

## Installation

```bash
pip install scam
```

For development:
```bash
git clone https://github.com/example/scam.git
cd scam
pip install -e .
```

## Quick Start

```python
from scam import TraceDB, Experiment, Series, Trace
import numpy as np

# Create database structure
db = TraceDB()
exp = db.get_or_create_experiment("aes_attack", metadata={"device": "STM32F4"})
series = exp.get_or_create_series("power_traces", metadata={"probe": "CT-1"})

# Add traces
for i in range(100):
    trace = Trace(
        samples=np.random.random(1000),  # Your measurement data
        stimulus=f"plaintext_{i}",
        response=f"ciphertext_{i}"
    )
    series.add_trace(trace)

# Save to HDF5 (v0.2.0+ merges by default, preventing data loss)
db.save_hdf5("experiment_data.h5")

# Load for analysis
analysis_db = TraceDB.load_hdf5("experiment_data.h5")
traces = analysis_db["aes_attack"]["power_traces"]
```

## Streaming for Large Datasets

```python
# For crash-safe real-time collection
series = Series("power_traces", traces=[])
series.open_hdf5_stream("live_data.h5", "experiment_name")

for measurement in data_stream:
    trace = Trace(samples=measurement)
    series.append_trace_to_stream(trace)
    
    # Flush every 50 traces for crash protection
    if len(series) % 50 == 0:
        series.flush_stream()

series.close_stream()
```

## Data Structure

### Trace
- **samples**: NumPy array of measurement data
- **timestamp**: Automatic timestamp (customizable)
- **stimulus**: Input that caused this trace (optional)
- **response**: Output from this trace (optional)
- **key**: Cryptographic key for white-box scenarios (optional)

### Series
- Collection of traces with identical dimensions
- Metadata for measurement conditions
- Streaming HDF5 support

### Experiment
- Collection of series from the same target device
- Metadata for experimental setup

### TraceDB
- Collection of experiments
- Complete HDF5 persistence

## Data Safety (v0.2.0+)

The library now protects your data by default:

```python
# First measurement run
db1 = TraceDB()
exp1 = db1.get_or_create_experiment("RV32I")
series1 = exp1.get_or_create_series("fixed_keys")
# ... add traces ...
db1.save_hdf5("data.h5")  # Creates new file

# Second measurement run
db2 = TraceDB()
exp2 = db2.get_or_create_experiment("RV32I")
series2 = exp2.get_or_create_series("random_keys")
# ... add traces ...
db2.save_hdf5("data.h5")  # MERGES with existing file (safe by default!)

# Both series are preserved in the file
loaded = TraceDB.load_hdf5("data.h5")
# loaded["RV32I"] now contains both "fixed_keys" and "random_keys" series
```

### Save Modes

- **`mode='update'` (default)**: Merges with existing file, preserving all data
- **`mode='overwrite'`**: Replaces file (requires `overwrite_ok=True` for safety)

```python
# Safe merge (default)
db.save_hdf5("data.h5")

# Explicit overwrite when intended
db.save_hdf5("data.h5", mode='overwrite', overwrite_ok=True)
```

## Examples

See the `examples/` directory for:
- `basic_usage.py` - Basic data collection and analysis
- `convenience_methods.py` - Get-or-create methods and warnings
- `safe_measurement_workflow.py` - Data safety features and best practices

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

Or use unittest:
```bash
python -m unittest tests.test_datamodel
```

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- h5py ≥ 3.0.0

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Changelog

### v0.2.0
- **CRITICAL FIX**: `save_hdf5()` now merges by default instead of overwriting
- Added `mode` parameter to `save_hdf5()` for explicit control
- Added `overwrite_ok` safety flag to prevent accidental data loss
- Warning system for duplicate series names
- New comprehensive safety tests
- New example: `safe_measurement_workflow.py`

### v0.1.0
- Initial release
- Basic data model with HDF5 persistence
- Streaming support with crash protection
- Convenience methods with warnings
- Comprehensive test suite