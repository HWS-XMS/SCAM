# SCA Data Model

A Python package for organizing and managing Side Channel Analysis (SCA) data with HDF5 persistence.

## Features

- **Hierarchical Organization**: TraceDB → Experiment → Series → Trace
- **HDF5 Persistence**: Efficient storage with metadata preservation
- **Crash-Safe Streaming**: Real-time data capture with flush protection
- **Convenient APIs**: Get-or-create methods with helpful warnings
- **Shape Validation**: Automatic trace compatibility checking
- **Flexible Metadata**: Store arbitrary information at any level

## Installation

```bash
pip install sca-datamodel
```

For development:
```bash
git clone https://github.com/example/sca-datamodel.git
cd sca-datamodel
pip install -e .
```

## Quick Start

```python
from sca_datamodel import TraceDB, Experiment, Series, Trace
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

# Save to HDF5
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

## Examples

See the `examples/` directory for:
- `basic_usage.py` - Basic data collection and analysis
- `convenience_methods.py` - Get-or-create methods and warnings

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

### v0.1.0
- Initial release
- Basic data model with HDF5 persistence
- Streaming support with crash protection
- Convenience methods with warnings
- Comprehensive test suite