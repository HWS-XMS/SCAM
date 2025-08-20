#!/usr/bin/env python3
"""
Basic usage example for SCAM (Side Channel Analysis Measurements).

Demonstrates:
- Creating experiments and series
- Adding traces with metadata
- Saving to HDF5
- Loading for analysis
"""

import numpy as np
from datetime import datetime
from scam import TraceDB, Experiment, Series, Trace


def basic_data_collection():
    """Demonstrate basic data collection workflow."""
    # Create database structure
    db = TraceDB()
    
    # Create experiment
    experiment = Experiment("aes_power_analysis", series=[])
    experiment.metadata = {
        "device": "STM32F4",
        "algorithm": "AES-128",
        "voltage": 3.3,
        "temperature": 25.0
    }
    
    # Create series for power measurements
    power_series = Series("power_consumption", traces=[])
    power_series.metadata = {
        "measurement_type": "power",
        "sampling_rate": 1000000000,  # 1 GHz
        "probe": "Current transformer CT-1",
        "amplification": 20
    }
    
    # Simulate data collection
    for i in range(100):
        # Simulate power measurement
        samples = np.random.random(1000) + 0.1 * np.sin(np.linspace(0, 10*np.pi, 1000))
        
        # Create trace with metadata
        trace = Trace(
            samples=samples,
            stimulus=f"plaintext_{i:04d}",
            response=f"ciphertext_{i:04d}"
        )
        
        power_series.add_trace(trace)
    
    # Add to experiment and database
    experiment.add_series(power_series)
    db.add_experiment(experiment)
    
    # Save to file
    db.save_hdf5("power_analysis.h5")
    
    print(f"Collected {len(power_series)} traces")
    print(f"Saved experiment '{experiment.name}' to power_analysis.h5")


def basic_data_analysis():
    """Demonstrate basic data analysis workflow."""
    # Load saved data
    db = TraceDB.load_hdf5("power_analysis.h5")
    
    # Access experiment and series
    experiment = db["aes_power_analysis"]
    power_traces = experiment["power_consumption"]
    
    print(f"Loaded experiment: {experiment.name}")
    print(f"Device: {experiment.metadata['device']}")
    print(f"Algorithm: {experiment.metadata['algorithm']}")
    
    # Open series for reading (lazy loading)
    power_traces.open_for_reading()
    
    print(f"Number of traces: {len(power_traces)}")
    print(f"Sampling rate: {power_traces.metadata['sampling_rate']} Hz")
    
    # Basic analysis
    trace_lengths = [len(trace.samples) for trace in power_traces]
    mean_power = np.mean([np.mean(trace.samples) for trace in power_traces])
    
    print(f"Trace length: {trace_lengths[0]} samples")
    print(f"Mean power consumption: {mean_power:.6f}")
    
    power_traces.close_reading()


def streaming_collection_example():
    """Demonstrate automatic streaming with new always-stream architecture."""
    # Create series
    power_series = Series("streaming_power", traces=[])
    power_series.metadata = {
        "measurement_type": "power",
        "sampling_rate": 500000000  # 500 MHz
    }
    
    # Open for writing - automatically streams everything to disk
    power_series.open_for_writing("streaming_data.h5", "streaming_experiment")
    
    try:
        # Simulate real-time data collection
        for i in range(50):
            # Simulate measurement
            samples = np.random.random(2000)
            
            trace = Trace(
                samples=samples,
                stimulus=f"input_{i}",
                response=f"output_{i}"
            )
            
            # Add trace - automatically persisted to disk!
            power_series.add_trace(trace)
            
            if (i + 1) % 10 == 0:
                print(f"Streamed {i + 1} traces...")
                
        print(f"Streamed {len(power_series.traces)} traces to disk")
        
    finally:
        # Always close writing
        power_series.close_writing()
    
    # Verify streaming worked
    db = TraceDB.load_hdf5("streaming_data.h5")
    loaded_traces = db["streaming_experiment"]["streaming_power"]
    loaded_traces.open_for_reading()
    print(f"Verified: {len(loaded_traces)} traces loaded from stream")
    loaded_traces.close_reading()


if __name__ == "__main__":
    print("SCAM - Basic Usage Examples")
    print("=" * 50)
    
    print("\n1. Basic Data Collection:")
    basic_data_collection()
    
    print("\n2. Basic Data Analysis:")
    basic_data_analysis()
    
    print("\n3. Streaming Collection:")
    streaming_collection_example()
    
    print("\nDone. Check the generated .h5 files.")