#!/usr/bin/env python3
"""
Demonstrates the new always-lazy loading and always-streaming writing patterns.
"""

from scam import TraceDB, Series, Trace
import numpy as np
from datetime import datetime
import time


def demo_writing_always_streams():
    """Show that writing always streams to disk."""
    print("=== Writing Always Streams ===\n")
    
    series = Series("measurements", traces=[])
    
    # Open for writing - will stream everything
    series.open_for_writing("streaming_data.h5", "experiment1")
    
    print("Adding 1000 traces (auto-streaming to disk)...")
    for i in range(1000):
        trace = Trace(
            samples=np.random.randn(5000),  # Large traces
            timestamp=datetime.now(),
            stimulus=f"input_{i}",
            response=f"output_{i}"
        )
        series.add_trace(trace)  # Automatically persisted!
        
        if (i + 1) % 250 == 0:
            print(f"  Streamed {i + 1} traces...")
    
    # Close and finalize
    series.close_writing()
    print("✅ All traces streamed to disk efficiently!\n")


def demo_reading_always_lazy():
    """Show that reading is always lazy."""
    print("=== Reading Always Lazy ===\n")
    
    # First create a large dataset
    print("Creating large dataset...")
    series = Series("large_dataset", traces=[])
    series.open_for_writing("large_data.h5", "experiment1")
    
    for i in range(10000):
        trace = Trace(samples=np.ones(1000) * i)
        series.add_trace(trace)
    
    series.close_writing()
    print(f"Created 10,000 traces\n")
    
    # Now load lazily
    print("Loading dataset lazily...")
    start_time = time.time()
    db = TraceDB.load_hdf5("large_data.h5")  # Instant! No data loaded
    load_time = time.time() - start_time
    print(f"✅ Load time: {load_time:.4f} seconds (no data loaded!)\n")
    
    # Access series
    series = db["experiment1"]["large_dataset"]
    print(f"Series has {len(series.traces)} traces in memory (lazy)")
    
    # Open for reading
    series.open_for_reading()
    print(f"Series has {len(series)} total traces (from metadata)")
    
    # Iterate over just first 5 traces
    print("\nIterating over first 5 traces only:")
    for i, trace in enumerate(series):
        print(f"  Trace {i}: sum = {trace.samples.sum()}")
        if i >= 4:
            break
    
    series.close_reading()
    print("\n✅ Only loaded what was needed!")


def demo_analysis_workflow():
    """Show typical SCA analysis workflow."""
    print("\n=== Typical SCA Analysis Workflow ===\n")
    
    # STEP 1: COLLECT - Always streams
    print("STEP 1: Collecting measurements...")
    series = Series("power_traces", traces=[])
    series.open_for_writing("sca_data.h5", "aes_attack")
    
    # Simulate collecting from oscilloscope
    for i in range(500):
        # Simulate measurement
        trace = Trace(
            samples=np.random.randn(10000) + np.sin(np.linspace(0, 10, 10000)) * (i % 16),
            timestamp=datetime.now(),
            stimulus=f"plaintext_{i:04x}",
            key=f"key_{i % 256:02x}"
        )
        series.add_trace(trace)
    
    series.close_writing()
    print(f"✅ Collected 500 traces\n")
    
    # STEP 2: ANALYZE - Always lazy
    print("STEP 2: Analyzing measurements...")
    db = TraceDB.load_hdf5("sca_data.h5")
    series = db["aes_attack"]["power_traces"]
    
    series.open_for_reading()
    
    # Example 1: Compute mean trace (efficient chunked loading)
    print("Computing mean trace...")
    matrix = series.to_matrix()  # Efficient chunked loading
    mean_trace = matrix.mean(axis=0)
    print(f"  Mean trace shape: {mean_trace.shape}")
    
    # Example 2: Process traces one at a time
    print("\nProcessing traces individually...")
    max_values = []
    for i, trace in enumerate(series):
        max_values.append(trace.samples.max())
        if i >= 9:  # Just first 10 for demo
            break
    print(f"  Max values from first 10 traces: {max_values}")
    
    # Example 3: Slice access
    print("\nAccessing specific trace range...")
    subset = series[100:105]
    print(f"  Got {len(subset)} traces from index 100-104")
    
    series.close_reading()
    print("\n✅ Analysis complete - minimal memory usage!")


def demo_append_workflow():
    """Show how to append more data to existing files."""
    print("\n=== Append Workflow ===\n")
    
    # Initial collection
    print("Initial data collection...")
    series1 = Series("initial", traces=[])
    series1.open_for_writing("append_test.h5", "experiment1")
    
    for i in range(100):
        series1.add_trace(Trace(samples=np.ones(100) * i))
    
    series1.close_writing()
    print("✅ Saved 100 traces\n")
    
    # Later, append more data
    print("Appending more data...")
    series2 = Series("additional", traces=[])
    series2.open_for_writing("append_test.h5", "experiment1", mode='a')
    
    for i in range(100, 150):
        series2.add_trace(Trace(samples=np.ones(100) * i))
    
    series2.close_writing()
    print("✅ Appended 50 more traces\n")
    
    # Load and verify
    print("Loading combined dataset...")
    db = TraceDB.load_hdf5("append_test.h5")
    
    series_initial = db["experiment1"]["initial"]
    series_additional = db["experiment1"]["additional"]
    
    series_initial.open_for_reading()
    series_additional.open_for_reading()
    
    print(f"  Initial series: {len(series_initial)} traces")
    print(f"  Additional series: {len(series_additional)} traces")
    
    series_initial.close_reading()
    series_additional.close_reading()
    
    print("\n✅ Both series preserved in same file!")


def cleanup_demo_files():
    """Clean up demo files."""
    import os
    for filename in ["streaming_data.h5", "large_data.h5", "sca_data.h5", "append_test.h5"]:
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    print("=" * 60)
    print("SCAM: Always-Stream Write, Always-Lazy Read")
    print("=" * 60 + "\n")
    
    demo_writing_always_streams()
    demo_reading_always_lazy()
    demo_analysis_workflow()
    demo_append_workflow()
    
    cleanup_demo_files()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("- Writing ALWAYS streams to disk (memory efficient)")
    print("- Reading ALWAYS lazy loads (scales to any size)")
    print("- Simple API with clear semantics")
    print("- Perfect for SCA workflows!")
    print("=" * 60)