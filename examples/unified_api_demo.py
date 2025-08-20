#!/usr/bin/env python3
"""
Demonstrates the unified API for both streaming and batch data collection.
The same add_trace() method works for both modes!
"""

from scam import TraceDB, Series, Trace
import numpy as np
from datetime import datetime
import time


def demo_batch_mode():
    """Traditional batch mode - collect all traces in memory, save at the end."""
    print("=== Batch Mode Demo ===")
    
    db = TraceDB()
    exp = db.get_or_create_experiment("batch_experiment")
    series = exp.get_or_create_series("batch_series")
    
    # Collect traces in memory (persist=False is default)
    for i in range(100):
        trace = Trace(
            samples=np.random.randn(1000),
            timestamp=datetime.now(),
            stimulus=f"input_{i}",
            response=f"output_{i}"
        )
        series.add_trace(trace)  # Default: persist=False
    
    print(f"Collected {len(series.traces)} traces in memory")
    
    # Save everything at once
    db.save_hdf5("batch_data.h5")
    print("Saved all traces to batch_data.h5")
    
    return series


def demo_streaming_mode():
    """Streaming mode - persist traces immediately to disk."""
    print("\n=== Streaming Mode Demo ===")
    
    series = Series("streaming_series", traces=[])
    
    # Enable streaming to HDF5
    series.enable_streaming(
        filename="streaming_data.h5",
        experiment_name="streaming_experiment",
        auto_flush_interval=10  # Auto-flush every 10 traces
    )
    
    print(f"Streaming enabled: {series.is_streaming()}")
    
    # Collect and persist traces immediately
    for i in range(100):
        trace = Trace(
            samples=np.random.randn(1000),
            timestamp=datetime.now(),
            stimulus=f"input_{i}",
            response=f"output_{i}"
        )
        # persist=True means write to disk immediately
        series.add_trace(trace, persist=True)
        
        if (i + 1) % 25 == 0:
            print(f"  Streamed {i + 1} traces (auto-flushed every 10)")
    
    # Close the stream
    series.disable_streaming()
    print("Streaming complete and file closed")
    
    return series


def demo_hybrid_mode():
    """Hybrid mode - collect in memory with optional persistence."""
    print("\n=== Hybrid Mode Demo ===")
    
    series = Series("hybrid_series", traces=[])
    
    # Start without streaming
    print("Phase 1: Collecting in memory only")
    for i in range(20):
        trace = Trace(
            samples=np.random.randn(1000),
            timestamp=datetime.now(),
            stimulus=f"mem_{i}"
        )
        series.add_trace(trace)  # Just memory, no persistence
    
    print(f"  Collected {len(series.traces)} traces in memory")
    
    # Enable streaming mid-collection
    print("\nPhase 2: Enabling streaming for critical data")
    series.enable_streaming(
        filename="hybrid_data.h5",
        experiment_name="hybrid_experiment",
        auto_flush_interval=5
    )
    
    # Now collect with immediate persistence
    for i in range(20, 40):
        trace = Trace(
            samples=np.random.randn(1000),
            timestamp=datetime.now(),
            stimulus=f"critical_{i}"
        )
        series.add_trace(trace, persist=True)  # Persist critical traces
    
    print(f"  Streamed 20 more traces (total: {len(series.traces)})")
    
    # Disable streaming and continue in memory
    print("\nPhase 3: Back to memory-only collection")
    series.disable_streaming()
    
    for i in range(40, 50):
        trace = Trace(
            samples=np.random.randn(1000),
            timestamp=datetime.now(),
            stimulus=f"post_{i}"
        )
        series.add_trace(trace)  # Back to memory-only
    
    print(f"  Final count: {len(series.traces)} traces")
    print("  (Note: Only traces 20-39 were persisted to HDF5)")
    
    return series


def demo_context_manager():
    """Use context manager for automatic stream management."""
    print("\n=== Context Manager Demo ===")
    
    series = Series("context_series", traces=[])
    series.enable_streaming("context_data.h5", "context_experiment")
    
    # Use context manager to ensure stream is closed
    with series:
        for i in range(30):
            trace = Trace(
                samples=np.random.randn(1000),
                timestamp=datetime.now(),
                stimulus=f"ctx_{i}"
            )
            series.add_trace(trace, persist=True)
        
        print(f"  Added {len(series.traces)} traces")
        # Stream automatically closes when exiting context
    
    print(f"  Stream closed: {not series.is_streaming()}")
    
    return series


def demo_manual_flush_control():
    """Demonstrate manual flush control for critical sections."""
    print("\n=== Manual Flush Control Demo ===")
    
    series = Series("flush_series", traces=[])
    series.enable_streaming(
        filename="flush_data.h5",
        experiment_name="flush_experiment",
        auto_flush_interval=100  # High interval, we'll flush manually
    )
    
    print("Collecting traces with manual flush control...")
    
    # Normal collection
    for i in range(10):
        trace = Trace(samples=np.random.randn(1000))
        series.add_trace(trace, persist=True)  # No auto-flush yet
    
    print("  10 traces added (not flushed)")
    
    # Critical trace - force flush
    critical_trace = Trace(
        samples=np.random.randn(1000),
        stimulus="CRITICAL_MEASUREMENT"
    )
    series.add_trace(critical_trace, persist=True, flush=True)  # Force flush
    print("  Critical trace added and flushed immediately")
    
    # More normal traces
    for i in range(10):
        trace = Trace(samples=np.random.randn(1000))
        series.add_trace(trace, persist=True)
    
    print(f"  Total traces: {len(series.traces)}")
    
    series.disable_streaming()
    
    return series


def compare_apis():
    """Show the API comparison between old and new methods."""
    print("\n=== API Comparison ===")
    print("\nOld streaming API:")
    print("  series.open_hdf5_stream(filename, exp_name)")
    print("  series.append_trace_to_stream(trace)")
    print("  series.flush_stream()")
    print("  series.close_stream()")
    
    print("\nNew unified API:")
    print("  series.enable_streaming(filename, exp_name)")
    print("  series.add_trace(trace, persist=True)  # Same method for both modes!")
    print("  series.add_trace(trace, persist=True, flush=True)  # With manual flush")
    print("  series.disable_streaming()")
    
    print("\nBenefits:")
    print("  - Single add_trace() method for all use cases")
    print("  - Seamless switching between batch and streaming")
    print("  - Automatic flush management with manual override")
    print("  - Context manager support for safe resource management")


if __name__ == "__main__":
    # Run all demos
    demo_batch_mode()
    demo_streaming_mode()
    demo_hybrid_mode()
    demo_context_manager()
    demo_manual_flush_control()
    compare_apis()
    
    # Clean up demo files
    import os
    for filename in ["batch_data.h5", "streaming_data.h5", "hybrid_data.h5", 
                     "context_data.h5", "flush_data.h5"]:
        if os.path.exists(filename):
            os.remove(filename)
    
    print("\n✅ All demos complete! (demo files cleaned up)")