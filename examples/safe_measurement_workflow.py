#!/usr/bin/env python3
"""
Example showing safe measurement workflow that prevents data loss.
"""

from scam import TraceDB, Trace
import numpy as np
from datetime import datetime

def run_measurement_campaign():
    """
    Demonstrates the proper workflow for running multiple measurement series
    without losing data.
    """
    
    # First measurement run with fixed keys
    print("=== First Measurement Run: Fixed Keys ===")
    db1 = TraceDB()
    exp1 = db1.get_or_create_experiment(
        name="RV32I", 
        metadata={
            "device": "SimpleRV32I on CW305 (Artix-100)",
            "algorithm": "Tiny AES (128)"
        }
    )
    
    series1 = exp1.get_or_create_series(
        name="fixed_keys",
        metadata={
            "measurement_type": "Power Side Channel",
            "key_type": "fixed"
        }
    )
    
    # Simulate adding traces
    for i in range(10):
        trace = Trace(
            samples=np.random.randn(1000),
            timestamp=datetime.now(),
            stimulus=f"plaintext_{i:02x}",
            key="00112233445566778899aabbccddeeff"
        )
        series1.add_trace(trace)
    
    # Save with default mode='update' - safe by default
    db1.save_hdf5("RV32I.h5")
    print(f"Saved {len(series1.traces)} traces to fixed_keys series")
    
    # Second measurement run with random keys
    print("\n=== Second Measurement Run: Random Keys ===")
    db2 = TraceDB()
    exp2 = db2.get_or_create_experiment(
        name="RV32I",  # Same experiment name
        metadata={
            "device": "SimpleRV32I on CW305 (Artix-100)",
            "algorithm": "Tiny AES (128)"
        }
    )
    
    series2 = exp2.get_or_create_series(
        name="random_keys",  # Different series name
        metadata={
            "measurement_type": "Power Side Channel",
            "key_type": "random"
        }
    )
    
    # Simulate adding traces with random keys
    for i in range(10):
        random_key = np.random.bytes(16).hex()
        trace = Trace(
            samples=np.random.randn(1000),
            timestamp=datetime.now(),
            stimulus=f"plaintext_{i:02x}",
            key=random_key
        )
        series2.add_trace(trace)
    
    # Save with mode='update' (default) - this MERGES with existing file
    db2.save_hdf5("RV32I.h5")  # Safe! Will merge, not overwrite
    print(f"Saved {len(series2.traces)} traces to random_keys series")
    
    # Verify both series are preserved
    print("\n=== Verification: Loading Combined Database ===")
    db_loaded = TraceDB.load_hdf5("RV32I.h5")
    
    for exp in db_loaded:
        print(f"Experiment: {exp.name}")
        for series in exp.series:
            print(f"  - Series: {series.name} with {len(series.traces)} traces")


def demonstrate_safety_features():
    """
    Demonstrates the safety features to prevent accidental data loss.
    """
    print("\n=== Safety Features Demo ===")
    
    # Create initial database
    db = TraceDB()
    exp = db.get_or_create_experiment("TestExp")
    series = exp.get_or_create_series("important_data")
    series.add_trace(Trace(samples=np.random.randn(100), timestamp=datetime.now()))
    
    # Save initial data
    db.save_hdf5("test_safety.h5")
    print("Initial data saved")
    
    # Try to overwrite without permission - will fail
    try:
        db2 = TraceDB()
        exp2 = db2.get_or_create_experiment("NewExp")
        db2.save_hdf5("test_safety.h5", mode='overwrite')  # Will raise error
    except ValueError as e:
        print(f"✓ Protected from accidental overwrite: {e}")
    
    # Explicit overwrite when intended
    db3 = TraceDB()
    exp3 = db3.get_or_create_experiment("ReplacementExp")
    db3.save_hdf5("test_safety.h5", mode='overwrite', overwrite_ok=True)
    print("✓ Explicit overwrite succeeded with overwrite_ok=True")
    
    # Duplicate series protection
    db4 = TraceDB()
    exp4 = db4.get_or_create_experiment("TestExp")
    series4 = exp4.get_or_create_series("important_data")  # Same name as before
    series4.add_trace(Trace(samples=np.random.randn(100), timestamp=datetime.now()))
    
    db4.save_hdf5("test_safety.h5", mode='update')
    print("✓ Warning issued for duplicate series name (data preserved)")


if __name__ == "__main__":
    # Run the main workflow example
    run_measurement_campaign()
    
    # Demonstrate safety features
    demonstrate_safety_features()
    
    print("\n=== Summary of Best Practices ===")
    print("1. Use mode='update' (default) to safely merge new data")
    print("2. Use unique series names within experiments to avoid conflicts")
    print("3. Use mode='overwrite' with overwrite_ok=True only when intentional")
    print("4. Always verify your data after saving by loading and checking")