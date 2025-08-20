#!/usr/bin/env python3
"""
Convenience methods example for SCA Data Model.

Demonstrates the get_or_create methods that simplify
experiment and series management.
"""

import numpy as np
import warnings
from sca_datamodel import TraceDB, Trace


def demonstrate_convenience_methods():
    """Show how get_or_create methods work."""
    db = TraceDB()
    
    # Create experiment and series in one step
    exp = db.get_or_create_experiment("crypto_analysis", metadata={
        "device": "FPGA_Spartan6",
        "algorithm": "RSA-2048"
    })
    
    series = exp.get_or_create_series("timing_measurements", metadata={
        "measurement_type": "timing",
        "precision": "nanosecond"
    })
    
    print(f"Created experiment: {exp.name}")
    print(f"Created series: {series.name}")
    
    # Add some data
    for i in range(10):
        trace = Trace(
            samples=np.random.random(500),
            stimulus=f"message_{i}"
        )
        series.add_trace(trace)
    
    print(f"Added {len(series)} traces")


def demonstrate_warnings():
    """Show warning system for duplicate names."""
    db = TraceDB()
    
    # First creation - no warning
    exp1 = db.get_or_create_experiment("test_experiment")
    
    # Second creation - will warn but return existing
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        exp2 = db.get_or_create_experiment("test_experiment")
        
        if w:
            print(f"Warning received: {w[0].message}")
        
        # Same object returned
        assert exp1 is exp2
        print("Same experiment object returned")


def realistic_workflow():
    """Show realistic workflow with convenience methods."""
    # Setup can be done in any order without KeyErrors
    db = TraceDB()
    
    # Multiple experiments
    experiments = [
        ("aes_round1", {"target": "first_round"}),
        ("aes_round10", {"target": "last_round"}),
        ("rsa_crt", {"target": "crt_operations"})
    ]
    
    for exp_name, metadata in experiments:
        exp = db.get_or_create_experiment(exp_name, metadata=metadata)
        
        # Multiple measurement types per experiment
        measurement_types = ["power", "electromagnetic", "timing"]
        
        for mtype in measurement_types:
            series = exp.get_or_create_series(f"{mtype}_traces", metadata={
                "measurement_type": mtype,
                "sampling_rate": 1e9 if mtype != "timing" else 1e12
            })
            
            # Add sample data
            for i in range(5):
                trace = Trace(samples=np.random.random(100))
                series.add_trace(trace)
    
    # Save everything
    db.save_hdf5("multi_experiment.h5")
    
    # Summary
    print(f"Created {len(db)} experiments:")
    for exp_name in db.experiments:
        exp = db[exp_name] 
        print(f"  {exp_name}: {len(exp)} series")
        for series in exp:
            print(f"    {series.name}: {len(series)} traces")


if __name__ == "__main__":
    print("SCA Data Model - Convenience Methods Examples")
    print("=" * 55)
    
    print("\n1. Basic Convenience Methods:")
    demonstrate_convenience_methods()
    
    print("\n2. Warning System:")
    demonstrate_warnings()
    
    print("\n3. Realistic Multi-Experiment Workflow:")
    realistic_workflow()
    
    print("\nDone.")