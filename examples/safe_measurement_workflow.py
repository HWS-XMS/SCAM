#!/usr/bin/env python3
"""
Example demonstrating safe measurement workflow with automatic UUID-based sessions.

This example shows how SCAM automatically generates session UUIDs to prevent
accidentally mixing data from different measurement runs.
"""

import numpy as np
from datetime import datetime
from scam import Series, Trace, get_session_uuid, new_session_uuid

def simple_uuid_workflow():
    """Simple example showing the UUID-based workflow."""
    print("=== Simple UUID Workflow ===\n")
    
    # Just create series and start writing - UUID is automatic!
    series = Series("power_traces", traces=[])
    series.open_for_writing("simple_demo.h5", "my_experiment")
    
    print(f"Automatic session UUID: {get_session_uuid()}")
    
    # Add some traces
    for i in range(5):
        trace = Trace(samples=np.random.randn(1000), stimulus=f"input_{i}")
        series.add_trace(trace)
    
    series.close_writing()
    print("✅ Added 5 traces")
    
    # Continue in same session - works seamlessly
    series2 = Series("power_traces", traces=[])
    series2.open_for_writing("simple_demo.h5", "my_experiment", mode='a')
    
    for i in range(5, 8):
        trace = Trace(samples=np.random.randn(1000), stimulus=f"input_{i}")
        series2.add_trace(trace)
    
    series2.close_writing()
    print("✅ Added 3 more traces - no prompts needed!")
    
    # Clean up
    import os
    if os.path.exists("simple_demo.h5"):
        os.remove("simple_demo.h5")
    
    print("✅ Simple workflow complete!\n")

def measurement_session_example():
    """Example of automatic UUID-based measurement sessions."""
    print("=== Safe Measurement Workflow with Auto UUIDs ===\n")
    
    # SCENARIO 1: Starting a new measurement session
    print("SCENARIO 1: Starting new measurement session...")
    print(f"Session UUID: {get_session_uuid()}")
    
    series = Series("aes_power_traces", traces=[])
    
    # No measurement_id needed - SCAM automatically uses session UUID
    series.open_for_writing("aes_measurements.h5", "experiment_001")
    
    # Collect traces
    print("Collecting 100 power traces...")
    for i in range(100):
        # Simulate power measurement
        power_trace = np.random.randn(5000) + np.sin(np.linspace(0, 10, 5000)) * (i % 16)
        
        trace = Trace(
            samples=power_trace,
            timestamp=datetime.now(),
            stimulus=f"plaintext_{i:04x}",
            key="fixed_secret_key"
        )
        series.add_trace(trace)
        
        if (i + 1) % 25 == 0:
            print(f"  Collected {i + 1} traces...")
    
    series.close_writing()
    print("✅ Initial measurement session complete!\n")
    
    # SCENARIO 2: Resuming the same measurement session (SAFE)
    print("SCENARIO 2: Resuming same measurement session...")
    print(f"Same session UUID: {get_session_uuid()}")
    
    series2 = Series("aes_power_traces", traces=[])
    
    # Same session UUID - works seamlessly without any prompts!
    series2.open_for_writing("aes_measurements.h5", "experiment_001", mode='a')
    
    print("Adding 50 more traces to same session...")
    for i in range(100, 150):
        power_trace = np.random.randn(5000) + np.sin(np.linspace(0, 10, 5000)) * (i % 16)
        trace = Trace(
            samples=power_trace,
            timestamp=datetime.now(),
            stimulus=f"plaintext_{i:04x}",
            key="fixed_secret_key"
        )
        series2.add_trace(trace)
    
    series2.close_writing()
    print("✅ Successfully resumed measurement session!\n")
    
    # SCENARIO 3: Simulating different measurement session (PROTECTED)
    print("SCENARIO 3: Simulating different measurement session...")
    
    old_uuid = get_session_uuid()
    new_uuid = new_session_uuid()  # Generate new session UUID
    print(f"Old session UUID: {old_uuid}")
    print(f"New session UUID: {new_uuid}")
    
    series3 = Series("aes_power_traces", traces=[])
    
    try:
        # This will fail because session UUID changed
        series3.open_for_writing("aes_measurements.h5", "experiment_001", mode='a')
        print("❌ ERROR: Should have been blocked!")
    except ValueError as e:
        print(f"✅ Correctly blocked: {e}")
    
    print("\nTo proceed with different measurement, you have options:")
    print("1. Use confirm_append=False to bypass safety")
    print("2. Create a new series name")
    print("3. Create a new experiment name\n")
    
    # SCENARIO 4: Manual override for different measurement
    print("SCENARIO 4: Manual override (bypassing safety)...")
    
    series4 = Series("aes_power_traces", traces=[])
    series4.open_for_writing(
        "aes_measurements.h5",
        "experiment_001",
        mode='a', 
        confirm_append=False  # Bypass safety for intentional override
    )
    
    print("Adding traces with new session UUID...")
    for i in range(3):
        trace = Trace(
            samples=np.random.randn(5000),
            timestamp=datetime.now(),
            stimulus=f"new_plaintext_{i}",
            key="different_key"
        )
        series4.add_trace(trace)
    
    series4.close_writing()
    print("✅ Successfully added with manual override\n")
    
    # SCENARIO 5: Best practice - use different series for different measurements
    print("SCENARIO 5: Best practice - separate series for separate measurements...")
    
    series5 = Series("aes_power_traces_v2", traces=[])  # Different series name
    series5.open_for_writing("aes_measurements.h5", "experiment_001", mode='a')
    
    print("Adding traces to new series...")
    for i in range(5):
        trace = Trace(
            samples=np.random.randn(5000),
            timestamp=datetime.now(),
            stimulus=f"v2_plaintext_{i}",
            key="v2_key"
        )
        series5.add_trace(trace)
    
    series5.close_writing()
    print("✅ Clean separation with different series names!\n")
    
    # VERIFICATION: Check final state
    print("=== VERIFICATION ===")
    from scam import TraceDB
    
    db = TraceDB.load_hdf5("aes_measurements.h5")
    exp = db["experiment_001"]
    
    print(f"Experiment contains {len(exp.series)} series:")
    for series in exp.series:
        series.open_for_reading()
        session_uuid = series._h5group.attrs.get('measurement_id', 'None')
        if isinstance(session_uuid, bytes):
            session_uuid = session_uuid.decode()
        print(f"  - {series.name}: {len(series)} traces (UUID: {session_uuid[:8]}...)")
        series.close_reading()
    
    # Cleanup
    import os
    if os.path.exists("aes_measurements.h5"):
        os.remove("aes_measurements.h5")
    
    print("\n✅ Safe measurement workflow demonstrated!")

def interactive_example():
    """Example that shows interactive confirmation."""
    print("\n=== Interactive Safety Demo ===")
    print("This would normally prompt for confirmation when appending to existing series.")
    print("Try running this interactively to see the prompts in action!")
    
    # Create initial data
    series = Series("interactive_test", traces=[])
    series.open_for_writing("interactive.h5", "test_exp", measurement_id="session_1")
    
    for i in range(3):
        trace = Trace(samples=np.ones(100) * i, timestamp=datetime.now())
        series.add_trace(trace)
    
    series.close_writing()
    print("Created initial series with 3 traces")
    
    # This would normally prompt: "Experiment: existent, Series: existent (3 traces), continue? [y/N]:"
    print("\nTo test interactively, uncomment the following code:")
    print("# series2 = Series('interactive_test', traces=[])")
    print("# series2.open_for_writing('interactive.h5', 'test_exp', mode='a', confirm_append=True)")
    
    # Cleanup
    import os
    if os.path.exists("interactive.h5"):
        os.remove("interactive.h5")

if __name__ == "__main__":
    simple_uuid_workflow()
    measurement_session_example()
    interactive_example()