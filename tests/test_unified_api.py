#!/usr/bin/env python3
"""
Tests for the unified streaming/batch API.
"""

import unittest
import tempfile
import os
import numpy as np
from datetime import datetime
import warnings
from scam import Series, Trace, TraceDB
import h5py


class TestUnifiedAPI(unittest.TestCase):
    """Test suite for unified add_trace API."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test.h5")
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_test_trace(self, index=0):
        """Helper to create a test trace."""
        return Trace(
            samples=np.random.randn(100),
            timestamp=datetime.now(),
            stimulus=f"stim_{index}",
            response=f"resp_{index}",
            key=f"key_{index}"
        )
    
    def test_batch_mode_default(self):
        """Test that add_trace defaults to batch mode (no persistence)."""
        series = Series("test", traces=[])
        
        # Add traces without persist flag (default batch mode)
        for i in range(10):
            trace = self.create_test_trace(i)
            series.add_trace(trace)
        
        # Verify traces are in memory
        self.assertEqual(len(series.traces), 10)
        
        # Verify no streaming is active
        self.assertFalse(series.is_streaming())
    
    def test_streaming_mode_persist(self):
        """Test streaming mode with persist=True."""
        series = Series("test", traces=[])
        
        # Enable streaming
        series.enable_streaming(self.test_file, "experiment")
        self.assertTrue(series.is_streaming())
        
        # Add traces with persist=True
        for i in range(10):
            trace = self.create_test_trace(i)
            series.add_trace(trace, persist=True)
        
        # Verify traces are in memory AND persisted
        self.assertEqual(len(series.traces), 10)
        
        # Close and verify file
        series.disable_streaming()
        self.assertFalse(series.is_streaming())
        
        # Load and verify persisted data
        with h5py.File(self.test_file, 'r') as f:
            self.assertIn('experiment', f)
            self.assertIn('test', f['experiment'])
            self.assertEqual(f['experiment']['test'].attrs['trace_count'], 10)
    
    def test_persist_without_stream_warning(self):
        """Test that persist=True without streaming gives a warning."""
        series = Series("test", traces=[])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trace = self.create_test_trace()
            series.add_trace(trace, persist=True)
            
            # Check for warning
            self.assertTrue(len(w) > 0)
            self.assertIn("no stream is open", str(w[0].message))
    
    def test_auto_flush_interval(self):
        """Test automatic flushing based on interval."""
        series = Series("test", traces=[])
        
        # Enable streaming with small auto-flush interval
        series.enable_streaming(
            self.test_file, 
            "experiment",
            auto_flush_interval=3
        )
        
        # Track flush count
        original_flush = series.flush_stream
        flush_count = 0
        
        def counting_flush():
            nonlocal flush_count
            flush_count += 1
            return original_flush()
        
        series.flush_stream = counting_flush
        
        # Add 10 traces - should trigger 3 auto-flushes
        for i in range(10):
            trace = self.create_test_trace(i)
            series.add_trace(trace, persist=True)
        
        # Should have flushed at traces 3, 6, 9
        self.assertEqual(flush_count, 3)
        
        series.disable_streaming()
    
    def test_manual_flush_override(self):
        """Test manual flush override."""
        series = Series("test", traces=[])
        
        # Enable streaming with large auto-flush interval
        series.enable_streaming(
            self.test_file,
            "experiment", 
            auto_flush_interval=100
        )
        
        # Track flush count
        flush_count = 0
        original_flush = series.flush_stream
        
        def counting_flush():
            nonlocal flush_count
            flush_count += 1
            return original_flush()
        
        series.flush_stream = counting_flush
        
        # Add traces without triggering auto-flush
        for i in range(5):
            trace = self.create_test_trace(i)
            series.add_trace(trace, persist=True)
        
        # No auto-flush yet
        self.assertEqual(flush_count, 0)
        
        # Force manual flush
        trace = self.create_test_trace(99)
        series.add_trace(trace, persist=True, flush=True)
        
        # Should have flushed once
        self.assertEqual(flush_count, 1)
        
        series.disable_streaming()
    
    def test_context_manager(self):
        """Test context manager for automatic stream closure."""
        series = Series("test", traces=[])
        series.enable_streaming(self.test_file, "experiment")
        
        with series:
            self.assertTrue(series.is_streaming())
            trace = self.create_test_trace()
            series.add_trace(trace, persist=True)
        
        # Stream should be closed after context
        self.assertFalse(series.is_streaming())
    
    def test_hybrid_workflow(self):
        """Test switching between batch and streaming modes."""
        series = Series("test", traces=[])
        
        # Phase 1: Batch mode
        for i in range(5):
            trace = self.create_test_trace(i)
            series.add_trace(trace)  # No persist
        
        self.assertEqual(len(series.traces), 5)
        self.assertFalse(series.is_streaming())
        
        # Phase 2: Enable streaming
        series.enable_streaming(self.test_file, "experiment")
        
        for i in range(5, 10):
            trace = self.create_test_trace(i)
            series.add_trace(trace, persist=True)
        
        self.assertEqual(len(series.traces), 10)
        self.assertTrue(series.is_streaming())
        
        # Phase 3: Disable streaming, back to batch
        series.disable_streaming()
        
        for i in range(10, 15):
            trace = self.create_test_trace(i)
            series.add_trace(trace)  # No persist
        
        self.assertEqual(len(series.traces), 15)
        self.assertFalse(series.is_streaming())
        
        # Verify only middle 5 traces were persisted
        with h5py.File(self.test_file, 'r') as f:
            self.assertEqual(f['experiment']['test'].attrs['trace_count'], 5)
    
    def test_shape_validation_with_persist(self):
        """Test that shape validation still works with persist flag."""
        series = Series("test", traces=[])
        series.enable_streaming(self.test_file, "experiment")
        
        # Add first trace
        trace1 = Trace(samples=np.random.randn(100))
        series.add_trace(trace1, persist=True)
        
        # Try to add trace with different shape
        trace2 = Trace(samples=np.random.randn(200))  # Different shape
        
        with self.assertRaises(ValueError) as context:
            series.add_trace(trace2, persist=True)
        
        self.assertIn("shape", str(context.exception))
        
        series.disable_streaming()
    
    def test_legacy_api_deprecation(self):
        """Test that legacy methods show deprecation warnings."""
        series = Series("test", traces=[])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Test deprecated open_hdf5_stream
            series.open_hdf5_stream(self.test_file, "experiment")
            self.assertTrue(any("deprecated" in str(warning.message) for warning in w))
            
            # Test deprecated append_trace_to_stream
            trace = self.create_test_trace()
            series.append_trace_to_stream(trace)
            self.assertTrue(any("deprecated" in str(warning.message) for warning in w))
        
        series.close_stream()
    
    def test_file_already_exists(self):
        """Test that streaming to existing file overwrites it."""
        series1 = Series("series1", traces=[])
        
        # First series creates file
        series1.enable_streaming(self.test_file, "experiment")
        for i in range(5):
            series1.add_trace(self.create_test_trace(i), persist=True)
        series1.disable_streaming()
        
        # Verify file exists with data
        with h5py.File(self.test_file, 'r') as f:
            self.assertEqual(f['experiment']['series1'].attrs['trace_count'], 5)
        
        # Second series overwrites file (current behavior)
        series2 = Series("series2", traces=[])
        series2.enable_streaming(self.test_file, "experiment2")
        for i in range(3):
            series2.add_trace(self.create_test_trace(i), persist=True)
        series2.disable_streaming()
        
        # Verify first series data is gone (overwritten)
        with h5py.File(self.test_file, 'r') as f:
            self.assertNotIn('experiment', f)  # Old experiment gone
            self.assertIn('experiment2', f)  # New experiment exists
            self.assertEqual(f['experiment2']['series2'].attrs['trace_count'], 3)
    
    def test_empty_series_streaming(self):
        """Test streaming with no traces added."""
        series = Series("test", traces=[])
        
        series.enable_streaming(self.test_file, "experiment")
        # Don't add any traces
        series.disable_streaming()
        
        # File should exist but with empty series
        with h5py.File(self.test_file, 'r') as f:
            self.assertEqual(f['experiment']['test'].attrs['trace_count'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)