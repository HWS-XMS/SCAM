#!/usr/bin/env python3
"""
Tests for the new always-lazy loading and always-streaming writing.
"""

import unittest
import tempfile
import os
import numpy as np
from datetime import datetime
import h5py
from scam import TraceDB, Experiment, Series, Trace


class TestLazyLoading(unittest.TestCase):
    """Test suite for lazy loading functionality."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test.h5")
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_test_file(self, num_traces=100):
        """Helper to create a test HDF5 file."""
        db = TraceDB()
        exp = db.get_or_create_experiment("test_exp", metadata={"device": "test"})
        series = exp.get_or_create_series("test_series", metadata={"type": "test"})
        
        for i in range(num_traces):
            trace = Trace(
                samples=np.ones(1000) * i,
                timestamp=datetime.now(),
                stimulus=f"stim_{i}",
                response=f"resp_{i}",
                key=f"key_{i}"
            )
            series.add_trace(trace)
        
        db.save_hdf5(self.test_file)
        return num_traces
    
    def test_lazy_loading_default(self):
        """Test that load_hdf5 is lazy by default."""
        # Create test file
        num_traces = self.create_test_file(50)
        
        # Load lazily
        db = TraceDB.load_hdf5(self.test_file)
        series = db["test_exp"]["test_series"]
        
        # Series should have source but no traces in memory
        self.assertEqual(len(series.traces), 0)
        self.assertIsNotNone(series._source)
        self.assertEqual(series._source, (self.test_file, "test_exp"))
    
    def test_lazy_iteration(self):
        """Test lazy iteration over traces."""
        # Create test file
        self.create_test_file(10)
        
        # Load lazily
        db = TraceDB.load_hdf5(self.test_file)
        series = db["test_exp"]["test_series"]
        
        # Open for reading
        series.open_for_reading()
        
        # Iterate and verify
        traces_list = list(series)
        self.assertEqual(len(traces_list), 10)
        
        # Verify trace content
        for i, trace in enumerate(traces_list):
            np.testing.assert_array_equal(trace.samples, np.ones(1000) * i)
            self.assertEqual(trace.stimulus, f"stim_{i}")
            self.assertEqual(trace.response, f"resp_{i}")
            self.assertEqual(trace.key, f"key_{i}")
        
        series.close_reading()
    
    def test_lazy_indexing(self):
        """Test lazy indexing of traces."""
        # Create test file
        self.create_test_file(20)
        
        # Load lazily
        db = TraceDB.load_hdf5(self.test_file)
        series = db["test_exp"]["test_series"]
        
        series.open_for_reading()
        
        # Test single index
        trace5 = series[5]
        np.testing.assert_array_equal(trace5.samples, np.ones(1000) * 5)
        
        # Test slicing
        traces_slice = series[10:15]
        self.assertEqual(len(traces_slice), 5)
        for i, trace in enumerate(traces_slice, 10):
            np.testing.assert_array_equal(trace.samples, np.ones(1000) * i)
        
        series.close_reading()
    
    def test_lazy_length(self):
        """Test getting length without loading data."""
        # Create test file
        num_traces = self.create_test_file(100)
        
        # Load lazily
        db = TraceDB.load_hdf5(self.test_file)
        series = db["test_exp"]["test_series"]
        
        series.open_for_reading()
        
        # Length should work without loading all data
        self.assertEqual(len(series), num_traces)
        
        # Memory traces should still be empty
        self.assertEqual(len(series.traces), 0)
        
        series.close_reading()
    
    def test_to_matrix_lazy(self):
        """Test efficient matrix building from lazy series."""
        # Create test file
        self.create_test_file(50)
        
        # Load lazily
        db = TraceDB.load_hdf5(self.test_file)
        series = db["test_exp"]["test_series"]
        
        series.open_for_reading()
        
        # Build matrix
        matrix = series.to_matrix()
        
        # Verify shape and content
        self.assertEqual(matrix.shape, (50, 1000))
        for i in range(50):
            np.testing.assert_array_equal(matrix[i], np.ones(1000) * i)
        
        series.close_reading()
    
    def test_writing_always_streams(self):
        """Test that writing mode always streams to disk."""
        series = Series("stream_test", traces=[])
        
        # Open for writing
        series.open_for_writing(self.test_file, "exp1")
        
        # Add traces - should auto-persist
        for i in range(10):
            trace = Trace(samples=np.ones(100) * i, timestamp=datetime.now())
            series.add_trace(trace)
        
        # Close writing
        series.close_writing()
        
        # Verify data was written
        with h5py.File(self.test_file, 'r') as f:
            self.assertIn('exp1', f)
            self.assertIn('stream_test', f['exp1'])
            self.assertEqual(f['exp1']['stream_test'].attrs['trace_count'], 10)
            
            # Verify trace data
            samples = f['exp1']['stream_test']['samples'][:]
            for i in range(10):
                np.testing.assert_array_equal(samples[i], np.ones(100) * i)
    
    def test_append_mode(self):
        """Test appending to existing file."""
        # Create initial file
        series1 = Series("series1", traces=[])
        series1.open_for_writing(self.test_file, "exp1")
        for i in range(5):
            series1.add_trace(Trace(samples=np.ones(100) * i))
        series1.close_writing()
        
        # Append more data
        series2 = Series("series2", traces=[])
        series2.open_for_writing(self.test_file, "exp1", mode='a')
        for i in range(5, 10):
            series2.add_trace(Trace(samples=np.ones(100) * i))
        series2.close_writing()
        
        # Verify both series exist
        with h5py.File(self.test_file, 'r') as f:
            self.assertIn('series1', f['exp1'])
            self.assertIn('series2', f['exp1'])
            self.assertEqual(f['exp1']['series1'].attrs['trace_count'], 5)
            self.assertEqual(f['exp1']['series2'].attrs['trace_count'], 5)
    
    def test_mode_transitions(self):
        """Test that mode transitions work correctly."""
        series = Series("test", traces=[])
        
        # Start in MEMORY mode
        from scam.series import SeriesMode
        self.assertEqual(series._mode, SeriesMode.MEMORY)
        
        # Transition to WRITING
        series.open_for_writing(self.test_file, "exp1")
        self.assertEqual(series._mode, SeriesMode.WRITING)
        
        # Cannot open for reading while writing
        with self.assertRaises(RuntimeError):
            series.open_for_reading()
        
        # Close writing
        series.close_writing()
        self.assertEqual(series._mode, SeriesMode.MEMORY)
        
        # Now can open for reading
        series.open_for_reading()
        self.assertEqual(series._mode, SeriesMode.READING)
        
        # Cannot add traces while reading
        with self.assertRaises(RuntimeError):
            series.add_trace(Trace(samples=np.ones(10)))
        
        # Close reading
        series.close_reading()
        self.assertEqual(series._mode, SeriesMode.MEMORY)
    
    def test_large_dataset_efficiency(self):
        """Test that large datasets are handled efficiently."""
        # Create large dataset
        series = Series("large", traces=[])
        series.open_for_writing(self.test_file, "exp1")
        
        # Write 10,000 traces
        for i in range(10000):
            trace = Trace(samples=np.random.randn(5000))
            series.add_trace(trace)
        
        series.close_writing()
        
        # Load lazily
        db = TraceDB.load_hdf5(self.test_file)
        series_lazy = db["exp1"]["large"]
        
        # Memory should be minimal (no traces loaded)
        self.assertEqual(len(series_lazy.traces), 0)
        
        # Open and iterate over just first 10
        series_lazy.open_for_reading()
        count = 0
        for trace in series_lazy:
            count += 1
            if count >= 10:
                break
        
        # Should have iterated 10 traces without loading all 10,000
        self.assertEqual(count, 10)
        
        series_lazy.close_reading()


if __name__ == '__main__':
    unittest.main(verbosity=2)