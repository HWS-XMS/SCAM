"""
Test suite for SCA data model components.

Tests core functionality including:
- Trace creation and timestamps
- Series management and HDF5 streaming
- Experiment organization
- TraceDB persistence
- Warning system for convenience methods
"""

import unittest
import warnings
import tempfile
import os
import numpy as np
from datetime import datetime
from sca_datamodel import TraceDB, Experiment, Series, Trace


class TestTrace(unittest.TestCase):
    """Test Trace class functionality."""
    
    def test_trace_auto_timestamp(self):
        """Test that traces automatically get timestamps."""
        before = datetime.now()
        trace = Trace(samples=np.array([1, 2, 3]))
        after = datetime.now()
        
        self.assertIsInstance(trace.timestamp, datetime)
        self.assertGreaterEqual(trace.timestamp, before)
        self.assertLessEqual(trace.timestamp, after)
    
    def test_trace_explicit_timestamp(self):
        """Test that explicit timestamps are preserved."""
        specific_time = datetime(2023, 1, 1, 12, 0, 0)
        trace = Trace(samples=np.array([1, 2, 3]), timestamp=specific_time)
        self.assertEqual(trace.timestamp, specific_time)
    
    def test_trace_with_metadata(self):
        """Test trace creation with stimulus, response, key."""
        trace = Trace(
            samples=np.array([1, 2, 3]),
            stimulus="plaintext",
            response="ciphertext", 
            key="secret_key"
        )
        
        self.assertEqual(trace.stimulus, "plaintext")
        self.assertEqual(trace.response, "ciphertext")
        self.assertEqual(trace.key, "secret_key")


class TestSeries(unittest.TestCase):
    """Test Series class functionality."""
    
    def setUp(self):
        self.series = Series("test_series", traces=[])
        self.trace1 = Trace(samples=np.array([1, 2, 3]))
        self.trace2 = Trace(samples=np.array([4, 5, 6]))
        self.trace_wrong_shape = Trace(samples=np.array([1, 2, 3, 4]))
    
    def test_series_creation(self):
        """Test basic series creation."""
        self.assertEqual(self.series.name, "test_series")
        self.assertEqual(len(self.series), 0)
    
    def test_add_trace(self):
        """Test adding traces to series."""
        self.series.add_trace(self.trace1)
        self.assertEqual(len(self.series), 1)
        
        self.series.add_trace(self.trace2)
        self.assertEqual(len(self.series), 2)
    
    def test_trace_shape_validation(self):
        """Test that traces must have compatible shapes."""
        self.series.add_trace(self.trace1)
        
        with self.assertRaises(ValueError):
            self.series.add_trace(self.trace_wrong_shape)
    
    def test_series_indexing(self):
        """Test series indexing and iteration."""
        self.series.add_trace(self.trace1)
        self.series.add_trace(self.trace2)
        
        self.assertEqual(self.series[0], self.trace1)
        self.assertEqual(self.series[1], self.trace2)
        
        traces = list(self.series)
        self.assertEqual(len(traces), 2)
        self.assertEqual(traces[0], self.trace1)
    
    def test_series_streaming_hdf5(self):
        """Test streaming HDF5 functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test_stream.h5")
            
            self.series.open_hdf5_stream(filename, "test_experiment")
            
            for i in range(10):
                trace = Trace(samples=np.array([i, i+1, i+2]))
                self.series.append_trace_to_stream(trace)
            
            self.series.flush_stream()
            self.series.close_stream()
            
            self.assertTrue(os.path.exists(filename))
            
            db = TraceDB.load_hdf5(filename)
            loaded_series = db["test_experiment"]["test_series"]
            self.assertEqual(len(loaded_series), 10)


class TestExperiment(unittest.TestCase):
    """Test Experiment class functionality."""
    
    def setUp(self):
        self.experiment = Experiment("test_exp", series=[])
        self.series1 = Series("series1", traces=[])
        self.series2 = Series("series2", traces=[])
    
    def test_experiment_creation(self):
        """Test basic experiment creation."""
        self.assertEqual(self.experiment.name, "test_exp")
        self.assertEqual(len(self.experiment), 0)
    
    def test_add_series(self):
        """Test adding series to experiment."""
        self.experiment.add_series(self.series1)
        self.assertEqual(len(self.experiment), 1)
        
        self.experiment.add_series(self.series2)
        self.assertEqual(len(self.experiment), 2)
    
    def test_duplicate_series_name(self):
        """Test that duplicate series names are rejected."""
        duplicate_series = Series("series1", traces=[])
        
        self.experiment.add_series(self.series1)
        with self.assertRaises(ValueError):
            self.experiment.add_series(duplicate_series)
    
    def test_experiment_indexing(self):
        """Test experiment indexing by name and index."""
        self.experiment.add_series(self.series1)
        self.experiment.add_series(self.series2)
        
        self.assertEqual(self.experiment["series1"], self.series1)
        self.assertEqual(self.experiment["series2"], self.series2)
        self.assertEqual(self.experiment[0], self.series1)
        self.assertEqual(self.experiment[1], self.series2)
    
    def test_get_or_create_series(self):
        """Test get_or_create_series method."""
        series = self.experiment.get_or_create_series("new_series", metadata={"test": "value"})
        self.assertEqual(series.name, "new_series")
        self.assertEqual(series.metadata["test"], "value")
        self.assertEqual(len(self.experiment), 1)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            same_series = self.experiment.get_or_create_series("new_series")
            self.assertEqual(len(w), 1)
            self.assertIn("already exists", str(w[0].message))
            self.assertIs(series, same_series)


class TestTraceDB(unittest.TestCase):
    """Test TraceDB class functionality."""
    
    def setUp(self):
        self.db = TraceDB()
        self.experiment = Experiment("test_exp", series=[])
    
    def test_db_creation(self):
        """Test basic database creation."""
        self.assertEqual(len(self.db), 0)
    
    def test_add_experiment(self):
        """Test adding experiments to database."""
        self.db.add_experiment(self.experiment)
        self.assertEqual(len(self.db), 1)
        self.assertEqual(self.db["test_exp"], self.experiment)
    
    def test_duplicate_experiment_name(self):
        """Test that duplicate experiment names are rejected."""
        duplicate_exp = Experiment("test_exp", series=[])
        
        self.db.add_experiment(self.experiment)
        with self.assertRaises(ValueError):
            self.db.add_experiment(duplicate_exp)
    
    def test_get_or_create_experiment(self):
        """Test get_or_create_experiment method."""
        exp = self.db.get_or_create_experiment("new_exp", metadata={"device": "STM32"})
        self.assertEqual(exp.name, "new_exp")
        self.assertEqual(exp.metadata["device"], "STM32")
        self.assertEqual(len(self.db), 1)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            same_exp = self.db.get_or_create_experiment("new_exp")
            self.assertEqual(len(w), 1)
            self.assertIn("already exists", str(w[0].message))
            self.assertIs(exp, same_exp)
    
    def test_hdf5_save_load(self):
        """Test complete HDF5 save and load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test_db.h5")
            
            exp = Experiment("test_experiment", series=[])
            exp.metadata = {"device": "STM32F4", "algorithm": "AES-128"}
            
            series = Series("power_traces", traces=[])
            series.metadata = {"probe": "CT-1", "sampling_rate": 1000000}
            
            for i in range(5):
                trace = Trace(
                    samples=np.array([i, i+1, i+2]),
                    stimulus=f"input_{i}",
                    response=f"output_{i}",
                    key=f"key_{i}"
                )
                series.add_trace(trace)
            
            exp.add_series(series)
            self.db.add_experiment(exp)
            
            self.db.save_hdf5(filename)
            self.assertTrue(os.path.exists(filename))
            
            loaded_db = TraceDB.load_hdf5(filename)
            
            self.assertEqual(len(loaded_db), 1)
            loaded_exp = loaded_db["test_experiment"]
            self.assertEqual(loaded_exp.metadata["device"], "STM32F4")
            
            loaded_series = loaded_exp["power_traces"]
            self.assertEqual(len(loaded_series), 5)
            self.assertEqual(loaded_series.metadata["probe"], "CT-1")
            
            loaded_trace = loaded_series[0]
            np.testing.assert_array_equal(loaded_trace.samples, np.array([0, 1, 2]))
            self.assertEqual(loaded_trace.stimulus, "input_0")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def test_complete_workflow(self):
        """Test complete data collection and analysis workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "integration_test.h5")
            
            db = TraceDB()
            exp = db.get_or_create_experiment("aes_attack", metadata={"device": "STM32"})
            series = exp.get_or_create_series("power_traces", metadata={"probe": "CT-1"})
            
            for i in range(20):
                measurement = np.random.random(1000)
                trace = Trace(
                    samples=measurement,
                    stimulus=f"plaintext_{i:02d}",
                    response=f"ciphertext_{i:02d}"
                )
                series.add_trace(trace)
            
            db.save_hdf5(filename)
            
            analysis_db = TraceDB.load_hdf5(filename)
            analysis_series = analysis_db["aes_attack"]["power_traces"]
            
            self.assertEqual(len(analysis_series), 20)
            
            for i, trace in enumerate(analysis_series):
                self.assertEqual(len(trace.samples), 1000)
                self.assertEqual(trace.stimulus, f"plaintext_{i:02d}")
                self.assertEqual(trace.response, f"ciphertext_{i:02d}")


if __name__ == "__main__":
    unittest.main()