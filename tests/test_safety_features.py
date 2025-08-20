#!/usr/bin/env python3
"""
Tests for the new safety features in TraceDB.save_hdf5()
"""

import unittest
import os
import tempfile
import warnings
import numpy as np
from datetime import datetime
from scam import TraceDB, Experiment, Series, Trace


class TestSafetyFeatures(unittest.TestCase):
    """Test suite for data safety features."""
    
    def setUp(self):
        """Create a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test.h5")
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_sample_db(self, exp_name="TestExp", series_name="TestSeries"):
        """Helper to create a sample database with data."""
        db = TraceDB()
        exp = db.get_or_create_experiment(exp_name, metadata={"test": "data"})
        series = exp.get_or_create_series(series_name, metadata={"type": "test"})
        
        for i in range(5):
            trace = Trace(
                samples=np.random.randn(100),
                timestamp=datetime.now(),
                stimulus=f"stim_{i}",
                response=f"resp_{i}",
                key=f"key_{i}"
            )
            series.add_trace(trace)
        
        return db
    
    def test_default_mode_is_update(self):
        """Test that default save mode is 'update' which merges data."""
        # Create and save first database
        db1 = self.create_sample_db("Exp1", "Series1")
        db1.save_hdf5(self.test_file)
        
        # Create and save second database with different series
        db2 = self.create_sample_db("Exp1", "Series2")
        db2.save_hdf5(self.test_file)  # Should merge, not overwrite
        
        # Load and verify both series exist
        db_loaded = TraceDB.load_hdf5(self.test_file)
        self.assertEqual(len(db_loaded.experiments), 1)
        exp = db_loaded.experiments["Exp1"]
        self.assertEqual(len(exp.series), 2)
        
        series_names = {s.name for s in exp.series}
        self.assertIn("Series1", series_names)
        self.assertIn("Series2", series_names)
    
    def test_overwrite_protection(self):
        """Test that overwrite mode requires explicit confirmation."""
        # Create and save initial database
        db1 = self.create_sample_db()
        db1.save_hdf5(self.test_file)
        
        # Try to overwrite without permission - should fail
        db2 = self.create_sample_db("NewExp", "NewSeries")
        with self.assertRaises(ValueError) as context:
            db2.save_hdf5(self.test_file, mode='overwrite')
        
        self.assertIn("already exists", str(context.exception))
        self.assertIn("overwrite_ok=True", str(context.exception))
        
        # Verify original data is still intact
        db_check = TraceDB.load_hdf5(self.test_file)
        self.assertIn("TestExp", db_check.experiments)
        self.assertNotIn("NewExp", db_check.experiments)
    
    def test_explicit_overwrite(self):
        """Test that explicit overwrite works with overwrite_ok=True."""
        # Create and save initial database
        db1 = self.create_sample_db("OldExp", "OldSeries")
        db1.save_hdf5(self.test_file)
        
        # Explicitly overwrite with new data
        db2 = self.create_sample_db("NewExp", "NewSeries")
        db2.save_hdf5(self.test_file, mode='overwrite', overwrite_ok=True)
        
        # Verify old data is gone and new data exists
        db_loaded = TraceDB.load_hdf5(self.test_file)
        self.assertNotIn("OldExp", db_loaded.experiments)
        self.assertIn("NewExp", db_loaded.experiments)
    
    def test_duplicate_series_warning(self):
        """Test that duplicate series names generate warnings."""
        # Create and save initial database
        db1 = self.create_sample_db("Exp1", "DupSeries")
        db1.save_hdf5(self.test_file)
        
        # Try to add another series with the same name
        db2 = self.create_sample_db("Exp1", "DupSeries")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            db2.save_hdf5(self.test_file, mode='update')
            
            # Check that a warning was issued
            self.assertTrue(len(w) > 0)
            warning_messages = [str(warning.message) for warning in w]
            self.assertTrue(
                any("already exists" in msg for msg in warning_messages),
                f"Expected warning about duplicate series, got: {warning_messages}"
            )
        
        # Verify original data is preserved
        db_loaded = TraceDB.load_hdf5(self.test_file)
        exp = db_loaded.experiments["Exp1"]
        self.assertEqual(len(exp.series), 1)  # Only one series (original preserved)
    
    def test_merge_multiple_experiments(self):
        """Test merging multiple experiments into one file."""
        # Create first experiment
        db1 = self.create_sample_db("Exp1", "Series1")
        db1.save_hdf5(self.test_file)
        
        # Create second experiment
        db2 = self.create_sample_db("Exp2", "Series2")
        db2.save_hdf5(self.test_file)  # Should merge
        
        # Create third experiment
        db3 = self.create_sample_db("Exp3", "Series3")
        db3.save_hdf5(self.test_file)  # Should merge
        
        # Verify all three experiments exist
        db_loaded = TraceDB.load_hdf5(self.test_file)
        self.assertEqual(len(db_loaded.experiments), 3)
        self.assertIn("Exp1", db_loaded.experiments)
        self.assertIn("Exp2", db_loaded.experiments)
        self.assertIn("Exp3", db_loaded.experiments)
    
    def test_merge_series_within_experiment(self):
        """Test adding new series to existing experiment."""
        # Create experiment with first series
        db1 = self.create_sample_db("ExpA", "Series1")
        db1.save_hdf5(self.test_file)
        
        # Add second series to same experiment
        db2 = self.create_sample_db("ExpA", "Series2")
        db2.save_hdf5(self.test_file)
        
        # Add third series to same experiment
        db3 = self.create_sample_db("ExpA", "Series3")
        db3.save_hdf5(self.test_file)
        
        # Verify all series exist in the same experiment
        db_loaded = TraceDB.load_hdf5(self.test_file)
        self.assertEqual(len(db_loaded.experiments), 1)
        exp = db_loaded.experiments["ExpA"]
        self.assertEqual(len(exp.series), 3)
        
        series_names = {s.name for s in exp.series}
        self.assertEqual(series_names, {"Series1", "Series2", "Series3"})
    
    def test_invalid_mode_raises_error(self):
        """Test that invalid mode parameter raises error."""
        db = self.create_sample_db()
        
        with self.assertRaises(ValueError) as context:
            db.save_hdf5(self.test_file, mode='invalid')
        
        self.assertIn("mode must be", str(context.exception))
    
    def test_data_integrity_after_merge(self):
        """Test that data integrity is maintained after merging."""
        # Create and save first dataset with known data
        db1 = TraceDB()
        exp1 = db1.get_or_create_experiment("IntegrityTest")
        series1 = exp1.get_or_create_series("Series1")
        
        # Add traces with specific values
        for i in range(3):
            trace = Trace(
                samples=np.ones(10) * i,  # Identifiable pattern
                timestamp=datetime.now(),
                stimulus=f"stim1_{i}",
                key=f"key1_{i}"
            )
            series1.add_trace(trace)
        
        db1.save_hdf5(self.test_file)
        
        # Create and save second dataset
        db2 = TraceDB()
        exp2 = db2.get_or_create_experiment("IntegrityTest")
        series2 = exp2.get_or_create_series("Series2")
        
        for i in range(3):
            trace = Trace(
                samples=np.ones(10) * (i + 10),  # Different pattern
                timestamp=datetime.now(),
                stimulus=f"stim2_{i}",
                key=f"key2_{i}"
            )
            series2.add_trace(trace)
        
        db2.save_hdf5(self.test_file)
        
        # Load and verify both datasets are intact
        db_loaded = TraceDB.load_hdf5(self.test_file)
        exp = db_loaded.experiments["IntegrityTest"]
        
        # Find series by name
        series1_loaded = next(s for s in exp.series if s.name == "Series1")
        series2_loaded = next(s for s in exp.series if s.name == "Series2")
        
        # Verify Series1 data
        self.assertEqual(len(series1_loaded.traces), 3)
        for i, trace in enumerate(series1_loaded.traces):
            np.testing.assert_array_equal(trace.samples, np.ones(10) * i)
            self.assertEqual(trace.stimulus, f"stim1_{i}")
            self.assertEqual(trace.key, f"key1_{i}")
        
        # Verify Series2 data
        self.assertEqual(len(series2_loaded.traces), 3)
        for i, trace in enumerate(series2_loaded.traces):
            np.testing.assert_array_equal(trace.samples, np.ones(10) * (i + 10))
            self.assertEqual(trace.stimulus, f"stim2_{i}")
            self.assertEqual(trace.key, f"key2_{i}")
    
    def test_empty_database_save(self):
        """Test saving an empty database."""
        db = TraceDB()
        db.save_hdf5(self.test_file)
        
        # Should create file and be loadable
        self.assertTrue(os.path.exists(self.test_file))
        db_loaded = TraceDB.load_hdf5(self.test_file)
        self.assertEqual(len(db_loaded.experiments), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)