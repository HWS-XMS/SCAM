#!/usr/bin/env python3
"""
Tests for always-lazy loading and always-streaming writing.
"""

import unittest
import tempfile
import os
import numpy as np
import h5py
from scam import TraceDB, Experiment, Series, Trace


class TestLazyLoading(unittest.TestCase):
    """Test suite for lazy loading functionality."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test.h5")

    def tearDown(self):
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
                samples=np.ones(1000, dtype=np.float32) * i,
                stimulus=f"stim_{i}",
                response=f"resp_{i}",
                key=f"key_{i}",
            )
            series.add_trace(trace)

        db.save_hdf5(self.test_file)
        return num_traces

    def test_lazy_loading_default(self):
        num_traces = self.create_test_file(50)
        db = TraceDB.load_hdf5(self.test_file)
        series = db["test_exp"]["test_series"]
        self.assertEqual(len(series.traces), 0)
        self.assertIsNotNone(series._source)
        self.assertEqual(series._source, (self.test_file, "test_exp"))

    def test_lazy_iteration(self):
        self.create_test_file(10)
        db = TraceDB.load_hdf5(self.test_file)
        series = db["test_exp"]["test_series"]
        series.open_for_reading()

        traces_list = list(series)
        self.assertEqual(len(traces_list), 10)
        for i, trace in enumerate(traces_list):
            np.testing.assert_array_almost_equal(trace.samples, np.ones(1000, dtype=np.float32) * i)
            self.assertEqual(trace.stimulus, f"stim_{i}")
            self.assertEqual(trace.response, f"resp_{i}")
            self.assertEqual(trace.key, f"key_{i}")

        series.close_reading()

    def test_lazy_indexing(self):
        self.create_test_file(20)
        db = TraceDB.load_hdf5(self.test_file)
        series = db["test_exp"]["test_series"]
        series.open_for_reading()

        trace5 = series[5]
        np.testing.assert_array_almost_equal(trace5.samples, np.ones(1000, dtype=np.float32) * 5)

        traces_slice = series[10:15]
        self.assertEqual(len(traces_slice), 5)
        for i, trace in enumerate(traces_slice, 10):
            np.testing.assert_array_almost_equal(trace.samples, np.ones(1000, dtype=np.float32) * i)

        series.close_reading()

    def test_lazy_length(self):
        num_traces = self.create_test_file(100)
        db = TraceDB.load_hdf5(self.test_file)
        series = db["test_exp"]["test_series"]
        series.open_for_reading()
        self.assertEqual(len(series), num_traces)
        self.assertEqual(len(series.traces), 0)
        series.close_reading()

    def test_to_matrix_lazy(self):
        self.create_test_file(50)
        db = TraceDB.load_hdf5(self.test_file)
        series = db["test_exp"]["test_series"]
        series.open_for_reading()

        matrix = series.to_matrix()
        self.assertEqual(matrix.shape, (50, 1000))
        for i in range(50):
            np.testing.assert_array_almost_equal(matrix[i], np.ones(1000, dtype=np.float32) * i)

        series.close_reading()

    def test_writing_always_streams(self):
        series = Series("stream_test", traces=[], trace_type=Trace)
        series.open_for_writing(self.test_file, "exp1")

        for i in range(10):
            trace = Trace(samples=np.ones(100, dtype=np.float32) * i)
            series.add_trace(trace)

        series.close_writing()

        with h5py.File(self.test_file, 'r') as f:
            self.assertIn('exp1', f)
            self.assertIn('stream_test', f['exp1'])
            samples = f['exp1']['stream_test']['samples'][:]
            self.assertEqual(samples.shape[0], 10)
            for i in range(10):
                np.testing.assert_array_almost_equal(samples[i], np.ones(100, dtype=np.float32) * i)

    def test_append_mode(self):
        series1 = Series("series1", traces=[], trace_type=Trace)
        series1.open_for_writing(self.test_file, "exp1")
        for i in range(5):
            series1.add_trace(Trace(samples=np.ones(100, dtype=np.float32) * i))
        series1.close_writing()

        series2 = Series("series2", traces=[], trace_type=Trace)
        series2.open_for_writing(self.test_file, "exp1", mode='a')
        for i in range(5, 10):
            series2.add_trace(Trace(samples=np.ones(100, dtype=np.float32) * i))
        series2.close_writing()

        with h5py.File(self.test_file, 'r') as f:
            self.assertIn('series1', f['exp1'])
            self.assertIn('series2', f['exp1'])
            self.assertEqual(f['exp1']['series1']['samples'].shape[0], 5)
            self.assertEqual(f['exp1']['series2']['samples'].shape[0], 5)

    def test_mode_transitions(self):
        series = Series("test", traces=[], trace_type=Trace)
        from scam.series import SeriesMode
        self.assertEqual(series._mode, SeriesMode.MEMORY)

        series.open_for_writing(self.test_file, "exp1")
        self.assertEqual(series._mode, SeriesMode.WRITING)

        with self.assertRaises(RuntimeError):
            series.open_for_reading()

        series.close_writing()
        self.assertEqual(series._mode, SeriesMode.MEMORY)

        series.open_for_reading()
        self.assertEqual(series._mode, SeriesMode.READING)

        with self.assertRaises(RuntimeError):
            series.add_trace(Trace(samples=np.ones(10, dtype=np.float32)))

        series.close_reading()
        self.assertEqual(series._mode, SeriesMode.MEMORY)

    def test_large_dataset_efficiency(self):
        series = Series("large", traces=[], trace_type=Trace)
        series.open_for_writing(self.test_file, "exp1")

        for i in range(10000):
            trace = Trace(samples=np.random.randn(5000).astype(np.float32))
            series.add_trace(trace)

        series.close_writing()

        db = TraceDB.load_hdf5(self.test_file)
        series_lazy = db["exp1"]["large"]
        self.assertEqual(len(series_lazy.traces), 0)

        series_lazy.open_for_reading()
        count = 0
        for trace in series_lazy:
            count += 1
            if count >= 10:
                break

        self.assertEqual(count, 10)
        series_lazy.close_reading()


if __name__ == '__main__':
    unittest.main(verbosity=2)
