#!/usr/bin/env python3
"""
Tests for the new schema system: Array, Scalar, bytes, custom dataclasses.
"""

import unittest
import tempfile
import os
import numpy as np
from dataclasses import dataclass
from scam import Array, Scalar, Series, SchemaError
from scam.schema import schema_fields, schema_to_json, schema_from_json, validate_schema_match


@dataclass
class MaskedAESTrace:
    samples:    Array[np.complex128]
    plaintext:  bytes
    ciphertext: bytes
    key:        bytes
    mask:       bytes


@dataclass
class SimpleTrace:
    samples: Array[np.float32]
    label:   Scalar[int]


@dataclass
class FullTrace:
    samples:   Array[np.float64]
    plaintext: bytes
    label:     Scalar[int]
    score:     Scalar[float]
    note:      Scalar[str]


class TestSchemaIntrospection(unittest.TestCase):

    def test_masked_aes_fields(self):
        fields = schema_fields(MaskedAESTrace)
        self.assertEqual(len(fields), 5)
        self.assertEqual(fields[0], ('samples', 'array', np.dtype('complex128'), np.ndarray))
        self.assertEqual(fields[1], ('plaintext', 'bytes', np.dtype('uint8'), bytes))
        self.assertEqual(fields[4], ('mask', 'bytes', np.dtype('uint8'), bytes))

    def test_simple_trace_fields(self):
        fields = schema_fields(SimpleTrace)
        self.assertEqual(len(fields), 2)
        self.assertEqual(fields[0], ('samples', 'array', np.dtype('float32'), np.ndarray))
        self.assertEqual(fields[1], ('label', 'scalar', np.dtype('int64'), int))

    def test_schema_json_roundtrip(self):
        json_str = schema_to_json(MaskedAESTrace)
        fields = schema_from_json(json_str)
        expected = schema_fields(MaskedAESTrace)
        self.assertEqual(len(fields), len(expected))
        for f, e in zip(fields, expected):
            self.assertEqual(f[0], e[0])  # name
            self.assertEqual(f[1], e[1])  # kind
            self.assertEqual(f[2], e[2])  # dtype

    def test_validate_schema_match_ok(self):
        json_str = schema_to_json(MaskedAESTrace)
        validate_schema_match(MaskedAESTrace, json_str)  # should not raise

    def test_validate_schema_mismatch(self):
        json_str = schema_to_json(SimpleTrace)
        with self.assertRaises(SchemaError):
            validate_schema_match(MaskedAESTrace, json_str)


class TestMaskedAESWriteRead(unittest.TestCase):
    """End-to-end test: write and read MaskedAESTrace with streaming."""

    def test_streaming_write_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test.h5")

            s = Series("test", trace_type=MaskedAESTrace)
            s.open_for_writing(filename, "exp1")
            s.add_trace(MaskedAESTrace(
                samples=np.random.randn(1000) + 1j * np.random.randn(1000),
                plaintext=bytes(range(16)),
                ciphertext=b'\x39\x25\x84\x1d\x02\xdc\x09\xfb\xdc\x11\x85\x97\x19\x6a\x0b\x32',
                key=b'>>XMS IS GREAT<<',
                mask=bytes(18),
            ))
            s.close_writing()

            s2 = Series("test", trace_type=MaskedAESTrace)
            s2.open_for_reading(filename, "exp1")
            t = s2[0]
            self.assertIsInstance(t, MaskedAESTrace)
            self.assertEqual(t.samples.dtype, np.complex128)
            self.assertEqual(len(t.samples), 1000)
            self.assertIsInstance(t.plaintext, bytes)
            self.assertEqual(len(t.plaintext), 16)
            self.assertEqual(t.plaintext, bytes(range(16)))
            self.assertIsInstance(t.mask, bytes)
            self.assertEqual(len(t.mask), 18)
            self.assertEqual(t.key, b'>>XMS IS GREAT<<')
            s2.close_reading()

    def test_multiple_traces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "multi.h5")

            s = Series("traces", trace_type=MaskedAESTrace)
            s.open_for_writing(filename, "exp1")

            for i in range(50):
                s.add_trace(MaskedAESTrace(
                    samples=np.ones(500, dtype=np.complex128) * (i + 1j * i),
                    plaintext=bytes([i % 256] * 16),
                    ciphertext=bytes([255 - i % 256] * 16),
                    key=b'0123456789abcdef',
                    mask=bytes(18),
                ))

            s.close_writing()

            s2 = Series("traces", trace_type=MaskedAESTrace)
            s2.open_for_reading(filename, "exp1")
            self.assertEqual(len(s2), 50)

            t10 = s2[10]
            expected = np.ones(500, dtype=np.complex128) * (10 + 10j)
            np.testing.assert_array_equal(t10.samples, expected)
            self.assertEqual(t10.plaintext, bytes([10] * 16))

            matrix = s2.to_matrix()
            self.assertEqual(matrix.shape, (50, 500))
            self.assertEqual(matrix.dtype, np.complex128)

            s2.close_reading()


class TestShapeLock(unittest.TestCase):

    def test_shape_mismatch_rejected(self):
        s = Series("test", trace_type=MaskedAESTrace)
        s.add_trace(MaskedAESTrace(
            samples=np.zeros(1000, dtype=np.complex128),
            plaintext=bytes(16),
            ciphertext=bytes(16),
            key=bytes(16),
            mask=bytes(18),
        ))

        with self.assertRaises(ValueError) as ctx:
            s.add_trace(MaskedAESTrace(
                samples=np.zeros(999, dtype=np.complex128),
                plaintext=bytes(16),
                ciphertext=bytes(16),
                key=bytes(16),
                mask=bytes(18),
            ))
        self.assertIn("samples", str(ctx.exception))
        self.assertIn("locked shape", str(ctx.exception))

    def test_bytes_shape_mismatch_rejected(self):
        s = Series("test", trace_type=MaskedAESTrace)
        s.add_trace(MaskedAESTrace(
            samples=np.zeros(100, dtype=np.complex128),
            plaintext=bytes(16),
            ciphertext=bytes(16),
            key=bytes(16),
            mask=bytes(18),
        ))

        with self.assertRaises(ValueError) as ctx:
            s.add_trace(MaskedAESTrace(
                samples=np.zeros(100, dtype=np.complex128),
                plaintext=bytes(15),  # wrong!
                ciphertext=bytes(16),
                key=bytes(16),
                mask=bytes(18),
            ))
        self.assertIn("plaintext", str(ctx.exception))


class TestTypeLock(unittest.TestCase):

    def test_wrong_type_rejected(self):
        from scam import Trace
        s = Series("test", trace_type=MaskedAESTrace)
        with self.assertRaises(TypeError):
            s.add_trace(Trace(samples=np.zeros(10, dtype=np.float32)))


class TestScalarFields(unittest.TestCase):

    def test_int_scalar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "scalar.h5")

            s = Series("test", trace_type=SimpleTrace)
            s.open_for_writing(filename, "exp1")
            for i in range(10):
                s.add_trace(SimpleTrace(samples=np.ones(5, dtype=np.float32) * i, label=i * 10))
            s.close_writing()

            s2 = Series("test", trace_type=SimpleTrace)
            s2.open_for_reading(filename, "exp1")
            t = s2[3]
            self.assertEqual(t.label, 30)
            self.assertIsInstance(t.label, int)
            np.testing.assert_array_equal(t.samples, np.ones(5, dtype=np.float32) * 3)
            s2.close_reading()

    def test_mixed_scalars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "full.h5")

            s = Series("test", trace_type=FullTrace)
            s.open_for_writing(filename, "exp1")
            s.add_trace(FullTrace(
                samples=np.array([1.0, 2.0, 3.0]),
                plaintext=b'\xab\xcd',
                label=42,
                score=3.14,
                note="hello",
            ))
            s.close_writing()

            s2 = Series("test", trace_type=FullTrace)
            s2.open_for_reading(filename, "exp1")
            t = s2[0]
            self.assertEqual(t.label, 42)
            self.assertAlmostEqual(t.score, 3.14)
            self.assertEqual(t.note, "hello")
            self.assertEqual(t.plaintext, b'\xab\xcd')
            s2.close_reading()


class TestSWMR(unittest.TestCase):
    """Tests for SWMR (Single Writer Multiple Reader) support."""

    def test_swmr_write_then_read(self):
        """Write with SWMR enabled, then read back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "swmr.h5")

            # Write two series, enable SWMR after both have their first trace
            s1 = Series("fixed", trace_type=MaskedAESTrace)
            s1.open_for_writing(filename, "exp1")
            s1.add_trace(MaskedAESTrace(
                samples=np.ones(100, dtype=np.complex128),
                plaintext=bytes(16), ciphertext=bytes(16),
                key=b'0123456789abcdef', mask=bytes(18),
            ))

            s2 = Series("random", trace_type=MaskedAESTrace)
            s2.open_for_writing(filename, "exp1")
            s2.add_trace(MaskedAESTrace(
                samples=np.ones(100, dtype=np.complex128) * 2,
                plaintext=bytes(16), ciphertext=bytes(16),
                key=b'fedcba9876543210', mask=bytes(18),
            ))

            # Both series have datasets — safe to enable SWMR
            s1.enable_swmr()

            # Write more traces after SWMR is active
            for i in range(9):
                s1.add_trace(MaskedAESTrace(
                    samples=np.ones(100, dtype=np.complex128) * (i + 2),
                    plaintext=bytes(16), ciphertext=bytes(16),
                    key=b'0123456789abcdef', mask=bytes(18),
                ))
                s2.add_trace(MaskedAESTrace(
                    samples=np.ones(100, dtype=np.complex128) * (i + 3),
                    plaintext=bytes(16), ciphertext=bytes(16),
                    key=b'fedcba9876543210', mask=bytes(18),
                ))

            s1.close_writing()
            s2.close_writing()

            # Read back and verify
            r1 = Series("fixed", trace_type=MaskedAESTrace)
            r1.open_for_reading(filename, "exp1")
            self.assertEqual(len(r1), 10)
            np.testing.assert_array_equal(r1[0].samples, np.ones(100, dtype=np.complex128))
            r1.close_reading()

            r2 = Series("random", trace_type=MaskedAESTrace)
            r2.open_for_reading(filename, "exp1")
            self.assertEqual(len(r2), 10)
            r2.close_reading()

    def test_swmr_concurrent_read(self):
        """Verify reader can open file while writer has SWMR active."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "swmr_concurrent.h5")

            writer = Series("data", trace_type=SimpleTrace)
            writer.open_for_writing(filename, "exp1")
            writer.add_trace(SimpleTrace(samples=np.ones(50, dtype=np.float32), label=1))
            writer.enable_swmr()

            # Reader opens while writer still has the file
            reader = Series("data", trace_type=SimpleTrace)
            reader.open_for_reading(filename, "exp1", swmr=True)

            # Reader sees what's been flushed
            reader.refresh()
            count = len(reader)
            self.assertGreaterEqual(count, 1)
            t = reader[0]
            self.assertEqual(t.label, 1)

            # Writer adds more
            writer.add_trace(SimpleTrace(samples=np.ones(50, dtype=np.float32) * 2, label=2))
            writer._h5file.flush()

            # Reader refreshes and sees new data
            reader.refresh()
            new_count = len(reader)
            self.assertEqual(new_count, 2)
            t2 = reader[1]
            self.assertEqual(t2.label, 2)

            reader.close_reading()
            writer.close_writing()

    def test_trace_count_from_shape(self):
        """Verify trace count is derived from dataset shape, not attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "shape_count.h5")

            s = Series("test", trace_type=SimpleTrace)
            s.open_for_writing(filename, "exp1")
            s.add_trace(SimpleTrace(samples=np.ones(10, dtype=np.float32), label=0))
            s.add_trace(SimpleTrace(samples=np.ones(10, dtype=np.float32), label=1))

            # Dataset shape == trace count (no trace_count attribute needed)
            self.assertEqual(s._h5group['samples'].shape[0], 2)
            self.assertEqual(s._h5group['label'].shape[0], 2)

            s.enable_swmr()

            # Can still add traces after SWMR — shape grows
            s.add_trace(SimpleTrace(samples=np.ones(10, dtype=np.float32), label=2))
            self.assertEqual(s._h5group['samples'].shape[0], 3)

            s.close_writing()


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error paths."""

    def test_type_lock_rejects_wrong_dataclass(self):
        """Adding a trace of the wrong dataclass type raises TypeError."""
        s = Series("test", trace_type=SimpleTrace)
        with self.assertRaises(TypeError) as ctx:
            s.add_trace(MaskedAESTrace(
                samples=np.ones(10, dtype=np.complex128),
                plaintext=bytes(16), ciphertext=bytes(16),
                key=bytes(16), mask=bytes(18),
            ))
        self.assertIn("SimpleTrace", str(ctx.exception))
        self.assertIn("MaskedAESTrace", str(ctx.exception))

    def test_schema_mismatch_on_read(self):
        """Opening a file with the wrong trace_type raises SchemaError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "mismatch.h5")

            s = Series("test", trace_type=MaskedAESTrace)
            s.open_for_writing(filename, "exp1")
            s.add_trace(MaskedAESTrace(
                samples=np.ones(100, dtype=np.complex128),
                plaintext=bytes(16), ciphertext=bytes(16),
                key=bytes(16), mask=bytes(18),
            ))
            s.close_writing()

            wrong = Series("test", trace_type=SimpleTrace)
            with self.assertRaises(SchemaError):
                wrong.open_for_reading(filename, "exp1")

    def test_remove_trace(self):
        """remove_trace works in MEMORY mode and rejects other modes."""
        s = Series("test", trace_type=SimpleTrace)
        t0 = SimpleTrace(samples=np.ones(5, dtype=np.float32), label=0)
        t1 = SimpleTrace(samples=np.ones(5, dtype=np.float32), label=1)
        t2 = SimpleTrace(samples=np.ones(5, dtype=np.float32), label=2)
        s.add_trace(t0)
        s.add_trace(t1)
        s.add_trace(t2)

        removed = s.remove_trace(1)
        self.assertEqual(removed.label, 1)
        self.assertEqual(len(s), 2)
        self.assertEqual(s[0].label, 0)
        self.assertEqual(s[1].label, 2)

        with self.assertRaises(IndexError):
            s.remove_trace(5)

    def test_empty_series_roundtrip(self):
        """Write zero traces, close, read back — should get empty series."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "empty.h5")

            s = Series("test", trace_type=SimpleTrace)
            s.open_for_writing(filename, "exp1")
            s.close_writing()

            r = Series("test", trace_type=SimpleTrace)
            r.open_for_reading(filename, "exp1")
            self.assertEqual(len(r), 0)
            self.assertEqual(list(r), [])
            r.close_reading()

    def test_to_matrix_with_field_name(self):
        """to_matrix can select a specific Array field by name."""
        @dataclass
        class TwoArrayTrace:
            power:  Array[np.float32]
            em:     Array[np.float64]

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "two_arrays.h5")

            s = Series("test", trace_type=TwoArrayTrace)
            s.open_for_writing(filename, "exp1")
            for i in range(5):
                s.add_trace(TwoArrayTrace(
                    power=np.ones(10, dtype=np.float32) * i,
                    em=np.ones(10, dtype=np.float64) * (i + 100),
                ))
            s.close_writing()

            r = Series("test", trace_type=TwoArrayTrace)
            r.open_for_reading(filename, "exp1")

            # Default: first Array field (power)
            mat_default = r.to_matrix()
            self.assertEqual(mat_default.dtype, np.float32)
            np.testing.assert_array_equal(mat_default[0], np.ones(10, dtype=np.float32) * 0)

            # Explicit field_name
            mat_em = r.to_matrix(field_name="em")
            self.assertEqual(mat_em.dtype, np.float64)
            np.testing.assert_array_equal(mat_em[2], np.ones(10, dtype=np.float64) * 102)

            r.close_reading()

    def test_access_without_open_for_reading(self):
        """Accessing traces from a loaded-but-not-opened series raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "noopen.h5")

            s = Series("test", trace_type=SimpleTrace)
            s.open_for_writing(filename, "exp1")
            s.add_trace(SimpleTrace(samples=np.ones(5, dtype=np.float32), label=0))
            s.close_writing()

            from scam import TraceDB
            db = TraceDB.load_hdf5(filename)
            loaded = db["exp1"]["test"]

            with self.assertRaises(RuntimeError) as ctx:
                _ = loaded[0]
            self.assertIn("open_for_reading", str(ctx.exception))

            with self.assertRaises(RuntimeError) as ctx:
                _ = list(loaded)
            self.assertIn("open_for_reading", str(ctx.exception))

    def test_bytes_high_values_roundtrip(self):
        """Bytes with 0xFF and full range survive write/read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "highbytes.h5")

            all_bytes_key = bytes(range(16))               # 0x00..0x0F
            high_mask = bytes([0xFF] * 18)                 # all 0xFF
            mixed_pt = bytes([0xDE, 0xAD, 0xBE, 0xEF] * 4)  # 16 bytes
            mixed_ct = bytes([i ^ 0xAA for i in range(16)])

            s = Series("test", trace_type=MaskedAESTrace)
            s.open_for_writing(filename, "exp1")
            s.add_trace(MaskedAESTrace(
                samples=np.ones(50, dtype=np.complex128),
                plaintext=mixed_pt,
                ciphertext=mixed_ct,
                key=all_bytes_key,
                mask=high_mask,
            ))
            s.close_writing()

            r = Series("test", trace_type=MaskedAESTrace)
            r.open_for_reading(filename, "exp1")
            t = r[0]
            self.assertEqual(t.plaintext, mixed_pt)
            self.assertEqual(t.ciphertext, mixed_ct)
            self.assertEqual(t.key, all_bytes_key)
            self.assertEqual(t.mask, high_mask)
            r.close_reading()


if __name__ == '__main__':
    unittest.main(verbosity=2)
