"""
Schema types and introspection for dataclass-to-HDF5 mapping.

Users define trace schemas as plain dataclasses using Array[T], Scalar[T],
and `bytes` annotations. schema_fields() introspects these into field
descriptors that drive HDF5 dataset creation and read/write logic.
"""

import dataclasses
import json
import typing
from datetime import datetime

import h5py
import numpy as np


T = typing.TypeVar('T')


class Array(typing.Generic[T]):
    """Per-trace array field. T = numpy dtype (np.complex128, np.float32, np.uint8, ...)"""
    pass


class Scalar(typing.Generic[T]):
    """Per-trace scalar field. T = python type (int, float, str, datetime)"""
    pass


class SchemaError(Exception):
    """Raised on schema mismatch between dataclass and HDF5."""
    pass


# Maps Python scalar types to numpy/HDF5 dtypes
_SCALAR_DTYPE_MAP = {
    int: np.dtype('int64'),
    float: np.dtype('float64'),
    str: None,       # uses h5py.string_dtype()
    datetime: None,  # stored as string, converted on read
}


def _get_origin(tp):
    """Get the origin of a generic type (e.g. Array from Array[np.float32])."""
    return getattr(tp, '__origin__', None)


def _get_args(tp):
    """Get the type arguments (e.g. (np.float32,) from Array[np.float32])."""
    return getattr(tp, '__args__', ())


def schema_fields(cls):
    """
    Introspect a dataclass and return field descriptors for HDF5 mapping.

    Returns:
        list of tuples: (name, kind, numpy_dtype, python_read_type)
        - kind: 'array' | 'bytes' | 'scalar'
        - numpy_dtype: np.dtype for HDF5 dataset (None for string types)
        - python_read_type: the Python type to return on read
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    hints = typing.get_type_hints(cls)
    fields = []

    for f in dataclasses.fields(cls):
        ann = hints[f.name]
        origin = _get_origin(ann)
        args = _get_args(ann)

        if origin is Array:
            # Array[np.complex128] -> ('array', np.dtype('complex128'), np.ndarray)
            if not args:
                raise TypeError(f"Array field '{f.name}' must specify a dtype, e.g. Array[np.float32]")
            dt = np.dtype(args[0])
            fields.append((f.name, 'array', dt, np.ndarray))

        elif ann is bytes:
            # bytes -> ('bytes', np.dtype('uint8'), bytes)
            fields.append((f.name, 'bytes', np.dtype('uint8'), bytes))

        elif origin is Scalar:
            if not args:
                raise TypeError(f"Scalar field '{f.name}' must specify a type, e.g. Scalar[int]")
            py_type = args[0]
            if py_type not in _SCALAR_DTYPE_MAP:
                raise TypeError(f"Scalar type {py_type} not supported. Use int, float, str, or datetime.")
            np_dt = _SCALAR_DTYPE_MAP[py_type]
            fields.append((f.name, 'scalar', np_dt, py_type))

        else:
            raise TypeError(
                f"Field '{f.name}' has unsupported annotation {ann}. "
                f"Use Array[dtype], bytes, or Scalar[type]."
            )

    return fields


def _to_hdf5_value(field_desc, value):
    """Convert a Python value to HDF5-writable form based on field descriptor."""
    name, kind, np_dtype, py_type = field_desc

    if kind == 'array':
        return np.asarray(value, dtype=np_dtype)

    elif kind == 'bytes':
        if isinstance(value, (bytes, bytearray)):
            return np.frombuffer(bytes(value), dtype=np.uint8)
        return np.asarray(value, dtype=np.uint8)

    elif kind == 'scalar':
        if py_type is datetime:
            return value.isoformat() if value is not None else ""
        elif py_type is str:
            return value if value is not None else ""
        else:
            return value


def _from_hdf5_value(field_desc, hdf5_value):
    """Convert an HDF5 value back to Python type based on field descriptor."""
    name, kind, np_dtype, py_type = field_desc

    if kind == 'array':
        return np.asarray(hdf5_value, dtype=np_dtype)

    elif kind == 'bytes':
        arr = np.asarray(hdf5_value, dtype=np.uint8)
        return bytes(arr)

    elif kind == 'scalar':
        if py_type is datetime:
            s = hdf5_value
            if isinstance(s, bytes):
                s = s.decode()
            return datetime.fromisoformat(s) if s else None
        elif py_type is str:
            s = hdf5_value
            if isinstance(s, bytes):
                s = s.decode()
            return s if s else None
        elif py_type is int:
            return int(hdf5_value)
        elif py_type is float:
            return float(hdf5_value)
        return hdf5_value


def schema_to_json(cls):
    """Serialize a dataclass schema to JSON string for HDF5 attribute storage."""
    fields = schema_fields(cls)
    entries = []
    for name, kind, np_dtype, py_type in fields:
        entry = {"name": name, "kind": kind}
        if np_dtype is not None:
            entry["dtype"] = str(np_dtype)
        if kind == 'scalar':
            entry["py_type"] = py_type.__name__
        entries.append(entry)
    return json.dumps(entries)


def schema_from_json(json_str):
    """Deserialize schema JSON back to field descriptors (for schemaless reading)."""
    _PY_TYPE_MAP = {'int': int, 'float': float, 'str': str, 'datetime': datetime}
    entries = json.loads(json_str)
    fields = []
    for entry in entries:
        name = entry["name"]
        kind = entry["kind"]
        if kind == 'array':
            np_dtype = np.dtype(entry["dtype"])
            fields.append((name, 'array', np_dtype, np.ndarray))
        elif kind == 'bytes':
            fields.append((name, 'bytes', np.dtype('uint8'), bytes))
        elif kind == 'scalar':
            py_type = _PY_TYPE_MAP[entry["py_type"]]
            np_dtype = _SCALAR_DTYPE_MAP[py_type]
            fields.append((name, 'scalar', np_dtype, py_type))
    return fields


def make_trace_type(json_str):
    """Build a dataclass from stored schema JSON.

    slots=True ensures AttributeError on invalid field access.
    """
    field_descriptors = schema_from_json(json_str)
    dc_fields = []
    for name, kind, np_dtype, py_type in field_descriptors:
        if kind == 'array':
            ann = Array[np_dtype.type]
        elif kind == 'bytes':
            ann = bytes
        elif kind == 'scalar':
            ann = Scalar[py_type]
        dc_fields.append((name, ann, dataclasses.field(default=None)))
    return dataclasses.make_dataclass('DynamicTrace', dc_fields, slots=True)


def validate_schema_match(cls, json_str):
    """Validate that a dataclass schema matches stored HDF5 schema JSON."""
    expected = schema_fields(cls)
    stored = schema_from_json(json_str)

    if len(expected) != len(stored):
        raise SchemaError(
            f"Schema field count mismatch: dataclass has {len(expected)}, "
            f"HDF5 has {len(stored)}"
        )

    for (e_name, e_kind, e_dtype, _), (s_name, s_kind, s_dtype, _) in zip(expected, stored):
        if e_name != s_name:
            raise SchemaError(f"Field name mismatch: expected '{e_name}', got '{s_name}'")
        if e_kind != s_kind:
            raise SchemaError(f"Field '{e_name}' kind mismatch: expected '{e_kind}', got '{s_kind}'")
        if e_dtype != s_dtype:
            raise SchemaError(f"Field '{e_name}' dtype mismatch: expected {e_dtype}, got {s_dtype}")
