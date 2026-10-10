"""Stable configuration hashing. Never use Python's salted hash()."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _canonicalize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float):
        return format(float(obj), ".17g")
    if isinstance(obj, np.floating):
        return format(float(obj), ".17g")
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _canonicalize(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, np.ndarray):
        return [_canonicalize(x) for x in obj.tolist()]
    return str(obj)


def canonical_json(obj: Any) -> str:
    return json.dumps(_canonicalize(obj), separators=(",", ":"), ensure_ascii=False)


def config_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def short_hash(obj: Any, n: int = 12) -> str:
    return config_hash(obj)[:n]
