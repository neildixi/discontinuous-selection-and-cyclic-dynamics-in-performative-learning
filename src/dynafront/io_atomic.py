"""Atomic writes: temp file, fsync, rename. Never overwrite a validated checkpoint."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TMP_MARKERS = (".tmp", ".partial", ".inprogress")


def is_temporary_path(path: Path | str) -> bool:
    name = Path(path).name.lower()
    return name.endswith(TMP_MARKERS) or ".tmp." in name or ".inprogress." in name


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path | str, data: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="._dynafront_", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return sha256_bytes(data)


def atomic_write_text(path: Path | str, text: str) -> str:
    return atomic_write_bytes(Path(path), text.encode("utf-8"))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)!r}")


def atomic_write_json(path: Path | str, obj: Any, *, indent: int = 2) -> str:
    payload = json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n"
    return atomic_write_text(path, payload)


def atomic_write_csv(path: Path | str, frame: pd.DataFrame) -> str:
    buf = io.StringIO()
    frame.to_csv(buf, index=False)
    return atomic_write_text(path, buf.getvalue())



def atomic_write_npz(path: Path | str, **arrays: Any) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return atomic_write_bytes(path, buf.getvalue())


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
