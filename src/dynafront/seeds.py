"""Named, process-independent seed streams via numpy SeedSequence.

Python's salted hash() is never used. Child seeds are derived from the master
seed, experiment name, configuration hash, and seed index.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

MASTER_SEED = 20260821


def _u32(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def spawn_key(experiment: str, config_hash: str, seed_index: int, *parts: Any) -> tuple[int, ...]:
    key = [_u32(experiment), _u32(config_hash), int(seed_index) % (2**32)]
    for part in parts:
        if isinstance(part, (bool, np.bool_)):
            key.append(1 if part else 0)
        elif isinstance(part, (int, np.integer)):
            key.append(int(part) % (2**32))
        elif isinstance(part, float):
            key.append(_u32(format(float(part), ".17g")))
        else:
            key.append(_u32(str(part)))
    return tuple(int(v) for v in key)


def seed_sequence(
    master_seed: int,
    experiment: str,
    config_hash: str,
    seed_index: int,
    *parts: Any,
) -> np.random.SeedSequence:
    return np.random.SeedSequence(
        entropy=int(master_seed),
        spawn_key=spawn_key(experiment, config_hash, seed_index, *parts),
    )


def default_rng(
    master_seed: int,
    experiment: str,
    config_hash: str,
    seed_index: int,
    *parts: Any,
) -> np.random.Generator:
    return np.random.default_rng(seed_sequence(master_seed, experiment, config_hash, seed_index, *parts))


def python_seed(
    master_seed: int,
    experiment: str,
    config_hash: str,
    seed_index: int,
    *parts: Any,
) -> int:
    ss = seed_sequence(master_seed, experiment, config_hash, seed_index, *parts)
    return int(ss.generate_state(1, dtype=np.uint32)[0])
