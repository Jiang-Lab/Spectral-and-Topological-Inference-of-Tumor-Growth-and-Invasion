"""Load and verify the four sealed initial-condition realizations."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


IC_NAMES = (
    "Random spaced",
    "One central cluster",
    "Many clusters",
    "Clusters + singles",
)

# SHA-256 of each contiguous float32 125x125 array in the sealed archive.
EXPECTED_ARRAY_SHA256 = {
    "Random spaced": "b7f90f08ac0c50c7b4247146f174d7e148aababcde4bb80bfec3dbf2e8c06739",
    "One central cluster": "d8a652f0ea7085fcf6a5890195358427fd9c6b8590eeec46037ed6d4951bb258",
    "Many clusters": "604a2ca8b54fbefdccb8a94543c345ead8cffd44cac5f26793db8b7d5ee09d9b",
    "Clusters + singles": "8cd916c98588459edd3a5a21f154b43974cbd24c47f62519d2b04666778cc62c",
}


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_initial_conditions(
    path: str | Path,
    *,
    verify: bool = True,
) -> dict[str, np.ndarray]:
    """Return the four IC fields as float64, unchanged from the sealed archive."""
    # No allow_pickle: nothing is deserialized before the hashes are checked.
    with np.load(Path(path)) as archive:
        order = tuple(str(v) for v in archive["order"])
        if order != IC_NAMES:
            raise ValueError(f"unexpected IC order: {order}")
        stored = {name: archive[name].copy() for name in order}

    if verify:
        for name, array in stored.items():
            digest = array_sha256(array)
            if digest != EXPECTED_ARRAY_SHA256[name]:
                raise ValueError(f"IC hash mismatch for {name}: {digest}")

    fields = {name: array.astype(np.float64) for name, array in stored.items()}
    for name, field in fields.items():
        if field.shape != (125, 125):
            raise ValueError(f"{name} has shape {field.shape}, expected (125, 125)")
    return fields

