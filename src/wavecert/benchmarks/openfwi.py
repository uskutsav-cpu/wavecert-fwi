from __future__ import annotations

from pathlib import Path

import numpy as np


def load_openfwi_velocity_batch(path: str | Path) -> np.ndarray:
    """Load an official OpenFWI 2-D velocity batch.

    OpenFWI's public 2-D Vel/Fault/Style datasets store velocity files as
    NumPy batches with canonical shape ``(N, 1, 70, 70)``. This adapter also
    accepts ``(N, H, W)`` for derived subsets and always returns ``(N,H,W)``.
    """

    arr = np.load(Path(path), mmap_mode="r")
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"expected OpenFWI velocity batch (N,1,H,W) or (N,H,W), got {arr.shape}")
    return np.asarray(arr)


def load_openfwi_seismic_batch(path: str | Path) -> np.ndarray:
    """Load OpenFWI seismic data, canonically shaped ``(N,5,1000,70)``."""

    arr = np.load(Path(path), mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"expected OpenFWI seismic batch (N,S,T,R), got {arr.shape}")
    return np.asarray(arr)
