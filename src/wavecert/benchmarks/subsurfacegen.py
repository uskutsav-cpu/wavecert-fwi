from __future__ import annotations

from pathlib import Path

import numpy as np

SUBSURFACEGEN_SETTINGS = (
    "penobscot",
    "f3",
    "gom",
    "fault",
    "salt_canopy",
    "seam",
)


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional data dependency
        raise ImportError("h5py is required for SubsurfaceGen HDF5 files") from exc
    return h5py


def load_subsurfacegen_velocity_slice(path: str | Path) -> np.ndarray:
    """Load a SubsurfaceGen 2-D HDF5 velocity slice (key: ``velocity``)."""

    h5py = _h5py()
    with h5py.File(Path(path), "r") as f:
        velocity = np.asarray(f["velocity"][:], dtype=np.float32)
    if velocity.ndim != 2:
        raise ValueError(f"expected 2-D SubsurfaceGen velocity slice, got {velocity.shape}")
    return velocity


def load_subsurfacegen_wavefield(path: str | Path) -> np.ndarray:
    h5py = _h5py()
    with h5py.File(Path(path), "r") as f:
        return np.asarray(f["wavefield"][:], dtype=np.float32)


def load_subsurfacegen_shot_gather_cube(path: str | Path) -> np.ndarray:
    h5py = _h5py()
    with h5py.File(Path(path), "r") as f:
        return np.asarray(f["shot_gather_cube"][:], dtype=np.float32)
