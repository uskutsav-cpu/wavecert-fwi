from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

# Public compact Marmousi model used by the reproducible SiameseFWI materials.
# The raw file is float32 and has 117 x 567 samples (265,356 bytes).
MARMOUSI_PUBLIC_URL = (
    "https://raw.githubusercontent.com/DeepWave-KAUST/"
    "Siamese_FWI-pub/main/data/mar_big_117_567.bin"
)


def load_marmousi_binary(
    path: str | Path,
    *,
    shape: tuple[int, int] = (117, 567),
    dtype: str | np.dtype = np.float32,
) -> np.ndarray:
    """Load the public compact Marmousi binary used by SiameseFWI.

    The function intentionally does not download data at import time. This
    keeps CI offline/reproducible while the accompanying script can fetch the
    public file on a networked research machine.
    """

    path = Path(path)
    values = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"expected {expected} samples for shape {shape}, got {values.size}")
    velocity = values.reshape(shape)
    if not np.isfinite(velocity).all() or np.any(velocity <= 0):
        raise ValueError("Marmousi velocity must be finite and strictly positive")
    return velocity.astype(np.float64)


def downsample_velocity(velocity: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=float)
    zoom = (shape[0] / velocity.shape[0], shape[1] / velocity.shape[1])
    return ndi.zoom(velocity, zoom=zoom, order=1, mode="nearest")


def marmousi_proxy(shape: tuple[int, int] = (64, 128)) -> np.ndarray:
    """Deterministic Marmousi-inspired fallback used only when data are absent.

    This is explicitly a *proxy*, not the Marmousi model. It gives CI and
    offline development a strongly layered/faulted model without fabricating a
    claim that the real public data were evaluated.
    """

    nz, nx = shape
    z = np.linspace(0.0, 1.0, nz)[:, None]
    x = np.linspace(-1.0, 1.0, nx)[None, :]
    velocity = np.broadcast_to(1.72 + 0.76 * z, shape).copy()
    # Several warped stratigraphic bands.
    for k, (amp, width, phase) in enumerate(
        [(0.15, 0.030, 0.0), (-0.11, 0.045, 0.8), (0.18, 0.035, 1.7), (-0.08, 0.028, 2.3)]
    ):
        horizon = 0.22 + 0.15 * k + 0.045 * np.sin((1.7 + 0.2 * k) * np.pi * x + phase)
        velocity += amp * np.exp(-((z - horizon) ** 2) / width)
    # Two fault offsets.
    for xpos, throw in [(-0.35, 0.10), (0.28, -0.08)]:
        side = x > xpos
        velocity += side * 0.10 * np.tanh((z - (0.48 + throw)) / 0.05)
    # Local high/low velocity lenses.
    velocity += 0.30 * np.exp(-((x + 0.10) ** 2 / 0.10 + (z - 0.66) ** 2 / 0.025))
    velocity -= 0.16 * np.exp(-((x - 0.52) ** 2 / 0.06 + (z - 0.45) ** 2 / 0.035))
    return np.clip(velocity, 1.60, 3.05)
