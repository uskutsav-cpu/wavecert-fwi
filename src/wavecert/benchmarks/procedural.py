from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

GEOLOGY_SETTINGS = ("penobscot", "f3", "gom", "fault", "salt_canopy", "seam")


def _base(shape: tuple[int, int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nz, nx = shape
    z = np.linspace(0.0, 1.0, nz)[:, None]
    x = np.linspace(-1.0, 1.0, nx)[None, :]
    velocity = np.broadcast_to(1.82 + 0.56 * z, shape).copy()
    velocity += rng.uniform(-0.025, 0.025)
    return velocity, z, x


def procedural_geology(
    setting: str,
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Compact stress-test proxies inspired by SubsurfaceGen setting classes.

    These are *not* SubsurfaceGen samples. They intentionally isolate familiar
    geological motifs so OOD logic can be tested offline before multi-GB/TB
    external datasets are attached.
    """

    if setting not in GEOLOGY_SETTINGS:
        raise ValueError(f"unknown geology setting: {setting}")
    v, z, x = _base(shape, rng)

    # Shared stratigraphy with setting-specific deformation.
    warp = np.zeros(shape)
    if setting in {"penobscot", "f3"}:
        warp = 0.04 * np.sin(2.0 * np.pi * x + rng.uniform(0, 2 * np.pi))
    elif setting in {"gom", "salt_canopy", "seam"}:
        warp = 0.07 * np.sin(np.pi * x) + 0.03 * np.sin(3.0 * np.pi * x)
    for k in range(4):
        horizon = 0.18 + 0.16 * k + warp
        amp = rng.uniform(-0.09, 0.11)
        v += amp * np.tanh((z - horizon) / 0.028)

    if setting == "penobscot":
        # Gentle basin and channels.
        v -= 0.12 * np.exp(-((x + 0.25) ** 2 / 0.22 + (z - 0.48) ** 2 / 0.035))
    elif setting == "f3":
        # A few strong normal faults.
        for xpos, throw in [(-0.45, 0.09), (0.05, -0.07), (0.48, 0.06)]:
            v += (x > xpos) * 0.08 * np.tanh((z - 0.45 - throw) / 0.05)
    elif setting == "gom":
        # Deep basin plus compact high-velocity diapir.
        v += 0.15 * z**2
        v += 0.30 * np.exp(-((x + 0.18) ** 2 / 0.08 + (z - 0.68) ** 2 / 0.06))
    elif setting == "fault":
        # Dense fault network.
        for xpos in np.linspace(-0.75, 0.75, 7):
            throw = rng.uniform(-0.08, 0.08)
            v += (x > xpos) * rng.uniform(-0.055, 0.055) * np.tanh((z - 0.5 - throw) / 0.035)
    elif setting == "salt_canopy":
        canopy = np.exp(-((z - (0.42 + 0.09 * np.sin(2.2 * np.pi * x))) ** 2) / 0.018)
        canopy *= np.exp(-(x**2) / 0.75)
        v += 0.55 * canopy
    elif setting == "seam":
        # Stronger salt and flank complexity.
        dome = np.exp(-((x + 0.12) ** 2 / 0.11 + (z - 0.56) ** 2 / 0.11))
        wing = np.exp(-((x - 0.52) ** 2 / 0.035 + (z - 0.66) ** 2 / 0.13))
        v += 0.52 * dome + 0.38 * wing
        v -= 0.12 * np.exp(-((x + 0.58) ** 2 / 0.05 + (z - 0.50) ** 2 / 0.09))

    texture = ndi.gaussian_filter(rng.normal(size=shape), sigma=1.2, mode="reflect")
    texture /= max(float(np.std(texture)), 1e-12)
    v += 0.014 * texture
    return np.clip(v, 1.60, 3.05)
