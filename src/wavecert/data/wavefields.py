from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

from wavecert.experiments.synthetic import velocity_to_m
from wavecert.physics.helmholtz import Helmholtz2D, SurveyGeometry

Array = np.ndarray


@dataclass(frozen=True)
class WavefieldDatasetArrays:
    models: Array
    source_indices: Array
    frequencies_hz: Array
    wavefields: Array
    receiver_indices: Array
    shape: tuple[int, int]
    spacing: float

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            models=self.models,
            source_indices=self.source_indices,
            frequencies_hz=self.frequencies_hz,
            wavefields=self.wavefields,
            receiver_indices=self.receiver_indices,
            shape=np.asarray(self.shape, dtype=int),
            spacing=np.asarray(self.spacing, dtype=float),
        )

    @classmethod
    def load(cls, path: str | Path) -> WavefieldDatasetArrays:
        z = np.load(path)
        shape = tuple(int(v) for v in z["shape"])
        return cls(
            models=z["models"],
            source_indices=z["source_indices"],
            frequencies_hz=z["frequencies_hz"],
            wavefields=z["wavefields"],
            receiver_indices=z["receiver_indices"],
            shape=shape,  # type: ignore[arg-type]
            spacing=float(z["spacing"]),
        )


def random_velocity_model(shape: tuple[int, int], rng: np.random.Generator) -> Array:
    """Generate a smooth, positive 2-D acoustic velocity model.

    The family intentionally contains layered structure, low-amplitude smooth
    texture, and several compact anomalies. It is simple enough for fast
    controlled operator-learning experiments while being richer than a single
    hand-crafted model.
    """

    nz, nx = shape
    z = np.linspace(0.0, 1.0, nz)[:, None]
    x = np.linspace(-1.0, 1.0, nx)[None, :]
    velocity = np.broadcast_to(1.95 + (0.32 + rng.uniform(-0.06, 0.08)) * z, shape).copy()

    # Smooth stratigraphic perturbations.
    for k in range(1, 4):
        amp = rng.uniform(-0.055, 0.055)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        velocity += amp * np.sin(2.0 * np.pi * k * z + phase)

    # Local anomalies with random polarity and aspect ratio.
    zz = 2.0 * z - 1.0
    for _ in range(int(rng.integers(2, 5))):
        cx = rng.uniform(-0.75, 0.75)
        cz = rng.uniform(-0.65, 0.7)
        sx = rng.uniform(0.10, 0.32)
        sz = rng.uniform(0.08, 0.28)
        amp = rng.uniform(-0.22, 0.30)
        velocity += amp * np.exp(-((x - cx) ** 2 / (2 * sx**2) + (zz - cz) ** 2 / (2 * sz**2)))

    texture = ndi.gaussian_filter(rng.normal(size=shape), sigma=rng.uniform(1.2, 2.6), mode="reflect")
    texture /= max(float(np.std(texture)), 1e-12)
    velocity += rng.uniform(0.01, 0.035) * texture
    return np.clip(velocity, 1.65, 2.85)


def build_wavefield_dataset(
    *,
    n_samples: int,
    shape: tuple[int, int] = (24, 24),
    spacing: float = 0.05,
    frequencies_hz: tuple[float, ...] = (2.5, 3.0, 3.5, 4.0),
    n_source_positions: int = 6,
    n_receivers: int = 20,
    seed: int = 20260910,
) -> WavefieldDatasetArrays:
    rng = np.random.default_rng(seed)
    physics = Helmholtz2D(
        shape=shape,
        spacing=spacing,
        damping_width=max(3, min(shape) // 8),
        damping_strength=2.0,
    )
    geometry = SurveyGeometry.from_grid(
        shape,
        n_sources=n_source_positions,
        n_receivers=n_receivers,
        source_depth=2,
        receiver_depth=2,
        margin=2,
    )

    models = np.empty((n_samples, *shape), dtype=np.float32)
    sources = np.empty(n_samples, dtype=np.int64)
    frequencies = np.empty(n_samples, dtype=np.float32)
    wavefields = np.empty((n_samples, *shape, 2), dtype=np.float32)

    for i in range(n_samples):
        velocity = random_velocity_model(shape, rng)
        m = velocity_to_m(velocity).astype(np.float64).reshape(-1)
        source_index = int(rng.choice(geometry.source_indices))
        frequency = float(rng.choice(frequencies_hz))
        q = physics.source(source_index)
        u = physics.solve_state(m, q, frequency).reshape(shape)

        models[i] = m.reshape(shape).astype(np.float32)
        sources[i] = source_index
        frequencies[i] = frequency
        wavefields[i, ..., 0] = u.real.astype(np.float32)
        wavefields[i, ..., 1] = u.imag.astype(np.float32)

    return WavefieldDatasetArrays(
        models=models,
        source_indices=sources,
        frequencies_hz=frequencies,
        wavefields=wavefields,
        receiver_indices=np.asarray(geometry.receiver_indices, dtype=np.int64),
        shape=shape,
        spacing=float(spacing),
    )
