from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi

from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock, SurveyGeometry


@dataclass(frozen=True)
class SyntheticProblem:
    physics: Helmholtz2D
    geometry: SurveyGeometry
    m_true: np.ndarray
    m0: np.ndarray
    blocks: tuple[SurveyBlock, ...]


def velocity_to_m(velocity: np.ndarray) -> np.ndarray:
    return 1.0 / np.asarray(velocity, dtype=float) ** 2


def build_synthetic_problem(
    *,
    shape: tuple[int, int] = (14, 14),
    spacing: float = 0.05,
    frequencies_hz: tuple[float, ...] = (3.0, 4.0),
    n_sources: int = 2,
    n_receivers: int = 10,
) -> SyntheticProblem:
    physics = Helmholtz2D(shape=shape, spacing=spacing, damping_width=3, damping_strength=2.0)
    geometry = SurveyGeometry.from_grid(
        shape,
        n_sources=n_sources,
        n_receivers=n_receivers,
        source_depth=2,
        receiver_depth=2,
        margin=2,
    )

    nz, nx = shape
    zz, xx = np.meshgrid(np.linspace(-1, 1, nz), np.linspace(-1, 1, nx), indexing="ij")
    base_v = 2.1 + 0.15 * (zz + 1.0)
    anomaly_fast = 0.45 * np.exp(-((xx - 0.25) ** 2 / 0.12 + (zz - 0.15) ** 2 / 0.08))
    anomaly_slow = -0.28 * np.exp(-((xx + 0.35) ** 2 / 0.08 + (zz + 0.15) ** 2 / 0.12))
    v_true = base_v + anomaly_fast + anomaly_slow
    v0 = ndi.gaussian_filter(v_true, sigma=2.5, mode="nearest")

    m_true = velocity_to_m(v_true).reshape(-1)
    m0 = velocity_to_m(v0).reshape(-1)

    blocks: list[SurveyBlock] = []
    for s_idx in geometry.source_indices:
        q = physics.source(s_idx)
        for f in frequencies_hz:
            u = physics.solve_state(m_true, q, f)
            observed = physics.restrict(u, geometry.receiver_indices)
            blocks.append(
                SurveyBlock(
                    source_index=s_idx,
                    frequency_hz=float(f),
                    observed=observed,
                    label=f"src{s_idx}-f{f:g}",
                )
            )

    return SyntheticProblem(physics, geometry, m_true, m0, tuple(blocks))
