"""Derivative-informed operator training baseline, using train/validation labels only.

This implements established Jacobian supervision as a comparator; it is not a
new WaveCert contribution. Label generation uses the same small discrete PDE as
the forward labels and is an offline training cost, not an online speedup.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from wavecert.data.wavefields import WavefieldDatasetArrays
from wavecert.physics.helmholtz import Helmholtz2D


@dataclass(frozen=True)
class TangentLabels:
    directions: np.ndarray
    tangents: np.ndarray
    metadata: dict


def make_tangent_labels(
    arrays: WavefieldDatasetArrays,
    *,
    role: str,
    seed: int,
) -> TangentLabels:
    """Create one fixed unit-L2 squared-slowness direction per allowed example.

    Exact states are recomputed at the stored float32 model converted to float64
    so the tangent label is consistent with the actual training input. Tangent
    solves are complex128, then real/imaginary channels are saved in float32.
    Calibration and evaluation roles are deliberately rejected.
    """

    if role not in {"train", "validation"}:
        raise ValueError("derivative labels are permitted only for train/model-selection validation")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("jacobian seed must be a nonnegative integer")
    physics = Helmholtz2D(
        shape=arrays.shape, spacing=arrays.spacing,
        damping_width=max(3, min(arrays.shape) // 8), damping_strength=2.0,
    )
    role_code = 0 if role == "train" else 1
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), role_code]))
    directions = rng.normal(size=arrays.models.shape)
    directions /= np.linalg.norm(directions.reshape(len(directions), -1), axis=1)[:, None, None]
    directions = directions.astype(np.float32)
    tangents = np.empty_like(arrays.wavefields, dtype=np.float32)
    for index, model in enumerate(arrays.models):
        m = model.astype(np.float64).reshape(-1)
        q = physics.source(int(arrays.source_indices[index]))
        frequency = float(arrays.frequencies_hz[index])
        state = physics.solve_state(m, q, frequency)
        tangent = physics.solve_tangent(m, state, directions[index].astype(np.float64).reshape(-1), frequency)
        tangent = tangent.reshape(arrays.shape)
        tangents[index, ..., 0] = tangent.real
        tangents[index, ..., 1] = tangent.imag
    metadata = {
        "role": role, "seed": int(seed), "role_seed_code": role_code,
        "n_examples": len(arrays.models), "directions_per_example": 1,
        "direction_distribution": "fixed Gaussian random squared-slowness directions, unit L2 over all model pixels",
        "reference_physics": "same discrete Helmholtz operator as grouped forward data; complex128 solves at stored model values",
        "saved_dtype": "float32 directions and physical real/imaginary tangent channels",
        "models_sha256": hashlib.sha256(np.ascontiguousarray(arrays.models).tobytes()).hexdigest(),
        "directions_sha256": hashlib.sha256(directions.tobytes()).hexdigest(),
        "tangents_sha256": hashlib.sha256(tangents.tobytes()).hexdigest(),
        "offline_label_physics_cost": physics.cost_snapshot(),
        "calibration_or_test_labels_used": False,
    }
    return TangentLabels(directions, tangents, metadata)


def normalized_value_and_tangent(model, encoder, models, sources, frequencies, directions):
    """JVP of normalized output with respect to physical squared slowness.

    Model parameters remain differentiable through the JVP so an outer
    derivative loss can train them. Source indices and frequencies are fixed.
    """

    import torch

    return torch.func.jvp(
        lambda values: model(encoder.encode_batch(values, sources, frequencies)),
        (models,), (directions,),
    )


def normalized_tangent_target(encoder, physical_tangents):
    """Normalize tangent channels by output standard deviation, without centering."""

    import torch

    norm = encoder.normalization
    scale = torch.as_tensor(
        [norm.output_std_real, norm.output_std_imag],
        dtype=physical_tangents.dtype, device=physical_tangents.device,
    )
    return (physical_tangents / scale).permute(0, 3, 1, 2)
