"""Additive source/frequency-batched adapter for the frozen real POD model.

The existing POD implementation is unchanged. Matrix RHS columns and lifts
replace repeated calls, using its same projected operator, LU and adjoint.
Reference private APIs are bound to exact source bytes at import time.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wavecert.surrogates.pod import PODGalerkinSurrogate

REFERENCE_SOURCE_SHA256 = {
    "physics/helmholtz.py": "caf51b58da24ed6d3633bda4eaaab804714f5c7932d9d5f343644763bc158131",
    "surrogates/pod.py": "b613dc2e62ba24f65a9c7587d40c3d1af4f7fc0d2e5fcab8daee649908bcd5ca",
    "physics/coarse_stability.py": "53e6ab1754966f63b97c62da6c3996b7babbd59648808df6f9412295e46c658a",
    "certificates/coarse_adjoint.py": "3d4f2d89c37116312f4f80c967ff4bd20d4be474efe60fe65461d758880ee565",
}


def _verify_reference_sources():
    package = Path(__file__).resolve().parents[1]
    targets = {"physics/helmholtz.py": ("Helmholtz2D", "operator"),
               "surrogates/pod.py": ("PODGalerkinSurrogate", "__init__"),
               "physics/coarse_stability.py": ("FastCoarseStability", "__init__"),
               "certificates/coarse_adjoint.py": ("PODCoarseAdjointCertifier", "__init__")}
    for name, expected in REFERENCE_SOURCE_SHA256.items():
        reference = importlib.import_module("wavecert."+name[:-3].replace("/", "."))
        intended = (package / name).resolve()
        actual = Path(reference.__file__).resolve()
        class_name, method_name = targets[name]
        implementation = getattr(getattr(reference, class_name), method_name)
        if actual != intended or Path(implementation.__code__.co_filename).resolve() != intended:
            raise RuntimeError(f"batched private adapter rejects a cached reference from a different checkout: {name}")
        if hashlib.sha256(actual.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"batched private adapter requires the frozen reference source: {name}")


_verify_reference_sources()


def _freeze(values):
    values.setflags(write=False)
    return values


def _delta(before, after):
    return {key: int(value-before.get(key, 0)) for key, value in after.items()
            if key not in {"full_dimension", "reduced_dimension"}}


def _survey_inputs(surrogate, m, blocks, receiver_indices):
    if type(surrogate) is not PODGalerkinSurrogate:
        raise TypeError("batched survey fields require the unchanged PODGalerkinSurrogate")
    raw = np.asarray(m)
    if raw.shape != (surrogate.n,) or np.iscomplexobj(raw):
        raise ValueError("the common model must be a real flat vector matching the POD grid")
    model = surrogate._model(raw)
    receivers = tuple(receiver_indices)
    if not receivers or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                            or not 0 <= i < surrogate.n for i in receivers):
        raise ValueError("receivers must be nonempty valid integer grid positions")
    blocks, seen, groups = tuple(blocks), set(), {}
    if not blocks:
        raise ValueError("at least one source/frequency block is required")
    for i, block in enumerate(blocks):
        source, frequency = block.source_index, block.frequency_hz
        if (isinstance(source, (bool, np.bool_)) or not isinstance(source, (int, np.integer))
                or not 0 <= source < surrogate.n):
            raise ValueError("source indices must be valid integers")
        surrogate._frequency(frequency)
        observed = np.asarray(block.observed)
        if observed.shape != (len(receivers),) or not np.all(np.isfinite(observed)):
            raise ValueError("each observed vector must be finite and match the common receivers")
        key = (int(source), float(frequency))
        if key in seen:
            raise ValueError("source/frequency blocks must be unique")
        seen.add(key)
        groups.setdefault(float(frequency), []).append(i)
    return model, blocks, tuple(int(i) for i in receivers), {f: tuple(groups[f]) for f in sorted(groups)}


@dataclass(frozen=True)
class PODSurveyFields:
    objective: float | None
    aggregate_gradient: np.ndarray | None
    block_states: np.ndarray
    block_adjoints: np.ndarray
    block_gradients: np.ndarray | None
    block_objectives: np.ndarray | None
    blocks: tuple
    receiver_indices: tuple[int, ...]
    group_indices: dict
    costs: dict
    batch_dispatches: dict

    @property
    def gradient(self):
        return self.aggregate_gradient

    @property
    def reference_source_sha256(self):
        return dict(REFERENCE_SOURCE_SHA256)


def batched_pod_survey_fields(surrogate, m, blocks, receiver_indices) -> PODSurveyFields:
    """Return ordered (grid, block) physical fields using frequency matrix RHS.

    Every RHS/projection/lift column is charged to the unchanged POD counters.
    Separate dispatch counts describe actual Python/BLAS solve/transform calls.
    A nonfinite physical contraction yields a null aggregate proposal; no
    stationarity or fine-PDE accuracy is inferred from it.
    """
    return _batched_pod_survey_fields(surrogate, m, blocks, receiver_indices, compute_physical=True)


def _batched_pod_survey_fields(surrogate, m, blocks, receiver_indices, *, compute_physical):
    model, blocks, receivers, groups = _survey_inputs(surrogate, m, blocks, receiver_indices)
    before, dispatch = surrogate.cost_snapshot(), {}

    def count(name):
        dispatch[name] = dispatch.get(name, 0)+1

    states = np.empty((surrogate.n, len(blocks)), dtype=np.complex128)
    adjoints = np.empty_like(states)
    gradients, objectives = np.empty(states.shape, dtype=float), np.empty(len(blocks))
    for frequency, indices in groups.items():
        columns = len(indices)
        count("pod_prepare_dispatches")
        _, omega2, factor = surrogate._prepare(model, frequency)
        source = np.zeros((surrogate.n, columns), dtype=np.complex128)
        source[[blocks[i].source_index for i in indices], np.arange(columns)] = 1/surrogate.physics_config["spacing"]**2
        count("source_rhs_assembly_dispatches")
        surrogate._count("source_projections", columns)
        count("pod_source_projection_dispatches")
        rhs = surrogate.basis.T @ source
        count("pod_forward_solve_dispatches")
        coefficient = surrogate._solve(factor, rhs, category="reduced_forward_rhs")
        surrogate._count("fine_state_lifts", columns)
        count("pod_state_lift_dispatches")
        state = surrogate.basis @ coefficient
        observed = np.stack([blocks[i].observed for i in indices], axis=1)
        residual = state[list(receivers), :]-observed
        injected = np.zeros_like(state, dtype=np.complex128)
        np.add.at(injected, np.asarray(receivers), residual)
        count("receiver_injection_dispatches")
        surrogate._count("adjoint_rhs_projections", columns)
        count("pod_adjoint_projection_dispatches")
        rhs = surrogate.basis.T @ injected
        count("pod_adjoint_solve_dispatches")
        coefficient = surrogate._solve(factor, rhs, category="reduced_adjoint_rhs", adjoint=True)
        surrogate._count("fine_adjoint_lifts", columns)
        count("pod_adjoint_lift_dispatches")
        adjoint = surrogate.basis @ coefficient
        states[:, indices], adjoints[:, indices] = state, adjoint
        if compute_physical:
            count("physical_gradient_dispatches")
            with np.errstate(over="ignore", invalid="ignore"):
                gradients[:, indices] = omega2*np.real(np.conjugate(adjoint)*state)
                objectives[list(indices)] = .5*np.sum(np.conjugate(residual)*residual, axis=0).real
    # Retain the scalar adapter's original block accumulation order.
    aggregate, objective = np.zeros(surrogate.n), 0.0
    if compute_physical:
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(len(blocks)):
                aggregate += gradients[:, i]
                objective += float(objectives[i])
    valid = compute_physical and np.all(np.isfinite(aggregate)) and np.all(np.isfinite(gradients)) and np.isfinite(objective)
    return PODSurveyFields(float(objective) if compute_physical and np.isfinite(objective) else None,
        _freeze(aggregate) if valid else None, _freeze(states), _freeze(adjoints),
        _freeze(gradients) if compute_physical else None, _freeze(objectives) if compute_physical else None,
        blocks, receivers, groups,
        {"pod": _delta(before, surrogate.cost_snapshot()),
         "dimensions": {"fine": surrogate.n, "pod": surrogate.reduced_dimension, "blocks": len(blocks)}}, dispatch)
