"""Inputs, isolated workers, and numerical checks for a batched FNO API control."""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock, SurveyGeometry
from wavecert.provenance import sha256_file, source_digest
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate


def make_neural_control_survey(batch_size: int):
    """Generate a fixed common model and nonzero synthetic misfit outside timing."""
    if isinstance(batch_size, bool) or batch_size not in (4, 12, 40):
        raise ValueError("the declared control batch sizes are 4, 12, and 40")
    shape = (24, 24)
    z, x = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]), indexing="ij")
    velocity = 2.2 + 0.15*z + 0.08*np.cos(np.pi*x)
    truth_velocity = velocity + 0.05*np.exp(-((z-0.2)**2+(x+0.15)**2)/0.12)
    model, truth = (1/velocity**2).reshape(-1), (1/truth_velocity**2).reshape(-1)
    geometry = SurveyGeometry.from_grid(shape, n_sources=10, n_receivers=20,
        source_depth=2, receiver_depth=2, margin=2)
    if batch_size == 4:
        sources = (geometry.source_indices[2], geometry.source_indices[7])
        frequencies = (3.0, 4.0)
    elif batch_size == 12:
        sources = tuple(geometry.source_indices[i] for i in (0, 4, 9))
        frequencies = (2.5, 3.0, 3.5, 4.0)
    else:
        sources, frequencies = geometry.source_indices, (2.5, 3.0, 3.5, 4.0)
    physics = Helmholtz2D(shape=shape, spacing=0.05, damping_width=3, damping_strength=2.0)
    blocks = tuple(SurveyBlock(int(source), frequency,
        physics.restrict(physics.solve_state(truth, physics.source(source), frequency), geometry.receiver_indices),
        f"{int(source)}@{frequency}") for source in sources for frequency in frequencies)
    input_record = {
        "shape": list(shape), "spacing": 0.05, "batch_size": batch_size,
        "common_model_sha256": hashlib.sha256(model.tobytes()).hexdigest(),
        "truth_model_sha256": hashlib.sha256(truth.tobytes()).hexdigest(),
        "source_indices": [b.source_index for b in blocks],
        "frequencies_hz": [b.frequency_hz for b in blocks],
        "receiver_indices": [int(r) for r in geometry.receiver_indices],
        "observed_sha256": hashlib.sha256(np.stack([b.observed for b in blocks]).tobytes()).hexdigest(),
        "observation_generation_costs_excluded_from_timing": physics.cost_snapshot(),
    }
    return model, blocks, geometry.receiver_indices, input_record


def evaluate_neural_control(surrogate, m, blocks, receivers, method):
    if method == "batch":
        return surrogate.survey_objective_and_gradient(m, blocks, receivers)
    if method != "serial":
        raise ValueError("method must be batch or serial")
    value, gradient, states, residuals = 0.0, np.zeros(m.size), [], []
    for block in blocks:
        source = np.zeros(m.size)
        source[block.source_index] = 400.0  # existing point-source mask encoding
        objective, grad, state, residual = surrogate.objective_and_gradient(
            m, source, block.observed, receivers, block.frequency_hz,
        )
        value += objective
        gradient += grad
        states.append(state)
        residuals.append(residual)
    return value, gradient, np.stack(states), np.stack(residuals)


def _payload(result, receivers):
    value, gradient, states, residuals = result
    sampled = states[:, list(receivers)]
    return {"objective": value, "gradient": gradient.tolist(),
        "receiver_state_real": sampled.real.tolist(), "receiver_state_imag": sampled.imag.tolist(),
        "residual_real": residuals.real.tolist(), "residual_imag": residuals.imag.tolist()}


def neural_control_agreement(reference: dict, candidate: dict) -> dict:
    """Compare full gradients and actual complex receiver outputs, not hashes."""
    result = {}
    objective_error = abs(reference["objective"]-candidate["objective"])
    result["objective_absolute_error"] = float(objective_error)
    result["objective_relative_error"] = float(objective_error/max(abs(reference["objective"]), 1e-15))
    pairs = {
        "gradient": (np.asarray(reference["gradient"]), np.asarray(candidate["gradient"])),
        "receiver_state": (
            np.asarray(reference["receiver_state_real"])+1j*np.asarray(reference["receiver_state_imag"]),
            np.asarray(candidate["receiver_state_real"])+1j*np.asarray(candidate["receiver_state_imag"])),
        "residual": (np.asarray(reference["residual_real"])+1j*np.asarray(reference["residual_imag"]),
                     np.asarray(candidate["residual_real"])+1j*np.asarray(candidate["residual_imag"])),
    }
    if not np.isfinite(reference["objective"]) or not np.isfinite(candidate["objective"]):
        raise ValueError("non-finite objective in numerical agreement check")
    for name, (expected, actual) in pairs.items():
        if expected.shape != actual.shape or not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
            raise ValueError("non-finite or incompatible output in numerical agreement check")
        difference = expected-actual
        result[f"{name}_relative_l2_error"] = float(np.linalg.norm(difference)/max(np.linalg.norm(expected), 1e-15))
        result[f"{name}_maximum_absolute_error"] = float(np.max(np.abs(difference)))
    result["passes_float32_tolerances"] = bool(
        result["objective_relative_error"] <= 5e-6
        and result["gradient_relative_l2_error"] <= 2e-4
        and result["receiver_state_relative_l2_error"] <= 3e-5
        and result["residual_relative_l2_error"] <= 3e-5
    )
    return result


def run_neural_control_worker(
    *, checkpoint: str | Path, method: str, batch_size: int, warm_repeats: int,
    repository_root: str | Path,
) -> dict:
    """Run one method in a fresh process; checkpoint load/synthesis are untimed."""
    import torch
    if isinstance(warm_repeats, bool) or not isinstance(warm_repeats, int) or warm_repeats < 1:
        raise ValueError("warm_repeats must be a positive integer")
    torch.set_num_threads(1)
    checkpoint, root = Path(checkpoint), Path(repository_root)
    before_hash, checkpoint_hash = source_digest(root), sha256_file(checkpoint)
    surrogate = TrainedFNOWavefieldSurrogate.load(checkpoint, device="cpu")
    if surrogate.shape != (24, 24):
        raise ValueError("this fixed benchmark requires a 24x24 checkpoint")
    m, blocks, receivers, input_record = make_neural_control_survey(batch_size)

    def evaluate():
        return evaluate_neural_control(surrogate, m, blocks, receivers, method)

    start = perf_counter()
    cold = evaluate()
    cold_seconds = perf_counter()-start
    evaluate()  # one explicit untimed warmup after the first invocation
    warm_seconds, agreements = [], []
    cold_payload = _payload(cold, receivers)
    for _ in range(warm_repeats):
        start = perf_counter()
        warm = evaluate()
        warm_seconds.append(perf_counter()-start)
        agreements.append(neural_control_agreement(cold_payload, _payload(warm, receivers)))
    if not all(a["passes_float32_tolerances"] for a in agreements):
        raise RuntimeError("cold and warm outputs disagree beyond declared float32 tolerances")
    return {
        "method": method, "batch_size": batch_size, "torch_threads": torch.get_num_threads(),
        "checkpoint_sha256": checkpoint_hash, "source_sha256": before_hash,
        "source_unchanged_during_worker": source_digest(root) == before_hash,
        "checkpoint_unchanged_during_worker": sha256_file(checkpoint) == checkpoint_hash,
        "input_record": input_record, "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds, "cold_output": cold_payload,
        "warm_output": _payload(warm, receivers), "cold_warm_agreements": agreements,
    }
