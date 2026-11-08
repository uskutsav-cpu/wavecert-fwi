"""Pure geometry, source, folding and diagnostic helpers; no wave execution."""
from __future__ import annotations

import hashlib

import numpy as np


def array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def geometry(protocol):
    config = protocol["geometry"]
    h = protocol["physics"]["spacing_coarse_m"]
    source = np.asarray([config["source_coarse_node_x_z"]], dtype=np.float64) * h
    receivers = np.column_stack((config["receiver_coarse_x_nodes"],
                                np.full(len(config["receiver_coarse_x_nodes"]), config["receiver_z_node"]))) * h
    if any(np.array_equal(source[0], receiver) for receiver in receivers):
        raise ValueError("source/receiver coincidence is forbidden")
    return source, receivers


def refine_2x(velocity):
    a = np.asarray(velocity, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 2:
        raise ValueError("2D velocity with at least two nodes per axis required")
    fine = np.empty(tuple(2 * n - 1 for n in a.shape), dtype=np.float64)
    fine[::2, ::2] = a
    fine[1::2, ::2] = .5 * (a[:-1] + a[1:])
    fine[::2, 1::2] = .5 * (a[:, :-1] + a[:, 1:])
    fine[1::2, 1::2] = .25 * (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:])
    return fine


def independent_edge_fold(padded, shape, width):
    padded = np.asarray(padded, dtype=np.float64)
    if padded.shape != tuple(n + 2 * width for n in shape):
        raise ValueError("padded gradient shape mismatch")
    physical = np.zeros(shape, dtype=np.float64)
    for x in range(padded.shape[0]):
        px = min(max(x - width, 0), shape[0] - 1)
        for z in range(padded.shape[1]):
            pz = min(max(z - width, 0), shape[1] - 1)
            physical[px, pz] += padded[x, z]
    return physical


def relative_record(error, reference):
    numerator, denominator = float(np.linalg.norm(error)), float(np.linalg.norm(reference))
    finite = bool(np.isfinite(numerator) and np.isfinite(denominator))
    return dict(numerator=numerator if finite else None, denominator=denominator if finite else None,
                ratio=numerator / denominator if finite and denominator > 0 else None,
                zero_denominator=denominator == 0, finite=finite)


def exact_common_indices(times_a, mask_a, times_b, mask_b):
    times_a, times_b = np.asarray(times_a), np.asarray(times_b)
    lookup = {float(value): i for i, value in enumerate(times_b) if mask_b[i]}
    ia = [i for i, value in enumerate(times_a) if mask_a[i] and float(value) in lookup]
    ib = [lookup[float(times_a[i])] for i in ia]
    if not ia or not np.array_equal(times_a[ia], times_b[ib]):
        raise ValueError("no exact common computed samples")
    return np.asarray(ia), np.asarray(ib)


def waveform_comparison(a, b):
    ia, ib = exact_common_indices(a["times_ms"], a["computed_mask"], b["times_ms"], b["computed_mask"])
    error, reference = a["receiver"][ia] - b["receiver"][ib], b["receiver"][ib]
    return dict(aggregate=relative_record(error, reference),
        per_receiver=[relative_record(error[:, i], reference[:, i]) for i in range(reference.shape[1])],
        exact_common_samples=len(ia), first_time_ms=float(a["times_ms"][ia[0]]),
        last_time_ms=float(a["times_ms"][ia[-1]]), physical_acceptance=None)


def parameter_direction(model, name):
    model = np.asarray(model, dtype=np.float64)
    mask = np.zeros(model.shape, dtype=bool)
    if name == "interior":
        mask[4:-4, 4:-4] = True
    elif name == "edge":
        mask[0] = mask[-1] = True
        mask[:, 0] = mask[:, -1] = True
    else:
        raise ValueError("unknown frozen direction")
    values = model * mask
    scale = float(np.linalg.norm(values))
    if scale == 0:
        raise ValueError("direction has no positive parameter support")
    return values / scale, scale, mask


def derivative_diagnostic(adjoint, finite_difference, gradient_norm, direction_norm):
    normscale = float(gradient_norm * direction_norm)
    informative = bool(normscale > 0 and abs(adjoint) > 1e-8 * normscale)
    error = abs(float(finite_difference) - float(adjoint))
    denominator = max(abs(float(adjoint)), abs(float(finite_difference)))
    return dict(adjoint_derivative=float(adjoint), finite_difference=float(finite_difference), absolute_error=error,
        informative=informative, relative_error=error / denominator if informative and denominator else None,
        normalized_absolute_error=error / normscale if normscale else None,
        scope="near-zero directional derivative is uninformative, never a zero-error pass")


def held_source(adapter, scale):
    """Capture geometry.src exactly once; caller passes this object explicitly."""
    source = adapter.geometry.src
    source.data[:] *= float(scale)
    times = np.asarray(source._time_range.time_values, dtype=np.float64)
    before = dict(source=np.array(source.data, copy=True), source_times_ms=times.copy(),
                  source_coordinates=np.array(source.coordinates.data, copy=True))
    return source, before


def ensure_no_wavefield_arrays(arrays, cap):
    if any(np.asarray(a).ndim > 2 for a in arrays.values()):
        raise ValueError("wavefield/time-history arrays cannot be saved")
    if sum(np.asarray(a).nbytes for a in arrays.values()) > cap:
        raise ValueError("artifact exceeds frozen small-array byte cap")
