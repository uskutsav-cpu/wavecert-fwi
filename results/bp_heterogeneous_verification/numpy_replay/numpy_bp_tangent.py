"""Source-informed variable-m OT2 forward/tangent port: NumPy and stdlib only.

This module adds no optimization, physical threshold or reference-data fitting.
Native numeric witnesses must remain unopened until all own solves complete.
"""
from __future__ import annotations

import hashlib

import numpy as np

# Literal values emitted by the pinned h=12.5m primary C, not rescaled h=10 values.
WEIGHTS = (5.33333333302289e-4, 8.53333333283663e-3, 3.2e-2)


def array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def laplace(field):
    """Logical full-grid stencil, with two fixed zero ghost cells outside it."""
    far, near, center = WEIGHTS
    return (far*(-field[:-4, 2:-2]-field[2:-2, :-4]-field[2:-2, 4:]-field[4:, 2:-2])
            +near*(field[1:-3, 2:-2]+field[2:-2, 1:-3]+field[2:-2, 3:-1]+field[3:-1, 2:-2])
            -center*field[2:-2, 2:-2])


def damping(shape, width, spacing):
    """Analytic face profiles from pinned model.py; summed, including corners."""
    result = np.zeros(shape, dtype=np.float64)
    if width == 0:
        return result
    if width < 1 or any(n <= 2*width for n in shape):
        raise ValueError("positive physical shape inside nonnegative edge width required")
    for axis, n in enumerate(shape):
        indices = np.arange(width, dtype=np.float64)
        left, right = (width-indices+1)/width, (indices+2)/width
        coefficient = 1.5*np.log(1000.)/(width*spacing)
        face = np.zeros(n, dtype=np.float64)
        face[:width] = coefficient*(left-np.sin(2*np.pi*left)/(2*np.pi))
        face[-width:] = coefficient*(right-np.sin(2*np.pi*right)/(2*np.pi))
        result += face[:, None] if axis == 0 else face[None, :]
    return result


def tangent_step(primal, tangents, time, mass, directions, damp, dt, q, source_index, source_inverse_mass):
    """Differentiate the actual quotient BEFORE the separate source injection.

    m and each direction live on the full logical padded grid. Directions must
    already be normalized physically, then extended with the constant-edge E.
    The injection value mirrors C's vp*vp; its real-arithmetic derivative uses
    -dt²*q*dm/m². This is ordinary numerical differentiation, not verified FP AD.
    """
    current, previous, following = time % 3, (time+2) % 3, (time+1) % 3
    u = primal[current]
    center, past = u[2:-2, 2:-2], primal[previous, 2:-2, 2:-2]
    r, s = 1/(dt*dt), 1/dt
    denominator = mass*r+s*damp
    following_core = (-mass*(-2*r*center+r*past)+s*damp*center+laplace(u))/denominator
    primal[following, 2:-2, 2:-2] = following_core
    for index, direction in enumerate(directions):
        w = tangents[index, current]
        wc, wp = w[2:-2, 2:-2], tangents[index, previous, 2:-2, 2:-2]
        numerator_derivative = (-direction*(-2*r*center+r*past)
                                -mass*(-2*r*wc+r*wp)+s*damp*wc+laplace(w))
        tangents[index, following, 2:-2, 2:-2] = (
            numerator_derivative-following_core*r*direction)/denominator
        x, z = source_index
        tangents[index, following, x+2, z+2] -= (dt*dt)*q*direction[x, z]/(mass[x, z]*mass[x, z])
    x, z = source_index
    primal[following, x+2, z+2] += (dt*dt)*source_inverse_mass*q
    return current, following


def physical_directions(mass):
    result = []
    for label in ["interior", "edge"]:
        mask = np.zeros(mass.shape, dtype=bool)
        if label == "interior":
            mask[4:-4, 4:-4] = True
        else:
            mask[0] = mask[-1] = True
            mask[:, 0] = mask[:, -1] = True
        vector = mass*mask
        scale = float(np.linalg.norm(vector))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("fixed physical direction has empty/nonfinite support")
        result.append(vector/scale)
    return np.asarray(result)


def propagate(physical_velocity, physical_perturbations, *, width, dt, nt, source_node, receiver_nodes):
    """Rolling primal plus fixed tangent directions, without native fields/data."""
    velocity = np.asarray(physical_velocity, dtype=np.float64)
    perturbations = np.asarray(physical_perturbations, dtype=np.float64)
    if (velocity.ndim != 2 or not np.isfinite(velocity).all() or (velocity <= 0).any()
            or perturbations.ndim != 3 or perturbations.shape[1:] != velocity.shape
            or not np.isfinite(perturbations).all()):
        raise ValueError("positive2D velocity and matching finite physical-m directions required")
    padded_velocity = np.pad(velocity, width, mode="edge")
    square = padded_velocity*padded_velocity
    mass = 1/square
    directions = np.asarray([np.pad(d, width, mode="edge") for d in perturbations])
    logical = mass.shape
    damp = damping(logical, width, 12.5)
    source = tuple(int(i+width) for i in source_node)
    receivers = np.asarray(receiver_nodes, dtype=np.int64)+width
    if (damp[source] != 0 or receivers.ndim != 2 or receivers.shape[1] != 2
            or any(i < 0 or i >= n for i, n in zip(source, logical, strict=True))):
        raise ValueError("fixed aligned undamped source/receiver geometry required")
    for axis in range(2):
        if ((receivers[:, axis] < 0) | (receivers[:, axis] >= logical[axis])).any():
            raise ValueError("receiver outside logical grid")
    state = np.zeros((3, logical[0]+4, logical[1]+4), dtype=np.float64)
    tangent = np.zeros((len(directions), *state.shape), dtype=np.float64)
    times = np.arange(nt, dtype=np.float64)*dt
    argument = np.pi*.005*(times-200.)
    source_values = (1-2*argument**2)*np.exp(-argument**2)
    traces = np.zeros((nt, len(receivers)), dtype=np.float64)
    tangent_traces = np.zeros((len(directions), nt, len(receivers)), dtype=np.float64)
    rx, rz = receivers[:, 0]+2, receivers[:, 1]+2
    for time in range(1, nt-1):
        current, _ = tangent_step(state, tangent, time, mass, directions, damp, dt,
                                  source_values[time], source, square[source])
        traces[time] = state[current, rx, rz]
        tangent_traces[:, time] = tangent[:, current, rx, rz]
    if not all(np.isfinite(a).all() for a in [state, tangent, traces, tangent_traces, damp]):
        raise FloatingPointError("nonfinite independent primal/tangent replay")
    return dict(receiver=traces, tangent_receiver=tangent_traces, times_ms=times,
                source=source_values[:, None], computed_mask=(np.arange(nt)>=1)&(np.arange(nt)<=nt-2)), dict(
        physical_velocity_sha256=array_sha(velocity), physical_m_sha256=array_sha(1/(velocity*velocity)),
        padded_m_sha256=array_sha(mass), damping_sha256=array_sha(damp),
        physical_direction_sha256=[array_sha(d) for d in perturbations],
        padded_direction_sha256=[array_sha(d) for d in directions],
        logical_shape=list(logical), rolling_state_shape=list(state.shape),
        primal_state_bytes=state.nbytes, tangent_state_bytes=tangent.nbytes,
        full_wavefield_history_saved=False, final_state_sha256=array_sha(state),
        final_tangent_state_sha256=array_sha(tangent), coefficients=list(WEIGHTS),
        time_bounds=dict(time_m=1, time_M=nt-2), receiver_samples_current_state=True,
        native_numeric_witnesses_used=False)
