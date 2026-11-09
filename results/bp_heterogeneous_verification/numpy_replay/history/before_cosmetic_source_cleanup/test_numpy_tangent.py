"""Manufactured one-step controls; no Devito imports or project data.

The dense reference uses a separately assembled matrix and a rearranged
recurrence. Epsilon values and tolerances are fixed before the first run.
"""
from __future__ import annotations

import numpy as np
import pytest

import numpy_bp_tangent as core


EXPECTED_WEIGHTS = (5.33333333302289e-4, 8.53333333283663e-3, 3.2e-2)
FD_EPSILONS = (1e-2, 1e-3, 1e-4)


def dense_stencil(shape):
    """Build every logical row, dropping only exterior-zero ghost entries."""
    nx, nz = shape
    far, near, center = EXPECTED_WEIGHTS
    matrix = np.zeros((nx*nz, nx*nz), dtype=np.float64)
    for x in range(nx):
        for z in range(nz):
            row = x*nz+z
            matrix[row, row] = -center
            for distance, weight in ((1, near), (2, -far)):
                for dx, dz in ((distance, 0), (-distance, 0),
                               (0, distance), (0, -distance)):
                    xx, zz = x+dx, z+dz
                    if 0 <= xx < nx and 0 <= zz < nz:
                        matrix[row, xx*nz+zz] += weight
    return matrix


def dense_forward(current, previous, mass, damp, dt, q, source,
                  source_inverse_mass=None):
    """Independent rearrangement: D=m+d*dt, not the core's r1/r2 form."""
    linear = (dense_stencil(mass.shape) @ current.ravel()).reshape(mass.shape)
    answer = ((2*mass+damp*dt)*current-mass*previous+dt**2*linear)/(mass+damp*dt)
    inverse_mass = 1/mass[source] if source_inverse_mass is None else source_inverse_mass
    answer[source] += dt**2*q*inverse_mass
    return answer


def dense_tangent(current, previous, current_dot, previous_dot,
                  mass, direction, damp, dt, q, source):
    """Analytic partials of the rearranged dense map, including injection."""
    matrix = dense_stencil(mass.shape)
    linear = (matrix @ current.ravel()).reshape(mass.shape)
    linear_dot = (matrix @ current_dot.ravel()).reshape(mass.shape)
    denominator = mass+damp*dt
    numerator = (2*mass+damp*dt)*current-mass*previous+dt**2*linear
    # Differentiate the history and mass arguments separately. The mass-only
    # partial is ((2u-u_prev)D-N)/D^2 before the separate source injection.
    answer = ((2*mass+damp*dt)*current_dot-mass*previous_dot+dt**2*linear_dot)/denominator
    answer += direction*((2*current-previous)*denominator-numerator)/denominator**2
    answer[source] -= dt**2*q*direction[source]/mass[source]**2
    return answer


def manufactured_case(time=1):
    generator = np.random.default_rng(724921)
    shape = (5, 4)
    mass = generator.uniform(0.4, 1.3, size=shape)
    damp = generator.uniform(0.03, 0.7, size=shape)
    directions = generator.uniform(-0.2, 0.2, size=(2, *shape))
    primal = np.zeros((3, shape[0]+4, shape[1]+4), dtype=np.float64)
    tangents = np.zeros((2, *primal.shape), dtype=np.float64)
    current, previous, following = time % 3, (time+2) % 3, (time+1) % 3
    for slot in (current, previous):
        primal[slot, 2:-2, 2:-2] = generator.normal(size=shape)
        tangents[:, slot, 2:-2, 2:-2] = generator.normal(size=(2, *shape))
    primal[following, 2:-2, 2:-2] = 12345.0
    tangents[:, following, 2:-2, 2:-2] = -12345.0
    return primal, tangents, mass, directions, damp, 0.5, 0.9, (1, 2)


def test_spatial_literals_are_the_printed_h12_5_values():
    assert tuple(core.WEIGHTS) == EXPECTED_WEIGHTS


@pytest.mark.parametrize("shape", [(1, 1), (4, 5)])
def test_laplace_matches_dense_all_nodes_including_boundaries(shape):
    logical = np.arange(1, np.prod(shape)+1, dtype=np.float64).reshape(shape)/7
    halo = np.zeros((shape[0]+4, shape[1]+4), dtype=np.float64)
    halo[2:-2, 2:-2] = logical
    expected = (dense_stencil(shape) @ logical.ravel()).reshape(shape)
    np.testing.assert_allclose(core.laplace(halo), expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("time", [1, 2, 3])
def test_variable_mass_damping_one_step_and_slot_ownership(time):
    primal, tangents, mass, directions, damp, dt, q, source = manufactured_case(time)
    primal_before, tangents_before = primal.copy(), tangents.copy()
    current, previous, following = time % 3, (time+2) % 3, (time+1) % 3
    u = primal_before[current, 2:-2, 2:-2]
    past = primal_before[previous, 2:-2, 2:-2]
    expected = dense_forward(u, past, mass, damp, dt, q, source)
    expected_dot = np.asarray([
        dense_tangent(u, past, tangents_before[k, current, 2:-2, 2:-2],
                      tangents_before[k, previous, 2:-2, 2:-2],
                      mass, direction, damp, dt, q, source)
        for k, direction in enumerate(directions)
    ])
    assert core.tangent_step(primal, tangents, time, mass, directions, damp, dt, q,
                             source, 1/mass[source]) == (current, following)
    np.testing.assert_allclose(primal[following, 2:-2, 2:-2], expected,
                               rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(tangents[:, following, 2:-2, 2:-2], expected_dot,
                               rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(primal[[current, previous]], primal_before[[current, previous]])
    np.testing.assert_array_equal(tangents[:, [current, previous]], tangents_before[:, [current, previous]])
    for state in (primal, *tangents):
        for region in (state[:, :2], state[:, -2:], state[:, :, :2], state[:, :, -2:]):
            np.testing.assert_array_equal(region, np.zeros_like(region))


@pytest.mark.parametrize("source_damping", [0.0, 0.8])
def test_source_only_derivative_and_supplied_inverse_mass(source_damping):
    shape, source, dt, q, time = (4, 5), (0, 0), 0.5, 1.7, 2
    mass = np.full(shape, 0.7, dtype=np.float64)
    damp = np.full(shape, source_damping, dtype=np.float64)
    directions = np.zeros((2, *shape), dtype=np.float64)
    directions[0, source[0], source[1]] = 0.13
    directions[1, -1, -1] = -0.2
    primal = np.zeros((3, shape[0]+4, shape[1]+4), dtype=np.float64)
    tangents = np.zeros((2, *primal.shape), dtype=np.float64)
    supplied_inverse_mass = 3.25
    _, following = core.tangent_step(primal, tangents, time, mass, directions, damp,
                                      dt, q, source, supplied_inverse_mass)
    expected = np.zeros(shape)
    expected[source] = dt**2*q*supplied_inverse_mass
    expected_dot = np.zeros((2, *shape))
    expected_dot[0, source[0], source[1]] = -dt**2*q*directions[0, source[0], source[1]]/mass[source]**2
    np.testing.assert_allclose(primal[following, 2:-2, 2:-2], expected, rtol=0, atol=0)
    np.testing.assert_allclose(tangents[:, following, 2:-2, 2:-2], expected_dot,
                               rtol=2e-15, atol=2e-15)


def test_fixed_history_central_differences_at_predeclared_epsilons():
    primal, tangents, mass, directions, damp, dt, q, source = manufactured_case()
    current, previous, following = 1, 0, 2
    u, past = primal[current, 2:-2, 2:-2].copy(), primal[previous, 2:-2, 2:-2].copy()
    w = tangents[:, current, 2:-2, 2:-2].copy()
    w_past = tangents[:, previous, 2:-2, 2:-2].copy()
    core.tangent_step(primal, tangents, 1, mass, directions, damp, dt, q,
                      source, 1/mass[source])
    exact = tangents[:, following, 2:-2, 2:-2]
    errors = []
    for epsilon in FD_EPSILONS:
        central = np.asarray([
            (dense_forward(u+epsilon*w[k], past+epsilon*w_past[k], mass+epsilon*mu,
                           damp, dt, q, source)
             -dense_forward(u-epsilon*w[k], past-epsilon*w_past[k], mass-epsilon*mu,
                            damp, dt, q, source))/(2*epsilon)
            for k, mu in enumerate(directions)
        ])
        relative = float(np.linalg.norm(central-exact)/np.linalg.norm(exact))
        errors.append(relative)
        assert relative <= 2*epsilon**2+2e-10
    print({"fixed_history_epsilons": FD_EPSILONS, "relative_errors": errors,
           "bounds": [2*epsilon**2+2e-10 for epsilon in FD_EPSILONS]})
    assert errors[-1] < 1e-7
