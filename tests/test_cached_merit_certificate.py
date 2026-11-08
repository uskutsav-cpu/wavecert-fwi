"""Synthetic coupled-error checks; no project arrays or performance studies.

The controlled issuer below exists solely to exercise substantial paid-field
errors and affine defects.  It is not an admission path for a native oracle.
Actual native-issued export/owner integration checks are separate below.
"""
import hashlib
import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse.linalg as spla

from wavecert.certificates.cached_merit_certificate import (
    evaluate_cached_merit,
    fixed_sweep_missing_fields,
)
from wavecert.physics.coarse_stability import FastCoarseStability
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock


def immutable(values):
    a = np.ascontiguousarray(values)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def digest(a):
    return hashlib.sha256(np.asarray(a, dtype="<f8").tobytes()).hexdigest()


@dataclass(frozen=True)
class Frame:
    frequency_hz: float
    source_indices: tuple
    receiver_indices: tuple
    unique_receiver_indices: tuple
    receiver_expand_indices: tuple
    observed_slots: np.ndarray
    cache_kind: str
    cached_field: np.ndarray
    residual_slots: np.ndarray
    residual_unique: np.ndarray
    source_amplitude: float
    transpose_symmetric: bool = True


class ControlledSyntheticIssuer:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def validate_merit_fields(self, snapshot):
        if snapshot is not self.snapshot:
            raise ValueError("unissued synthetic mathematical fixture")


def fixture(route, *, perturb_paid=False, affine_defect=False, frequencies=(1.2, 1.6)):
    physics = Helmholtz2D(shape=(5, 7), spacing=.11, damping_strength=.9)
    model = np.linspace(.08, .15, physics.n)
    sources = (2, 7) if route == "forward_states" else (2, 7, 12, 18, 24, 30)
    receivers = (2, 7, 7, 14, 20, 20) if route == "forward_states" else (2, 7, 7, 2, 7)
    unique = tuple(dict.fromkeys(receivers))
    expand = tuple(unique.index(i) for i in receivers)
    p = np.eye(physics.n)[list(receivers)]
    q = np.column_stack([physics.source(i) for i in sources])
    a = 1/physics.spacing**2
    rng = np.random.default_rng(44)
    frames, missing, truth, contexts = [], {}, {}, {}
    engine = FastCoarseStability(physics, 12)
    for frequency in frequencies:
        factor = spla.splu(physics.operator(model, frequency).tocsc())
        state = factor.solve(q)
        observed = .3*(rng.normal(size=(len(receivers), len(sources)))+1j*rng.normal(size=(len(receivers), len(sources))))
        residual = p@state-observed
        adjoint = factor.solve(p.T@residual, trans="H")
        gradient = (2*np.pi*frequency)**2*np.real(np.sum(np.conjugate(adjoint)*state, axis=1))
        if route == "forward_states":
            paid = state.copy()
            if perturb_paid:
                paid += .03*la.norm(paid)/np.sqrt(paid.size)*(rng.normal(size=paid.shape)+1j*rng.normal(size=paid.shape))
            cached_residual = p@paid-observed
            if affine_defect:
                cached_residual += .02*(rng.normal(size=cached_residual.shape)+1j*rng.normal(size=cached_residual.shape))
            proposal = factor.solve(p.T@cached_residual, trans="H")
            true_first, true_second = state, adjoint
            cached_field = paid
        else:
            z = factor.solve(np.eye(physics.n)[:, list(unique)].astype(complex))
            paid = z.copy()
            if perturb_paid:
                paid += .03*la.norm(paid)/np.sqrt(paid.size)*(rng.normal(size=paid.shape)+1j*rng.normal(size=paid.shape))
            cached_residual = (a*paid[list(sources)].T)[list(expand)]-observed
            if affine_defect:
                cached_residual += .02*(rng.normal(size=cached_residual.shape)+1j*rng.normal(size=cached_residual.shape))
            summed = np.zeros((len(unique), len(sources)), dtype=complex)
            np.add.at(summed, np.asarray(expand), cached_residual)
            proposal = factor.solve(q@summed.conjugate().T)
            true_summed = np.zeros_like(summed)
            np.add.at(true_summed, np.asarray(expand), residual)
            true_first, true_second = z, factor.solve(q@true_summed.conjugate().T)
            cached_field = paid.T
        summed = np.zeros((len(unique), len(sources)), dtype=complex)
        np.add.at(summed, np.asarray(expand), cached_residual)
        frame = Frame(frequency, sources, receivers, unique, expand, immutable(observed), route,
                      immutable(cached_field), immutable(cached_residual), immutable(summed), a)
        frames.append(frame)
        missing[frequency] = proposal
        truth[frequency] = (gradient, true_first, true_second, paid, p, q, observed)
        contexts[frequency] = engine.evaluate(model, frequency)
        assert contexts[frequency].available
    snapshot = SimpleNamespace(receipt_key="controlled-synthetic-math-only", model_sha256=digest(model),
        model=immutable(model), shape=physics.shape, spacing=physics.spacing, damping=immutable(physics.damping),
        frequencies=tuple(frames))
    return ControlledSyntheticIssuer(snapshot), snapshot, contexts, missing, truth, physics


def component_errors(context, error):
    nz, nx = context.physics_config["shape"]
    z, x = np.arange(1, nz+1)[:, None], np.arange(1, nx+1)[None, :]
    b = np.column_stack([(2/np.sqrt((nz+1)*(nx+1))*np.sin(np.pi*k*z/(nz+1))
                          * np.sin(np.pi*p*x/(nx+1))).ravel() for k, p in context._engine.mode_indices])
    coarse = b.T@error
    return np.stack((la.norm(coarse, axis=0), la.norm(error-b@coarse, axis=0)))


@pytest.mark.parametrize("route", ["forward_states", "receiver_map"])
@pytest.mark.parametrize("perturb_paid,affine_defect", [(False, False), (True, False), (False, True), (True, True)])
def test_both_coupled_routes_cover_independent_lu_and_preserve_all_terms(route, perturb_paid, affine_defect):
    owner, snapshot, contexts, missing, truth, physics = fixture(route, perturb_paid=perturb_paid, affine_defect=affine_defect)
    answer = evaluate_cached_merit(owner, snapshot, missing, contexts)
    assert answer.available and answer.evaluation_complete
    exact_gradient = sum(pack[0] for pack in truth.values())
    error = abs(float((exact_gradient-answer.approximate_gradient)@answer.direction))
    assert error <= answer.directional_error_bound+1e-11
    for record, frame in zip(answer.frequency_records, snapshot.frequencies, strict=True):
        gradient, first, second, paid, *_ = truth[frame.frequency_hz]
        del gradient
        eta1 = np.asarray(record["paid_field_error_components"])
        eta2 = np.asarray(record["missing_field_error_components"])
        actual1 = component_errors(contexts[frame.frequency_hz], first-paid)
        actual2 = component_errors(contexts[frame.frequency_hz], second-missing[frame.frequency_hz])
        assert np.all(actual1 <= eta1+1e-12)
        assert np.all(actual2 <= eta2+1e-11)
        assert np.asarray(record["directional_bound_terms"]).shape[0] == 3
        if perturb_paid:
            assert np.max(eta1) > 1e-8  # Measured paid-field uncertainty was not set to zero.
        if affine_defect:
            assert record["affine_data_defect_norm"] > .01
        if route == "receiver_map":
            assert record["receiver_multiplicities"] == [2, 3]
            if perturb_paid:
                assert max(record["source_forcing_coefficient_bounds"]) > .01
    if not perturb_paid and not affine_defect:
        np.testing.assert_allclose(answer.approximate_gradient, exact_gradient, rtol=1e-10, atol=1e-10)
        assert answer.certified_descent
    json.dumps(answer.to_dict(), allow_nan=False)
    # Independent objective FD, including duplicate slots and physical h^-2 source scale.
    direction = np.random.default_rng(77).normal(size=physics.n)
    direction /= la.norm(direction)
    def objective(m):
        total = 0.
        for frequency, pack in truth.items():
            _, _, _, _, p, q, observed = pack
            value = p@spla.splu(physics.operator(m, frequency).tocsc()).solve(q)-observed
            total += .5*np.vdot(value, value).real
        return total
    h = 1e-5
    fd = (objective(snapshot.model+h*direction)-objective(snapshot.model-h*direction))/(2*h)
    assert fd == pytest.approx(exact_gradient@direction, rel=3e-6, abs=3e-7)


@pytest.mark.parametrize("route", ["forward_states", "receiver_map"])
def test_fixed_sweeps_share_normfree_proposal_and_all_query_costs(route):
    from wavecert.physics.coarse_dst_reference import CoarseDSTEngine
    owner, snapshot, stability, _, _, _ = fixture(route)
    factory = CoarseDSTEngine(snapshot.shape, snapshot.spacing, snapshot.damping, 12)
    contexts = {f.frequency_hz: factory.prepare(snapshot.model, f.frequency_hz) for f in snapshot.frequencies}
    proposals, receipt = fixed_sweep_missing_fields(owner, snapshot, contexts, sweeps=2)
    answer = evaluate_cached_merit(owner, snapshot, proposals, stability)
    assert answer.evaluation_complete
    assert receipt["query_costs"]["preconditioner_columns"] == 2*sum(x.shape[1] for x in proposals.values())
    assert receipt["query_costs"]["certificate_svd_evaluations"] == 0
    assert receipt["query_costs"]["fine_direct_rhs"] == 0
    assert not receipt["setup_costs_included"]
    for field in proposals.values():
        with pytest.raises(ValueError):
            field.setflags(write=True)


@pytest.mark.parametrize("route", ["forward_states", "receiver_map"])
def test_zero_numeric_fail_and_context_mismatch_fail_closed(route):
    owner, snapshot, contexts, missing, _, physics = fixture(route)
    zero = {key: np.zeros_like(value) for key, value in missing.items()}
    answer = evaluate_cached_merit(owner, snapshot, zero, contexts)
    assert not answer.available and answer.exact_fallback_required and answer.evaluation_complete
    assert "zero" in answer.reason and not answer.to_dict()["stationarity_claim"]
    broken = {key: np.full_like(value, np.nan) for key, value in missing.items()}
    bad = evaluate_cached_merit(owner, snapshot, broken, contexts)
    assert not bad.available and not bad.evaluation_complete and bad.approximate_gradient is None
    json.dumps(bad.to_dict(), allow_nan=False)
    huge = {key: np.full_like(value, 1e308) for key, value in missing.items()}
    bad = evaluate_cached_merit(owner, snapshot, huge, contexts)
    assert not bad.available and not bad.evaluation_complete
    json.dumps(bad.to_dict(), allow_nan=False)
    wrong = dict(contexts)
    frequency = snapshot.frequencies[0].frequency_hz
    wrong[frequency] = FastCoarseStability(physics, 12).evaluate(snapshot.model*1.01, frequency)
    with pytest.raises(ValueError, match="model mismatch"):
        evaluate_cached_merit(owner, snapshot, missing, wrong)
    with pytest.raises(ValueError, match="unissued"):
        evaluate_cached_merit(owner, SimpleNamespace(**snapshot.__dict__), missing, contexts)


def test_unavailable_frequency_keeps_complete_proposal_and_abstains():
    owner, snapshot, contexts, missing, _, _ = fixture("receiver_map")
    contexts[snapshot.frequencies[0].frequency_hz]._unavailable("synthetic availability boundary")
    answer = evaluate_cached_merit(owner, snapshot, missing, contexts)
    assert answer.evaluation_complete and not answer.available
    assert answer.approximate_gradient is not None and len(answer.frequency_records) == 2
    assert answer.directional_error_bound is None and answer.certified_upper_derivative is None
    json.dumps(answer.to_dict(), allow_nan=False)


def test_native_export_admission_both_routes_mutation_owner_and_zero_marginal():
    from pathlib import Path

    from wavecert.physics.exact_merit import ExactMeritOracle
    physics = Helmholtz2D(shape=(5, 7), spacing=.11, damping_strength=.9)
    model = np.linspace(.08, .15, physics.n)
    reference_root = Path(__import__("wavecert.physics.helmholtz", fromlist=["x"]).__file__).parents[1]
    rng = np.random.default_rng(9)
    for sources, receivers, expected in [((2, 7), (2, 7, 7, 14), "forward_states"),
                                          ((2, 7, 12, 18), (2, 7, 7), "receiver_map"),
                                          ((2, 7), (2, 7, 7), "forward_states")]:
        blocks = [SurveyBlock(source, 1.2, rng.normal(size=len(receivers))+1j*rng.normal(size=len(receivers)), str(i))
                  for i, source in enumerate(sources)]
        oracle = ExactMeritOracle(physics, blocks, receivers, reference_package_root=reference_root)
        receipt = oracle.evaluate(model)
        snapshot = oracle.export_merit_fields(receipt)
        assert snapshot.frequencies[0].cache_kind == expected
        context = FastCoarseStability(physics, 12).evaluate(model, 1.2)
        frame = snapshot.frequencies[0]
        factor = spla.splu(physics.operator(model, 1.2).tocsc())
        if expected == "forward_states":
            rhs = np.zeros_like(frame.cached_field)
            np.add.at(rhs, np.asarray(receivers), frame.residual_slots)
            missing = factor.solve(rhs, trans="H")
        else:
            q = np.column_stack([physics.source(i) for i in sources])
            missing = factor.solve(q@frame.residual_unique.conjugate().T)
        answer = evaluate_cached_merit(oracle, snapshot, {1.2: missing}, {1.2: context})
        exact = oracle.gradient(receipt)
        np.testing.assert_allclose(answer.approximate_gradient, exact.gradient, rtol=2e-10, atol=2e-10)
        assert answer.certified_descent
        if len(sources) == len(set(receivers)):
            assert exact.additional_rhs == 0
        foreign = ExactMeritOracle(physics, blocks, receivers, reference_package_root=reference_root)
        with pytest.raises((ValueError, RuntimeError)):
            evaluate_cached_merit(foreign, snapshot, {1.2: missing}, {1.2: context})
        with pytest.raises((ValueError, RuntimeError)):
            evaluate_cached_merit(oracle, replace(snapshot, model=immutable(model*1.01)), {1.2: missing}, {1.2: context})
        with pytest.raises((ValueError, RuntimeError)):
            evaluate_cached_merit(oracle, replace(snapshot, data_sha256="0"*64), {1.2: missing}, {1.2: context})
        changed = replace(frame, cached_field=immutable(frame.cached_field*1.01))
        with pytest.raises((ValueError, RuntimeError)):
            evaluate_cached_merit(oracle, replace(snapshot, frequencies=(changed,)), {1.2: missing}, {1.2: context})
