"""Manufactured dense algebra only: no production imports, grids or PDEs."""
from __future__ import annotations

import copy
import gc
import hashlib
import weakref

import numpy as np
import pytest
from cached_jacobian import CachedJacobianCallbacks, Layout, array_sha
from public_map_builder import PublicReceiverMapBuilder
from scipy.optimize import least_squares
from scipy.sparse.linalg import lsqr


class Record:
    pass


def record(**items):
    result = Record()
    result.__dict__.update(items)
    return result


def sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


class DenseToyNative:
    """Issuer with tiny manufactured dense matrices, not native Helmholtz."""
    def __init__(self, layout, initial, *, nonsymmetric=False, amplitudes=None, observed=None):
        self.layout, self.initial = layout, initial.copy()
        n = layout.n
        self.k = np.diag(np.linspace(7., 9., n))
        if n > 1:
            self.k += np.diag(np.full(n-1, .17), 1)+np.diag(np.full(n-1, .17), -1)
            if nonsymmetric:
                self.k[0, 1] = .31
        self.d = np.linspace(.2, .4, n)
        self.amplitudes = {f: tuple(amplitudes or (2+3j,)*len(s)) for f, s, _ in layout.frames}
        self.observed = observed or {f: self.states(initial*.93, f, s)[list(layout.receivers)]
                                    for f, s, _ in layout.frames}
        self.counts = {"evaluations": 0, "exports": 0, "validations": 0,
                       "factorizations": 0, "forward_rhs": 0}
        self.receipts, self.exports = weakref.WeakSet(), weakref.WeakKeyDictionary()
        self.held = []
        self.contract_changed = self.fail_evaluate = self.fail_cost = False

    def matrix(self, model, frequency):
        omega = 2*np.pi*frequency
        return self.k-np.diag(omega**2*model)+1j*omega*np.diag(self.d)

    def states(self, model, frequency, sources):
        q = np.zeros((self.layout.n, len(sources)), dtype=np.complex128)
        q[list(sources), np.arange(len(sources))] = self.amplitudes[frequency]
        return np.linalg.solve(self.matrix(model, frequency), q)

    def cost_snapshot(self):
        if self.fail_cost:
            raise RuntimeError("manufactured snapshot interruption")
        return dict(self.counts)

    def evaluate(self, model):
        self.counts["evaluations"] += 1
        if self.fail_evaluate:
            raise RuntimeError("manufactured numerical dispatch interruption")
        frames, objective = [], 0.
        for frequency, sources, blocks in self.layout.frames:
            state = self.states(model, frequency, sources)
            self.counts["factorizations"] += 1
            self.counts["forward_rhs"] += len(sources)
            residual = state[list(self.layout.receivers)]-self.observed[frequency]
            objective += .5*np.vdot(residual, residual).real
            unique = tuple(dict.fromkeys(self.layout.receivers))
            frames.append(record(frequency_hz=frequency, source_indices=sources, block_indices=blocks,
                receiver_indices=self.layout.receivers, unique_receiver_indices=unique,
                receiver_expand_indices=np.array([unique.index(i) for i in self.layout.receivers]),
                source_amplitude=np.array(self.amplitudes[frequency], dtype=np.complex128),
                transpose_symmetric=np.array_equal(self.matrix(model, frequency), self.matrix(model, frequency).T),
                cache_kind="forward_states", cached_field=state, residual_slots=residual,
                observed_slots=self.observed[frequency].copy()))
        model = model.copy()
        receipt = record(model=model, model_sha256=array_sha(model), objective=float(objective),
            operator_sha256=sha(("operator", array_sha(model))), geometry_sha256=sha(self.layout),
            data_sha256=sha(tuple(array_sha(v) for v in self.observed.values())), frames=frames)
        receipt.key = sha((receipt.operator_sha256, receipt.geometry_sha256, receipt.data_sha256))
        self.receipts.add(receipt)
        self.held = (self.held+[receipt])[-2:]
        return receipt

    def export_merit_fields(self, receipt):
        self.counts["exports"] += 1
        if receipt not in self.receipts:
            raise ValueError("foreign receipt")
        snapshot = record(model=receipt.model.copy(), model_sha256=receipt.model_sha256,
            receipt_key=receipt.key, operator_sha256=receipt.operator_sha256,
            geometry_sha256=receipt.geometry_sha256, data_sha256=receipt.data_sha256,
            shape=(1, self.layout.n), spacing=.025, damping_width=0, damping_strength=0.,
            frequencies=copy.deepcopy(receipt.frames))
        self.exports[snapshot] = (weakref.ref(receipt), self.fingerprint(snapshot))
        return snapshot

    @staticmethod
    def fingerprint(snapshot):
        return (array_sha(snapshot.model), tuple((array_sha(f.cached_field), array_sha(f.residual_slots),
            array_sha(f.observed_slots)) for f in snapshot.frequencies))

    def validate_merit_fields(self, snapshot):
        self.counts["validations"] += 1
        if self.contract_changed:
            raise ValueError("manufactured survey contract changed")
        if snapshot not in self.exports:
            raise ValueError("foreign snapshot")
        reference, fingerprint = self.exports[snapshot]
        if reference() is None or self.fingerprint(snapshot) != fingerprint:
            raise ValueError("expired or altered snapshot")


class DenseToyMapPhysics:
    KEYS = ("factorizations", "forward_rhs", "tangent_rhs", "adjoint_rhs", "stability_rhs",
            "dense_inverse_rhs", "svd_evaluations", "cache_hits", "analytic_bound_evaluations")

    def __init__(self, native, *, fail=False, snapshot_fail=False):
        self.k, self.d = native.k.copy(), native.d.copy()
        self.counts = dict.fromkeys(self.KEYS, 0)
        self.fail, self.snapshot_fail = fail, snapshot_fail

    def cost_snapshot(self):
        if self.snapshot_fail and self.counts["factorizations"]:
            raise RuntimeError("manufactured map counter interruption")
        return dict(self.counts)

    def receiver_resolvent_matrix(self, model, frequency, receivers):
        omega = 2*np.pi*frequency
        matrix = self.k-np.diag(omega**2*model)+1j*omega*np.diag(self.d)
        rhs = np.zeros((len(model), len(receivers)), dtype=np.complex128)
        rhs[list(receivers), np.arange(len(receivers))] = 1.
        self.counts["factorizations"] += 1
        if self.fail:
            raise RuntimeError("manufactured solve after factorization interrupted")
        response = np.linalg.solve(matrix.conj().T, rhs)
        self.counts["stability_rhs"] += len(receivers)
        return response.conj().T


def make(*, n=3, receivers=(0, 1, 1, 2), sources=(2, 0), frequency=.23,
         nonsymmetric=False, amplitudes=None, observed=None, map_fail=False, snapshot_fail=False):
    layout = Layout(n, receivers, ((frequency, sources, tuple(range(len(sources)))),))
    initial = np.linspace(.25, .35, n, dtype=np.float64)
    native = DenseToyNative(layout, initial, nonsymmetric=nonsymmetric, amplitudes=amplitudes, observed=observed)
    builder = PublicReceiverMapBuilder(lambda config: DenseToyMapPhysics(native, fail=map_fail, snapshot_fail=snapshot_fail))
    callbacks = CachedJacobianCallbacks(native, builder, initial_model=initial, layout=layout)
    return callbacks, native, builder, initial


def manual_jacobian(native, model):
    rows = np.zeros((native.layout.blocks*len(native.layout.receivers), native.layout.n), dtype=np.complex128)
    for frequency, sources, blocks in native.layout.frames:
        inverse = np.linalg.inv(native.matrix(model, frequency))
        state = native.states(model, frequency, sources)
        for column, block in enumerate(blocks):
            rows[block*len(native.layout.receivers):(block+1)*len(native.layout.receivers)] = (
                (2*np.pi*frequency)**2*inverse[list(native.layout.receivers)]*state[:, column])
    return rows


def test_scalar_manual_phi_gradient_and_two_centered_differences():
    callbacks, native, _, initial = make(n=1, receivers=(0,), sources=(0,), amplitudes=(2+3j,))
    residual = callbacks.fun(initial)
    operator = callbacks.jac(initial)
    frequency = native.layout.frames[0][0]
    omega2 = (2*np.pi*frequency)**2
    a = native.matrix(initial, frequency)[0, 0]
    u = (2+3j)/a
    expected = omega2*(2+3j)/a**2
    assert np.isclose(u, native.states(initial, frequency, (0,))[0, 0], rtol=1e-14)
    direction = np.array([.7])
    actual = operator@direction
    reference = np.array([expected.real, expected.imag])*.7/np.sqrt(callbacks.phi0)
    np.testing.assert_allclose(actual, reference, rtol=2e-13, atol=1e-13)
    assert np.isclose(.5*residual@residual, 1., rtol=2e-13)
    for step in (1e-5, 5e-6):
        plus, minus = callbacks.fun(initial+step*direction), callbacks.fun(initial-step*direction)
        np.testing.assert_allclose((plus-minus)/(2*step), actual, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(operator.rmatvec(residual), operator.T@residual)


@pytest.mark.parametrize("nonsymmetric", [False, True])
@pytest.mark.parametrize("amplitudes", [(2+3j, -1+2j), (1600., 1600.)])
def test_manual_dense_real_adjoint_and_complex_amplitude(nonsymmetric, amplitudes):
    callbacks, native, builder, initial = make(nonsymmetric=nonsymmetric, amplitudes=amplitudes)
    residual = callbacks.fun(initial)
    operator = callbacks.jac(initial)
    matrix = manual_jacobian(native, initial)
    expected = np.vstack((matrix.real, matrix.imag))/np.sqrt(callbacks.phi0)
    p = np.array([.3, -.7, .2])
    y = np.linspace(-.8, .9, expected.shape[0])
    np.testing.assert_allclose(operator@p, expected@p, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(operator.T@y, expected.T@y, rtol=2e-12, atol=2e-12)
    assert np.isclose(y@(operator@p), p@(operator.T@y), rtol=2e-12, atol=2e-12)
    native_gradient = expected.T@residual
    for step in (1e-5, 5e-6):
        fp, fm = callbacks.fun(initial+step*p), callbacks.fun(initial-step*p)
        np.testing.assert_allclose((fp-fm)/(2*step), expected@p, rtol=3e-8, atol=3e-8)
        dd = (.5*(fp@fp)-.5*(fm@fm))/(2*step)
        assert np.isclose(dd, native_gradient@p, rtol=3e-8, atol=3e-8)
    batch = builder.cost_snapshot()["cold_owner_batches"][0]
    assert batch["last_known_physics_prefix"]["factorizations"] == 1
    assert batch["last_known_physics_prefix"]["stability_rhs"] == (3 if nonsymmetric else 1)


def test_full_eight_frequency_1920_real_terms_preserve_all_blocks():
    frequencies = (.25, .5, 1., 2., 2.5, 3., 3.5, 4.)
    sources = (0, 4, 8, 11, 15, 19)
    layout = Layout(20, tuple(range(20)), tuple((f, sources, tuple(range(6*i, 6*i+6))) for i, f in enumerate(frequencies)))
    initial = np.full(20, .001, dtype=np.float64)
    native = DenseToyNative(layout, initial, amplitudes=(1600.,)*6)
    builder = PublicReceiverMapBuilder(lambda config: DenseToyMapPhysics(native))
    callbacks = CachedJacobianCallbacks(native, builder, initial_model=initial, layout=layout)
    residual = callbacks.fun(initial)
    operator = callbacks.jac(initial)
    assert residual.shape == (1920,) and operator.shape == (1920, 20)
    expected_complex = np.concatenate([(native.states(initial, f, s)[list(layout.receivers)]-native.observed[f]).T.reshape(-1)
                                       for f, s, _ in layout.frames])
    np.testing.assert_allclose(residual, np.r_[expected_complex.real, expected_complex.imag]/np.sqrt(callbacks.phi0))
    assert np.isclose(.5*residual@residual, 1.)
    batches = builder.cost_snapshot()["cold_owner_batches"]
    assert len(batches) == 8
    assert sum(row["last_known_physics_prefix"]["factorizations"] for row in batches) == 8
    assert sum(row["last_known_physics_prefix"]["stability_rhs"] for row in batches) == 112
    before = builder.cost_snapshot()
    operator@np.linspace(-1., 1., 20)
    operator.T@np.linspace(-1., 1., 1920)
    assert builder.cost_snapshot() == before
    assert callbacks.jac(initial) is operator
    assert builder.cost_snapshot() == before


def test_all_covered_receivers_need_no_new_physics_owner():
    callbacks, _, builder, initial = make(receivers=(0, 2, 0))
    callbacks.fun(initial)
    callbacks.jac(initial)
    assert builder.cost_snapshot()["cold_owner_batches"] == []


def test_old_operator_keeps_its_field_point_after_new_trial():
    callbacks, _, builder, initial = make()
    callbacks.fun(initial)
    old = callbacks.jac(initial)
    vector = np.array([.3, -.2, .7])
    before = old@vector
    callbacks.fun(initial*1.07)
    new = callbacks.jac(initial*1.07)
    np.testing.assert_array_equal(old@vector, before)
    assert not np.allclose(new@vector, before, rtol=1e-7, atol=1e-7)
    batches = builder.cost_snapshot()
    old@vector
    assert builder.cost_snapshot() == batches
    with pytest.raises(ValueError):
        callbacks.latest_jac.frames[0].state.setflags(write=True)


def test_changed_contract_invalidates_even_old_operator():
    callbacks, native, _, initial = make()
    callbacks.fun(initial)
    operator = callbacks.jac(initial)
    native.contract_changed = True
    with pytest.raises(ValueError, match="contract"):
        operator@np.ones(3)


def test_cold_native_owner_and_unique_policy_ownership():
    callbacks, native, builder, initial = make()
    with pytest.raises(ValueError, match="another live"):
        CachedJacobianCallbacks(native, builder, initial_model=initial, layout=native.layout)
    callbacks.fun(initial)
    foreign = DenseToyNative(native.layout, initial)
    foreign.evaluate(initial)
    separate_builder = PublicReceiverMapBuilder(lambda config: DenseToyMapPhysics(foreign))
    with pytest.raises(ValueError, match="zero executed work"):
        CachedJacobianCallbacks(foreign, separate_builder, initial_model=initial, layout=native.layout)


def test_no_hidden_merit_query_for_jac_or_wrong_initial():
    callbacks, native, _, initial = make()
    for operation in (callbacks.fun, callbacks.jac):
        with pytest.raises(ValueError):
            operation(initial*1.01)
    assert native.counts["evaluations"] == 0


def test_foreign_and_expired_native_exports_are_rejected():
    callbacks, native, _, initial = make()
    receipt = native.evaluate(initial)
    snapshot = native.export_merit_fields(receipt)
    foreign = DenseToyNative(native.layout, initial)
    with pytest.raises(ValueError, match="foreign"):
        foreign.validate_merit_fields(snapshot)
    native.held.clear()
    del receipt
    gc.collect()
    with pytest.raises(ValueError, match="expired"):
        native.validate_merit_fields(snapshot)
    assert callbacks.phi0 is None


@pytest.mark.parametrize("invalid", [np.array([0., .3, .4]), np.array([np.nan, .3, .4]),
    np.array([.2, .3, .4], dtype=np.float32), np.array([.2+0j, .3, .4]), np.ones((3, 1))])
def test_invalid_native_model(invalid):
    callbacks, _, _, _ = make()
    with pytest.raises(ValueError):
        callbacks.fun(invalid)


@pytest.mark.parametrize("kind", ["frequency", "blocks", "receivers", "amplitude", "state", "objective", "route"])
def test_invalid_complete_snapshot_fields_rejected(kind):
    callbacks, native, _, initial = make()
    original = native.export_merit_fields

    def corrupt(receipt):
        value = original(receipt)
        frame = value.frequencies[0]
        if kind == "frequency":
            value.frequencies = []
        elif kind == "blocks":
            frame.block_indices = (0, 0)
        elif kind == "receivers":
            frame.receiver_indices = (0, 1, 2, 2)
        elif kind == "amplitude":
            frame.source_amplitude = 0.
        elif kind == "state":
            frame.cached_field = frame.cached_field[:, :1]
        elif kind == "objective":
            receipt.objective *= 2
        else:
            frame.cache_kind = "receiver_map"
        native.exports[value] = (weakref.ref(receipt), native.fingerprint(value))
        return value

    native.export_merit_fields = corrupt
    with pytest.raises(ValueError):
        callbacks.fun(initial)


def test_map_failed_prefix_retains_lu_and_unknown_rhs_work():
    callbacks, _, builder, initial = make(map_fail=True)
    callbacks.fun(initial)
    with pytest.raises(RuntimeError, match="interrupted"):
        callbacks.jac(initial)
    row = builder.cost_snapshot()["cold_owner_batches"][0]
    assert row["last_known_physics_prefix"]["factorizations"] == 1
    assert row["last_known_physics_prefix"]["stability_rhs"] == 0
    assert row["unknown_work"] and not row["map_completed"]
    assert callbacks.last_failure["map_builder_prefix"]["unknown_work_present"]


def test_map_snapshot_failure_does_not_invent_zero_work():
    callbacks, _, builder, initial = make(snapshot_fail=True)
    callbacks.fun(initial)
    with pytest.raises(RuntimeError):
        callbacks.jac(initial)
    row = builder.cost_snapshot()["cold_owner_batches"][0]
    assert row["map_completed"] and row["unknown_work"]
    assert row["last_known_physics_prefix"] is None


def test_native_failure_and_snapshot_failure_preserve_last_prefix():
    callbacks, native, _, initial = make()
    callbacks.fun(initial)
    prefix = callbacks.cost_snapshot()["native"]["last_known_cumulative"]
    native.fail_evaluate = native.fail_cost = True
    with pytest.raises(RuntimeError):
        callbacks.fun(initial*1.01)
    result = callbacks.last_failure["native_cost"]
    assert result["last_known_cumulative"] == prefix
    assert result["unknown_since_prefix"] and result["failed_dispatch_work_may_exceed_prefix"]


def test_native_failed_dispatch_unknown_even_when_counter_snapshot_available():
    callbacks, native, _, initial = make()
    native.fail_evaluate = True
    with pytest.raises(RuntimeError):
        callbacks.fun(initial)
    assert callbacks.last_failure["native_cost"]["last_known_cumulative"]["evaluations"] == 1
    assert callbacks.last_failure["native_cost"]["unknown_since_prefix"]
    native.fail_evaluate = False
    callbacks.fun(initial)
    assert callbacks.cost_snapshot()["native"]["unknown_since_prefix"]
    assert callbacks.cost_snapshot()["native"]["failed_dispatch_work_may_exceed_prefix"]


def test_map_builder_cannot_be_shared_between_live_policies():
    callbacks, native, builder, initial = make()
    foreign = DenseToyNative(native.layout, initial)
    with pytest.raises(ValueError, match="map builder already"):
        CachedJacobianCallbacks(foreign, builder, initial_model=initial, layout=native.layout)
    assert callbacks.phi0 is None


@pytest.mark.parametrize("corruption", ["model", "residual", "map"])
def test_altered_copied_or_native_arrays_fail_closed(corruption):
    callbacks, _, _, initial = make()
    callbacks.fun(initial)
    operator = callbacks.jac(initial)
    point = callbacks.latest_jac
    if corruption == "model":
        object.__setattr__(point, "model", initial*1.01)
    elif corruption == "residual":
        point.native_snapshot.frequencies[0].residual_slots[0, 0] += 1.
    else:
        # Immutable byte-backed maps cannot enable write access; demonstrate
        # the same property through the copied field array exposed for audit.
        with pytest.raises(ValueError):
            point.frames[0].state.setflags(write=True)
        return
    with pytest.raises(ValueError):
        operator@np.ones(3)


def test_interrupted_map_batch_retains_clock_and_partial_work():
    callbacks, _, builder, initial = make(map_fail=True)
    times = iter((10., 10.75))
    builder.clock = lambda: next(times)
    callbacks.fun(initial)
    with pytest.raises(RuntimeError):
        callbacks.jac(initial)
    row = callbacks.last_failure["map_builder_prefix"]["cold_owner_batches"][0]
    assert row["elapsed_seconds"] == .75 and row["unknown_work"]


def test_native_keyboard_interrupt_keeps_failure_evidence():
    callbacks, native, _, initial = make()

    def interrupted(model):
        native.counts["evaluations"] += 1
        raise KeyboardInterrupt

    native.evaluate = interrupted
    with pytest.raises(KeyboardInterrupt):
        callbacks.fun(initial)
    assert callbacks.last_failure["exception_type"] == "KeyboardInterrupt"
    assert callbacks.last_failure["native_cost"]["unknown_since_prefix"]


def test_reused_map_owner_is_rejected_without_counting_borrowed_work_as_new():
    callbacks, native, builder, initial = make()
    owner = DenseToyMapPhysics(native)
    builder.factory = lambda config: owner
    callbacks.fun(initial)
    callbacks.jac(initial)
    callbacks.fun(initial*1.01)
    with pytest.raises(ValueError, match="reused"):
        callbacks.jac(initial*1.01)
    assert builder.cost_snapshot()["cold_owner_batches"][-1]["unknown_work"]


def test_bounded_trf_synthetic_quadratic_with_component_linear_operator():
    callbacks, _, _, initial = make(n=1, receivers=(0,), sources=(0,))
    f0 = callbacks.fun(initial)
    operator = callbacks.jac(initial)
    dense = operator@np.ones(1)
    unconstrained = initial[0]-(dense@f0)/(dense@dense)
    lower, upper = initial[0]-.001, initial[0]+.001
    expected = np.clip(unconstrained, lower, upper)
    result = least_squares(lambda x: f0+operator@(x-initial), initial,
        jac=lambda x: operator, bounds=([lower], [upper]), method="trf", tr_solver="lsmr",
        loss="linear", x_scale=1., ftol=1e-12, xtol=1e-12, gtol=1e-12,
        tr_options={"regularize": True, "atol": 1e-10, "btol": 1e-10, "maxiter": 10})
    assert result.success and lower <= result.x[0] <= upper
    assert abs(result.x[0]-expected) < 1e-9
    assert result.cost < .5*f0@f0


def test_damped_lsqr_manual_rank_deficient_reference_and_no_recovery_credit():
    matrix = np.array([[2., 0.]])
    rhs = np.array([3.])
    damp = .5
    actual = lsqr(matrix, rhs, damp=damp, atol=1e-12, btol=1e-12)[0]
    np.testing.assert_allclose(actual, [6/(4+damp*damp), 0.], rtol=1e-13, atol=1e-13)
    # Fit can improve while moving away from a manufactured initial truth.
    initial, truth = np.array([0., 1.]), np.array([0., 1.])
    final = initial+actual
    assert np.linalg.norm(matrix@final-rhs) < np.linalg.norm(matrix@initial-rhs)
    assert np.linalg.norm(final-truth) > np.linalg.norm(initial-truth)


@pytest.mark.parametrize("receivers,frames", [((0,), ()), ((0,), ((.2, (0,), (1,)),)),
    ((0,), ((.2, (0,), (0,)), (.2, (1,), (1,)))), ((-1,), ((.2, (0,), (0,)),))])
def test_invalid_layout_rejected(receivers, frames):
    with pytest.raises(ValueError):
        Layout(3, receivers, frames)
