"""Small manufactured matrix/tangent controls only; no project data."""
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import run_probe as probe


def setup():
    shape, h = (4, 5), .25
    k, damping = probe.graph(shape, h)
    initial = np.linspace(.3, .7, 20)
    truth = initial+.015*np.sin(np.arange(20)+.4)
    rhs = np.zeros((20, 2), dtype=np.complex128)
    rhs[[6, 13], [0, 1]] = 1/h**2
    receivers = np.array([0, 3, 8, 12, 19])
    operations = {name: dict(status="unattempted", factorizations_attempted=0, factorizations_completed=0,
        rhs_columns_attempted=0, rhs_columns_completed=0) for name in ("truth", "initial", "tangent")}
    return k, damping, initial, truth, rhs, receivers, operations


def test_graph_matches_independent_dense_equation_on_boundaries():
    k, damping = probe.graph((3, 4), .25)
    expected = np.diag(np.full(12, 64.))
    for row in range(12):
        z, x = divmod(row, 4)
        for col in range(12):
            zz, xx = divmod(col, 4)
            if abs(z-zz)+abs(x-xx) == 1:
                expected[row, col] = -16.
    np.testing.assert_array_equal(k.toarray(), expected)
    np.testing.assert_allclose(damping.reshape(3, 4), [[2., 2., 2., 2.], [2., 50/36, 50/36, 2.], [2., 2., 2., 2.]], rtol=2e-15, atol=0)


@pytest.mark.parametrize("frequency", [.5, 2.])
def test_tangent_matches_fixed_central_differences_and_complex_adjoint(frequency):
    k, damping, initial, truth, rhs, receivers, operations = setup()
    retained = {}
    result = probe.frequency_probe(initial, truth, k, damping, rhs, receivers, frequency,
        operations, on_receiver=lambda name, value: retained.update({name: value}))
    omega = 2*np.pi*frequency
    matrix = k+sp.diags(-omega**2*initial+1j*omega*damping)
    delta = truth-initial
    tangent = retained["tangent"]
    errors = []
    for epsilon in (1e-2, 1e-3, 1e-4):
        plus = spla.spsolve(matrix-sp.diags(omega**2*epsilon*delta), rhs)
        minus = spla.spsolve(matrix+sp.diags(omega**2*epsilon*delta), rhs)
        fd = (plus[receivers]-minus[receivers])/(2*epsilon)
        error = probe.norm(fd-tangent)/probe.norm(tangent)
        errors.append(error)
        assert error < 2*epsilon**2+2e-8
    state = spla.spsolve(matrix, rhs)
    residual = retained["initial"]-retained["truth"]
    adjoint_rhs = np.zeros_like(rhs)
    adjoint_rhs[receivers] = residual
    adjoint = spla.spsolve(matrix.conjugate().T, adjoint_rhs)
    gradient = omega**2*np.real(np.sum(np.conjugate(adjoint)*state, axis=1))
    assert probe.real_dot(gradient, delta) == pytest.approx(result["g_data_dot_delta"], rel=1e-11, abs=1e-13)
    assert result["g_data_dot_delta"] == pytest.approx(-probe.real_dot(retained["truth"]-retained["initial"], tangent))
    assert sum(r["factorizations_completed"] for r in operations.values()) == 2
    assert sum(r["rhs_columns_completed"] for r in operations.values()) == 6
    assert errors[-1] < 1e-7
    print({"manufactured_frequency": frequency, "fixed_epsilons": [1e-2, 1e-3, 1e-4], "tangent_FD_relative_errors": errors})


def test_zero_data_change_does_not_become_easy_relative_or_alignment_pass():
    values = np.zeros((5, 2), dtype=np.complex128)
    result = probe.diagnostics(values, values, values)
    assert result["remainder_relative_to_data_change"] is None
    assert result["real_alignment"] is None and result["zero_data_change"] and result["zero_tangent"]


def test_failed_factorization_preserves_stage_and_zero_RHS_attempts(monkeypatch):
    k, damping, initial, truth, rhs, receivers, operations = setup()
    def fail(*args, **kwargs):
        raise RuntimeError("manufactured factorization failure")
    monkeypatch.setattr(spla, "splu", fail)
    with pytest.raises(RuntimeError, match="manufactured"):
        probe.frequency_probe(initial, truth, k, damping, rhs, receivers, .5, operations)
    assert operations["truth"]["status"] == "failed"
    assert operations["truth"]["factorizations_attempted"] == 1
    assert operations["truth"]["rhs_columns_attempted"] == 0
    assert operations["initial"]["status"] == operations["tangent"]["status"] == "unattempted"


def test_default_cli_and_full_planned_denominators(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("No project inputs or solves may be read")
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(probe, "execute", forbidden)
    assert probe.main([]) == 0
    assert "metadata_only_no_project_arrays_or_solves" in capsys.readouterr().out
    protocol = probe.load_protocol(probe.Path(probe.__file__).with_name("protocol.json"))
    rows = probe.planned(protocol)
    assert len(rows) == 8 and sum(len(r["frequencies"]) for r in rows) == 64
    assert sum(op["rhs_columns_planned"] for r in rows for f in r["frequencies"] for op in f["operations"].values()) == 1152
