"""Manufactured math/API/retention controls only; no BP wave execution."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "wavecert-fwi"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


maths = load("_bp_guard_math_test", "guard_math.py")
driver = load("_bp_guard_driver_test", "run_guards.py")


def protocol():
    return driver.load_protocol(HERE / "protocol.json")[0]


def test_same_bilinear_field_preserves_extent_and_nodes():
    x, z = np.meshgrid(np.arange(4.), np.arange(3.), indexing="ij")
    coarse = 2 + x + 2 * z + .25 * x * z
    fine = maths.refine_2x(coarse)
    fx, fz = np.meshgrid(np.arange(0, 3.01, .5), np.arange(0, 2.01, .5), indexing="ij")
    assert fine.shape == (7, 5)
    np.testing.assert_array_equal(fine[::2, ::2], coarse)
    np.testing.assert_array_equal(fine, 2 + fx + 2 * fz + .25 * fx * fz)
    assert np.all((fine >= coarse.min()) & (fine <= coarse.max()))


def test_exact_common_times_use_computed_intersection_only():
    a, b = np.arange(0, 3.5, .5), np.arange(0, 3.25, .25)
    ma, mb = np.ones(len(a), bool), np.ones(len(b), bool)
    ma[[0, -1]] = False
    mb[[0, -1]] = False
    ia, ib = maths.exact_common_indices(a, ma, b, mb)
    np.testing.assert_array_equal(ia, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(ib, [2, 4, 6, 8, 10])
    with pytest.raises(ValueError, match="no exact"):
        maths.exact_common_indices(a, ma, b + .01, mb)


def test_independent_edge_pullback_duality_and_production_match():
    sys.path.insert(0, str(REPO / "src"))
    from wavecert.physics.devito_backend import _edge_padding_pullback

    shape, width = (3, 4), 2
    padded = np.arange(56., dtype=float).reshape(7, 8) - 20
    perturbation = np.arange(12., dtype=float).reshape(shape) / 11
    independent = maths.independent_edge_fold(padded, shape, width)
    production = _edge_padding_pullback(padded, shape, width)
    np.testing.assert_array_equal(independent, production)
    assert np.sum(padded * np.pad(perturbation, width, mode="edge")) == pytest.approx(np.sum(independent * perturbation))
    assert not np.array_equal(independent, padded[width:-width, width:-width])


@pytest.mark.parametrize("name", ["interior", "edge"])
def test_fixed_parameter_directions_are_positive_bounded_and_normalized(name):
    model = np.linspace(.04, .3, 120).reshape(12, 10)
    direction, scale, mask = maths.parameter_direction(model, name)
    assert np.linalg.norm(direction) == pytest.approx(1)
    for step in protocol()["gradient"]["relative_steps"]:
        perturbation = step * scale * direction
        np.testing.assert_allclose(perturbation[mask] / model[mask], step, rtol=1e-15)
        assert np.all((model - perturbation) > 0)
    assert np.all(direction[~mask] == 0)


def test_near_zero_derivatives_and_zero_norm_are_uninformative():
    diagnostic = maths.derivative_diagnostic(1e-12, 1e-12, 10., 1.)
    assert diagnostic["informative"] is False and diagnostic["relative_error"] is None
    assert maths.relative_record(np.zeros(2), np.zeros(2))["ratio"] is None
    assert maths.derivative_diagnostic(1., .9999, 10., 1.)["relative_error"] == pytest.approx(.0001)


def test_real_ricker_source_is_captured_once_and_explicitly_retained(monkeypatch):
    from devito import Grid
    from devito.operator.operator import Operator
    from examples.seismic import RickerSource, TimeAxis

    def forbidden(*args, **kwargs):
        raise AssertionError("Operator.apply is forbidden in source-only API control")

    monkeypatch.setattr(Operator, "apply", forbidden)
    grid = Grid(shape=(11, 11), extent=(100., 100.), dtype=np.float64)
    axis = TimeAxis(start=0., stop=400., step=.5)

    class Geometry:
        accesses = 0

        @property
        def src(self):
            self.accesses += 1
            return RickerSource(name="manufactured_source", grid=grid, f0=.005, time_range=axis,
                                coordinates=np.asarray([[50., 50.]]), dtype=np.float64)

    geometry = Geometry()
    source, before = maths.held_source(SimpleNamespace(geometry=geometry), 4.)
    assert geometry.accesses == 1
    t = axis.time_values
    expected = 4 * (1 - 2 * (np.pi * .005 * (t - 200)) ** 2) * np.exp(-(np.pi * .005 * (t - 200)) ** 2)
    np.testing.assert_array_equal(source.data[:, 0], expected)
    np.testing.assert_array_equal(source.data, before["source"])
    assert source.data[np.argmin(abs(t - 200)), 0] == 4
    assert 4 * 6.25 ** 2 == 12.5 ** 2


def test_no_wavefield_can_be_written_and_plan_has_all57_calls():
    with pytest.raises(ValueError, match="wavefield"):
        maths.ensure_no_wavefield_arrays({"u": np.zeros((3, 2, 2))}, 1000)
    with pytest.raises(ValueError, match="cap"):
        maths.ensure_no_wavefield_arrays({"x": np.zeros((3, 3))}, 10)
    rows = driver.plan(protocol())
    assert len(rows) == len({(r["case_id"], r["call_id"]) for r in rows}) == 57
    assert sum(row["kind"] == "adjoint" for row in rows) == 3
    assert all(row["status"] == "unattempted" for row in rows)


def test_forward_wrapper_passes_the_held_source_and_records_raw_endpoints(monkeypatch):
    p = protocol()
    times = np.arange(2001, dtype=np.float64) * .5
    coords, rec_coords = maths.geometry(p)
    source = SimpleNamespace(data=((1 - 2 * (np.pi * .005 * (times - 200)) ** 2)
        * np.exp(-(np.pi * .005 * (times - 200)) ** 2))[:, None],
        coordinates=SimpleNamespace(data=coords), _time_range=SimpleNamespace(time_values=times))
    receiver = SimpleNamespace(data=np.zeros((2001, 10)), coordinates=SimpleNamespace(data=rec_coords),
                               _time_range=SimpleNamespace(time_values=times))

    class Geometry:
        accesses = 0
        rec = receiver

        @property
        def src(self):
            self.accesses += 1
            return source

    geometry = Geometry()

    def solve(**kwargs):
        assert kwargs["src"] is source and geometry.accesses == 1
        assert kwargs["save"] is False and kwargs["dt"] == .5
        receiver.data[1:-1] = np.sin(times[1:-1, None] / 40)
        return receiver, kwargs["u"], None

    storage = np.zeros((3, 8, 8))
    field = SimpleNamespace(shape_allocated=(3, 8, 8), data_with_halo=storage, _data=storage)
    solver = SimpleNamespace(forward=solve, op_fwd=lambda **k: SimpleNamespace(
        arguments=lambda **kwargs: dict(time_m=1, time_M=1999)))
    adapter = SimpleNamespace(update_velocity=lambda v: None, geometry=geometry, solver=solver,
        model=SimpleNamespace(vp=object(), grid=SimpleNamespace(origin=(0., 0.))), dt=.5,
        time_values=times, config=SimpleNamespace(spacing=(12.5, 12.5)))
    runtime = dict(np=np, helpers=maths, devito=SimpleNamespace(TimeFunction=lambda **kwargs: field))
    monkeypatch.setattr(driver, "model_record", lambda *a: {})
    arrays, record, returned_field, held = driver.forward(adapter, np.full((4, 4), 2.), 4., False, runtime, p)
    assert held is source and returned_field is field and geometry.accesses == 1
    assert source.data[400, 0] == 4 and record["physical_point_strength_m2"] == 625
    assert arrays["computed_mask"].sum() == 1999 and np.all(arrays["receiver"][[0, -1]] == 0)
    np.testing.assert_array_equal(arrays["source_before"], arrays["source_after"])


def test_default_cli_does_not_load_runtime(monkeypatch, capsys):
    monkeypatch.setattr(driver, "load_runtime", lambda *a: pytest.fail("runtime import"))
    assert driver.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "metadata_only_no_BP_or_PDE" and report["planned_wave_calls"] == 57


def test_worker_interrupt_retains_attempt_and_all19_denominator(monkeypatch, tmp_path):
    p = protocol()
    case = p["inputs"][0]
    output = tmp_path / "case"
    runtime = dict(np=np, helpers=maths)
    monkeypatch.setattr(driver, "load_runtime", lambda *a: runtime)
    monkeypatch.setattr(driver, "source_record", lambda *a: {"input_sha256": case["sha256"]})
    monkeypatch.setattr(np, "load", lambda *a, **k: np.full((96, 192), 2000., dtype=np.float32))
    monkeypatch.setattr(driver, "make_adapter", lambda *a: object())

    def interrupt(*args):
        persisted = json.loads((output / "status.json").read_text())
        assert len(persisted["calls"]) == 19 and persisted["calls"][0]["status"] == "attempted"
        raise KeyboardInterrupt("manufactured interruption")

    monkeypatch.setattr(driver, "forward", interrupt)
    with pytest.raises(KeyboardInterrupt):
        driver.worker(REPO, tmp_path, output, HERE / "protocol.json", case["id"])
    result = json.loads((output / "status.json").read_text())
    assert result["status"] == "failed_or_interrupted"
    assert result["calls"][0]["status"] == "interrupted"
    assert len(result["calls"]) == 19 and sum(r["status"] == "unattempted" for r in result["calls"]) == 18
    assert not (output / "complete_receipt.json").exists()


@pytest.mark.parametrize("cleanup_interrupt", [False, True])
def test_parent_interrupt_preserves_all57_and_stops_child(monkeypatch, tmp_path, cleanup_interrupt):
    events = []

    class Process:
        pid = 123456
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1 or (cleanup_interrupt and self.calls == 2):
                raise KeyboardInterrupt("synthetic parent interrupt")
            events.append(("reaped", timeout))
            return 0

        def poll(self):
            return None

    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(driver.os, "killpg", lambda pid, sig: events.append((pid, sig)))
    monkeypatch.setattr(driver.shutil, "disk_usage", lambda *a: SimpleNamespace(free=2 ** 30))
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        monkeypatch.setenv(key, "1")
    output = tmp_path / "all"
    args = ["--execute-guards", "--repo", str(REPO), "--inputs", str(tmp_path), "--output", str(output),
            "--expected-source-sha256", driver.sha((HERE / "run_guards.py").read_bytes()),
            "--expected-protocol-sha256", driver.sha((HERE / "protocol.json").read_bytes())]
    with pytest.raises(KeyboardInterrupt):
        driver.main(args)
    result = json.loads((output / "status.json").read_text())
    assert len(result["calls"]) == 57 and len(result["case_processes"]) == 3
    assert result["case_processes"][0]["status"] == "interrupted"
    assert all(row["status"] == "unattempted" for row in result["case_processes"][1:])
    assert events[0] == (123456, driver.signal.SIGINT) and events[-1][0] == "reaped"
    if cleanup_interrupt:
        assert events[1] == (123456, driver.signal.SIGKILL)
    assert not (output / "complete_receipt.json").exists()


def test_completed_fd_receipts_survive_mid_sequence_interrupt(monkeypatch, tmp_path):
    sys.path.insert(0, str(REPO / "src"))
    from wavecert.physics.devito_backend import _edge_padding_pullback

    p, output = protocol(), tmp_path / "fd_prefix"
    case = p["inputs"][0]

    class Adapter:
        def __init__(self, velocity):
            self.model = SimpleNamespace(vp=SimpleNamespace(data=None), shape=velocity.shape,
                nbl=40, grid=SimpleNamespace(origin=(0., 0.)), critical_dt=1.)
            self.dt = .5
            self.geometry = SimpleNamespace(time_axis=object(), rec_positions=np.zeros((10, 2)))
            self.solver = SimpleNamespace(gradient=lambda **kw: kw["grad"].data.fill(1.),
                op_grad=lambda: SimpleNamespace(arguments=lambda **kw: dict(time_m=1, time_M=1)))
            self.update_velocity(velocity)

        def update_velocity(self, velocity):
            self.model.vp.data = np.pad(velocity, 40, mode="edge")

    runtime = dict(np=np, helpers=maths, backend=SimpleNamespace(_edge_padding_pullback=_edge_padding_pullback),
        devito=SimpleNamespace(Function=lambda **kw: SimpleNamespace(data=np.zeros((272, 176))),
            TimeFunction=lambda **kw: SimpleNamespace(data_with_halo=np.zeros((3, 2, 2)), shape_allocated=(3, 2, 2))),
        seismic=SimpleNamespace(Receiver=lambda **kw: SimpleNamespace(data=np.zeros((3, 10)))))
    monkeypatch.setattr(driver, "load_runtime", lambda *a: runtime)
    monkeypatch.setattr(driver, "source_record", lambda *a: {"input_sha256": case["sha256"]})
    monkeypatch.setattr(np, "load", lambda *a, **k: np.full((96, 192), 2000., dtype=np.float32))
    monkeypatch.setattr(driver, "make_adapter", lambda velocity, *a: Adapter(velocity))
    monkeypatch.setattr(driver, "save_arrays", lambda path, *a: path.write_bytes(b"manufactured-placeholder"))
    count = 0

    def manufactured_forward(adapter, velocity, *args):
        nonlocal count
        count += 1
        if count == 9:
            raise KeyboardInterrupt("after first complete FD pair")
        adapter.update_velocity(velocity)
        return dict(receiver=np.full((3, 10), float(np.mean(velocity))), computed_mask=np.array([False, True, False])), {}, object(), object()

    monkeypatch.setattr(driver, "forward", manufactured_forward)
    with pytest.raises(KeyboardInterrupt):
        driver.worker(REPO, tmp_path, output, HERE / "protocol.json", case["id"])
    status = json.loads((output / "status.json").read_text())
    assert len(status["calls"]) == 19 and len(status["derivative_checks"]) == 6
    first = status["derivative_checks"][0]
    assert first["status"] == "complete" and first["plus"]["artifact_sha256"] and first["minus"]["restored_probe_model"]
    assert status["derivative_checks"][1]["status"] == "attempted"
    assert next(r for r in status["calls"] if r["call_id"] == "fd_interior_0.001_plus")["result"]["objective"] > 0
    assert next(r for r in status["calls"] if r["call_id"] == "probe_adjoint")["result"]["artifact_sha256"]
    assert next(r for r in status["calls"] if r["call_id"] == "fd_interior_0.0003_plus")["restoration"]["status"] == "restored"
    assert not (output / "complete_receipt.json").exists()


@pytest.mark.filterwarnings("ignore:Passing `SubDomain`s to `Grid` on instantiation:DeprecationWarning")
def test_real_manufactured_forward_adjoint_and_fd_integration(capsys):
    """21x21 constant synthetic velocity; five short wave calls, no BP inputs."""
    import copy

    p = copy.deepcopy(protocol())
    runtime = driver.load_runtime(REPO, p)
    backend, devito = runtime["backend"], runtime["devito"]
    velocity = np.full((21, 21), 2., dtype=np.float64)
    source = np.asarray([[125., 50.]])
    receivers = np.column_stack((np.asarray([2, 3, 4, 5, 6, 14, 15, 16, 17, 18]) * 12.5, np.full(10, 50.)))
    adapter = backend.DevitoAcousticFWI(velocity, source_coordinates=source, receiver_coordinates=receivers,
        config=backend.DevitoAcousticConfig(spacing=(12.5, 12.5), nbl=40, t0=0., tn=80., f0=.005,
                                           dt=.5, max_velocity=2.1, space_order=4, dtype="float64"))
    # Test-only layout clone: project protocol and its824MB limit/shape remain unchanged.
    nt = len(adapter.time_values)
    metadata = devito.TimeFunction(name="test_layout", grid=adapter.model.grid, time_order=2, space_order=4,
                                  save=nt, dtype=np.float64)
    assert metadata._data is None
    p["memory"]["actual_saved_layout"] = [int(n) for n in metadata.shape_allocated]
    p["memory"]["saved_byte_cap"] = 32 * 1024 ** 2
    p["memory"]["minimum_available_RAM_before_saved_allocation_bytes"] = 64 * 1024 ** 2
    observed, _, _, _ = driver.forward(adapter, velocity, 1., False, runtime, p)
    probe = .99 * velocity
    predicted, record, u, held = driver.forward(adapter, probe, 1., True, runtime, p)
    residual = predicted["receiver"] - observed["receiver"]
    residual[~predicted["computed_mask"]] = 0
    assert np.linalg.norm(residual) > 0
    actual_bounds = []
    arrays, adjoint = driver.adjoint_gradient(adapter, u, held, residual, runtime, actual_bounds.append)
    assert adjoint["pullback_numerical_agreement"] is True
    assert actual_bounds == [dict(time_m=1, time_M=nt - 2)]
    gradient = arrays["production_fold"]
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0
    del u
    model = 1 / probe ** 2
    direction = model / np.linalg.norm(model)
    epsilon = .0003 * np.linalg.norm(model)
    objectives = []
    try:
        for sign in [1., -1.]:
            trial = 1 / np.sqrt(model + sign * epsilon * direction)
            result, _, _, _ = driver.forward(adapter, trial, 1., False, runtime, p)
            mask = result["computed_mask"]
            objectives.append(.5 * float(np.sum((result["receiver"][mask] - observed["receiver"][mask]) ** 2)))
    finally:
        adapter.update_velocity(probe)
    restored = driver.model_record(adapter, probe, runtime)
    assert restored["actual_physical_m_sha256"] == maths.array_sha(model)
    fd = (objectives[0] - objectives[1]) / (2 * epsilon)
    diagnostic = maths.derivative_diagnostic(np.sum(gradient * direction), fd, np.linalg.norm(gradient), 1.)
    assert diagnostic["informative"] and diagnostic["relative_error"] <= .01
    assert np.all(observed["receiver"][[0, -1]] == 0)
    assert record["time_bounds"] == actual_bounds[0]
    # A single visible manufactured-only result; this is not a BP/project outcome.
    with capsys.disabled():
        print(json.dumps(dict(manufactured_only=True, shape=[21, 21], nt=nt,
            saved_bytes=record["materialized_wavefield_bytes"], observed_norm=float(np.linalg.norm(observed["receiver"])),
            residual_norm=float(np.linalg.norm(residual)), gradient_norm=float(np.linalg.norm(gradient)),
            forward_bounds=record["time_bounds"], adjoint_bounds=actual_bounds[0], derivative=diagnostic), allow_nan=False))
