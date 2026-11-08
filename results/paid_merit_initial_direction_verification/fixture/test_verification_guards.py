"""Pure/mock regressions only: no project arrays, production imports or PDE solves."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("mock_test_initial_runner", "run_initial_direction_verification.py")
replayer = load("mock_test_initial_replayer", "replay_raw_witnesses.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def test_default_commands_load_no_project_inputs_or_numeric_bindings(capsys, monkeypatch):
    monkeypatch.delattr(runner, "np", raising=False)
    monkeypatch.delattr(replayer, "np", raising=False)
    assert "np" not in runner.__dict__
    assert "np" not in replayer.__dict__
    assert runner.main([]) == replayer.main([]) == 0
    output = capsys.readouterr().out
    assert "source_only" in output
    assert "np" not in runner.__dict__
    assert "np" not in replayer.__dict__


@pytest.mark.parametrize("raise_in_original", [False, True])
def test_capture_preserves_argument_identity_return_object_and_restores_binding(monkeypatch, raise_in_original):
    monkeypatch.setattr(runner, "np", np, raising=False)
    snapshot = SimpleNamespace(frequencies=[], damping=np.zeros(3), to_dict=lambda: {"mock_snapshot": True})
    missing, contexts = {1.0: object()}, {1.0: object()}
    certificate = SimpleNamespace(direction=np.ones(3), approximate_gradient=np.ones(3),
                                  to_dict=lambda: {"mock_certificate": True})
    observed_calls = []

    def original(issuing_oracle, actual_snapshot, actual_missing, actual_contexts):
        observed_calls.append((issuing_oracle, actual_snapshot, actual_missing, actual_contexts))
        assert actual_snapshot is snapshot and actual_missing is missing and actual_contexts is contexts
        if raise_in_original:
            raise ValueError("mock original certificate failure")
        return certificate

    class Oracle:
        def __init__(self, *args):
            self.receipt = SimpleNamespace(objective=1.0)

        def evaluate(self, _):
            return self.receipt

        def accept(self, actual):
            assert actual is self.receipt

        def cost_snapshot(self):
            return {}

    adapter = SimpleNamespace(evaluate_cached_merit=original)

    class Proxy:
        def __init__(self, oracle, *args, **kwargs):
            self.oracle = oracle
            assert kwargs["mode"] == "fixed_sweeps_certified"
            assert kwargs["rank"] == 64 and kwargs["sweeps"] == 4

        def propose(self, _, receipt):
            assert receipt is self.oracle.receipt
            result = adapter.evaluate_cached_merit(self.oracle, snapshot, missing, contexts)
            assert result is certificate
            return SimpleNamespace(direction=result.direction, fallback_required=False,
                                   certified_descent=True, diagnostics={})

        def cost_snapshot(self):
            return {}

    adapter.CachedFieldProposal = Proxy
    helpers = SimpleNamespace(Helmholtz2D=lambda **kwargs: object(), ExactMeritOracle=Oracle)
    prepared = {"initial": np.ones(3), "blocks": (), "receivers": ()}
    original_protocol = {"physics": {}, "gmres": {}}
    if raise_in_original:
        with pytest.raises(ValueError, match="mock original certificate failure"):
            runner.production_witness(helpers, adapter, prepared, original_protocol, {})
    else:
        result = runner.production_witness(helpers, adapter, prepared, original_protocol, {})
        assert result["certificate"] == {"mock_certificate": True}
    assert len(observed_calls) == 1
    assert adapter.evaluate_cached_merit is original


@pytest.mark.parametrize("exception_class", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("location", ["independent_gradient", "production_witness"])
def test_handled_interrupt_saves_partial_arrays_and_preserves_other_seven_slots(
        tmp_path, monkeypatch, exception_class, location):
    # Only static source/protocol bytes are copied. Raw project NPZ is never read.
    repo = tmp_path / "invented_repo"
    candidate = repo / "results/provisional_software_benchmark/matched_cost_pilot"
    candidate.mkdir(parents=True)
    real_repo = HERE.parents[1] / "wavecert-fwi"
    original_raw = (real_repo / "results/provisional_software_benchmark/matched_cost_pilot/protocol.json").read_bytes()
    (candidate / "protocol.json").write_bytes(original_raw)
    benchmark = tmp_path / "invented_terminal_benchmark"
    benchmark.mkdir()
    write_json(benchmark / "status.json", {"status": "complete_engineering_benchmark"})
    objective = tmp_path / "independent_source.py"
    objective.write_bytes((HERE.parent / "paid_merit_objective_audit/audit_saved_models.py").read_bytes())

    class Checks:
        def __init__(self):
            self.count, self.failures = 0, []

        def require(self, condition, context, detail=None):
            self.count += 1
            if not condition:
                self.failures.append({"context": context, "detail": detail})

    class InventedCase:
        def __init__(self, *args):
            self.truth = self.initial = np.full(2401, 0.1)
            self.damping = np.zeros(2401)
            self.rhs = np.zeros((2401, 6), dtype=np.complex128)
            self.sources, self.receivers = list(range(6)), list(range(20))

    independent = SimpleNamespace(
        source_digest=lambda _: {"sha256": runner.PRODUCTION_SHA, "files": 161},
        Checks=Checks, load_cases=lambda *args: [np.full((24, 24), 0.1) for _ in range(8)],
        IndependentCase=InventedCase, scalar_norm=lambda values: float(np.linalg.norm(values)))
    monkeypatch.setattr(runner, "import_file", lambda *args, **kwargs: independent)
    helpers = SimpleNamespace(prepare_case=lambda *args: {"initial": np.full(2401, 0.1)})
    monkeypatch.setattr(runner, "load_production", lambda _: (helpers, object()))

    def fake_gradient(case, model, protocol, checks, arrays):
        arrays["mock_partial_gradient_bytes"] = np.arange(7, dtype=float)
        if location == "independent_gradient":
            raise exception_class("invented interruption after partial bytes")
        return np.ones(2401), 1.0, []

    def fake_production(*args):
        raise exception_class("invented interruption before certificate return")

    monkeypatch.setattr(runner, "independent_gradient", fake_gradient)
    monkeypatch.setattr(runner, "production_witness", fake_production)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    output = tmp_path / "new_mock_witness"
    with pytest.raises(exception_class):
        runner.run(repo, benchmark, output, objective, HERE / "protocol.json",
                   digest(HERE / "run_initial_direction_verification.py"), digest(HERE / "protocol.json"))
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed_or_interrupted"
    assert status["cases"][0]["status"] == "interrupted"
    assert all(row["status"] == "unattempted" for row in status["cases"][1:])
    receipt = json.loads((output / "case_00_receipt.json").read_text())
    assert receipt["status"] == "interrupted" and receipt["raw_arrays_available"] is True
    assert receipt["raw_witness_complete"] is False
    with np.load(output / receipt["raw_arrays_artifact"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["mock_partial_gradient_bytes"], np.arange(7, dtype=float))
    assert not (output / "complete_receipt.json").exists()


def array_descriptor(values, *, snapshot=False):
    values = np.asarray(values)
    return {"shape": list(values.shape), "dtype": str(values.dtype) if snapshot else values.dtype.str,
            "sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()}


def mock_witness_bundle(tmp_path, monkeypatch, *, scenario="clear"):
    """Invented vectors and a mocked stencil; exercises audit decisions, no PDE."""
    directory = tmp_path / "invented_raw_bundle"
    directory.mkdir()
    frequencies = [2.5, 3.0, 3.5, 4.0]
    protocol = {"frequencies_hz": frequencies, "numeric_checks": {
        "scalar_relative_allowance": 5e-10, "scale_floor": 1e-12,
        "vector_relative_L2_maximum": 2e-10, "observation_relative_L2_maximum": 1e-10,
        "linear_relative_residual_maximum": 1e-9, "residual_norm_action_scale_allowance": 1e-12}}
    write_json(directory / "protocol.json", protocol)
    model, damping = np.full(2401, 0.1), np.zeros(2401)
    sources, receivers = np.arange(6, dtype=np.int64), np.arange(20, dtype=np.int64)
    state, missing = np.ones((2401, 6), dtype=np.complex128), np.full((2401, 6), 3.0, dtype=np.complex128)
    sign = -1 if scenario == "false_certification" else 1
    adjoint = np.full((2401, 6), sign * 2.0, dtype=np.complex128)
    residual = np.ones((20, 6), dtype=np.complex128)
    adjoint_rhs = np.zeros((2401, 6), dtype=np.complex128)
    adjoint_rhs[receivers, :] = residual
    approximate, exact = np.zeros(2401), np.zeros(2401)
    arrays = {"initial_model": model, "production_damping": damping, "independent_damping": damping,
              "source_rhs": np.zeros_like(state), "source_indices": sources, "receiver_indices": receivers}
    contributions = []
    for fi, frequency in enumerate(frequencies):
        omega = 2.0 * math.pi * frequency
        contribution = omega ** 2 * np.real(np.sum(np.conjugate(missing) * state, axis=1))
        approximate += contribution
        contributions.append(contribution)
        exact += omega ** 2 * np.real(np.sum(np.conjugate(adjoint) * state, axis=1))
        for name, values in (("state", state), ("adjoint", adjoint),
                             ("observed", np.zeros_like(residual)), ("residual", residual),
                             ("adjoint_rhs", adjoint_rhs)):
            arrays[f"f{fi}_independent_{name}"] = values
        for name, values in (("cached_field", state), ("missing_field", missing),
                             ("observed_slots", np.zeros_like(residual)), ("residual_slots", residual),
                             ("residual_unique", residual), ("receiver_expand_indices", receivers)):
            arrays[f"f{fi}_production_{name}"] = values
    direction = -approximate / math.hypot(*(float(v) for v in approximate))
    arrays.update(direction=direction, approximate_gradient=approximate, independent_true_gradient=exact)
    dd, center = replayer.dot(exact, direction), replayer.dot(approximate, direction)
    discrepancy = abs(dd - center)
    if scenario == "historical_miss":
        bound = discrepancy / 2
    elif scenario == "literal_miss_only":
        allowance = 1e-9 * max(abs(dd), abs(center), discrepancy, 1.0)
        bound = discrepancy - allowance / 4
    else:
        bound = discrepancy * 1.1
    allowance = 1e-9 * max(abs(dd), abs(center), bound, 1.0)
    frames, frequency_records = [], []
    for fi, frequency in enumerate(frequencies):
        frame = {"source_indices": sources.tolist(), "receiver_indices": receivers.tolist()}
        for name in ("cached_field", "observed_slots", "residual_slots", "residual_unique", "receiver_expand_indices"):
            frame[name] = array_descriptor(arrays[f"f{fi}_production_{name}"], snapshot=True)
        frames.append(frame)
        frequency_records.append({"frequency_hz": frequency,
            "missing_raw_residual_norms": [0.0] * 6,
            "approximate_directional_derivative": replayer.dot(contributions[fi], direction),
            "directional_error_bound": bound / 4,
            "directional_bound_terms": [[bound / 72] * 6 for _ in range(3)]})
    certified = scenario == "false_certification" or center + bound < 0
    certificate = {"frequency_records": frequency_records, "approximate_directional_derivative": center,
        "directional_error_bound": bound, "certified_upper_derivative": center + bound,
        "certified_descent": certified,
        "direction_sha256": array_descriptor(direction)["sha256"],
        "approximate_gradient_sha256": array_descriptor(approximate)["sha256"]}
    snapshot = {"model_sha256": array_descriptor(model)["sha256"],
                "model": array_descriptor(model, snapshot=True),
                "damping": array_descriptor(damping, snapshot=True)}
    if scenario == "model_snapshot_tamper":
        snapshot["model"]["sha256"] = "0" * 64
    if scenario == "damping_snapshot_tamper":
        snapshot["damping"]["sha256"] = "0" * 64
    array_path = directory / "toy_arrays.npz"
    array_path.write_bytes(b"MOCK NPZ intercepted by test-only loader; no project data")
    rows = []
    for index in range(8):
        receipt = {"status": "complete", "raw_arrays_artifact": array_path.name,
            "raw_arrays_file_sha256": digest(array_path),
            "arrays": {name: array_descriptor(value) for name, value in arrays.items()},
            "production": {"snapshot": snapshot, "certificate": certificate, "frames": frames,
                "proposal_diagnostics": {"field_sha256": {str(f): array_descriptor(missing)["sha256"] for f in frequencies}}},
            "independent_true_directional_derivative": dd,
            "false_certification_using_independent_gradient": bool(certified and dd >= 0),
            "bound_covers_without_extra_allowance": discrepancy <= bound,
            "historical_helper_FP_allowance": allowance,
            "bound_covers_with_historical_helper_allowance": discrepancy <= bound + allowance}
        receipt_path = directory / f"case_{index:02d}.json"
        write_json(receipt_path, receipt)
        rows.append({"case_index": index, "receipt_artifact": receipt_path.name, "receipt_sha256": digest(receipt_path)})
    status = {"status": "complete_initial_direction_verification", "cases": rows,
              "protocol_sha256": digest(directory / "protocol.json"),
              "verification_result": "issues_require_review" if scenario == "upstream_issues" else "clear"}
    write_json(directory / "status.json", status)
    complete = {**status, "artifact_sha256": {p.name: digest(p) for p in directory.iterdir() if p.is_file()}}
    write_json(directory / "complete_receipt.json", complete)

    class MockArchive:
        files = list(arrays)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, key):
            return arrays[key]

    monkeypatch.setattr(np, "load", lambda *args, **kwargs: MockArchive())
    monkeypatch.setattr(replayer, "apply", lambda model, damping, field, frequency, adjoint=False:
                        adjoint_rhs.copy() if adjoint else np.zeros_like(field))
    return directory


@pytest.mark.parametrize("scenario,expected,context", [
    ("clear", "clear_raw_witness_replay", None),
    ("literal_miss_only", "clear_raw_witness_replay", None),
    ("false_certification", "issues_require_review", "no_independently_replayed_false_certification"),
    ("historical_miss", "issues_require_review", "no_replayed_bound_miss_beyond_historical_allowance"),
    ("upstream_issues", "issues_require_review", "upstream_verification_must_be_clear"),
    ("model_snapshot_tamper", "issues_require_review", "raw_initial_model_actual_snapshot_binding"),
    ("damping_snapshot_tamper", "issues_require_review", "raw_production_damping_actual_snapshot_binding"),
])
def test_raw_replay_adverse_gates_are_independent_of_self_consistent_flags(
        tmp_path, monkeypatch, scenario, expected, context):
    directory = mock_witness_bundle(tmp_path, monkeypatch, scenario=scenario)
    output = tmp_path / "new_replay_report.json"
    result = replayer.run(directory, output, digest(HERE / "replay_raw_witnesses.py"))
    assert result["result"] == expected
    report = json.loads(output.read_text())
    assert report["all8_case_slots_retained"] and report["directions_assessed"] == 8
    if context is not None:
        assert any(context in issue["context"] for issue in report["issues"])
    if scenario == "literal_miss_only":
        assert report["literal_bound_miss_count"] == 8
        assert all(row["bound_covers_with_historical_helper_allowance"] for row in report["cases"])
