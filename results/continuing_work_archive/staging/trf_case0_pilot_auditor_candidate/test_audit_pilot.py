"""Manufactured closed-packet controls only; never load a project NPZ or worker."""
import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_scalar_core as S

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("manufactured_audit_wrapper", HERE / "audit_pilot.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
C = S.C


def trace(records):
    h = hashlib.sha256(b"bounded-exact-record-stream-v1\0").digest()
    for index, record in enumerate(records, 1):
        raw = C.canonical(record)
        h = hashlib.sha256(h + index.to_bytes(8, "big") + len(raw).to_bytes(8, "big") + raw).digest()
    total = len(records)
    return {"total_count": total, "finalized_count": total, "retained_count": min(16, total),
        "omitted_count": max(0, total - 16), "head_limit": 8, "tail_limit": 8,
        "pending_record": None, "pending_record_unsealed": False, "chain_sha256": h.hex()}


def wait_row(pid=987654):
    return {"actual_pid": pid, "observed_pgid": pid, "actual_wait_returncode": 0, "wait_observed": True,
        "terminal_state_known": True, "owned_process_group_alive_after_cleanup": False,
        "closed_log": True, "unknown_work": False, "post_wait_IO_errors": [], "deferred_parent_signals": [],
        "clock": "time.perf_counter_same_host", "wait_completed_perf_counter": 90.}


class ManufacturedPacket:
    """A source-shaped fake packet. Its mock receipts confer no real OS credit."""
    def __init__(self, root):
        self.root, self.packet = Path(root), Path(root) / "packet"
        self.packet.mkdir()
        self.manifest = A.bounded_json(HERE / "AUDITOR_SOURCE_MANIFEST.json")
        self.protocol = A.bounded_json(HERE.parent / "trf_realcase_worker_candidate/project_protocol.json")
        self.plan = A.bounded_json(HERE.parent / "trf_owned_serial_pilot_parent_candidate/PILOT_PLAN.json")
        self.initial, self.selected, self.truth = [.20] * 2401, [.24] * 2401, [.25] * 2401
        self.protocol["cases"][0]["raw_float64_sha256"] = C.sha(C.f64_bytes([.25] * 576))
        self.protocol["cases"][0]["initial_float64_sha256"] = C.sha(C.f64_bytes(self.initial))
        self.protocol_path = self.js(self.root / "protocol.json", self.protocol)
        self.protocol_sha = A.digest(self.protocol_path)
        self.plan["worker_contract"]["project_protocol.json"] = self.protocol_sha
        self.plan_path = self.js(self.root / "plan.json", self.plan)
        self.parent_contract = dict(A.PARENT_CONTRACT, **{"PILOT_PLAN.json": A.digest(self.plan_path)})
        self.sources = {"worker_contract": self.plan["worker_contract"], "parent_contract": self.parent_contract,
            "role_bindings": {"auditor": A.digest(HERE / "AUDITOR_SOURCE_MANIFEST.json"), "criteria": A.CRITERIA_SHA}}
        self.raw_path = self.root / "raw.npz"
        values = {k: [0] * math.prod(shape) for k, (shape, _) in C.RAW_SCHEMA.items()}
        values["models"] = [.25] * (192 * 576)
        values["shape"], values["spacing"] = [24, 24], [1.]
        self.npz(self.raw_path, C.RAW_SCHEMA, values)
        self.initial_path = self.root / "initial.npz"
        self.npz(self.initial_path, C.INITIAL_SCHEMA, {"initial_m": self.initial * 8, "case_indices": list(range(8)), "shape": [49, 49]})
        old = self.protocol["historical_API"]["source_sha256"]
        self.go = {k: True for k in ("execution_authorized", "independent_worker_source_review_clear", "external_supervisor_review_clear",
            "independent_auditor_review_clear", "runtime_review_clear", "mandatory_frozen_controls_passed")}
        self.go.update(authorization="GO_TRF_CASE0_TWO_RETURN_INTEGRATION_PILOT", allowed_slot_indices=[0, 1], full48_authorized=False,
            source_files=self.sources["worker_contract"], parent_contract=self.parent_contract, review_bindings=self.sources["role_bindings"],
            auditor_source_sha256=A.digest(HERE / "AUDITOR_SOURCE_MANIFEST.json"), audit_criteria_sha256=A.CRITERIA_SHA, admission_sha256="b" * 64)
        self.go_sha = A.digest(self.js(self.packet / "supplied_root_go.json", self.go))
        self.js(self.packet / "project_plan.json", self.plan)
        for name in ("source_before.json", "source_after.json"):
            self.js(self.packet / name, self.sources)
        self.js(self.packet / "admission.json", {"original_root_go_sha256": self.go_sha, "packet_root_go_sha256": self.go_sha,
            "observed_sources": self.sources, "original_admission_sha256": "b" * 64,
            "registered_controls": {"groups": self.plan["required_controls"], "all_actual_counts_verified": True},
            "observed_runtime": {"packages": {k: self.plan["control_runtime"][k] for k in ("numpy", "scipy", "pytest")},
                "threads": self.plan["threads"], "numerical_imports_in_parent": 0, "python": "3.11.18 manufactured", "platform": "Linux manufactured"}})
        self.attempts = []
        for index in range(2):
            directory = self.packet / f"slot_{index:02d}"
            worker = directory / "worker"
            worker.mkdir(parents=True)
            started, age = 100. + index * 7., 6.
            request = {"parent_pid": 654321, "absolute_started": started, "optimizer_absolute_deadline": started + 7.5,
                "complete_return_absolute_deadline": started + 8., "slot_index": index, "arm": self.plan["slots"][index]["arm"],
                "case_index": 0, "clock": "time.perf_counter_same_host", "source_contract_sha256": C.sha(C.canonical(self.sources["worker_contract"]))}
            request_path = self.js(directory / "request.json", request)
            accepted = {"slot": self.plan["slots"][index], "absolute_started": started, "pre_admission_age": .1,
                "source_files": self.sources["worker_contract"], "mode": "case0_two_return_integration_pilot", "external_wait_and_audit_required": True,
                "root_go_sha256": self.go_sha, "request_sha256": A.digest(request_path), "source": {"production_source_sha256": A.PRODUCTION_SHA,
                    "production_files": 161, "historical_API_files": 25, "source_files": self.sources["worker_contract"]}}
            self.js(worker / "admission.json", accepted)
            self.js(worker / "project_plan.json", self.protocol)
            self.js(worker / "source_contract_before.json", self.sources["worker_contract"])
            for name in ("source_before.json", "source_after.json"):
                self.js(directory / name, self.sources)
            payload = S.trf_payload()
            payload["initial_model"], payload["selected_model"] = self.initial, self.selected
            ih, sh = C.sha(C.f64_bytes(self.initial)), C.sha(C.f64_bytes(self.selected))
            for event in payload["TRF_scalar_events"]:
                if event["model_sha256"] == C.sha(C.f64_bytes([.25] * 2401)):
                    event["model_sha256"] = ih
            costs, context, _, _, _ = S.cost_payload(1 if index else 0)
            payload.update(costs, arm=self.plan["slots"][index]["arm"], timing_origin="supplied_pre_entry_absolute", pre_entry_seconds=.2,
                admission_receipt={"project_protocol_sha256": self.protocol_sha, "historical_API": {"actual_public_SciPy_TRF_function": True,
                    "components": {k: old[k] for k in ("bounded_exact.py", "cached_jacobian.py", "public_map_builder.py")}}})
            if index == 0:
                normalization = {"current_model_sha256": sh, "initial_header": {"model_sha256": ih, "raw_objective": 10.},
                    "native_last_known_cumulative": costs["native_cumulative_once"], "unknown_work": False, "failed_dispatch_work_may_exceed_prefix": False}
                checkpoints = [{"kind": kind, "model_sha256": h, "elapsed_seconds": age_, "within_budget": True,
                    "objective": objective, "online_costs": {"oracle": {"native": costs["native_cumulative_once"]}}}
                    for kind, h, age_, objective in (("initial", ih, 1., 1.), ("accepted", sh, 2., .8))]
                payload.update(component_prefix=None, normalization_prefix=normalization, exact_engine={"status": payload["status"],
                    "initial_objective": 1., "final_objective": .8, "checkpoints": checkpoints, "checkpoint_trace": trace(checkpoints),
                    "steps": [], "step_trace": trace([]), "all_library_evaluation_trace": {k: trace([])[k] for k in ("total_count", "chain_sha256")}})
            prepared = {"raw24_m": [.25] * 576, "initial49_m": self.initial, "truth49_m": self.truth}
            self.npz(worker / "prepared_models.npz", C.PREPARED_SCHEMA, prepared)
            self.npz(worker / "observations.npz", C.OBSERVATION_SCHEMA, {"observed": [1 + 2j] * 960})
            self.npz(worker / "returned_models.npz", C.RETURNED_SCHEMA, {"initial_m": self.initial, "selected_m": self.selected})
            survey = {"frequencies_hz": self.protocol["frequencies_hz"], "sources": self.protocol["source_indices"], "receivers": self.protocol["receiver_indices"]}
            binding = {"case_index": 0, **{key + "_sha256": C.sha(C.f64_bytes(value)) for key, value in prepared.items()},
                "all8_observations_sha256": C.sha(b"".join(__import__("struct").pack("<dd", 1., 2.) for _ in range(960))), "survey_sha256": C.sha(C.canonical(survey))}
            quality = {"computed_after_selection_returned": True, "used_for_selection": False,
                "initial_relative_velocity_error": math.sqrt(1.25) - 1., "selected_relative_velocity_error": math.sqrt(25 / 24) - 1.}
            result = {"status": "WORKER_RETURNED_PENDING_EXTERNAL_WAIT_AND_INDEPENDENT_AUDIT", "slot": self.plan["slots"][index],
                "source_files": self.sources["worker_contract"], "method_policy_valid": True, "complete_OS_process_eligibility_available": False,
                "actual_external_wait_required": True, "method_status": payload["status"], "entry_origin": started, "worker_pre_result_age": 5.,
                "API_return_gate_only": {"eligible": True, "cap_seconds": 8., "actual_complete_return_seconds": 4.},
                "case_binding": binding, "post_return_quality": quality, "evidence": {"context": dict(context, admission=accepted), "stop_signals": []}}
            suffixes = {"numpy": "/numpy/__init__.py", "scipy": "/scipy/__init__.py", "wavecert.physics.helmholtz": "/src/wavecert/physics/helmholtz.py",
                "wavecert.physics.exact_merit": "/src/wavecert/physics/exact_merit.py", "matched_driver": "/api/matched_driver.py", "public_map_builder": "/api/public_map_builder.py"}
            origins = {k: "/manufactured" + v for k, v in suffixes.items()}
            result["evidence"]["context"]["runtime_origins"] = origins
            events = [{"kind": "exact_GO_source_request_persisted_before_numerical_imports"},
                {"kind": "actual_native_numerical_origins_before_case_work", "origins": origins, "manufactured_runtime": False},
                {"kind": "raw_initial_truth_prepared_before_observations", "initial_sha256": ih, "truth_sha256": binding["truth49_m_sha256"], "old_truth_bitguard": False}]
            for batch, frequency in enumerate(self.protocol["frequencies_hz"]):
                for completed in (False, True):
                    count = batch + int(completed)
                    prefix = {"observation_dispatches_attempted": batch + 1, "observation_dispatches_completed": count,
                        "observation_call_inflight": not completed, "unknown_work": not completed, "active_frequency_hz": frequency,
                        "last_known_observation_counts": C.expected_counts(factorizations=count, forward_rhs=count * 6)}
                    events.append({"kind": "observation_dispatch_returned" if completed else "observation_dispatch_attempt", "prefix": prefix})
            events += [{"kind": "all8_data_prepared_before_policy", "binding": binding, "counts": payload["preparation_costs_disjoint"]},
                {"kind": "genuine_cold_native_owner_constructed", "native_class": "ExactMeritOracle", "prefix": {"physics": C.expected_counts(), "oracle": {}}},
                {"kind": "API_returned_selection_frozen", "selected": sh, "observed_return": started + 4.},
                {"kind": "post_return_quality_completed_without_reselection", "metrics": quality}]
            for sequence, event in enumerate(events):
                event.update(sequence=sequence, observed_perf_counter=started + sequence / 10.)
            (worker / "events.jsonl").write_bytes(b"\n".join(A.canonical(event) for event in events) + b"\n")
            payload_path = self.js(worker / "returned_payload.json", payload)
            result_path = self.js(worker / "result.json", result)
            row = dict(wait_row(987650 + index), slot=self.plan["slots"][index], status="attempted", eligible=True,
                eligible_before_external_return_observation=True, parent_signals_outside_slot=[], absolute_started=started,
                external_return_observed_perf_counter=started + age, external_complete_return_seconds=age,
                external_wait_cleanup_logclose_seconds=5.5, case_binding=binding, worker_result_sha256=A.digest(result_path),
                returned_payload_sha256=A.digest(payload_path))
            row["actual_wait_receipt_sha256"] = A.digest(self.js(directory / "actual_wait.json", row))
            self.js(directory / "launch_requested.json", {"launch_state": "NOT_ATTEMPTED", "absolute_started": started})
            self.js(directory / "spawned.json", {"launch_state": "HANDLE_ASSIGNED", "actual_pid": row["actual_pid"], "observed_pgid": row["observed_pgid"]})
            self.js(directory / "slot_return.json", row)
            self.attempts.append(row)
        self.js(self.packet / "attempts.json", self.attempts)
        self.js(self.packet / "pilot_result.json", {"attempts": self.attempts, "unknown_work": False, "failure": None,
            "status": "PILOT_TWO_RETURNS_PENDING_INDEPENDENT_AUDIT_AND_PARENT_EXTERNAL_WAIT", "parent_terminal_externally_observed": False,
            "deferred_parent_signals": []})
        sources = A.control_source_contract(self.protocol, self.sources["worker_contract"], self.parent_contract)
        parent_controls = self.controls("parent_controls", self.plan["required_controls"], sources)
        auditor_controls = self.controls("auditor_controls", {"auditor": self.manifest["registered_control_count"]},
            {"auditor": {k: v["sha256"] for k, v in self.manifest["files"].items()}})
        observer = self.root / "observer.py"
        observer.write_text("# MANUFACTURED SOURCE, not an actual observer\n")
        log = self.root / "external.log"
        log.write_text("MANUFACTURED external terminal log\n")
        self.terminal = dict(wait_row(654321), schema="trf_case0_external_parent_terminal_v1", observation_scope="independent_launcher_actual_wait",
            parent_exit_inferred_from_packet=False, observer_source_sha256=A.digest(observer),
            observer_contract={"wrapper_sha256": A.digest(observer), "launcher_helper_sha256": "c" * 64},
            parent_contract=self.parent_contract, external_parent_started_perf_counter=99., wait_completed_perf_counter=114.,
            parent_complete_observed_perf_counter=115., external_parent_complete_return_seconds=16., closed_log_sha256=A.digest(log))
        terminal_path = self.js(self.root / "terminal.json", self.terminal)
        criteria = HERE.parent / "trf_case0_pilot_audit_criteria_candidate.json"
        self.admission = {"schema": "trf_case0_postreturn_audit_go_v1", "authorization": "GO_POSTRETURN_TRF_CASE0_PILOT_AUDIT",
            "execution_authorized": True, "native_execution_authorized": False, "expected_sources": self.sources,
            "pilot_root_go_sha256": self.go_sha, "observer_contract": self.terminal["observer_contract"],
            "files": {k: self.bound(v) for k, v in {"source_manifest": HERE / "AUDITOR_SOURCE_MANIFEST.json", "criteria": criteria,
                "protocol": self.protocol_path, "parent_plan": self.plan_path, "raw_validation": self.raw_path, "frozen_initial": self.initial_path,
                "external_parent_terminal": terminal_path, "external_observer_source": observer, "external_parent_log": log,
                "parent_controls": parent_controls, "auditor_controls": auditor_controls}.items()}}
        self.refresh()

    def js(self, path, value):
        path.write_bytes(A.canonical(value))
        return path

    def bound(self, path):
        return {"path": str(path), "sha256": A.digest(path)}

    def npz(self, path, schema, values):
        return S.archive(path, [(k + ".npy", S.npy(shape, dtype, values[k])) for k, (shape, dtype) in schema.items()])

    def controls(self, name, groups, sources):
        base = self.root / name
        base.mkdir()
        rows = {}
        for group, count in groups.items():
            log = base / (group + ".log")
            log.write_text("MANUFACTURED group log, no actual control proof\n")
            result = base / (group + (".json" if group == "parent" else ".xml"))
            if group == "parent":
                self.js(result, {"tests": count, "failures": 0, "errors": 0, "skipped": 0, "test_ids": [str(i) for i in range(count)]})
            else:
                result.write_text('<testsuite tests="%d" failures="0" errors="0" skipped="0">' % count
                    + ''.join('<testcase classname="manufactured" name="%d"/>' % i for i in range(count)) + '</testsuite>')
            identities = self.js(base / (group + "_sources.json"), sources[group])
            wait = self.js(base / (group + "_wait.json"), wait_row())
            rows[group] = {k: {"path": p.name, "sha256": A.digest(p)} for k, p in
                {"log": log, "result": result, "source_identities": identities, "actual_wait": wait}.items()}
        return self.js(base / "receipt.json", {"status": "REGISTERED_PILOT_CONTROLS_ACTUAL_PASS", "runtime": self.plan["control_runtime"], "groups": rows})

    def refresh(self):
        files = {p.relative_to(self.packet).as_posix(): {"sha256": A.digest(p), "bytes": p.stat().st_size, "bounded_hash_completed": True}
            for p in self.packet.rglob("*") if p.is_file() and p.name != "artifact_inventory.json"}
        inv = {"complete_inventory": True, "known_children_terminal_before_hashing": True, "inventory_excludes_itself": True,
            "files": files, "hashed_bytes": sum(row["bytes"] for row in files.values())}
        inv_sha = A.digest(self.js(self.packet / "artifact_inventory.json", inv))
        self.terminal["packet_inventory_sha256"] = self.admission["packet_inventory_sha256"] = inv_sha
        terminal_path = self.js(self.root / "terminal.json", self.terminal)
        self.admission["files"]["external_parent_terminal"] = self.bound(terminal_path)
        self.admission_path = self.js(self.root / "audit_go.json", self.admission)
        self.admission_sha = A.digest(self.admission_path)

    def evaluate(self):
        with patch.multiple(A, PROTOCOL_SHA=self.protocol_sha, RAW_SHA=A.digest(self.raw_path), INITIAL_SHA=A.digest(self.initial_path),
                PARENT_CONTRACT=self.parent_contract), patch.object(A, "load_scalar", return_value=C):
            return A.evaluate(self.packet, self.admission_path, self.admission_sha)


class WrapperControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.f = ManufacturedPacket(self.directory.name)

    def test_manufactured_two_arm_packet_complete_path(self):
        report = self.f.evaluate()
        self.assertTrue(report["verified_pilot_credit"], report["issues"])
        self.assertEqual(report["array_members_decoded_count"], 22)
        self.assertFalse(report["scientific_goals_complete"])
        self.assertEqual(report["details"]["registered_parent_controls"]["parent"], 11)
        self.assertEqual(report["details"]["slots"][1]["costs"]["map_disjoint"]["stability_rhs"], 112)

    def test_no_go_and_wrong_go_hash_load_no_companion_or_arrays(self):
        with patch.object(A, "load_scalar", side_effect=AssertionError("must not import")) as load:
            report = A.evaluate(self.f.packet, self.f.admission_path, "0" * 64)
        self.assertFalse(report["verified_pilot_credit"])
        self.assertEqual(report["decoded_array_members"], [])
        load.assert_not_called()

    def test_changed_companion_manifest_fails_before_dynamic_import(self):
        manifest = copy.deepcopy(self.f.manifest)
        manifest["files"]["scalar_core.py"]["sha256"] = "0" * 64
        with patch.object(A.importlib.util, "spec_from_file_location", side_effect=AssertionError("must not load")) as load:
            with self.assertRaises(ValueError):
                A.load_scalar(manifest)
        load.assert_not_called()

    def test_parent_self_report_nonzero_live_group_and_unclosed_log_rejected(self):
        for key, value in (("observation_scope", "packet_self_report"), ("actual_wait_returncode", 2),
                ("owned_process_group_alive_after_cleanup", True), ("closed_log", False), ("clock", "other_clock")):
            original = copy.deepcopy(self.f.terminal)
            self.f.terminal[key] = value
            self.f.refresh()
            report = self.f.evaluate()
            self.assertFalse(report["verified_pilot_credit"], key)
            self.assertEqual(report["decoded_array_members"], [], key)
            self.f.terminal = original

    def test_worker_admission_and_api_origin_checked_before_any_array(self):
        worker = self.f.packet / "slot_01/worker"
        result = A.bounded_json(worker / "result.json")
        result["entry_origin"] += .1
        self.f.js(worker / "result.json", result)
        self.f.refresh()
        report = self.f.evaluate()
        self.assertFalse(report["verified_pilot_credit"])
        self.assertEqual(report["decoded_array_members"], [])

    def test_actual_caller_late_or_overlapping_rejected_before_any_array(self):
        for key, value in (("external_complete_return_seconds", 8.01), ("absolute_started", 101.)):
            attempts = copy.deepcopy(self.f.attempts)
            attempts[1][key] = value
            self.f.js(self.f.packet / "attempts.json", attempts)
            self.f.refresh()
            report = self.f.evaluate()
            self.assertFalse(report["verified_pilot_credit"])
            self.assertEqual(report["decoded_array_members"], [])

    def test_changed_packet_byte_or_added_file_fails_before_array_decode(self):
        (self.f.packet / "unexpected.txt").write_text("unregistered")
        report = self.f.evaluate()
        self.assertFalse(report["verified_pilot_credit"])
        self.assertEqual(report["decoded_array_members"], [])

    def test_final_rehash_detects_mutation_after_last_numeric_member(self):
        original = C.audit_numeric_slot
        calls = []
        def mutate(*args):
            answer = original(*args)
            calls.append(True)
            if len(calls) == 2:
                (self.f.packet / "source_after.json").write_bytes(b"{}")
            return answer
        with patch.object(C, "audit_numeric_slot", side_effect=mutate):
            report = self.f.evaluate()
        self.assertFalse(report["verified_pilot_credit"])
        self.assertGreater(report["array_members_decoded_count"], 0)
        self.assertTrue(any("artifact" in row["message"] for row in report["issues"]), report["issues"])

    def test_frozen_controls_time_source_and_xml_denominator(self):
        receipt_path = self.f.root / "auditor_controls/receipt.json"
        receipt = A.bounded_json(receipt_path)
        row = receipt["groups"]["auditor"]
        path = receipt_path.parent / row["actual_wait"]["path"]
        wait = A.bounded_json(path)
        wait["wait_completed_perf_counter"] = 100.01
        self.f.js(path, wait)
        row["actual_wait"]["sha256"] = A.digest(path)
        self.f.js(receipt_path, receipt)
        self.f.admission["files"]["auditor_controls"] = self.f.bound(receipt_path)
        self.f.refresh()
        report = self.f.evaluate()
        self.assertFalse(report["verified_pilot_credit"])
        self.assertEqual(report["decoded_array_members"], [])

    def test_exact_admission_roles_and_external_log_binding(self):
        log = self.f.root / "external.log"
        log.write_text("changed externally closed log")
        self.f.admission["files"]["external_parent_log"] = self.f.bound(log)
        self.f.refresh()
        report = self.f.evaluate()
        self.assertFalse(report["verified_pilot_credit"])
        self.assertEqual(report["decoded_array_members"], [])

    def test_default_actual_isolated_module_inventory(self):
        script = "import runpy,sys;runpy.run_path(sys.argv[1],run_name='__main__');assert not any(k.split('.')[0] in ('numpy','scipy','wavecert') for k in sys.modules);assert '_admitted_independent_case0_scalar_core' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-I", "-Werror", "-c", script, str(HERE / "audit_pilot.py")], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["project_packet_decodes"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
