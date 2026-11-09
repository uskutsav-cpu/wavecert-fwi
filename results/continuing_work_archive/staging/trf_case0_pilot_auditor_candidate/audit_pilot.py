"""Held stdlib post-return case0 pilot auditor; default metadata only.

Audit GO is distinct from worker execution GO. Source/criteria/input/terminal
pins are admitted before dynamic scalar import or any packet-array decoding.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_NAMES = {"audit_pilot.py", "scalar_core.py", "test_scalar_core.py", "test_audit_pilot.py", "run_controls.py"}
CRITERIA_SHA = "a13f07f86b34d4a6f8f39da7f7b9ef288d97b31fe17ea3dd3634a29de5049e56"
PROTOCOL_SHA = "537cb3914c2e47e5d9afa327607bbfacd992ecec8f2020b8128a5005c9b59272"
RAW_SHA = "36d11aba94ae9093cac888e9b28a2d01edcef1aa3a54b12f8e10269b963db2fc"
INITIAL_SHA = "73d3e5b42ee3df6d40c0a9d8dc872c7603ad65d25ccbb459e0c0bbd40cdb4fc1"
PRODUCTION_SHA = "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"
PARENT_CONTRACT = {"serial_pilot.py": "7eaa78f8a4f3b1108378e89f4792a13089b0b9bace37555836f3144c72586382",
    "test_parent.py": "cfc3edafc47eb0eb35e35ffcc49e26201a66b5af97c3b79553a0cc559d7ef5cc",
    "PILOT_PLAN.json": "ebded996fc6d0fd01eccd4b2e9a6523753f3eab35f66af596c367aebe897211d"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def bounded_json(path, cap=16 * 1024 * 1024):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= cap, "bounded regular JSON")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    need(len(raw) <= cap, "actual JSON read cap")
    return strict_json(raw)


def bound_file(base, row, cap=16 * 1024 * 1024, *, within=False):
    need(type(row) is dict and re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))), "bound file SHA")
    original = Path(row["path"])
    if not original.is_absolute():
        original = Path(base) / original
    path = original.resolve()
    if within:
        need(not Path(row["path"]).is_absolute() and path.is_relative_to(Path(base).resolve()), "relative control path remains in its evidence root")
    need(original.is_file() and not original.is_symlink() and original.stat().st_size <= cap
        and digest(path) == row["sha256"], "bound file bytes/origin/cap")
    return path


def known_wait(value, *, parent=False):
    need(type(value.get("actual_pid")) is int and value["actual_pid"] > 0
        and type(value.get("observed_pgid")) is int and value["observed_pgid"] == value["actual_pid"]
        and type(value.get("actual_wait_returncode")) is int and value["actual_wait_returncode"] == 0
        and value.get("wait_observed") is True and value.get("terminal_state_known") is True
        and value.get("owned_process_group_alive_after_cleanup") is False
        and value.get("closed_log") is True and value.get("unknown_work") is False,
        "actual waited exit0/owned group gone/closed log/no unknown")
    need(not value.get("post_wait_IO_errors", []) and not value.get("deferred_parent_signals", [])
        and not value.get("cleanup_error") and not value.get("exception_type"), "no wait/cleanup/stop error credit")
    if parent:
        need(value.get("schema") == "trf_case0_external_parent_terminal_v1"
            and value.get("observation_scope") == "independent_launcher_actual_wait"
            and value.get("parent_exit_inferred_from_packet") is False, "independent external parent terminal only")


def source_manifest(manifest):
    need(manifest.get("schema") == "trf_case0_compound_auditor_source_v1"
        and set(manifest["files"]) == SOURCE_NAMES and type(manifest["registered_control_count"]) is int
        and manifest["registered_control_count"] > 0 and manifest.get("excludes_itself") is True,
        "complete exact executing auditor source manifest")
    need({p.name for p in HERE.glob("test_*.py")} <= SOURCE_NAMES, "all current test sources in compound manifest")
    for name, row in manifest["files"].items():
        path = HERE / name
        need(path.is_file() and not path.is_symlink() and type(row["bytes"]) is int
            and path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], "executing auditor source changed")


def parent_terminal(terminal, admission, files):
    known_wait(terminal, parent=True)
    need(terminal.get("clock") == "time.perf_counter_same_host"
        and terminal["observer_source_sha256"] == digest(files["external_observer_source"])
        and terminal["observer_contract"] == admission["observer_contract"]
        and terminal["observer_contract"]["wrapper_sha256"] == terminal["observer_source_sha256"]
        and re.fullmatch(r"[0-9a-f]{64}", terminal["observer_contract"]["launcher_helper_sha256"])
        and terminal["packet_inventory_sha256"] == admission["packet_inventory_sha256"]
        and terminal["parent_contract"] == admission["expected_sources"]["parent_contract"], "source-bound actual parent observation")
    times = [terminal[k] for k in ("external_parent_started_perf_counter", "wait_completed_perf_counter", "parent_complete_observed_perf_counter")]
    need(all(type(v) in (int, float) and math.isfinite(v) for v in times) and 0 <= times[0] <= times[1] <= times[2],
        "external same-host actual wait/complete chronology")
    need(terminal["closed_log_sha256"] == digest(files["external_parent_log"]), "actual externally closed parent log binding")


def verify_controls(receipt_path, runtime, expected_groups, expected_sources, before):
    receipt = bounded_json(receipt_path, 1024 * 1024)
    need(receipt.get("status") == "REGISTERED_PILOT_CONTROLS_ACTUAL_PASS"
        and receipt.get("runtime") == runtime and set(receipt.get("groups", {})) == set(expected_groups), "registered control groups/runtime")
    counts = {}
    for name, count in expected_groups.items():
        row = receipt["groups"][name]
        paths = {key: bound_file(Path(receipt_path).parent, row[key], 4 * 1024 * 1024, within=True)
            for key in ("log", "result", "source_identities", "actual_wait")}
        identities = bounded_json(paths["source_identities"], 1024 * 1024)
        need(all(identities.get(k) == v for k, v in expected_sources[name].items()), "control source prehash")
        wait = bounded_json(paths["actual_wait"])
        known_wait(wait)
        completed = wait.get("wait_completed_perf_counter")
        need(type(completed) in (int, float) and math.isfinite(completed) and completed <= before
            and wait.get("clock") == "time.perf_counter_same_host", "actual controls before first native slot")
        need(paths["log"].stat().st_size > 0, "retained raw control log")
        if name == "parent":
            result = bounded_json(paths["result"])
            totals = {k: result.get(k) for k in ("tests", "failures", "errors", "skipped")}
            ids = result.get("test_ids", [])
        else:
            root = ET.fromstring(paths["result"].read_bytes())
            suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
            totals = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
            ids = [(c.get("classname"), c.get("name")) for c in root.iter("testcase")]
            need(not any(list(root.iter(k)) for k in ("failure", "error", "skipped")), "no nested control failures/skips")
        need(totals == {"tests": count, "failures": 0, "errors": 0, "skipped": 0}
            and len(ids) == len(set(map(str, ids))) == count, "unique actual registered control counts")
        counts[name] = count
    return counts


def admit(admission_path, expected_sha):
    admission_path = Path(admission_path).resolve()
    need(re.fullmatch(r"[0-9a-f]{64}", expected_sha) and digest(admission_path) == expected_sha, "exact explicit audit GO SHA")
    admission = bounded_json(admission_path, 1024 * 1024)
    need(admission.get("schema") == "trf_case0_postreturn_audit_go_v1"
        and admission.get("authorization") == "GO_POSTRETURN_TRF_CASE0_PILOT_AUDIT"
        and admission.get("execution_authorized") is True and admission.get("native_execution_authorized") is False,
        "separate post-return audit GO, never native GO")
    files = {name: bound_file(admission_path.parent, row) for name, row in admission["files"].items()}
    required = {"source_manifest", "criteria", "protocol", "parent_plan", "raw_validation", "frozen_initial",
        "external_parent_terminal", "external_observer_source", "external_parent_log", "parent_controls", "auditor_controls"}
    need(set(files) == required, "complete explicit audit input/terminal/source roles")
    manifest = bounded_json(files["source_manifest"], 1024 * 1024)
    need(files["source_manifest"].resolve() == (HERE / "AUDITOR_SOURCE_MANIFEST.json").resolve(), "executing manifest path")
    source_manifest(manifest)
    need(digest(files["criteria"]) == CRITERIA_SHA and digest(files["protocol"]) == PROTOCOL_SHA
        and digest(files["raw_validation"]) == RAW_SHA and digest(files["frozen_initial"]) == INITIAL_SHA,
        "frozen prospective criteria/project/raw/initial pins")
    criteria, protocol, plan = (bounded_json(files[k]) for k in ("criteria", "protocol", "parent_plan"))
    need(criteria["numeric_rules"] == {"truth_and_metric_comparison_rtol": 2e-12, "truth_and_metric_comparison_atol": 2e-14,
        "optimizer_completion_deadline_seconds": 7.5, "actual_complete_process_cap_seconds": 8.,
        "squared_slowness_lower": 1 / 36, "squared_slowness_upper": 1., "model_parameters": 2401,
        "real_data_terms": 1920, "observation_blocks": 48}, "fixed numeric criteria")
    terminal = bounded_json(files["external_parent_terminal"])
    parent_terminal(terminal, admission, files)
    need(admission["expected_sources"]["parent_contract"] == PARENT_CONTRACT
        and PARENT_CONTRACT["PILOT_PLAN.json"] == digest(files["parent_plan"])
        and admission["expected_sources"]["worker_contract"] == plan["worker_contract"]
        and plan["criteria_sha256"] == CRITERIA_SHA and plan["production_source_sha256"] == PRODUCTION_SHA,
        "new parent/worker/source/criteria contract")
    need(plan["mode"] == "case0_two_return_integration_pilot" and plan["full48_authorized"] is False
        and len(plan["slots"]) == 2 and [s["run_index"] for s in plan["slots"]] == [0, 1]
        and all(s["case_index"] == s["repetition"] == 0 and s["status"] == "unattempted" for s in plan["slots"]), "fixed two-slot pilot only")
    need(plan["required_controls"] == {"component": 46, "API": 18, "worker": 10, "parent": 11}, "all85 frozen parent control groups")
    return admission, files, manifest, criteria, protocol, plan, terminal


def load_scalar(manifest):
    source_manifest(manifest)  # Recheck immediately before executing companion bytes.
    name = "_admitted_independent_case0_scalar_core"
    need(name not in sys.modules, "fresh admitted scalar import")
    spec = importlib.util.spec_from_file_location(name, HERE / "scalar_core.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inventory(packet, expected_sha, caps):
    path = packet / "artifact_inventory.json"
    need(digest(path) == expected_sha, "externally bound terminal inventory")
    value = bounded_json(path, 1024 * 1024)
    need(value["complete_inventory"] is True and value["known_children_terminal_before_hashing"] is True
        and value["inventory_excludes_itself"] is True, "stable closed child packet only")
    actual = set()
    for item in packet.rglob("*"):
        need(not item.is_symlink(), "no symlink in packet")
        if item.is_file() and item != path:
            actual.add(item.relative_to(packet).as_posix())
    need(set(value["files"]) == actual and len(actual) <= caps["inventory_file_count"], "exact inventory denominator")
    used = 0
    for name, row in value["files"].items():
        original = packet / name
        need(not Path(name).is_absolute() and original.resolve().is_relative_to(packet)
            and row["bounded_hash_completed"] is True and type(row["bytes"]) is int
            and 0 <= row["bytes"] <= caps["individual_file_bytes"] and original.stat().st_size == row["bytes"]
            and digest(original) == row["sha256"], "stable bounded artifact bytes")
        used += row["bytes"]
    need(used == value["hashed_bytes"] and used <= caps["packet_bytes"] - caps["failure_reserve_bytes"], "bounded complete packet sum")
    need(not any("failure" in n.lower() or "partial" in n.lower() or "UNSTABLE" in n for n in actual), "no failed/partial packet receives credit")
    return value


def worker_events(worker, result, payload, started, returned, core):
    path = worker / "events.jsonl"
    need(path.stat().st_size <= 2 * 1024 * 1024, "bounded worker event stream")
    events = [strict_json(line) for line in path.read_bytes().splitlines()]
    kinds = ["exact_GO_source_request_persisted_before_numerical_imports",
        "actual_native_numerical_origins_before_case_work", "raw_initial_truth_prepared_before_observations"]
    kinds += [kind for _ in range(8) for kind in ("observation_dispatch_attempt", "observation_dispatch_returned")]
    kinds += ["all8_data_prepared_before_policy", "genuine_cold_native_owner_constructed",
        "API_returned_selection_frozen", "post_return_quality_completed_without_reselection"]
    need([row["kind"] for row in events] == kinds, "complete chronological worker event lifecycle")
    previous = started
    for index, event in enumerate(events):
        now = event["observed_perf_counter"]
        need(type(event["sequence"]) is int and event["sequence"] == index
            and type(now) in (int, float) and math.isfinite(now) and previous <= now <= returned,
            "same-origin retained worker chronology")
        previous = now
    origins = result["evidence"]["context"]["runtime_origins"]
    suffixes = {"numpy": "/numpy/__init__.py", "scipy": "/scipy/__init__.py",
        "wavecert.physics.helmholtz": "/src/wavecert/physics/helmholtz.py",
        "wavecert.physics.exact_merit": "/src/wavecert/physics/exact_merit.py",
        "matched_driver": "/api/matched_driver.py", "public_map_builder": "/api/public_map_builder.py"}
    need(set(origins) == set(suffixes) and all(type(origins[k]) is str and origins[k].startswith("/")
        and origins[k].endswith(v) for k, v in suffixes.items())
        and events[1]["origins"] == origins and events[1]["manufactured_runtime"] is False,
        "loaded public native/API/numerical origin metadata")
    for batch in range(8):
        attempt, complete = events[3 + 2 * batch]["prefix"], events[4 + 2 * batch]["prefix"]
        need(attempt["observation_dispatches_attempted"] == batch + 1
            and attempt["observation_dispatches_completed"] == batch and attempt["observation_call_inflight"] is True
            and attempt["unknown_work"] is True and complete["observation_dispatches_attempted"] == batch + 1
            and complete["observation_dispatches_completed"] == batch + 1
            and complete["observation_call_inflight"] is False and complete["unknown_work"] is False,
            "observation attempt/return denominator")
        need(attempt["active_frequency_hz"] == complete["active_frequency_hz"] == [.25, .5, 1., 2., 2.5, 3., 3.5, 4.][batch], "all8 ordered observation event frequencies")
        need(core.physics_counts(complete["last_known_observation_counts"])
            == core.expected_counts(factorizations=batch + 1, forward_rhs=6 * (batch + 1)), "actual observation cumulative event prefix")
    binding = result["case_binding"]
    need(events[2]["initial_sha256"] == binding["initial49_m_sha256"]
        and events[2]["truth_sha256"] == binding["truth49_m_sha256"] and events[2]["old_truth_bitguard"] is False
        and events[19]["binding"] == binding and events[19]["counts"] == payload["preparation_costs_disjoint"], "prepared arrays/observations event bindings")
    need(events[20]["native_class"] == "ExactMeritOracle"
        and core.count_tree(events[20]["prefix"]) == {"physics": core.expected_counts(), "oracle": {}}, "one genuine cold source-bound policy owner")
    api = result["API_return_gate_only"]
    need(events[21]["selected"] == payload["selected"]["model_sha256"]
        and math.isclose(events[21]["observed_return"] - started, api["actual_complete_return_seconds"], rel_tol=1e-12, abs_tol=1e-9)
        and events[22]["metrics"] == result["post_return_quality"], "selection fixed before retained truth metrics")
    return {"retained_worker_events": len(events), "complete_worker_preparation_lifecycle_checked": True,
        "native_inner_queries_independently_replayed": False}


def control_source_contract(protocol, worker_contract, parent_contract):
    old = protocol["historical_API"]["source_sha256"]
    return {"component": {"cached_jacobian.py": old["cached_jacobian.py"], "public_map_builder.py": old["public_map_builder.py"],
            "component_controls.py": "e5dfc6db6dde088000064cd8d5c5eaae6668749746f4523f692349032999407e"},
        "API": {"matched_driver.py": old["matched_driver.py"], "normalized_exact.py": old["normalized_exact.py"],
            "test_driver.py": "2e105f64a62cfa505436cc08e8496f3142daee2f36b4b6038b79be4be07573e3",
            "protocol.json": "e967fdf4096ab820b3fa68b37ce711b99dbfd752b3fe5a88e100589a9a30a41e"},
        "worker": {"realcase_worker.py": worker_contract["realcase_worker.py"], "case_runtime.py": worker_contract["case_runtime.py"],
            "project_protocol.json": worker_contract["project_protocol.json"],
            "test_worker.py": "35ad9287e58d2b0f0e5e0eb12d3886093df8552b67db9bf279178fbdf345c4cf"},
        "parent": parent_contract}


def slot_preflight(packet, attempts, plan):
    previous_return = -1.
    for index, row in enumerate(attempts):
        need(row["slot"] == plan["slots"][index] and row["status"] == "attempted" and row["eligible"] is True
            and row["unknown_work"] is False and row["eligible_before_external_return_observation"] is True
            and not row["parent_signals_outside_slot"], "both actual fixed eligible slots before array decode")
        known_wait(row)
        wait_path = packet / f"slot_{index:02d}/actual_wait.json"
        known_wait(bounded_json(wait_path))
        need(digest(wait_path) == row["actual_wait_receipt_sha256"], "raw child actual-wait receipt before array decode")
        started, returned, age = row["absolute_started"], row["external_return_observed_perf_counter"], row["external_complete_return_seconds"]
        need(all(type(v) in (int, float) and math.isfinite(v) for v in (started, returned, age))
            and started >= 0 and (index == 0 or 0 <= previous_return <= started), "serial finite earlier origin")
        need(0 <= age <= 8 and math.isclose(returned - started, age, rel_tol=1e-12, abs_tol=1e-12)
            and row["external_wait_cleanup_logclose_seconds"] <= age, "fully serialized child-slot return8 before array decode")
        previous_return = returned


def worker_preflight(packet, attempts, before, terminal, protocol, root_go_sha):
    """All worker source/admission/caller clocks are checked before ANY array."""
    retained = []
    for index, row in enumerate(attempts):
        directory = packet / f"slot_{index:02d}"
        started, returned, age = row["absolute_started"], row["external_return_observed_perf_counter"], row["external_complete_return_seconds"]
        wait = bounded_json(directory / "actual_wait.json")
        need(all(wait[k] == row[k] for k in ("actual_pid", "observed_pgid", "actual_wait_returncode", "absolute_started")), "raw actual wait identity/origin")
        request = bounded_json(directory / "request.json")
        need(type(request["parent_pid"]) is int and request["parent_pid"] == terminal["actual_pid"]
            and request["absolute_started"] == started and request["optimizer_absolute_deadline"] == started + 7.5
            and request["complete_return_absolute_deadline"] == started + 8. and request["slot_index"] == index
            and request["arm"] == row["slot"]["arm"] and request["case_index"] == 0
            and request["clock"] == "time.perf_counter_same_host"
            and request["source_contract_sha256"] == hashlib.sha256(canonical(before["worker_contract"])).hexdigest(), "same actual parent/origin/slot/source request")
        spawned, launch, slot_return = (bounded_json(directory / name) for name in ("spawned.json", "launch_requested.json", "slot_return.json"))
        need(launch["launch_state"] == "NOT_ATTEMPTED" and launch["absolute_started"] == started
            and spawned["launch_state"] == "HANDLE_ASSIGNED" and spawned["actual_pid"] == row["actual_pid"]
            and spawned["observed_pgid"] == row["observed_pgid"] and slot_return["eligible_before_external_return_observation"] is True
            and all(slot_return[k] == row[k] for k in ("actual_pid", "absolute_started", "actual_wait_returncode", "actual_wait_receipt_sha256",
                "worker_result_sha256", "returned_payload_sha256", "case_binding")), "owned launch and serialized caller return")
        need(bounded_json(directory / "source_before.json") == bounded_json(directory / "source_after.json") == before, "per-slot source stability")
        worker = directory / "worker"
        result, payload = bounded_json(worker / "result.json"), bounded_json(worker / "returned_payload.json")
        need(digest(worker / "result.json") == row["worker_result_sha256"] and digest(worker / "returned_payload.json") == row["returned_payload_sha256"]
            and result["status"] == "WORKER_RETURNED_PENDING_EXTERNAL_WAIT_AND_INDEPENDENT_AUDIT"
            and result["slot"] == row["slot"] and result["source_files"] == before["worker_contract"]
            and result["method_policy_valid"] is True and result["complete_OS_process_eligibility_available"] is False
            and result["actual_external_wait_required"] is True and result["method_status"] == payload["status"]
            and payload["arm"] == row["slot"]["arm"] and payload["policy_valid"] is True
            and payload["unknown_work"] is False and not result["evidence"]["stop_signals"], "retained worker/API policy/source binding")
        need(bounded_json(worker / "project_plan.json") == protocol
            and bounded_json(worker / "source_contract_before.json") == before["worker_contract"], "worker pre-import protocol/source pins")
        accepted = bounded_json(worker / "admission.json")
        need(accepted == result["evidence"]["context"]["admission"] and accepted["slot"] == row["slot"]
            and accepted["absolute_started"] == result["entry_origin"] == started
            and accepted["source_files"] == before["worker_contract"]
            and accepted["mode"] == "case0_two_return_integration_pilot" and accepted["external_wait_and_audit_required"] is True
            and accepted["root_go_sha256"] == root_go_sha and accepted["request_sha256"] == digest(directory / "request.json")
            and accepted["source"] == {"production_source_sha256": PRODUCTION_SHA, "production_files": 161,
                "historical_API_files": accepted["source"]["historical_API_files"], "source_files": before["worker_contract"]}
            and type(accepted["source"]["historical_API_files"]) is int and accepted["source"]["historical_API_files"] > 0, "worker actual pre-import source/admission receipt")
        api_age, pre_age, worker_age = result["API_return_gate_only"]["actual_complete_return_seconds"], accepted["pre_admission_age"], result["worker_pre_result_age"]
        need(all(type(v) in (int, float) and math.isfinite(v) for v in (api_age, pre_age, worker_age, payload["pre_entry_seconds"]))
            and 0 <= pre_age <= payload["pre_entry_seconds"] <= api_age <= worker_age <= age <= 8
            and pre_age < 7.5 and payload["timing_origin"] == "supplied_pre_entry_absolute"
            and result["API_return_gate_only"]["eligible"] is True and result["API_return_gate_only"]["cap_seconds"] == 8., "earlier-origin worker/API/actual caller clocks")
        need(payload["admission_receipt"]["project_protocol_sha256"] == PROTOCOL_SHA
            and payload["admission_receipt"]["historical_API"]["actual_public_SciPy_TRF_function"] is True
            and payload["admission_receipt"]["historical_API"]["components"]
            == {k: protocol["historical_API"]["source_sha256"][k] for k in ("bounded_exact.py", "cached_jacobian.py", "public_map_builder.py")}, "new project admission above authentic historical API")
        retained.append((worker, result, payload))
    return retained


def inspect_packet(packet, admission, files, manifest, protocol, plan, terminal, core, decoded):
    """Details only; only evaluate() can issue global pilot verified credit."""
    packet = Path(packet).resolve()
    inv = inventory(packet, admission["packet_inventory_sha256"], plan["caps"])
    need(bounded_json(packet / "project_plan.json") == plan, "complete before-first-child plan")
    before, after = (bounded_json(packet / n) for n in ("source_before.json", "source_after.json"))
    need(before == after == admission["expected_sources"], "unchanged exact parent/worker/source/input contracts")
    go, observed = (bounded_json(packet / n) for n in ("supplied_root_go.json", "admission.json"))
    need(digest(packet / "supplied_root_go.json") == admission["pilot_root_go_sha256"]
        == observed["original_root_go_sha256"] == observed["packet_root_go_sha256"], "exact original pilot GO bytes")
    flags = ("execution_authorized", "independent_worker_source_review_clear", "external_supervisor_review_clear",
        "independent_auditor_review_clear", "runtime_review_clear", "mandatory_frozen_controls_passed")
    need(all(go.get(k) is True for k in flags) and go["authorization"] == "GO_TRF_CASE0_TWO_RETURN_INTEGRATION_PILOT"
        and go["allowed_slot_indices"] == [0, 1] and go["full48_authorized"] is False
        and go["source_files"] == before["worker_contract"] and go["parent_contract"] == before["parent_contract"]
        and go["review_bindings"] == before["role_bindings"]
        and go["auditor_source_sha256"] == digest(files["source_manifest"])
        and go["audit_criteria_sha256"] == CRITERIA_SHA, "complete frozen pilot GO/review contracts")
    need(observed["observed_sources"] == before and observed["original_admission_sha256"] == go["admission_sha256"]
        and observed["registered_controls"] == {"groups": plan["required_controls"], "all_actual_counts_verified": True}, "source/runtime/control admission binding")
    runtime = observed["observed_runtime"]
    need(runtime["packages"] == {k: plan["control_runtime"][k] for k in ("numpy", "scipy", "pytest")}
        and runtime["threads"] == plan["threads"] and runtime["numerical_imports_in_parent"] == 0
        and runtime["python"].startswith(plan["runtime"]["python"]) and "Linux" in runtime["platform"], "actual registered target runtime/thread metadata")
    attempts = bounded_json(packet / "attempts.json")
    need(type(attempts) is list and len(attempts) == 2, "exact two attempted rows, no replacement")
    slot_preflight(packet, attempts, plan)
    pilot = bounded_json(packet / "pilot_result.json")
    need(pilot["attempts"] == attempts and pilot["unknown_work"] is False and pilot["failure"] is None
        and pilot["status"] == "PILOT_TWO_RETURNS_PENDING_INDEPENDENT_AUDIT_AND_PARENT_EXTERNAL_WAIT"
        and pilot["parent_terminal_externally_observed"] is False and not pilot["deferred_parent_signals"], "pilot result is not terminal proof")
    first_start = attempts[0]["absolute_started"]
    parent_control_sources = control_source_contract(protocol, before["worker_contract"], before["parent_contract"])
    parent_controls = verify_controls(files["parent_controls"], plan["control_runtime"], plan["required_controls"], parent_control_sources, first_start)
    auditor_controls = verify_controls(files["auditor_controls"], plan["control_runtime"], {"auditor": manifest["registered_control_count"]},
        {"auditor": {k: row["sha256"] for k, row in manifest["files"].items()}}, first_start)
    retained = worker_preflight(packet, attempts, before, terminal, protocol, admission["pilot_root_go_sha256"])
    read = lambda path, schema, h, label: core.read_npz(path, schema, h, on_member=lambda k: decoded.append(label + ":" + k))
    raw = read(files["raw_validation"], core.RAW_SCHEMA, admission["files"]["raw_validation"]["sha256"], "raw")
    frozen = read(files["frozen_initial"], core.INITIAL_SCHEMA, admission["files"]["frozen_initial"]["sha256"], "initial")
    results = []
    for index, row in enumerate(attempts):
        need(row["slot"] == plan["slots"][index] and row["status"] == "attempted" and row["eligible"] is True
            and row["unknown_work"] is False and row["eligible_before_external_return_observation"] is True
            and not row["parent_signals_outside_slot"], "eligible fixed slot/caller boundary")
        known_wait(row)
        started, returned, age = row["absolute_started"], row["external_return_observed_perf_counter"], row["external_complete_return_seconds"]
        need(type(started) in (int, float) and type(returned) in (int, float) and math.isfinite(started) and math.isfinite(returned)
            and 0 <= age <= 8 and math.isclose(returned - started, age, rel_tol=1e-12, abs_tol=1e-12)
            and row["external_wait_cleanup_logclose_seconds"] <= age, "actual fully serialized complete parent slot return<=8")
        directory = packet / f"slot_{index:02d}"
        wait = bounded_json(directory / "actual_wait.json")
        known_wait(wait)
        need(digest(directory / "actual_wait.json") == row["actual_wait_receipt_sha256"]
            and all(wait[k] == row[k] for k in ("actual_pid", "observed_pgid", "actual_wait_returncode", "absolute_started")), "raw actual wait binding")
        request = bounded_json(directory / "request.json")
        need(request["parent_pid"] == terminal["actual_pid"] and request["absolute_started"] == started
            and request["optimizer_absolute_deadline"] == started + 7.5 and request["complete_return_absolute_deadline"] == started + 8.
            and request["slot_index"] == index and request["arm"] == row["slot"]["arm"]
            and request["clock"] == "time.perf_counter_same_host"
            and request["source_contract_sha256"] == hashlib.sha256(canonical(before["worker_contract"])).hexdigest(), "same actual parent/origin/slot/source request")
        spawned = bounded_json(directory / "spawned.json")
        launch = bounded_json(directory / "launch_requested.json")
        slot_return = bounded_json(directory / "slot_return.json")
        need(launch["launch_state"] == "NOT_ATTEMPTED" and launch["absolute_started"] == started
            and spawned["launch_state"] == "HANDLE_ASSIGNED" and spawned["actual_pid"] == row["actual_pid"]
            and spawned["observed_pgid"] == row["observed_pgid"]
            and slot_return["eligible_before_external_return_observation"] is True
            and all(slot_return[k] == row[k] for k in ("actual_pid", "absolute_started", "actual_wait_returncode",
                "actual_wait_receipt_sha256", "worker_result_sha256", "returned_payload_sha256", "case_binding")),
            "launch/owned handoff/fully serialized slot return binding")
        need(bounded_json(directory / "source_before.json") == bounded_json(directory / "source_after.json") == before, "per-slot source stability")
        worker = directory / "worker"
        result, payload = bounded_json(worker / "result.json"), bounded_json(worker / "returned_payload.json")
        need(digest(worker / "result.json") == row["worker_result_sha256"] and digest(worker / "returned_payload.json") == row["returned_payload_sha256"]
            and result["status"] == "WORKER_RETURNED_PENDING_EXTERNAL_WAIT_AND_INDEPENDENT_AUDIT"
            and result["slot"] == row["slot"] and result["source_files"] == before["worker_contract"]
            and result["method_policy_valid"] is True and result["API_return_gate_only"]["eligible"] is True
            and result["complete_OS_process_eligibility_available"] is False and result["actual_external_wait_required"] is True
            and result["method_status"] == payload["status"] and payload["arm"] == row["slot"]["arm"], "genuine retained worker/API return binding")
        need(payload["admission_receipt"]["project_protocol_sha256"] == PROTOCOL_SHA
            and payload["admission_receipt"]["historical_API"]["actual_public_SciPy_TRF_function"] is True
            and payload["admission_receipt"]["historical_API"]["components"]
            == {k: protocol["historical_API"]["source_sha256"][k] for k in ("bounded_exact.py", "cached_jacobian.py", "public_map_builder.py")},
            "new project admission above authentic historical API")
        need((worker, result, payload) == retained[index], "worker metadata unchanged since pre-decode admission")
        arrays = {}
        for filename, schema in (("prepared_models.npz", core.PREPARED_SCHEMA), ("observations.npz", core.OBSERVATION_SCHEMA), ("returned_models.npz", core.RETURNED_SCHEMA)):
            name = f"slot_{index:02d}/worker/{filename}"
            arrays[filename] = read(worker / filename, schema, inv["files"][name]["sha256"], f"slot{index}/" + filename)
        numeric = core.audit_numeric_slot(raw, frozen, arrays["prepared_models.npz"], arrays["observations.npz"], arrays["returned_models.npz"], payload, result, protocol)
        need(numeric["case_binding"] == row["case_binding"], "parent numeric binding equality")
        cost = core.audit_costs(payload, result["evidence"]["context"], protocol["frequencies_hz"], protocol["receiver_indices"], protocol["source_indices"])
        need(not result["evidence"]["stop_signals"], "no worker stop receives pilot credit")
        chronology = worker_events(worker, result, payload, started, returned, core)
        results.append({"slot_index": index, "external_complete_return_seconds": age, "numeric": numeric, "costs": cost, "chronology": chronology})
    need(results[0]["numeric"]["case_binding"] == results[1]["numeric"]["case_binding"], "both arms identical new truth/initial/all8obs/survey bytes")
    need(attempts[0]["external_return_observed_perf_counter"] <= attempts[1]["absolute_started"], "serial fixed slot chronology")
    cold_start, cold_return, cold_age = (terminal[k] for k in ("external_parent_started_perf_counter", "parent_complete_observed_perf_counter", "external_parent_complete_return_seconds"))
    need(all(type(v) in (int, float) and math.isfinite(v) for v in (cold_start, cold_return, cold_age))
        and 0 <= cold_start <= attempts[0]["absolute_started"] and attempts[1]["external_return_observed_perf_counter"] <= cold_return
        and math.isclose(cold_return - cold_start, cold_age, rel_tol=1e-12, abs_tol=1e-9)
        and cold_age >= sum(r["external_complete_return_seconds"] for r in attempts), "separate actual cold parent/shared-tail observation")
    need(inventory(packet, admission["packet_inventory_sha256"], plan["caps"]) == inv, "whole packet rehashed after scalar audit")
    return {"slots": results, "registered_parent_controls": parent_controls, "registered_auditor_controls": auditor_controls,
        "parent_terminal_scope": "Independent externally waited parent0 plus both actual child waits/group cleanup; never inferred from status.",
        "shared_overhead": {**{k: pilot.get(k) for k in ("serial_parent_wall_before_report_seconds", "sum_observed_slot_complete_return_seconds", "shared_serial_preplan_between_and_tail_seconds")},
            "external_parent_cold_start_through_terminal_seconds": cold_age,
            "scope": "Separately paid observed shared startup/between/final tail; not silently assigned zero or pooled with old timings."}}


def evaluate(packet_root, admission_path, expected_admission_sha256):
    decoded = []
    report = {"status": "PILOT_TECHNICAL_INTEGRATION_FAIL", "verified_pilot_credit": False,
        "issues": [], "decoded_array_members": decoded, "scientific_goals_complete": False, "student_final_design": False,
        "full48_study_credit": False, "physical_acceptance": None, "novelty_or_recovery_success_claim": False}
    try:
        admitted = admit(admission_path, expected_admission_sha256)
        admission, files, manifest, criteria, protocol, plan, terminal = admitted
        core = load_scalar(manifest)
        report["details"] = inspect_packet(Path(packet_root), admission, files, manifest, protocol, plan, terminal, core, decoded)
        need(digest(admission_path) == expected_admission_sha256, "audit GO unchanged after audit")
        need(all(digest(files[k]) == row["sha256"] for k, row in admission["files"].items()), "all external input/terminal/control sources unchanged after audit")
        source_manifest(manifest)
        report.update(status="PILOT_TECHNICAL_INTEGRATION_PASS", verified_pilot_credit=True,
            criteria_sha256=digest(files["criteria"]), auditor_source_manifest_sha256=digest(files["source_manifest"]))
    except Exception as exc:  # noqa: BLE001 -- fail closed, preserve every decoded-member prefix
        report["issues"].append({"exception_type": type(exc).__name__, "message": str(exc)})
    report["array_members_decoded_count"] = len(decoded)
    report["limits"] = ["No independent numerical Helmholtz objective or observation reproduction; objective scalar/receipt/source consistency only.",
        "Independent scalar truth/velocity-model errors and exact retained input/returned-array bindings only.",
        "Bounded engine retained head/tail/commitments; omitted middle and LSMR internal algebra are source-bound, not independently replayed.",
        "Two reused case0 returns do not establish full48 recovery, independent geology, physical acceptance, novelty, human reproduction or student ownership."]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--packet-root", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--admission-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "HELD_SOURCE_ONLY_CASE0_POSTRETURN_AUDITOR", "execution_authorized": False,
            "numerical_or_producer_imports": 0, "project_packet_decodes": 0, "PDE_calls": 0, "scientific_goals_complete": False}))
        return 0
    if any(getattr(args, k) is None for k in ("packet_root", "admission", "admission_sha256", "output")):
        parser.error("explicit packet/admission/hash/new output required")
    result = evaluate(args.packet_root, args.admission, args.admission_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return 0 if result["verified_pilot_credit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
