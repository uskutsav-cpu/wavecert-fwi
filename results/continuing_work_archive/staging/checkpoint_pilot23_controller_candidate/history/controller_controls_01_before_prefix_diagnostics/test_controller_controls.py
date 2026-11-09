"""Stdlib manufactured admission and real dummy-process lifecycle controls."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("manufactured_checkpoint23_controller", ROOT/"control_pilot23.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
C = D.C


def manufactured_probe(_group):
    return {"available": True, "members": [], "group_RSS_bytes": 0,
        "live_member_CPU_seconds": 0, "scope": "MANUFACTURED_RESOURCE_STUB_NO_RSS_CLAIM"}


def policy():
    value = C.read_json(ROOT/"controller_policy.json")
    value["poll_seconds"] = 0.01
    return value


def dummy(code):
    return [sys.executable, "-B", "-I", "-u", "-c", code]


def run_dummy(code, root, selected_policy=None):
    slot = root/"dummy"
    slot.mkdir()
    result = D.run_one(dummy(code), slot, root, policy() if selected_policy is None else selected_policy,
        D.StopState(), "manufactured_dummy", slot/"worker", resource_probe=manufactured_probe,
        evidence_scope="MANUFACTURED_DUMMY_ONLY_NO_PDE_OR_ENVIRONMENT_AUTHORITY")
    return slot, result


def helper(directory):
    stop = D.StopState()
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, stop.handler)
    result = D.run_one(dummy("import time; print('manufactured-ready',flush=True); time.sleep(30)"),
        directory, directory, policy(), stop, "manufactured_signal", directory/"worker",
        resource_probe=manufactured_probe, evidence_scope="MANUFACTURED_SIGNAL_CONTROL_ONLY")
    C.record(directory/"helper_terminal.json", {"result": result, "manufactured_only": True})
    return 0


class ControllerControls(unittest.TestCase):
    def test_default_isolated_CLI_blocks_numerical_imports_and_creates_nothing(self):
        guard = '''import sys,runpy
def block(event,args):
    if event == "import" and args[0].split(".")[0] in ("numpy","devito","pyrevolve"):
        raise RuntimeError("numerical import forbidden in manufactured control")
sys.addaudithook(block)
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name="__main__")
'''
        with tempfile.TemporaryDirectory(prefix="checkpoint23-default-") as temporary:
            child = subprocess.run([sys.executable, "-B", "-I", "-c", guard, str(ROOT/"control_pilot23.py")],
                cwd=temporary, capture_output=True, text=True, check=False)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertEqual(json.loads(child.stdout)["status"], "HELD_CHECKPOINT23_NO_CHILD_NO_NUMERICAL_IMPORT")
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_complete_plan_has_exact23_independent_slot_denominator_and_order(self):
        protocol = C.verify_worker()
        plan = C.verify_plan(protocol)
        rows = plan["slots"]
        self.assertEqual(len(rows), 23)
        self.assertEqual([row["slot_id"] for row in rows[:3]], ["saved", "checkpoint2", "checkpoint4"])
        self.assertEqual([row["checkpoints"] for row in rows[:3]], [None, 2, 4])
        self.assertEqual(rows[-1]["slot_id"], "fd_left_edge_4_1")
        self.assertEqual(len({row["slot_id"] for row in rows}), 23)
        self.assertEqual({row["direction"] for row in rows[3:]}, {"interior", "left_edge"})
        self.assertTrue(all(row["state"] == "unattempted" and row["actual_returncode"] is None for row in rows))

    def test_reordered_or_inferred_K_plan_is_rejected(self):
        protocol = C.verify_worker()
        plan = C.verify_plan(protocol)
        with tempfile.TemporaryDirectory(prefix="checkpoint23-plan-") as temporary:
            path = Path(temporary)
            changed = deepcopy(plan)
            changed["slots"][1]["checkpoints"] = None
            C.record(path/"FROZEN_PLAN23.json", changed)
            with patch.object(C, "ROOT", path), self.assertRaises(ValueError):
                C.verify_plan(protocol)

    def test_false_registration_rejects_before_environment_or_Popen(self):
        with tempfile.TemporaryDirectory(prefix="checkpoint23-reject-") as temporary:
            path = Path(temporary)
            files = {}
            for name, value in (("registration", {"execution_authorized": False}),
                                ("go", {"manufactured_not_a_GO": True}),
                                ("environment", {"manufactured_not_an_environment": True})):
                file = path/(name+".json")
                C.record(file, value)
                files[name] = file
            args = argparse.Namespace(registration=files["registration"], registration_sha256=C.sha(files["registration"]),
                root_go=files["go"], root_go_sha256=C.sha(files["go"]), environment=files["environment"],
                environment_sha256=C.sha(files["environment"]), output=path/"fresh")
            with patch.object(D.subprocess, "Popen") as popen, patch.object(C, "verify_environment") as env:
                with self.assertRaises(ValueError):
                    D.admit(args)
                popen.assert_not_called()
                env.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_actual_dummy_zero_has_real_PID_group_wait_and_complete_wall(self):
        with tempfile.TemporaryDirectory(prefix="checkpoint23-zero-") as temporary:
            slot, result = run_dummy("import os,json,time; print(json.dumps({'PID':os.getpid(),'group':os.getpgrp()}),flush=True); time.sleep(0.1)", Path(temporary))
            printed = json.loads((slot/"worker_console.log").read_text())
            self.assertEqual(result["actual_returncode"], 0)
            self.assertTrue(result["observed_terminal"])
            self.assertEqual(printed["PID"], result["PID"])
            self.assertEqual(printed["group"], result["PID"])
            self.assertGreaterEqual(result["external_wall_seconds"], 0.1)
            self.assertFalse(result["unknown_work"])
            self.assertIn("MANUFACTURED", result["evidence_scope"])

    def test_actual_dummy_nonzero_is_preserved_without_retry(self):
        with tempfile.TemporaryDirectory(prefix="checkpoint23-nonzero-") as temporary:
            slot, result = run_dummy("import sys; print('one-attempt'); sys.exit(7)", Path(temporary))
            self.assertEqual(result["actual_returncode"], 7)
            self.assertTrue(result["observed_terminal"])
            self.assertEqual(result["reason"], "actual_nonzero_or_signal_exit")
            self.assertEqual((slot/"worker_console.log").read_text().count("one-attempt"), 1)

    def test_started_record_write_failure_kills_and_waits_known_dummy(self):
        original = C.record
        def fail_started(path, value):
            if path.name == "started.json":
                raise OSError("manufactured started-receipt write failure")
            return original(path, value)
        with tempfile.TemporaryDirectory(prefix="checkpoint23-write-") as temporary:
            with patch.object(C, "record", side_effect=fail_started):
                slot, result = run_dummy("import time; time.sleep(30)", Path(temporary))
            self.assertTrue(result["observed_terminal"])
            self.assertEqual(result["actual_returncode"], -signal.SIGKILL)
            self.assertIn("manufactured started-receipt write failure", result["reason"])
            self.assertTrue((slot/"spawn_requested.json").exists())
            self.assertTrue((slot/"exit.json").exists())
            with self.assertRaises(ProcessLookupError):
                os.kill(result["PID"], 0)

    def test_resource_probe_failure_still_cleans_up_known_child(self):
        with tempfile.TemporaryDirectory(prefix="checkpoint23-probe-") as temporary:
            path = Path(temporary)
            slot = path/"dummy"
            slot.mkdir()
            def failure(_pid):
                raise OSError("manufactured proc read failure")
            result = D.run_one(dummy("import time; time.sleep(30)"), slot, path, policy(), D.StopState(),
                "manufactured_probe_error", slot/"worker", resource_probe=failure,
                evidence_scope="MANUFACTURED_PROBE_FAILURE_ONLY")
            self.assertEqual(result["actual_returncode"], -signal.SIGKILL)
            self.assertTrue(result["observed_terminal"])
            self.assertTrue(result["unknown_work"])
            with self.assertRaises(ProcessLookupError):
                os.kill(result["PID"], 0)

    def test_output_cap_stops_actual_dummy_and_preserves_overshoot(self):
        with tempfile.TemporaryDirectory(prefix="checkpoint23-cap-") as temporary:
            selected = policy()
            selected["per_console_cap_bytes"] = 100
            slot, result = run_dummy("import time; print('x'*10000,flush=True); time.sleep(30)", Path(temporary), selected)
            self.assertEqual(result["actual_returncode"], -signal.SIGKILL)
            self.assertTrue(result["observed_terminal"])
            self.assertGreater((slot/"worker_console.log").stat().st_size, 100)
            self.assertIn("cap exceeded", result["reason"])

    def test_actual_signal_forwards_and_reaps_known_dummy(self):
        with tempfile.TemporaryDirectory(prefix="checkpoint23-signal-") as temporary:
            directory = Path(temporary)
            helper_child = subprocess.Popen([sys.executable, "-B", "-I", str(Path(__file__).resolve()),
                "--helper", str(directory)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, start_new_session=True)
            try:
                deadline = time.monotonic()+5
                while not (directory/"started.json").exists():
                    if helper_child.poll() is not None or time.monotonic() > deadline:
                        self.fail("manufactured helper did not record its child start")
                    time.sleep(0.01)
                os.kill(helper_child.pid, signal.SIGTERM)
                stdout, stderr = helper_child.communicate(timeout=5)
                self.assertEqual(helper_child.returncode, 0, stdout+stderr)
                result = C.read_json(directory/"helper_terminal.json")["result"]
                self.assertTrue(result["observed_terminal"])
                self.assertIn(result["actual_returncode"], (-signal.SIGTERM, -signal.SIGKILL))
                self.assertTrue(result["observed_signals"])
                with self.assertRaises(ProcessLookupError):
                    os.kill(result["PID"], 0)
            finally:
                if helper_child.poll() is None:
                    os.killpg(helper_child.pid, signal.SIGKILL)
                    helper_child.wait()

    def test_plan_is_saved_before_dispatch_and_failure_leaves22_unattempted(self):
        with tempfile.TemporaryDirectory(prefix="checkpoint23-prefix-") as temporary:
            path = Path(temporary).resolve()
            inputs = []
            for name in ("registration", "go", "environment"):
                file = path/(name+".json")
                C.record(file, {"manufactured_fixture_not_native_authority": name})
                inputs.append(file)
            args = argparse.Namespace(registration=inputs[0], registration_sha256=C.sha(inputs[0]), root_go=inputs[1],
                root_go_sha256=C.sha(inputs[1]), environment=inputs[2], environment_sha256=C.sha(inputs[2]), output=path/"fresh")
            protocol = C.verify_worker()
            plan = C.verify_plan(protocol)
            calls = []
            def stop_before_real_Popen(command, slot_dir, output, selected_policy, stop, slot_id, worker_output):
                actual_plan = C.read_json(output/"plan.json")
                self.assertEqual(len(actual_plan["slots"]), 23)
                self.assertTrue(actual_plan["all23_rows_written_before_any_Popen"])
                calls.append(slot_id)
                raise D.SlotError("manufactured boundary stop; no Popen", {"PID": None, "unknown_work": False,
                    "actual_returncode": None, "observed_terminal": False})
            with patch.object(D, "admit", return_value=({}, protocol, plan, {},
                    {"worker_go": {"manufactured_not_a_native_GO": True}}, {"modules": {}}, {})), \
                    patch.object(D, "child_command", return_value=["UNUSED_MANUFACTURED_NO_EXEC"]), \
                    patch.object(C, "verify_bundle", return_value={}), \
                    patch.object(D, "resources_before_spawn", return_value={"manufactured": True}), \
                    patch.object(D, "run_one", side_effect=stop_before_real_Popen):
                self.assertEqual(D.execute(args), 2)
            terminal = C.read_json(args.output/"terminal_children.json")
            self.assertEqual(calls, ["saved"])
            self.assertEqual(len(terminal["slots"]), 23)
            self.assertEqual(sum(row["state"] == "unattempted" for row in terminal["slots"]), 22)
            self.assertEqual(terminal["status"], "INCOMPLETE_OR_FAILED_CHILD_PREFIX")
            self.assertFalse((args.output/"comparison.json").exists())

    def test_native_event_prefix_keeps_unreturned_work_unknown_and_replay_distinct(self):
        with tempfile.TemporaryDirectory(prefix="checkpoint23-events-") as temporary:
            path = Path(temporary)
            rows = [{"kind": "native_attempt", "role": "forward", "phase": "initial_forward", "native_steps": 7},
                    {"kind": "native_returned", "role": "forward", "phase": "initial_forward", "native_steps": 7},
                    {"kind": "native_attempt", "role": "forward", "phase": "reverse_with_paid_replay", "native_steps": 3}]
            (path/"events.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
            observed = D.native_prefix(path)
            self.assertTrue(observed["unknown_partial_native_steps"])
            self.assertEqual(observed["counters"]["native_forward_attempted_steps"], 10)
            self.assertEqual(observed["counters"]["native_forward_returned_steps"], 7)
            self.assertEqual(observed["counters"]["replay_forward_attempted_steps"], 3)
            self.assertEqual(observed["counters"]["replay_forward_returned_steps"], 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.helper:
        return helper(args.helper.resolve())
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ControllerControls))
    receipt = {"status": "MANUFACTURED_DUMMY_LIFECYCLE_PASS" if result.wasSuccessful() else "FAIL",
        "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "source_sha256": {name: C.sha(ROOT/name) for name in
            ("control_pilot23.py", "pilot23_common.py", "worker_entry.py", "controller_policy.json", "FROZEN_PLAN23.json", "test_controller_controls.py")},
        "native_workers_launched": 0, "numerical_imports": 0, "project_arrays_decoded": 0,
        "PDE_calls": 0, "dummy_processes_only": True,
        "manufactured_resource_probes_are_stubs_not_RSS_claims": True,
        "bounded_waits_used_only_in_test_harness_not_native_controller": True}
    if args.receipt:
        C.record(args.receipt, receipt)
    print(json.dumps(receipt))
    return 0 if result.wasSuccessful() else 2


if __name__ == "__main__":
    raise SystemExit(main())
