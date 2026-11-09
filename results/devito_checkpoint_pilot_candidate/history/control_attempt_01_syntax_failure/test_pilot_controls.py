"""Manufactured checks of accounting/admission/failure behavior, never physics."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("held_checkpoint_pilot_control", ROOT/"run_checkpoint_pilot.py")
DRIVER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DRIVER)
A = DRIVER._accounting


class FakeKernel:
    def __init__(self, offset=1, action=None):
        self.start_offset, self.action = offset, action
        self.calls = []

    def _prepare_args(self, start, end):
        return {"time_m": start+self.start_offset, "time_M": end-1+self.start_offset}

    def apply(self, t_start, t_end):
        self.calls.append((t_start, t_end))
        if self.action:
            self.action()


class FakeStorage:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def save(self, key, pointers):
        self.calls.append(("save", key, [p.nbytes for p in pointers]))
        if self.fail:
            raise RuntimeError("manufactured copy failure")

    def load(self, key, pointers):
        self.calls.append(("load", key, [p.nbytes for p in pointers]))


class Pointer:
    def __init__(self, nbytes):
        self.nbytes = nbytes


class PilotControls(unittest.TestCase):
    def test_default_isolated_CLI_blocks_numerical_imports_and_creates_nothing(self):
        wrapper = (
            "import sys,runpy; "
            "exec('def block(event,args):\\n if event == \\\"import\\\" and args[0].split(\\\".\\\")[0] in '
            ' (\\\"numpy\\\",\\\"devito\\\",\\\"pyrevolve\\\"):\\n  raise RuntimeError(\\\"numerical import blocked\\\")'); "
            "sys.addaudithook(block); sys.argv=[sys.argv[1]]; "
            "runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        with tempfile.TemporaryDirectory(prefix="checkpoint-default-") as temporary:
            result = subprocess.run([sys.executable, "-B", "-I", "-c", wrapper,
                str(ROOT/"run_checkpoint_pilot.py")], cwd=temporary, text=True,
                capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "HELD_METADATA_ONLY_NO_IMPORT_NO_PDE")
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_invalid_GO_cannot_authorize_or_create_output(self):
        with tempfile.TemporaryDirectory(prefix="checkpoint-GO-") as temporary:
            root = Path(temporary)
            go = root/"manufactured_rejection.json"
            go.write_text(json.dumps({"execution_authorized": False, "source_files": DRIVER.contract()}))
            with self.assertRaises(ValueError):
                DRIVER.admit(go)
            self.assertEqual([p.name for p in root.iterdir()], [go.name])

    def test_constructor_and_invalid_range_perform_no_native_call(self):
        native, ledger = FakeKernel(), A.Ledger()
        observed = A.AuditedKernel(native, "forward", 7, ledger, {})
        self.assertEqual(native.calls, [])
        self.assertEqual(ledger.events, [])
        for begin, end in [(0, 0), (-1, 3), (2, 8), (False, 1)]:
            with self.assertRaises(ValueError):
                observed.apply(begin, end)
        self.assertEqual(native.calls, [])

    def test_real_wrapper_offset_and_half_open_steps_are_recorded(self):
        native, ledger = FakeKernel(offset=3), A.Ledger()
        observed = A.AuditedKernel(native, "forward", 9, ledger, {})
        observed.apply(2, 7)
        event = ledger.events[0]
        self.assertEqual((event["native_time_m"], event["native_time_M"], event["native_steps"]), (5, 9, 5))
        self.assertEqual(native.calls, [(2, 7)])
        self.assertEqual(observed.counts()["returned_steps"], 5)

    def test_replay_forward_work_is_paid_in_actual_sequence(self):
        ledger = A.Ledger()
        observed = A.AuditedKernel(FakeKernel(), "forward", 6, ledger, {})
        ledger.phase = "initial_forward"
        observed.apply(0, 6)
        initial = observed.counts()["returned_steps"]
        ledger.phase = "reverse_with_paid_replay"
        observed.apply(0, 3)
        observed.apply(4, 5)
        self.assertEqual(observed.counts()["returned_steps"], 10)
        self.assertEqual(observed.counts()["returned_steps"]-initial, 4)
        self.assertEqual([x["phase"] for x in ledger.events if x["kind"] == "native_attempt"],
                         ["initial_forward", "reverse_with_paid_replay", "reverse_with_paid_replay"])

    def test_native_exception_preserves_attempt_prefix_without_return_claim(self):
        def failure():
            raise RuntimeError("manufactured native failure")
        ledger = A.Ledger()
        observed = A.AuditedKernel(FakeKernel(action=failure), "reverse", 6, ledger, {})
        with self.assertRaises(RuntimeError):
            observed.apply(1, 2)
        self.assertEqual([x["kind"] for x in ledger.events], ["native_attempt", "native_raised"])
        self.assertEqual(observed.counts(), {"attempted_calls": 1, "returned_calls": 0,
                         "attempted_steps": 1, "returned_steps": 0})

    def test_replay_source_mutation_is_rejected_after_paid_actual_call(self):
        state = {"source": "early-and-late-wavelet"}
        def change():
            state["source"] = "wrong-restarted-phase"
        native, ledger = FakeKernel(action=change), A.Ledger()
        observed = A.AuditedKernel(native, "forward", 6, ledger, {"source": lambda: state["source"]})
        with self.assertRaisesRegex(ValueError, "during"):
            observed.apply(0, 1)
        self.assertEqual(native.calls, [(0, 1)])
        self.assertEqual(observed.counts()["returned_calls"], 1)
        self.assertEqual(ledger.events[-1]["kind"], "immutable_input_changed")

    def test_changed_reverse_dual_is_rejected_before_any_native_call(self):
        state = {"dual": "fixed"}
        native = FakeKernel()
        observed = A.AuditedKernel(native, "reverse", 6, A.Ledger(), {"dual": lambda: state["dual"]})
        state["dual"] = "overwritten-by-replay-scratch"
        with self.assertRaisesRegex(ValueError, "before"):
            observed.apply(0, 1)
        self.assertEqual(native.calls, [])

    def test_checkpoint_budget_uses_elements_times_itemsize_and_explicit_K(self):
        result = A.checkpoint_budget(100, 8, 4, 3200)
        self.assertEqual((result["checkpoint_bytes"], result["capacity_bytes"]), (800, 3200))
        for k in (None, True, 0, -2, 2.0):
            with self.assertRaises(ValueError):
                A.checkpoint_budget(100, 8, k, 3200)
        with self.assertRaises(ValueError):
            A.checkpoint_budget(100, 8, 5, 3200)

    def test_storage_counts_supplied_two_plane_copy_bytes_and_actual_calls(self):
        native, ledger = FakeStorage(), A.Ledger()
        observed = A.AuditedStorage(native, 64, ledger)
        self.assertEqual(native.calls, [])
        observed.save(1, [Pointer(32), Pointer(32)])
        observed.load(1, [Pointer(32), Pointer(32)])
        self.assertEqual(native.calls, [("save", 1, [32, 32]), ("load", 1, [32, 32])])
        self.assertEqual(sum(row["bytes"] for row in observed.calls), 128)
        with self.assertRaises(ValueError):
            observed.save(0, [Pointer(32)])
        self.assertEqual(len(native.calls), 2)

    def test_storage_failure_preserves_attempt_and_does_not_claim_success(self):
        native, ledger = FakeStorage(fail=True), A.Ledger()
        observed = A.AuditedStorage(native, 64, ledger)
        with self.assertRaises(RuntimeError):
            observed.save(0, [Pointer(32), Pointer(32)])
        self.assertEqual(observed.calls[0]["status"], "raised")
        self.assertEqual([x["kind"] for x in ledger.events], ["checkpoint_copy_attempt", "checkpoint_copy_raised"])

    def test_edge_pullback_preserves_independent_transpose_identity_and_corners(self):
        physical = [[1, 2, 3], [4, 5, 6]]
        extended = [[1, 1, 2, 3, 3], [1, 1, 2, 3, 3],
                    [4, 4, 5, 6, 6], [4, 4, 5, 6, 6]]
        padded = [[1+5*i+j for j in range(5)] for i in range(4)]
        pulled = A.edge_pullback(padded, (2, 3), 1)
        self.assertEqual(pulled, [[16, 11, 28], [56, 31, 68]])
        left = sum(x*y for rowx, rowy in zip(extended, padded) for x, y in zip(rowx, rowy))
        right = sum(x*y for rowx, rowy in zip(physical, pulled) for x, y in zip(rowx, rowy))
        self.assertEqual(left, right)

    def test_centered_difference_keeps_expected_cubic_second_order_error(self):
        x = 2.0
        errors = []
        for epsilon in (0.1, 0.05):
            difference = A.centered_difference((x+epsilon)**3, (x-epsilon)**3, epsilon)
            errors.append(abs(difference-3*x*x))
        self.assertAlmostEqual(errors[0]/errors[1], 4.0, places=9)
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                A.centered_difference(value, 1.0, 0.1)

    def test_registered_K_and_all_FD_coordinates_are_required(self):
        protocol = json.loads((ROOT/"protocol.json").read_bytes())
        args = argparse.Namespace(phase="checkpoint2", checkpoints=None, direction=None,
                                  epsilon=None, sign=None, inputs=None)
        with self.assertRaises(ValueError):
            DRIVER.validate_worker_args(args, protocol)
        args.checkpoints = 2
        DRIVER.validate_worker_args(args, protocol)
        args.phase, args.checkpoints, args.direction = "fd", None, "interior"
        args.epsilon, args.sign = 0.002, 1
        with self.assertRaises(ValueError):
            DRIVER.validate_worker_args(args, protocol)
        args.epsilon = 0.001
        DRIVER.validate_worker_args(args, protocol)

    def test_manufactured_worker_error_preserves_files_and_never_retries(self):
        protocol = json.loads((ROOT/"protocol.json").read_bytes())
        calls = []
        def failure(args, protocol, output, ledger):
            calls.append(args.phase)
            ledger.add("manufactured_attempt_prefix", not_a_native_PDE=True)
            raise RuntimeError("manufactured worker failure")
        with tempfile.TemporaryDirectory(prefix="checkpoint-prefix-") as temporary:
            out = Path(temporary)/"fresh"
            argv = [str(ROOT/"run_checkpoint_pilot.py"), "--execute", "--root-go", "UNREAD_MANUFACTURED_HANDLE",
                    "--output", str(out), "--phase", "saved"]
            with patch.object(sys, "argv", argv), patch.object(DRIVER, "admit", return_value={"sha256": "manufactured-not-a-root-GO"}), \
                    patch.object(DRIVER, "environment", return_value={"manufactured": True}), \
                    patch.object(DRIVER, "worker", side_effect=failure), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(DRIVER.main(), 2)
            self.assertEqual(calls, ["saved"])
            self.assertTrue((out/"events.jsonl").is_file())
            self.assertTrue((out/"failure.json").is_file())
            self.assertFalse((out/"result.json").exists())
            self.assertIn("manufactured worker failure", (out/"failure.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PilotControls))
    receipt = {"status": "MANUFACTURED_ACCOUNTING_LIFECYCLE_PASS" if result.wasSuccessful() else "FAIL",
        "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "source_files": DRIVER.contract(), "control_source_sha256": DRIVER.sha(Path(__file__)),
        "Devito_PyRevolve_NumPy_imports": 0, "operators_constructed": 0, "PDE_calls": 0,
        "project_model_values_decoded": 0, "physics_or_gradient_agreement_claim": False}
    if args.receipt:
        DRIVER.record(args.receipt, receipt)
    print(json.dumps(receipt))
    return 0 if result.wasSuccessful() else 2


if __name__ == "__main__":
    raise SystemExit(main())
