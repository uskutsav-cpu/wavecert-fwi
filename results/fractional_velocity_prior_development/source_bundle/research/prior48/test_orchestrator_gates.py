"""Stdlib-only manufactured metadata gates; no study/data/network invocation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).with_name("orchestrate.py")
SPEC = importlib.util.spec_from_file_location("_manufactured_remote_gates", SOURCE)
ORCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORCH)


class GateControls(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "runtime": {
                "driver_sha256": "d" * 64,
                "protocol_sha256": "p" * 64,
                "schedule_sha256": "s" * 64,
            }
        }
        self.state = {
            "repository": ORCH.REPOSITORY,
            "ref": "refs/heads/" + ORCH.BRANCH,
            "event": "push",
            "attempt": 1,
            "public": True,
            "forced": False,
            "deleted": False,
            "sha": "b" * 40,
            "parents": ["a" * 40],
            "changed_paths": [ORCH.REGISTRATION],
        }
        self.control = {
            "schema_version": 1,
            "mode": "controls_only",
            "execution_authorized": False,
            "baseline_commit": ORCH.BASELINE,
            "production_source_sha256": ORCH.PRODUCTION,
            "production_file_count": 161,
            "expected_control_counts": {n: c for n, _p, c in ORCH.CONTROL_GROUPS},
            "python_version": "3.11.18",
            "numpy_version": "2.4.6",
            "scipy_version": "1.17.1",
            "bundle_manifest_sha256": "m" * 64,
            "root_go": None,
            "root_go_sha256": None,
            "source_bundle_commit": None,
            "prior_successful_control": None,
        }
        self.active = copy.deepcopy(self.control)
        self.active.update(
            mode="study",
            execution_authorized=True,
            source_bundle_commit="a" * 40,
            prior_successful_control={"commit": "a" * 40, "run_id": 123, "run_attempt": 1},
            root_go={
                "authorization": "GO_CLASSICAL_PRIOR48_DEVELOPMENT",
                "driver_sha256": "d" * 64,
                "protocol_sha256": "p" * 64,
                "schedule_sha256": "s" * 64,
                "production_source_sha256": ORCH.PRODUCTION,
                "independent_source_review_clear": True,
                "role": "MANUFACTURED metadata only; never dispatched",
            },
        )
        self.active["root_go_sha256"] = hashlib.sha256(
            ORCH.encoded(self.active["root_go"])
        ).hexdigest()
        self.api = {
            "id": 123,
            "run_attempt": 1,
            "status": "completed",
            "conclusion": "success",
            "head_sha": "a" * 40,
            "head_branch": ORCH.BRANCH,
            "event": "push",
            "name": ORCH.WORKFLOW_NAME,
            "path": ORCH.WORKFLOW,
            "repository": {"full_name": ORCH.REPOSITORY},
            "head_repository": {"full_name": ORCH.REPOSITORY},
        }

    def admit(self, registration=None, state=None, previous=None, api=None):
        return ORCH.validate_admission(
            registration or self.active,
            state or self.state,
            self.manifest,
            self.control if previous is None else previous,
            self.api if api is None else api,
        )

    def test_false_registration_is_controls_only(self):
        self.assertEqual(
            ORCH.validate_admission(self.control, self.state, self.manifest), "controls_only"
        )

    def test_only_fully_bound_manufactured_activation_admits_study(self):
        self.assertEqual(self.admit(), "study")

    def test_host_event_branch_attempt_public_force_guards(self):
        for key, value in (
            ("repository", "other/repo"),
            ("ref", "refs/heads/main"),
            ("event", "workflow_dispatch"),
            ("attempt", 2),
            ("public", False),
            ("forced", True),
            ("deleted", True),
            ("sha", "bad"),
        ):
            with self.subTest(key=key):
                state = copy.deepcopy(self.state)
                state[key] = value
                with self.assertRaises(ValueError):
                    self.admit(state=state)

    def test_authorization_is_strict_boolean(self):
        value = copy.deepcopy(self.active)
        value["execution_authorized"] = 1
        with self.assertRaises(ValueError):
            self.admit(registration=value)

    def test_false_registration_cannot_carry_GO(self):
        value = copy.deepcopy(self.control)
        value["root_go"] = {"not": "allowed"}
        with self.assertRaises(ValueError):
            ORCH.validate_admission(value, self.state, self.manifest)

    def test_parent_must_be_exact_single_bundle_commit(self):
        for parents in ([], ["c" * 40], ["a" * 40, "c" * 40]):
            with self.subTest(parents=parents):
                state = copy.deepcopy(self.state)
                state["parents"] = parents
                with self.assertRaises(ValueError):
                    self.admit(state=state)

    def test_activation_can_change_registration_only(self):
        for paths in ([], [ORCH.REGISTRATION, ORCH.WORKFLOW], ["research/prior48/orchestrate.py"]):
            with self.subTest(paths=paths):
                state = copy.deepcopy(self.state)
                state["changed_paths"] = paths
                with self.assertRaises(ValueError):
                    self.admit(state=state)

    def test_prior_run_binding_is_exact(self):
        for key, value in (
            ("commit", "c" * 40),
            ("run_id", 0),
            ("run_id", "123"),
            ("run_attempt", 2),
        ):
            with self.subTest(key=key, value=value):
                registration = copy.deepcopy(self.active)
                registration["prior_successful_control"][key] = value
                with self.assertRaises(ValueError):
                    self.admit(registration=registration)

    def test_parent_registration_must_be_same_control_bundle(self):
        for key, value in (
            ("execution_authorized", True),
            ("mode", "study"),
            ("bundle_manifest_sha256", "changed"),
        ):
            with self.subTest(key=key):
                previous = copy.deepcopy(self.control)
                previous[key] = value
                with self.assertRaises(ValueError):
                    self.admit(previous=previous)

    def test_failed_or_unrelated_GitHub_run_does_not_authorize(self):
        for key, value in (
            ("id", 999),
            ("run_attempt", 2),
            ("status", "in_progress"),
            ("conclusion", "failure"),
            ("head_sha", "c" * 40),
            ("head_branch", "main"),
            ("event", "workflow_dispatch"),
            ("name", "different"),
            ("path", ".github/workflows/other.yml"),
        ):
            with self.subTest(key=key):
                api = copy.deepcopy(self.api)
                api[key] = value
                with self.assertRaises(ValueError):
                    self.admit(api=api)

    def test_fork_control_run_is_rejected(self):
        for key in ("repository", "head_repository"):
            with self.subTest(key=key):
                api = copy.deepcopy(self.api)
                api[key] = {"full_name": "fork/repository"}
                with self.assertRaises(ValueError):
                    self.admit(api=api)

    def test_GO_fields_and_exact_bytes_required(self):
        for key in (
            "authorization",
            "driver_sha256",
            "protocol_sha256",
            "schedule_sha256",
            "production_source_sha256",
            "independent_source_review_clear",
        ):
            with self.subTest(key=key):
                registration = copy.deepcopy(self.active)
                registration["root_go"][key] = "wrong"
                registration["root_go_sha256"] = hashlib.sha256(
                    ORCH.encoded(registration["root_go"])
                ).hexdigest()
                with self.assertRaises(ValueError):
                    self.admit(registration=registration)
        registration = copy.deepcopy(self.active)
        registration["root_go_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.admit(registration=registration)

    def test_fixed_runtime_production_and_counts_cannot_change(self):
        for key, value in (
            ("baseline_commit", "c" * 40),
            ("production_source_sha256", "changed"),
            ("production_file_count", 160),
            ("python_version", "3.12.0"),
            ("numpy_version", "changed"),
            ("scipy_version", "changed"),
            ("expected_control_counts", {}),
        ):
            with self.subTest(key=key):
                registration = copy.deepcopy(self.active)
                registration[key] = value
                with self.assertRaises(ValueError):
                    self.admit(registration=registration)

    def test_registered_bundle_hash_and_file_tamper_rejected(self):
        with tempfile.TemporaryDirectory(prefix="prior48-gate-metadata-") as temp:
            root = Path(temp).resolve()
            bundle = root / "research/prior48"
            bundle.mkdir(parents=True)
            source = bundle / "dummy.py"
            source.write_text("# manufactured inert source\n")
            manifest = {
                "baseline_commit": ORCH.BASELINE,
                "production_source_sha256": ORCH.PRODUCTION,
                "files": {"research/prior48/dummy.py": ORCH.sha(source)},
            }
            path = bundle / "BUNDLE_MANIFEST.json"
            path.write_bytes(ORCH.encoded(manifest))
            registration = {"bundle_manifest_sha256": ORCH.sha(path)}
            self.assertEqual(ORCH.verify_bundle(root, registration), manifest)
            source.write_text("# changed\n")
            with self.assertRaises(ValueError):
                ORCH.verify_bundle(root, registration)

    def test_JUnit_counts_skips_and_duplicate_identities_rejected(self):
        with tempfile.TemporaryDirectory(prefix="prior48-gate-xml-") as temp:
            p = Path(temp).resolve() / "r.xml"
            p.write_text(
                '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase classname="x" name="one"/></testsuite></testsuites>'
            )
            self.assertEqual(ORCH.verify_xml(p, 1)["tests"], 1)
            with self.assertRaises(ValueError):
                ORCH.verify_xml(p, 2)
            p.write_text(
                '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0"><testcase classname="x" name="same"/><testcase classname="x" name="same"/></testsuite></testsuites>'
            )
            with self.assertRaises(ValueError):
                ORCH.verify_xml(p, 2)

    def test_control_upload_excludes_temporary_arrays_and_study(self):
        with tempfile.TemporaryDirectory(prefix="prior48-gate-upload-") as temp:
            root = Path(temp).resolve()
            live = root / "live"
            live.mkdir()
            (live / "metadata").mkdir()
            (live / "metadata/proof.json").write_text("{}")
            (live / "control_tmp").mkdir()
            (live / "control_tmp/dummy.npy").write_bytes(b"NOT_AN_ARRAY")
            (live / "study").mkdir()
            (live / "study/dummy.bin").write_bytes(b"NOT_MODEL_DATA")
            ORCH.collect_upload(live, root / "upload", "controls_only")
            self.assertFalse((root / "upload/control_tmp").exists())
            self.assertFalse((root / "upload/study").exists())
            self.assertTrue((root / "upload/metadata/proof.json").is_file())

    def test_upload_budget_violation_has_explicit_omission_inventory(self):
        with tempfile.TemporaryDirectory(prefix="prior48-gate-budget-") as temp:
            root = Path(temp).resolve()
            live = root / "live"
            live.mkdir()
            (live / "study").mkdir()
            (live / "study/dummy.bin").write_bytes(b"x" * 17)
            previous = ORCH.STUDY_CAP
            try:
                ORCH.STUDY_CAP = 16
                with self.assertRaises(ValueError):
                    ORCH.collect_upload(live, root / "upload", "study")
            finally:
                ORCH.STUDY_CAP = previous
            inventory = json.loads((root / "upload/UPLOAD_INVENTORY.json").read_bytes())
            self.assertFalse(inventory["complete_evidence_collection"])
            self.assertEqual(inventory["omitted"][0]["bytes"], 17)

    def test_mutation_during_copy_has_actual_target_hash_and_incomplete_inventory(self):
        with tempfile.TemporaryDirectory(prefix="prior48-gate-copy-") as temp:
            root = Path(temp).resolve()
            live = root / "live"
            live.mkdir()
            (live / "metadata").mkdir()
            source = live / "metadata/mutable.json"
            source.write_bytes(b"original")
            original = ORCH.copy_snapshot

            def mutate_after_copy(src, dst, size):
                original(src, dst, size)
                src.write_bytes(b"changed")

            try:
                ORCH.copy_snapshot = mutate_after_copy
                with self.assertRaises(ValueError):
                    ORCH.collect_upload(live, root / "upload", "controls_only")
            finally:
                ORCH.copy_snapshot = original
            inventory = json.loads((root / "upload/UPLOAD_INVENTORY.json").read_bytes())
            self.assertFalse(inventory["complete_evidence_collection"])
            actual = root / "upload/metadata/mutable.json"
            self.assertEqual(inventory["retained"][0]["sha256"], ORCH.sha(actual))
            self.assertEqual(actual.read_bytes(), b"original")

    def test_control_environment_contains_no_child_tokens(self):
        controlled = ORCH.controlled_environment()
        self.assertTrue(all(k not in controlled for k in ORCH.TOKEN_NAMES))
        self.assertTrue(all(controlled[k] == v for k, v in ORCH.THREADS.items()))
        self.assertEqual(controlled["PYTEST_DISABLE_PLUGIN_AUTOLOAD"], "1")

    def test_four_frozen_fresh_control_groups_are_exact(self):
        self.assertEqual(
            [(x[0], x[2]) for x in ORCH.CONTROL_GROUPS],
            [("prior", 39), ("adapter", 21), ("driver", 30), ("launcher", 50)],
        )
        self.assertEqual(sum(x[2] for x in ORCH.CONTROL_GROUPS), 140)

    def test_interrupted_dispatch_handoff_recovers_only_recorded_owned_PID(self):
        with tempfile.TemporaryDirectory(prefix="prior48-gate-handoff-") as temp:
            job = Path(temp).resolve()
            receipt = {"supervisor_PID": 12345, "dispatch_only": True, "detached_session": True}
            ORCH.record(job / "dispatch_receipt.json", receipt)
            # Models a stop after dispatch persisted/returned, before assignment.
            self.assertEqual(
                ORCH.recover_supervisor_pid(job, None, None), (12345, "immutable_dispatch_receipt")
            )
            self.assertEqual(
                ORCH.recover_supervisor_pid(job, receipt, None), (12345, "returned_dispatch")
            )
            self.assertEqual(
                ORCH.recover_supervisor_pid(job, None, 12345), (12345, "assigned_supervisor_PID")
            )
            (job / "dispatch_receipt.json").unlink()
            self.assertEqual(
                ORCH.recover_supervisor_pid(job, {"supervisor_PID": True}, None),
                (None, "no_verified_handle"),
            )

    def test_default_CLI_cannot_launch(self):
        with tempfile.TemporaryDirectory(prefix="prior48-gate-default-") as temp:
            result = subprocess.run(
                [sys.executable, "-B", "-I", str(SOURCE)],
                cwd=temp,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(json.loads(result.stdout)["status"], "HELD_METADATA_ONLY_NO_LAUNCH")
            self.assertEqual(list(Path(temp).resolve().iterdir()), [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(GateControls)
    )
    receipt = {
        "status": "MANUFACTURED_METADATA_GATE_CONTROLS_PASS"
        if result.wasSuccessful()
        else "FAILED",
        "tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "source_sha256": ORCH.sha(Path(__file__)),
        "orchestrator_sha256": ORCH.sha(SOURCE),
        "actual_study_launches": 0,
        "project_arrays_decoded": 0,
        "PDE_calls": 0,
        "network_requests": 0,
    }
    if args.receipt:
        ORCH.record(args.receipt, receipt)
    print(json.dumps(receipt))
    return 0 if result.wasSuccessful() else 2


if __name__ == "__main__":
    raise SystemExit(main())
