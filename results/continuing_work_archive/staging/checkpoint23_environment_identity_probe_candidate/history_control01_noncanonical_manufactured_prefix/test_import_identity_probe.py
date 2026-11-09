"""Stdlib-only synthetic identity/guard controls; no real runtime imports."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "checkpoint_pilot23_runtime_contract_v2_candidate"
spec = importlib.util.spec_from_file_location("manufactured_import_only_probe", HERE / "import_identity_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
BLOCKED_IMPORT_ATTEMPTS = []


def forbid_runtime_imports(event, args):
    if event == "import" and args[0].split(".")[0] in probe.TARGET_ROOTS:
        BLOCKED_IMPORT_ATTEMPTS.append(args[0])
        raise RuntimeError("real runtime import forbidden in manufactured controls")
    if event == "subprocess.Popen":
        raise RuntimeError("no process is authorized by these controls")


sys.addaudithook(forbid_runtime_imports)


def args_for(root, value):
    path = root / "manufactured_false_or_malformed_go.json"
    probe.write(path, value)
    prefix = root / "manufactured_prefix"
    prefix.mkdir()
    return argparse.Namespace(root_preparation_go=path,
        root_preparation_go_sha256=probe.sha(path), protocol=V2 / "worker/protocol.json",
        pins=V2 / "worker/RUNTIME_SOURCE_PINS.json", prefix=prefix, output=root / "not_created")


def synthetic_origins(root):
    """Tiny fake origins only, never an approved environment or GO receipt."""
    pins = probe.read_json(V2 / "worker/RUNTIME_SOURCE_PINS.json")
    primary_names = [probe.module_name(row["path"]) for row in pins["files"]]
    all_names = sorted(set(primary_names) | {"pyrevolve", "pyrevolve.crevolve", "numpy", "scipy"})
    modules = {}
    for index, name in enumerate(all_names):
        path = root / (str(index) + ".synthetic_origin")
        path.write_bytes(b"MANUFACTURED_ORIGIN_ONLY\n")
        module = types.ModuleType(name)
        module.__file__ = str(path)
        modules[name] = module
    primary = {name: {"path": modules[name].__file__, "sha256": probe.sha(Path(modules[name].__file__))}
        for name in primary_names}
    modules["devito"].DevitoCheckpoint = types.SimpleNamespace(__module__="devito.checkpointing.checkpoint")
    modules["devito"].CheckpointOperator = types.SimpleNamespace(__module__="devito.checkpointing.checkpoint")
    sentinel = object()
    modules["pyrevolve"].Revolver = sentinel
    modules["pyrevolve.pyrevolve"].MemoryRevolver = sentinel
    return primary, modules


class ImportProbeControls(unittest.TestCase):
    def assert_preimport_rejection(self, args, message):
        with patch.object(probe, "preimport") as pre, patch.object(probe.importlib, "import_module") as target:
            with self.assertRaisesRegex(ValueError, message):
                probe.run(args)
            pre.assert_not_called()
            target.assert_not_called()
        self.assertFalse(args.output.exists())

    def test_default_main_creates_no_evidence_or_target_import(self):
        out = io.StringIO()
        with patch.object(sys, "argv", [str(HERE / "import_identity_probe.py")]), redirect_stdout(out):
            with patch.object(probe, "read_json") as read, patch.object(probe, "sha") as digest:
                with patch.object(probe.importlib, "import_module") as target:
                    self.assertEqual(probe.main(), 0)
                    target.assert_not_called()
                    read.assert_not_called()
                    digest.assert_not_called()
        result = json.loads(out.getvalue())
        self.assertEqual(result["status"], "HELD_IMPORT_ONLY_PROBE_NO_TARGET_IMPORT_NO_FILES")
        self.assertFalse(result["native_use_review_clear"])

    def test_false_authorization_stops_before_metadata_imports_and_output(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-false-") as temp:
            args = args_for(Path(temp), {"execution_authorized": False, "manufactured_control": True})
            self.assert_preimport_rejection(args, "preparation-only")

    def test_nondict_go_has_clean_rejection_before_imports(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-shape-") as temp:
            self.assert_preimport_rejection(args_for(Path(temp), []), "GO must be a JSON object")

    def test_nondict_nested_go_has_clean_rejection_before_imports(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-projection-") as temp:
            self.assert_preimport_rejection(args_for(Path(temp), {"import_identity_probe": False}), "projection must be a JSON object")

    def test_wrong_go_hash_stops_before_imports(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-hash-") as temp:
            args = args_for(Path(temp), {"execution_authorized": False})
            args.root_preparation_go_sha256 = "0" * 64
            self.assert_preimport_rejection(args, "GO hash")

    def test_wrong_protocol_bytes_stop_before_imports(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-source-") as temp:
            root = Path(temp)
            args = args_for(root, {"execution_authorized": False})
            args.protocol = root / "manufactured_wrong_protocol.json"
            args.protocol.write_bytes(b"{}\n")
            self.assert_preimport_rejection(args, "V2 protocol")

    def test_grid_constructor_body_is_blocked(self):
        reached = []
        class Grid:
            __module__ = "devito.synthetic_control"
            def __init__(self):
                reached.append("body")
        guard = probe.ImportCallGuard()
        with self.assertRaisesRegex(RuntimeError, "forbids tracked"):
            with guard:
                Grid()
        self.assertEqual(reached, [])
        self.assertEqual(len(guard.blocked), 1)
        self.assertIsNone(sys.getprofile())

    def test_inherited_constructor_self_mro_is_blocked(self):
        reached = []
        class Grid:
            __module__ = "devito.synthetic_control"
            def __init__(self):
                reached.append("body")
        class RenamedChild(Grid):
            pass
        with self.assertRaisesRegex(RuntimeError, "forbids tracked"):
            with probe.ImportCallGuard():
                RenamedChild()
        self.assertEqual(reached, [])

    def test_inherited_new_cls_mro_is_blocked_in_foreign_helper(self):
        reached = []
        class GenericFactory:
            def __new__(cls):
                reached.append("allocation")
                return object.__new__(cls)
        class TimeFunction(GenericFactory):
            __module__ = "devito.synthetic_control"
        with self.assertRaisesRegex(RuntimeError, "forbids tracked"):
            with probe.ImportCallGuard():
                TimeFunction()
        self.assertEqual(reached, [])

    def test_operator_apply_body_is_blocked_through_self_mro(self):
        reached = []
        class Operator:
            __module__ = "devito.synthetic_control"
            def apply(self):
                reached.append("PDE")
        object_only = object.__new__(Operator)
        with self.assertRaisesRegex(RuntimeError, "forbids tracked"):
            with probe.ImportCallGuard():
                object_only.apply()
        self.assertEqual(reached, [])

    def test_revolver_subclass_name_is_blocked(self):
        reached = []
        class MemoryRevolver:
            __module__ = "pyrevolve.synthetic_control"
            def __init__(self):
                reached.append("schedule")
        with self.assertRaisesRegex(RuntimeError, "forbids tracked"):
            with probe.ImportCallGuard():
                MemoryRevolver()
        self.assertEqual(reached, [])

    def test_unrelated_classes_and_foreign_grid_remain_usable(self):
        reached = []
        class Grid:
            def __init__(self):
                reached.append("foreign")
        class Timer:
            __module__ = "devito.synthetic_control"
            def __init__(self):
                reached.append("unrelated")
        guard = probe.ImportCallGuard()
        with guard:
            Grid()
            Timer()
        self.assertEqual(reached, ["foreign", "unrelated"])
        self.assertEqual(guard.blocked, [])

    def test_preexisting_profile_hook_is_not_overwritten(self):
        def previous(_frame, _event, _arg):
            return None
        sys.setprofile(previous)
        try:
            with self.assertRaisesRegex(ValueError, "pre-existing"):
                with probe.ImportCallGuard():
                    self.fail("must not enter")
            self.assertIs(sys.getprofile(), previous)
        finally:
            sys.setprofile(None)

    def test_exact17_synthetic_origin_set_is_returned_without_real_imports(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-origins-") as temp:
            root = Path(temp)
            primary, modules = synthetic_origins(root)
            with patch.dict(sys.modules, modules), patch.object(probe.importlib, "import_module", side_effect=modules.__getitem__):
                selected = probe.imported_origins(primary, root)
            self.assertEqual(len(selected), 17)
            self.assertEqual(set(selected), set(modules))
            self.assertTrue(all(set(row) == {"path", "sha256"} for row in selected.values()))

    def test_late_import_mutation_of_earlier_file_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-mutation-") as temp:
            root = Path(temp)
            primary, modules = synthetic_origins(root)
            def import_fake(name):
                if name == "scipy":
                    Path(modules["devito"].__file__).write_bytes(b"LATE_MUTATION\n")
                return modules[name]
            with patch.dict(sys.modules, modules), patch.object(probe.importlib, "import_module", side_effect=import_fake):
                with self.assertRaisesRegex(ValueError, "origin changed during imports"):
                    probe.imported_origins(primary, root)

    def test_late_import_repointing_earlier_origin_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-repoint-") as temp:
            root = Path(temp)
            primary, modules = synthetic_origins(root)
            alternate = root / "alternate.synthetic_origin"
            alternate.write_bytes(b"MANUFACTURED_ORIGIN_ONLY\n")
            def import_fake(name):
                if name == "scipy":
                    modules["devito"].__file__ = str(alternate)
                return modules[name]
            with patch.dict(sys.modules, modules), patch.object(probe.importlib, "import_module", side_effect=import_fake):
                with self.assertRaisesRegex(ValueError, "origin changed during imports"):
                    probe.imported_origins(primary, root)

    def test_missing_primary_denominator_rejects_before_any_import(self):
        with patch.object(probe.importlib, "import_module") as target:
            with self.assertRaisesRegex(ValueError, "exact17"):
                probe.imported_origins({}, HERE)
            target.assert_not_called()


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ImportProbeControls))
    if BLOCKED_IMPORT_ATTEMPTS:
        raise RuntimeError("a real runtime import was attempted despite the manufactured-only scope")
    raise SystemExit(0 if result.wasSuccessful() else 1)
