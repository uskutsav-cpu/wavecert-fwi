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
    """Self-authored classes/origins only; no approved environment or GO."""
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
    revolve_source = "class Operator:\n    pass\nclass Checkpoint:\n    @property\n    def nbytes(self):\n        return 0\nclass BaseRevolver:\n"
    for name in ("__init__", "resetStorageList", "addStorage", "addNumpyStorage", "apply_forward", "apply_reverse", "save_checkpoint", "load_checkpoint", "remove_checkpoint"):
        revolve_source += "    def " + name + "(self, *args, **kwargs):\n        pass\n"
    revolve_source += "class SingleLevelRevolver(BaseRevolver):\n    def __init__(self, *args, **kwargs):\n        pass\nclass MemoryRevolver(SingleLevelRevolver):\n    def __init__(self, *args, **kwargs):\n        pass\nRevolver = MemoryRevolver\n"
    checkpoint_source = "class CheckpointOperator(Operator):\n"
    for name in ("__init__", "_prepare_args", "apply"):
        checkpoint_source += "    def " + name + "(self, *args, **kwargs):\n        pass\n"
    checkpoint_source += "class DevitoCheckpoint(Checkpoint):\n"
    for name in ("__init__", "get_data", "get_data_location"):
        checkpoint_source += "    def " + name + "(self, *args, **kwargs):\n        pass\n"
    for name in ("dtype", "size"):
        checkpoint_source += "    @property\n    def " + name + "(self):\n        return 0\n"
    checkpoint_source += "def get_symbol_data(symbol, timestep):\n    return None\n"
    rev, cp = modules["pyrevolve.pyrevolve"], modules["devito.checkpointing.checkpoint"]
    for module, source in ((rev, revolve_source), (cp, checkpoint_source)):
        if module is cp:
            vars(module).update(Checkpoint=rev.Checkpoint, Operator=rev.Operator)
        path = Path(module.__file__)
        path.write_text(source)
        # Only this self-authored class-definition fixture is executed. Its
        # methods are never called and no actual runtime package is imported.
        exec(compile(source, str(path), "exec", dont_inherit=True, optimize=0), vars(module))
    modules["devito"].DevitoCheckpoint = cp.DevitoCheckpoint
    modules["devito"].CheckpointOperator = cp.CheckpointOperator
    modules["pyrevolve"].Revolver = rev.MemoryRevolver
    primary = {name: {"path": modules[name].__file__, "sha256": probe.sha(Path(modules[name].__file__))}
        for name in primary_names}
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
            root = Path(temp).resolve()
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
            root = Path(temp).resolve()
            primary, modules = synthetic_origins(root)
            with patch.dict(sys.modules, modules), patch.object(probe.importlib, "import_module", side_effect=modules.__getitem__):
                bindings = {}
                selected = probe.imported_origins(primary, root, bindings)
            self.assertEqual(len(selected), 17)
            self.assertEqual(set(selected), set(modules))
            self.assertTrue(all(set(row) == {"path", "sha256"} for row in selected.values()))
            self.assertEqual(bindings["callable_count"], 21)
            self.assertEqual(len(bindings["callables"]), 21)
            self.assertEqual({row["method"] for row in bindings["callables"] if row.get("property_getter")}, {"nbytes", "size", "dtype"})

    def test_namespace_checkpoint_placeholder_in_both_aliases_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-placeholder-") as temp:
            root = Path(temp).resolve()
            primary, modules = synthetic_origins(root)
            placeholder = types.SimpleNamespace(__module__="devito.checkpointing.checkpoint")
            modules["devito.checkpointing.checkpoint"].DevitoCheckpoint = placeholder
            modules["devito"].DevitoCheckpoint = placeholder
            with patch.dict(sys.modules, modules), patch.object(probe.importlib, "import_module", side_effect=modules.__getitem__):
                with self.assertRaisesRegex(ValueError, "genuine pinned-module class"):
                    probe.imported_origins(primary, root)

    def test_namespace_operator_placeholder_in_both_aliases_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-op-placeholder-") as temp:
            root = Path(temp).resolve()
            primary, modules = synthetic_origins(root)
            placeholder = types.SimpleNamespace(__module__="devito.checkpointing.checkpoint")
            modules["devito.checkpointing.checkpoint"].CheckpointOperator = placeholder
            modules["devito"].CheckpointOperator = placeholder
            with patch.dict(sys.modules, modules), patch.object(probe.importlib, "import_module", side_effect=modules.__getitem__):
                with self.assertRaisesRegex(ValueError, "genuine pinned-module class"):
                    probe.imported_origins(primary, root)

    def test_genuine_but_wrong_exported_checkpoint_alias_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-wrong-alias-") as temp:
            root = Path(temp).resolve()
            primary, modules = synthetic_origins(root)
            modules["devito"].DevitoCheckpoint = type("DevitoCheckpoint", (), {"__module__": "devito.checkpointing.checkpoint"})
            with patch.dict(sys.modules, modules), patch.object(probe.importlib, "import_module", side_effect=modules.__getitem__):
                with self.assertRaisesRegex(ValueError, "class alias differs"):
                    probe.imported_origins(primary, root)

    def test_nonclass_memory_revolver_aliases_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-memory-placeholder-") as temp:
            root = Path(temp).resolve()
            primary, modules = synthetic_origins(root)
            placeholder = object()
            modules["pyrevolve.pyrevolve"].MemoryRevolver = placeholder
            modules["pyrevolve.pyrevolve"].Revolver = placeholder
            modules["pyrevolve"].Revolver = placeholder
            with patch.dict(sys.modules, modules), patch.object(probe.importlib, "import_module", side_effect=modules.__getitem__):
                with self.assertRaisesRegex(ValueError, "genuine pinned-module class"):
                    probe.imported_origins(primary, root)

    def test_foreign_method_with_forged_module_label_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-foreign-method-") as temp:
            root = Path(temp).resolve()
            primary, modules = synthetic_origins(root)
            def foreign(self, *args, **kwargs):
                return "foreign"
            foreign.__module__ = "devito.checkpointing.checkpoint"
            foreign.__qualname__ = "CheckpointOperator.apply"
            modules["devito.checkpointing.checkpoint"].CheckpointOperator.apply = foreign
            with patch.dict(sys.modules, modules), patch.object(probe.importlib, "import_module", side_effect=modules.__getitem__):
                with self.assertRaisesRegex(ValueError, "genuine pinned-module Python function"):
                    probe.imported_origins(primary, root)

    def test_foreign_method_with_same_filename_globals_labels_still_fails_code_binding(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-forged-code-") as temp:
            root = Path(temp).resolve()
            primary, modules = synthetic_origins(root)
            cp = modules["devito.checkpointing.checkpoint"]
            old = cp.CheckpointOperator.apply
            compiled = compile("def apply(self, *args, **kwargs):\n    return 777\n", cp.__file__, "exec", dont_inherit=True, optimize=0)
            code = next(value for value in compiled.co_consts if isinstance(value, types.CodeType))
            code = code.replace(co_firstlineno=old.__code__.co_firstlineno, co_qualname=old.__qualname__)
            foreign = types.FunctionType(code, vars(cp), name="apply")
            foreign.__module__ = cp.__name__
            foreign.__qualname__ = old.__qualname__
            cp.CheckpointOperator.apply = foreign
            with patch.dict(sys.modules, modules), patch.object(probe.importlib, "import_module", side_effect=modules.__getitem__):
                with self.assertRaisesRegex(ValueError, "method code differs"):
                    probe.imported_origins(primary, root)

    def test_checkpoint_property_replaced_with_plain_method_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-property-") as temp:
            root = Path(temp).resolve()
            primary, modules = synthetic_origins(root)
            cp = modules["devito.checkpointing.checkpoint"]
            cp.DevitoCheckpoint.size = cp.DevitoCheckpoint.get_data
            with patch.dict(sys.modules, modules), patch.object(probe.importlib, "import_module", side_effect=modules.__getitem__):
                with self.assertRaisesRegex(ValueError, "actual property"):
                    probe.imported_origins(primary, root)

    def test_inherited_revolver_method_replaced_by_foreign_function_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-revolver-method-") as temp:
            root = Path(temp).resolve()
            primary, modules = synthetic_origins(root)
            def foreign(self):
                return None
            modules["pyrevolve.pyrevolve"].BaseRevolver.apply_forward = foreign
            with patch.dict(sys.modules, modules), patch.object(probe.importlib, "import_module", side_effect=modules.__getitem__):
                with self.assertRaisesRegex(ValueError, "genuine pinned-module Python function"):
                    probe.imported_origins(primary, root)

    def test_late_import_mutation_of_earlier_file_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="import-probe-mutation-") as temp:
            root = Path(temp).resolve()
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
            root = Path(temp).resolve()
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
