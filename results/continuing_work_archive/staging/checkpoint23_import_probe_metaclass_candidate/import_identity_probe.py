"""Held import-only checkpoint23 environment probe; default performs no imports.

Only a separately source-bound environment-preparation GO admits target imports.
This program never installs packages, constructs a model/operator/revolver,
executes a worker, or grants native-use approval. Library import initializers do
execute native extension code; the Python call guard is deliberately limited.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import sys
import sysconfig
import time
import types

PROTOCOL_SHA = "27e5a331d6bd9eff0dd6b4fa0ca1bc666c6014ed70ffb3da01821c62f83fdcbc"
PINS_SHA = "bfbc2e31cd4ff7ae2a0f0342dc3633c5e4dc0189daea0d06eaeb12de1040350f"
PACKAGES = {"devito": "4.8.23", "pyrevolve": "2.2.8", "numpy": "2.4.3", "scipy": "1.17.1"}
THREADS = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
NATIVE_ENV = {"DEVITO_LANGUAGE": "C", "DEVITO_MPI": "0", "DEVITO_AUTOTUNING": "off",
    "DEVITO_AUTOPADDING": "0", "DEVITO_SAFE_MATH": "1", "DEVITO_JIT_BACKDOOR": "0",
    "DEVITO_FIRST_TOUCH": "0", "MPLBACKEND": "Agg"}
FRAMEWORKS = ("devito", "examples.seismic", "pyrevolve")
TARGET_ROOTS = {"devito", "examples", "pyrevolve", "numpy", "scipy"}
FORBIDDEN_CLASSES = {"Grid", "Model", "SeismicModel", "TimeFunction", "SparseTimeFunction",
    "Operator", "DevitoCheckpoint", "CheckpointOperator", "Revolver"}
CONSTRUCTOR_METHODS = {"__init__", "__new__", "__call__"}
APPLY_METHODS = {"apply", "apply_forward", "apply_reverse", "apply_adjoint", "apply_jacobian"}


def sha(path, cap=64 * 1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > cap:
        raise ValueError("bounded non-symlink regular file required: " + str(path))
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            value.update(block)
    return value.hexdigest()


def canonical(path):
    value = Path(path)
    if not value.is_absolute() or value != value.resolve():
        raise ValueError("canonical absolute path required")
    return value


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
        raise ValueError("bounded non-symlink JSON required")
    def unique(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def write(path, value):
    body = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(body) > 1024**2:
        raise ValueError("probe evidence JSON exceeds fixed bound")
    with Path(path).open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def module_name(path):
    return path.removesuffix(".py").replace("/", ".").removesuffix(".__init__")


def framework(name):
    return isinstance(name, str) and any(name == root or name.startswith(root + ".") for root in FRAMEWORKS)


def forbidden_class_name(name):
    return name in FORBIDDEN_CLASSES or name.endswith("Revolver")


class ImportCallGuard:
    """Reject observed Python constructors/apply calls before their bodies run.

    This does not intercept every C-extension call, all allocations, subprocess
    code, or a deliberately evasive library. It does not sandbox package imports.
    """
    def __init__(self):
        self.blocked = []
        self.python_call_events = 0
        self.previous = None

    def __call__(self, frame, event, arg):
        if event != "call":
            return
        self.python_call_events += 1
        method = frame.f_code.co_name
        if method not in CONSTRUCTOR_METHODS | APPLY_METHODS:
            return
        # A Python metaclass initializer creates a class, not its instances.
        if method == "__init__" and frame.f_code.co_argcount:
            receiver = frame.f_locals.get(frame.f_code.co_varnames[0])
            if isinstance(receiver, type):
                initializer = inspect.getattr_static(type(receiver), "__init__", None)
                if (isinstance(initializer, types.FunctionType)
                        and initializer.__code__ is frame.f_code):
                    return
        module = frame.f_globals.get("__name__", "")
        classes = []
        candidate = frame.f_locals.get("cls")
        if isinstance(candidate, type):
            classes.extend(candidate.__mro__)
        if "self" in frame.f_locals:
            classes.extend(type(frame.f_locals["self"]).__mro__)
        related = [cls for cls in classes if framework(cls.__module__) and forbidden_class_name(cls.__name__)]
        lexical = any(forbidden_class_name(part) for part in frame.f_code.co_qualname.split(".")[:-1])
        reject = bool(related) or (framework(module) and (method in APPLY_METHODS or lexical))
        if reject:
            row = {"module": module, "qualname": frame.f_code.co_qualname,
                "method": method, "related_classes": sorted({cls.__module__ + "." + cls.__name__ for cls in related})}
            self.blocked.append(row)
            raise RuntimeError("import-only probe forbids tracked constructor/apply call: " + module + "." + frame.f_code.co_qualname)

    def __enter__(self):
        self.previous = sys.getprofile()
        if self.previous is not None:
            raise ValueError("fresh process with no pre-existing Python profile hook required")
        sys.setprofile(self)
        return self

    def __exit__(self, *_error):
        sys.setprofile(self.previous)

    def evidence(self):
        return {"python_call_events_observed": self.python_call_events,
            "tracked_forbidden_python_calls_attempted": len(self.blocked), "blocked_calls": self.blocked,
            "scope": "Python constructor/apply call events, including inherited cls/self MRO; not all native calls or allocations. Import initializers/BLAS/compiled scheduler loading are expected and are not PDE evidence."}


def authorize(args):
    """Source/GO/path checks only; no package metadata or target imports here."""
    if sha(args.root_preparation_go) != args.root_preparation_go_sha256:
        raise ValueError("exact root preparation GO hash required")
    if sha(args.protocol) != PROTOCOL_SHA or sha(args.pins) != PINS_SHA:
        raise ValueError("exact reviewed V2 protocol and primary pins required")
    expected = {"import_identity_probe.py": sha(Path(__file__).resolve()),
        "protocol.json": PROTOCOL_SHA, "RUNTIME_SOURCE_PINS.json": PINS_SHA}
    go = read_json(args.root_preparation_go)
    if not isinstance(go, dict):
        raise ValueError("root preparation GO must be a JSON object")
    scope = go.get("import_identity_probe", {})
    if not isinstance(scope, dict):
        raise ValueError("import identity GO projection must be a JSON object")
    if (go.get("authorization") != "GO_CHECKPOINT23_ENVIRONMENT_PREPARATION_V1"
            or go.get("execution_authorized") is not True
            or go.get("native_execution_authorized") is not False
            or go.get("import_only_identity_probe_authorized") is not True
            or scope.get("source_files") != expected):
        raise ValueError("separate exact environment-preparation-only import authorization required")
    prefix, output = canonical(args.prefix), canonical(args.output)
    source = Path(__file__).resolve().parent
    if (scope.get("prefix") != str(prefix) or not prefix.is_dir() or output.exists()
            or not output.parent.is_dir() or output.is_relative_to(prefix) or output.is_relative_to(source)
            or any(output.is_relative_to(Path(p).resolve().parent) for p in (args.protocol, args.pins))):
        raise ValueError("fresh separate output and exact registered installed prefix required")
    return prefix, output, expected


def preimport(prefix, protocol_path, pins_path):
    """Read exact package metadata and13 primary file bytes before any import."""
    if any(name.split(".")[0] in TARGET_ROOTS for name in sys.modules):
        raise ValueError("fresh target-import-free process required")
    protocol, pins = read_json(protocol_path), read_json(pins_path)
    runtime = protocol["runtime"]
    if (platform.system() != "Linux" or platform.machine() != "x86_64"
            or platform.python_version() != "3.11.18" or runtime["python"] != "3.11.18"
            or any(runtime[name] != value for name, value in PACKAGES.items())
            or Path(sys.prefix).resolve() != prefix or not Path(sys.executable).absolute().is_relative_to(prefix)
            or not sys.flags.isolated or not sys.dont_write_bytecode):
        raise ValueError("exact isolated Linuxx86_64/Python3.11.18 reviewed prefix required")
    if any(os.environ.get(key) != value for key, value in (THREADS | NATIVE_ENV).items()):
        raise ValueError("registered thread/Devito import environment required")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("controlled bytecode/hash-seed environment required")
    if any(key in os.environ for key in ("PYTHONPATH", "PYTHONHOME", "GITHUB_TOKEN", "GH_TOKEN",
            "ACTIONS_RUNTIME_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_TOKEN")):
        raise ValueError("foreign import overrides or credential variables forbidden")
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    if versions != PACKAGES:
        raise ValueError("exact V2 runtime distributions required")
    rows = pins["files"]
    if len(rows) != 13 or len({module_name(row["path"]) for row in rows}) != 13:
        raise ValueError("exact13 primary source denominator required")
    checked = {}
    for row in rows:
        original = Path(importlib.metadata.distribution(row["distribution"]).locate_file(row["path"]))
        path = original.resolve()
        if original.is_symlink() or not path.is_relative_to(prefix) or sha(path) != row["sha256"]:
            raise ValueError("installed primary source differs: " + row["path"])
        checked[module_name(row["path"])] = {"path": str(path), "sha256": row["sha256"]}
    return checked


def code_identity(code):
    """Canonical Python-code fingerprint; no code object is executed here."""
    def constant(value):
        if isinstance(value, types.CodeType):
            return ["code", payload(value)]
        if value is None:
            return ["None"]
        if value is Ellipsis:
            return ["Ellipsis"]
        if type(value) is bool:
            return ["bool", value]
        if type(value) is int:
            return ["int", str(value)]
        if type(value) is float:
            return ["float", value.hex()]
        if type(value) is complex:
            return ["complex", value.real.hex(), value.imag.hex()]
        if type(value) is str:
            return ["str", value]
        if type(value) is bytes:
            return ["bytes", value.hex()]
        if type(value) is tuple:
            return ["tuple", [constant(item) for item in value]]
        if type(value) is frozenset:
            return ["frozenset", sorted((constant(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))]
        raise ValueError("unsupported constant in selected Python method code")
    def payload(value):
        scalar = ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals", "co_stacksize",
            "co_flags", "co_filename", "co_name", "co_qualname", "co_firstlineno")
        result = {key: getattr(value, key) for key in scalar}
        for key in ("co_code", "co_linetable", "co_exceptiontable"):
            result[key] = getattr(value, key).hex()
        for key in ("co_names", "co_varnames", "co_freevars", "co_cellvars"):
            result[key] = list(getattr(value, key))
        result["co_consts"] = [constant(item) for item in value.co_consts]
        return result
    if not isinstance(code, types.CodeType):
        raise ValueError("selected method must have an actual Python code object")
    return hashlib.sha256(json.dumps(payload(code), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_code_index(module, selected):
    """Compile already hash-bound source without importing or executing it."""
    name = module.__name__
    row = selected[name]
    path = canonical(row["path"])
    if sha(path) != row["sha256"]:
        raise ValueError("selected method source file changed: " + name)
    root = compile(path.read_bytes(), str(path), "exec", dont_inherit=True, optimize=0)
    classes, functions = {}, {}
    for item in root.co_consts:
        if not isinstance(item, types.CodeType):
            continue
        if item.co_name in ("DevitoCheckpoint", "CheckpointOperator", "Checkpoint", "Operator",
                "BaseRevolver", "SingleLevelRevolver", "MemoryRevolver"):
            classes[item.co_name] = {method.co_name: method for method in item.co_consts if isinstance(method, types.CodeType)}
        else:
            functions[item.co_name] = item
    return {"path": str(path), "sha256": row["sha256"], "classes": classes, "functions": functions}


def verify_class_bindings(selected):
    """Bind used class objects and Python method code to selected primary bytes."""
    cp_module = sys.modules["devito.checkpointing.checkpoint"]
    rev_module = sys.modules["pyrevolve.pyrevolve"]
    devito, pyrevolve = sys.modules["devito"], sys.modules["pyrevolve"]
    owners = {}
    for module, names in ((cp_module, ("DevitoCheckpoint", "CheckpointOperator")),
            (rev_module, ("Checkpoint", "Operator", "BaseRevolver", "SingleLevelRevolver", "MemoryRevolver"))):
        for name in names:
            cls = vars(module).get(name)
            if (not isinstance(cls, type) or cls.__module__ != module.__name__
                    or cls.__name__ != name or cls.__qualname__ != name):
                raise ValueError("genuine pinned-module class required: " + module.__name__ + "." + name)
            owners[name] = cls
    for exported, name in ((devito, "DevitoCheckpoint"), (devito, "CheckpointOperator")):
        if vars(exported).get(name) is not owners[name]:
            raise ValueError("exported checkpoint class alias differs: " + name)
    if vars(pyrevolve).get("Revolver") is not owners["MemoryRevolver"] or vars(rev_module).get("Revolver") is not owners["MemoryRevolver"]:
        raise ValueError("exported MemoryRevolver class alias differs")
    for name, bases in (("DevitoCheckpoint", ("Checkpoint",)), ("CheckpointOperator", ("Operator",)),
            ("SingleLevelRevolver", ("BaseRevolver",)), ("MemoryRevolver", ("SingleLevelRevolver", "BaseRevolver"))):
        if owners[name].__mro__ != (owners[name], *(owners[base] for base in bases), object):
            raise ValueError("pinned checkpoint/revolver class MRO differs: " + name)
    indexes = {module.__name__: source_code_index(module, selected) for module in (cp_module, rev_module)}
    rows = []
    def bind(lookup, owner, name, property_getter=False):
        cls = owners[lookup]
        descriptor = inspect.getattr_static(cls, name)
        if descriptor is not vars(owners[owner]).get(name):
            raise ValueError("selected method owner differs: " + lookup + "." + name)
        if property_getter:
            if type(descriptor) is not property:
                raise ValueError("selected getter must be an actual property: " + name)
            function = descriptor.fget
        else:
            function = descriptor
        module = cp_module if owner in ("DevitoCheckpoint", "CheckpointOperator") else rev_module
        index = indexes[module.__name__]
        if (not isinstance(function, types.FunctionType) or function.__globals__ is not vars(module)
                or function.__module__ != module.__name__ or function.__qualname__ != owner + "." + name):
            raise ValueError("selected method must be a genuine pinned-module Python function: " + lookup + "." + name)
        expected = index["classes"][owner][name]
        if function.__code__.co_filename != index["path"] or code_identity(function.__code__) != code_identity(expected):
            raise ValueError("selected method code differs from pinned source: " + lookup + "." + name)
        rows.append({"lookup_class": lookup, "defining_class": owner, "method": name,
            "property_getter": property_getter, "module": module.__name__, "source_path": index["path"],
            "source_sha256": index["sha256"], "code_sha256": code_identity(expected),
            "code_first_line": expected.co_firstlineno})
    for name in ("__init__", "get_data", "get_data_location"):
        bind("DevitoCheckpoint", "DevitoCheckpoint", name)
    for name in ("dtype", "size"):
        bind("DevitoCheckpoint", "DevitoCheckpoint", name, True)
    bind("DevitoCheckpoint", "Checkpoint", "nbytes", True)
    for name in ("__init__", "_prepare_args", "apply"):
        bind("CheckpointOperator", "CheckpointOperator", name)
    for owner in ("MemoryRevolver", "SingleLevelRevolver", "BaseRevolver"):
        bind(owner, owner, "__init__")
    for name in ("resetStorageList", "addStorage", "addNumpyStorage", "apply_forward", "apply_reverse",
            "save_checkpoint", "load_checkpoint", "remove_checkpoint"):
        bind("MemoryRevolver", "BaseRevolver", name)
    helper = vars(cp_module).get("get_symbol_data")
    index = indexes[cp_module.__name__]
    expected = index["functions"]["get_symbol_data"]
    if (not isinstance(helper, types.FunctionType) or helper.__globals__ is not vars(cp_module)
            or helper.__code__.co_filename != index["path"] or code_identity(helper.__code__) != code_identity(expected)):
        raise ValueError("checkpoint data helper code differs from pinned source")
    rows.append({"module_function": "get_symbol_data", "module": cp_module.__name__,
        "source_path": index["path"], "source_sha256": index["sha256"],
        "code_sha256": code_identity(expected), "code_first_line": expected.co_firstlineno})
    return {"status": "SELECTED_ALIAS_CLASSES_AND_21_PYTHON_CALLABLES_BOUND_TO_PRIMARY_SOURCE",
        "class_aliases": {"devito.DevitoCheckpoint": "devito.checkpointing.checkpoint.DevitoCheckpoint",
            "devito.CheckpointOperator": "devito.checkpointing.checkpoint.CheckpointOperator",
            "pyrevolve.Revolver": "pyrevolve.pyrevolve.MemoryRevolver"},
        "callable_count": len(rows), "callables": rows,
        "scope": "Exact class identity/MRO, static descriptors, globals and canonical Python code fingerprints compiled without execution from selected source bytes. No instance construction, mutable global dependency closure or native ABI attestation."}


def imported_origins(primary, prefix, binding_evidence=None):
    """Only explicit operations are module imports, identity reads and hashing."""
    required = set(primary) | {"pyrevolve", "pyrevolve.crevolve", "numpy", "scipy"}
    if len(required) != 17:
        raise ValueError("exact17 actual module denominator required")
    selected = {}
    for name in sorted(required):
        module = importlib.import_module(name)
        original = Path(module.__file__)
        path = original.resolve()
        if original.is_symlink() or not path.is_relative_to(prefix):
            raise ValueError("actual imported origin outside canonical reviewed prefix: " + name)
        selected[name] = {"path": str(path), "sha256": sha(path)}
        if name in primary and selected[name] != primary[name]:
            raise ValueError("actual imported primary source differs: " + name)
    bindings = verify_class_bindings(selected)
    if binding_evidence is not None:
        binding_evidence.update(bindings)
    # A later module import must not silently mutate an earlier observed file.
    for name, original in selected.items():
        module = sys.modules[name]
        current_path = Path(module.__file__).resolve()
        if (str(current_path) != original["path"] or Path(module.__file__).is_symlink()
                or sha(current_path) != original["sha256"]):
            raise ValueError("selected module origin changed during imports: " + name)
    return selected


def process_cost():
    own = resource.getrusage(resource.RUSAGE_SELF)
    return {"user_CPU_seconds": own.ru_utime, "system_CPU_seconds": own.ru_stime,
        "process_peak_RSS_bytes": int(own.ru_maxrss) * (1024 if platform.system() == "Linux" else 1),
        "scope": "Own process cumulative peak and CPU, including admission/import/hash instrumentation. Full external wall/actual wait belongs to the controller."}


def run(args):
    entered = time.perf_counter()
    prefix, output, sources = authorize(args)
    primary = preimport(prefix, args.protocol, args.pins)
    output.mkdir()
    write(output / "admission.json", {"status": "ENVIRONMENT_IMPORT_ONLY_ADMITTED_NOT_NATIVE",
        "root_preparation_go_sha256": args.root_preparation_go_sha256, "source_files": sources,
        "prefix": str(prefix), "PID": os.getpid(), "PPID": os.getppid(),
        "process_group_ID": os.getpgrp(), "session_ID": os.getsid(0), "native_use_review_clear": False})
    guard = ImportCallGuard()
    bindings = {}
    imports_started = time.perf_counter()
    try:
        with guard:
            selected = imported_origins(primary, prefix, bindings)
        imports_finished = time.perf_counter()
        invocation = Path(sys.executable).absolute()
        receipt = {"schema": "checkpoint_pilot_ephemeral_linux_environment_v1",
            "status": "CANDIDATE_IMPORT_IDENTITIES_ONLY_REQUIRES_ROOT_ENVIRONMENT_REVIEW",
            "ephemeral": True, "native_use_review_clear": False,
            "system": platform.system(), "machine": platform.machine(),
            "python_version": platform.python_version(), "python_executable": str(invocation),
            "python_realpath": str(invocation.resolve()), "python_binary_sha256": sha(invocation.resolve()),
            "prefix": str(prefix), "packages": PACKAGES, "modules": selected}
        cfg = prefix / "pyvenv.cfg"
        evidence = {"status": "SELECTED17_IMPORTS_MATCH_PRIMARY13_CANDIDATE_ONLY",
            "source_files": sources, "root_preparation_go_sha256": args.root_preparation_go_sha256,
            "PID": os.getpid(), "PPID": os.getppid(), "process_group_ID": os.getpgrp(),
            "session_ID": os.getsid(0), "python_base_prefix": sys.base_prefix,
            "python_base_exec_prefix": sys.base_exec_prefix, "python_version_full": sys.version,
            "sys_path": list(sys.path), "sysconfig_paths": sysconfig.get_paths(),
            "pyvenv_cfg": {"path": str(cfg), "sha256": sha(cfg), "text": cfg.read_text()} if cfg.is_file() else None,
            "controlled_environment": {key: os.environ.get(key) for key in THREADS | NATIVE_ENV},
            "selected_module_count": len(selected), "primary_source_count": len(primary),
            "import_seconds": imports_finished - imports_started,
            "entry_through_pre_evidence_seconds": time.perf_counter() - entered,
            "process_cost": process_cost(), "call_guard": guard.evidence(),
            "alias_class_and_method_bindings": bindings,
            "explicit_PDE_calls_by_probe": 0, "checkpoint_workers_started": 0,
            "native_execution_authorized": False, "native_use_review_clear": False,
            "limitations": "Selected17 origins are not full-runtime/binary attestation. Imports load native extensions and can probe CPU/compiler/cache. The conservative Python guard is not a sandbox and does not measure all native calls. No numerical agreement or physics claim."}
        write(output / "candidate_environment.json", receipt)
        write(output / "identity_evidence.json", evidence)
        return 0
    except BaseException as error:
        write(output / "failure.json", {"status": "IMPORT_ONLY_PROBE_FAILED_NO_APPROVAL",
            "exception_type": type(error).__name__, "message": str(error), "call_guard": guard.evidence(),
            "unknown_partial_import_initialization_work": True, "native_use_review_clear": False,
            "explicit_PDE_calls_by_probe": 0, "process_cost": process_cost(),
            "entry_through_failure_seconds": time.perf_counter() - entered})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-preparation-go", type=Path)
    parser.add_argument("--root-preparation-go-sha256")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--pins", type=Path)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "HELD_IMPORT_ONLY_PROBE_NO_TARGET_IMPORT_NO_FILES",
            "runtime_install_authorized": False, "native_use_review_clear": False,
            "PDE_calls": 0, "target_module_imports": 0}))
        return 0
    if any(getattr(args, name) is None for name in ("root_preparation_go", "root_preparation_go_sha256",
            "protocol", "pins", "prefix", "output")):
        raise ValueError("explicit source-bound preparation authorization and paths required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
