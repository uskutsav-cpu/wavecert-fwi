"""AI-authored provisional software verification helpers, without a study runner.

These engineering helpers are archived for review. No execution CLI or project
study loop is provided while student research ownership is unresolved. They do
not specify the student final research question, hypothesis or design.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib
import io
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import RegularGridInterpolator

from wavecert.physics.exact_merit import ExactMeritOracle
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.provenance import source_digest

from . import adapter

PROTOCOL_SHA256 = "fb6b304fd719441be8cd6eb8d101ca864d0856254f40618583d8d986fd1c945c"
METHODS = adapter.METHODS
MODULES = {
    "wavecert.physics.exact_merit": "physics/exact_merit.py",
    "wavecert.physics.coarse_dst_reference": "physics/coarse_dst_reference.py",
    "wavecert.certificates.cached_merit_certificate": "certificates/cached_merit_certificate.py",
    "wavecert.optimization.paid_descent": "optimization/paid_descent.py",
    "wavecert.physics.helmholtz": "physics/helmholtz.py",
    "wavecert.physics.coarse_stability": "physics/coarse_stability.py",
    "wavecert.physics.coarse_batch": "physics/coarse_batch.py",
}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read_protocol(path):
    raw = Path(path).read_bytes()
    if _sha(raw) != PROTOCOL_SHA256:
        raise ValueError("provisional software protocol bytes changed; amendment/review required")
    protocol = json.loads(raw)
    if (protocol["methods"] != list(METHODS) or protocol["project_execution_authorized"]
            or protocol["student_final_research_design"] or protocol["competition_evidence"]):
        raise ValueError("software-only scope/controls changed")
    return protocol, raw


def _new_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def module_receipt(root):
    package = (Path(root)/"src/wavecert").resolve()
    hashes = {}
    for name, relative in MODULES.items():
        module = importlib.import_module(name)
        expected = package/relative
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError("foreign cached benchmark module:"+name)
        hashes[name] = _sha(expected.read_bytes())
    # Actual executing implementation paths are also checked, not only sibling
    # bytes or class names in a mixed-checkout interpreter.
    functions = (ExactMeritOracle.evaluate, Helmholtz2D.operator,
        adapter.CoarseDSTEngine.prepare, adapter.FastCoarseStability.evaluate,
        adapter.fixed_sweep_missing_fields, adapter.evaluate_cached_merit,
        adapter.run_paid_descent, adapter.run_exact_lbfgsb)
    expected_paths = {package/relative for relative in MODULES.values()}
    for function in functions:
        raw = getattr(function, "__wrapped__", function)
        if Path(raw.__code__.co_filename).resolve() not in expected_paths:
            raise RuntimeError("foreign executing implementation in benchmark")
    return hashes


def source_receipt(root, protocol_raw):
    root = Path(root).resolve()
    return dict(protocol_sha256=_sha(protocol_raw), repository_source_sha256=source_digest(root),
        backend_modules=module_receipt(root),
        adapter_sha256=_sha(Path(adapter.__file__).read_bytes()),
        verification_helpers_sha256=_sha(Path(__file__).read_bytes()),
        scope="AI-authored provisional SOFTWARE benchmark only; not execution authorization or student final design")


def verify_freeze(root, protocol_raw, receipt):
    if source_receipt(root, protocol_raw) != receipt:
        raise ValueError("source/protocol/backend freeze changed")


def scheduled_runs(protocol):
    rng = np.random.default_rng(protocol["order_seed"])
    schedule = []
    for repetition in range(protocol["repetitions"]):
        for case in range(len(protocol["identities"])):
            for position, method in enumerate(rng.permutation(METHODS).tolist()):
                schedule.append(dict(run_index=len(schedule), repetition=repetition,
                    case_index=case, order_position=position, method=method))
    return schedule


def load_raw_inputs(root, protocol):
    """Read only pinned development models/metadata; never inspect heldout arrays."""
    payloads = {}
    for relative, expected in protocol["inputs_sha256"].items():
        path = (Path(root)/relative).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError("input escapes repository")
        raw = path.read_bytes()
        if _sha(raw) != expected:
            raise ValueError("input SHA mismatch:"+relative)
        payloads[relative] = raw
    prefix = "data/reference/subsurfacegen_parent_split/"
    metadata = list(csv.DictReader(io.StringIO(payloads[prefix+"validation_metadata.csv"].decode())))
    manifest = list(csv.DictReader(io.StringIO(payloads[prefix+"split_manifest.csv"].decode())))
    with np.load(io.BytesIO(payloads[prefix+"validation.npz"]), allow_pickle=False) as archive:
        models = archive["models"].copy()
        source_indices = archive["source_indices"].copy()
        frequencies = archive["frequencies_hz"].copy()
        receivers = archive["receiver_indices"].copy()
        shape = archive["shape"].copy()
        spacing = float(archive["spacing"])
    geometry = protocol["geometry"]
    if (models.shape != (192, 24, 24) or models.dtype != np.float32
            or np.any(~np.isfinite(models)) or np.any(models <= 0)
            or source_indices.shape != (192,) or frequencies.shape != (192,)
            or len(metadata) != 192 or shape.tolist() != [24, 24] or spacing != .05
            or not np.array_equal(receivers, geometry["original_receiver_indices"])):
        raise ValueError("raw development fixture schema/domain changed")
    cases = []
    pairs = {(s, f) for s in geometry["original_source_indices"] for f in geometry["frequencies_hz"]}
    for identity in protocol["identities"]:
        selected = [i for i, row in enumerate(metadata) if row["slice_id"] == identity["slice_id"]]
        if (len(selected) != 24 or selected[0] != identity["array_index"]
                or {(int(source_indices[i]), float(frequencies[i])) for i in selected} != pairs
                or any(metadata[i]["split"] != "validation" or metadata[i]["model_id"] != identity["parent_id"]
                    or metadata[i]["source_file"] != identity["source_file"]
                    or metadata[i]["source_sha256"] != identity["source_sha256"] for i in selected)
                or any(not np.array_equal(models[i], models[selected[0]]) for i in selected)):
            raise ValueError("development identity/block/model-repeat binding failed")
        rows = [r for r in manifest if r["slice_id"] == identity["slice_id"]]
        if len(rows) != 1 or rows[0]["split"] != "validation" or rows[0]["model_id"] != identity["parent_id"]:
            raise ValueError("development split manifest binding failed")
        cases.append(dict(identity=dict(identity), raw_model=models[selected[0]].astype(np.float64)))
    if {r["slice_id"] for r in metadata} != {i["slice_id"] for i in protocol["identities"]}:
        raise ValueError("unexpected/deleted development slice")
    return cases


def refined_velocity(raw_model, n):
    if np.asarray(raw_model).shape != (24, 24) or n != 49:
        raise ValueError("this provisional benchmark only refines24 to49")
    velocity = 1/np.sqrt(raw_model)
    coordinates = np.arange(26, dtype=float)
    nodes = np.arange(1, n+1)/(n+1)*25
    zz, xx = np.meshgrid(nodes, nodes, indexing="ij")
    return RegularGridInterpolator((coordinates, coordinates), np.pad(velocity, 1, mode="edge"))(
        np.stack((zz, xx), axis=-1))


def refined_index(index, n=49):
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < 576 or n != 49:
        raise ValueError("invalid original grid index")
    z, x = divmod(int(index), 24)
    return (2*(z+1)-1)*n+2*(x+1)-1


def prepare_case(case, protocol):
    """Shared offline data generation; no solver object is passed to a policy."""
    started = perf_counter()
    config = protocol["physics"]
    velocity = refined_velocity(case["raw_model"], 49)
    truth = (1/velocity**2).reshape(-1)
    initial = (1/ndi.gaussian_filter(velocity, sigma=.08/config["spacing"], mode="reflect")**2).reshape(-1)
    lower, upper = protocol["squared_slowness_bounds"]
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("fixed box excludes initial model; no clipping/resampling allowed")
    physics = Helmholtz2D(**{**config, "shape": tuple(config["shape"])}, factorization_cache_size=0)
    receivers = tuple(refined_index(i) for i in protocol["geometry"]["original_receiver_indices"])
    # An offline oracle independently computes noiseless data using a matrix RHS
    # per frequency. It is destroyed before any policy is constructed.
    sources = tuple(refined_index(i) for i in protocol["geometry"]["original_source_indices"])
    blocks = []
    for frequency in protocol["geometry"]["frequencies_hz"]:
        rhs = np.column_stack([physics.source(source) for source in sources])
        state = physics._solve(truth, rhs, frequency, category="forward_rhs")
        for column, source in enumerate(sources):
            blocks.append(SurveyBlock(source, frequency, state[list(receivers), column].copy(), f"{source}@{frequency}"))
    return dict(identity=case["identity"], initial=initial, truth=truth, blocks=tuple(blocks), receivers=receivers,
        preparation_costs=physics.cost_snapshot(), preparation_seconds=perf_counter()-started,
        raw_model_sha256=adapter.array_sha(case["raw_model"]), initial_sha256=adapter.array_sha(initial),
        truth_sha256=adapter.array_sha(truth))


def budget_endpoints(result, prepared, protocol):
    """No post-budget endpoint credit, including work finishing across a ladder cap."""
    endpoints = []
    initial_objective = result["initial_objective"]
    for budget in protocol["endpoint_budgets_seconds"]:
        eligible = [row for row in result["checkpoints"] if row["within_budget"] and row["elapsed_seconds"] <= budget]
        checkpoint = eligible[-1] if eligible else None
        model = prepared["initial"] if checkpoint is None else checkpoint["model"]
        endpoints.append(dict(budget_seconds=budget, model=model,
            model_sha256=adapter.array_sha(model), checkpoint_elapsed_seconds=None if checkpoint is None else checkpoint["elapsed_seconds"],
            completed_paid_checkpoint=checkpoint is not None,
            online_objective=None if checkpoint is None else checkpoint["objective"],
            relative_online_objective=None if checkpoint is None or initial_objective is None or initial_objective <= 0
                else checkpoint["objective"]/initial_objective))
    targets = []
    for ratio in protocol["reported_objective_ratios"]:
        eligible = [] if initial_objective is None else [row for row in result["checkpoints"]
            if row["within_budget"] and row["objective"] <= ratio*initial_objective]
        first = eligible[0] if eligible else None
        targets.append(dict(objective_ratio=ratio, reached=first is not None,
            elapsed_seconds=None if first is None else first["elapsed_seconds"],
            model_sha256=None if first is None else first["model_sha256"]))
    return endpoints, targets


def offline_audit(result, prepared, protocol, endpoints):
    """Fresh LU endpoint/DD diagnostics after the clock; never policy inputs."""
    started = perf_counter()
    config = protocol["physics"]
    physics = Helmholtz2D(**{**config, "shape": tuple(config["shape"])})
    oracle = ExactMeritOracle(physics, prepared["blocks"], prepared["receivers"])
    initial = oracle.evaluate(prepared["initial"])
    for endpoint in endpoints:
        receipt = oracle.evaluate(endpoint["model"])
        endpoint["offline_objective"] = receipt.objective
        endpoint["offline_objective_ratio"] = None if initial.objective <= 0 else receipt.objective/initial.objective
        endpoint["relative_velocity_error"] = float(np.linalg.norm(1/np.sqrt(endpoint["model"])-1/np.sqrt(prepared["truth"]))
            / np.linalg.norm(1/np.sqrt(prepared["truth"])))
    by_sha = {row["model_sha256"]: row["model"] for row in result["checkpoints"]}
    by_sha.setdefault(prepared["initial_sha256"], prepared["initial"])
    certificate_rows = []
    for step in result["steps"]:
        proposal = step.get("proposal")
        diagnostic = None if proposal is None else proposal.get("diagnostics")
        certificate = None if diagnostic is None else diagnostic.get("certificate")
        if certificate is None:
            continue
        row = dict(iteration=step["iteration"], model_sha256=step["model_sha256"],
            available=certificate["available"], certified_descent=certificate["certified_descent"],
            true_derivative=None, bound_covers=None, false_certification=None)
        direction = proposal.get("direction")
        if direction is not None:
            reference = oracle.gradient(oracle.evaluate(by_sha[step["model_sha256"]]))
            dd = float(reference.gradient@direction)
            row["true_derivative"] = dd
            if certificate["directional_error_bound"] is not None:
                center, bound = certificate["approximate_directional_derivative"], certificate["directional_error_bound"]
                tolerance = 1e-9*max(abs(dd), abs(center), bound, 1.)
                row["bound_covers"] = bool(abs(dd-center) <= bound+tolerance)
            row["false_certification"] = bool(certificate["certified_descent"] and dd >= 0)
        certificate_rows.append(row)
    return dict(seconds=perf_counter()-started, costs=oracle.cost_snapshot(),
        certificate_rows=certificate_rows, initial_objective=initial.objective,
        policy_access=False, scope="offline numerical diagnostics only; ordinary FP tolerance, no enclosure")


def compact_trajectory(result):
    """Retain ALL costs/time/Phi/hash events without duplicate dense vector lists."""
    def compact(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("object arrays cannot be stored as an unambiguous numeric receipt")
            canonical = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<"))
            return dict(array_shape=list(value.shape), array_dtype=canonical.dtype.str,
                sha256=_sha(canonical.tobytes()), dense_values_saved=False)
        if isinstance(value, dict):
            return {str(key): compact(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [compact(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value
    return compact(result)


def _save_run(output, index, result, endpoints, targets, audit):
    name = f"run_{index:03d}"
    paths = [output/(name+suffix) for suffix in ("_models.npz", "_models.tmp.npz", ".json.gz", ".tmp.json.gz")]
    if any(path.exists() for path in paths):
        raise ValueError("software receipt already exists; overwriting is forbidden")
    models = {"initial": result["checkpoints"][0]["model"]} if result["checkpoints"] else {}
    models.update({f"budget_{i}": row["model"] for i, row in enumerate(endpoints)})
    models["final"] = result["final_model"]
    npz = output/(name+"_models.npz")
    temporary = output/(name+"_models.tmp.npz")
    np.savez_compressed(temporary, **models)
    temporary.replace(npz)
    record = dict(trajectory=compact_trajectory(result), endpoints=compact_trajectory(endpoints),
        targets=targets, offline_audit=audit, models_artifact=npz.name, models_sha256=_sha(npz.read_bytes()))
    temporary = output/(name+".tmp.json.gz")
    with gzip.open(temporary, "wt") as stream:
        json.dump(record, stream, allow_nan=False, separators=(",", ":"))
    temporary.replace(output/(name+".json.gz"))
