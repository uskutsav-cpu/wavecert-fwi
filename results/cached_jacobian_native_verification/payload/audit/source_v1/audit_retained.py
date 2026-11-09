"""Prospective stdlib-only cached-J retained-array audit; default metadata only.

No NumPy/SciPy/producer imports, solver, installation, GO setter or network.
The source-derived criteria must be frozen before an actual packet is admitted.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import re
import stat
import struct
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_PINS = {
    "verify_native_fields.py": "c660a025badbebd5e9d9a4d6ddbfc684c95193c6111f2de349f25112976d1d6b",
    "protocol.json": "3b608f942f3d08f40705e8203375fc7e24763592bd27d7cc70bac12b351391e6",
    "FROZEN_PRODUCTION_SOURCE.json": "e7ba5deef7a651f42d02b9f8e0bcbee9624291b8217ab378bdde7586c57cd4de",
    "cached_jacobian.py": "f613370617d7e0bd7815cf441dd8ba5f41816ad4791275e86f33de228657c79b",
    "public_map_builder.py": "622a624f54b17d9d882da9d88af547c516b26ee42c87d80ec68d5fe99a687ec1",
}
FREQUENCIES = (.25, .5, 1., 2., 2.5, 3., 3.5, 4.)
SOURCES = (250, 258, 266, 272, 280, 288)
RECEIVERS = tuple(range(250, 289, 2))
BASELINE = "2918a47c68ac8bee3d33e532c21a96d335d1718a"
REPOSITORY = "uskutsav-cpu/wavecert-fwi"
BRANCH = "research-cached-j-native-01"
MODEL_N, COMPLEX_N, REAL_N = 2401, 960, 1920
MAX_JSON = 8*1024**2
MAX_ARCHIVE, MAX_PAYLOAD, MAX_PACKET = 16*1024**2, 8*1024**2, 32*1024**2
SCHEMA = {
    "m0": ((2401,), "<f8"), "observation_model": ((2401,), "<f8"),
    "observations": ((48, 20), "<c16"), "direction": ((2401,), "<f8"),
    "dual": ((1920,), "<f8"), "initial_residual": ((1920,), "<f8"),
    "cached_jvp": ((1920,), "<f8"), "reference_jvp": ((1920,), "<f8"),
    "cached_vjp": ((2401,), "<f8"), "reference_vjp": ((2401,), "<f8"),
    "fd_residual_vectors": ((4, 1920), "<f8"), "native_U": ((8, 2401, 6), "<c16"),
    "reference_C_covered": ((8, 6, 2401), "<c16"), "complex_U": ((8, 2401, 6), "<c16"),
}
PHYSICS_KEYS = ("factorizations", "forward_rhs", "tangent_rhs", "adjoint_rhs", "stability_rhs")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_constant(value):
    raise ValueError("nonfinite JSON constant: "+value)


def read_json(path):
    if path.stat().st_size > MAX_JSON:
        raise ValueError("JSON exceeds8MiB")
    return json.loads(path.read_bytes(), parse_constant=reject_constant)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def count(value):
    return type(value) is int and value >= 0


def digest_text(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def safe_file(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("invalid evidence path")
    path = root/name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("missing/escaping/symlink evidence: "+name)
    return path


class Checks:
    def __init__(self):
        self.count = self.numeric = 0
        self.issues = []

    def require(self, condition, label, detail=None):
        self.count += 1
        if not condition:
            self.issues.append(dict(check=label, detail=detail))
        return bool(condition)

    def scalar(self, actual, expected, rtol, atol, label):
        self.numeric += 1
        ok = number(actual) and number(expected) and abs(actual-expected) <= atol+rtol*abs(expected)
        return self.require(ok, label, dict(independent=actual, recorded=expected))


def decode_npy(raw, shape, dtype):
    """Adapted bounded literal-header reader; no pickle/NumPy or implicit casts."""
    if len(raw) < 10 or raw[:6] != b"\x93NUMPY":
        raise ValueError("NPY magic/truncation")
    version = tuple(raw[6:8])
    if version == (1, 0):
        length, start = struct.unpack("<H", raw[8:10])[0], 10
    elif version == (2, 0) and len(raw) >= 12:
        length, start = struct.unpack("<I", raw[8:12])[0], 12
    else:
        raise ValueError("bounded NPYv1/v2 required")
    if not 0 < length <= 4096 or start+length > len(raw):
        raise ValueError("NPY header extent")
    header = ast.literal_eval(raw[start:start+length].decode("latin1").strip())
    if (not isinstance(header, dict) or set(header) != {"descr", "shape", "fortran_order"}
            or dtype not in {"<f8", "<c16"} or header["descr"] != dtype
            or header["fortran_order"] is not False or not isinstance(header["shape"], tuple)
            or any(type(n) is not int or n <= 0 for n in header["shape"])
            or header["shape"] != tuple(shape)):
        raise ValueError("NPY exact dtype/order/shape")
    body, width = raw[start+length:], 8 if dtype == "<f8" else 16
    if len(body) != width*math.prod(shape) or len(body) > MAX_PAYLOAD:
        raise ValueError("NPY exact bounded payload")
    if dtype == "<f8":
        values = tuple(v[0] for v in struct.iter_unpack("<d", body))
        finite = all(math.isfinite(v) for v in values)
    else:
        values = tuple(complex(a, b) for a, b in struct.iter_unpack("<dd", body))
        finite = all(math.isfinite(v.real) and math.isfinite(v.imag) for v in values)
    if not finite:
        raise ValueError("nonfinite retained array")
    return dict(shape=tuple(shape), dtype=dtype, values=values, bytes=len(body), data_sha256=hashlib.sha256(body).hexdigest())


def decode_npz(path, schema=SCHEMA):
    if path.stat().st_size > MAX_ARCHIVE:
        raise ValueError("NPZ exceeds16MiB")
    names = {name+".npy": spec for name, spec in schema.items()}
    answer, payload = {}, 0
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
        entries = archive.infolist()
        if len(entries) != len(names) or {e.filename for e in entries} != set(names):
            raise ValueError("exact unique NPZ member denominator")
        for entry in entries:
            shape, dtype = names[entry.filename]
            maximum = (8 if dtype == "<f8" else 16)*math.prod(shape)+4108
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if (entry.flag_bits & 1 or entry.compress_type != zipfile.ZIP_STORED
                    or mode not in (0, stat.S_IFREG) or entry.file_size > maximum):
                raise ValueError("NPZ unsafe member/encryption/compression/extent")
            with archive.open(entry) as stream:
                raw = stream.read(maximum+1)
                if len(raw) > maximum or stream.read(1) or len(raw) != entry.file_size:
                    raise ValueError("actual NPZ read differs from bounded extent")
            value = decode_npy(raw, shape, dtype)
            payload += value["bytes"]
            if payload > MAX_PAYLOAD:
                raise ValueError("aggregate retained payload exceeds8MiB")
            answer[entry.filename[:-4]] = value
    return answer


def norm(values):
    return math.hypot(*(abs(x) for x in values))


def real_dot(left, right):
    return math.fsum(a*b for a, b in zip(left, right, strict=True))


def relative(left, right):
    if len(left) != len(right):
        raise ValueError("vector length")
    denominator = norm(right)
    if not math.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError("uninformative vector")
    return norm(tuple(a-b for a, b in zip(left, right, strict=True)))/denominator


def check_vector(left, right, tolerance, checks, label):
    checks.numeric += 1
    value = relative(left, right)
    checks.require(math.isfinite(value) and value <= tolerance, label, value)
    return value


def centered_fd(minus, plus, step):
    return tuple((b-a)/(2*step) for a, b in zip(minus, plus, strict=True))


def fd_objective(minus, plus, step):
    return (.5*real_dot(plus, plus)-.5*real_dot(minus, minus))/(2*step)


def check_counts(rows, expected, checks):
    phases = {"observation_generation": (8, 48, 0, 0, 0), "policy_template": (0, 0, 0, 0, 0),
        "public_jvp": (8, 48, 48, 0, 0), "public_vjp": (8, 48, 0, 48, 0),
        "complex_source_validation": (8, 48, 0, 0, 0), "full_receiver_map_validation": (8, 0, 0, 0, 160),
        "actual_native_policy": (16, 96, 0, 0, 0), "actual_foreign_native_validation": (0, 0, 0, 0, 0),
        "foreign_template": (0, 0, 0, 0, 0), "component_cold_map_batches": (8, 0, 0, 0, 112)}
    phases.update({f"fd_{step}_{sign}": (8, 48, 0, 0, 0) for step in (1e-4, 5e-5) for sign in (-1, 1)})
    checks.require(set(rows) == set(phases), "costs/exact14_disjoint_owner_roles")
    totals = dict.fromkeys(PHYSICS_KEYS, 0)
    for name, target in phases.items():
        row = rows.get(name, {})
        valid = isinstance(row, dict) and all(isinstance(k, str) and count(v) for k, v in row.items())
        checks.require(valid and all(row.get(k) == v for k, v in zip(PHYSICS_KEYS, target, strict=True)),
            "costs/actual_phase/"+name)
        if valid:
            for k in PHYSICS_KEYS:
                totals[k] += row.get(k, 0)
    checks.require(totals == expected, "costs/final_snapshots_once_not_setup_plus_final", totals)
    return totals


def expected_dispatch_labels():
    labels = ["observation_generation"]*8+["policy"]*4
    labels += [label for _ in range(48) for label in ("public_jvp", "public_vjp")]
    labels += ["complex_source_validation"]*8+["full_receiver_map_validation"]*8
    labels += [f"fd_{step}_{sign}" for step in (1e-4, 5e-5) for sign in (-1, 1) for _ in range(8)]
    return labels+["policy", "policy", "expected_model_rejection", "foreign_owner_validation"]


def numerical_audit(arrays, result, protocol, checks):
    c = protocol["criteria_pending_independent_review"]
    a = {key: record["values"] for key, record in arrays.items()}
    header = result["native_initial_header"]
    phi0 = header["objective"]
    checks.require(number(phi0) and phi0 > 0, "objective/paid_initialPhi0_positive")
    if not number(phi0) or phi0 <= 0:
        return {}
    for name in ("m0", "observation_model"):
        expected = 1/protocol["initial_velocity" if name == "m0" else "observation_velocity"]**2
        checks.require(all(v == expected for v in a[name]), "input/constant_model/"+name)
    checks.require(arrays["m0"]["data_sha256"] == header["model_sha256"], "objective/actual_native_header_model_bytes")
    p_expected = tuple(.25*(1+.2*math.sin(2*math.pi*i/48)*math.cos(2*math.pi*j/48))/1.2
        for i in range(49) for j in range(49))
    check_vector(a["direction"], p_expected, c["relative_vector_error"], checks, "input/independent_fixed_direction")
    raw_dual = tuple(math.sin(.017*k)+math.cos(.013*k) for k in range(1, 1921))
    dual_expected = tuple(v/norm(raw_dual) for v in raw_dual)
    check_vector(a["dual"], dual_expected, c["relative_vector_error"], checks, "input/independent_fixed_dual")
    residual = tuple(a["native_U"][(f*MODEL_N+r)*6+s]-a["observations"][(f*6+s)*20+j]
        for f in range(8) for s in range(6) for j, r in enumerate(RECEIVERS))
    phi_own = .5*math.fsum(v.real*v.real+v.imag*v.imag for v in residual)
    checks.scalar(phi_own, phi0, c["normalized_objective_relative_error"], c["normalized_objective_absolute_error"],
        "objective/raw_U_minus_observations_full960Phi")
    scale = math.sqrt(phi0)
    own_f0 = tuple(v.real/scale for v in residual)+tuple(v.imag/scale for v in residual)
    check_vector(a["initial_residual"], own_f0, c["relative_vector_error"], checks, "objective/real_imag_block_order_fixed_normalization")
    initial_cost = .5*real_dot(a["initial_residual"], a["initial_residual"])
    checks.scalar(initial_cost, 1., c["normalized_objective_relative_error"], c["normalized_objective_absolute_error"], "objective/Phi_initial_over_fixedPhi0")
    errors = dict(jvp=check_vector(a["cached_jvp"], a["reference_jvp"], c["relative_vector_error"], checks, "vectors/cachedJVP_vs_public_redoU"),
        vjp=check_vector(a["cached_vjp"], a["reference_vjp"], c["relative_vector_error"], checks, "vectors/cachedJstar_vs_public_redoU"))
    left, right = real_dot(a["dual"], a["cached_jvp"]), real_dot(a["direction"], a["cached_vjp"])
    adjoint_error = abs(left-right)/max(abs(left), abs(right), norm(a["dual"])*norm(a["cached_jvp"]), 1e-12)
    checks.numeric += 1
    checks.require(adjoint_error <= c["relative_real_adjoint_error"], "real_adjoint/zJp_equals_pJstarz", adjoint_error)
    dd = real_dot(a["initial_residual"], a["cached_jvp"])
    cosine = abs(dd)/(norm(a["initial_residual"])*norm(a["cached_jvp"]))
    checks.require(cosine >= c["nontrivial_relative_dd_cosine"] and abs(dd) > c["nonzero_norm_floor"], "FD/both_steps_informative_fixed_direction")
    fd_rows = []
    for index, step in enumerate(protocol["fd_steps"]):
        minus = a["fd_residual_vectors"][(2*index)*REAL_N:(2*index+1)*REAL_N]
        plus = a["fd_residual_vectors"][(2*index+1)*REAL_N:(2*index+2)*REAL_N]
        j_error = check_vector(centered_fd(minus, plus, step), a["cached_jvp"], c["fd_jvp_relative_error_each_step"], checks, f"FD/centered_residual/epsilon{step}")
        derivative = fd_objective(minus, plus, step)
        objective_error = abs(derivative-dd)/max(abs(derivative), abs(dd), 1e-12)
        checks.numeric += 1
        checks.require(objective_error <= c["fd_objective_relative_error_each_step"], f"FD/centered_objective/epsilon{step}", objective_error)
        fd_rows.append(dict(step=step, jvp_relative_error=j_error, objective_relative_error=objective_error,
            independent_objective_FD=derivative, normalized_objective_DD=dd))
    amplitude = 1/protocol["physics"]["spacing"]**2
    multipliers = tuple(complex(*v) for v in protocol["complex_source_multipliers"])
    row_errors = []
    for f, frequency in enumerate(FREQUENCIES):
        for s, source in enumerate(SOURCES):
            covered = a["reference_C_covered"][(f*6+s)*MODEL_N:(f*6+s+1)*MODEL_N]
            native = tuple(a["native_U"][(f*MODEL_N+k)*6+s]/amplitude for k in range(MODEL_N))
            challenge = tuple(a["complex_U"][(f*MODEL_N+k)*6+s]/(multipliers[s]*amplitude) for k in range(MODEL_N))
            native_error = check_vector(native, covered, c["receiver_row_relative_error"], checks, f"rows/nativeU_plain_transpose/f{frequency}/source{source}")
            complex_error = check_vector(challenge, covered, c["receiver_row_relative_error"], checks, f"rows/complexU_divide_amplitude_no_conjugation/f{frequency}/source{source}")
            row_errors.append(dict(frequency_hz=frequency, source_index=source, native_relative_error=native_error, complex_source_relative_error=complex_error))
    checks.require(len(fd_rows) == 2 and len(row_errors) == 48, "numeric/full2FD_and48plus48_rows_no_selection")
    return dict(native_Phi0=phi0, scalar_recomputed_Phi0=phi_own, normalized_initial_cost=initial_cost,
        relative_vector_errors=errors, real_adjoint_relative_error=adjoint_error, left_zJp=left, right_pJstarz=right,
        fixed_objective_direction_cosine=cosine, FD=fd_rows, covered_rows=row_errors)


def mandatory_controls(packet, checks):
    summary = read_json(safe_file(packet, "mandatory58_controls.json"))
    text = safe_file(packet, "lifecycle14.log").read_text()
    lifecycle = json.loads(text.strip().splitlines()[-1], parse_constant=reject_constant)
    checks.require(summary.get("total_mandatory_tests") == 58 and summary.get("lifecycle14") == lifecycle,
        "controls/mandatory58_summary_matches_raw14")
    checks.require(lifecycle.get("status") == "MANUFACTURED_LIFECYCLE_PASS"
        and all(count(lifecycle.get(k)) for k in ("tests", "failures", "errors"))
        and (lifecycle.get("tests"), lifecycle.get("failures"), lifecycle.get("errors")) == (14, 0, 0)
        and lifecycle.get("source_files") == SOURCE_PINS
        and lifecycle.get("control_source_sha256") == "2526a76b539af952a9a9707f6723493ad571723012ed48c8e3738b8432a80414"
        and all(type(lifecycle.get(k)) is int and lifecycle[k] == 0 for k in ("numerical_imports", "native_owners_constructed", "PDE_calls")),
        "controls/actual14_source_count_no_numerics")
    names = re.findall(r"^(test_.*) \.\.\. ok$", text, re.MULTILINE)
    checks.require(len(names) == len(set(names)) == 14
        and "... skipped" not in text and "OK (skipped=" not in text, "controls/actual14_names_no_skips")
    xml_path = safe_file(packet, "component44.xml")
    if xml_path.stat().st_size > 1024**2:
        raise ValueError("44-test XML exceeds1MiB")
    xml = xml_path.read_bytes()
    if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
        raise ValueError("XML entities/DTD not admitted")
    root = ET.fromstring(xml)
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    checks.require(totals == dict(tests=44, failures=0, errors=0, skipped=0)
        and summary.get("component44") == totals and len(root.findall(".//testcase")) == 44
        and len({(row.get("classname"), row.get("name")) for row in root.findall(".//testcase")}) == 44
        and not root.findall(".//testcase/skipped") and not root.findall(".//testcase/failure") and not root.findall(".//testcase/error"),
        "controls/actual44_xml_no_failures_errors_skips")


def wait_check(row, label, checks):
    checks.require(row.get("label") == label and type(row.get("actual_wait_returncode")) is int
        and row["actual_wait_returncode"] == 0 and row.get("wait_observed") is True
        and row.get("requested_new_session") is True and row.get("launch_state") == "HANDLE_ASSIGNED"
        and row.get("terminal_state_known") is True and row.get("owned_process_group_alive_after_cleanup") is False
        and row.get("deferred_parent_signals") == []
        and count(row.get("actual_pid")) and row["actual_pid"] > 0
        and count(row.get("observed_pgid")) and row.get("observed_pgid") == row["actual_pid"],
        "actual_wait/known_successful_owned_terminal/"+label)
    checks.require(number(row.get("external_spawn_through_wait_wall_seconds"))
        and row["external_spawn_through_wait_wall_seconds"] >= 0, "actual_wait/real_duration/"+label)


def inventory_check(packet, inventory, checks):
    files = inventory.get("files", {})
    checks.require(isinstance(files, dict) and inventory.get("any_omitted") is False
        and inventory.get("inventory_excludes_itself") is True, "inventory/complete_stable_scope")
    actual = {p.relative_to(packet).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
        for p in packet.rglob("*") if p.is_file() and p.name != "artifact_inventory.json"}
    checks.require(set(actual) == set(files), "inventory/exact_file_membership")
    for name, recorded in files.items():
        path = safe_file(packet, name)
        checks.require(recorded.get("included") is True and count(recorded.get("bytes"))
            and recorded["bytes"] == path.stat().st_size and recorded.get("sha256") == sha(path), "inventory/bytes_and_sha/"+name)
    total = sum(row["bytes"] for row in actual.values())
    checks.require(inventory.get("copied_bytes") == total and inventory.get("artifact_cap_bytes") == MAX_PACKET
        and total+safe_file(packet, "artifact_inventory.json").stat().st_size <= MAX_PACKET, "inventory/actual_complete_byte_cap")
    return actual


def metadata_preflight(args, packet, protocol, checks):
    observation = read_json(args.external_observation)
    inventory_path = safe_file(packet, "artifact_inventory.json")
    route_path = safe_file(packet, "route_result.json")
    checks.require(sha(args.external_observation) == args.expected_external_sha256, "external/explicit_actual_observation_sha")
    checks.require(type(observation.get("exit_code")) is int and observation["exit_code"] == 0
        and observation.get("artifact_inventory_sha256") == sha(inventory_path)
        and observation.get("route_result_sha256") == sha(route_path)
        and observation.get("repository") == REPOSITORY and type(observation.get("run_attempt")) is int and observation.get("run_attempt") == 1,
        "external/actual_route0_and_exact_packet_binding")
    inventory = read_json(inventory_path)
    files = inventory_check(packet, inventory, checks)
    route = read_json(route_path)
    checks.require(route.get("status") == "NATIVE_CHILD_ACTUAL_EXIT0_SELF_REPORT_PRESENT_AUDIT_PENDING"
        and route.get("native_child_attempted") is True and route.get("mandatory_controls") == 58
        and route.get("scientific_goals_complete") is False and route.get("independent_numerical_audit_required") is True,
        "route/native_attempt_and_audit_pending_not_science")
    for label in ("lifecycle14", "component44", "native"):
        row = read_json(safe_file(packet, label+"_actual_wait.json"))
        wait_check(row, label, checks)
        if label == "native":
            checks.require(route.get("actual_native_wait") == row, "route/exact_native_actual_wait_record")
    mandatory_controls(packet, checks)
    before, after = (read_json(safe_file(packet, "source_"+suffix+".json")) for suffix in ("before", "after"))
    frozen = read_json(HERE/"FROZEN_PRODUCTION_SOURCE.json")
    checks.require(before == after and before.get("baseline_source_sha256") == frozen["source_sha256"]
        and before.get("baseline_files") == frozen["files"] and len(frozen["files"]) == 161,
        "source/exact161_frozen9f_before_after")
    checks.require(before.get("route_manifest_sha256") == args.expected_route_manifest_sha256
        and before.get("baseline_actual_head") == BASELINE and before.get("experiment_actual_head") == observation.get("experiment_commit")
        and before.get("experiment_worktree_status") == before.get("baseline_worktree_status") == "", "source/route_manifest_and_clean_exact_heads")
    route_sources = before.get("route_files", {})
    checks.require(route_sources.get("experiments/cached_j_native_01/run_route.py", {}).get("sha256") == args.expected_route_source_sha256,
        "source/exact_reviewed_final_route_not_held7001")
    for name, expected in SOURCE_PINS.items():
        checks.require(route_sources.get("experiments/cached_j_native_01/runtime/"+name, {}).get("sha256") == expected,
            "source/runtime_component/"+name)
    checks.require(route_sources.get("experiments/cached_j_native_01/runtime/test_lifecycle.py", {}).get("sha256") == "2526a76b539af952a9a9707f6723493ad571723012ed48c8e3738b8432a80414"
        and route_sources.get("experiments/cached_j_native_01/runtime/test_manufactured.py", {}).get("sha256") == "c7d7d9d9e152b4b0c037b5cd2efc411356d2c1206b5ce3824ed8a351acda3104", "source/actual_mandatory14_and44_sources")
    env = read_json(safe_file(packet, "actual_github_context.json"))
    event = read_json(safe_file(packet, "actual_github_event.json"))
    head = observation.get("experiment_commit")
    checks.require(env.get("GITHUB_EVENT_NAME") == "push" and env.get("GITHUB_REPOSITORY") == REPOSITORY
        and env.get("GITHUB_REF") == "refs/heads/"+BRANCH and env.get("GITHUB_RUN_ATTEMPT") == "1"
        and env.get("GITHUB_SHA") == env.get("GITHUB_WORKFLOW_SHA") == head
        and env.get("GITHUB_WORKFLOW_REF") == REPOSITORY+"/.github/workflows/cached-j-native.yml@refs/heads/"+BRANCH,
        "context/actual_first_push_attempt1_workflow")
    checks.require(event.get("ref") == "refs/heads/"+BRANCH and event.get("created") is True
        and event.get("forced") is False and event.get("deleted") is False and event.get("before") == "0"*40
        and event.get("after") == event.get("head_commit", {}).get("id") == head
        and event.get("repository", {}).get("full_name") == REPOSITORY and event["repository"].get("private") is False,
        "context/public_nonforced_branch_creation")
    runtime = read_json(safe_file(packet, "actual_runtime.json"))
    checks.require(runtime.get("packages") == dict(numpy="2.4.6", scipy="1.17.1", pytest="9.1.1")
        and isinstance(runtime.get("python"), str) and runtime["python"].startswith("3.11.18")
        and isinstance(runtime.get("platform"), str) and runtime["platform"].startswith("Linux")
        and runtime.get("numerical_imports_in_controller") == 0, "context/actual_fixed_runtime")
    registration = read_json(safe_file(packet, "actual_registration.json"))
    go = read_json(safe_file(packet, "supplied_root_go.json"))
    checks.require(registration.get("repository") == REPOSITORY and registration.get("branch") == BRANCH
        and registration.get("single_parent") == BASELINE and registration.get("native_execution_registered") is True
        and registration.get("root_go") == go and registration.get("route_manifest_sha256") == args.expected_route_manifest_sha256,
        "admission/actual_registered_conditional_native_GO")
    checks.require(go.get("authorization") == protocol["root_go_authorization"]
        and go.get("execution_authorized") is True and go.get("independent_source_review_clear") is True
        and go.get("criteria_frozen_before_execution") is True and go.get("source_files") == SOURCE_PINS
        and go.get("route_manifest_sha256") == args.expected_route_manifest_sha256, "admission/explicit_source_bound_GO_not_created_by_auditor")
    admission = read_json(safe_file(packet, "native/admission.json"))
    checks.require(admission.get("root_go") == go and admission.get("root_go_sha256") == sha(safe_file(packet, "supplied_root_go.json"))
        and admission.get("source_files") == SOURCE_PINS and admission.get("production_source_sha256") == frozen["source_sha256"], "native/exact_admission_source_GO")
    checks.require(read_json(safe_file(packet, "native/plan.json")) == protocol, "native/exact_fixed_plan")
    checks.require("native/failure.json" not in files and "route_failure.json" not in files
        and "UNSTABLE_CHILD_PREFIX_ONLY.json" not in files, "native/no_failure_or_unstable_inventory")
    result_path = safe_file(packet, "native/result.json")
    checks.require(sha(result_path) == args.expected_result_sha256, "native/explicit_result_sha")
    result = read_json(result_path)
    checks.require(result.get("status") == "NATIVE_FIELD_INTEGRATION_CHECKS_PASS_PENDING_EXTERNAL_WAIT"
        and result.get("source_files") == SOURCE_PINS and result.get("actual_external_wait_required") is True
        and result.get("scientific_goals_complete") is False, "native/completed_source_bound_worker_report")
    artifacts = result.get("output_artifacts", {})
    native_files = {name.removeprefix("native/"): row["sha256"] for name, row in files.items()
        if name.startswith("native/") and name != "native/result.json"}
    checks.require(artifacts == native_files and set(artifacts) == {"plan.json", "admission.json", "events.jsonl", "arrays.npz"}, "native/exact_pre_result_artifact_inventory")
    checks.require(count(result.get("observed_peak_RSS_bytes")) and result["observed_peak_RSS_bytes"] <= 1024**3,
        "native/self_peak_before_final_serialization_scope_only")
    return result, files


def ledger_check(packet, result, protocol, checks):
    labels = expected_dispatch_labels()
    evidence = result["actual_public_attempt_prefixes"]
    calls = evidence["calls"]
    checks.require(len(calls) == len(labels) == 160, "events/all160_public_dispatches_retained_not58_tests")
    for index, label in enumerate(labels):
        if index >= len(calls):
            break
        call = calls[index]
        guard = index >= len(labels)-2
        checks.require(count(call.get("index")) and call.get("index") == index and call.get("label") == label
            and call.get("status") == ("raised" if guard else "returned"), f"events/call{index}/identity_status")
        checks.require(call.get("before", {}).get("unknown_after_prefix") is False
            and call.get("after", {}).get("unknown_after_prefix") is False, f"events/call{index}/known_snapshots")
        if guard:
            checks.require(call.get("exception_type") == "ValueError", f"events/call{index}/expected_source_guard_only")
    events = [json.loads(line, parse_constant=reject_constant) for line in safe_file(packet, "native/events.jsonl").read_text().splitlines()]
    kinds = Counter(e.get("kind") for e in events)
    checks.require(events[0].get("kind") == "full_plan_and_admission_persisted_before_numerical_imports"
        and kinds == Counter(full_plan_and_admission_persisted_before_numerical_imports=1,
            public_dispatch_attempt=160, public_dispatch_returned=158, public_dispatch_raised=2,
            all_synthetic_observations_complete_before_policy=1, expected_source_bound_model_guard_rejection=1,
            expected_source_bound_owner_guard_rejection=1), "events/full_durable_event_denominators")
    attempts = [e for e in events if e.get("kind") == "public_dispatch_attempt"]
    endings = [e for e in events if e.get("kind") in {"public_dispatch_returned", "public_dispatch_raised"}]
    checks.require([e.get("index") for e in attempts] == list(range(160))
        and [e.get("label") for e in attempts] == labels and [e.get("index") for e in endings] == list(range(160)), "events/ordered_attempt_return_pairs")
    age = -1.
    for e in events:
        checks.require(number(e.get("elapsed_seconds")) and e["elapsed_seconds"] >= age, "events/monotone_nonadditive_global_ages")
        if number(e.get("elapsed_seconds")):
            age = e["elapsed_seconds"]
    native_checks = result["checks"]
    checks.require(isinstance(native_checks, list) and all(row.get("pass") is True for row in native_checks), "native/all_retained_dynamic_checks_pass")
    names = Counter(row["check"] for row in native_checks)
    for label, multiplicity in {"actual native transpose symmetry measured": 8,
            "fixed centered nativePhi FD": 2, "fixed centered residual FD relative error": 2,
            "native U.T/a covered rows relative error": 8, "complex U.T/a with no conjugation relative error": 8,
            "old field operator fixed after new native trial": 1,
            "unissued model query rejected before new physics work": 1,
            "foreign export rejected without physics solve": 1}.items():
        checks.require(names[label] == multiplicity, "native/exact_required_check_multiplicity/"+label)
    for label in ("observation_generation", "complex_source_validation", "fd_0.0001_-1", "fd_0.0001_1", "fd_5e-05_-1", "fd_5e-05_1"):
        checks.require(names[label+" finite source fields"] == 8, "native/all8_forward_field_checks/"+label)
    component = result["component_costs"]
    checks.require(component["native"]["unknown_since_prefix"] is False
        and component["map_builder"]["unknown_work_present"] is False, "costs/completed_component_not_unknown")
    batches = component["map_builder"]["cold_owner_batches"]
    missing = [r for r in RECEIVERS if r not in SOURCES]
    checks.require(len(batches) == 8, "costs/all8_separate_missing14_receiver_batches")
    header = result["native_initial_header"]
    identity = [header[k] for k in ("key", "model_sha256", "operator_sha256", "geometry_sha256", "data_sha256")]
    for index, row in enumerate(batches[:8]):
        checks.require(row.get("index") == index and row.get("frequency_hz") == FREQUENCIES[index]
            and row.get("receivers") == missing and row.get("point_identity") == identity
            and row.get("unknown_work") is False and row.get("actual_complete_batch_counts_verified") is True
            and row.get("cold_owner_verified") is True and row.get("last_known_physics_prefix", {}).get("factorizations") == 1
            and row["last_known_physics_prefix"].get("stability_rhs") == 14, f"costs/mapbatch{index}/context_and_actual_disjoint_work")
    rows = result["actual_disjoint_physics_counts"]
    totals = check_counts(rows, protocol["expected_complete_disjoint_counts"], checks)
    checks.require(totals == result.get("total_actual_physics_counts")
        and component["native"]["last_known_cumulative"]["physics"] == rows["actual_native_policy"], "costs/actual_native_cumulative_once_binding")
    return dict(actual_dispatches=len(calls), expected_guard_rejections=2,
        mandatory_controls=58, recorded_dynamic_check_count=len(native_checks), disjoint_totals=totals)


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def evaluate(args):
    """Only explicit future terminal/audit GO can reach retained packet reads."""
    checks = Checks()
    packet, output = args.packet.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(packet) or output.is_relative_to(HERE):
        raise ValueError("fresh output outside input packet/frozen auditor required")
    if sha(Path(__file__)) != args.expected_source_sha256 or sha(HERE/"criteria.json") != args.expected_criteria_sha256:
        raise ValueError("auditor/criteria changed after review")
    if sha(HERE/"native_protocol.json") != SOURCE_PINS["protocol.json"] or sha(HERE/"FROZEN_PRODUCTION_SOURCE.json") != SOURCE_PINS["FROZEN_PRODUCTION_SOURCE.json"]:
        raise ValueError("copied source context changed")
    protocol = read_json(HERE/"native_protocol.json")
    criteria = read_json(HERE/"criteria.json")
    if criteria["source_pins"] != SOURCE_PINS or criteria["criteria_unchanged_from_native_protocol"] != protocol["criteria_pending_independent_review"]:
        raise ValueError("fixed source/criteria context differs")
    report = dict(status="ADMISSION_PENDING", input_packet=str(packet), npz_arrays_decoded=0,
        source_sha256=sha(Path(__file__)), criteria_sha256=sha(HERE/"criteria.json"), source_pins=SOURCE_PINS,
        planned_array_members=14, planned_FD_steps=2, planned_native_covered_rows=48,
        planned_complex_source_covered_rows=48, planned_public_dispatches=160,
        mandatory_controls=58, scalar_results={}, ledger={}, verified_credit=False,
        physical_acceptance=None, scientific_goals_complete=False, student_final_design=False)
    before = {p.relative_to(packet).as_posix(): sha(p) for p in packet.rglob("*") if p.is_file()}
    try:
        result, _files = metadata_preflight(args, packet, protocol, checks)
        if not checks.issues:
            report["ledger"] = ledger_check(packet, result, protocol, checks)
        if not checks.issues:
            arrays = decode_npz(safe_file(packet, "native/arrays.npz"))
            report["npz_arrays_decoded"] = len(arrays)
            metadata = result.get("array_metadata", {})
            checks.require(set(metadata) == set(SCHEMA), "arrays/exact14_declared_metadata")
            for name, array in arrays.items():
                recorded = metadata.get(name, {})
                checks.require(recorded.get("shape") == list(array["shape"])
                    and recorded.get("dtype") == ("float64" if array["dtype"] == "<f8" else "complex128")
                    and recorded.get("bytes") == array["bytes"] and recorded.get("sha256") == array["data_sha256"],
                    "arrays/actual_shape_dtype_bytes_payloadSHA/"+name)
            if not checks.issues:
                report["scalar_results"] = numerical_audit(arrays, result, protocol, checks)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, SyntaxError, OverflowError, struct.error, zipfile.BadZipFile, ET.ParseError) as exc:
        checks.require(False, "missing_invalid_or_inconsistent_retained_evidence", f"{type(exc).__name__}: {exc}")
    after = {p.relative_to(packet).as_posix(): sha(p) for p in packet.rglob("*") if p.is_file()}
    checks.require(before == after, "read_only/all_packet_bytes_unchanged")
    report.update(status="RETAINED_ARRAY_SCALAR_AUDIT_CLEAR" if not checks.issues else "INVALID_OR_INCOMPLETE_NO_CREDIT",
        verified_credit=not checks.issues, checks=checks.count, numeric_comparisons=checks.numeric,
        issues=checks.issues, files_before=before, files_after=after,
        limits=[
            "Stdlib scalar/array-only recomputation; no native/PDE/NumPy/producer import or full-J reconstruction.",
            "Only six covered rows per frequency are retained; missing14 receiver-map rows are not independently replayed.",
            "Old-operator repeat vector and native receipt identities are not reconstructible from arrays; source-bound checks/events/counters retain those controls.",
            "Mandatory58 means actual14 lifecycle plus44 manufactured component tests, not58 numerical conditions or160 native dispatches.",
            "Actual source/context/wait evidence is externally recorded; self-reported fields are not independent OS/remote authenticity proof.",
            "Worker selfRSS is measured before final serialization; parent RUSAGE_CHILDREN includes controls and must not be assigned solely to native.",
            "Ordinary floating-point scalar agreement at one fixed constant model/direction/dual, not physical accuracy, rigorous enclosure, recovery, speedup, novelty or student work."])
    output.mkdir()
    (output/"report.json").write_text(json.dumps(clean(report), sort_keys=True, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status=report["status"], checks=checks.count, numeric_comparisons=checks.numeric,
        issues=len(checks.issues), npz_arrays_decoded=report["npz_arrays_decoded"], verified_credit=report["verified_credit"])))
    return 0 if not checks.issues else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-after-external-terminal", action="store_true")
    for name in ("packet", "external-observation", "output"):
        parser.add_argument("--"+name, type=Path)
    for name in ("source", "criteria", "external", "result", "route-source", "route-manifest"):
        parser.add_argument("--expected-"+name+"-sha256")
    args = parser.parse_args(argv)
    if not args.execute_after_external_terminal:
        print(json.dumps(dict(status="PROSPECTIVE_METADATA_ONLY_NO_PACKET_OR_NUMERICAL_READ", source_sha256=sha(Path(__file__)),
            criteria_sha256=sha(HERE/"criteria.json"), arrays=14, native_rows=48, complex_rows=48,
            FD_steps=2, mandatory_tests=58, native_dispatches=160, PDE_calls=0)))
        return 0
    if not all((args.packet, args.external_observation, args.output, args.expected_source_sha256,
            args.expected_criteria_sha256, args.expected_external_sha256, args.expected_result_sha256,
            args.expected_route_source_sha256, args.expected_route_manifest_sha256)):
        parser.error("explicit actual terminal/audit GO, paths and all frozen hashes required")
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
