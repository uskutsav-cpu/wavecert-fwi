"""Held stdlib-only comparison of23 source-bound checkpoint-control workers.

Import/default CLI reads no worker output and imports no numerical library.
Only evaluate() decodes retained arrays after full external terminal admission.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import stat
import struct
import zipfile
from pathlib import Path

SOURCES = {
    "run_checkpoint_pilot.py": "31969bd3e072dcbf554e0cd98f022b0e35dd5438ef12566f9c23fa98eee76e9b",
    "pilot_accounting.py": "1b0094af99fb57e44966e18808aa80f939a2a83771ed98c53a94a5e1460f33f2",
    "protocol.json": "27e5a331d6bd9eff0dd6b4fa0ca1bc666c6014ed70ffb3da01821c62f83fdcbc",
    "RUNTIME_SOURCE_PINS.json": "bfbc2e31cd4ff7ae2a0f0342dc3633c5e4dc0189daea0d06eaeb12de1040350f",
}
PROTOCOL_OBJECT_SHA = "7604536e8c4ee136227a336fbb700c938011e1bcd19af92645296f7e272c4dde"
RUNTIME_FILES_SHA = "318a09f325468ebdb4c56aa2c1984eaa4bdae2483c55e04ccccd7ab21723541d"
VARIANTS = ("saved", "checkpoint2", "checkpoint4")
DIRECTIONS = ("interior", "left_edge")
EPSILONS = (0.01, 0.003, 0.001, 0.0003, 0.0001)
MAX_JSON_BYTES = 8 * 1024**2
MAX_NPZ_BYTES = 1024**2
THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
NATIVE_ENV = dict(
    DEVITO_LANGUAGE="C",
    DEVITO_MPI="0",
    DEVITO_AUTOTUNING="off",
    DEVITO_AUTOPADDING="0",
    DEVITO_SAFE_MATH="1",
    DEVITO_JIT_BACKDOOR="0",
    DEVITO_FIRST_TOUCH="0",
    MPLBACKEND="Agg",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            value.update(block)
    return value.hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


class Checks:
    def __init__(self):
        self.count = 0
        self.numeric = 0
        self.issues = []

    def require(self, condition, label, detail=None):
        self.count += 1
        if not condition:
            self.issues.append(dict(check=label, detail=clean(detail)))

    def norm_scaled(self, own, recorded, scale, tolerance, label):
        self.numeric += 1
        ok = all(finite(v) for v in (own, recorded, scale)) and scale > 0
        error = abs(own - recorded) / scale if ok else math.inf
        self.require(
            ok and error <= tolerance,
            label,
            dict(independent=own, recorded=recorded, norm_scaled_error=error, threshold=tolerance),
        )
        return error


def read_json(path):
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON evidence exceeds8MiB cap")
    return json.loads(path.read_bytes())


def safe_file(directory, name):
    path = directory / name
    if (
        Path(name).name != name
        or path.is_symlink()
        or not path.resolve().is_relative_to(directory.resolve())
        or not path.is_file()
    ):
        raise ValueError("missing/escaping/symlinked evidence: " + name)
    return path


def decode_npy(raw, expected_shape):
    """Bounded literal header and C-order little-endian float64, no pickle."""
    if len(raw) < 10 or raw[:6] != b"\x93NUMPY":
        raise ValueError("NPY magic/truncation")
    version = tuple(raw[6:8])
    if version == (1, 0):
        length, start = struct.unpack("<H", raw[8:10])[0], 10
    elif version == (2, 0) and len(raw) >= 12:
        length, start = struct.unpack("<I", raw[8:12])[0], 12
    else:
        raise ValueError("only bounded NPYv1/v2 admitted")
    if length < 1 or length > 4096 or start + length > len(raw):
        raise ValueError("NPY header length")
    header = ast.literal_eval(raw[start : start + length].decode("latin1").strip())
    if (
        not isinstance(header, dict)
        or set(header) != {"descr", "fortran_order", "shape"}
        or header["descr"] != "<f8"
        or header["fortran_order"] is not False
        or not isinstance(header["shape"], tuple)
        or any(type(n) is not int or n <= 0 for n in header["shape"])
        or header["shape"] != tuple(expected_shape)
    ):
        raise ValueError("NPY dtype/order/shape differs")
    body = raw[start + length :]
    if len(body) != 8 * math.prod(expected_shape):
        raise ValueError("NPY exact payload length")
    values = tuple(v[0] for v in struct.iter_unpack("<d", body))
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite retained NPY vector")
    return dict(
        shape=tuple(expected_shape),
        values=values,
        bytes=len(body),
        data_sha256=hashlib.sha256(body).hexdigest(),
    )


def decode_npz(path, shapes, on_decoded=None):
    if path.stat().st_size > MAX_NPZ_BYTES:
        raise ValueError("NPZ exceeds1MiB cap")
    names = {name + ".npy": shape for name, shape in shapes.items()}
    answer = {}
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
        entries = archive.infolist()
        if len(entries) != len(names) or {e.filename for e in entries} != set(names):
            raise ValueError("NPZ exact unique array denominator")
        for entry in entries:
            mode = stat.S_IFMT(entry.external_attr >> 16)
            maximum = 8 * math.prod(names[entry.filename]) + 4108
            # The frozen worker uses np.savez, which writes uncompressed members.
            if (
                entry.flag_bits & 1
                or entry.compress_type != zipfile.ZIP_STORED
                or mode not in (0, stat.S_IFREG)
                or entry.file_size > maximum
            ):
                raise ValueError("NPZ unsafe type/encryption/compression/expansion")
            with archive.open(entry) as stream:
                raw = stream.read(maximum + 1)
                if len(raw) > maximum or stream.read(1) or len(raw) != entry.file_size:
                    raise ValueError("NPZ actual member read exceeds bounded declared extent")
            value = decode_npy(raw, names[entry.filename])
            answer[entry.filename[:-4]] = value
            if on_decoded is not None:
                on_decoded(entry.filename[:-4])
    return answer


def norm(values):
    return math.hypot(*values)


def dot(left, right):
    if len(left) != len(right):
        raise ValueError("dot vector length")
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def pullback(padded, shape, nbl):
    nx, nz = shape
    if len(padded) != (nx + 2 * nbl) * (nz + 2 * nbl):
        raise ValueError("padded pullback dimensions")
    terms = [[] for _ in range(nx * nz)]
    for i in range(nx + 2 * nbl):
        for j in range(nz + 2 * nbl):
            ii, jj = min(max(i - nbl, 0), nx - 1), min(max(j - nbl, 0), nz - 1)
            terms[ii * nz + jj].append(padded[i * (nz + 2 * nbl) + j])
    return tuple(math.fsum(v) for v in terms)


def direction(name, shape, m0):
    nx, nz = shape
    values = []
    for i in range(nx):
        for j in range(nz):
            if name == "interior":
                value = (
                    math.exp(-(((i - 20) / 5) ** 2) - ((j - 16) / 4) ** 2)
                    if 8 <= i <= 32 and 8 <= j <= 24
                    else 0.0
                )
            elif name == "left_edge":
                value = (
                    math.exp(-((i / 3) ** 2) - ((j - 16) / 5) ** 2)
                    if i <= 10 and 6 <= j <= 26
                    else 0.0
                )
            else:
                raise ValueError("unregistered direction")
            values.append(value)
    scale = max(abs(v) for v in values)
    if scale <= 0:
        raise ValueError("zero manufactured direction")
    return tuple(m0 * v / scale for v in values)


def dual(nt, nr):
    return tuple(
        0.0 if t in (0, nt - 1) else math.sin(0.037 * t) * math.cos(0.29 * (j + 1))
        for t in range(nt)
        for j in range(nr)
    )


def vector_error(reference, candidate):
    if len(reference) != len(candidate) or norm(reference) <= 0:
        raise ValueError("vector dimensions/nontrivial reference")
    return norm(tuple(a - b for a, b in zip(reference, candidate, strict=True))) / norm(reference)


def relative_error(a, b):
    if not all(finite(v) for v in (a, b)):
        raise ValueError("nonfinite finite-difference scalar")
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def expected_slots():
    rows = [
        dict(
            slot_id=name,
            phase=name,
            checkpoints=None if name == "saved" else int(name[-1]),
            direction=None,
            epsilon=None,
            epsilon_index=None,
            sign=None,
        )
        for name in VARIANTS
    ]
    for name in DIRECTIONS:
        for index, epsilon in enumerate(EPSILONS):
            for sign in (-1, 1):
                rows.append(
                    dict(
                        slot_id=f"fd_{name}_{index}_{sign}",
                        phase="fd",
                        checkpoints=None,
                        direction=name,
                        epsilon=epsilon,
                        epsilon_index=index,
                        sign=sign,
                    )
                )
    return [dict(slot_index=i, **row) for i, row in enumerate(rows)]


def _counts(calls):
    return dict(
        attempted_calls=len(calls),
        returned_calls=len(calls),
        attempted_steps=sum(v["native_steps"] for v in calls),
        returned_steps=sum(v["native_steps"] for v in calls),
    )


def audit_events(path, row, p, checks, label):
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("event evidence exceeds16MiB cap")
    events = [json.loads(line) for line in path.read_text().splitlines()]
    checks.require(0 < len(events) <= 8192, label + "/bounded_events")
    pending, native, copied, initial, previous, capacity = None, [], [], None, -1.0, None
    for index, event in enumerate(events):
        prefix = label + f"/event{index}"
        checks.require(
            type(event["sequence"]) is int
            and event["sequence"] == index
            and finite(event["seconds_since_ledger"])
            and event["seconds_since_ledger"] >= previous,
            prefix + "/chronology",
        )
        previous = event["seconds_since_ledger"]
        kind = event["kind"]
        if kind in (
            "runtime_source_gate_pass",
            "manufactured_setup_complete",
            "checkpoint_capacity_admitted_before_storage_allocation",
        ):
            checks.require(event["phase"] == "setup", prefix + "/setup_phase")
            if kind == "checkpoint_capacity_admitted_before_storage_allocation":
                checks.require(
                    capacity is None
                    and not native
                    and not copied
                    and pending is None
                    and row["checkpoint_budget"] is not None
                    and all(event.get(k) == v for k, v in row["checkpoint_budget"].items()),
                    prefix + "/exact_budget_before_propagation_or_copy",
                )
                capacity = event
        elif kind == "initial_forward_complete":
            checks.require(
                initial is None and event["phase"] == "initial_forward",
                prefix + "/single_initial_completion",
            )
            initial = event
        elif kind in ("native_attempt", "checkpoint_copy_attempt"):
            checks.require(
                pending is None and event["status"] == "attempted",
                prefix + "/single_pending_attempt",
            )
            pending = event
        elif kind in ("native_returned", "checkpoint_copy_returned"):
            checks.require(
                pending is not None and event["status"] == "returned", prefix + "/matching_return"
            )
            if pending is None:
                continue
            fields = (
                (
                    "role",
                    "scheduled_start",
                    "scheduled_end",
                    "native_time_m",
                    "native_time_M",
                    "native_steps",
                )
                if kind == "native_returned"
                else ("operation", "key", "pointer_count", "bytes")
            )
            checks.require(
                pending["kind"]
                == ("native_attempt" if kind == "native_returned" else "checkpoint_copy_attempt")
                and all(event[k] == pending[k] for k in fields)
                and event["phase"] == pending["phase"],
                prefix + "/exact_attempt_return_fields",
            )
            checks.require(finite(event["seconds"]) and event["seconds"] >= 0, prefix + "/duration")
            if kind == "native_returned":
                start, end = event["scheduled_start"], event["scheduled_end"]
                checks.require(
                    type(start) is int and type(end) is int and 0 <= start < end <= p["nt"] - 2,
                    prefix + "/scheduled_range",
                )
                checks.require(
                    (event["native_time_m"], event["native_time_M"], event["native_steps"])
                    == (start + 1, end, end - start),
                    prefix + "/native_offset_and_steps",
                )
                checks.require(
                    all(
                        type(event[k]) is int
                        for k in ("native_time_m", "native_time_M", "native_steps")
                    ),
                    prefix + "/integer_native_range",
                )
                checks.require(
                    event["role"] in ("forward", "reverse")
                    and event["phase"] in ("initial_forward", "reverse_with_paid_replay"),
                    prefix + "/role_phase",
                )
                checks.require(
                    event["phase"] != "initial_forward" or event["role"] == "forward",
                    prefix + "/initial_forward_only",
                )
                native.append(event)
            else:
                checks.require(
                    event["operation"] in ("save", "load")
                    and type(event["key"]) is int
                    and row["checkpoints"] is not None
                    and 0 <= event["key"] < row["checkpoints"],
                    prefix + "/copy_key",
                )
                checks.require(
                    event["pointer_count"] == 2
                    and row["checkpoint_budget"] is not None
                    and event["bytes"] == row["checkpoint_budget"]["checkpoint_bytes"],
                    prefix + "/actual_copy_bytes",
                )
                copied.append(event)
            pending = None
        else:
            checks.require(False, prefix + "/unexpected_or_failed_event", kind)
    checks.require(pending is None and initial is not None, label + "/all_attempts_returned")
    fw = [e for e in native if e["role"] == "forward"]
    first = [e for e in fw if e["phase"] == "initial_forward"]
    reverse = [e for e in native if e["role"] == "reverse"]
    expected = list(range(1, p["nt"] - 1))

    def covered(calls):
        return sorted(i for e in calls for i in range(e["native_time_m"], e["native_time_M"] + 1))

    checks.require(covered(first) == expected, label + "/once_only_initial_native_mask")
    ordered_forward = [i for e in first for i in range(e["native_time_m"], e["native_time_M"] + 1)]
    checks.require(ordered_forward == expected, label + "/ordered_initial_forward_native_traversal")
    checks.require(
        row["native_forward"] == _counts(fw) and row["initial_native_forward"] == _counts(first),
        label + "/forward_counts",
    )
    if initial is not None:
        checks.require(
            initial["native_counts"] == _counts(first) and initial["objective"] == row["objective"],
            label + "/initial_event_counts_objective",
        )
    for tree in (row["native_forward"], row["initial_native_forward"], row["native_reverse"]):
        if tree is not None:
            checks.require(
                set(tree) == set(_counts([]))
                and all(type(v) is int and v >= 0 for v in tree.values()),
                label + "/strict_nonnegative_count_trees",
            )
    replay = sum(e["native_steps"] for e in fw) - sum(e["native_steps"] for e in first)
    checks.require(row["native_replay_forward_steps"] == replay, label + "/paid_replay_steps")
    if row["phase"] == "fd":
        checks.require(
            not reverse and row["native_reverse"] is None and not copied and replay == 0,
            label + "/FD_forward_only",
        )
    else:
        checks.require(
            covered(reverse) == expected and row["native_reverse"] == _counts(reverse),
            label + "/once_only_reverse_native_mask",
        )
        ordered_reverse = [
            i for e in reverse for i in range(e["native_time_M"], e["native_time_m"] - 1, -1)
        ]
        checks.require(
            ordered_reverse == list(reversed(expected)), label + "/ordered_reverse_native_traversal"
        )
        checks.require(
            row["phase"] != "saved" or not copied and replay == 0,
            label + "/saved_no_checkpoint_replay",
        )
    projected = [
        {k: e[k] for k in ("operation", "key", "pointer_count", "bytes", "status", "seconds")}
        for e in copied
    ]
    checks.require(projected == row["checkpoint_copy_calls"], label + "/all_copy_calls_counted")
    if row["phase"].startswith("checkpoint"):
        checks.require(
            capacity is not None
            and replay > 0
            and {e["operation"] for e in copied} == {"save", "load"},
            label + "/nontrivial_replay_and_storage",
        )
    else:
        checks.require(capacity is None, label + "/no_checkpoint_capacity_event")
    return dict(
        events=len(events),
        initial_forward_steps=len(expected),
        replay_forward_steps=replay,
        reverse_steps=sum(e["native_steps"] for e in reverse),
        copy_calls=len(copied),
        save_bytes=sum(e["bytes"] for e in copied if e["operation"] == "save"),
        load_bytes=sum(e["bytes"] for e in copied if e["operation"] == "load"),
    )


def _admit(children_root, terminal, protocol, expected_sources, expected_go, checks):
    checks.require(expected_sources == SOURCES, "registered_exact_worker_source4")
    checks.require(object_sha(protocol) == PROTOCOL_OBJECT_SHA, "registered_protocol_object")
    checks.require(digest(expected_go), "registered_worker_GO_hash")
    checks.require(
        terminal.get("schema") == "checkpoint_pilot23_external_v1"
        and terminal.get("status") == "ALL_23_CHILDREN_WAITED_EXIT0"
        and terminal.get("slot_count") == 23
        and terminal.get("unknown_work") is False,
        "external_terminal_full23_known_exit0",
    )
    checks.require(
        terminal.get("source_files") == expected_sources
        and terminal.get("worker_go_sha256") == expected_go,
        "external_terminal_source_GO",
    )
    entries = list(children_root.iterdir())
    checks.require(
        {p.name for p in entries} == {s["slot_id"] for s in expected_slots()}
        and len(entries) == 23
        and all(p.is_dir() and not p.is_symlink() for p in entries),
        "children_exact23_registered_nonsymlink_directories",
    )
    checks.require(
        type(terminal.get("controller_PID")) is int
        and terminal["controller_PID"] > 0
        and digest(terminal.get("environment_sha256"))
        and isinstance(terminal.get("approved_environment_modules"), dict)
        and len(terminal["approved_environment_modules"]) == 17,
        "external_terminal_controller_environment",
    )
    slots = terminal.get("slots", [])
    checks.require(isinstance(slots, list) and len(slots) == 23, "all23_external_slots")
    if not isinstance(slots, list) or len(slots) != 23:
        return []
    for slot, expected in zip(slots, expected_slots(), strict=True):
        label = expected["slot_id"]
        checks.require(
            all(
                slot.get(k) == v
                and (type(slot.get(k)) is type(v) if v is not None else slot.get(k) is None)
                for k, v in expected.items()
            ),
            label + "/registered_slot_parameters",
        )
        checks.require(
            slot.get("state") == "waited_exit0"
            and type(slot.get("actual_returncode")) is int
            and slot["actual_returncode"] == 0
            and type(slot.get("PID")) is int
            and slot["PID"] > 0,
            label + "/external_actual_wait",
        )
        worker = children_root / label / "worker"
        checks.require(
            slot.get("worker_output") == str(worker.resolve()), label + "/canonical_worker_output"
        )
        checks.require(
            worker.is_dir()
            and not worker.is_symlink()
            and worker.resolve().is_relative_to(children_root),
            label + "/worker_directory_confined",
        )
        checks.require(
            finite(slot.get("external_wall_seconds")) and slot["external_wall_seconds"] > 0,
            label + "/actual_whole_child_wall",
        )
    return slots


def _worker_evidence(
    children_root, slot, terminal, protocol, expected_sources, expected_go, checks
):
    label = slot["slot_id"]
    parent = children_root / label
    directory = parent / "worker"
    for name, key in (
        ("started.json", "started_sha256"),
        ("exit.json", "exit_sha256"),
        ("import_identity.json", "import_identity_sha256"),
    ):
        checks.require(
            file_sha(safe_file(parent, name)) == slot.get(key), label + "/external_" + name
        )
    exit_record, started = read_json(parent / "exit.json"), read_json(parent / "started.json")
    identity = read_json(parent / "import_identity.json")
    for source in (exit_record, started):
        checks.require(
            source.get("slot_id") == label
            and source.get("PID") == slot["PID"]
            and source.get("worker_output") == str(directory.resolve()),
            label + "/external_receipt_identity",
        )
    checks.require(
        exit_record.get("observed_terminal") is True
        and type(exit_record.get("actual_returncode")) is int
        and exit_record["actual_returncode"] == 0
        and exit_record.get("external_wall_seconds") == slot["external_wall_seconds"]
        and exit_record.get("unknown_work") is False
        and exit_record.get("unknown_partial_native_work") is False
        and exit_record.get("reason") is None,
        label + "/external_receipt_actual_exit",
    )
    checks.require(
        identity.get("status") == "SELECTED_ACTUAL_IMPORT_ORIGINS_MATCH_APPROVED_ENVIRONMENT"
        and identity.get("slot_id") == label
        and identity.get("worker_output") == str(directory.resolve())
        and identity.get("PID")
        == identity.get("process_group_ID")
        == identity.get("session_ID")
        == slot["PID"]
        and identity.get("PPID") == terminal["controller_PID"]
        and identity.get("environment_sha256") == terminal["environment_sha256"]
        and identity.get("worker_go_sha256") == expected_go
        and identity.get("controlled_native_environment") == NATIVE_ENV
        and identity.get("native_PDE_calls_before_worker") == 0,
        label + "/external_actual_import_gate_identity",
    )
    checks.require(
        identity.get("selected_modules") == terminal["approved_environment_modules"],
        label + "/all17_approved_actual_import_origins",
    )
    for name, meta in identity["selected_modules"].items():
        checks.require(
            isinstance(meta.get("path"), str)
            and Path(meta["path"]).is_absolute()
            and digest(meta.get("sha256")),
            label + "/import_origin_metadata/" + name,
        )
    checks.require(
        file_sha(safe_file(directory, "result.json")) == slot.get("result_sha256"),
        label + "/external_result_hash",
    )
    row = read_json(directory / "result.json")
    checks.require(
        row.get("source_files") == expected_sources and row.get("root_go_sha256") == expected_go,
        label + "/worker_source_GO",
    )
    checks.require(
        row.get("status") == "WORKER_RETURNED_NO_SCIENTIFIC_SUCCESS_INFERRED"
        and type(row.get("intended_worker_return_code")) is int
        and row["intended_worker_return_code"] == 0,
        label + "/worker_selfreport_distinct_from_actual_exit",
    )
    checks.require(
        (
            row.get("phase"),
            row.get("checkpoints"),
            row.get("fd_direction"),
            row.get("fd_epsilon"),
            row.get("fd_sign"),
        )
        == (slot["phase"], slot["checkpoints"], slot["direction"], slot["epsilon"], slot["sign"]),
        label + "/worker_parameters",
    )
    required = {"admission.json", "runtime.json", "events.jsonl", "forward.c"}
    if slot["phase"] != "fd":
        required |= {"arrays.npz", "reverse.c"}
    artifacts = row["output_artifacts"]
    checks.require(
        all(p.is_file() and not p.is_symlink() for p in directory.iterdir()),
        label + "/flat_nonsymlink_worker_output",
    )
    checks.require(
        set(artifacts)
        == required
        == {p.name for p in directory.iterdir() if p.is_file() and p.name != "result.json"},
        label + "/full_worker_output_denominator",
    )
    for name, expected in artifacts.items():
        checks.require(
            digest(expected) and file_sha(safe_file(directory, name)) == expected,
            label + "/worker_artifact/" + name,
        )
    admission, runtime = (
        read_json(directory / "admission.json"),
        read_json(directory / "runtime.json"),
    )
    go = admission["root_go"]
    go_bytes = (
        json.dumps(go["content"], indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    checks.require(
        admission["source_files"] == expected_sources
        and go["sha256"] == expected_go
        and hashlib.sha256(go_bytes).hexdigest() == expected_go
        and go["content"].get("source_files") == expected_sources
        and go["content"].get("authorization") == "GO_CONSTANT_MEDIUM_CHECKPOINT_PILOT"
        and all(
            go["content"].get(k) is True
            for k in (
                "execution_authorized",
                "independent_source_review_clear",
                "criteria_frozen_before_execution",
            )
        ),
        label + "/worker_admission",
    )
    checks.require(
        runtime["python"] == protocol["runtime"]["python"]
        and runtime["packages"]
        == {k: protocol["runtime"][k] for k in ("devito", "pyrevolve", "numpy", "scipy")},
        label + "/actual_runtime",
    )
    checks.require(runtime["threads"] == {k: "1" for k in THREADS}, label + "/runtime_threads")
    checks.require(
        runtime["devito_configuration_environment"]
        == dict(
            DEVITO_LANGUAGE="C",
            DEVITO_MPI="0",
            DEVITO_AUTOTUNING="off",
            DEVITO_AUTOPADDING="0",
            DEVITO_SAFE_MATH="1",
        ),
        label + "/fixed_native_configuration",
    )
    source_rows = [
        {k: v for k, v in r.items() if k != "installed_path"} for r in runtime["source_checks"]
    ]
    checks.require(
        len(source_rows) == 13 and object_sha(source_rows) == RUNTIME_FILES_SHA,
        label + "/all13_primary_source_pins",
    )
    p = protocol["constant_control"]
    checks.require(
        row["time_axis"] == dict(start=p["t0_ms"], stop=p["tn_ms"], step=p["dt_ms"], num=p["nt"])
        and row["native_schedule_steps"] == p["nt"] - 2,
        label + "/fixed_time_axis",
    )
    checks.require(
        row["prepared_bounds"]
        == dict(
            forward=[1, p["nt"] - 2],
            forward_offset=1,
            reverse_offset=None if row["phase"] == "fd" else 1,
        ),
        label + "/native_time_masks",
    )
    checks.require(finite(row["objective"]), label + "/finite_objective")
    checks.require(
        finite(row["worker_entry_through_pre_result_record_seconds"])
        and 0
        <= row["worker_entry_through_pre_result_record_seconds"]
        <= slot["external_wall_seconds"],
        label + "/whole_child_time_scope",
    )
    for record in (row["resources"], row["resources_before_result_record"]):
        checks.require(
            type(record["worker_peak_RSS_bytes"]) is int
            and 0
            < record["worker_peak_RSS_bytes"]
            <= protocol["memory"]["whole_child_peak_RSS_review_cap_bytes"],
            label + "/reported_peak_RSS_cap",
        )
    for name, metadata in row["generated_operators_and_JIT_metadata"].items():
        path = directory / (name + ".c")
        checks.require(
            metadata["generated_C_sha256"] == file_sha(path)
            and metadata["generated_C_bytes"] == path.stat().st_size
            and metadata["cache_hit_or_cold_compile_inferred"] is False,
            label + "/retained_generated_C_JIT_scope",
        )
    allocations = row["array_allocations"]
    required_allocations = {
        "forward_u_allocated",
        "vp_allocated",
        "damp_allocated",
        "source",
        "source_coordinates",
        "forward_receiver_scratch",
        "forward_receiver_coordinates",
        "fixed_receiver_dual",
        "dual_coordinates",
    }
    if row["phase"] != "fd":
        required_allocations |= {"reverse_v_allocated", "gradient_allocated"}
    checks.require(
        set(allocations) == required_allocations, label + "/complete_working_array_inventory"
    )
    for name, meta in allocations.items():
        checks.require(
            meta["dtype"] == "float64"
            and isinstance(meta["shape"], list)
            and all(type(v) is int and v > 0 for v in meta["shape"])
            and meta["bytes"] == 8 * math.prod(meta["shape"])
            and digest(meta["sha256"]),
            label + "/buffer_metadata/" + name,
        )
    u_shape = allocations["forward_u_allocated"]["shape"]
    checks.require(
        len(u_shape) == 3
        and u_shape[0] == (p["nt"] if row["phase"] == "saved" else 3)
        and all(u_shape[i + 1] >= p["shape"][i] + 2 * p["nbl"] for i in range(2)),
        label + "/full_history_or_rolling_levels",
    )
    if row["phase"] != "fd":
        checks.require(
            allocations["reverse_v_allocated"]["shape"] == [3, *u_shape[1:]],
            label + "/rolling_reverse_levels",
        )
    for name, shape in (
        ("source", [p["nt"], 1]),
        ("source_coordinates", [1, 2]),
        ("forward_receiver_scratch", [p["nt"], len(p["receiver_positions_m"])]),
        ("fixed_receiver_dual", [p["nt"], len(p["receiver_positions_m"])]),
        ("forward_receiver_coordinates", [len(p["receiver_positions_m"]), 2]),
        ("dual_coordinates", [len(p["receiver_positions_m"]), 2]),
    ):
        checks.require(allocations[name]["shape"] == shape, label + "/fixed_sparse_shape/" + name)
    for key, positions in (
        ("source_coordinates", p["source_position_m"]),
        ("dual_coordinates", p["receiver_positions_m"]),
    ):
        raw = b"".join(struct.pack("<d", float(v)) for point in positions for v in point)
        checks.require(
            row["input_hashes"][key] == hashlib.sha256(raw).hexdigest(),
            label + "/independent_fixed_coordinates/" + key,
        )
    if row["phase"].startswith("checkpoint"):
        budget = row["checkpoint_budget"]
        elements = 2 * math.prod(u_shape[1:])
        checks.require(
            budget
            == dict(
                size_elements=elements,
                itemsize=8,
                checkpoint_bytes=8 * elements,
                checkpoints=slot["checkpoints"],
                capacity_bytes=slot["checkpoints"] * 8 * elements,
                cap_bytes=protocol["memory"]["checkpoint_capacity_cap_bytes"],
            )
            and budget["capacity_bytes"] <= budget["cap_bytes"],
            label + "/explicit_K_actual_allocated_plane_budget",
        )
    else:
        checks.require(row["checkpoint_budget"] is None, label + "/no_checkpoint_allocation")
    for name, key in (
        ("source", "source"),
        ("source_coordinates", "source_coordinates"),
        ("fixed_receiver_dual", "dual"),
        ("dual_coordinates", "dual_coordinates"),
    ):
        checks.require(
            allocations[name]["sha256"] == row["input_hashes"][key],
            label + "/fixed_allocated_input/" + name,
        )
    row["independent_event_audit"] = audit_events(directory / "events.jsonl", row, p, checks, label)
    return row


def evaluate(children_root, terminal, protocol, expected_sources, expected_worker_go_sha256):
    """Return all check issues; never mutate evidence or launch numerical work."""
    checks = Checks()
    report = dict(
        status="IMPLEMENTATION_AGREEMENT_FAIL",
        implementation_agreement_pass=False,
        planned_worker_outputs=23,
        vector_comparisons=[],
        finite_difference=[],
        worker_evidence=[],
        npz_arrays_decoded=0,
        native_PDE_calls=0,
        scientific_goals_complete=False,
        fair_speed_or_matched_cost_claim=False,
        all23_externally_waited_exit0=False,
    )
    try:
        root = Path(children_root).resolve()
        slots = _admit(
            root, terminal, protocol, expected_sources, expected_worker_go_sha256, checks
        )
        if checks.issues:
            return _finish(report, checks)
        rows = {}
        for slot in slots:
            try:
                row = _worker_evidence(
                    root,
                    slot,
                    terminal,
                    protocol,
                    expected_sources,
                    expected_worker_go_sha256,
                    checks,
                )
                rows[slot["slot_id"]] = row
                report["worker_evidence"].append(
                    dict(
                        slot_id=slot["slot_id"],
                        external_actual_returncode=slot["actual_returncode"],
                        worker_intended_returncode=row["intended_worker_return_code"],
                        external_wall_seconds=slot["external_wall_seconds"],
                        reported_worker_peak_RSS_bytes=row["resources"]["worker_peak_RSS_bytes"],
                        events=row["independent_event_audit"],
                    )
                )
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                IndexError,
                OverflowError,
                AttributeError,
                SyntaxError,
                zipfile.BadZipFile,
            ) as exc:
                checks.require(
                    False,
                    slot["slot_id"] + "/unreadable_or_invalid_evidence",
                    type(exc).__name__ + ": " + str(exc),
                )
        if checks.issues:
            return _finish(report, checks)
        report["all23_externally_waited_exit0"] = len(rows) == 23
        p, criteria = protocol["constant_control"], protocol["candidate_criteria_pending_review"]
        shapes = dict(
            receiver=(p["nt"], len(p["receiver_positions_m"])),
            padded_gradient=tuple(v + 2 * p["nbl"] for v in p["shape"]),
            physical_gradient=tuple(p["shape"]),
        )
        arrays = {}

        def decoded(_name):
            report["npz_arrays_decoded"] += 1

        for name in VARIANTS:
            arrays[name] = decode_npz(root / name / "worker/arrays.npz", shapes, on_decoded=decoded)
        for name in VARIANTS:
            checks.require(
                rows[name]["input_hashes"] == rows["saved"]["input_hashes"],
                name + "/same_full_primal_source_dual_damping",
            )
            initial_mass = b"".join(struct.pack("<d", 0.25) for _ in range(math.prod(p["shape"])))
            checks.require(
                rows[name]["physical_m"]["shape"] == p["shape"]
                and rows[name]["physical_m"]["dtype"] == "float64"
                and rows[name]["physical_m"]["bytes"] == len(initial_mass)
                and rows[name]["physical_m"]["sha256"] == hashlib.sha256(initial_mass).hexdigest(),
                name + "/constant_base_m",
            )
            vp = b"".join(struct.pack("<d", 2.0) for _ in range(math.prod(p["shape"])))
            padded_vp = b"".join(
                struct.pack("<d", 2.0) for _ in range(math.prod(shapes["padded_gradient"]))
            )
            checks.require(
                rows[name]["physical_vp"]
                == dict(
                    shape=p["shape"],
                    dtype="float64",
                    bytes=len(vp),
                    sha256=hashlib.sha256(vp).hexdigest(),
                )
                and rows[name]["input_hashes"]["model_vp"] == hashlib.sha256(padded_vp).hexdigest(),
                name + "/constant_base_vp_and_padded_primal",
            )
            receiver = arrays[name]["receiver"]["values"]
            nr = len(p["receiver_positions_m"])
            checks.require(
                all(v == 0 for v in receiver[:nr] + receiver[-nr:]),
                name + "/receiver_first_last_time_mask",
            )
            for quantity in shapes:
                checks.require(
                    norm(arrays[name][quantity]["values"]) > 0, name + "/nontrivial_" + quantity
                )
            r = dual(p["nt"], nr)
            checks.norm_scaled(
                dot(receiver, r),
                rows[name]["objective"],
                norm(receiver) * norm(r),
                criteria["receiver_relative_L2"],
                name + "/independent_fixed_dual_objective",
            )
            extended = pullback(arrays[name]["padded_gradient"]["values"], p["shape"], p["nbl"])
            error = vector_error(extended, arrays[name]["physical_gradient"]["values"])
            checks.numeric += 1
            checks.require(
                error <= criteria["physical_gradient_relative_L2"],
                name + "/independent_E_transpose_faces_corners",
                error,
            )
        for name in VARIANTS[1:]:
            for quantity, threshold in (
                ("receiver", criteria["receiver_relative_L2"]),
                ("padded_gradient", criteria["padded_gradient_relative_L2"]),
                ("physical_gradient", criteria["physical_gradient_relative_L2"]),
            ):
                error = vector_error(
                    arrays["saved"][quantity]["values"], arrays[name][quantity]["values"]
                )
                checks.numeric += 1
                passed = error <= threshold
                checks.require(passed, name + "/saved_agreement/" + quantity, error)
                report["vector_comparisons"].append(
                    dict(
                        variant=name,
                        quantity=quantity,
                        relative_L2=error,
                        threshold=threshold,
                        pass_check=passed,
                    )
                )
        for name in DIRECTIONS:
            dm = direction(name, p["shape"], 1 / p["velocity_m_per_ms"] ** 2)
            dots, cosines = {}, {}
            for variant in VARIANTS:
                gradient = arrays[variant]["physical_gradient"]["values"]
                dots[variant] = dot(gradient, dm)
                scale = norm(gradient) * norm(dm)
                cosines[variant] = abs(dots[variant]) / scale if scale > 0 else 0.0
                checks.require(
                    cosines[variant] >= criteria["nontrivial_direction_cosine_minimum"],
                    variant + "/informative_direction/" + name,
                    cosines[variant],
                )
                checks.norm_scaled(
                    dots[variant],
                    rows[variant]["directional_adjoint"][name],
                    scale,
                    criteria["physical_gradient_relative_L2"],
                    variant + "/independent_gradient_dot_dm/" + name,
                )
            for index, epsilon in enumerate(EPSILONS):
                pair = {sign: rows[f"fd_{name}_{index}_{sign}"] for sign in (-1, 1)}
                for sign, row in pair.items():
                    checks.require(
                        all(
                            row["input_hashes"][key] == rows["saved"]["input_hashes"][key]
                            for key in (
                                "source",
                                "source_coordinates",
                                "dual",
                                "dual_coordinates",
                                "damping",
                            )
                        ),
                        f"fd_{name}_{index}_{sign}/same_fixed_inputs",
                    )
                    checks.require(
                        row["physical_m"]["shape"] == p["shape"]
                        and row["physical_m"]["dtype"] == "float64"
                        and row["physical_m"]["bytes"] == 8 * math.prod(p["shape"])
                        and digest(row["physical_m"]["sha256"]),
                        f"fd_{name}_{index}_{sign}/physical_m_metadata_only",
                    )
                checks.require(
                    pair[-1]["physical_m"]["sha256"] != pair[1]["physical_m"]["sha256"]
                    and all(
                        r["physical_m"]["sha256"] != rows["saved"]["physical_m"]["sha256"]
                        for r in pair.values()
                    ),
                    f"fd_{name}_{index}/distinct_registered_perturbation_digests",
                )
                difference = (pair[1]["objective"] - pair[-1]["objective"]) / (2 * epsilon)
                required = epsilon in protocol["finite_difference"]["required_middle_epsilon"]
                for variant in VARIANTS:
                    error = relative_error(dots[variant], difference)
                    passed = cosines[variant] >= criteria[
                        "nontrivial_direction_cosine_minimum"
                    ] and (not required or error <= criteria["fd_middle_relative_error"])
                    checks.numeric += 1
                    checks.require(passed, f"{variant}/{name}/FD{index}", error)
                    report["finite_difference"].append(
                        dict(
                            variant=variant,
                            direction=name,
                            epsilon=epsilon,
                            independent_adjoint=dots[variant],
                            centered_difference=difference,
                            relative_error=error,
                            direction_cosine=cosines[variant],
                            required_middle_step=required,
                            pass_check=passed,
                        )
                    )
        checks.require(
            len(report["vector_comparisons"]) == 6
            and len(report["finite_difference"]) == 30
            and sum(r["required_middle_step"] for r in report["finite_difference"]) == 18,
            "complete_6_vectors_30_FD_18_required_middle",
        )
        report["implementation_agreement_pass"] = not checks.issues
        report["status"] = (
            "IMPLEMENTATION_AGREEMENT_PASS"
            if not checks.issues
            else "IMPLEMENTATION_AGREEMENT_FAIL"
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        OverflowError,
        AttributeError,
        SyntaxError,
        struct.error,
        zipfile.BadZipFile,
    ) as exc:
        checks.require(
            False, "unexpected_schema_or_retained_array", type(exc).__name__ + ": " + str(exc)
        )
    return _finish(report, checks)


def _finish(report, checks):
    report.update(
        checks=checks.count,
        numeric_comparisons=checks.numeric,
        issues=checks.issues,
        verified_worker_outputs=len(report["worker_evidence"]),
        source_files=SOURCES,
        comparison_source_sha256=file_sha(Path(__file__)),
        interpretation="Standard checkpoint implementation agreement only; no new method, field accuracy, FWI recovery, independent geology, fair speed or student ownership.",
        limits="FD physical-m metadata hashes are bound to exact worker source/parameters, not bitwise regenerated using math.exp. Fixed source/dual/damping hashes repeat across children; scalar objective uses independently generated fixed dual at the existing receiver norm tolerance. Native events/copy counts and memory inventories are retained wrapper evidence, not OS/compiled-kernel proof. Shared JIT cache can be warm. Array storage subtotals are not RSS or nonconcurrent compiler-peak sums.",
    )
    if checks.issues:
        report.update(implementation_agreement_pass=False, status="IMPLEMENTATION_AGREEMENT_FAIL")
    return clean(report)


if __name__ == "__main__":
    print(
        json.dumps(
            dict(
                status="HELD_METADATA_ONLY_NO_WORKER_READ_NO_IMPORT_NO_PDE",
                comparison_source_sha256=file_sha(Path(__file__)),
                expected_workers=23,
                vector_comparisons=6,
                finite_difference_rows=30,
                required_middle_rows=18,
            )
        )
    )
