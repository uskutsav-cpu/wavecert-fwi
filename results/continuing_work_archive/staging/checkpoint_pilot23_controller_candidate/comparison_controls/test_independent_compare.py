"""Manufactured scalar/vector/NPY evidence only; no numerical library or PDE."""

import copy
import hashlib
import io
import json
import math
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import independent_compare as c
import pytest

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "fixtures/protocol.json").read_bytes())
PINS = json.loads((HERE / "fixtures/RUNTIME_SOURCE_PINS.json").read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def raw(values):
    return b"".join(struct.pack("<d", v) for v in values)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def npy(values, shape, *, version=1, dtype="<f8", fortran=False):
    header = repr(dict(descr=dtype, fortran_order=fortran, shape=tuple(shape)))
    start = 10 if version == 1 else 12
    header = (header + " " * ((16 - (start + len(header) + 1) % 16) % 16) + "\n").encode()
    return (
        b"\x93NUMPY"
        + bytes((version, 0))
        + struct.pack("<H" if version == 1 else "<I", len(header))
        + header
        + raw(values)
    )


def npz(path, arrays, *, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, (values, shape) in arrays.items():
            archive.writestr(name + ".npy", npy(values, shape))


def meta(values, shape):
    return dict(
        shape=list(shape), dtype="float64", bytes=8 * math.prod(shape), sha256=digest(raw(values))
    )


def symbolic_meta(shape, label):
    return dict(
        shape=list(shape),
        dtype="float64",
        bytes=8 * math.prod(shape),
        sha256=digest(label.encode()),
    )


def physical_direction(name):
    answer = []
    for i in range(41):
        for j in range(33):
            if name == "interior":
                v = (
                    math.exp(-((i - 20) ** 2) / 25 - (j - 16) ** 2 / 16)
                    if 8 <= i <= 32 and 8 <= j <= 24
                    else 0
                )
            else:
                v = math.exp(-i * i / 9 - (j - 16) ** 2 / 25) if i <= 10 and 6 <= j <= 26 else 0
            answer.append(0.25 * v)
    return answer


def make_events(phase, budget, objective):
    events = []

    def add(kind, event_phase="setup", **value):
        events.append(
            dict(
                sequence=len(events),
                seconds_since_ledger=len(events) / 100,
                phase=event_phase,
                kind=kind,
                **value,
            )
        )

    add("runtime_source_gate_pass")
    add("manufactured_setup_complete")
    if budget:
        add("checkpoint_capacity_admitted_before_storage_allocation", **budget)

    def native(role, event_phase):
        fields = dict(
            role=role,
            scheduled_start=0,
            scheduled_end=255,
            native_time_m=1,
            native_time_M=255,
            native_steps=255,
        )
        add("native_attempt", event_phase, status="attempted", **fields)
        add("native_returned", event_phase, status="returned", seconds=0.001, **fields)

    native("forward", "initial_forward")
    first = dict(attempted_calls=1, returned_calls=1, attempted_steps=255, returned_steps=255)
    add("initial_forward_complete", "initial_forward", native_counts=first, objective=objective)
    copies = []
    if budget:
        for operation in ("save", "load"):
            fields = dict(
                operation=operation, key=0, pointer_count=2, bytes=budget["checkpoint_bytes"]
            )
            add("checkpoint_copy_attempt", "reverse_with_paid_replay", status="attempted", **fields)
            add(
                "checkpoint_copy_returned",
                "reverse_with_paid_replay",
                status="returned",
                seconds=0.001,
                **fields,
            )
            copies.append(dict(status="returned", seconds=0.001, **fields))
            if operation == "save":
                native("forward", "reverse_with_paid_replay")
    if phase != "fd":
        native("reverse", "reverse_with_paid_replay")
    total = dict(first)
    if budget:
        total = dict(attempted_calls=2, returned_calls=2, attempted_steps=510, returned_steps=510)
    return events, first, total, copies


def make_cohort(directory):
    root = directory / "children"
    root.mkdir()
    go_content = dict(
        authorization="GO_CONSTANT_MEDIUM_CHECKPOINT_PILOT",
        execution_authorized=True,
        independent_source_review_clear=True,
        criteria_frozen_before_execution=True,
        source_files=c.SOURCES,
        manufactured_control_only=True,
    )
    go_sha = digest((json.dumps(go_content, indent=2, sort_keys=True) + "\n").encode())
    approved = {
        f"manufactured.module{i}": dict(
            path=f"/manufactured/approved/module{i}.py", sha256=digest(str(i).encode())
        )
        for i in range(17)
    }
    terminal = dict(
        schema="checkpoint_pilot23_external_v1",
        status="ALL_23_CHILDREN_WAITED_EXIT0",
        slot_count=23,
        source_files=c.SOURCES,
        worker_go_sha256=go_sha,
        unknown_work=False,
        controller_PID=999,
        environment_sha256="b" * 64,
        approved_environment_modules=approved,
        slots=[],
    )
    receiver = [0.0] * (257 * 7)
    receiver[7] = 1.0
    padded = [1.0] * (61 * 53)
    gradient = [
        float((11 if i in (0, 40) else 1) * (11 if j in (0, 32) else 1))
        for i in range(41)
        for j in range(33)
    ]
    base_objective = math.sin(0.037) * math.cos(0.29)
    adjoint = {
        d: math.fsum(g * v for g, v in zip(gradient, physical_direction(d), strict=True))
        for d in c.DIRECTIONS
    }
    mass = meta([0.25] * 1353, (41, 33))
    vp = meta([2.0] * 1353, (41, 33))
    input_hashes = dict(
        source="c" * 64,
        source_coordinates=digest(raw([200.0, 120.0])),
        dual="d" * 64,
        dual_coordinates=digest(
            raw(
                [v for point in PROTOCOL["constant_control"]["receiver_positions_m"] for v in point]
            )
        ),
        damping="e" * 64,
        model_vp=digest(raw([2.0] * (61 * 53))),
    )
    for slot in c.expected_slots():
        parent = root / slot["slot_id"]
        parent.mkdir()
        worker = parent / "worker"
        worker.mkdir()
        PID = 1000 + slot["slot_index"]
        phase = slot["phase"]
        K = slot["checkpoints"]
        budget = (
            None
            if K is None
            else dict(
                size_elements=2 * 65 * 57,
                itemsize=8,
                checkpoint_bytes=2 * 65 * 57 * 8,
                checkpoints=K,
                capacity_bytes=K * 2 * 65 * 57 * 8,
                cap_bytes=16777216,
            )
        )
        objective = (
            base_objective
            if phase != "fd"
            else base_objective + slot["sign"] * slot["epsilon"] * adjoint[slot["direction"]]
        )
        arrays = dict(
            forward_u_allocated=symbolic_meta((257 if phase == "saved" else 3, 65, 57), "forwardu"),
            vp_allocated=symbolic_meta((65, 57), "vp"),
            damp_allocated=symbolic_meta((65, 57), "damp"),
            source=symbolic_meta((257, 1), "source"),
            source_coordinates=meta([200.0, 120.0], (1, 2)),
            forward_receiver_scratch=symbolic_meta((257, 7), "scratch"),
            forward_receiver_coordinates=meta(
                [
                    v
                    for point in PROTOCOL["constant_control"]["receiver_positions_m"]
                    for v in point
                ],
                (7, 2),
            ),
            fixed_receiver_dual=symbolic_meta((257, 7), "dual"),
            dual_coordinates=meta(
                [
                    v
                    for point in PROTOCOL["constant_control"]["receiver_positions_m"]
                    for v in point
                ],
                (7, 2),
            ),
        )
        arrays["source"]["sha256"] = input_hashes["source"]
        arrays["fixed_receiver_dual"]["sha256"] = input_hashes["dual"]
        if phase != "fd":
            arrays.update(
                reverse_v_allocated=symbolic_meta((3, 65, 57), "reverse"),
                gradient_allocated=symbolic_meta((63, 55), "gradient"),
            )
            npz(
                worker / "arrays.npz",
                dict(
                    receiver=(receiver, (257, 7)),
                    padded_gradient=(padded, (61, 53)),
                    physical_gradient=(gradient, (41, 33)),
                ),
            )
        events, first, total, copies = make_events(phase, budget, objective)
        (worker / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        write(
            worker / "admission.json",
            dict(
                source_files=c.SOURCES,
                root_go=dict(sha256=go_sha, content=go_content, path="/manufactured/workergo.json"),
            ),
        )
        write(
            worker / "runtime.json",
            dict(
                python="3.11.18",
                packages={
                    name: PROTOCOL["runtime"][name]
                    for name in ("devito", "pyrevolve", "numpy", "scipy")
                },
                threads={name: "1" for name in c.THREADS},
                source_checks=[
                    dict(r, installed_path="/manufactured/" + r["path"]) for r in PINS["files"]
                ],
                devito_configuration_environment=dict(
                    DEVITO_LANGUAGE="C",
                    DEVITO_MPI="0",
                    DEVITO_AUTOTUNING="off",
                    DEVITO_AUTOPADDING="0",
                    DEVITO_SAFE_MATH="1",
                ),
            ),
        )
        compiled = {}
        for name in ("forward",) if phase == "fd" else ("forward", "reverse"):
            source = worker / (name + ".c")
            source.write_text("/* manufactured text; never compiled */\n")
            compiled[name] = dict(
                generated_C_sha256=c.file_sha(source),
                generated_C_bytes=source.stat().st_size,
                cache_hit_or_cold_compile_inferred=False,
            )
        m = copy.deepcopy(mass)
        v = copy.deepcopy(vp)
        if phase == "fd":
            m["sha256"] = digest(slot["slot_id"].encode())
            v["sha256"] = digest((slot["slot_id"] + "/vp").encode())
        row = dict(
            status="WORKER_RETURNED_NO_SCIENTIFIC_SUCCESS_INFERRED",
            source_files=c.SOURCES,
            root_go_sha256=go_sha,
            intended_worker_return_code=0,
            phase=phase,
            checkpoints=K,
            fd_direction=slot["direction"],
            fd_epsilon=slot["epsilon"],
            fd_sign=slot["sign"],
            objective=objective,
            directional_adjoint=None if phase == "fd" else adjoint,
            input_hashes=copy.deepcopy(input_hashes),
            physical_m=m,
            physical_vp=v,
            time_axis=dict(start=0.0, stop=256.0, step=1.0, num=257),
            native_schedule_steps=255,
            prepared_bounds=dict(
                forward=[1, 255], forward_offset=1, reverse_offset=None if phase == "fd" else 1
            ),
            worker_entry_through_pre_result_record_seconds=0.5,
            resources=dict(worker_peak_RSS_bytes=1024),
            resources_before_result_record=dict(worker_peak_RSS_bytes=1024),
            generated_operators_and_JIT_metadata=compiled,
            array_allocations=arrays,
            checkpoint_budget=budget,
            native_forward=total,
            initial_native_forward=first,
            native_reverse=None if phase == "fd" else first,
            native_replay_forward_steps=255 if K else 0,
            checkpoint_copy_calls=copies,
            output_artifacts={p.name: c.file_sha(p) for p in worker.iterdir() if p.is_file()},
        )
        write(worker / "result.json", row)
        common = dict(slot_id=slot["slot_id"], PID=PID, worker_output=str(worker.resolve()))
        write(parent / "started.json", common)
        write(
            parent / "exit.json",
            dict(
                common,
                actual_returncode=0,
                observed_terminal=True,
                external_wall_seconds=1.0,
                unknown_work=False,
                unknown_partial_native_work=False,
                reason=None,
            ),
        )
        write(
            parent / "import_identity.json",
            dict(
                common,
                status="SELECTED_ACTUAL_IMPORT_ORIGINS_MATCH_APPROVED_ENVIRONMENT",
                PPID=999,
                process_group_ID=PID,
                session_ID=PID,
                environment_sha256="b" * 64,
                worker_go_sha256=go_sha,
                native_PDE_calls_before_worker=0,
                selected_modules=approved,
                controlled_native_environment=c.NATIVE_ENV,
            ),
        )
        terminal["slots"].append(
            dict(
                slot,
                state="waited_exit0",
                PID=PID,
                actual_returncode=0,
                worker_output=str(worker.resolve()),
                external_wall_seconds=1.0,
                result_sha256=c.file_sha(worker / "result.json"),
                **{
                    key: c.file_sha(parent / name)
                    for name, key in [
                        ("started.json", "started_sha256"),
                        ("exit.json", "exit_sha256"),
                        ("import_identity.json", "import_identity_sha256"),
                    ]
                },
            )
        )
    return root, terminal, go_sha


@pytest.fixture
def cohort(tmp_path):
    with TemporaryDirectory(prefix="manufactured23-", dir=tmp_path) as directory:
        yield make_cohort(Path(directory))


def evaluate(cohort):
    root, terminal, go = cohort
    return c.evaluate(root, terminal, PROTOCOL, c.SOURCES, go)


def rebind(cohort, slot_id, filename, change):
    root, terminal, _ = cohort
    slot = next(r for r in terminal["slots"] if r["slot_id"] == slot_id)
    parent = root / slot_id
    if filename in ("exit.json", "started.json", "import_identity.json"):
        path = parent / filename
        v = json.loads(path.read_bytes())
        change(v)
        write(path, v)
        slot[
            {
                "exit.json": "exit_sha256",
                "started.json": "started_sha256",
                "import_identity.json": "import_identity_sha256",
            }[filename]
        ] = c.file_sha(path)
        return
    worker = parent / "worker"
    row = json.loads((worker / "result.json").read_bytes())
    if filename == "result.json":
        change(row)
    elif filename == "events.jsonl":
        events = [json.loads(line) for line in (worker / filename).read_text().splitlines()]
        change(events)
        (worker / filename).write_text("".join(json.dumps(e) + "\n" for e in events))
        row["output_artifacts"][filename] = c.file_sha(worker / filename)
    elif filename == "arrays.npz":
        arrays = c.decode_npz(
            worker / filename,
            dict(receiver=(257, 7), padded_gradient=(61, 53), physical_gradient=(41, 33)),
        )
        editable = {k: [list(v["values"]), v["shape"]] for k, v in arrays.items()}
        change(editable)
        npz(worker / filename, editable)
        row["output_artifacts"][filename] = c.file_sha(worker / filename)
    write(worker / "result.json", row)
    slot["result_sha256"] = c.file_sha(worker / "result.json")


def test_full23_manufactured_evidence_passes_all_fixed_comparisons(cohort):
    before = {name for name in sys.modules if name.startswith(("numpy", "devito", "pyrevolve"))}
    result = evaluate(cohort)
    assert not result["issues"], result["issues"]
    assert (
        result["implementation_agreement_pass"] is True
        and result["all23_externally_waited_exit0"] is True
    )
    assert result["verified_worker_outputs"] == 23 and result["npz_arrays_decoded"] == 9
    assert len(result["vector_comparisons"]) == 6 and len(result["finite_difference"]) == 30
    assert sum(r["required_middle_step"] for r in result["finite_difference"]) == 18
    assert result["native_PDE_calls"] == 0 and result["fair_speed_or_matched_cost_claim"] is False
    assert {
        name for name in sys.modules if name.startswith(("numpy", "devito", "pyrevolve"))
    } == before


def test_pullback_sums_faces_and_corners_and_preserves_adjoint_dot():
    padded = tuple(range(1, 21))
    expected = (16, 11, 28, 56, 31, 68)
    assert c.pullback(padded, (2, 3), 1) == expected
    physical = (1, 2, 3, 4, 5, 6)
    extended = tuple(
        physical[min(max(i - 1, 0), 1) * 3 + min(max(j - 1, 0), 2)]
        for i in range(4)
        for j in range(5)
    )
    assert c.dot(extended, padded) == c.dot(physical, expected)


def test_directions_masks_scale_and_nontriviality_are_fixed():
    interior = c.direction("interior", (41, 33), 0.25)
    edge = c.direction("left_edge", (41, 33), 0.25)
    assert interior[20 * 33 + 16] == edge[16] == 0.25 and max(interior) == max(edge) == 0.25
    assert interior[7 * 33 + 16] == edge[11 * 33 + 16] == edge[5] == 0
    assert math.isclose(c.relative_error(2.0, 2.001), 0.001 / 2.001, rel_tol=1e-12)


def test_default_CLI_is_metadata_only_without_numeric_imports_or_worker_files(tmp_path):
    result = subprocess.run(
        [sys.executable, "-B", "-I", str(HERE / "independent_compare.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        result.returncode == 0
        and json.loads(result.stdout)["status"]
        == "HELD_METADATA_ONLY_NO_WORKER_READ_NO_IMPORT_NO_PDE"
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", [1, 2])
def test_npy_float64_versions_and_exact_payload(version):
    value = c.decode_npy(npy([1.0, -2.0, 3.0, 4.0], (2, 2), version=version), (2, 2))
    assert value["values"] == (1.0, -2.0, 3.0, 4.0) and value["data_sha256"] == digest(
        raw([1.0, -2.0, 3.0, 4.0])
    )


@pytest.mark.parametrize("kind", ["object", "fortran", "shape", "trailing", "nan", "header"])
def test_npy_malformed_nonfinite_and_nonliteral_headers_rejected(kind):
    data = npy(
        [1.0, 2.0, 3.0, 4.0],
        (2, 2),
        dtype="|O8" if kind == "object" else "<f8",
        fortran=kind == "fortran",
    )
    expected = (3, 2) if kind == "shape" else (2, 2)
    if kind == "trailing":
        data += b"x"
    if kind == "nan":
        data = npy([1.0, 2.0, float("nan"), 4.0], (2, 2))
    if kind == "header":
        data = b"\x93NUMPY\x01\x00" + struct.pack("<H", 5000) + b" " * 5000
    with pytest.raises((ValueError, SyntaxError)):
        c.decode_npy(data, expected)


def test_npz_compressed_or_duplicate_names_are_rejected(tmp_path):
    path = tmp_path / "compressed.npz"
    npz(path, dict(x=([0.0] * 4, (2, 2))), compression=zipfile.ZIP_DEFLATED)
    with pytest.raises(ValueError):
        c.decode_npz(path, dict(x=(2, 2)))
    path = tmp_path / "duplicate.npz"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("x.npy", npy([0.0] * 4, (2, 2)))
        with pytest.warns(UserWarning):
            z.writestr("x.npy", npy([0.0] * 4, (2, 2)))
    with pytest.raises(ValueError):
        c.decode_npz(path, dict(x=(2, 2)))


def test_npz_actual_read_is_bounded_even_if_metadata_lies(tmp_path, monkeypatch):
    path = tmp_path / "lying.npz"
    path.write_bytes(b"placeholder")
    observed = []

    class BoundedFake(io.BytesIO):
        def read(self, size=-1):
            observed.append(size)
            assert size > 0
            return super().read(size)

    class Archive:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def infolist(self):
            return [
                SimpleNamespace(
                    filename="x.npy",
                    external_attr=0,
                    flag_bits=0,
                    compress_type=zipfile.ZIP_STORED,
                    file_size=1,
                )
            ]

        def open(self, _entry):
            return BoundedFake(b"X" * 10000)

    monkeypatch.setattr(c.zipfile, "ZipFile", Archive)
    with pytest.raises(ValueError):
        c.decode_npz(path, dict(x=(2, 2)))
    assert observed == [8 * 4 + 4108 + 1]


@pytest.mark.parametrize(
    "kind",
    [
        "unknown_terminal",
        "missing_slot",
        "nonzero_exit",
        "import_origin",
        "source",
        "time_mask",
        "copy_bytes",
        "event_sequence",
    ],
)
def test_bound_but_inconsistent_metadata_blocks_array_decode(cohort, kind):
    if kind == "unknown_terminal":
        cohort[1]["unknown_work"] = True
    elif kind == "missing_slot":
        cohort[1]["slots"].pop()
    elif kind == "nonzero_exit":
        rebind(cohort, "saved", "exit.json", lambda v: v.update(actual_returncode=7))
    elif kind == "import_origin":
        rebind(
            cohort,
            "saved",
            "import_identity.json",
            lambda v: v["selected_modules"].update(
                extra=dict(path="/wrong/module.py", sha256="a" * 64)
            ),
        )
    elif kind == "source":
        rebind(cohort, "saved", "result.json", lambda v: v.update(source_files={}))
    elif kind == "time_mask":
        rebind(cohort, "saved", "result.json", lambda v: v["time_axis"].update(step=2))
    elif kind == "copy_bytes":

        def bad(events):
            for e in events:
                if e["kind"].startswith("checkpoint_copy_"):
                    e["bytes"] -= 8

        rebind(cohort, "checkpoint2", "events.jsonl", bad)
    elif kind == "event_sequence":
        rebind(cohort, "saved", "events.jsonl", lambda events: events[1].update(sequence=True))
    result = evaluate(cohort)
    assert result["issues"] and result["implementation_agreement_pass"] is False
    assert result["npz_arrays_decoded"] == 0


@pytest.mark.parametrize("kind", ["pullback_crop", "receiver_vector", "FD_middle", "zero_gradient"])
def test_rehashed_vector_or_middle_FD_tamper_cannot_pass(cohort, kind):
    if kind == "pullback_crop":
        rebind(
            cohort,
            "checkpoint2",
            "arrays.npz",
            lambda a: a["physical_gradient"][0].__setitem__(0, 1.0),
        )
    elif kind == "receiver_vector":
        rebind(cohort, "checkpoint2", "arrays.npz", lambda a: a["receiver"][0].__setitem__(14, 1.0))
    elif kind == "FD_middle":
        rebind(
            cohort,
            "fd_interior_2_1",
            "events.jsonl",
            lambda events: [
                e.update(objective=e["objective"] + 1)
                for e in events
                if e["kind"] == "initial_forward_complete"
            ],
        )
        rebind(
            cohort,
            "fd_interior_2_1",
            "result.json",
            lambda r: r.update(objective=r["objective"] + 1),
        )
    else:

        def zero(arrays):
            for name in ("padded_gradient", "physical_gradient"):
                arrays[name][0][:] = [0.0] * len(arrays[name][0])

        rebind(cohort, "saved", "arrays.npz", zero)
    result = evaluate(cohort)
    assert result["issues"] and result["implementation_agreement_pass"] is False


def test_endpoint_FD_is_retained_but_does_not_replace_fixed_middle_steps(cohort):
    rebind(
        cohort,
        "fd_interior_0_1",
        "events.jsonl",
        lambda events: [
            e.update(objective=e["objective"] + 1)
            for e in events
            if e["kind"] == "initial_forward_complete"
        ],
    )
    rebind(
        cohort, "fd_interior_0_1", "result.json", lambda r: r.update(objective=r["objective"] + 1)
    )
    result = evaluate(cohort)
    assert result["implementation_agreement_pass"] is True, result["issues"]
    selected = [
        r
        for r in result["finite_difference"]
        if r["direction"] == "interior" and r["epsilon"] == 0.01
    ]
    assert len(selected) == 3 and all(
        not r["required_middle_step"] and r["relative_error"] > 0.1 for r in selected
    )
    assert (
        len(result["finite_difference"]) == 30
        and sum(r["required_middle_step"] for r in result["finite_difference"]) == 18
    )


def test_artifact_hash_tamper_blocks_NPZ_decode(cohort):
    path = cohort[0] / "saved/worker/arrays.npz"
    path.write_bytes(path.read_bytes() + b"tampered")
    result = evaluate(cohort)
    assert (
        result["issues"]
        and not result["implementation_agreement_pass"]
        and result["npz_arrays_decoded"] == 0
    )


def test_changed_protocol_cannot_obtain_credit_under_old_source_labels(cohort, monkeypatch):
    changed = copy.deepcopy(PROTOCOL)
    changed["candidate_criteria_pending_review"]["fd_middle_relative_error"] = 0.1
    monkeypatch.setattr(
        c, "read_json", lambda *args: pytest.fail("worker values read before protocol admission")
    )
    result = c.evaluate(cohort[0], cohort[1], changed, c.SOURCES, cohort[2])
    assert (
        result["issues"]
        and not result["implementation_agreement_pass"]
        and result["npz_arrays_decoded"] == 0
    )


@pytest.mark.parametrize("role", ["forward", "reverse"])
def test_permuted_native_ranges_fail_even_with_complete_sorted_coverage(cohort, role):
    counts = dict(attempted_calls=2, returned_calls=2, attempted_steps=255, returned_steps=255)

    def permute(events):
        answer = []
        inserted = False
        for event in events:
            if event["kind"] in ("native_attempt", "native_returned") and event["role"] == role:
                if not inserted:
                    ranges = [(127, 255), (0, 127)] if role == "forward" else [(0, 127), (127, 255)]
                    for start, end in ranges:
                        for kind, status in (
                            ("native_attempt", "attempted"),
                            ("native_returned", "returned"),
                        ):
                            row = dict(
                                event,
                                kind=kind,
                                status=status,
                                scheduled_start=start,
                                scheduled_end=end,
                                native_time_m=start + 1,
                                native_time_M=end,
                                native_steps=end - start,
                            )
                            if kind == "native_returned":
                                row["seconds"] = 0.001
                            answer.append(row)
                    inserted = True
                continue
            if role == "forward" and event["kind"] == "initial_forward_complete":
                event["native_counts"] = counts
            answer.append(event)
        for index, event in enumerate(answer):
            event.update(sequence=index, seconds_since_ledger=index / 100)
        events[:] = answer

    rebind(cohort, "saved", "events.jsonl", permute)
    if role == "forward":
        rebind(
            cohort,
            "saved",
            "result.json",
            lambda row: row.update(native_forward=counts, initial_native_forward=counts),
        )
    else:
        rebind(cohort, "saved", "result.json", lambda row: row.update(native_reverse=counts))
    result = evaluate(cohort)
    assert result["implementation_agreement_pass"] is False and result["npz_arrays_decoded"] == 0
    assert any(
        "ordered_" in issue["check"] and "native_traversal" in issue["check"]
        for issue in result["issues"]
    )


@pytest.mark.parametrize("kind", ["extra", "missing", "symlink", "worker_nested"])
def test_full_child_directory_denominator_and_flat_worker_output_are_enforced(cohort, kind):
    root = cohort[0]
    if kind == "extra":
        (root / "unregistered24th").mkdir()
    elif kind in ("missing", "symlink"):
        original = root / "saved"
        moved = root.parent / "moved_manufactured_saved"
        original.rename(moved)
        if kind == "symlink":
            original.symlink_to(moved, target_is_directory=True)
    else:
        (root / "saved/worker/unregistered_nested_output").mkdir()
    result = evaluate(cohort)
    assert result["issues"] and not result["implementation_agreement_pass"]
    assert result["npz_arrays_decoded"] == 0
