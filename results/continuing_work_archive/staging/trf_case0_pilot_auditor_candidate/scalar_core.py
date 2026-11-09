"""Independent stdlib retained-array/scalar pilot audit; no numerical or producer import.

ZIP/NPY admission is adapted from the review_inputs.py decoder bound by the
395-check input review (81bcc4a1 report). It adds complex128 and pure helpers;
it never executes headers, loads pickle, derives a PDE, or imports worker code.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

PHYSICS_KEYS = ("factorizations", "forward_rhs", "tangent_rhs", "adjoint_rhs",
    "stability_rhs", "dense_inverse_rhs", "svd_evaluations", "cache_hits", "analytic_bound_evaluations")
RAW_SCHEMA = {"models": ((192, 24, 24), "<f4"), "source_indices": ((192,), "<i8"),
    "frequencies_hz": ((192,), "<f4"), "wavefields": ((192, 24, 24, 2), "<f4"),
    "receiver_indices": ((20,), "<i8"), "shape": ((2,), "<i8"), "spacing": ((), "<f8")}
INITIAL_SCHEMA = {"initial_m": ((8, 2401), "<f8"), "case_indices": ((8,), "<i8"), "shape": ((2,), "<i8")}
PREPARED_SCHEMA = {"raw24_m": ((24, 24), "<f8"), "initial49_m": ((2401,), "<f8"), "truth49_m": ((2401,), "<f8")}
OBSERVATION_SCHEMA = {"observed": ((48, 20), "<c16")}
RETURNED_SCHEMA = {"initial_m": ((2401,), "<f8"), "selected_m": ((2401,), "<f8")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class Array:
    shape: tuple
    descr: str
    body: bytes
    values: tuple

    @property
    def payload_sha256(self):
        return sha(self.body)


def decode_npy(member, expected_shape, expected_descr):
    require(len(member) >= 10 and member[:6] == b"\x93NUMPY", "NPY magic/extent")
    version = tuple(member[6:8])
    require(version in ((1, 0), (2, 0)), "NPY version")
    prefix = 10 if version == (1, 0) else 12
    require(len(member) >= prefix, "NPY length field extent")
    header_len = struct.unpack("<H" if prefix == 10 else "<I", member[8:prefix])[0]
    require(0 < header_len <= 4096, "bounded NPY header")
    raw_header = member[prefix:prefix + header_len]
    require(len(raw_header) == header_len and raw_header.endswith(b"\n"), "NPY header extent/newline")
    header = ast.literal_eval(raw_header.decode("latin1"))
    require(type(header) is dict and set(header) == {"shape", "descr", "fortran_order"}, "NPY header keys")
    require(type(header["shape"]) is tuple and all(type(n) is int and n >= 0 for n in header["shape"])
        and header["shape"] == expected_shape and header["descr"] == expected_descr
        and header["fortran_order"] is False, "exact C-order numeric NPY schema")
    formats = {"<f4": (4, "<f"), "<f8": (8, "<d"), "<i8": (8, "<q"), "<c16": (16, "<dd")}
    require(expected_descr in formats, "numeric dtype allowlist")
    width, code = formats[expected_descr]
    body = member[prefix + header_len:]
    require(len(body) == math.prod(expected_shape) * width, "exact NPY payload and EOF")
    rows = struct.iter_unpack(code, body)
    if expected_descr == "<c16":
        values = tuple(complex(real, imag) for real, imag in rows)
        require(all(math.isfinite(v.real) and math.isfinite(v.imag) for v in values), "finite complex payload")
    else:
        values = tuple(row[0] for row in rows)
        require(all(finite(v) for v in values), "finite real/integer payload")
    return Array(expected_shape, expected_descr, body, values)


def read_npz(path, schema, expected_sha256, *, byte_cap=2 * 1024 * 1024, on_member=None):
    """Bound actual container and inflation; caller must admit all pins first."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= byte_cap, "bounded regular NPZ file")
    with path.open("rb") as stream:
        raw = stream.read(byte_cap + 1)
    require(len(raw) <= byte_cap and sha(raw) == expected_sha256, "NPZ bytes/hash admission")
    result = {}
    widths = {"<f4": 4, "<f8": 8, "<i8": 8, "<c16": 16}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        require(len(infos) == len(schema) and {i.filename for i in infos} == {k + ".npy" for k in schema},
            "exact unique NPZ members")
        for info in infos:
            key = info.filename[:-4]
            shape, descr = schema[key]
            require(descr in widths, "NPZ numeric dtype allowlist")
            cap = math.prod(shape) * widths[descr] + 4112
            require(not info.flag_bits & 1 and info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                and info.file_size <= cap, "bounded nonencrypted ZIP member")
            with archive.open(info) as stream:
                member = stream.read(cap + 1)
                require(len(member) <= cap and not stream.read(1), "bounded actual ZIP inflation/EOF")
            result[key] = decode_npy(member, shape, descr)
            if on_member is not None:
                on_member(key)
    return result


def f64_bytes(values):
    require(all(finite(v) for v in values), "finite float64 serialization")
    return struct.pack("<" + str(len(values)) + "d", *values)


def checked_model(values, *, interior=False):
    require(len(values) == 2401 and all(finite(v) for v in values), "finite 2401-vector model")
    lower, upper = 1 / 36, 1.
    if interior:
        require(all(lower + 1e-10 < v < upper - 1e-10 for v in values), "strict direct initializer box")
    else:
        require(all(lower <= v <= upper for v in values), "selected direct model box")


def reconstruct_truth(raw_m):
    """Own scalar edge-padded velocity bilinear interpolation at half indices."""
    require(len(raw_m) == 576 and all(finite(v) and v > 0 for v in raw_m), "positive raw24 model")
    velocity = [1 / math.sqrt(v) for v in raw_m]

    def padded(row, col):
        return velocity[min(23, max(0, row - 1)) * 24 + min(23, max(0, col - 1))]

    fine_m = []
    for row in range(1, 50):
        r0, rw = divmod(row, 2)
        rw *= .5
        for col in range(1, 50):
            c0, cw = divmod(col, 2)
            cw *= .5
            value = ((1 - rw) * ((1 - cw) * padded(r0, c0) + cw * padded(r0, c0 + 1))
                + rw * ((1 - cw) * padded(r0 + 1, c0) + cw * padded(r0 + 1, c0 + 1)))
            fine_m.append(1 / (value * value))
    require(all(finite(v) and v > 0 for v in fine_m), "finite independently reconstructed truth")
    return tuple(fine_m)


def velocity_error(model, truth):
    require(len(model) == len(truth) and len(model) > 0, "metric vector shape")
    require(all(finite(v) and v > 0 for v in (*model, *truth)), "positive metric vectors")
    reference = [1 / math.sqrt(v) for v in truth]
    difference = [1 / math.sqrt(v) - r for v, r in zip(model, reference)]
    denominator = math.sqrt(math.fsum(v * v for v in reference))
    result = math.sqrt(math.fsum(v * v for v in difference)) / denominator
    require(finite(result), "finite relative velocity L2")
    return result


def compare_vectors(actual, reference, *, rtol=2e-12, atol=2e-14):
    require(len(actual) == len(reference) and len(actual) > 0, "numeric comparison shape")
    require(all(finite(v) for v in (*actual, *reference)), "finite numeric comparison")
    require(all(math.isclose(a, b, rel_tol=rtol, abs_tol=atol) for a, b in zip(actual, reference)),
        "independent scalar vector comparison")
    return {"terms": len(actual), "maximum_absolute_difference": max(abs(a - b) for a, b in zip(actual, reference))}


def audit_trace(records, summary):
    """Replay only complete retained streams; omitted middle is a commitment."""
    require(type(records) is list and type(summary) is dict, "recorded trace types")
    total = summary["total_count"]
    require(type(total) is int and total >= 0 and summary["finalized_count"] == total
        and summary["retained_count"] == len(records) == min(total, 16)
        and summary["omitted_count"] == total - len(records)
        and summary["head_limit"] == summary["tail_limit"] == 8
        and summary["pending_record"] is None and summary["pending_record_unsealed"] is False,
        "bounded trace denominator/finalization")
    digest = summary["chain_sha256"]
    require(type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "trace commitment")
    if total == len(records):
        own = hashlib.sha256(b"bounded-exact-record-stream-v1\0").digest()
        for index, record in enumerate(records, 1):
            payload = canonical(record)
            own = hashlib.sha256(own + index.to_bytes(8, "big") + len(payload).to_bytes(8, "big") + payload).digest()
        require(own.hex() == digest, "independent complete retained trace commitment")
    return {"total_records": total, "retained_records": len(records), "omitted_middle_records": total - len(records),
        "complete_stream_replayed": total == len(records), "omitted_middle_independently_replayed": False}


def audit_engine_steps(engine):
    steps = engine["steps"]
    trace = audit_trace(steps, engine["step_trace"])
    previous_end, previous_iteration, retained_library_count = 0., -1, 0
    complete_library, library_records = True, []
    for step in steps:
        start, end = step["elapsed_start"], step["elapsed_end"]
        require(finite(start) and finite(end) and 0 <= previous_end <= start <= end
            and finite(step["seconds"]) and math.isclose(end - start, step["seconds"], rel_tol=2e-12, abs_tol=2e-14)
            and type(step["iteration"]) is int and step["iteration"] > previous_iteration, "bounded engine step chronology")
        previous_end, previous_iteration = end, step["iteration"]
        require(type(step["accepted"]) is bool, "recorded step accepted flag")
        if step["accepted"]:
            require(finite(step["accepted_elapsed_seconds"]) and start <= step["accepted_elapsed_seconds"] <= min(end, 7.5)
                and finite(step["objective_before"]) and finite(step["objective_after"])
                and 0 <= step["objective_after"] < step["objective_before"], "accepted bounded step time/scalar")
        count_tree(step["online_cost_delta"])
        observed = audit_trace(step["library_evaluations"], step["library_evaluations_summary"])
        retained_library_count += observed["total_records"]
        complete_library = complete_library and observed["complete_stream_replayed"]
        library_records.extend(step["library_evaluations"])
    library = engine["all_library_evaluation_trace"]
    require(type(library["total_count"]) is int and library["total_count"] >= retained_library_count
        and (trace["omitted_middle_records"] > 0 or library["total_count"] == retained_library_count)
        and type(library["chain_sha256"]) is str and len(library["chain_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in library["chain_sha256"]), "bounded all-library commitment denominator")
    if trace["complete_stream_replayed"] and complete_library:
        own = hashlib.sha256(b"bounded-exact-record-stream-v1\0").digest()
        for index, record in enumerate(library_records, 1):
            raw = canonical(record)
            own = hashlib.sha256(own + index.to_bytes(8, "big") + len(raw).to_bytes(8, "big") + raw).digest()
        require(own.hex() == library["chain_sha256"], "complete retained all-library commitment")
    return {"step_trace": trace, "retained_steps_library_records": retained_library_count,
        "all_library_records": library["total_count"], "LSMR_internal_algebra_replayed": False}


def audit_selection(payload):
    require(payload["policy_valid"] is True and payload["error"] is None
        and payload["status"] in ("solver_success", "solver_terminated", "budget_exhausted", "maximum_steps")
        and payload["truth_used_for_policy"] is False, "known eligible truth-free returned policy")
    selected = payload["selected"]
    require(type(selected) is dict and finite(selected["elapsed_seconds"])
        and 0 <= selected["elapsed_seconds"] <= 7.5, "selected completion by7.5")
    initial_sha = sha(f64_bytes(payload["initial_model"]))
    selected_sha = sha(f64_bytes(payload["selected_model"]))
    require(selected_sha == selected["model_sha256"], "serialized selected model hash")
    phi0, phi = payload["Phi0"], selected["raw_Phi"]
    require(finite(phi0) and phi0 > 0 and finite(phi) and 0 <= phi <= phi0, "genuine positive normalization/nonincreasing scalar Phi")
    require(finite(payload["common_Phi_ratio"]) and math.isclose(payload["common_Phi_ratio"], phi / phi0,
        rel_tol=2e-12, abs_tol=2e-14), "raw Phi/Phi0 scalar consistency")
    if payload["exact_engine"] is not None:
        engine = payload["exact_engine"]
        require(engine["status"] == payload["status"] and engine["initial_objective"] == 1., "normalized exact engine status/Phi0")
        records = engine["checkpoints"]
        trace = audit_trace(records, engine["checkpoint_trace"])
        steps = audit_engine_steps(engine)
        require(records and records[0]["kind"] == "initial" and records[0]["model_sha256"] == initial_sha,
            "completed paid initial checkpoint")
        previous_time, previous_objective = -1., math.inf
        for index, row in enumerate(records):
            require(finite(row["elapsed_seconds"]) and previous_time <= row["elapsed_seconds"] <= 7.5
                and row["within_budget"] is True and finite(row["objective"]) and row["objective"] >= 0,
                "retained eligible checkpoint chronology")
            require(row["kind"] == ("initial" if index == 0 else "accepted")
                and row["objective"] < previous_objective, "strict accepted endpoint scalar decrease")
            previous_time, previous_objective = row["elapsed_seconds"], row["objective"]
        last = records[-1]
        require(last["model_sha256"] == selected_sha and last["elapsed_seconds"] == selected["elapsed_seconds"]
            and last["objective"] == engine["final_objective"]
            and math.isclose(last["objective"] * phi0, phi, rel_tol=2e-12, abs_tol=2e-14), "last eligible exact endpoint only")
        normalization = payload["normalization_prefix"]
        require(normalization["current_model_sha256"] == selected_sha
            and normalization["initial_header"]["model_sha256"] == initial_sha
            and normalization["initial_header"]["raw_objective"] == phi0, "native normalized current/initial receipt binding")
        for row in records:
            prefix_at_most(row["online_costs"]["oracle"]["native"], payload["native_cumulative_once"])
        return {"selected_model_sha256": selected_sha, "selected_elapsed_seconds": selected["elapsed_seconds"],
            "receipt_scope": "Authentic frozen-source receipt/scalar consistency; no independent Helmholtz objective reproduction.",
            "checkpoint_trace": trace, "steps": steps, "selection": "Last retained eligible exact native checkpoint; omitted middle source-bound."}
    events = payload["TRF_scalar_events"]
    require(type(events) is list and events, "TRF scalar event stream")
    eligible, previous_time, previous_phi, residuals = [], -1., math.inf, set()
    for index, event in enumerate(events):
        require(type(event["sequence"]) is int and event["sequence"] == index
            and finite(event["elapsed_seconds"]) and event["elapsed_seconds"] >= max(0., previous_time), "TRF recorded scalar chronology")
        previous_time = event["elapsed_seconds"]
        require(event["kind"] in ("residual_completed", "initial", "accepted_callback", "equal_callback_no_new_endpoint"), "TRF recorded event kind")
        if event["kind"] != "equal_callback_no_new_endpoint":
            identity = event["model_sha256"]
            require(type(identity) is str and len(identity) == 64 and all(c in "0123456789abcdef" for c in identity), "TRF model identity")
        if event["kind"] == "residual_completed":
            require(event["elapsed_seconds"] <= 7.5, "completed residual within absolute optimizer budget")
            residuals.add(event["model_sha256"])
        if event["kind"] in ("initial", "accepted_callback"):
            require(event["within_budget"] is (event["elapsed_seconds"] <= 7.5)
                and event["model_sha256"] in residuals and (event["kind"] == "initial") is (not eligible)
                and finite(event["raw_Phi"]) and 0 <= event["raw_Phi"] < previous_phi, "genuine strict TRF promotion scalar")
            if event["within_budget"]:
                eligible.append(event)
                previous_phi = event["raw_Phi"]
    require(eligible and eligible[0]["kind"] == "initial" and eligible[0]["model_sha256"] == initial_sha
        and eligible[0]["raw_Phi"] == phi0, "completed TRF initial fallback")
    last = eligible[-1]
    require(last["model_sha256"] == selected_sha and last["elapsed_seconds"] == selected["elapsed_seconds"]
        and last["raw_Phi"] == phi, "last eligible native TRF callback only; result.x/rejected/late ignored")
    return {"selected_model_sha256": selected_sha, "selected_elapsed_seconds": selected["elapsed_seconds"],
        "recorded_scalar_events": len(events), "selection": "Last eligible initial/strict native accepted callback, never result.x.",
        "receipt_scope": "Authentic frozen-source receipt/scalar consistency; no independent Helmholtz objective reproduction."}


def audit_numeric_slot(raw, frozen, prepared, observations, returned, payload, result, protocol):
    """Independent scalar truth/errors and exact stored-array identity bindings."""
    case = protocol["cases"][0]
    require(result["slot"]["case_index"] == result["slot"]["repetition"] == 0, "case0 repetition0 only")
    models = raw["models"]
    first = models.body[:576 * 4]
    require(all(models.body[index * 576 * 4:(index + 1) * 576 * 4] == first for index in range(24)), "24 identical case0 models")
    raw_m = tuple(v[0] for v in struct.iter_unpack("<f", first))
    require(f64_bytes(raw_m) == prepared["raw24_m"].body
        and sha(f64_bytes(raw_m)) == case["raw_float64_sha256"], "exact source-pinned raw24 model")
    require(frozen["case_indices"].values == tuple(range(8)) and frozen["shape"].values == (49, 49), "frozen initializer indexing")
    exact_initial = frozen["initial_m"].body[:2401 * 8]
    require(exact_initial == prepared["initial49_m"].body == returned["initial_m"].body
        == f64_bytes(payload["initial_model"]) and sha(exact_initial) == case["initial_float64_sha256"], "exact retained initializer bytes")
    checked_model(returned["initial_m"].values, interior=True)
    checked_model(returned["selected_m"].values)
    require(returned["selected_m"].body == f64_bytes(payload["selected_model"]), "exact serialized selected model artifact")
    independent_truth = reconstruct_truth(raw_m)
    truth_comparison = compare_vectors(prepared["truth49_m"].values, independent_truth)
    survey = {"frequencies_hz": protocol["frequencies_hz"], "sources": protocol["source_indices"], "receivers": protocol["receiver_indices"]}
    binding = {"case_index": 0, "raw24_m_sha256": prepared["raw24_m"].payload_sha256,
        "initial49_m_sha256": prepared["initial49_m"].payload_sha256,
        "truth49_m_sha256": prepared["truth49_m"].payload_sha256,
        "all8_observations_sha256": observations["observed"].payload_sha256, "survey_sha256": sha(canonical(survey))}
    require(result["case_binding"] == binding, "new stored truth/initial/all8 observation/survey binding")
    metrics = result["post_return_quality"]
    initial_error = velocity_error(returned["initial_m"].values, independent_truth)
    selected_error = velocity_error(returned["selected_m"].values, independent_truth)
    require(metrics["computed_after_selection_returned"] is True and metrics["used_for_selection"] is False, "post-return truth metric scope")
    compare_vectors((metrics["initial_relative_velocity_error"], metrics["selected_relative_velocity_error"]), (initial_error, selected_error))
    selection = audit_selection(payload)
    return {"case_binding": binding, "independent_initial_relative_velocity_L2": initial_error,
        "independent_selected_relative_velocity_L2": selected_error, "truth_comparison": truth_comparison,
        "selection": selection, "all8_observation_numerical_PDE_reproduction_performed": False,
        "full_objective_numerical_PDE_reproduction_performed": False}


def physics_counts(value):
    require(type(value) is dict and set(value) == set(PHYSICS_KEYS), "complete physical counter keys")
    require(all(type(v) is int and v >= 0 for v in value.values()), "actual nonnegative integer counters")
    return dict(value)


def count_tree(value):
    require(type(value) is dict, "nested counter dictionary")
    result = {}
    for key, item in value.items():
        require(type(key) is str, "counter string key")
        if type(item) is dict:
            result[key] = count_tree(item)
        else:
            require(type(item) is int and item >= 0, "nonnegative integer counter tree")
            result[key] = item
    return result


def prefix_at_most(prefix, final):
    count_tree(prefix)
    count_tree(final)
    for key, value in prefix.items():
        if type(value) is dict:
            require(type(final.get(key)) is dict, "counter prefix branch")
            prefix_at_most(value, final[key])
        else:
            require(value <= final.get(key, 0), "counter prefix exceeds final cumulative")


def expected_counts(**values):
    result = {key: 0 for key in PHYSICS_KEYS}
    require(set(values) <= set(result), "expected counter keys")
    result.update(values)
    return result


def audit_costs(payload, context, frequencies, receivers, sources):
    """Consumed prefixes, not summed overlapping snapshots or request estimates."""
    preparation = physics_counts(payload["preparation_costs_disjoint"])
    require(preparation == expected_counts(factorizations=8, forward_rhs=48), "actual complete preparation8LU48RHS")
    prefix = payload["preparation_prefix"]
    require(prefix["observation_dispatches_attempted"] == prefix["observation_dispatches_completed"] == 8
        and prefix["unknown_work"] is False and prefix["observation_call_inflight"] is False
        and prefix["last_known_observation_counts"] == preparation, "complete known preparation prefix")
    require(not any(k in prefix for k in ("original_observation_exception", "partial_observation_persistence_error")),
        "no preparation failure receives complete credit")
    require(payload["unknown_work"] is False and payload["preparation_work_unknown"] is False, "no unknown work credit")
    native = count_tree(payload["native_cumulative_once"])
    require(set(native) == {"physics", "oracle"}, "genuine native cumulative branches")
    policy_physics = physics_counts(native["physics"])
    require(context.get("native_final_cumulative_0") == native
        and not any(k.startswith("native_snapshot_failure_") for k in context), "one final native cumulative binding")
    require(not any(k.startswith("native_final_cumulative_") and k != "native_final_cumulative_0" for k in context),
        "one policy native owner")
    map_total = expected_counts()
    map_sets = 0
    component = payload["component_prefix"]
    normalization = payload["normalization_prefix"]
    if component is not None:
        require(normalization is None, "one normalization policy branch")
        require(component["native"]["unknown_since_prefix"] is False
            and component["map_builder"]["unknown_work_present"] is False, "known component/map work")
        prefix_at_most(component["native"]["last_known_cumulative"], native)
        rows = component["map_builder"]["cold_owner_batches"]
        require(len(rows) % 8 == 0, "no partial map construction set receives complete credit")
        missing = tuple(receiver for receiver in dict.fromkeys(receivers) if receiver not in sources)
        require(len(missing) == 14, "registered missing receiver geometry")
        for index, row in enumerate(rows):
            require(type(row["index"]) is int and row["index"] == index and row["unknown_work"] is False
                and all(row[k] is True for k in ("factory_attempted", "factory_completed", "map_dispatch_attempted",
                    "map_completed", "cold_owner_verified", "actual_complete_batch_counts_verified", "final_snapshot_completed")),
                "complete distinct cold map batch")
            require(tuple(row["receivers"]) == missing, "ordered missing receiver batch")
            identity = tuple(row["point_identity"])
            require(len(identity) == 5 and all(type(v) is str and len(v) == 64
                and all(c in "0123456789abcdef" for c in v) for v in identity), "map point identity")
            frequency = row["frequency_hz"]
            require(frequency == frequencies[index % 8], "ordered8-frequency map construction set")
            require(tuple(rows[index - index % 8]["point_identity"]) == identity, "same point within one map set")
            require(physics_counts(row["before_physics"]) == expected_counts(), "cold map before counts")
            actual = physics_counts(row["last_known_physics_prefix"])
            require(actual == expected_counts(factorizations=1, stability_rhs=14), "one fresh map LU14RHS")
            require(finite(row["elapsed_seconds"]) and row["elapsed_seconds"] >= 0, "map elapsed metadata")
            for key, value in actual.items():
                map_total[key] += value
        # Canonical point identity may recur after a later point evicts the
        # last operator. That is new consumed work, not a cache hit or a
        # forbidden duplicate. Every fresh sequential8-batch set counts once.
        map_sets = len(rows) // 8
    else:
        require(normalization is not None and normalization["unknown_work"] is False
            and normalization["failed_dispatch_work_may_exceed_prefix"] is False, "known exact normalization work")
        prefix_at_most(normalization["native_last_known_cumulative"], native)
    total = {key: preparation[key] + policy_physics[key] + map_total[key] for key in PHYSICS_KEYS}
    return {"preparation_disjoint": preparation, "policy_native_once": policy_physics,
        "map_disjoint": map_total, "complete_distinct_map_sets": map_sets, "consumed_total": total,
        "overlapping_snapshots_summed": False, "request_counts_are_RHS": False,
        "LSMR_iterations_and_internal_FLOPs_or_copies": None}


if __name__ == "__main__":
    print(json.dumps({"status": "HELD_SCALAR_AUDITOR_CORE_SOURCE_ONLY", "execution_authorized": False,
        "numerical_or_producer_imports": 0, "PDE_calls": 0, "scientific_goals_complete": False}))
