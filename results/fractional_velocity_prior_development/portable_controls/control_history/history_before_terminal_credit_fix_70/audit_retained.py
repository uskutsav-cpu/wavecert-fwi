"""Independent retained prior48 scalar audit, no producer imports or PDE solves.

Project input decoding/run reading requires an exact root terminal observation.
Default reads only the fixed prospective criteria. No source/model writes occur.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import platform
import random
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CRITERIA_SHA = "b06e9618b4d980ed0aa08c05a0da136d0ba3e37ecc422be4df411a5f3b96edf6"
ARMS = ("normalized_exact_data_only", "normalized_exact_fractional_update_prior")
PHYSICS_KEYS = ("factorizations", "forward_rhs")
BACKENDS = (
    "physics.exact_merit",
    "physics.coarse_dst_reference",
    "certificates.cached_merit_certificate",
    "optimization.paid_descent",
    "physics.helmholtz",
    "physics.coarse_stability",
    "physics.coarse_batch",
)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for raw in iter(lambda: stream.read(1024**2), b""):
            digest.update(raw)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def json_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def model_sha(values):
    return hashlib.sha256(b"".join(struct.pack("<d", float(x)) for x in values)).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def digest(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def safe_file(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError("missing/escaping retained artifact: " + str(name))
    return path


def json_file(path, maximum=32 * 1024**2):
    if Path(path).stat().st_size > maximum:
        raise ValueError("retained artifact byte cap exceeded")
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as stream:
            raw = stream.read(maximum + 1)
    else:
        raw = Path(path).read_bytes()
    if len(raw) > maximum:
        raise ValueError("retained JSON expansion cap exceeded")
    return json.loads(raw)


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


class Checks:
    def __init__(self):
        self.count = 0
        self.numeric_count = 0
        self.failures = []

    def require(self, ok, label, detail=None):
        self.count += 1
        if not bool(ok):
            self.failures.append(dict(label=label, detail=clean_json(detail)))

    def near(self, actual, expected, *, rtol, atol, label):
        self.numeric_count += 1
        ok = finite(actual) and finite(expected)
        allowance = atol + rtol * abs(actual) if ok else None
        error = abs(actual - expected) if ok else None
        self.require(
            ok and error <= allowance,
            label,
            dict(independent=actual, recorded=expected, error=error, allowance=allowance),
        )


def counted(tree, checks, label):
    checks.require(isinstance(tree, dict), label + "/tree")
    if not isinstance(tree, dict):
        return {}
    answer = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            answer[key] = counted(value, checks, label + "/" + str(key))
        else:
            checks.require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                label + "/" + str(key),
            )
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                answer[key] = value
    return answer


def monotone(before, after, checks, label):
    for key, value in before.items():
        other = after.get(key, {} if isinstance(value, dict) else 0)
        if isinstance(value, dict):
            checks.require(isinstance(other, dict), label + "/shape")
            if isinstance(other, dict):
                monotone(value, other, checks, label + "/" + key)
        else:
            checks.require(
                isinstance(other, int) and not isinstance(other, bool) and other >= value,
                label + "/" + key,
            )


def scalar_refinement(raw):
    """Explicit half-index interpolation in velocity, no project interpolator."""
    if len(raw) != 24 or any(len(r) != 24 for r in raw):
        raise ValueError("raw24x24 required")
    velocity = [[1 / math.sqrt(float(x)) for x in row] for row in raw]
    fine = []
    for z in range(49):
        za = (z + 1) / 2
        zi = int(math.floor(za))
        fz = za - zi
        z0, z1 = max(0, min(23, zi - 1)), max(0, min(23, zi))
        line = []
        for x in range(49):
            xa = (x + 1) / 2
            xi = int(math.floor(xa))
            fx = xa - xi
            x0, x1 = max(0, min(23, xi - 1)), max(0, min(23, xi))
            line.append(
                (1 - fz) * (1 - fx) * velocity[z0][x0]
                + fz * (1 - fx) * velocity[z1][x0]
                + (1 - fz) * fx * velocity[z0][x1]
                + fz * fx * velocity[z1][x1]
            )
        fine.append(line)
    truth = [1 / (v * v) for row in fine for v in row]
    return truth, fine


def initial_from_generic_filter(fine):
    import numpy as np
    from scipy.ndimage import gaussian_filter

    smoothed = gaussian_filter(np.asarray(fine, dtype=np.float64), sigma=3.2, mode="reflect")
    return [1 / (float(v) * float(v)) for v in smoothed.ravel()]


def quality(values, truth):
    if len(values) != len(truth) or any(not finite(x) or x <= 0 for x in [*values, *truth]):
        raise ValueError("matching finite positive model/truth vectors required")
    mass_norm = math.sqrt(math.fsum(t * t for t in truth))
    velocity = [1 / math.sqrt(t) for t in truth]
    v_norm = math.sqrt(math.fsum(v * v for v in velocity))
    return dict(
        relative_squared_slowness_L2=math.sqrt(
            math.fsum((a - b) ** 2 for a, b in zip(values, truth, strict=True))
        )
        / mass_norm,
        relative_physical_velocity_L2=math.sqrt(
            math.fsum((1 / math.sqrt(a) - v) ** 2 for a, v in zip(values, velocity, strict=True))
        )
        / v_norm,
    )


def scalar_R(values, initial, shape, spacing, length):
    if len(values) != len(initial) or len(values) != math.prod(shape):
        raise ValueError("prior shape mismatch")
    q = [math.sqrt(float(a) / float(b)) - 1 for a, b in zip(initial, values, strict=True)]
    rows, cols = shape
    terms = [v * v for v in q]
    for z in range(rows):
        for x in range(cols):
            i = z * cols + x
            if z + 1 < rows:
                terms.append((length / spacing) ** 2 * (q[i] - q[i + cols]) ** 2)
            if x + 1 < cols:
                terms.append((length / spacing) ** 2 * (q[i] - q[i + 1]) ** 2)
    return math.fsum(terms) / (2 * len(q))


def expected_geometry(p):
    def mapped(index):
        z, x = divmod(index, 24)
        return (2 * z + 1) * 49 + 2 * x + 1

    sources = [mapped(i) for i in p["geometry"]["original_source_indices"]]
    return dict(
        ordered_receivers=[mapped(i) for i in p["geometry"]["original_receiver_indices"]],
        ordered_blocks=[
            dict(source=s, frequency_hz=float(f), label=f"{s}@{f}")
            for f in p["frequencies_hz"]
            for s in sources
        ],
    )


def native_header(header, model, p, checks, label):
    checks.require(isinstance(header, dict), label + "/header")
    if not isinstance(header, dict):
        return
    for key in ("key", "model_sha256", "operator_sha256", "geometry_sha256", "data_sha256"):
        checks.require(digest(header.get(key)), label + "/" + key)
    checks.require(header.get("model_sha256") == model_sha(model), label + "/model")
    checks.require(finite(header.get("objective")) and header["objective"] >= 0, label + "/rawPhi")
    checks.require(
        header.get("geometry_sha256") == json_sha(expected_geometry(p)),
        label + "/registered_geometry",
    )
    checks.require(
        header.get("key")
        == json_sha(
            dict(
                operator=header.get("operator_sha256"),
                geometry=header.get("geometry_sha256"),
                data=header.get("data_sha256"),
            )
        ),
        label + "/receipt_key_hash",
    )
    frequencies = {str(float(f)) for f in p["frequencies_hz"]}
    checks.require(
        set(header.get("frequency_merit_routes", {})) == frequencies, label + "/all8frequencies"
    )
    checks.require(
        set(header.get("frequency_field_shapes", {})) == frequencies, label + "/all8fieldshapes"
    )
    checks.require(
        all(
            value == "forward_states" for value in header.get("frequency_merit_routes", {}).values()
        ),
        label + "/six_source_route",
    )
    checks.require(
        all(value == [2401, 6] for value in header.get("frequency_field_shapes", {}).values()),
        label + "/six_source_shapes",
    )
    checks.require(isinstance(header.get("gradient_cached"), bool), label + "/gradient_cache_flag")


def raw_objective(record, header, initial_phi, c, checks, label):
    checks.require(
        finite(record.get("objective")) and record["objective"] >= 0,
        label + "/nonnegative_stored_rawPhi",
    )
    checks.near(
        record["objective"],
        header["objective"],
        rtol=c["objective_rtol"],
        atol=c["objective_atol_initial_scale"]
        * max(initial_phi, c["objective_atol_initial_floor"]),
        label=label,
    )
    terms = record.get("per_frequency_objectives", [])
    checks.require(
        len(terms) == 8 and all(finite(t) and t >= 0 for t in terms),
        label + "/8stored_frequency_terms",
    )
    if len(terms) == 8 and all(finite(t) for t in terms):
        checks.near(
            math.fsum(terms),
            record["objective"],
            rtol=c["objective_rtol"],
            atol=c["objective_atol_initial_scale"]
            * max(initial_phi, c["objective_atol_initial_floor"]),
            label=label + "/stored_sum",
        )


def audit_row(
    row,
    payload,
    graph,
    raw,
    p,
    c,
    checks,
    trace_function,
    initial_builder=initial_from_generic_filter,
):
    label = f"run{row['run_index']}"
    start = len(checks.failures)
    truth, fine = scalar_refinement(raw)
    expected_initial = initial_builder(fine)
    models = {key: payload[key] for key in ("initial_model", "final_model")}
    for name, values in models.items():
        checks.require(
            isinstance(values, list) and len(values) == 2401, label + "/" + name + "/shape"
        )
        if len(values) != 2401 or any(not finite(x) or x <= 0 for x in values):
            raise ValueError("invalid full returned native model")
        checks.require(
            all(
                p["squared_slowness_bounds"][0] <= x <= p["squared_slowness_bounds"][1]
                for x in values
            ),
            label + "/" + name + "/box",
        )
    checks.require(
        all(
            abs(a - b) <= c["initial_model_atol"] + c["initial_model_rtol"] * abs(b)
            for a, b in zip(models["initial_model"], expected_initial, strict=True)
        ),
        label + "/independent_initial",
    )
    own_quality = {key: quality(value, truth) for key, value in models.items()}
    for name, record in graph["records"].items():
        checks.require(name in own_quality, label + "/graph_record_name")
        if name in own_quality:
            checks.require(
                record["model_sha256"] == model_sha(models[name]), label + "/graph_model_sha"
            )
            for metric, value in own_quality[name].items():
                checks.require(
                    finite(record.get(metric)) and record[metric] >= 0,
                    label + "/nonnegative_graph_metric/" + metric,
                )
                checks.near(
                    value,
                    record[metric],
                    rtol=c["quality_rtol"],
                    atol=c["quality_atol"],
                    label=label + "/graph_" + name + "/" + metric,
                )
    checks.require(set(graph["records"]) == set(models), label + "/both_graph_records")
    engine = payload["engine"]
    traced = trace_function(engine, checks, c, label + "/engine")
    weight = row["weight"]
    checks.require(payload.get("weight") == weight, label + "/weight")
    checks.require(
        payload.get("physical_acceptance") is None and payload.get("no_certificate_claim") is True,
        label + "/scope",
    )
    binding = row["common_binding"]
    checks.require(
        binding["initial_native_m_sha256"] == model_sha(models["initial_model"]),
        label + "/common_initial_sha",
    )
    checks.require(
        binding["raw_model_sha256"] == model_sha([float(x) for line in raw for x in line]),
        label + "/raw24sha",
    )
    checks.require(
        digest(binding.get("truth_sha256")) and digest(binding.get("all8_observations_sha256")),
        label + "/retained_truth_observation_digests",
    )
    selected = traced.get("last_eligible")
    method_valid = bool(
        traced.get("method_valid")
        and selected is not None
        and selected.get("model_sha256") == model_sha(models["final_model"])
    )
    checks.require(payload.get("method_valid") is method_valid, label + "/derived_method_valid")
    timing = row["timing"]
    wall = timing.get("actual_complete_return_seconds")
    inner = timing.get("inner_return_age")
    checks.require(finite(wall) and finite(inner) and 0 <= inner <= wall, label + "/complete_clock")
    checks.require(timing.get("complete_return_cap_seconds") == 8.0, label + "/strict_cap")
    clock_eligible = method_valid and finite(wall) and wall <= 8.0
    checks.require(timing.get("eligible") is clock_eligible, label + "/derived_timing_eligibility")
    checks.require(
        timing.get("event_seconds") == (wall if clock_eligible else None), label + "/late_censor"
    )
    graph_ok = isinstance(graph.get("failures"), list) and not graph["failures"]
    checks.require(isinstance(graph.get("failures"), list), label + "/graph_failure_list")
    checks.require(row.get("integrity_valid") is graph_ok, label + "/stored_graph_integrity_flag")
    factors = graph.get("factorizations")
    checks.require(
        isinstance(factors, int) and not isinstance(factors, bool) and factors in (16, 24, 32),
        label + "/stored_graph_factor_count",
    )
    checks.require(graph.get("forward_rhs") == 6 * factors, label + "/stored_graph_rhs")
    checks.require(
        finite(graph.get("maximum_relative_linear_residual"))
        and 0
        <= graph["maximum_relative_linear_residual"]
        <= c["linear_solve_relative_residual_maximum"],
        label + "/stored_graph_residual",
    )
    checks.require(
        graph.get("numeric_comparisons") == (3 + int(weight > 0) if method_valid else 0),
        label + "/stored_graph_scalar_count",
    )
    checks.require(
        graph.get("checks") == 6 * factors + 9 + (5 + int(weight > 0) if method_valid else 0),
        label + "/stored_graph_check_count",
    )
    preparation = counted(row.get("preparation_costs"), checks, label + "/preparation")
    checks.require(
        preparation.get("factorizations") == 8 and preparation.get("forward_rhs") == 48,
        label + "/fresh8LU48RHS",
    )
    checks.require(
        row.get("preparation_costs_unknown") is False, label + "/complete_preparation_count_scope"
    )
    native = payload.get("final_native_owner_cumulative_costs")
    if native is not None:
        counted(native, checks, label + "/one_final_native_snapshot")
        prior_native = engine.get("total_online_costs", {}).get("oracle", {}).get("native", {})
        monotone(prior_native, native, checks, label + "/native_final_dominates_engine_snapshot")
    tail = counted(payload["metric_tail_counts"], checks, label + "/metric_tail")
    checks.require(
        tail.get("native_snapshot_calls") == int(native is not None),
        label + "/one_final_snapshot_call",
    )
    checks.require(
        tail.get("native_evaluate_attempts") == int(method_valid)
        and tail.get("native_evaluate_completed") == int(method_valid),
        label + "/paid_final_native_dispatch",
    )
    checks.require(
        tail.get("prior_evaluation_attempts") == int(method_valid and weight > 0)
        and tail.get("prior_evaluations_completed") == int(method_valid and weight > 0),
        label + "/paid_final_prior_dispatch",
    )
    checks.require(
        tail.get("final_metric_prior_anchor_constructions", 0) == int(method_valid and weight > 0),
        label + "/extra_metric_prior_anchor",
    )
    adapter = engine.get("adapter")
    if adapter is not None:
        adapter_counts = counted(adapter["adapter_counts"], checks, label + "/adapter_arithmetic")
        checks.require(adapter.get("weight") == weight, label + "/adapter_weight")
        if weight == 0:
            checks.require(
                adapter_counts.get("prior_evaluations_completed") == 0,
                label + "/zero_arm_no_R_queries",
            )
    else:
        adapter_counts = None
    ratio = None
    if method_valid:
        initial_header = payload["initial_native_header"]
        current = adapter["current_accepted"]
        checks.require(
            initial_header == adapter["initial_accepted"], label + "/initial_header_identity"
        )
        raw0 = initial_header["native_data_receipt_header"]
        rawfinal = payload["final_raw_native_header"]
        native_header(raw0, models["initial_model"], p, checks, label + "/initial_native")
        native_header(rawfinal, models["final_model"], p, checks, label + "/final_native")
        phi0 = adapter["initial_paid_Phi0"]
        checks.require(
            finite(phi0) and phi0 > 0 and phi0 == raw0["objective"], label + "/positive_paid_Phi0"
        )
        checks.require(
            raw0["data_sha256"] == rawfinal["data_sha256"], label + "/same_initial_final_data"
        )
        for name, header in (("initial_model", raw0), ("final_model", rawfinal)):
            raw_objective(
                graph["records"][name],
                header,
                graph["records"]["initial_model"]["objective"],
                c,
                checks,
                label + "/" + name + "/stored_graph_Phi",
            )
        R = (
            scalar_R(
                models["final_model"],
                models["initial_model"],
                p["physics"]["shape"],
                p["physics"]["spacing"],
                p["prior_smoothing_length_km"],
            )
            if weight
            else 0.0
        )
        if weight:
            checks.require(
                finite(payload.get("final_R")) and payload["final_R"] >= 0,
                label + "/nonnegative_prior_R",
            )
            checks.near(
                R,
                payload["final_R"],
                rtol=c["prior_rtol"],
                atol=c["prior_atol"],
                label=label + "/independent_R",
            )
            checks.near(
                R,
                current["weighted_prior_value"] / weight,
                rtol=c["prior_rtol"],
                atol=c["prior_atol"],
                label=label + "/accepted_R",
            )
        else:
            checks.require(
                payload["final_R"] is None and current["weighted_prior_value"] == 0,
                label + "/zero_arm_prior_absence",
            )
        F = graph["records"]["final_model"]["objective"] / phi0 + weight * R
        checks.require(
            finite(payload.get("final_F")) and payload["final_F"] >= 0,
            label + "/nonnegative_final_F",
        )
        checks.near(
            F,
            payload["final_F"],
            rtol=c["composed_F_rtol"],
            atol=c["composed_F_atol"],
            label=label + "/stored_graph_and_scalar_F",
        )
        checks.near(
            F,
            engine["final_objective"],
            rtol=c["composed_F_rtol"],
            atol=c["composed_F_atol"],
            label=label + "/returned_composed_F",
        )
        checks.require(
            current["model_sha256"] == model_sha(models["final_model"]),
            label + "/last_composed_model",
        )
        checks.require(
            all(
                current["native_data_receipt_header"][key] == rawfinal[key]
                for key in (
                    "objective",
                    "key",
                    "model_sha256",
                    "operator_sha256",
                    "geometry_sha256",
                    "data_sha256",
                    "frequency_merit_routes",
                    "frequency_field_shapes",
                )
            ),
            label + "/actual_final_raw_header",
        )
        if raw0["model_sha256"] == rawfinal["model_sha256"]:
            checks.require(
                raw0["operator_sha256"] == rawfinal["operator_sha256"]
                and raw0["key"] == rawfinal["key"]
                and raw0["objective"] == rawfinal["objective"],
                label + "/same_model_same_native_receipt_scalar",
            )
        checks.require(
            initial_header["objective"] == 1.0
            and initial_header["normalized_data_value"] == 1.0
            and initial_header["weighted_prior_value"] == 0.0,
            label + "/both_arms_normalized_initial_F",
        )
        ratio = rawfinal["objective"] / phi0
        checks.near(
            ratio,
            payload["common_raw_Phi_ratio"],
            rtol=c["objective_ratio_rtol"],
            atol=c["objective_ratio_atol"],
            label=label + "/native_raw_ratio",
        )
        checks.near(
            ratio,
            current["normalized_data_value"],
            rtol=c["objective_ratio_rtol"],
            atol=c["objective_ratio_atol"],
            label=label + "/accepted_normalized_data",
        )
        checks.near(
            ratio,
            row["common_raw_Phi_ratio"],
            rtol=c["objective_ratio_rtol"],
            atol=c["objective_ratio_atol"],
            label=label + "/row_common_raw_ratio",
        )
    else:
        checks.require(
            all(
                payload.get(key) is None
                for key in ("final_raw_native_header", "final_R", "final_F", "common_raw_Phi_ratio")
            ),
            label + "/invalid_method_no_final_metric_credit",
        )
    for metric, value in own_quality["initial_model"].items():
        checks.require(
            finite(row["initial_quality"].get(metric)) and row["initial_quality"][metric] >= 0,
            label + "/nonnegative_initial_metric/" + metric,
        )
        checks.near(
            value,
            row["initial_quality"][metric],
            rtol=c["quality_rtol"],
            atol=c["quality_atol"],
            label=label + "/row_initial_quality/" + metric,
        )
    independently_eligible = clock_eligible and graph_ok and len(checks.failures) == start
    checks.require(
        row.get("final_study_eligible") is independently_eligible,
        label + "/independent_study_eligibility",
    )
    if independently_eligible:
        for metric, value in own_quality["final_model"].items():
            checks.require(
                finite(row["primary_quality"].get(metric)) and row["primary_quality"][metric] >= 0,
                label + "/nonnegative_final_metric/" + metric,
            )
            checks.near(
                value,
                row["primary_quality"][metric],
                rtol=c["quality_rtol"],
                atol=c["quality_atol"],
                label=label + "/row_final_quality/" + metric,
            )
    else:
        checks.require(row.get("primary_quality") is None, label + "/late_failed_quality_censored")
    return dict(
        run_index=row["run_index"],
        case_index=row["case_index"],
        repetition=row["repetition"],
        method=row["method"],
        weight=weight,
        eligible=independently_eligible and len(checks.failures) == start,
        initial_quality=own_quality["initial_model"],
        final_quality=own_quality["final_model"]
        if independently_eligible and len(checks.failures) == start
        else None,
        common_raw_Phi_ratio=ratio,
        wall=wall,
        common_binding=binding,
        saved_native_data_sha256=(
            payload["initial_native_header"]["native_data_receipt_header"]["data_sha256"]
            if payload.get("initial_native_header") is not None
            else None
        ),
        preparation_counts=preparation,
        final_native_snapshot=native,
        adapter_counts=adapter_counts,
        metric_tail_counts=tail,
        stored_postgraph_counts=dict(factorizations=factors, forward_rhs=graph.get("forward_rhs")),
        trace=traced,
        stored_graph_failures=graph.get("failures"),
        new_issue_count=len(checks.failures) - start,
    )


def bind_case_results(results, checks, case):
    """An acquisition/initializer mismatch invalidates both arms of the case."""
    selected = [
        r for r in results if r["case_index"] == case and r.get("common_binding") is not None
    ]
    before = len(checks.failures)
    if selected:
        checks.require(
            all(r["common_binding"] == selected[0]["common_binding"] for r in selected),
            f"case{case}/all6_common_initial_truth_observation_binding",
        )
    native_data = [
        r["saved_native_data_sha256"]
        for r in selected
        if r.get("saved_native_data_sha256") is not None
    ]
    checks.require(
        all(digest(value) for value in native_data)
        and (not native_data or len(set(native_data)) == 1),
        f"case{case}/all_native_data_headers_same",
    )
    if len(checks.failures) > before:
        for row in results:
            if row["case_index"] == case:
                row.update(eligible=False, final_quality=None, case_binding_integrity_valid=False)
    return len(checks.failures) == before


def apply_global_integrity_gate(summary, checks):
    value = dict(summary, global_integrity_gate_pass=not checks.failures)
    if checks.failures:
        value.update(all48_eligible=False, proposed_development_continuation_gate_pass=False)
    return value


def summarize(rows, c):
    full = {(case, rep, method) for case in range(8) for rep in range(3) for method in ARMS}
    groups = []
    pairs = []
    slots = {(r["case_index"], r["repetition"], r["method"]) for r in rows}
    complete = len(rows) == 48 and slots == full and all(r.get("eligible") is True for r in rows)
    for case in range(8):
        pair = {}
        for method in ARMS:
            selected = [r for r in rows if r["case_index"] == case and r["method"] == method]
            valid = (
                len(selected) == 3
                and {r["repetition"] for r in selected} == {0, 1, 2}
                and all(r.get("eligible") is True for r in selected)
            )

            def med(values):
                return sorted(values)[1]

            group = dict(
                case_index=case,
                method=method,
                eligible_repetitions=sum(r.get("eligible") is True for r in selected),
                all3_eligible=valid,
                initial_error=None,
                final_error=None,
                raw_ratio=None,
            )
            if valid:
                group.update(
                    initial_error=med(
                        [r["initial_quality"]["relative_physical_velocity_L2"] for r in selected]
                    ),
                    final_error=med(
                        [r["final_quality"]["relative_physical_velocity_L2"] for r in selected]
                    ),
                    raw_ratio=med([r["common_raw_Phi_ratio"] for r in selected]),
                )
            groups.append(group)
            pair[method] = group
        a, z = pair[ARMS[0]], pair[ARMS[1]]
        valid = a["all3_eligible"] and z["all3_eligible"]
        result = dict(
            case_index=case,
            all3_each_arm_eligible=valid,
            successful_case=False,
            harm_gate_pass=None,
        )
        if valid:
            result.update(
                initial_error=a["initial_error"],
                baseline_error=a["final_error"],
                candidate_error=z["final_error"],
                candidate_raw_Phi_ratio=z["raw_ratio"],
                successful_case=z["final_error"] <= 0.9 * a["initial_error"]
                and z["final_error"] <= 0.9 * a["final_error"]
                and z["raw_ratio"] <= 0.75,
                harm_gate_pass=z["final_error"] <= 1.1 * a["initial_error"],
            )
        pairs.append(result)
    return dict(
        all48_eligible=complete,
        case_arm_groups=groups,
        cases=pairs,
        proposed_development_continuation_gate_pass=complete
        and sum(r["successful_case"] for r in pairs) >= 6
        and all(r["harm_gate_pass"] is True for r in pairs),
        scientific_goal_complete=False,
    )


def source_digest(repo):
    files = [
        repo / "pyproject.toml",
        *repo.glob("requirements*.txt"),
        *(repo / "src").rglob("*.py"),
        *(p for p in (repo / "configs").rglob("*") if p.is_file()),
        *(p for p in (repo / "scripts").rglob("*") if p.is_file() and p.suffix in (".py", ".sh")),
        *(repo / ".github/workflows").glob("*.yml"),
    ]
    state = hashlib.sha256()
    count = 0
    for path in sorted(set(files)):
        if path.is_file():
            state.update(
                path.relative_to(repo).as_posix().encode() + b"\0" + file_sha(path).encode() + b"\0"
            )
            count += 1
    return state.hexdigest(), count


def read_cases(repo, p):
    import numpy as np

    payloads = {name: safe_file(repo, name).read_bytes() for name in p["inputs_sha256"]}
    if any(
        hashlib.sha256(raw).hexdigest() != p["inputs_sha256"][name]
        for name, raw in payloads.items()
    ):
        raise ValueError("frozen input bytes changed")
    prefix = "data/reference/subsurfacegen_parent_split/"
    metadata = list(
        csv.DictReader(io.StringIO(payloads[prefix + "validation_metadata.csv"].decode()))
    )
    manifest = list(csv.DictReader(io.StringIO(payloads[prefix + "split_manifest.csv"].decode())))
    with np.load(io.BytesIO(payloads[prefix + "validation.npz"]), allow_pickle=False) as archive:
        models = archive["models"]
        sources = archive["source_indices"]
        freqs = archive["frequencies_hz"]
        receivers = archive["receiver_indices"]
        shape = archive["shape"].tolist()
        spacing = float(archive["spacing"])
    geometry = p["original_geometry_context"]
    if (
        models.shape != (192, 24, 24)
        or models.dtype != np.float32
        or len(metadata) != 192
        or shape != [24, 24]
        or spacing != 0.05
        or sources.shape != (192,)
        or freqs.shape != (192,)
        or not np.isfinite(models).all()
        or not (models > 0).all()
        or receivers.tolist() != geometry["original_receiver_indices"]
    ):
        raise ValueError("pinned input shape/type/geometry/domain changed")
    if {r["slice_id"] for r in metadata} != {i["slice_id"] for i in p["identities"]}:
        raise ValueError("input case denominator differs")
    cases = []
    expected = {
        (s, f) for s in geometry["original_source_indices"] for f in geometry["frequencies_hz"]
    }
    for identity in p["identities"]:
        positions = [i for i, r in enumerate(metadata) if r["slice_id"] == identity["slice_id"]]
        if (
            len(positions) != 24
            or positions[0] != identity["array_index"]
            or {(int(sources[i]), float(freqs[i])) for i in positions} != expected
        ):
            raise ValueError("pinned source/old4frequency input coverage changed")
        for i in positions:
            r = metadata[i]
            if (
                r["model_id"] != identity["parent_id"]
                or r["split"] != "validation"
                or r["source_file"] != identity["source_file"]
                or r["source_sha256"] != identity["source_sha256"]
                or not np.array_equal(models[i], models[positions[0]])
            ):
                raise ValueError("pinned metadata/raw-model repeat changed")
        selected = [r for r in manifest if r["slice_id"] == identity["slice_id"]]
        if (
            len(selected) != 1
            or selected[0]["model_id"] != identity["parent_id"]
            or selected[0]["split"] != "validation"
        ):
            raise ValueError("source manifest identity changed")
        cases.append([[float(v) for v in line] for line in models[positions[0]]])
    return cases


def schedule_expected(p):
    rng = random.Random(2026110901)
    rows = []
    for repetition in range(3):
        for case in range(8):
            arms = [(ARMS[0], 0.0), (ARMS[1], 100.0)]
            rng.shuffle(arms)
            for position, (method, weight) in enumerate(arms):
                rows.append(
                    dict(
                        run_index=len(rows),
                        case_index=case,
                        identity=p["identities"][case],
                        repetition=repetition,
                        order_position=position,
                        method=method,
                        weight=weight,
                    )
                )
    return rows


def bind_run(run, repo, c, checks):
    status = json_file(safe_file(run, "status.json"))
    protocol_path = safe_file(run, "protocol.json")
    if file_sha(protocol_path) != c["protocol_sha256"]:
        raise ValueError("exact registered protocol bytes required before dynamic path use")
    p = json_file(protocol_path)
    checks.require(file_sha(protocol_path) == c["protocol_sha256"], "source/protocol")
    checks.require(file_sha(run / "driver_source.py") == c["driver_sha256"], "source/driver")
    checks.require(
        file_sha(run / "FROZEN48_SCHEDULE.json") == c["schedule_sha256"], "source/schedule"
    )
    for name, value in p["local_dependency_sha256"].items():
        checks.require(file_sha(safe_file(run, name)) == value, "source/local/" + name)
    for name, value in p["candidate_files_sha256"].items():
        checks.require(
            file_sha(safe_file(repo / p["original_candidate_relative_directory"], name)) == value,
            "source/preparation/" + name,
        )
    actual, count = source_digest(repo)
    checks.require(
        actual == c["production_sha256"] and count == c["production_file_count"],
        "source/production161",
    )
    before = status["source_before"]
    checks.require(
        before["driver_sha256"] == c["driver_sha256"]
        and before["protocol_sha256"] == c["protocol_sha256"],
        "source/before_driver_protocol",
    )
    checks.require(before["local"] == p["local_dependency_sha256"], "source/before_all_local")
    production = before["production"]
    checks.require(
        production["repository_source_sha256"] == c["production_sha256"],
        "source/before_native_production",
    )
    checks.require(
        production["protocol_sha256"] == p["original_protocol_sha256"],
        "source/original_preparation_protocol",
    )
    checks.require(
        production["adapter_sha256"] == p["candidate_files_sha256"]["adapter.py"]
        and production["verification_helpers_sha256"]
        == p["candidate_files_sha256"]["verification.py"],
        "source/raw_preparation_helpers",
    )
    expected_backends = {
        "wavecert." + name: file_sha(
            safe_file(repo / "src/wavecert", name.replace(".", "/") + ".py")
        )
        for name in BACKENDS
    }
    checks.require(production["backend_modules"] == expected_backends, "source/all_native_backends")
    go = json_file(safe_file(run, "root_go.json"))
    expected_go = dict(
        authorization="GO_CLASSICAL_PRIOR48_DEVELOPMENT",
        driver_sha256=c["driver_sha256"],
        protocol_sha256=c["protocol_sha256"],
        schedule_sha256=c["schedule_sha256"],
        production_source_sha256=c["production_sha256"],
        independent_source_review_clear=True,
    )
    checks.require(all(go.get(k) == v for k, v in expected_go.items()), "source/separate_root_GO")
    checks.require(
        status.get("status")
        in ("running", "failed_or_interrupted", "complete_engineering_attempts"),
        "status/known_terminal_or_preserved_prefix_state",
    )
    checks.require(
        finite(status.get("shared_bootstrap_seconds")) and status["shared_bootstrap_seconds"] >= 0,
        "cost/shared_bootstrap_time",
    )
    complete_path = run / "complete_receipt.json"
    complete = json_file(complete_path) if complete_path.is_file() else None
    if complete is not None:
        checks.require(
            status.get("status") == "complete_engineering_attempts"
            and all(row.get("status") == "complete" for row in status["runs"]),
            "complete/all48_attempts_terminal",
        )
        checks.require(
            status.get("numerical_audits_all_pass")
            is all(row.get("integrity_valid") is True for row in status["runs"]),
            "complete/numerical_graph_flag",
        )
        checks.require(
            {k: v for k, v in complete.items() if k != "artifact_sha256"} == status,
            "complete/status_identity",
        )
        checks.require(
            set(complete["artifact_sha256"])
            == {x.name for x in run.iterdir() if x.is_file() and x.name != "complete_receipt.json"},
            "complete/full_artifact_denominator",
        )
        for name, value in complete["artifact_sha256"].items():
            checks.require(file_sha(safe_file(run, name)) == value, "complete/artifact/" + name)
        checks.require(status.get("source_after") == before, "source/before_after_identity")
    checks.require(
        status.get("all48_scheduled_before_project_inputs") is True,
        "schedule/before_project_inputs",
    )
    rows = status["runs"]
    expected = schedule_expected(p)
    checks.require(len(rows) == 48, "schedule/full48_denominator")
    saved = json_file(run / "FROZEN48_SCHEDULE.json")
    checks.require(
        len(saved) == 48
        and all({k: r.get(k) for k in x} == x for r, x in zip(saved, expected, strict=True)),
        "schedule/independent_frozen_seed_order",
    )
    checks.require(
        len(rows) == 48
        and all({k: r.get(k) for k in x} == x for r, x in zip(rows, expected, strict=True)),
        "schedule/actual_all48_order_identity_weights",
    )
    checks.require(p["continuation_gates"] == c["continuation_gates"], "fixed/development_gates")
    for key in ("engine_deadline_seconds", "complete_return_cap_seconds"):
        checks.require(p[key] == c[key], "fixed/" + key)
    checks.require(
        p["frequencies_hz"] == [0.25, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
        and p["observation_blocks"] == 48,
        "fixed/common8",
    )
    return status, p, complete


def incomplete_row(run, row, checks):
    label = f"run{row['run_index']}"
    checks.require(
        row.get("final_study_eligible") is False and row.get("primary_quality") is None,
        label + "/incomplete_no_credit",
    )
    information = {}
    for artifact_key, hash_key in (
        ("return_artifact", "return_sha256"),
        ("audit_artifact", "audit_sha256"),
        ("failure_prefix_artifact", "failure_prefix_sha256"),
    ):
        if artifact_key in row:
            path = safe_file(run, row[artifact_key])
            checks.require(file_sha(path) == row.get(hash_key), label + "/" + artifact_key)
            value = json_file(path)
            if artifact_key == "failure_prefix_artifact":
                checks.require(
                    value.get("quality_success_claim") is False, label + "/failed_prefix_no_quality"
                )
                checks.require(
                    isinstance(value.get("attempt"), dict),
                    label + "/all_available_attempt_branches",
                )
                information["retained_failure_branches"] = list(value.get("attempt", {}))
                information["current_native_work_unknown"] = value.get(
                    "current_native_work_unknown"
                )
                information["final_native_snapshot_available"] = value.get(
                    "final_native_snapshot_available"
                )
    return dict(
        run_index=row["run_index"],
        case_index=row["case_index"],
        repetition=row["repetition"],
        method=row["method"],
        eligible=False,
        initial_quality=None,
        final_quality=None,
        common_raw_Phi_ratio=None,
        status=row["status"],
        incomplete_work_unknown=True,
        retained_evidence=information,
    )


def terminal_context(terminal, run):
    """Local audited copy is hash-bound; original runner paths are context only."""
    exit_code = terminal.get("actual_exit_code")
    if exit_code is not None and (not isinstance(exit_code, int) or isinstance(exit_code, bool)):
        raise ValueError("terminal exit code must be actual integer or explicit unknown")
    if terminal.get("observed_terminal") is not True or terminal.get("run_directory") != str(run):
        raise ValueError("root must independently observe and hash-bind this exact audited copy")
    remote = {
        key: value
        for key, value in terminal.items()
        if key.startswith("remote_")
        or key in ("child_exit_evidence", "child_exit_receipt_sha256", "supervisor_exit_evidence")
    }
    return dict(
        audited_run_directory=str(run),
        original_executed_run_directory=terminal.get("original_executed_run_directory"),
        root_reported_remote_execution_context=remote,
        scope="Artifact relocation is allowed only with exact local status/complete/artifact identities. Original runner paths and remote job/child-exit evidence are root-reported context, not locally regenerated timing or independently queried remote evidence.",
    )


def execute(args, c):
    run, repo = args.run.resolve(), args.repo.resolve()
    terminal_path = args.terminal_observation.resolve()
    if file_sha(terminal_path) != args.expected_terminal_sha256:
        raise ValueError("exact root terminal observation hash required")
    terminal = json_file(terminal_path)
    execution_context = terminal_context(terminal, run)
    if terminal.get("status_sha256") != file_sha(safe_file(run, "status.json")):
        raise ValueError("terminal observation/status identity changed")
    complete_path = run / "complete_receipt.json"
    actual_complete = file_sha(complete_path) if complete_path.is_file() else None
    if terminal.get("complete_receipt_sha256") != actual_complete:
        raise ValueError("terminal completed receipt identity changed")
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(x) for x in (run, repo, HERE)):
        raise ValueError("fresh external audit output required")
    output.mkdir()
    prefix = dict(
        status="attempted",
        criteria_sha256=CRITERIA_SHA,
        auditor_sha256=file_sha(Path(__file__)),
        trace_source_sha256=args.expected_trace_sha256,
        terminal_observation_sha256=args.expected_terminal_sha256,
        no_project_PDE=True,
        no_producer_imports=True,
        full_planned_denominator=48,
    )
    (output / "attempt_started.json").write_text(json.dumps(prefix, indent=2) + "\n")
    checks = Checks()
    try:
        trace_path = HERE / "trace_checks.py"
        if file_sha(trace_path) != args.expected_trace_sha256:
            raise ValueError("exact independent trace source pin required")
        spec = importlib.util.spec_from_file_location("_independent_prior48_trace", trace_path)
        trace_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trace_module)
        status, p, complete = bind_run(run, repo, c, checks)
        if checks.failures:
            raise ValueError("run source/input-independent metadata bindings failed")
        forbidden_modules = (
            "run_prior48",
            "composed_exact_adapter",
            "bounded_exact",
            "fractional_velocity_prior",
            "independent_saved_models",
        )
        checks.require(
            not any(
                name.startswith("wavecert") or name in forbidden_modules for name in sys.modules
            ),
            "no_producer_imports_before_decode",
        )
        cases = read_cases(repo, p)
        results = []
        for row in status["runs"]:
            if row["status"] != "complete":
                results.append(incomplete_row(run, row, checks))
                continue
            try:
                paths = {
                    key: safe_file(run, row[key]) for key in ("return_artifact", "audit_artifact")
                }
                for key, path in paths.items():
                    checks.require(
                        file_sha(path)
                        == row[
                            {"return_artifact": "return_sha256", "audit_artifact": "audit_sha256"}[
                                key
                            ]
                        ],
                        f"run{row['run_index']}/" + key,
                    )
                result = audit_row(
                    row,
                    json_file(paths["return_artifact"]),
                    json_file(paths["audit_artifact"]),
                    cases[row["case_index"]],
                    p,
                    c,
                    checks,
                    trace_module.audit_engine,
                )
                results.append(result)
            except (
                KeyError,
                TypeError,
                ValueError,
                IndexError,
                OverflowError,
                ZeroDivisionError,
                OSError,
            ) as exc:
                checks.require(
                    False,
                    f"run{row['run_index']}/unexpected_saved_schema",
                    f"{type(exc).__name__}: {exc}",
                )
                results.append(
                    dict(
                        run_index=row["run_index"],
                        case_index=row["case_index"],
                        repetition=row["repetition"],
                        method=row["method"],
                        eligible=False,
                        initial_quality=None,
                        final_quality=None,
                        common_raw_Phi_ratio=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        for case in range(8):
            bind_case_results(results, checks, case)
        summary = summarize(results, c)
        if complete is not None:
            recorded = status["continuation"]
            checks.require(
                recorded["all48_eligible"] == summary["all48_eligible"], "summary/all48_eligibility"
            )
            checks.require(
                recorded["proposed_development_continuation_gate_pass"]
                == summary["proposed_development_continuation_gate_pass"],
                "summary/fixed_continuation_gate",
            )
            for a, b in zip(summary["cases"], recorded["cases"], strict=True):
                for key in (
                    "case_index",
                    "all3_each_arm_eligible",
                    "successful_case",
                    "harm_gate_pass",
                ):
                    checks.require(a[key] == b[key], f"summary/case{a['case_index']}/" + key)
                for key in (
                    "initial_error",
                    "baseline_error",
                    "candidate_error",
                    "candidate_raw_Phi_ratio",
                ):
                    if key in a:
                        checks.near(
                            a[key],
                            b[key],
                            rtol=c["quality_rtol"]
                            if key != "candidate_raw_Phi_ratio"
                            else c["objective_ratio_rtol"],
                            atol=c["quality_atol"]
                            if key != "candidate_raw_Phi_ratio"
                            else c["objective_ratio_atol"],
                            label=f"summary/case{a['case_index']}/" + key,
                        )
        for name, value in p["inputs_sha256"].items():
            checks.require(file_sha(safe_file(repo, name)) == value, "after/input/" + name)
        checks.require(
            source_digest(repo) == (c["production_sha256"], c["production_file_count"]),
            "after/production",
        )
        checks.require(file_sha(terminal_path) == args.expected_terminal_sha256, "after/terminal")
        checks.require(file_sha(run / "status.json") == terminal["status_sha256"], "after/status")
        checks.require(
            (file_sha(complete_path) if complete_path.is_file() else None) == actual_complete,
            "after/complete_receipt",
        )
        if complete is not None:
            for name, value in complete["artifact_sha256"].items():
                checks.require(file_sha(safe_file(run, name)) == value, "after/artifact/" + name)
        checks.require(file_sha(args.criteria) == CRITERIA_SHA, "after/fixed_criteria")
        checks.require(
            file_sha(Path(__file__)) == args.expected_source_sha256
            and file_sha(trace_path) == args.expected_trace_sha256,
            "after/auditor_sources",
        )
        checks.require(
            not any(
                name.startswith("wavecert") or name in forbidden_modules for name in sys.modules
            ),
            "no_producer_imports_after_audit",
        )
        import numpy as np
        import scipy

        summary = apply_global_integrity_gate(summary, checks)
        report = dict(
            status="RETAINED_SCALAR_AUDIT_COMPLETE",
            producer_environment=status.get("environment"),
            shared_producer_bootstrap_seconds=status.get("shared_bootstrap_seconds"),
            auditor_environment=dict(
                python=sys.version,
                platform=platform.platform(),
                numpy=np.__version__,
                scipy=scipy.__version__,
            ),
            checks=checks.count,
            numeric_comparisons=checks.numeric_count,
            issues=checks.failures,
            software_integrity_checks_pass=not checks.failures,
            verified_full_study_credit=not checks.failures
            and complete is not None
            and terminal.get("actual_exit_code") == 0
            and summary["all48_eligible"],
            rows=results,
            summary=summary,
            source_bindings=prefix,
            execution_context=execution_context,
            graph_scope="Stored source-bound rawPhi/observation/solve-residual evidence inspected only, no independent matrix/PDE replay. Paid nativePhi0 normalization separately checked against graph initialPhi.",
            cost_scope="Per-row preparation plus ONE final cumulative native snapshot; adapter/metric/prior/storedpostgraph distinct, no overlapping snapshot sums or dispatch-toRHS equivalence.",
            limits="Retained header/chain consistency cannot reconstruct live receipt ownership, observation arrays not retained in producer payload, or omitted middle history. Data/observation digests are structurally bound and repeated, not independently regenerated without PDE. NumPy decoder and SciPy Gaussian initializer shared, scalartruth/prior/errors independent.",
            scientific_goals_complete=False,
            project_PDE_calls=0,
        )
        (output / "report.json").write_text(
            json.dumps(clean_json(report), indent=2, allow_nan=False) + "\n"
        )
        return 0 if not checks.failures else 2
    except BaseException as exc:
        (output / "failure.json").write_text(
            json.dumps(
                dict(
                    error=f"{type(exc).__name__}: {exc}",
                    checks=checks.count,
                    issues=checks.failures,
                    valid_study_credit=False,
                    full_planned_denominator=48,
                ),
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-after-root-terminal", action="store_true")
    parser.add_argument("--criteria", type=Path, default=HERE / "criteria.json")
    for key in ("run", "repo", "terminal-observation", "output"):
        parser.add_argument("--" + key, type=Path)
    for key in ("source", "trace", "terminal"):
        parser.add_argument("--expected-" + key + "-sha256")
    args = parser.parse_args()
    if file_sha(args.criteria) != CRITERIA_SHA:
        raise ValueError("fixed prospective criteria changed")
    c = json_file(args.criteria)
    if not args.execute_after_root_terminal:
        print(
            json.dumps(
                dict(
                    status="METADATA_ONLY_NO_PROJECT_ARRAYS_RUN_OR_PDE",
                    criteria_sha256=CRITERIA_SHA,
                    planned_runs=48,
                )
            )
        )
        return 0
    if not all(
        (
            args.run,
            args.repo,
            args.terminal_observation,
            args.output,
            args.expected_source_sha256,
            args.expected_trace_sha256,
            args.expected_terminal_sha256,
        )
    ):
        parser.error(
            "exact source/trace/terminal pins, run/repo/freshoutput and root terminal observation required"
        )
    if file_sha(Path(__file__)) != args.expected_source_sha256:
        raise ValueError("auditor source changed")
    return execute(args, c)


if __name__ == "__main__":
    raise SystemExit(main())
