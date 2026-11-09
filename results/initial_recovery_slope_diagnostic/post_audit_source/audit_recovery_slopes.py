"""Independent scalar recovery-slope post-audit. No PDE or optimizer.

NumPy only decodes/canonicalizes NPZ files; sqrt/fsum arithmetic independently
reconstructs all eight initial cases, finite differences and1440 endpoint rows.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import gzip
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

SOURCE9F = "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"
PRODUCER799 = "7998631fc3542cdd42ebce0bcc236d87755a33647b1135327ebebda6f8dec46e"
PROTOCOL_B110 = "b1104b4159ffc0065fb5b3a47a1c552a8cb94fc329daa8f7b315418452d7a8d2"
PRODUCER_RECEIPT = "f6ec029cb50f6e8e6518256b6e397face4ecdf2ed88e8d40bc5c084d21a5a5f4"
WITNESS_RECEIPT = "71b9de071e4d04557784805bf9bfc4ea55171c8b07e9c9625c866d8ab68aaba2"
BENCHMARK_RECEIPT = "e4939a3dfeda9315273e7f8918c8d2654ef177636d47cfee2862aa351586779a"
ORIGINAL_PROTOCOL = "fb6b304fd719441be8cd6eb8d101ca864d0856254f40618583d8d986fd1c945c"
METRICS = ("half_relative_squared_m_error", "half_relative_squared_velocity_error")
RELATIVE = ("relative_squared_slowness_L2", "relative_physical_velocity_L2")
MACHINE_EPS = sys.float_info.epsilon


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(Path(path).read_bytes())


def owned(root, name):
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("registered single filename required")
    path = (root / name).resolve()
    if path.parent != root.resolve() or not path.is_file():
        raise ValueError("missing/escaping artifact")
    return path


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def length(a):
    return math.sqrt(math.fsum(x * x for x in a))


def difference(a, b):
    return [x - y for x, y in zip(a, b, strict=True)]


def metrics(m, truth):
    velocity = [1 / math.sqrt(x) for x in m]
    target = [1 / math.sqrt(x) for x in truth]
    em = math.fsum((x - t) ** 2 for x, t in zip(m, truth, strict=True)) / dot(truth, truth)
    ev = math.fsum((x - t) ** 2 for x, t in zip(velocity, target, strict=True)) / dot(target, target)
    return dict(zip(METRICS + RELATIVE, (em / 2, ev / 2, math.sqrt(em), math.sqrt(ev)), strict=True))


def sign(value, scale):
    threshold = 1e-10 * max(scale, 1e-300)
    literal = "positive" if value > 0 else "negative" if value < 0 else "zero"
    return {"value": value, "scale": scale, "near_zero_threshold": threshold,
            "literal_sign": literal, "numerical_sign": "near_zero" if abs(value) <= threshold else literal}


def feasibility(values):
    low, high = 1 / 36, 1.0
    finite = [x for x in values if math.isfinite(x)]
    below, above = [x for x in finite if x < low], [x for x in finite if x > high]

    def near_low(x):
        return abs(x - low) <= 64 * MACHINE_EPS * max(abs(x), abs(low))

    def near_high(x):
        return abs(x - high) <= 64 * MACHINE_EPS * max(abs(x), abs(high))

    return {"cell_count": len(values), "finite_cells": len(finite),
            "nonfinite_cells": len(values) - len(finite),
            "nonpositive_finite_cells": sum(x <= 0 for x in finite),
            "finite_positive": len(finite) == len(values) and all(x > 0 for x in finite),
            "finite_minimum": min(finite) if finite else None, "finite_maximum": max(finite) if finite else None,
            "lower_bound": low, "upper_bound": high, "literal_below_lower_cells": len(below),
            "literal_above_upper_cells": len(above),
            "literal_box_feasible": len(finite) == len(values) and not below and not above,
            "exact_lower_bound_cells": sum(x == low for x in finite),
            "exact_upper_bound_cells": sum(x == high for x in finite),
            "near_lower_FP_band_cells": sum(near_low(x) for x in finite),
            "near_upper_FP_band_cells": sum(near_high(x) for x in finite),
            "literal_below_but_inside_lower_FP_band_cells": sum(near_low(x) for x in below),
            "literal_above_but_inside_upper_FP_band_cells": sum(near_high(x) for x in above)}


class Audit:
    def __init__(self):
        self.checks = self.numeric = 0
        self.issues, self.near_zero_literal_disagreements = [], []
        self.phase = "before_inputs"

    def require(self, condition, context, detail=None):
        self.checks += 1
        if not bool(condition):
            self.issues.append({"context": context, "detail": detail})

    def near(self, own, saved, context, *, atol=2e-13, rtol=5e-11, scale=None):
        self.numeric += 1
        if own is None or saved is None:
            self.require(own is None and saved is None, context)
            return
        magnitude = max(abs(own), abs(saved), 1e-12) if scale is None else max(scale, 1e-12)
        allowance = atol + rtol * magnitude
        self.require(math.isfinite(own) and math.isfinite(saved) and abs(own - saved) <= allowance,
                     context, {"independent": own, "recorded": saved, "allowance": allowance})

    def fields(self, own, saved, context):
        for key, value in own.items():
            if isinstance(value, (int, bool)) or value is None:
                self.require(saved.get(key) == value, context + "/" + key,
                             {"independent": value, "recorded": saved.get(key)})
            else:
                self.near(value, saved[key], context + "/" + key)

    def sign(self, own, saved, context):
        self.near(own["value"], saved["value"], context + "/value", scale=own["scale"],
                  atol=1e-12 * max(own["scale"], 1e-12))
        for key in ("scale", "near_zero_threshold"):
            self.near(own[key], saved[key], context + "/" + key, atol=0, rtol=2e-12)
        recomputed_saved = sign(saved["value"], saved["scale"])
        self.require(all(saved[k] == recomputed_saved[k] for k in ("literal_sign", "numerical_sign")),
                     context + "/recorded_value_sign_self_consistency")
        if own["literal_sign"] != saved["literal_sign"]:
            both_near = (abs(own["value"]) <= own["near_zero_threshold"]
                         and abs(saved["value"]) <= saved["near_zero_threshold"])
            self.require(both_near, context + "/independent_literal_sign_or_both_near_zero")
            self.near_zero_literal_disagreements.append(
                {"context": context, "independent": own, "recorded": saved, "both_near_zero": both_near})
        self.require(own["numerical_sign"] == saved["numerical_sign"], context + "/independent_numerical_sign")


def manifest(root, expected, key):
    receipt = owned(root, "complete_receipt.json")
    if checksum(receipt) != expected:
        raise ValueError("immutable complete-receipt SHA mismatch")
    complete, status = load(receipt), load(owned(root, "status.json"))
    if {k: v for k, v in complete.items() if k != key} != status:
        raise ValueError("upstream complete/status mismatch")
    for name, expected_sha in complete[key].items():
        if checksum(owned(root, name)) != expected_sha:
            raise ValueError("registered artifact SHA mismatch: " + name)
    return status, {"complete_receipt_sha256": expected, "registered_hashes_checked": len(complete[key])}


def source_pin(repo):
    paths = [repo / "pyproject.toml"] + list((repo / "src").rglob("*.py"))
    paths += [p for p in (repo / "configs").rglob("*") if p.is_file()]
    paths += [p for p in (repo / "scripts").rglob("*") if p.suffix in {".py", ".sh"}]
    paths += list((repo / ".github/workflows").glob("*.yml")) + list(repo.glob("requirements*.txt"))
    digest, count = hashlib.sha256(), 0
    for path in sorted(set(paths)):
        if path.is_file():
            digest.update(path.relative_to(repo).as_posix().encode() + b"\0" + checksum(path).encode() + b"\0")
            count += 1
    return {"sha256": digest.hexdigest(), "files": count}


def array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def decode(path, names, descriptors=None):
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in names:
            if name not in archive.files:
                result[name] = None
                continue
            values = archive[name].copy()
            if values.shape != (2401,) or values.dtype != np.float64:
                raise ValueError("raw vector schema differs")
            if descriptors is not None:
                record = descriptors[name]
                if record["sha256"] != array_sha(values) or record["shape"] != [2401]:
                    raise ValueError("raw vector descriptor differs")
            result[name] = [float(x) for x in values]
            if not all(math.isfinite(x) for x in result[name]):
                raise ValueError("nonfinite raw vector")
    return result


def metric_slopes(m, truth, direction):
    velocity, target = [1 / math.sqrt(x) for x in m], [1 / math.sqrt(x) for x in truth]
    v_direction = [-0.5 * v / x * p for v, x, p in zip(velocity, m, direction, strict=True)]
    dm, dv = dot(truth, truth), dot(target, target)
    return {
        METRICS[0]: {"analytic": dot(difference(m, truth), direction) / dm,
            "uncancelled_derivative_scale": math.fsum((abs(x) + abs(t)) * abs(p)
                for x, t, p in zip(m, truth, direction, strict=True)) / dm,
            "metric_evaluation_scale": (dot(m, m) + dm) / (2 * dm)},
        METRICS[1]: {"analytic": dot(difference(velocity, target), v_direction) / dv,
            "uncancelled_derivative_scale": math.fsum((abs(v) + abs(t)) * abs(p)
                for v, t, p in zip(velocity, target, v_direction, strict=True)) / dv,
            "metric_evaluation_scale": (dot(velocity, velocity) + dv) / (2 * dv)}}


def audit_direction(audit, m, truth, gradient, direction, saved, context):
    audit.require(saved["available"] is True, context + "/available")
    dn, gn = length(direction), length(gradient)
    correction = difference(truth, m)
    audit.near(dn, saved["direction_l2_norm"], context + "/direction_norm", atol=0, rtol=2e-12)
    data_dd = dot(gradient, direction)
    audit.sign(sign(data_dd, gn * dn), saved["data_directional_derivative"], context + "/dataDD")
    cosine = None if dn == 0 or length(correction) == 0 else dot(direction, correction) / (dn * length(correction))
    if cosine is None:
        audit.require(saved["native_m_cosine_direction_to_truth_correction"] is None, context + "/undefined_cosine")
    else:
        audit.sign(sign(cosine, 1), saved["native_m_cosine_direction_to_truth_correction"], context + "/cosine")
    slopes = metric_slopes(m, truth, direction)
    for name, row in slopes.items():
        stored = saved["half_relative_squared_metric_derivatives"][name]
        audit.fields(row, stored, context + "/" + name)
        audit.sign(sign(row["analytic"], row["uncancelled_derivative_scale"]), stored["sign"], context + "/" + name + "/sign")
        literal = data_dd < 0 and row["analytic"] > 0
        numerical = (sign(data_dd, gn * dn)["numerical_sign"] == "negative"
                     and sign(row["analytic"], row["uncancelled_derivative_scale"])["numerical_sign"] == "positive")
        audit.require(saved["numerically_classified_local_opposition"][name] is numerical, context + "/opposition/" + name)
        if saved["literal_local_opposition"][name] is not literal:
            audit.require(abs(row["analytic"]) <= 1e-10 * max(row["uncancelled_derivative_scale"], 1e-300),
                          context + "/literal_opposition_diff_only_near_zero/" + name)
    fd = saved["finite_difference_metric_validation"]
    if dn == 0:
        audit.require(fd["status"] == "zero_direction_trivial" and fd["actual_steps"] == [], context + "/zero_FD")
        return {"dataDD": data_dd, "slopes": slopes, "FD_status": "zero_direction_trivial"}
    ceiling = min(x / abs(p) for x, p in zip(m, direction, strict=True) if p != 0)
    steps = [1e-4 * ceiling, 5e-5 * ceiling]
    audit.require(fd["status"] == "computed", context + "/FD_computed")
    audit.near(ceiling, fd["positivity_ceiling_step"], context + "/FD_ceiling", atol=0, rtol=2e-12)
    samples = collections.defaultdict(list)
    for i, step in enumerate(steps):
        audit.near(step, fd["actual_steps"][i], context + f"/FD_step{i}", atol=0, rtol=2e-12)
        plus = [x + step * p for x, p in zip(m, direction, strict=True)]
        minus = [x - step * p for x, p in zip(m, direction, strict=True)]
        metadata = {"dimensionless_epsilon": [1e-4, 5e-5][i], "actual_step_in_native_m_coordinates": step,
            "maximum_relative_coordinate_perturbation": max(abs(step * p) / x for x, p in zip(m, direction, strict=True)),
            "minimum_positive_model_plus": min(plus), "minimum_positive_model_minus": min(minus),
            "box_violations_plus": sum(x < 1 / 36 or x > 1 for x in plus),
            "box_violations_minus": sum(x < 1 / 36 or x > 1 for x in minus), "no_projection_or_clipping": True}
        audit.fields(metadata, fd["step_metadata"][i], context + f"/FD_metadata{i}")
        qp, qm = metrics(plus, truth), metrics(minus, truth)
        for name, row in slopes.items():
            value = (qp[name] - qm[name]) / (2 * step)
            proxy = 64 * MACHINE_EPS * row["metric_evaluation_scale"] / step
            allowance = 2e-7 * row["uncancelled_derivative_scale"] + 8 * proxy
            stored = fd["metric_checks"][name][i]
            samples[name].append((value, proxy))
            comparison_atol = 8 * proxy + 1e-12 * max(row["uncancelled_derivative_scale"], 1e-12)
            audit.near(value, stored["FD"], context + f"/FD{i}/" + name, atol=comparison_atol)
            audit.near(proxy, stored["roundoff_cancellation_proxy"], context + f"/FDproxy{i}/" + name, atol=0, rtol=2e-12)
            audit.near(allowance, stored["fixed_agreement_allowance"], context + f"/FDallowance{i}/" + name, atol=0, rtol=2e-12)
            audit.near(abs(value - row["analytic"]), stored["analytic_absolute_discrepancy"],
                       context + f"/FDdiscrepancy{i}/" + name, atol=8 * proxy)
            audit.require(abs(value - row["analytic"]) <= allowance, context + f"/independent_FD_original_allowance{i}/" + name)
    for name, values in samples.items():
        estimate = abs(values[1][0] - values[0][0]) / 3
        audit.near(estimate, fd["metric_checks"][name][1]["two_scale_difference_over_three"],
                   context + "/two_scale/" + name, atol=8 * (values[0][1] + values[1][1]))
    return {"dataDD": data_dd, "slopes": slopes, "FD_steps": steps, "FD_status": "computed",
            "scope": "unprojected local numerical statistics; no finite clipped-step or causal inference"}


def statistics_of(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {"count": len(values), "mean": math.fsum(values) / len(values), "median": statistics.median(values),
            "minimum": min(values), "maximum": max(values)}


def perform(repo, witness, benchmark, producer, audit):
    global np
    audit.phase = "input_hashes_and_source"
    ws, wp = manifest(witness, WITNESS_RECEIPT, "artifact_sha256")
    bs, bp = manifest(benchmark, BENCHMARK_RECEIPT, "artifacts")
    ps, pp = manifest(producer, PRODUCER_RECEIPT, "artifact_sha256")
    if ps["status"] != "complete_post_outcome_software_diagnostic" or ps["result"] != "clear":
        raise ValueError("pinned producer must be terminal and clear")
    if checksum(owned(producer, "diagnostic_source.py")) != PRODUCER799 or checksum(owned(producer, "protocol.json")) != PROTOCOL_B110:
        raise ValueError("producer source/protocol pins differ")
    if checksum(owned(benchmark, "historical_candidate_protocol.json")) != ORIGINAL_PROTOCOL:
        raise ValueError("original protocol pin differs")
    original = load(owned(benchmark, "historical_candidate_protocol.json"))
    source_before = source_pin(repo)
    if source_before["sha256"] != SOURCE9F:
        raise ValueError("current production source differs from9f")
    audit.require(ps["source_before"] == ps["source_after"] == source_before, "producer/source")
    audit.require(ws["source_before"]["sha256"] == SOURCE9F and ws["source_before"] == ws["source_after"], "witness/source")
    audit.require(bs["source_before"]["candidate"]["repository_source_sha256"] == SOURCE9F
                  and bs["source_before"] == bs["source_after"], "benchmark/source")
    for relative, expected in original["inputs_sha256"].items():
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo) or checksum(path) != expected:
            raise ValueError("original raw input file SHA differs")
    import numpy as np

    control = metric_slopes([1.0], [4.0], [2.0])
    audit.require(metrics([1.0], [4.0])[METRICS[0]] == 9 / 32 and metrics([1.0], [4.0])[METRICS[1]] == 1 / 2,
                  "independent_manufactured_metric_controls")
    audit.require(control[METRICS[0]]["analytic"] == -3 / 8 and control[METRICS[1]]["analytic"] == -2,
                  "independent_manufactured_derivative_controls")
    cases, case_reports, recorded_cases = {}, [], []
    audit.require(len(ws["cases"]) == len(ps["case_slots"]) == 8, "all8_case_slots")
    for row in ws["cases"]:
        index = row["case_index"]
        audit.phase = f"initial_case_{index}"
        raw = load(owned(witness, row["receipt_artifact"]))
        raw_path = owned(witness, raw["raw_arrays_artifact"])
        if checksum(raw_path) != raw["raw_arrays_file_sha256"]:
            raise ValueError("initial raw NPZ SHA differs")
        values = decode(raw_path, ("initial_model", "independent_truth_model", "independent_true_gradient",
                                  "approximate_gradient", "direction"), raw["arrays"])
        if any(values[k] is None for k in ("initial_model", "independent_truth_model", "independent_true_gradient")):
            raise ValueError("required initial raw vectors missing")
        cases[index] = values
        saved = load(owned(producer, f"case_{index:02d}_slopes.json"))
        recorded_cases.append(saved)
        m, truth, gradient = (values[k] for k in ("initial_model", "independent_truth_model", "independent_true_gradient"))
        audit.require(saved["identity"] == original["identities"][index], f"case{index}/identity")
        audit.fields(metrics(m, truth), saved["initial_metrics"], f"case{index}/initial_metrics")
        audit.fields(feasibility(m), saved["initial_model_box_feasibility"], f"case{index}/initial_box")
        audit.fields(feasibility(truth), saved["truth_model_box_feasibility"], f"case{index}/truth_box")
        audit.near(length(gradient), saved["independent_true_gradient_l2_norm"], f"case{index}/gradient_norm", atol=0, rtol=2e-12)
        correction = difference(truth, m)
        truth_dd = sign(dot(gradient, correction), length(gradient) * length(correction))
        audit.sign(truth_dd, saved["data_gradient_dot_toward_truth_correction"], f"case{index}/truthDD")
        report = {"case_index": index, "initial_box": feasibility(m), "truth_box": feasibility(truth),
                  "truthDD": truth_dd, "directions": {}}
        exact = None if length(gradient) == 0 else [-g / length(gradient) for g in gradient]
        for name, direction in (("actual_raw_proposal", values["direction"]),
                                ("exact_normalized_negative_data_gradient", exact)):
            if direction is None:
                audit.require(saved["directions"][name]["available"] is False, f"case{index}/{name}/unknown_not_zero")
                report["directions"][name] = {"available": False}
            else:
                report["directions"][name] = audit_direction(
                    audit, m, truth, gradient, direction, saved["directions"][name], f"case{index}/{name}")
        if values["direction"] is not None and values["approximate_gradient"] is not None:
            ag, direction = values["approximate_gradient"], values["direction"]
            audit.near(length(ag), saved["approximate_gradient_l2_norm"], f"case{index}/approx_norm", atol=0, rtol=2e-12)
            audit.sign(sign(dot(ag, direction), length(ag) * length(direction)),
                       saved["approximate_data_gradient_dot_actual_proposal"], f"case{index}/approxDD")
        case_reports.append(report)
    table = load(owned(producer, "all_original_internal_endpoint_baselines.json"))
    rows_by_key = {(r["run_index"], r["internal_budget_seconds"]): r for r in table["all_rows"]}
    audit.require(len(table["all_rows"]) == len(rows_by_key) == 1440, "all1440_unique_endpoint_slots")
    groups, endpoint_reports = collections.defaultdict(list), []
    for row in bs["runs"]:
        index, case_index = row["run_index"], row["case_index"]
        audit.phase = f"original_run_{index}"
        core = load(owned(benchmark, row["core_artifact"]))
        old = load(owned(benchmark, row["audit_artifact"]))
        arrays = decode(owned(benchmark, row["models_artifact"]),
                        ("prepared_initial", "internal_budget_0", "internal_budget_1", "internal_budget_2", "internal_budget_3"))
        initial, truth = arrays["prepared_initial"], cases[case_index]["independent_truth_model"]
        audit.require(array_sha(initial) == array_sha(cases[case_index]["initial_model"]), f"run{index}/initial_binding")
        baseline = metrics(initial, truth)
        for bi, budget in enumerate((1.0, 2.0, 4.0, 8.0)):
            saved = rows_by_key[(index, budget)]
            endpoint, model = core["internal_budget_endpoints"][bi], arrays[f"internal_budget_{bi}"]
            audit.require(array_sha(model) == endpoint["model_sha256"] == saved["model_sha256"], f"run{index}/budget{budget}/model")
            paid = endpoint["completed_paid_checkpoint"]
            final = row["status"] == "complete" and row["final_study_eligible"] is True
            audit.require(saved["paid_checkpoint"] is paid and saved["terminal_final_study_eligible"] is final
                          and saved["eligible_internal_diagnostic"] is (paid and final), f"run{index}/budget{budget}/eligibility")
            audit.require(saved["identity"] == row["identity"] and saved["mode"] == row["measurement_mode"]
                          and saved["method"] == row["method"] and saved["repetition"] == row["repetition"],
                          f"run{index}/budget{budget}/dimensions")
            result = metrics(model, truth)
            audit.fields(result, saved["raw_saved_state_quality"], f"run{index}/budget{budget}/metrics")
            audit.fields(baseline, saved["initial_quality"], f"run{index}/budget{budget}/baseline")
            audit.fields(feasibility(model), saved["saved_endpoint_box_feasibility_and_active_bounds"], f"run{index}/budget{budget}/box")
            old_quality = old["credited_reconstruction_quality"]["internal_budget_states"][bi]
            audit.require(old_quality["eligible_paid_checkpoint"] is paid
                          and saved["original_credited_relative_quality"] == old_quality["quality"],
                          f"run{index}/budget{budget}/original_quality_retention")
            if paid:
                for name in RELATIVE:
                    audit.near(result[name], old_quality["quality"][name], f"run{index}/budget{budget}/old/" + name)
            else:
                audit.require(old_quality["quality"] is None, f"run{index}/budget{budget}/unpaid_no_credit")
            deltas = {}
            for name in METRICS:
                deltas[name] = sign(result[name] - baseline[name], abs(result[name]) + abs(baseline[name]))
                audit.sign(deltas[name], saved["half_relative_squared_error_change_from_own_initial"][name],
                           f"run{index}/budget{budget}/delta/" + name)
            for name in RELATIVE:
                audit.near(result[name] - baseline[name], saved["relative_L2_change_from_own_initial"][name],
                           f"run{index}/budget{budget}/relative_delta/" + name)
            result_row = {"run_index": index, "case_index": case_index, "budget": budget,
                          "mode": row["measurement_mode"], "method": row["method"], "eligible": paid and final,
                          "half_squared_error_deltas": deltas}
            endpoint_reports.append(result_row)
            groups[(row["measurement_mode"], row["method"], budget)].append((saved, result_row))
    supplied_groups = {(g["mode"], g["method"], g["internal_budget_seconds"]): g for g in table["all_groups"]}
    audit.require(len(groups) == len(supplied_groups) == 60 and len(endpoint_reports) == 1440, "all1440_and60groups")
    group_reports = []
    for key, rows in sorted(groups.items()):
        saved_group = supplied_groups[key]
        audit.require(saved_group["planned_endpoint_instances"] == len(rows) == 24
                      and saved_group["eligible_internal_diagnostic_instances"] == sum(r[1]["eligible"] for r in rows),
                      str(key) + "/denominators")
        audit.require(saved_group["unpaid_or_ineligible_instances_retained"] == sum(not r[1]["eligible"] for r in rows),
                      str(key) + "/unpaid_or_ineligible_denominator")
        report = {"mode": key[0], "method": key[1], "budget": key[2], "metrics": {}}
        for name in METRICS:
            old = [r[0]["half_relative_squared_error_change_from_own_initial"][name] for r in rows]
            own = [r[1]["half_squared_error_deltas"][name] for r in rows]
            eligible = [r[1]["half_squared_error_deltas"][name] for r in rows if r[1]["eligible"]]
            target = saved_group["metrics"][name]
            eligible_old = [r[0]["half_relative_squared_error_change_from_own_initial"][name]
                            for r in rows if r[1]["eligible"]]
            audit.require(target["eligible_denominator"] == len(eligible)
                          and target["full_planned_denominator"] == len(rows),
                          str(key) + "/metric_denominators/" + name)
            audit.require(target["all_raw_literal_sign_counts"] == dict(collections.Counter(r["literal_sign"] for r in old)),
                          str(key) + "/literal_group_self_consistency/" + name)
            audit.require(target["all_raw_numerical_sign_counts"] == dict(collections.Counter(r["numerical_sign"] for r in own)),
                          str(key) + "/independent_numerical_counts/" + name)
            audit.require(target["eligible_internal_literal_sign_counts"] ==
                          dict(collections.Counter(r["literal_sign"] for r in eligible_old)),
                          str(key) + "/eligible_literal_group_self_consistency/" + name)
            audit.fields(statistics_of([r["value"] for r in own]), target["all_raw_saved_states_delta_statistics"],
                         str(key) + "/all_stats/" + name)
            audit.fields(statistics_of([r["value"] for r in eligible]), target["eligible_internal_delta_statistics"],
                         str(key) + "/eligible_stats/" + name)
            report["metrics"][name] = {"all": statistics_of([r["value"] for r in own]),
                "eligible": statistics_of([r["value"] for r in eligible]),
                "literal_sign_counts": dict(collections.Counter(r["literal_sign"] for r in own)),
                "numerical_sign_counts": dict(collections.Counter(r["numerical_sign"] for r in own))}
        group_reports.append(report)
    summary = load(owned(producer, "summary.json"))
    audit.require(summary["issues"] == ps["issues"] == [] and summary["checks"] == ps["checks"],
                  "summary/status_check_result_agreement")
    audit.require(summary["new_PDE_solves"] == summary["new_optimizer_trajectories"] == 0
                  and summary["truth_used_for_policy"] is False, "summary/software_only_scope")
    audit.require(summary["all8_cases_retained"] is True and summary["original_internal_endpoint_instances"] == 1440,
                  "summary/denominators")
    audit.require(summary["truth_models_literal_box_feasible_count"] == sum(feasibility(c["independent_truth_model"])["literal_box_feasible"] for c in cases.values()),
                  "summary/truth_feasibility")
    audit.require(summary["initial_models_literal_box_feasible_count"] == sum(feasibility(c["initial_model"])["literal_box_feasible"] for c in cases.values()),
                  "summary/initial_feasibility")
    for field, sign_key in (("toward_truth_data_slope_literal_sign_counts", "literal_sign"),
                            ("toward_truth_data_slope_numerical_sign_counts", "numerical_sign")):
        audit.require(summary[field] == dict(collections.Counter(c["data_gradient_dot_toward_truth_correction"][sign_key] for c in recorded_cases)),
                      "summary/" + field)
    for direction_name in ("actual_raw_proposal", "exact_normalized_negative_data_gradient"):
        available = [c["directions"][direction_name] for c in recorded_cases
                     if c["directions"][direction_name]["available"]]
        target = summary["direction_summaries"][direction_name]
        audit.require(target["full_case_denominator"] == 8 and target["available_directions"] == len(available)
                      and target["unavailable_cases_retained"] == 8 - len(available),
                      "summary/" + direction_name + "/denominators")
        for name in METRICS:
            calculated = {
                "literal_local_opposition_cases": sum(c["literal_local_opposition"][name] for c in available),
                "numerically_classified_local_opposition_cases":
                    sum(c["numerically_classified_local_opposition"][name] for c in available),
                "metric_derivative_literal_sign_counts": dict(collections.Counter(
                    c["half_relative_squared_metric_derivatives"][name]["sign"]["literal_sign"] for c in available)),
                "metric_derivative_numerical_sign_counts": dict(collections.Counter(
                    c["half_relative_squared_metric_derivatives"][name]["sign"]["numerical_sign"] for c in available))}
            audit.require(target["metrics"][name] == calculated,
                          "summary/" + direction_name + "/all_case_metric_counts/" + name)
    source_after = source_pin(repo)
    audit.require(source_after == source_before, "source_after_unchanged")
    audit.phase = "finished"
    return {"input_pins": {"producer": pp, "witness": wp, "benchmark": bp}, "source_before": source_before,
            "source_after": source_after, "cases_checked": len(case_reports), "endpoint_instances_checked": len(endpoint_reports),
            "mode_method_budget_groups_checked": len(group_reports), "initial_case_reports": case_reports,
            "endpoint_reports": endpoint_reports, "group_reports": group_reports,
            "near_zero_literal_disagreements": audit.near_zero_literal_disagreements,
            "scope": "independent scalar local/unprojected checks; no finite clipped-step or causal inference"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "witness", "benchmark", "producer", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--execute-frozen-audit", action="store_true")
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args(argv)
    if not args.execute_frozen_audit:
        print(json.dumps({"status": "source_only_no_project_reads_or_numeric_imports",
                          "source_sha256": checksum(Path(__file__))}))
        return 0
    if not all((args.repo, args.witness, args.benchmark, args.producer, args.output, args.expected_source_sha256)):
        parser.error("explicit audit requires all paths and reviewed source SHA")
    if checksum(Path(__file__)) != args.expected_source_sha256:
        raise ValueError("reviewed independent auditor SHA mismatch")
    roots = [p.resolve() for p in (args.repo, args.witness, args.benchmark, args.producer)]
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(root) for root in roots):
        raise FileExistsError("fresh outer audit JSON required")
    audit = Audit()
    report = {"role": "independent AI scalar post-audit, not human review/student conclusions",
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "auditor_source_sha256": checksum(Path(__file__)), "new_PDE_solves": 0, "new_optimizer_trajectories": 0}
    try:
        report.update(perform(*roots, audit))
    except BaseException as exc:
        audit.require(False, "fatal_audit_failure", f"{type(exc).__name__}: {exc}")
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
    report.update(result="clear_independent_scalar_audit" if not audit.issues else "issues_require_review",
                  checks=audit.checks, numeric_comparisons=audit.numeric, issues=audit.issues,
                  final_phase=audit.phase, criteria_fixed_before_evaluation=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write((json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    print(json.dumps({"result": report["result"], "checks": audit.checks, "issues": len(audit.issues),
                      "output": str(output), "sha256": checksum(output)}))
    return 0 if not audit.issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
