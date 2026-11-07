"""Independently inspect a completed trajectory study, including raw-table completeness.

A green workflow or a saved summary alone does not establish that every planned
model/policy/iteration was executed. This checks the raw Cartesian design, joins,
cost/time totals, objective trajectories, quantile and declared frozen source.
It does not establish geological realism, novelty or statistical exchangeability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np


def _read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, label):
    result = float(value)
    _require(math.isfinite(result), f"nonfinite {label}")
    return result


def _integer(value, label):
    result = _number(value, label)
    _require(result >= 0 and result.is_integer(), f"invalid nonnegative integer {label}")
    return int(result)


def _boolean(value):
    _require(value in {"True", "False"}, f"invalid boolean {value!r}")
    return value == "True"


def _same(actual, expected, label):
    _require(math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12),
             f"{label}: {actual} != {expected}")


def _key(row):
    return (int(row["seed"]), int(row["case_index"]), row["scenario"], row["policy"])


def _check_interval(saved, values, label):
    values = np.asarray(values, dtype=float)
    _require(saved["n_independent_models"] == len(values), f"{label}: incorrect sample count")
    _same(saved["mean"], values.mean(), label)
    if len(values) == 1:
        _require(saved["ci95"] is None, f"{label}: unsupported one-model interval")
    else:
        rng = np.random.default_rng(71)
        resampled = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
        for observed, expected in zip(saved["ci95"], np.quantile(resampled, [0.025, 0.975]), strict=True):
            _same(observed, expected, f"{label}: model bootstrap endpoint")


def _is_source_path(name):
    path = Path(name)
    return (name == "pyproject.toml" or
            (name.startswith("src/") and path.suffix == ".py") or
            name.startswith("configs/") or
            (name.startswith("scripts/") and path.suffix in {".py", ".sh"}) or
            (path.parent.as_posix() == ".github/workflows" and path.suffix == ".yml") or
            (path.parent.as_posix() == "." and path.match("requirements*.txt")))


def audit_study(directory, *, repository=None, expected_models=None, expected_iterations=None,
                expected_policies=None, expected_scenarios=None):
    directory = Path(directory)
    configuration = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    _require(status["status"] == "complete", "study is not marked complete")
    _require(summary["source_unchanged_during_run"] is True, "source changed during run")
    _require(configuration == summary["configuration"], "summary/configuration mismatch")
    models = len(configuration["seeds"]) * configuration["cases_per_seed"]
    iterations = configuration["iterations"]
    _require(expected_models is None or models == expected_models, "unexpected model count")
    _require(expected_iterations is None or iterations == expected_iterations,
             "unexpected iteration count")
    _require(expected_policies is None or len(configuration["policies"]) == expected_policies,
             "unexpected policy count")
    _require(expected_scenarios is None or len(configuration["scenarios"]) == expected_scenarios,
             "unexpected scenario count")
    expected = set(itertools.product(configuration["seeds"],
                                     range(configuration["cases_per_seed"]),
                                     configuration["scenarios"], configuration["policies"]))
    cases = _read_csv(directory / "cases.csv")
    trajectories = _read_csv(directory / "trajectories.csv")
    _require(len(cases) == len(expected) and {_key(r) for r in cases} == expected,
             "case table is incomplete, duplicated, or outside the planned design")
    _require(len(trajectories) == len(expected) * (iterations + 1),
             "unexpected trajectory row count")
    by_case = defaultdict(list)
    for row in trajectories:
        _require(_key(row) in expected, "unplanned trajectory case")
        by_case[_key(row)].append(row)
    initial_values = {}
    targets = defaultdict(lambda: defaultdict(list))
    total_increases = 0
    for case in cases:
        key = _key(case)
        records = sorted(by_case[key], key=lambda r: int(r["iteration"]))
        _require([int(r["iteration"]) for r in records] == list(range(iterations + 1)),
                 f"incomplete or duplicated iterations: {key}")
        for column, value in case.items():
            if column not in {"scenario", "policy"}:
                _number(value, column)
        for record in records:
            for column in ("true_objective", "model_relative_l2", "algorithm_seconds",
                           "validation_seconds", "step_size"):
                _require(_number(record[column], column) >= 0, f"negative {column}")
            _boolean(record["accepted"])
            _require(record["step_acceptance"] == configuration["step_acceptance"],
                     "per-record acceptance protocol mismatch")
            _require(record.get("effective_step_acceptance", record["step_acceptance"]) in
                     {"surrogate", "exact-decrease", "exact-armijo"},
                     "invalid effective acceptance criterion")
            if record["method_objective"]:
                _number(record["method_objective"], "method_objective")
        objectives = [float(r["true_objective"]) for r in records]
        _same(case["initial_objective"], objectives[0], "initial objective")
        _same(case["final_objective"], objectives[-1], "final objective")
        _same(case["initial_model_error"], records[0]["model_relative_l2"], "initial model error")
        _same(case["final_model_error"], records[-1]["model_relative_l2"], "final model error")
        _same(case["objective_ratio"], objectives[-1] / max(objectives[0], 1e-15), "objective ratio")
        _same(case["algorithm_seconds"], sum(float(r["algorithm_seconds"]) for r in records),
              "algorithm time total")
        _same(case["validation_seconds"], sum(float(r["validation_seconds"]) for r in records),
              "validation time total")
        for column in ("exact_gradient_blocks", "verification_gradient_blocks", "line_search_trials"):
            _same(case[column], sum(_integer(r[column], column) for r in records), f"total {column}")
        _same(case["accepted_steps"], sum(_boolean(r["accepted"]) for r in records), "accepted steps")
        cumulative = 0
        for record in records:
            cumulative += int(record["exact_gradient_blocks"])
            _same(record["cumulative_exact_gradient_blocks"], cumulative, "cumulative gradient blocks")
        increases = sum(b - a > 1e-10 * max(objectives[0], 1e-15)
                        for a, b in zip(objectives, objectives[1:], strict=False))
        _same(case["objective_increases"], increases, "objective increases")
        total_increases += increases
        if configuration["step_acceptance"] in {"exact-decrease", "exact-armijo"}:
            _require(increases == 0, f"true-objective increase under paid acceptance: {key}")
        for index, record in enumerate(records[:-1]):
            if not _boolean(record["accepted"]):
                _same(objectives[index + 1], objectives[index], "rejected update changed objective")
                _same(records[index + 1]["model_relative_l2"], record["model_relative_l2"],
                      "rejected update changed model error")
                _same(record["step_size"], 0, "rejected update has a nonzero step")
            else:
                _require(float(record["step_size"]) > 0, "accepted update has zero step")
                effective = record.get("effective_step_acceptance", record["step_acceptance"])
                if effective in {"exact-decrease", "exact-armijo"}:
                    _require(objectives[index + 1] < objectives[index],
                             "accepted paid update did not strictly decrease true objective")
        final = records[-1]
        _require(final["mode"] == "final-evaluation" and not _boolean(final["accepted"]),
                 "invalid final-evaluation record")
        for column in ("algorithm_seconds", "exact_gradient_blocks", "verification_gradient_blocks",
                       "line_search_trials", "step_size"):
            _same(final[column], 0, f"final evaluation has nonzero {column}")
        for column in case:
            if column.startswith("online_"):
                _integer(case[column], column)
        # Specific to this FNO/reference-solver study: each counted exact or
        # verification gradient block executes one forward and one adjoint RHS.
        # Receiver-map construction is separately charged to stability_rhs.
        gradient_blocks = int(case["exact_gradient_blocks"]) + int(case["verification_gradient_blocks"])
        _same(case["online_adjoint_rhs"], gradient_blocks, "adjoint RHS/gradient-block count")
        _require(int(case["online_forward_rhs"]) >= gradient_blocks,
                 "forward RHS count omits gradient work")
        model_scenario = key[:3]
        if model_scenario in initial_values:
            _same(objectives[0], initial_values[model_scenario], "paired initial objective")
        initial_values[model_scenario] = objectives[0]
        elapsed = 0.0
        for target in (0.1, 0.05):
            elapsed = 0.0
            reached = None
            for record, objective in zip(records, objectives, strict=True):
                if objective <= target * objectives[0]:
                    reached = elapsed
                    break
                elapsed += float(record["algorithm_seconds"])
            targets[(key[2], key[3])][target].append(reached)

    for scenario, policy in itertools.product(configuration["scenarios"], configuration["policies"]):
        selected = [r for r in cases if r["scenario"] == scenario and r["policy"] == policy]
        aggregate = summary["comparisons"][scenario][policy]
        _require(aggregate["n_models"] == models, "summary model count mismatch")
        for column in ("objective_ratio", "final_model_error", "algorithm_seconds"):
            _check_interval(aggregate[column], [float(r[column]) for r in selected],
                            f"summary {scenario}/{policy}/{column}")
        _same(aggregate["objective_increases"], sum(int(r["objective_increases"]) for r in selected),
              "summary objective increase count")
        _same(aggregate["mean_exact_gradient_blocks"], mean(int(r["exact_gradient_blocks"]) for r in selected),
              "summary mean exact blocks")
        for baseline in ("exact", "random-repair", "residual-repair"):
            reference = {_key(r)[:2]: r for r in cases
                         if r["scenario"] == scenario and r["policy"] == baseline}
            if not reference:
                continue
            field = ("paired_final_model_error_difference_to_exact" if baseline == "exact" else
                     f"paired_model_error_difference_to_{baseline}")
            _check_interval(aggregate[field], [float(r["final_model_error"]) -
                                              float(reference[_key(r)[:2]]["final_model_error"])
                                              for r in selected], f"paired error vs {baseline}")
            _check_interval(aggregate[f"paired_runtime_ratio_to_{baseline}"],
                            [float(r["algorithm_seconds"]) /
                             float(reference[_key(r)[:2]]["algorithm_seconds"])
                             for r in selected], f"paired runtime vs {baseline}")
        for column in selected[0]:
            if column.startswith("online_"):
                _same(aggregate[f"mean_{column}"], mean(float(r[column]) for r in selected),
                      f"summary cost {column}")

    calibration = _read_csv(directory / "calibration.csv")
    _require(sorted(int(r["case_index"]) for r in calibration) ==
             list(range(configuration["calibration_cases"])), "incomplete calibration table")
    scores = sorted(_number(r["pathwise_score"], "calibration score") for r in calibration)
    _require(min(scores) >= 0, "negative calibration score")
    rank = math.ceil((len(scores) + 1) * (1 - configuration["alpha"]))
    _require(rank <= len(scores), "unsupported finite quantile")
    _same(summary["pathwise_scale"], scores[rank - 1] or 1.0, "conformal order statistic")
    if "clean" in configuration["scenarios"]:
        coverage = _read_csv(directory / "held_out_initial_path_coverage.csv")
        _require(len(coverage) == models and
                 {(int(r["seed"]), int(r["case_index"])) for r in coverage} ==
                 set(itertools.product(configuration["seeds"], range(configuration["cases_per_seed"]))),
                 "incomplete held-out path table")
        for row in coverage:
            _require(_boolean(row["covered"]) ==
                     (float(row["pathwise_score"]) <= summary["pathwise_scale"]),
                     "incorrect held-out coverage flag")
        _same(summary["clean_initial_pathwise_coverage"]["observed_coverage"],
              mean(_boolean(r["covered"]) for r in coverage), "path coverage")
        from scipy.stats import beta

        uncertainty = summary["clean_initial_pathwise_coverage"]["failure_uncertainty"]
        failures = sum(not _boolean(r["covered"]) for r in coverage)
        _require(uncertainty["n_units"] == models and uncertainty["failures"] == failures,
                 "coverage uncertainty counts mismatch")
        _same(uncertainty["observed_failure_rate"], failures / models, "path failure rate")
        upper = 1.0 if failures == models else beta.ppf(
            uncertainty["confidence"], failures + 1, models - failures)
        _same(uncertainty["one_sided_failure_probability_upper"], upper,
              "path failure probability upper confidence limit")

    provenance = configuration["provenance"]
    manifest = provenance["source_files"]
    if repository is not None:
        tree_paths = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", provenance["git_revision"]],
            cwd=repository, text=True).splitlines()
        _require(set(manifest) == {name for name in tree_paths if _is_source_path(name)},
                 "source manifest is incomplete for the declared source schema/commit")
        checkpoint = subprocess.check_output(
            ["git", "show", f"{provenance['git_revision']}:{configuration['checkpoint']}"],
            cwd=repository)
        _require(hashlib.sha256(checkpoint).hexdigest() == configuration["checkpoint_sha256"],
                 "checkpoint hash differs from the declared commit")
    digest = hashlib.sha256()
    for name, checksum in sorted(manifest.items()):
        digest.update(name.encode() + b"\0" + checksum.encode() + b"\0")
        if repository is not None:
            content = subprocess.check_output(
                ["git", "show", f"{provenance['git_revision']}:{name}"], cwd=repository)
            _require(hashlib.sha256(content).hexdigest() == checksum,
                     f"recorded source differs from declared commit: {name}")
    _same_source = digest.hexdigest()
    _require(_same_source == provenance["source_sha256"] == summary["source_sha256_at_end"],
             "source manifest/hash mismatch")
    time_targets = {}
    for (scenario, policy), thresholds in sorted(targets.items()):
        time_targets.setdefault(scenario, {})[policy] = {
            str(target): {
                "reached_models": sum(value is not None for value in values),
                "total_models": len(values),
                "median_seconds_among_reached_only": median(
                    value for value in values if value is not None
                ) if any(value is not None for value in values) else None,
            } for target, values in thresholds.items()
        }
    return {
        "audit_passed": True, "models": models, "case_rows": len(cases),
        "trajectory_rows": len(trajectories), "iterations": iterations,
        "calibration_cases": len(calibration), "objective_increases": total_increases,
        "declared_source_commit": provenance["git_revision"],
        "source_manifest_checked_against_git_commit": repository is not None,
        "checkpoint_checked_against_git_commit": repository is not None,
        "source_sha256": _same_source,
        "time_to_objective_ratio": time_targets,
        "time_to_target_scope": "online seconds accumulated before each recorded state; unreached cases remain censored, and the reached-only median is not an overall speedup estimate",
        "limitations": [
            "This audit checks internal completion/accounting and declared source bytes; it does not prove scientific validity, exchangeability or field-scale performance.",
            "The frozen study saves RHS totals per case, not per iteration. This checks their summary means and nonnegative-integral form, not an independent reconstruction of every executed solve.",
            "Recomputing the declared bootstrap checks reported arithmetic; it does not validate the assumed independent sampling units.",
        ],
        "artifact_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("config.json", "summary.json", "status.json", "cases.csv",
                         "trajectories.csv", "calibration.csv", "held_out_initial_path_coverage.csv")
            if (directory / name).is_file()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--expected-models", type=int)
    parser.add_argument("--expected-iterations", type=int)
    parser.add_argument("--expected-policies", type=int)
    parser.add_argument("--expected-scenarios", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_study(args.directory, repository=args.repository,
                         expected_models=args.expected_models, expected_iterations=args.expected_iterations,
                         expected_policies=args.expected_policies, expected_scenarios=args.expected_scenarios)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"time_to_objective_ratio", "artifact_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
