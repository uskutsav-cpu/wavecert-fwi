"""Post-hoc comparison of established marginal and IID tolerance calibration.

Reads saved initial-state path scores only. The conventional alpha=.1,
delta=.05 comparison is fixed here, not optimized using held-out coverage.
It does not rerun inversion, change a deployed gate, or overwrite the study.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import binom

from wavecert.provenance import environment_record, sha256_file, source_digest
from wavecert.statistics.conformal import split_conformal_quantile
from wavecert.statistics.tolerance import pac_tolerance_threshold

TARGET_ALPHA = 0.10
CALIBRATION_DELTA = 0.05
INPUT_NAMES = (
    "config.json", "summary.json", "status.json", "calibration.csv",
    "held_out_initial_path_coverage.csv",
)


def _csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _score_values(rows):
    values = np.asarray([float(row["pathwise_score"]) for row in rows])
    if not values.size or np.any(np.isnan(values)) or np.any(values < 0):
        raise ValueError("path scores must be nonempty and nonnegative, without NaN")
    return values


def _observed_coverage(scores, threshold):
    covered = int(np.count_nonzero(scores <= threshold))
    return {
        "n_held_out_paths": int(len(scores)), "covered": covered,
        "failures": int(len(scores) - covered), "observed_coverage": covered / len(scores),
        "scope": "Descriptive post-hoc reuse of already inspected held-out initial-state scores; no new confirmation or trajectory experiment.",
    }


def run_diagnostic(study_dir, output_dir, *, repository_root=None):
    study, output = Path(study_dir).resolve(), Path(output_dir).resolve()
    root = Path(__file__).resolve().parents[1] if repository_root is None else Path(repository_root)
    if output == study or output.is_relative_to(study):
        raise ValueError("diagnostic output must be outside the original study directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("diagnostic output must be new or empty; preserve previous runs")
    output.mkdir(parents=True, exist_ok=True)
    provenance = environment_record(root)
    status_path = output / "status.json"

    def save_status(status, **extra):
        status_path.write_text(json.dumps({"status": status, **extra}, indent=2, allow_nan=False) + "\n")

    save_status("running")
    try:
        input_hashes = {name: sha256_file(study / name) for name in INPUT_NAMES}
        config = json.loads((study / "config.json").read_text())
        saved = json.loads((study / "summary.json").read_text())
        source_status = json.loads((study / "status.json").read_text())
        if (source_status.get("status") != "complete" or
                saved.get("source_unchanged_during_run") is not True or
                saved["configuration"] != config):
            raise ValueError("source study must be complete with consistent frozen-source metadata")
        if config["alpha"] != TARGET_ALPHA or "clean" not in config["scenarios"]:
            raise ValueError("this fixed diagnostic requires the saved alpha=.1 clean initial-path study")
        calibration_rows = _csv(study / "calibration.csv")
        if sorted(int(row["case_index"]) for row in calibration_rows) != list(range(config["calibration_cases"])):
            raise ValueError("calibration score identities are incomplete or duplicated")
        held_out_rows = _csv(study / "held_out_initial_path_coverage.csv")
        expected = {(seed, case) for seed in config["seeds"] for case in range(config["cases_per_seed"])}
        if (len(held_out_rows) != len(expected) or
                {(int(row["seed"]), int(row["case_index"])) for row in held_out_rows} != expected):
            raise ValueError("held-out score identities are incomplete or duplicated")
        calibration, held_out = _score_values(calibration_rows), _score_values(held_out_rows)
        raw_marginal = split_conformal_quantile(calibration, TARGET_ALPHA)
        # Preserve the original study's explicitly conservative zero-scale rule.
        saved_scale = 1.0 if raw_marginal == 0 else raw_marginal
        if not math.isfinite(saved_scale) or not math.isclose(saved_scale, saved["pathwise_scale"], rel_tol=1e-12):
            raise ValueError("saved marginal scale disagrees with the original calibration order statistic")
        marginal_coverage = _observed_coverage(held_out, saved_scale)
        if (any(row["covered"] not in {"True", "False"} for row in held_out_rows) or
                [row["covered"] == "True" for row in held_out_rows] != (held_out <= saved_scale).tolist() or
                not math.isclose(marginal_coverage["observed_coverage"],
                                 saved["clean_initial_pathwise_coverage"]["observed_coverage"])):
            raise ValueError("saved held-out marginal coverage is inconsistent with its raw scores")
        pac = pac_tolerance_threshold(calibration, alpha=TARGET_ALPHA, delta=CALIBRATION_DELTA)
        marginal_rank = math.ceil((len(calibration) + 1) * (1 - TARGET_ALPHA))
        source_same = source_digest(root) == provenance["source_sha256"]
        inputs_same = all(sha256_file(study / name) == value for name, value in input_hashes.items())
        result = {
            "schema_version": 1,
            "status": "complete" if source_same and inputs_same else "stale",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_status": "Post-hoc established-method diagnostic; no new preregistered or confirmatory experiment.",
            "fixed_analysis_parameters": {"alpha": TARGET_ALPHA, "delta": CALIBRATION_DELTA},
            "marginal": {
                "rank": marginal_rank, "raw_quantile": raw_marginal,
                "original_deployed_scale": saved_scale,
                "zero_quantile_conservative_substitution": raw_marginal == 0,
                "observed_held_out": marginal_coverage,
                "iid_continuous_score_probability_population_coverage_at_least_target": float(
                    binom.cdf(marginal_rank - 1, len(calibration), 1 - TARGET_ALPHA)
                ),
                "rank_probability_scope": "Exact for an IID continuous score law, conservative with ties; not an observed guarantee conditional on this particular calibration set.",
            },
            "pac_tolerance": {**pac.to_dict(), "observed_held_out": _observed_coverage(held_out, pac.threshold)},
            "source_study": str(study),
            "source_study_commit": config["provenance"]["git_revision"],
            "source_study_sha256": config["provenance"]["source_sha256"],
            "input_sha256": input_hashes,
            "source_unchanged_during_run": source_same,
            "input_files_unchanged_during_run": inputs_same,
            "provenance": provenance,
            "limitations": [
                "The original marginal result and inversion/cost artifacts are unchanged. The PAC threshold is not applied to an inversion policy here.",
                "The optional tolerance theorem requires IID calibration and future units, conditional on a frozen training/scoring/selection rule; metadata and this calculation do not prove IID sampling.",
                "Neither method guarantees the observed coverage fraction in one finite test set, object-conditional reliability, OOD reliability, or coverage of later adaptive inversion states.",
                "Parameters are conventional fixed values for this post-hoc comparison, selected after the original results were inspected; improved descriptive coverage is not independent validation.",
                "An unsupported rank or selected infinite score has an infinite mathematical threshold. Strict JSON reports null plus explicit flags; deployment must abstain/fall back rather than clip it.",
            ],
            "references": [
                "https://proceedings.mlr.press/v25/vovk12/vovk12.pdf",
                "https://arxiv.org/pdf/2210.14735",
            ],
        }
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if not source_same or not inputs_same:
            raise RuntimeError("source or input changed during the diagnostic")
        save_status("complete", source_unchanged_during_run=True, input_files_unchanged_during_run=True)
        return result
    except Exception as error:
        save_status("failed", error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("results/evidence_study"))
    parser.add_argument("--output", type=Path, default=Path("results/tolerance_calibration_diagnostic"))
    args = parser.parse_args()
    result = run_diagnostic(args.study, args.output)
    print(json.dumps({"status": result["status"],
                      "marginal": result["marginal"]["observed_held_out"],
                      "pac_tolerance": result["pac_tolerance"]["observed_held_out"]}, indent=2))


if __name__ == "__main__":
    main()
