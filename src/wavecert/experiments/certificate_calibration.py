from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows], dtype=float)


def _bool(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([r[key].strip().lower() == "true" for r in rows], dtype=bool)


def _split_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal upper quantile."""

    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("empty calibration scores")
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(scores[k - 1])


def run_certificate_calibration(
    *,
    phase4_case_csv: str | Path = "results/phase4/certificate_cases.csv",
    phase4_block_csv: str | Path = "results/phase4/certificate_blocks.csv",
    output_dir: str | Path = "results/phase5",
    calibration_fraction: float = 0.625,
    alpha: float = 0.10,
) -> dict:
    """Calibrate a practical statistical gate without altering the rigorous bound.

    Phase 4 produces a deterministic upper bound that is safe but conservative.
    Phase 5 retains that bound as the reference theorem and separately performs
    split conformal calibration of its *scale*.  The resulting gate has a
    distribution-free finite-sample marginal coverage interpretation under the
    usual exchangeability assumption; it is not described as a deterministic
    PDE guarantee.

    For selective block repair, each source/frequency label is calibrated
    separately at ``alpha / n_blocks``.  A union bound then targets simultaneous
    block coverage of at least ``1-alpha`` for a fixed four-block case.
    """

    case_rows = _read(Path(phase4_case_csv))
    block_rows = _read(Path(phase4_block_csv))
    if not case_rows or not block_rows:
        raise ValueError("Phase-4 results are required")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    case_ids = sorted({int(r["case_id"]) for r in case_rows})
    n_cal = int(round(len(case_ids) * calibration_fraction))
    n_cal = min(max(n_cal, 1), len(case_ids) - 1)
    cal_ids = set(case_ids[:n_cal])
    eval_ids = set(case_ids[n_cal:])
    cal_cases = [r for r in case_rows if int(r["case_id"]) in cal_ids]
    eval_cases = [r for r in case_rows if int(r["case_id"]) in eval_ids]

    cal_realized = _float(cal_cases, "realized_directional_error")
    cal_base = _float(cal_cases, "directional_error_bound")
    case_ratios = cal_realized / np.maximum(cal_base, 1e-15)
    q_case = _split_quantile(case_ratios, alpha)

    eval_realized = _float(eval_cases, "realized_directional_error")
    eval_base = _float(eval_cases, "directional_error_bound")
    eval_dd = _float(eval_cases, "surrogate_directional_derivative")
    eval_exact_dd = _float(eval_cases, "exact_directional_derivative")
    conformal_bound = q_case * eval_base
    covered = eval_realized <= conformal_bound + 1e-12
    certified = eval_dd + conformal_bound < 0.0
    false_cert = certified & (eval_exact_dd >= 0.0)

    # A more conservative 95% case-level operating point is reported too.
    q_case_95 = _split_quantile(case_ratios, 0.05)
    conformal_bound_95 = q_case_95 * eval_base
    covered_95 = eval_realized <= conformal_bound_95 + 1e-12
    certified_95 = eval_dd + conformal_bound_95 < 0.0
    false_cert_95 = certified_95 & (eval_exact_dd >= 0.0)

    labels = sorted({r["label"] for r in block_rows})
    alpha_block = alpha / len(labels)
    block_scales: dict[str, float] = {}
    block_eval_coverage: dict[str, float] = {}
    simultaneous_by_case: dict[int, bool] = {i: True for i in eval_ids}
    for label in labels:
        cal = [r for r in block_rows if r["label"] == label and int(r["case_id"]) in cal_ids]
        ev = [r for r in block_rows if r["label"] == label and int(r["case_id"]) in eval_ids]
        ratios = _float(cal, "realized_directional_error") / np.maximum(
            _float(cal, "directional_error_bound"), 1e-15
        )
        q = _split_quantile(ratios, alpha_block)
        block_scales[label] = q
        ev_err = _float(ev, "realized_directional_error")
        ev_bound = q * _float(ev, "directional_error_bound")
        ok = ev_err <= ev_bound + 1e-12
        block_eval_coverage[label] = float(np.mean(ok))
        for row, is_ok in zip(ev, ok, strict=True):
            simultaneous_by_case[int(row["case_id"])] &= bool(is_ok)

    simultaneous_coverage = float(np.mean(list(simultaneous_by_case.values())))

    rigorous_eff = eval_base / np.maximum(eval_realized, 1e-15)
    conformal_eff = conformal_bound / np.maximum(eval_realized, 1e-15)
    summary = {
        "n_cases": len(case_rows),
        "n_calibration_cases": len(cal_cases),
        "n_evaluation_cases": len(eval_cases),
        "calibration_case_ids": sorted(cal_ids),
        "evaluation_case_ids": sorted(eval_ids),
        "exchangeability_assumption": True,
        "deterministic_reference": {
            "evaluation_coverage": float(np.mean(eval_realized <= eval_base + 1e-12)),
            "median_effectivity": float(np.median(rigorous_eff)),
        },
        "conformal_case_gate": {
            "alpha": alpha,
            "target_coverage": 1.0 - alpha,
            "scale": q_case,
            "evaluation_coverage": float(np.mean(covered)),
            "certified_descent_fraction": float(np.mean(certified)),
            "false_certifications": int(np.count_nonzero(false_cert)),
            "median_effectivity": float(np.median(conformal_eff)),
        },
        "conformal_case_gate_95pct": {
            "alpha": 0.05,
            "target_coverage": 0.95,
            "scale": q_case_95,
            "evaluation_coverage": float(np.mean(covered_95)),
            "certified_descent_fraction": float(np.mean(certified_95)),
            "false_certifications": int(np.count_nonzero(false_cert_95)),
        },
        "conformal_block_gate": {
            "family_alpha": alpha,
            "per_block_alpha": alpha_block,
            "scales_by_label": block_scales,
            "evaluation_coverage_by_label": block_eval_coverage,
            "simultaneous_case_coverage": simultaneous_coverage,
        },
        "exit_criteria": {
            "disjoint_calibration_and_evaluation": cal_ids.isdisjoint(eval_ids),
            "case_eval_coverage_at_least_target_minus_sampling_tolerance": float(np.mean(covered)) >= (1.0 - alpha - 0.10),
            "zero_observed_false_certifications": int(np.count_nonzero(false_cert)) == 0,
            "block_scales_saved": len(block_scales) == len(labels),
        },
    }
    summary["passed"] = all(summary["exit_criteria"].values())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "conformal_scales.json").write_text(
        json.dumps(
            {
                "alpha": alpha,
                "case_scale": q_case,
                "case_scale_95pct": q_case_95,
                "block_family_alpha": alpha,
                "block_scales_by_label": block_scales,
                "calibration_case_ids": sorted(cal_ids),
            },
            indent=2,
        )
        + "\n"
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.9), constrained_layout=True)
    x = np.arange(len(eval_cases))
    ax.scatter(x, eval_realized, s=24, label="realized error")
    ax.scatter(x, conformal_bound, s=24, marker="x", label=f"{int((1-alpha)*100)}% conformal bound")
    ax.set_yscale("log")
    ax.set_xlabel("held-out evaluation case")
    ax.set_ylabel("directional-derivative error / bound")
    ax.set_title("Phase 5: split-conformal residual calibration")
    ax.legend(fontsize=8)
    fig.savefig(out / "conformal_bounds.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.1, 4.6), constrained_layout=True)
    ax.boxplot(
        [np.log10(rigorous_eff), np.log10(conformal_eff)],
        tick_labels=["deterministic", "conformal"],
        showfliers=False,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_ylabel("log10(effectivity)")
    ax.set_title("Calibration reduces conservatism")
    fig.savefig(out / "effectivity_comparison.png", dpi=220)
    plt.close(fig)

    return summary
