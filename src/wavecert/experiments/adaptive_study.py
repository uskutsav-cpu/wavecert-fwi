from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

from wavecert.data.wavefields import random_velocity_model
from wavecert.experiments.synthetic import velocity_to_m
from wavecert.metrics import cosine_similarity
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock, SurveyGeometry
from wavecert.repair.adaptive import adaptive_hybrid_direction, global_fallback_direction
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate


@dataclass(frozen=True)
class AdaptiveStudyRow:
    case_id: int
    policy: str
    exact_blocks_used: int
    exact_block_fraction: float
    certified_descent: bool
    true_directional_derivative: float
    true_descent: bool
    cosine_to_exact_descent_direction: float
    mode: str


def _append_policy_row(
    rows: list[AdaptiveStudyRow],
    *,
    case_id: int,
    policy: str,
    direction: np.ndarray,
    exact_used: int,
    certified: bool,
    mode: str,
    exact_grad_total: np.ndarray,
    exact_direction: np.ndarray,
    total_blocks: int,
) -> None:
    """Append one policy outcome without closing over loop-scoped state."""

    true_dd = float(np.dot(exact_grad_total, direction))
    rows.append(
        AdaptiveStudyRow(
            case_id=case_id,
            policy=policy,
            exact_blocks_used=int(exact_used),
            exact_block_fraction=float(exact_used / total_blocks),
            certified_descent=bool(certified),
            true_directional_derivative=true_dd,
            true_descent=bool(true_dd < 0.0),
            cosine_to_exact_descent_direction=float(
                cosine_similarity(exact_direction, direction)
            ),
            mode=mode,
        )
    )


def _make_case(shape: tuple[int, int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    velocity_true = random_velocity_model(shape, rng)
    sigma = float(rng.uniform(0.7, 1.7))
    velocity_eval = ndi.gaussian_filter(velocity_true, sigma=sigma, mode="reflect")
    noise = ndi.gaussian_filter(rng.normal(size=shape), sigma=2.0, mode="reflect")
    noise /= max(np.std(noise), 1e-12)
    velocity_eval = np.clip(velocity_eval + rng.uniform(0.006, 0.022) * noise, 1.65, 2.85)
    return velocity_true, velocity_eval


def run_adaptive_study(
    *,
    checkpoint_path: str | Path = "checkpoints/fno_phase2.pt",
    output_dir: str | Path = "results/phase6",
    conformal_scales_path: str | Path = "results/phase5/conformal_scales.json",
    n_cases: int = 30,
    seed: int = 20261010,
) -> dict:
    """Compare neural-only, global fallback, selective repair, and exact directions.

    Exact full gradients are computed after each policy solely for offline
    validation. Those validation solves are *not* counted in the policy's
    ``exact_blocks_used`` metric.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(Path(conformal_scales_path).read_text())
    case_scale = float(calibration["case_scale"])
    block_scales = {k: float(v) for k, v in calibration["block_scales_by_label"].items()}
    surrogate = TrainedFNOWavefieldSurrogate.load(checkpoint_path)
    shape = surrogate.shape
    physics = Helmholtz2D(
        shape=shape,
        spacing=0.05,
        damping_width=max(3, min(shape) // 8),
        damping_strength=2.0,
    )
    geometry = SurveyGeometry.from_grid(
        shape,
        n_sources=6,
        n_receivers=min(20, shape[1] - 4),
        source_depth=2,
        receiver_depth=2,
        margin=2,
    )
    block_sources = (geometry.source_indices[1], geometry.source_indices[-2])
    block_frequencies = (3.0, 4.0)

    rng = np.random.default_rng(seed)
    rows: list[AdaptiveStudyRow] = []

    for case_id in range(n_cases):
        velocity_true, velocity_eval = _make_case(shape, rng)
        m_true = velocity_to_m(velocity_true).reshape(-1)
        m_eval = velocity_to_m(velocity_eval).reshape(-1)

        blocks: list[SurveyBlock] = []
        exact_grad_total = np.zeros(physics.n, dtype=float)
        neural_grad_total = np.zeros(physics.n, dtype=float)
        for source_index in block_sources:
            q = physics.source(source_index)
            for frequency in block_frequencies:
                u_true = physics.solve_state(m_true, q, frequency)
                observed = physics.restrict(u_true, geometry.receiver_indices)
                label = f"s{source_index}_f{frequency:g}"
                block = SurveyBlock(
                    source_index=int(source_index),
                    frequency_hz=float(frequency),
                    observed=observed,
                    label=label,
                )
                blocks.append(block)
                _, eg, _, _ = physics.objective_and_gradient(
                    m_eval, q, observed, geometry.receiver_indices, frequency
                )
                _, ng, _, _ = surrogate.objective_and_gradient(
                    m_eval, q, observed, geometry.receiver_indices, frequency
                )
                exact_grad_total += eg
                neural_grad_total += ng

        exact_direction = -exact_grad_total / max(np.linalg.norm(exact_grad_total), 1e-14)
        neural_direction = -neural_grad_total / max(np.linalg.norm(neural_grad_total), 1e-14)

        total_blocks = len(blocks)
        _append_policy_row(
            rows,
            case_id=case_id,
            policy="neural-only",
            direction=neural_direction,
            exact_used=0,
            certified=False,
            mode="unverified",
            exact_grad_total=exact_grad_total,
            exact_direction=exact_direction,
            total_blocks=total_blocks,
        )

        global_result = global_fallback_direction(
            physics=physics,
            surrogate=surrogate,
            m=m_eval,
            blocks=blocks,
            receiver_indices=geometry.receiver_indices,
            stability_mode="directional",
            bound_scale=case_scale,
        )
        _append_policy_row(
            rows,
            case_id=case_id,
            policy="conformal-global",
            direction=global_result.direction,
            exact_used=global_result.exact_block_evaluations,
            certified=global_result.certified_descent,
            mode=global_result.mode,
            exact_grad_total=exact_grad_total,
            exact_direction=exact_direction,
            total_blocks=total_blocks,
        )

        selective_result = adaptive_hybrid_direction(
            physics=physics,
            surrogate=surrogate,
            m=m_eval,
            blocks=blocks,
            receiver_indices=geometry.receiver_indices,
            stability_mode="directional",
            bound_scales=block_scales,
        )
        _append_policy_row(
            rows,
            case_id=case_id,
            policy="conformal-selective",
            direction=selective_result.direction,
            exact_used=selective_result.exact_block_evaluations,
            certified=selective_result.certified_descent,
            mode=selective_result.mode,
            exact_grad_total=exact_grad_total,
            exact_direction=exact_direction,
            total_blocks=total_blocks,
        )

        _append_policy_row(
            rows,
            case_id=case_id,
            policy="exact",
            direction=exact_direction,
            exact_used=total_blocks,
            certified=True,
            mode="exact",
            exact_grad_total=exact_grad_total,
            exact_direction=exact_direction,
            total_blocks=total_blocks,
        )

    path = out / "adaptive_study.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for row in rows:
            w.writerow(asdict(row))

    summary: dict[str, object] = {
        "seed": seed,
        "n_cases": n_cases,
        "blocks_per_case": len(block_sources) * len(block_frequencies),
        "statistical_certificate": {
            "case_scale": case_scale,
            "block_scales_by_label": block_scales,
            "coverage_semantics": "split-conformal marginal coverage under exchangeability; not deterministic PDE certification",
        },
        "policies": {},
    }
    for policy in ("neural-only", "conformal-global", "conformal-selective", "exact"):
        r = [x for x in rows if x.policy == policy]
        exact_used = np.asarray([x.exact_blocks_used for x in r], dtype=float)
        descent = np.asarray([x.true_descent for x in r], dtype=bool)
        cosine = np.asarray([x.cosine_to_exact_descent_direction for x in r], dtype=float)
        certified = np.asarray([x.certified_descent for x in r], dtype=bool)
        mode_counts = {m: sum(x.mode == m for x in r) for m in sorted({x.mode for x in r})}
        summary["policies"][policy] = {
            "true_descent_rate": float(np.mean(descent)),
            "certified_rate": float(np.mean(certified)),
            "mean_exact_blocks": float(np.mean(exact_used)),
            "median_exact_blocks": float(np.median(exact_used)),
            "mean_exact_block_fraction": float(np.mean(exact_used) / len(blocks)),
            "mean_cosine_to_exact_direction": float(np.mean(cosine)),
            "mode_counts": mode_counts,
        }

    selective = summary["policies"]["conformal-selective"]
    global_ = summary["policies"]["conformal-global"]
    summary["savings"] = {
        "selective_exact_block_reduction_vs_exact": float(
            1.0 - selective["mean_exact_blocks"] / len(blocks)
        ),
        "selective_exact_block_reduction_vs_global_fallback": float(
            0.0
            if global_["mean_exact_blocks"] <= 0
            else 1.0 - selective["mean_exact_blocks"] / global_["mean_exact_blocks"]
        ),
    }
    summary["exit_criteria"] = {
        "conformal_selective_empirical_safety_at_least_90pct": selective["true_descent_rate"] >= 0.90,
        "conformal_global_empirical_safety_at_least_90pct": global_["true_descent_rate"] >= 0.90,
        "exact_baseline_safe": summary["policies"]["exact"]["true_descent_rate"] >= 1.0 - 1e-12,
        "policy_costs_reported": True,
    }
    summary["passed"] = all(summary["exit_criteria"].values())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    policies = ["neural-only", "conformal-global", "conformal-selective", "exact"]
    xs = [summary["policies"][p]["mean_exact_blocks"] for p in policies]
    ys = [summary["policies"][p]["true_descent_rate"] for p in policies]
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    ax.scatter(xs, ys, s=75)
    for x, y, label in zip(xs, ys, policies, strict=True):
        ax.annotate(label, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("mean exact source-frequency blocks used")
    ax.set_ylabel("fraction of directions that are true descent")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Phase 6: safety versus exact-physics cost")
    fig.savefig(out / "safety_vs_exact_blocks.png", dpi=220)
    plt.close(fig)

    return summary
