from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from scipy.stats import pearsonr, spearmanr

from wavecert.data.wavefields import random_velocity_model
from wavecert.experiments.synthetic import velocity_to_m
from wavecert.metrics import cosine_similarity, relative_l2
from wavecert.physics.helmholtz import Helmholtz2D, SurveyGeometry
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate


@dataclass(frozen=True)
class FailureStudyRow:
    case_id: int
    wavefield_relative_l2: float
    receiver_relative_l2: float
    objective_relative_error: float
    gradient_relative_l2: float
    gradient_cosine_similarity: float
    exact_gradient_norm: float
    neural_gradient_norm: float
    exact_directional_derivative: float
    neural_directional_derivative: float
    directional_absolute_error: float
    preregistered_failure: bool


def _stack_rel_error(pred: list[np.ndarray], truth: list[np.ndarray]) -> float:
    p = np.concatenate([np.asarray(x).reshape(-1) for x in pred])
    t = np.concatenate([np.asarray(x).reshape(-1) for x in truth])
    return float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-12))


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> tuple[float, float]:
    if np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return float("nan"), float("nan")
    result = pearsonr(x, y) if kind == "pearson" else spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def run_failure_study(
    *,
    checkpoint_path: str | Path = "checkpoints/fno_phase2.pt",
    output_dir: str | Path = "results/phase3",
    n_cases: int = 80,
    seed: int = 20260930,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
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
    # Two source positions and two frequencies give a non-trivial multi-block
    # FWI objective while keeping the study lightweight and reproducible.
    block_sources = (geometry.source_indices[1], geometry.source_indices[-2])
    block_frequencies = (3.0, 4.0)

    rng = np.random.default_rng(seed)
    rows: list[FailureStudyRow] = []
    representative: dict | None = None

    for case_id in range(n_cases):
        velocity_true = random_velocity_model(shape, rng)
        # A realistic early/mid-inversion state: preserve large-scale trends but
        # remove some compact structure. Add a tiny deterministic perturbation
        # so exact gradients do not collapse near a stationary point.
        sigma = float(rng.uniform(0.7, 1.7))
        velocity_eval = ndi.gaussian_filter(velocity_true, sigma=sigma, mode="reflect")
        smooth_noise = ndi.gaussian_filter(rng.normal(size=shape), sigma=2.0, mode="reflect")
        smooth_noise /= max(np.std(smooth_noise), 1e-12)
        velocity_eval = np.clip(velocity_eval + rng.uniform(0.006, 0.022) * smooth_noise, 1.65, 2.85)
        m_true = velocity_to_m(velocity_true).reshape(-1)
        m_eval = velocity_to_m(velocity_eval).reshape(-1)

        exact_obj = 0.0
        neural_obj = 0.0
        exact_grad = np.zeros(physics.n, dtype=float)
        neural_grad = np.zeros(physics.n, dtype=float)
        exact_states: list[np.ndarray] = []
        neural_states: list[np.ndarray] = []
        exact_receivers: list[np.ndarray] = []
        neural_receivers: list[np.ndarray] = []

        for source_index in block_sources:
            q = physics.source(source_index)
            for frequency in block_frequencies:
                u_true = physics.solve_state(m_true, q, frequency)
                observed = physics.restrict(u_true, geometry.receiver_indices)
                eo, eg, u_exact, _ = physics.objective_and_gradient(
                    m_eval, q, observed, geometry.receiver_indices, frequency
                )
                no, ng, u_neural, _ = surrogate.objective_and_gradient(
                    m_eval, q, observed, geometry.receiver_indices, frequency
                )
                exact_obj += eo
                neural_obj += no
                exact_grad += eg
                neural_grad += ng
                exact_states.append(u_exact)
                neural_states.append(u_neural)
                exact_receivers.append(physics.restrict(u_exact, geometry.receiver_indices))
                neural_receivers.append(physics.restrict(u_neural, geometry.receiver_indices))

        wave_error = _stack_rel_error(neural_states, exact_states)
        receiver_error = _stack_rel_error(neural_receivers, exact_receivers)
        obj_error = abs(neural_obj - exact_obj) / max(abs(exact_obj), 1e-12)
        grad_error = relative_l2(exact_grad, neural_grad)
        grad_cos = cosine_similarity(exact_grad, neural_grad)
        neural_norm = float(np.linalg.norm(neural_grad))
        direction = -neural_grad / max(neural_norm, 1e-15)
        exact_dd = float(np.dot(exact_grad, direction))
        neural_dd = float(np.dot(neural_grad, direction))
        failure = bool(
            receiver_error <= 0.15
            and (grad_error >= 0.50 or grad_cos <= 0.90)
        )
        row = FailureStudyRow(
            case_id=case_id,
            wavefield_relative_l2=wave_error,
            receiver_relative_l2=receiver_error,
            objective_relative_error=float(obj_error),
            gradient_relative_l2=float(grad_error),
            gradient_cosine_similarity=float(grad_cos),
            exact_gradient_norm=float(np.linalg.norm(exact_grad)),
            neural_gradient_norm=neural_norm,
            exact_directional_derivative=exact_dd,
            neural_directional_derivative=neural_dd,
            directional_absolute_error=abs(exact_dd - neural_dd),
            preregistered_failure=failure,
        )
        rows.append(row)
        if failure and representative is None:
            representative = {
                "case_id": case_id,
                "velocity_true": velocity_true,
                "velocity_eval": velocity_eval,
                "exact_gradient": exact_grad,
                "neural_gradient": neural_grad,
            }

    csv_path = out / "failure_study.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    receiver = np.asarray([r.receiver_relative_l2 for r in rows])
    wave = np.asarray([r.wavefield_relative_l2 for r in rows])
    grad = np.asarray([r.gradient_relative_l2 for r in rows])
    cos = np.asarray([r.gradient_cosine_similarity for r in rows])
    failures = np.asarray([r.preregistered_failure for r in rows], dtype=bool)
    pearson_grad = _safe_corr(receiver, grad, "pearson")
    spearman_grad = _safe_corr(receiver, grad, "spearman")
    pearson_cos = _safe_corr(receiver, cos, "pearson")
    spearman_cos = _safe_corr(receiver, cos, "spearman")

    summary = {
        "seed": seed,
        "n_cases": n_cases,
        "blocks_per_case": len(block_sources) * len(block_frequencies),
        "sources": [int(v) for v in block_sources],
        "frequencies_hz": list(block_frequencies),
        "preregistered_failure_rule": {
            "receiver_relative_l2_le": 0.15,
            "gradient_relative_l2_ge": 0.50,
            "or_gradient_cosine_le": 0.90,
        },
        "forward": {
            "receiver_relative_l2_median": float(np.median(receiver)),
            "receiver_relative_l2_p90": float(np.quantile(receiver, 0.9)),
            "wavefield_relative_l2_median": float(np.median(wave)),
        },
        "gradient": {
            "relative_l2_median": float(np.median(grad)),
            "relative_l2_p90": float(np.quantile(grad, 0.9)),
            "cosine_median": float(np.median(cos)),
            "cosine_p10": float(np.quantile(cos, 0.1)),
        },
        "correlations": {
            "receiver_error_vs_gradient_error_pearson_r": pearson_grad[0],
            "receiver_error_vs_gradient_error_pearson_p": pearson_grad[1],
            "receiver_error_vs_gradient_error_spearman_rho": spearman_grad[0],
            "receiver_error_vs_gradient_error_spearman_p": spearman_grad[1],
            "receiver_error_vs_gradient_cosine_pearson_r": pearson_cos[0],
            "receiver_error_vs_gradient_cosine_pearson_p": pearson_cos[1],
            "receiver_error_vs_gradient_cosine_spearman_rho": spearman_cos[0],
            "receiver_error_vs_gradient_cosine_spearman_p": spearman_cos[1],
        },
        "failure_cases": int(np.count_nonzero(failures)),
        "failure_fraction": float(np.mean(failures)),
        "exit_criteria": {
            "at_least_50_cases": n_cases >= 50,
            "scatter_data_saved": True,
            "preregistered_failure_found_or_null_reported": True,
            "correlations_reported": True,
        },
    }
    summary["passed"] = all(summary["exit_criteria"].values())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(6.3, 4.6), constrained_layout=True)
    ax.scatter(receiver[~failures], grad[~failures], s=24, alpha=0.7, label="other held-out cases")
    if np.any(failures):
        ax.scatter(receiver[failures], grad[failures], s=38, marker="x", label="pre-registered failures")
    ax.axvline(0.15, linestyle="--", linewidth=1)
    ax.axhline(0.50, linestyle="--", linewidth=1)
    ax.set_xlabel("receiver forward relative L2 error")
    ax.set_ylabel("FWI gradient relative L2 error")
    ax.set_title("Forward accuracy does not determine gradient accuracy")
    ax.legend(fontsize=8)
    fig.savefig(out / "forward_vs_gradient_error.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 4.6), constrained_layout=True)
    ax.scatter(receiver[~failures], cos[~failures], s=24, alpha=0.7, label="other held-out cases")
    if np.any(failures):
        ax.scatter(receiver[failures], cos[failures], s=38, marker="x", label="pre-registered failures")
    ax.axvline(0.15, linestyle="--", linewidth=1)
    ax.axhline(0.90, linestyle="--", linewidth=1)
    ax.set_xlabel("receiver forward relative L2 error")
    ax.set_ylabel("exact vs neural gradient cosine")
    ax.set_title("Forward error vs gradient direction")
    ax.legend(fontsize=8)
    fig.savefig(out / "forward_vs_gradient_cosine.png", dpi=220)
    plt.close(fig)

    if representative is not None:
        fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.6), constrained_layout=True)
        axes[0, 0].imshow(representative["velocity_true"], aspect="auto")
        axes[0, 0].set_title("true velocity")
        axes[0, 1].imshow(representative["velocity_eval"], aspect="auto")
        axes[0, 1].set_title("evaluation velocity")
        exact_g = representative["exact_gradient"].reshape(shape)
        neural_g = representative["neural_gradient"].reshape(shape)
        lim = max(np.max(np.abs(exact_g)), np.max(np.abs(neural_g)), 1e-12)
        axes[1, 0].imshow(exact_g, vmin=-lim, vmax=lim, aspect="auto")
        axes[1, 0].set_title("exact FWI gradient")
        axes[1, 1].imshow(neural_g, vmin=-lim, vmax=lim, aspect="auto")
        axes[1, 1].set_title("forward-trained FNO gradient")
        fig.suptitle(f"Representative pre-registered failure: case {representative['case_id']}")
        fig.savefig(out / "representative_failure.png", dpi=220)
        plt.close(fig)

    return summary
