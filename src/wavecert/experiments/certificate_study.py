from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from scipy.stats import spearmanr

from wavecert.certificates.directional import certify_direction
from wavecert.data.wavefields import random_velocity_model
from wavecert.experiments.synthetic import velocity_to_m
from wavecert.metrics import cosine_similarity, relative_l2
from wavecert.physics.helmholtz import Helmholtz2D, SurveyGeometry
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate


@dataclass(frozen=True)
class CertificateCaseRow:
    case_id: int
    receiver_relative_l2: float
    gradient_relative_l2: float
    gradient_cosine_similarity: float
    preregistered_failure: bool
    exact_directional_derivative: float
    surrogate_directional_derivative: float
    realized_directional_error: float
    directional_error_bound: float
    directional_effectivity: float
    directional_certified_descent: bool
    true_descent: bool
    primal_residual_sum: float
    tangent_residual_sum: float


@dataclass(frozen=True)
class CertificateBlockRow:
    case_id: int
    label: str
    frequency_hz: float
    exact_directional_derivative: float
    surrogate_directional_derivative: float
    realized_directional_error: float
    primal_residual_norm: float
    tangent_residual_norm: float
    receiver_resolvent_norm: float
    directional_tangent_resolvent_norm: float
    directional_error_bound: float


def _make_case(shape: tuple[int, int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate the exact Phase-3 held-out case distribution."""

    velocity_true = random_velocity_model(shape, rng)
    sigma = float(rng.uniform(0.7, 1.7))
    velocity_eval = ndi.gaussian_filter(velocity_true, sigma=sigma, mode="reflect")
    smooth_noise = ndi.gaussian_filter(rng.normal(size=shape), sigma=2.0, mode="reflect")
    smooth_noise /= max(np.std(smooth_noise), 1e-12)
    velocity_eval = np.clip(
        velocity_eval + rng.uniform(0.006, 0.022) * smooth_noise,
        1.65,
        2.85,
    )
    return velocity_true, velocity_eval


def _effectivity(bound: float, realized: float) -> float:
    if realized <= 1e-15:
        return 1.0 if bound <= 1e-15 else float("inf")
    return float(bound / realized)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3 or np.std(x[mask]) <= 1e-15 or np.std(y[mask]) <= 1e-15:
        return float("nan"), float("nan")
    r = spearmanr(x[mask], y[mask])
    return float(r.statistic), float(r.pvalue)


def run_certificate_study(
    *,
    checkpoint_path: str | Path = "checkpoints/fno_phase2.pt",
    output_dir: str | Path = "results/phase4",
    n_cases: int = 80,
    seed: int = 20260930,
) -> dict:
    """Validate three nested residual certificates on held-out FWI states.

    The study deliberately reuses the Phase-3 held-out distribution. It tests
    the theorem on states where neural gradients are already known to fail,
    rather than selecting easy post-hoc examples.
    """

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
    block_sources = (geometry.source_indices[1], geometry.source_indices[-2])
    block_frequencies = (3.0, 4.0)

    rng = np.random.default_rng(seed)
    case_rows: list[CertificateCaseRow] = []
    block_rows: list[CertificateBlockRow] = []

    for case_id in range(n_cases):
        velocity_true, velocity_eval = _make_case(shape, rng)
        m_true = velocity_to_m(velocity_true).reshape(-1)
        m_eval = velocity_to_m(velocity_eval).reshape(-1)

        block_data: list[dict] = []
        exact_grad_total = np.zeros(physics.n, dtype=float)
        neural_grad_total = np.zeros(physics.n, dtype=float)
        exact_receiver_all: list[np.ndarray] = []
        neural_receiver_all: list[np.ndarray] = []

        for source_index in block_sources:
            q = physics.source(source_index)
            for frequency in block_frequencies:
                u_true = physics.solve_state(m_true, q, frequency)
                observed = physics.restrict(u_true, geometry.receiver_indices)
                _, exact_grad, u_exact, _ = physics.objective_and_gradient(
                    m_eval,
                    q,
                    observed,
                    geometry.receiver_indices,
                    frequency,
                )
                _, neural_grad, u_neural, _ = surrogate.objective_and_gradient(
                    m_eval,
                    q,
                    observed,
                    geometry.receiver_indices,
                    frequency,
                )
                exact_grad_total += exact_grad
                neural_grad_total += neural_grad
                exact_receiver_all.append(physics.restrict(u_exact, geometry.receiver_indices))
                neural_receiver_all.append(physics.restrict(u_neural, geometry.receiver_indices))
                block_data.append(
                    {
                        "source_index": int(source_index),
                        "frequency": float(frequency),
                        "q": q,
                        "observed": observed,
                        "exact_grad": exact_grad,
                    }
                )

        neural_norm = float(np.linalg.norm(neural_grad_total))
        if neural_norm <= 1e-14:
            direction = -exact_grad_total / max(np.linalg.norm(exact_grad_total), 1e-14)
        else:
            direction = -neural_grad_total / neural_norm

        exact_dd_total = float(np.dot(exact_grad_total, direction))
        neural_dd_total = float(np.dot(neural_grad_total, direction))
        realized_total = abs(exact_dd_total - neural_dd_total)

        exact_r = np.concatenate(exact_receiver_all)
        neural_r = np.concatenate(neural_receiver_all)
        receiver_error = float(
            np.linalg.norm(neural_r - exact_r) / max(np.linalg.norm(exact_r), 1e-12)
        )
        grad_error = relative_l2(exact_grad_total, neural_grad_total)
        grad_cos = cosine_similarity(exact_grad_total, neural_grad_total)
        failure = bool(
            receiver_error <= 0.15 and (grad_error >= 0.50 or grad_cos <= 0.90)
        )

        # A(m, f) is source-independent.  Form only the receiver resolvent
        # P A^{-1} with sparse multi-RHS solves; no dense inverse is required.
        constants: dict[float, dict[str, object]] = {}
        for frequency in block_frequencies:
            receiver_matrix = physics.receiver_resolvent_matrix(
                m_eval, frequency, geometry.receiver_indices
            )
            alpha = physics.receiver_resolvent_norm(
                m_eval,
                frequency,
                geometry.receiver_indices,
                receiver_resolvent_matrix=receiver_matrix,
            )
            kappa = physics.directional_receiver_tangent_resolvent_norm(
                m_eval,
                frequency,
                geometry.receiver_indices,
                direction,
                receiver_resolvent_matrix=receiver_matrix,
            )
            constants[float(frequency)] = {
                "receiver_matrix": receiver_matrix,
                "alpha": alpha,
                "kappa": kappa,
            }

        directional_bound_total = 0.0
        primal_sum = 0.0
        tangent_sum = 0.0

        for block in block_data:
            frequency = float(block["frequency"])
            c = constants[frequency]
            cert_directional = certify_direction(
                physics=physics,
                surrogate=surrogate,
                m=m_eval,
                q=block["q"],
                observed=block["observed"],
                receiver_indices=geometry.receiver_indices,
                frequency_hz=frequency,
                direction=direction,
                stability_mode="directional",
                receiver_resolvent_matrix=np.asarray(c["receiver_matrix"]),
                receiver_resolvent_norm=float(c["alpha"]),
                directional_tangent_resolvent_norm=float(c["kappa"]),
            )
            exact_block_dd = float(np.dot(np.asarray(block["exact_grad"]), direction))
            realized_block = abs(exact_block_dd - cert_directional.surrogate_directional_derivative)
            directional_bound_total += cert_directional.directional_error_bound
            primal_sum += cert_directional.primal_residual_norm
            tangent_sum += cert_directional.tangent_residual_norm
            block_rows.append(
                CertificateBlockRow(
                    case_id=case_id,
                    label=f"s{block['source_index']}_f{frequency:g}",
                    frequency_hz=frequency,
                    exact_directional_derivative=exact_block_dd,
                    surrogate_directional_derivative=cert_directional.surrogate_directional_derivative,
                    realized_directional_error=realized_block,
                    primal_residual_norm=cert_directional.primal_residual_norm,
                    tangent_residual_norm=cert_directional.tangent_residual_norm,
                    receiver_resolvent_norm=float(c["alpha"]),
                    directional_tangent_resolvent_norm=float(c["kappa"]),
                    directional_error_bound=cert_directional.directional_error_bound,
                )
            )

        true_descent = exact_dd_total < 0.0
        case_rows.append(
            CertificateCaseRow(
                case_id=case_id,
                receiver_relative_l2=receiver_error,
                gradient_relative_l2=float(grad_error),
                gradient_cosine_similarity=float(grad_cos),
                preregistered_failure=failure,
                exact_directional_derivative=exact_dd_total,
                surrogate_directional_derivative=neural_dd_total,
                realized_directional_error=realized_total,
                directional_error_bound=float(directional_bound_total),
                directional_effectivity=_effectivity(directional_bound_total, realized_total),
                directional_certified_descent=neural_dd_total + directional_bound_total < 0.0,
                true_descent=true_descent,
                primal_residual_sum=float(primal_sum),
                tangent_residual_sum=float(tangent_sum),
            )
        )

    case_path = out / "certificate_cases.csv"
    with case_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(case_rows[0]).keys()))
        w.writeheader()
        for row in case_rows:
            w.writerow(asdict(row))

    block_path = out / "certificate_blocks.csv"
    with block_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(block_rows[0]).keys()))
        w.writeheader()
        for row in block_rows:
            w.writerow(asdict(row))

    realized = np.asarray([r.realized_directional_error for r in case_rows])
    true_descent = np.asarray([r.true_descent for r in case_rows], dtype=bool)
    failures = np.asarray([r.preregistered_failure for r in case_rows], dtype=bool)
    tangent = np.asarray([r.tangent_residual_sum for r in case_rows])
    directional_bound = np.asarray([r.directional_error_bound for r in case_rows])
    directional_certified = np.asarray(
        [r.directional_certified_descent for r in case_rows], dtype=bool
    )
    covered = realized <= directional_bound + 1e-8 * np.maximum(1.0, directional_bound)
    effectivity = directional_bound / np.maximum(realized, 1e-15)
    false_cert = directional_certified & ~true_descent
    mode_summary = {
        "coverage": float(np.mean(covered)),
        "violations": int(np.count_nonzero(~covered)),
        "certified_descent_fraction": float(np.mean(directional_certified)),
        "certified_among_true_descent_fraction": float(
            np.mean(directional_certified[true_descent]) if np.any(true_descent) else 0.0
        ),
        "false_certifications": int(np.count_nonzero(false_cert)),
        "effectivity_median": float(np.median(effectivity)),
        "effectivity_p90": float(np.quantile(effectivity, 0.9)),
        "effectivity_max": float(np.max(effectivity)),
        "bound_vs_realized_spearman_rho": _safe_spearman(directional_bound, realized)[0],
    }

    summary = {
        "seed": seed,
        "n_cases": n_cases,
        "blocks_per_case": len(block_sources) * len(block_frequencies),
        "failure_cases_reproduced": int(np.count_nonzero(failures)),
        "true_descent_fraction_for_neural_direction": float(np.mean(true_descent)),
        "tangent_residual_vs_realized_error_spearman_rho": _safe_spearman(tangent, realized)[0],
        "directional_reference_certificate": mode_summary,
        "exit_criteria": {
            "at_least_50_cases": n_cases >= 50,
            "directional_bound_full_coverage": mode_summary["coverage"] >= 1.0 - 1e-12,
            "no_false_directional_certifications": mode_summary["false_certifications"] == 0,
            "block_level_results_saved": True,
        },
    }
    summary["passed"] = all(summary["exit_criteria"].values())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    positive = realized > 1e-14
    fig, ax = plt.subplots(figsize=(6.5, 5.1), constrained_layout=True)
    ax.scatter(realized[positive], directional_bound[positive], s=24, alpha=0.7)
    if np.any(positive):
        lo = max(float(np.min(realized[positive])) * 0.6, 1e-12)
        hi = max(float(np.max(directional_bound[positive])) * 1.3, lo * 10)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, label="ideal bound")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("realized |DΦ - DΦ̂|")
    ax.set_ylabel("direction-aware a posteriori upper bound")
    ax.set_title("Phase 4: deterministic residual certificate")
    ax.legend(fontsize=8)
    fig.savefig(out / "bound_vs_realized_error.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    ax.scatter(tangent[~failures], realized[~failures], s=24, alpha=0.65, label="other cases")
    if np.any(failures):
        ax.scatter(tangent[failures], realized[failures], s=38, marker="x", label="Phase-3 failures")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sum of tangent PDE residual norms")
    ax.set_ylabel("realized directional-derivative error")
    ax.set_title("Tangent residual tracks derivative inconsistency")
    ax.legend(fontsize=8)
    fig.savefig(out / "tangent_residual_vs_error.png", dpi=220)
    plt.close(fig)

    return summary
