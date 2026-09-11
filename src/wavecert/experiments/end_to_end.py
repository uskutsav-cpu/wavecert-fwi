from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

from wavecert.data.wavefields import random_velocity_model
from wavecert.experiments.synthetic import velocity_to_m
from wavecert.optimization.fwi import InversionRun, run_inversion, velocity_from_m
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock, SurveyGeometry
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate


def _make_problem(shape: tuple[int, int], rng: np.random.Generator):
    velocity_true = random_velocity_model(shape, rng)
    velocity0 = ndi.gaussian_filter(velocity_true, sigma=float(rng.uniform(1.8, 2.8)), mode="reflect")
    # Introduce a low-wavenumber bias to make the inversion non-trivial.
    nz, nx = shape
    z = np.linspace(-1.0, 1.0, nz)[:, None]
    x = np.linspace(-1.0, 1.0, nx)[None, :]
    velocity0 = velocity0 + rng.uniform(-0.035, 0.035) + rng.uniform(-0.025, 0.025) * z
    velocity0 = velocity0 + rng.uniform(-0.015, 0.015) * x
    velocity0 = np.clip(velocity0, 1.65, 2.85)
    return velocity_true, velocity0


def _blocks_for_model(
    physics: Helmholtz2D,
    geometry: SurveyGeometry,
    m_true: np.ndarray,
) -> tuple[SurveyBlock, ...]:
    block_sources = (geometry.source_indices[1], geometry.source_indices[-2])
    block_frequencies = (3.0, 4.0)
    blocks: list[SurveyBlock] = []
    for source_index in block_sources:
        q = physics.source(source_index)
        for frequency in block_frequencies:
            state = physics.solve_state(m_true, q, frequency)
            observed = physics.restrict(state, geometry.receiver_indices)
            blocks.append(
                SurveyBlock(
                    source_index=int(source_index),
                    frequency_hz=float(frequency),
                    observed=observed,
                    label=f"s{source_index}_f{frequency:g}",
                )
            )
    return tuple(blocks)


def run_end_to_end_study(
    *,
    checkpoint_path: str | Path = "checkpoints/fno_phase2.pt",
    conformal_scales_path: str | Path = "results/phase5/conformal_scales.json",
    output_dir: str | Path = "results/phase7",
    n_cases: int = 4,
    iterations: int = 8,
    seed: int = 20261101,
    initial_step: float = 0.02,
) -> dict:
    """Run complete multi-iteration inversion trajectories for four policies."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    surrogate = TrainedFNOWavefieldSurrogate.load(checkpoint_path)
    calibration = json.loads(Path(conformal_scales_path).read_text())
    case_scale = float(calibration["case_scale"])
    block_scales = {k: float(v) for k, v in calibration["block_scales_by_label"].items()}

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
    policies = ("neural-only", "conformal-global", "conformal-selective", "exact")
    rng = np.random.default_rng(seed)
    all_runs: dict[tuple[int, str], InversionRun] = {}
    case_models: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict] = []

    for case_id in range(n_cases):
        velocity_true, velocity0 = _make_problem(shape, rng)
        m_true = velocity_to_m(velocity_true).reshape(-1)
        m0 = velocity_to_m(velocity0).reshape(-1)
        blocks = _blocks_for_model(physics, geometry, m_true)
        case_models[case_id] = (velocity_true, velocity0)

        for policy in policies:
            run = run_inversion(
                policy=policy,
                physics=physics,
                surrogate=surrogate,
                m0=m0,
                m_true=m_true,
                blocks=blocks,
                receiver_indices=geometry.receiver_indices,
                iterations=iterations,
                initial_step=initial_step,
                case_scale=case_scale,
                block_scales=block_scales,
            )
            all_runs[(case_id, policy)] = run
            for record in run.records:
                row = asdict(record)
                row["case_id"] = case_id
                row["policy"] = policy
                rows.append(row)

    with (out / "trajectories.csv").open("w", newline="") as f:
        fieldnames = ["case_id", "policy", *asdict(next(iter(all_runs.values())).records[0]).keys()]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    policy_summary: dict[str, dict] = {}
    for policy in policies:
        runs = [all_runs[(i, policy)] for i in range(n_cases)]
        initial_errors = np.asarray([r.records[0].model_relative_l2 for r in runs])
        final_errors = np.asarray([r.records[-1].model_relative_l2 for r in runs])
        initial_objectives = np.asarray([r.records[0].true_objective for r in runs])
        final_objectives = np.asarray([r.records[-1].true_objective for r in runs])
        exact_blocks = np.asarray([r.total_exact_gradient_blocks for r in runs], dtype=float)
        seconds = np.asarray([r.total_algorithm_seconds for r in runs], dtype=float)
        accepted = np.asarray(
            [np.mean([rec.accepted for rec in r.records[:-1]]) for r in runs], dtype=float
        )
        policy_summary[policy] = {
            "initial_model_relative_l2_mean": float(np.mean(initial_errors)),
            "final_model_relative_l2_mean": float(np.mean(final_errors)),
            "model_error_reduction_fraction_mean": float(
                np.mean(1.0 - final_errors / np.maximum(initial_errors, 1e-15))
            ),
            "true_objective_reduction_fraction_mean": float(
                np.mean(1.0 - final_objectives / np.maximum(initial_objectives, 1e-15))
            ),
            "mean_exact_gradient_blocks": float(np.mean(exact_blocks)),
            "mean_algorithm_seconds": float(np.mean(seconds)),
            "accepted_step_fraction_mean": float(np.mean(accepted)),
        }

    exact_cost = policy_summary["exact"]["mean_exact_gradient_blocks"]
    selective_cost = policy_summary["conformal-selective"]["mean_exact_gradient_blocks"]
    summary = {
        "seed": seed,
        "n_cases": n_cases,
        "iterations": iterations,
        "blocks_per_iteration": 4,
        "validation_cost_semantics": (
            "true objectives are computed offline for every recorded iterate and are excluded "
            "from policy exact-gradient-block budgets"
        ),
        "step_policy": (
            "normalised descent direction with backtracking on each policy's own merit objective; "
            "true PDE objective is not used to choose neural/WaveCert steps"
        ),
        "policies": policy_summary,
        "savings": {
            "selective_exact_gradient_block_reduction_vs_exact": float(
                1.0 - selective_cost / max(exact_cost, 1e-15)
            )
        },
        "exit_criteria": {
            "all_four_policies_completed": len(all_runs) == 4 * n_cases,
            "full_trajectories_saved": len(rows) == n_cases * len(policies) * (iterations + 1),
            "exact_objective_reported_each_iteration": True,
            "policy_costs_reported": True,
            "selective_uses_fewer_exact_blocks_than_exact": selective_cost < exact_cost,
        },
    }
    summary["passed"] = all(summary["exit_criteria"].values())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Mean convergence curves.
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    for policy in policies:
        matrix = np.asarray(
            [[rec.true_objective for rec in all_runs[(i, policy)].records] for i in range(n_cases)]
        )
        relative = matrix / np.maximum(matrix[:, :1], 1e-15)
        ax.plot(np.arange(iterations + 1), np.mean(relative, axis=0), marker="o", label=policy)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean true objective / initial true objective")
    ax.set_title("Phase 7: complete FWI trajectories")
    ax.legend(fontsize=8)
    fig.savefig(out / "objective_trajectories.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    for policy in policies:
        final_error = policy_summary[policy]["final_model_relative_l2_mean"]
        exact_blocks = policy_summary[policy]["mean_exact_gradient_blocks"]
        ax.scatter(exact_blocks, final_error, s=75)
        ax.annotate(policy, (exact_blocks, final_error), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("mean exact gradient blocks over trajectory")
    ax.set_ylabel("mean final model relative L2 error")
    ax.set_title("Accuracy versus exact-physics gradient cost")
    fig.savefig(out / "accuracy_vs_exact_cost.png", dpi=220)
    plt.close(fig)

    # Representative reconstruction panel from case 0.
    true_v, initial_v = case_models[0]
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.4), constrained_layout=True)
    panels = [("truth", true_v), ("initial", initial_v)]
    for policy in policies:
        panels.append((policy, velocity_from_m(all_runs[(0, policy)].final_model).reshape(shape)))
    vmin = min(float(np.min(p[1])) for p in panels)
    vmax = max(float(np.max(p[1])) for p in panels)
    for ax, (title, image) in zip(axes.flat, panels, strict=True):
        im = ax.imshow(image, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="velocity")
    fig.savefig(out / "representative_reconstructions.png", dpi=220)
    plt.close(fig)

    return summary
