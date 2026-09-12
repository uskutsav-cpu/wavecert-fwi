from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wavecert.experiments.end_to_end import run_end_to_end_study
from wavecert.experiments.heuristic_baselines import run_heuristic_baselines


def run_baseline_study(
    *,
    output_dir: str | Path = "results/baselines",
    seeds: tuple[int, ...] = (20261101, 20261102, 20261103),
    cases: int = 4,
    iterations: int = 8,
    heuristic_cases: int = 80,
) -> dict:
    """Run trajectory repetitions plus heuristic fallback ablations."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trajectory_summaries: list[dict] = []
    for seed in seeds:
        summary = run_end_to_end_study(
            output_dir=out / f"trajectory_seed_{seed}",
            n_cases=cases,
            iterations=iterations,
            seed=seed,
        )
        trajectory_summaries.append(summary)

    policies = ("neural-only", "conformal-global", "conformal-selective", "exact")
    aggregate: dict[str, dict] = {}
    for policy in policies:
        final_error = [
            s["policies"][policy]["final_model_relative_l2_mean"] for s in trajectory_summaries
        ]
        exact_blocks = [
            s["policies"][policy]["mean_exact_gradient_blocks"] for s in trajectory_summaries
        ]
        objective_reduction = [
            s["policies"][policy]["true_objective_reduction_fraction_mean"]
            for s in trajectory_summaries
        ]
        aggregate[policy] = {
            "final_model_relative_l2_mean": float(np.mean(final_error)),
            "final_model_relative_l2_std": float(np.std(final_error, ddof=1))
            if len(final_error) > 1
            else 0.0,
            "exact_gradient_blocks_mean": float(np.mean(exact_blocks)),
            "objective_reduction_mean": float(np.mean(objective_reduction)),
        }

    heuristics = run_heuristic_baselines(
        output_dir=out / "heuristics",
        n_cases=heuristic_cases,
    )
    result = {
        "seeds": list(seeds),
        "cases_per_seed": cases,
        "iterations": iterations,
        "trajectory_policies": aggregate,
        "heuristic_direction_baselines": heuristics["policies"],
        "baseline_semantics": heuristics["semantics"],
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
