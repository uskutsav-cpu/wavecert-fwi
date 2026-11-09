"""Reproduce the availability and limitations of the no-SVD stability bound."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from wavecert.experiments.certificate_study import _make_case
from wavecert.experiments.synthetic import velocity_to_m
from wavecert.physics.helmholtz import Helmholtz2D


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/stability_benchmark")
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, aggregates = [], []
    for size in (16, 24, 64):
        shape = (size, size)
        physics = Helmholtz2D(
            shape=shape, spacing=0.05, damping_width=max(3, size // 8), damping_strength=2.0,
        )
        rng = np.random.default_rng(args.seed)
        models = [velocity_to_m(_make_case(shape, rng)[1]).reshape(-1) for _ in range(args.cases)]
        for frequency in (1.0, 2.5, 3.0, 4.0):
            beta_values, seconds, exact_ratios = [], [], []
            physics.reset_cost_counters(clear_cache=True)
            online_costs = {key: 0 for key in physics.cost_snapshot()}
            for case_id, model in enumerate(models):
                before = physics.cost_snapshot()
                start = time.perf_counter()
                bound = physics.analytical_stability_bound(model, frequency)
                elapsed = time.perf_counter() - start
                after = physics.cost_snapshot()
                for key in online_costs:
                    online_costs[key] += after[key] - before[key]
                beta_values.append(bound.beta_lower_bound)
                seconds.append(elapsed)
                row = {
                    "shape": list(shape), "frequency_hz": frequency, "case_id": case_id,
                    "seconds": elapsed, **bound.to_dict(),
                }
                # Reference checks are explicitly offline and never performed at64².
                if size <= 24 and case_id < min(3, args.cases):
                    exact = physics.smallest_singular_value(model, frequency)
                    row["offline_svd_beta"] = exact
                    row["lower_bound_valid_against_svd"] = bound.beta_lower_bound <= exact + 1e-10
                    exact_ratios.append(bound.beta_lower_bound / exact)
                rows.append(row)
            costs = physics.cost_snapshot()
            aggregates.append({
                "shape": list(shape), "spacing": physics.spacing,
                "damping_width": physics.damping_width, "frequency_hz": frequency,
                "n_cases": args.cases,
                "available_cases": int(np.count_nonzero(np.asarray(beta_values) > 0)),
                "median_beta_lower_bound": float(np.median(beta_values)),
                "mean_analytic_seconds": float(np.mean(seconds)),
                "median_lower_to_svd_ratio_offline_subset": (
                    float(np.median(exact_ratios)) if exact_ratios else None
                ),
                "costs_including_offline_svd_checks": costs,
                "online_costs": online_costs,
            })
    checked = [row for row in rows if "lower_bound_valid_against_svd" in row]
    summary = {
        "schema_version": 1,
        "seed": args.seed, "cases_per_shape": args.cases,
        "model_generator": "wavecert.experiments.certificate_study._make_case; smoothed Phase4 eval models",
        "scope": (
            "Fixed-spacing discrete five-point Dirichlet Helmholtz sufficient bounds. "
            "Changing shape changes physical domain size; this is not a fixed-domain convergence study. "
            "No success is inferred for real field data, PML, time-domain physics, or arbitrary OOD geology."
        ),
        "failure_behavior": "zero is unavailable; deployment must use explicit exact fallback or abstain",
        "roundoff_semantics": "ordinary floating point with downward allowance; not interval arithmetic",
        "python": platform.python_version(), "numpy": np.__version__,
        "aggregates": aggregates, "cases": rows,
        "offline_svd_checks": len(checked),
        "all_offline_svd_checks_pass": all(row["lower_bound_valid_against_svd"] for row in checked),
        "no_svd_or_linear_solve_in_online_bound": all(
            a["online_costs"][key] == 0
            for a in aggregates for key in (
                "factorizations", "forward_rhs", "adjoint_rhs", "tangent_rhs", "stability_rhs", "svd_evaluations",
            )
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
