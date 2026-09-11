from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from wavecert.experiments.synthetic import build_synthetic_problem
from wavecert.physics.verification import verify_reference_problem


def main() -> None:
    out = Path("results/phase1")
    out.mkdir(parents=True, exist_ok=True)
    problem = build_synthetic_problem(
        shape=(18, 18), frequencies_hz=(3.0,), n_sources=1, n_receivers=18
    )
    block = problem.blocks[0]
    q = problem.physics.source(block.source_index)
    report = verify_reference_problem(
        problem.physics,
        problem.m0,
        q,
        block.observed,
        problem.geometry.receiver_indices,
        block.frequency_hz,
        seed=20260910,
        n_trials=5,
    )
    payload = report.to_dict()
    payload["exit_criteria"] = {
        "dot": payload["max_dot_relative_error"] < 1e-8,
        "gradient": payload["max_gradient_relative_error"] < 5e-4,
        "taylor": payload["min_taylor_slope"] >= 1.8,
    }
    payload["passed"] = all(payload["exit_criteria"].values())
    (out / "verification.json").write_text(json.dumps(payload, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for i, result in enumerate(report.taylor_tests):
        ax.loglog(result.epsilons, result.remainders, marker="o", label=f"trial {i+1}: slope={result.fitted_slope:.3f}")
    ax.set_xlabel("step size ε")
    ax.set_ylabel("|Φ(m+εv)-Φ(m)-ε gᵀv|")
    ax.set_title("Exact adjoint-gradient Taylor verification")
    ax.legend(fontsize=8)
    fig.savefig(out / "taylor_test.png", dpi=200)
    plt.close(fig)

    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit("Phase 1 exit criteria failed")


if __name__ == "__main__":
    main()
