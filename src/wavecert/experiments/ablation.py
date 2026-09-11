from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavecert.certificates.directional import certify_direction
from wavecert.experiments.synthetic import build_synthetic_problem
from wavecert.metrics import cosine_similarity, relative_l2
from wavecert.repair.policies import BlockCertificate, selectively_verify
from wavecert.surrogates.low_fidelity import SmoothedHelmholtzSurrogate


def run_surrogate_quality_ablation(output_dir: str | Path = "results/ablation") -> list[dict]:
    """Sweep controlled surrogate mismatch and measure certification/fallback behavior."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    problem = build_synthetic_problem()
    physics = problem.physics

    exact_grad = np.zeros(physics.n)
    exact_obj = 0.0
    for block in problem.blocks:
        q = physics.source(block.source_index)
        f, g, _, _ = physics.objective_and_gradient(
            problem.m0, q, block.observed, problem.geometry.receiver_indices, block.frequency_hz
        )
        exact_obj += f
        exact_grad += g

    betas = {
        f: physics.smallest_singular_value(problem.m0, f)
        for f in sorted({b.frequency_hz for b in problem.blocks})
    }

    settings = [
        ("very-close", 0.10, 1.0005),
        ("close", 0.20, 1.0010),
        ("moderate", 0.30, 1.0020),
        ("challenging", 0.40, 1.0030),
        ("poor", 0.55, 1.0060),
        ("severe", 0.70, 1.0100),
    ]

    rows: list[dict] = []
    for name, sigma, scale in settings:
        surrogate = SmoothedHelmholtzSurrogate(
            physics=physics,
            smoothing_sigma=sigma,
            model_scale=scale,
            model_offset=0.0,
        )
        surrogate_grad = np.zeros(physics.n)
        surrogate_obj = 0.0
        for block in problem.blocks:
            q = physics.source(block.source_index)
            f, g, _, _ = surrogate.objective_and_gradient(
                problem.m0, q, block.observed, problem.geometry.receiver_indices, block.frequency_hz
            )
            surrogate_obj += f
            surrogate_grad += g

        direction = -surrogate_grad / max(np.linalg.norm(surrogate_grad), 1e-15)
        certs = []
        policy_blocks = []
        for block in problem.blocks:
            q = physics.source(block.source_index)
            cert = certify_direction(
                physics=physics,
                surrogate=surrogate,
                m=problem.m0,
                q=q,
                observed=block.observed,
                receiver_indices=problem.geometry.receiver_indices,
                frequency_hz=block.frequency_hz,
                direction=direction,
                beta=betas[block.frequency_hz],
                validate_exact=True,
            )
            certs.append(cert)

            def exact_eval(block=block, q=q, direction=direction):
                return physics.directional_derivative(
                    problem.m0,
                    q,
                    block.observed,
                    problem.geometry.receiver_indices,
                    block.frequency_hz,
                    direction,
                )

            policy_blocks.append(
                BlockCertificate(
                    label=block.label,
                    surrogate_derivative=cert.surrogate_directional_derivative,
                    error_bound=cert.directional_error_bound,
                    exact_evaluator=exact_eval,
                )
            )

        policy = selectively_verify(policy_blocks)
        total_eta = float(sum(c.directional_error_bound for c in certs))
        realized = float(sum(c.realized_error or 0.0 for c in certs))
        rows.append(
            {
                "name": name,
                "smoothing_sigma": sigma,
                "model_scale": scale,
                "objective_relative_error": abs(surrogate_obj - exact_obj) / max(abs(exact_obj), 1e-15),
                "gradient_relative_error": relative_l2(exact_grad, surrogate_grad),
                "gradient_cosine": cosine_similarity(exact_grad, surrogate_grad),
                "certificate_error_bound_sum": total_eta,
                "realized_block_error_sum": realized,
                "bound_covers_all_blocks": all(
                    (c.realized_error or 0.0) <= c.directional_error_bound * (1 + 1e-10) + 1e-10
                    for c in certs
                ),
                "certified_without_fallback": sum(
                    c.surrogate_directional_derivative + c.directional_error_bound for c in certs
                )
                < 0,
                "exact_blocks_needed": policy.exact_evaluations,
                "fallback_fraction": policy.exact_evaluations / len(problem.blocks),
            }
        )

    (out / "ablation.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (out / "ablation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.5, 4.0), constrained_layout=True)
    ax.plot(x, [r["gradient_cosine"] for r in rows], marker="o", label="gradient cosine")
    ax.plot(x, [1.0 - r["fallback_fraction"] for r in rows], marker="s", label="fraction accepted without exact blocks")
    ax.set_xticks(x, [r["name"] for r in rows], rotation=20)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Surrogate mismatch vs gradient quality and exact-physics demand")
    ax.legend()
    fig.savefig(out / "quality_vs_fallback.png", dpi=180)
    plt.close(fig)

    return rows
