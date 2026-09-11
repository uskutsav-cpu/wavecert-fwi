from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavecert.certificates.directional import certify_direction
from wavecert.experiments.synthetic import build_synthetic_problem
from wavecert.metrics import cosine_similarity, relative_l2
from wavecert.repair.policies import BlockCertificate, selectively_verify
from wavecert.surrogates.low_fidelity import SmoothedHelmholtzSurrogate


def run_demo(output_dir: str | Path = "results/demo", *, validate_exact: bool = True) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    problem = build_synthetic_problem()
    physics = problem.physics
    surrogate = SmoothedHelmholtzSurrogate(physics=physics, smoothing_sigma=0.3, model_scale=1.002, model_offset=0.0)

    exact_obj = 0.0
    surrogate_obj = 0.0
    exact_grad = np.zeros(physics.n)
    surrogate_grad = np.zeros(physics.n)
    per_block_cache: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {}

    for block in problem.blocks:
        q = physics.source(block.source_index)
        eo, eg, _, _ = physics.objective_and_gradient(
            problem.m0, q, block.observed, problem.geometry.receiver_indices, block.frequency_hz
        )
        so, sg, _, _ = surrogate.objective_and_gradient(
            problem.m0, q, block.observed, problem.geometry.receiver_indices, block.frequency_hz
        )
        exact_obj += eo
        surrogate_obj += so
        exact_grad += eg
        surrogate_grad += sg
        per_block_cache[block.label] = (eg, sg, eo, so)

    scale = max(np.linalg.norm(surrogate_grad), 1e-15)
    direction = -surrogate_grad / scale

    certificates = []
    policy_blocks = []
    for block in problem.blocks:
        q = physics.source(block.source_index)
        beta = physics.smallest_singular_value(problem.m0, block.frequency_hz)
        cert = certify_direction(
            physics=physics,
            surrogate=surrogate,
            m=problem.m0,
            q=q,
            observed=block.observed,
            receiver_indices=problem.geometry.receiver_indices,
            frequency_hz=block.frequency_hz,
            direction=direction,
            beta=beta,
            validate_exact=validate_exact,
        )
        certificates.append((block.label, cert))

        def exact_eval(block=block, q=q):
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

    total_surrogate_dd = float(sum(c.surrogate_directional_derivative for _, c in certificates))
    total_eta = float(sum(c.directional_error_bound for _, c in certificates))
    total_upper = total_surrogate_dd + total_eta
    total_exact_dd = float(np.dot(exact_grad, direction))
    policy = selectively_verify(policy_blocks)

    summary = {
        "problem": {
            "shape": list(physics.shape),
            "n_parameters": physics.n,
            "n_blocks": len(problem.blocks),
            "n_sources": len(problem.geometry.source_indices),
            "frequencies_hz": sorted({b.frequency_hz for b in problem.blocks}),
        },
        "objective": {
            "exact": float(exact_obj),
            "surrogate": float(surrogate_obj),
            "relative_error": abs(surrogate_obj - exact_obj) / max(abs(exact_obj), 1e-15),
        },
        "gradient": {
            "relative_l2_error": relative_l2(exact_grad, surrogate_grad),
            "cosine_similarity": cosine_similarity(exact_grad, surrogate_grad),
            "exact_norm": float(np.linalg.norm(exact_grad)),
            "surrogate_norm": float(np.linalg.norm(surrogate_grad)),
        },
        "directional_certificate": {
            "surrogate_derivative": total_surrogate_dd,
            "error_bound": total_eta,
            "certified_upper_derivative": total_upper,
            "exact_derivative": total_exact_dd,
            "realized_error": abs(total_exact_dd - total_surrogate_dd),
            "valid": bool(abs(total_exact_dd - total_surrogate_dd) <= total_eta * (1 + 1e-10) + 1e-10),
            "certified_descent_without_fallback": bool(total_upper < 0),
        },
        "selective_verification": {
            "certified_descent": policy.certified_descent,
            "repaired_blocks": list(policy.repaired_blocks),
            "exact_evaluations": policy.exact_evaluations,
            "final_upper_derivative": policy.final_upper_derivative,
        },
        "blocks": {label: cert.to_dict() for label, cert in certificates},
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    # Model figure.
    v_true = (1.0 / np.sqrt(problem.m_true)).reshape(physics.shape)
    v0 = (1.0 / np.sqrt(problem.m0)).reshape(physics.shape)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True)
    im0 = axes[0].imshow(v_true, aspect="auto")
    axes[0].set_title("True velocity")
    axes[1].imshow(v0, aspect="auto")
    axes[1].set_title("Initial velocity")
    fig.colorbar(im0, ax=axes, shrink=0.75, label="velocity (arb. units)")
    fig.savefig(out / "models.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True)
    lim = max(np.max(np.abs(exact_grad)), np.max(np.abs(surrogate_grad)), 1e-15)
    axes[0].imshow(exact_grad.reshape(physics.shape), vmin=-lim, vmax=lim, aspect="auto")
    axes[0].set_title("Exact gradient")
    axes[1].imshow(surrogate_grad.reshape(physics.shape), vmin=-lim, vmax=lim, aspect="auto")
    axes[1].set_title("Surrogate gradient")
    fig.savefig(out / "gradients.png", dpi=180)
    plt.close(fig)

    labels = [label for label, _ in certificates]
    realized = [c.realized_error or 0.0 for _, c in certificates]
    bounds = [c.directional_error_bound for _, c in certificates]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 3.8), constrained_layout=True)
    ax.bar(x - 0.18, realized, width=0.36, label="realized |error|")
    ax.bar(x + 0.18, bounds, width=0.36, label="certificate bound")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("directional-derivative error")
    ax.set_title("Per-block a-posteriori certificate")
    ax.legend()
    fig.savefig(out / "certificate_coverage.png", dpi=180)
    plt.close(fig)

    return summary
