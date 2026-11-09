"""Diagnose geological FNO errors using training/model-selection data only.

This is a development diagnostic, not a new confirmation experiment. All
checkpoint comparisons use identical sampled examples and fresh random tangent
directions. No calibration or test wavefield arrays are loaded. The smoothed
state gradient decomposition fixes either the data residual or the Jacobian:

    g = J* r,  g_hat = J_hat* r_hat,
    g_same_residual = J_hat* r,  g_same_jacobian = J* r_hat.

Here * is the real adjoint for complex data and real squared slowness. These
diagnostics separate errors in predicted residuals from errors in derivatives;
they do not supply a certificate or establish an inversion/runtime benefit.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import torch

from wavecert.data.splits import verify_calibration_checkpoint, verify_dataset_provenance
from wavecert.data.wavefields import WavefieldDatasetArrays
from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.provenance import environment_record, sha256_file, source_digest
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate


def _relative_error(prediction, truth) -> float:
    """Relative L2 with an explicitly recorded 1e-15 denominator floor."""
    return float(np.linalg.norm(prediction - truth) / max(np.linalg.norm(truth), 1e-15))


def _real_cosine(left, right) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.real(np.vdot(left, right)) / denominator) if denominator > 0 else 0.0


def _sample_indices(count: int, requested: int, seed: int, role_code: int, stream: int):
    rng = np.random.default_rng(np.random.SeedSequence([seed, role_code, stream]))
    return np.sort(rng.choice(count, size=min(count, requested), replace=False))


def _neural_data_vjp(surrogate, model, source_index, frequency, receivers, data_dual):
    """Apply the neural Jacobian's real adjoint to a fixed complex data vector."""
    model_tensor = torch.tensor(model, dtype=torch.float32, requires_grad=True)
    wave = surrogate._forward_tensor(model_tensor, source_index, frequency)
    pred = wave.reshape(-1, 2)[torch.as_tensor(receivers, dtype=torch.long)]
    dual = torch.as_tensor(
        np.stack([data_dual.real, data_dual.imag], axis=-1), dtype=pred.dtype,
    )
    (gradient,) = torch.autograd.grad(torch.sum(pred * dual), model_tensor)
    return gradient.detach().numpy().astype(float)


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows, keys):
    by_parent = defaultdict(list)
    for row in rows:
        by_parent[row["model_id"]].append(row)
    return {
        "n_blocks": len(rows),
        "n_parent_models": len(by_parent),
        "block_medians": {key: float(np.median([row[key] for row in rows])) for key in keys},
        "parent_medians": {
            parent: {
                "n_blocks": len(group),
                **{key: float(np.median([row[key] for row in group])) for key in keys},
            }
            for parent, group in sorted(by_parent.items())
        },
    }


def _identity(metadata, index, checkpoint, role):
    row = metadata[index]
    return {
        "checkpoint": checkpoint, "role": role, "example_index": int(index),
        "slice_id": row["slice_id"], "model_id": row["model_id"],
        "model_type": row["model_type"], "source_index": int(row["source_index"]),
        "frequency_hz": float(row["frequency_hz"]),
    }


def run(args):
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    provenance = environment_record(root)
    data_provenance = verify_dataset_provenance(args.dataset)
    arrays = {}
    metadata = {}
    for role in ("train", "validation"):
        arrays[role] = WavefieldDatasetArrays.load(Path(args.dataset) / f"{role}.npz")
        with (Path(args.dataset) / f"{role}_metadata.csv").open(newline="") as stream:
            metadata[role] = list(csv.DictReader(stream))
        if len(metadata[role]) != len(arrays[role].models):
            raise ValueError(f"{role} metadata and array lengths disagree")
    shape = arrays["train"].shape
    if arrays["validation"].shape != shape:
        raise ValueError("training and validation shapes differ")

    forward_rows, tangent_rows, smoothed_rows = [], [], []
    checkpoint_summaries = {}
    for checkpoint in args.checkpoint:
        verify_calibration_checkpoint(checkpoint, data_provenance)
        surrogate = TrainedFNOWavefieldSurrogate.load(checkpoint, device="cpu")
        if surrogate.shape != shape:
            raise ValueError("checkpoint and data shapes differ")
        label = Path(checkpoint).name
        if label in checkpoint_summaries:
            raise ValueError("checkpoint filenames must be unique")
        checkpoint_summaries[label] = {
            "path": str(checkpoint), "sha256": sha256_file(checkpoint), "roles": {},
        }
        for role_code, role in enumerate(("train", "validation")):
            a = arrays[role]
            receivers = tuple(a.receiver_indices.tolist())
            physics = Helmholtz2D(
                shape=shape, spacing=a.spacing,
                damping_width=max(3, min(shape) // 8), damping_strength=2.0,
            )
            for start in range(0, len(a.models), 64):
                stop = min(start + 64, len(a.models))
                with torch.no_grad():
                    inputs = surrogate.encoder.encode_batch(
                        torch.as_tensor(a.models[start:stop], dtype=torch.float32),
                        torch.as_tensor(a.source_indices[start:stop]),
                        torch.as_tensor(a.frequencies_hz[start:stop]),
                    )
                    predictions = surrogate.encoder.decode_output(surrogate.model(inputs)).numpy()
                for offset, prediction in enumerate(predictions):
                    index = start + offset
                    truth = a.wavefields[index]
                    forward_rows.append({
                        **_identity(metadata[role], index, label, role),
                        "wavefield_relative_l2": _relative_error(prediction, truth),
                        "receiver_relative_l2": _relative_error(
                            prediction.reshape(-1, 2)[list(receivers)],
                            truth.reshape(-1, 2)[list(receivers)],
                        ),
                    })

            tangent_indices = _sample_indices(
                len(a.models), args.tangent_examples, args.seed, role_code, 0,
            )
            direction_rng = np.random.default_rng(np.random.SeedSequence([args.seed, role_code, 1]))
            for index in tangent_indices:
                m = a.models[index].astype(float).reshape(-1)
                direction = direction_rng.normal(size=m.size)
                direction /= np.linalg.norm(direction)
                frequency = float(a.frequencies_hz[index])
                q = physics.source(int(a.source_indices[index]))
                state = physics.solve_state(m, q, frequency)
                tangent = physics.solve_tangent(m, state, direction, frequency)
                predicted_tangent = surrogate.jvp(m, direction, q, frequency)
                tangent_rows.append({
                    **_identity(metadata[role], index, label, role),
                    "full_relative_l2": _relative_error(predicted_tangent, tangent),
                    "full_real_cosine": _real_cosine(predicted_tangent, tangent),
                    "receiver_relative_l2": _relative_error(
                        predicted_tangent[list(receivers)], tangent[list(receivers)],
                    ),
                    "receiver_real_cosine": _real_cosine(
                        predicted_tangent[list(receivers)], tangent[list(receivers)],
                    ),
                })

            # Evaluate identical smoothing diagnostics on training and validation
            # parents to distinguish state shift from parent generalization.
            smooth_indices = _sample_indices(
                len(a.models), args.smoothed_examples, args.seed, role_code, 2,
            )
            for index in smooth_indices:
                m_raw = a.models[index].astype(float).reshape(-1)
                velocity = (1.0 / np.sqrt(m_raw)).reshape(shape)
                smooth_velocity = ndi.gaussian_filter(velocity, args.smoothing_sigma, mode="reflect")
                m = (1.0 / smooth_velocity**2).reshape(-1)
                frequency = float(a.frequencies_hz[index])
                source = int(a.source_indices[index])
                q = physics.source(source)
                stored = a.wavefields[index]
                observed = (stored[..., 0] + 1j * stored[..., 1]).reshape(-1)[list(receivers)]
                _, gradient, state, residual = physics.objective_and_gradient(
                    m, q, observed, receivers, frequency,
                )
                _, neural_gradient, neural_state, neural_residual = surrogate.objective_and_gradient(
                    m, q, observed, receivers, frequency,
                )
                neural_same_residual = _neural_data_vjp(
                    surrogate, m, source, frequency, receivers, residual,
                )
                exact_same_jacobian = physics.data_vjp(
                    m, q, receivers, frequency, neural_residual,
                )
                reconstructed_neural = _neural_data_vjp(
                    surrogate, m, source, frequency, receivers, neural_residual,
                )
                identity_error = _relative_error(reconstructed_neural, neural_gradient)
                if identity_error > 1e-5:
                    raise AssertionError("neural gradient and fixed-residual VJP disagree")
                smoothed_rows.append({
                    **_identity(metadata[role], index, label, role),
                    "model_relative_shift": _relative_error(m, m_raw),
                    "true_residual_relative_norm": float(
                        np.linalg.norm(residual) / max(np.linalg.norm(state[list(receivers)]), 1e-15)
                    ),
                    "receiver_prediction_relative_l2": _relative_error(
                        neural_state[list(receivers)], state[list(receivers)],
                    ),
                    "actual_gradient_cosine": _real_cosine(gradient, neural_gradient),
                    "fixed_true_residual_gradient_cosine": _real_cosine(gradient, neural_same_residual),
                    "fixed_exact_jacobian_gradient_cosine": _real_cosine(gradient, exact_same_jacobian),
                    "neural_vjp_identity_relative_error": identity_error,
                })

            selected_forward = [r for r in forward_rows if r["checkpoint"] == label and r["role"] == role]
            selected_tangent = [r for r in tangent_rows if r["checkpoint"] == label and r["role"] == role]
            selected_smooth = [r for r in smoothed_rows if r["checkpoint"] == label and r["role"] == role]
            checkpoint_summaries[label]["roles"][role] = {
                "raw_forward": _aggregate(selected_forward, ("wavefield_relative_l2", "receiver_relative_l2")),
                "fresh_random_tangent": _aggregate(selected_tangent, (
                    "full_relative_l2", "full_real_cosine", "receiver_relative_l2", "receiver_real_cosine",
                )),
                "smoothed_gradient_decomposition": _aggregate(selected_smooth, (
                    "model_relative_shift", "true_residual_relative_norm", "receiver_prediction_relative_l2",
                    "actual_gradient_cosine", "fixed_true_residual_gradient_cosine",
                    "fixed_exact_jacobian_gradient_cosine", "neural_vjp_identity_relative_error",
                )),
                "offline_diagnostic_physics_cost": physics.cost_snapshot(),
            }

    for filename, rows in (("raw_forward.csv", forward_rows), ("fresh_random_tangents.csv", tangent_rows),
                           ("smoothed_gradient_decomposition.csv", smoothed_rows)):
        _write_csv(output / filename, rows)
    summary = {
        "schema_version": 1,
        "study_status": "training/model-selection development diagnostic; no untouched confirmation claim",
        "roles_loaded": ["train", "validation"],
        "calibration_or_test_wavefields_loaded": False,
        "seed": args.seed,
        "sampling": {
            "tangent_examples_per_role": args.tangent_examples,
            "smoothed_examples_per_role": args.smoothed_examples,
            "selection": "uniform block sampling without replacement, identical across checkpoints",
            "direction_distribution": "fresh independent Gaussian directions in physical squared slowness, unit L2",
            "direction_seed": "SeedSequence([seed, role_code, 1]); role_code train=0, validation=1",
            "selection_seed": "SeedSequence([seed, role_code, stream]); tangent stream=0, smoothing stream=2",
            "smoothing": {"variable": "velocity", "sigma_grid_cells": args.smoothing_sigma, "mode": "reflect"},
            "relative_error_denominator_floor": 1e-15,
            "zero_norm_cosine": 0.0,
        },
        "identities": {
            "actual": "g=J* r compared with g_hat=J_hat* r_hat",
            "fixed_true_residual": "g=J* r compared with J_hat* r",
            "fixed_exact_jacobian": "g=J* r compared with J* r_hat",
            "adjoint": "real model-space adjoint under Re<v,w> for complex data",
        },
        "dataset_manifest_sha256": data_provenance["dataset_manifest_sha256"],
        "role_manifest_sha256": data_provenance["role_manifest_sha256"],
        "dataset_directory": str(args.dataset),
        "loaded_array_sha256": {
            role: sha256_file(Path(args.dataset) / f"{role}.npz") for role in arrays
        },
        "checkpoints": checkpoint_summaries,
        "limitations": [
            "Training errors are in-sample; validation errors use the same parents used for model selection.",
            "Samples and blocks are correlated within parents. Block medians are descriptive and no population confidence interval is asserted.",
            "This diagnostic computes no calibration/test scores and does not modify checkpoint settings; it was designed after earlier development outcomes had been inspected.",
            "The evaluated checkpoints and held-out results were already inspected in development; this does not reset a final-test boundary.",
            "Fresh random tangents differ from the fixed training-supervision directions and are not a full Jacobian norm estimate.",
            "Stored float32 synthetic truth observations and float32 neural derivatives are numerical approximations.",
            "These are small Dirichlet reference-operator diagnostics, not time-domain or field-data validation.",
        ],
        "provenance": provenance,
        "source_sha256_at_end": source_digest(root),
    }
    summary["source_unchanged_during_run"] = summary["source_sha256_at_end"] == provenance["source_sha256"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    status = "complete" if summary["source_unchanged_during_run"] else "stale-source"
    (output / "status.json").write_text(json.dumps({"status": status}, indent=2) + "\n")
    print(json.dumps({
        "status": status,
        "checkpoints": {
            label: {role: {
                "receiver_forward_median": result["raw_forward"]["block_medians"]["receiver_relative_l2"],
                "fresh_tangent_relative_l2_median": result["fresh_random_tangent"]["block_medians"]["full_relative_l2"],
                "smoothed_gradient_cosine_median": result["smoothed_gradient_decomposition"]["block_medians"]["actual_gradient_cosine"],
            } for role, result in item["roles"].items()} for label, item in checkpoint_summaries.items()
        },
    }, indent=2, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/reference/subsurfacegen_parent_split")
    parser.add_argument("--checkpoint", action="append")
    parser.add_argument("--output", default="results/development_generalization")
    parser.add_argument("--seed", type=int, default=2026110709)
    parser.add_argument("--tangent-examples", type=int, default=48)
    parser.add_argument("--smoothed-examples", type=int, default=24)
    parser.add_argument("--smoothing-sigma", type=float, default=1.6)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.seed < 0 or min(args.tangent_examples, args.smoothed_examples, args.torch_threads) < 1:
        parser.error("seed must be nonnegative and sample/thread counts must be positive")
    if not np.isfinite(args.smoothing_sigma) or args.smoothing_sigma <= 0:
        parser.error("smoothing sigma must be finite and positive")
    if args.checkpoint is None:
        args.checkpoint = ["checkpoints/fno_parent_grouped.pt", "checkpoints/fno_parent_grouped_jvp.pt"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "status.json").write_text('{"status": "running"}\n')
    try:
        summary = run(args)
        _plot(output, summary)
    except BaseException as exc:
        (output / "status.json").write_text(json.dumps({
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
        }, indent=2) + "\n")
        raise


def _plot(output, summary):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
    labels = []
    forward, tangent, cosine = [], [], []
    for label, checkpoint in summary["checkpoints"].items():
        short = "Forward" if "jvp" not in label else "JVP"
        for role, values in checkpoint["roles"].items():
            labels.append(f"{short}\n{role}")
            forward.append(values["raw_forward"]["block_medians"]["receiver_relative_l2"])
            tangent.append(values["fresh_random_tangent"]["block_medians"]["full_relative_l2"])
            cosine.append(values["smoothed_gradient_decomposition"]["block_medians"]["actual_gradient_cosine"])
    colors = ["#0072B2", "#56B4E9", "#D55E00", "#E69F00"]
    for axis, values, title in zip(axes, (forward, tangent, cosine), (
        "Raw receiver prediction error", "Fresh random tangent error", "Smoothed-state gradient cosine",
    ), strict=True):
        axis.bar(np.arange(len(labels)), values, color=colors[:len(labels)])
        axis.set_xticks(np.arange(len(labels)), labels, fontsize=8)
        axis.set_title(title, fontsize=10)
        axis.axhline(0, color="#777777", linewidth=.7)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Median relative L2 (lower is better)")
    axes[1].set_ylabel("Median relative L2 (lower is better)")
    axes[2].set_ylabel("Median real cosine (higher is better)")
    axes[2].set_ylim(-1, 1)
    fig.suptitle("Development diagnostics: training and model-selection parents only", fontsize=12)
    fig.savefig(output / "development_diagnostics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
