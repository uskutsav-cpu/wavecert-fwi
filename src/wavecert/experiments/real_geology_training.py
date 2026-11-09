from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from wavecert.data.splits import audit_dataset_roles, verify_dataset_provenance
from wavecert.data.wavefields import WavefieldDatasetArrays
from wavecert.experiments.derivative_training import (
    make_tangent_labels,
    normalized_tangent_target,
    normalized_value_and_tangent,
)
from wavecert.provenance import sha256_file
from wavecert.surrogates.fno import (
    FNO2dWavefield,
    FNOConfig,
    FNOInputEncoder,
    WavefieldNormalization,
    checkpoint_payload,
)


def _relative_l2_batch(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    p = pred.reshape(pred.shape[0], -1)
    t = truth.reshape(truth.shape[0], -1)
    return np.linalg.norm(p - t, axis=1) / np.maximum(np.linalg.norm(t, axis=1), 1e-12)


def _predict(model, encoder, arrays, device: str, batch_size: int = 64) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(arrays.models, dtype=torch.float32),
            torch.as_tensor(arrays.source_indices, dtype=torch.long),
            torch.as_tensor(arrays.frequencies_hz, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for models, sources, frequencies in loader:
            models = models.to(device)
            sources = sources.to(device)
            frequencies = frequencies.to(device)
            y = encoder.decode_output(model(encoder.encode_batch(models, sources, frequencies)))
            out.append(y.cpu().numpy())
    return np.concatenate(out, axis=0)


def _evaluate(model, encoder, arrays, device: str) -> tuple[dict, np.ndarray, np.ndarray]:
    pred = _predict(model, encoder, arrays, device)
    wave_error = _relative_l2_batch(pred, arrays.wavefields)
    n = len(pred)
    pred_rec = pred.reshape(n, -1, 2)[:, arrays.receiver_indices]
    true_rec = arrays.wavefields.reshape(n, -1, 2)[:, arrays.receiver_indices]
    receiver_error = _relative_l2_batch(pred_rec, true_rec)
    return (
        {
            "n_examples": int(n),
            "wavefield_relative_l2_mean": float(np.mean(wave_error)),
            "wavefield_relative_l2_median": float(np.median(wave_error)),
            "wavefield_relative_l2_p90": float(np.quantile(wave_error, 0.9)),
            "receiver_relative_l2_mean": float(np.mean(receiver_error)),
            "receiver_relative_l2_median": float(np.median(receiver_error)),
            "receiver_relative_l2_p90": float(np.quantile(receiver_error, 0.9)),
        },
        wave_error,
        receiver_error,
    )


def run_real_geology_training(
    *,
    dataset_dir: str | Path = "data/external/subsurfacegen-wavecert",
    checkpoint_path: str | Path = "checkpoints/fno_subsurfacegen_v2.pt",
    output_dir: str | Path = "results/real_geology_v2",
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 15,
    seed: int = 20260911,
    device: str = "cpu",
    config: FNOConfig | None = None,
    jacobian_weight: float = 0.0,
    jacobian_seed: int = 20261107,
) -> dict:
    """Train forward-only or derivative-informed FNO using train/validation only.

    A positive ``jacobian_weight`` adds one fixed random-direction exact tangent
    label per training/model-selection example. The model-selection score is
    forward normalized MSE plus the weighted normalized tangent MSE. This is
    an established derivative-supervision baseline, not a novel WaveCert claim.
    Calibration/test derivatives are never generated for training or selection.
    """

    started = time.perf_counter()
    if not np.isfinite(jacobian_weight) or jacobian_weight < 0:
        raise ValueError("jacobian_weight must be finite and nonnegative")
    if isinstance(jacobian_seed, bool) or int(jacobian_seed) != jacobian_seed or jacobian_seed < 0:
        raise ValueError("jacobian_seed must be a nonnegative integer")

    if config is None:
        config = FNOConfig(modes_z=12, modes_x=12, width=48, depth=4, padding=2)

    dataset_dir = Path(dataset_dir)
    split_audit = audit_dataset_roles(dataset_dir)
    provenance = verify_dataset_provenance(dataset_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train = WavefieldDatasetArrays.load(dataset_dir / "train.npz")
    validation = WavefieldDatasetArrays.load(dataset_dir / "validation.npz")
    test_id = WavefieldDatasetArrays.load(dataset_dir / "test_id.npz")
    test_ood = WavefieldDatasetArrays.load(dataset_dir / "test_ood.npz")
    if not (train.shape == validation.shape == test_id.shape == test_ood.shape):
        raise ValueError("all real-geology datasets must share one grid")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cpu":
        torch.set_num_threads(min(8, max(1, torch.get_num_threads())))

    norm = WavefieldNormalization.from_arrays(train.models, train.frequencies_hz, train.wavefields)
    encoder = FNOInputEncoder(train.shape, norm)
    model = FNO2dWavefield(config).to(device)

    training_tensors = [
        torch.as_tensor(train.models, dtype=torch.float32),
        torch.as_tensor(train.source_indices, dtype=torch.long),
        torch.as_tensor(train.frequencies_hz, dtype=torch.float32),
        torch.as_tensor(train.wavefields, dtype=torch.float32),
    ]
    jacobian_labels = {}
    label_started = time.perf_counter()
    if jacobian_weight > 0:
        labels = {}
        for role, arrays in (("train", train), ("validation", validation)):
            labels[role] = make_tangent_labels(arrays, role=role, seed=jacobian_seed)
            path = out / f"jacobian_{role}_labels.npz"
            np.savez_compressed(path, directions=labels[role].directions, tangents=labels[role].tangents)
            jacobian_labels[role] = {
                **labels[role].metadata,
                "artifact": path.name,
                "artifact_sha256": sha256_file(path),
            }
        training_tensors += [
            torch.as_tensor(labels["train"].directions, dtype=torch.float32),
            torch.as_tensor(labels["train"].tangents, dtype=torch.float32),
        ]
        vd = torch.as_tensor(labels["validation"].directions, dtype=torch.float32, device=device)
        vt = torch.as_tensor(labels["validation"].tangents, dtype=torch.float32, device=device)
    label_seconds = time.perf_counter() - label_started
    loader = DataLoader(
        TensorDataset(*training_tensors),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    vm = torch.as_tensor(validation.models, dtype=torch.float32, device=device)
    vs = torch.as_tensor(validation.source_indices, dtype=torch.long, device=device)
    vf = torch.as_tensor(validation.frequencies_hz, dtype=torch.float32, device=device)
    vy = torch.as_tensor(validation.wavefields, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=1e-4
    )
    receiver_idx = torch.as_tensor(train.receiver_indices, dtype=torch.long, device=device)
    scale = max(0.5 * (norm.output_std_real + norm.output_std_imag), 1e-8)

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0

    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        jacobian_total = 0.0
        for batch in loader:
            m, s, f, y = batch[:4]
            m, s, f, y = m.to(device), s.to(device), f.to(device), y.to(device)
            if jacobian_weight > 0:
                direction, tangent = (tensor.to(device) for tensor in batch[4:])
                pred_norm, pred_tangent = normalized_value_and_tangent(model, encoder, m, s, f, direction)
                jacobian_loss = torch.mean((pred_tangent - normalized_tangent_target(encoder, tangent)) ** 2)
            else:
                pred_norm = model(encoder.encode_batch(m, s, f))
                jacobian_loss = torch.zeros((), dtype=pred_norm.dtype, device=device)
            target_norm = encoder.normalize_target(y).permute(0, 3, 1, 2)
            loss_full = torch.mean((pred_norm - target_norm) ** 2)
            pred = encoder.decode_output(pred_norm)
            p = pred.reshape(pred.shape[0], -1, 2)[:, receiver_idx]
            t = y.reshape(y.shape[0], -1, 2)[:, receiver_idx]
            loss_receiver = torch.mean(((p - t) / scale) ** 2)
            loss = loss_full + 0.25 * loss_receiver + jacobian_weight * jacobian_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(m)
            jacobian_total += float(jacobian_loss.detach()) * len(m)
            seen += len(m)

        model.eval()
        with torch.no_grad():
            if jacobian_weight > 0:
                val_pred, val_tangent = normalized_value_and_tangent(model, encoder, vm, vs, vf, vd)
                val_jacobian_loss = float(torch.mean((val_tangent - normalized_tangent_target(encoder, vt)) ** 2).cpu())
            else:
                val_pred = model(encoder.encode_batch(vm, vs, vf))
                val_jacobian_loss = 0.0
            val_target = encoder.normalize_target(vy).permute(0, 3, 1, 2)
            val_forward_loss = float(torch.mean((val_pred - val_target) ** 2).cpu())
            val_loss = val_forward_loss + jacobian_weight * val_jacobian_loss
            if not np.isfinite(val_loss):
                raise FloatingPointError(f"non-finite model-selection loss at epoch {epoch}")
        scheduler.step()
        train_loss = total / max(seen, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_forward_loss": val_forward_loss,
                "validation_jacobian_loss": val_jacobian_loss,
                "training_jacobian_loss": jacobian_total / max(seen, 1),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    training_seconds = time.perf_counter() - training_started

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)

    id_metrics, _, id_receiver = _evaluate(model, encoder, test_id, device)
    ood_metrics, _, ood_receiver = _evaluate(model, encoder, test_ood, device)

    summary = {
        "dataset": "SubsurfaceGen real geology with WaveCert reference wavefields",
        "dataset_provenance": provenance,
        "split_audit": split_audit,
        "calibration_role_use": "not loaded for training, early stopping, normalization, or checkpoint selection",
        "training_objective": "forward full-field normalized MSE + 0.25 receiver MSE + jacobian_weight * normalized tangent MSE",
        "model_selection_objective": "validation forward normalized MSE + jacobian_weight * normalized tangent MSE",
        "jacobian_weight": float(jacobian_weight),
        "jacobian_seed": int(jacobian_seed),
        "jacobian_labels": jacobian_labels,
        "jacobian_label_generation_seconds": label_seconds,
        "training_seconds": training_seconds,
        "seconds_before_artifact_writes": time.perf_counter() - started,
        "derivative_baseline_scope": "Existing derivative-supervision strategy used as a comparator; float32 neural derivatives and stored tangent labels are numerical approximations, not a new certification theorem.",
        "seed": seed,
        "shape": list(train.shape),
        "model_config": asdict(config),
        "normalization": asdict(norm),
        "examples": {
            "train": int(len(train.models)),
            "validation": int(len(validation.models)),
            "test_id": int(len(test_id.models)),
            "test_ood": int(len(test_ood.models)),
        },
        "best_epoch": best_epoch,
        "best_validation_normalized_mse": history[best_epoch - 1]["validation_forward_loss"],
        "best_validation_selection_score": best_val,
        "id": id_metrics,
        "ood": ood_metrics,
        "ood_to_id_receiver_error_ratio": float(
            ood_metrics["receiver_relative_l2_median"]
            / max(id_metrics["receiver_relative_l2_median"], 1e-12)
        ),
        "exit_criteria": {
            "validation_selected_checkpoint": True,
            "id_receiver_median_below_0_30": id_metrics["receiver_relative_l2_median"] < 0.30,
            "id_receiver_median_below_0_20": id_metrics["receiver_relative_l2_median"] < 0.20,
        },
    }

    torch.save(
        checkpoint_payload(
            model,
            config,
            norm,
            train.shape,
            seed=seed,
            best_epoch=best_epoch,
            history=history,
            receiver_indices=train.receiver_indices.tolist(),
            dataset="SubsurfaceGen",
            dataset_manifest_sha256=provenance["dataset_manifest_sha256"],
            training_manifest_sha256=provenance["role_manifest_sha256"],
            calibration_role_used_for_model_selection=False,
            jacobian_weight=float(jacobian_weight),
            jacobian_seed=int(jacobian_seed),
            jacobian_labels=jacobian_labels,
        ),
        checkpoint_path,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (out / "training_history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")

    fig, ax = plt.subplots(figsize=(6.8, 4.5), constrained_layout=True)
    ax.semilogy([r["epoch"] for r in history], [r["train_loss"] for r in history], label="train")
    ax.semilogy(
        [r["epoch"] for r in history],
        [r["validation_loss"] for r in history],
        label="validation",
    )
    ax.axvline(best_epoch, linestyle="--", linewidth=1, label=f"best {best_epoch}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("normalized MSE")
    ax.set_title("Real-geology FNO training")
    ax.legend()
    fig.savefig(out / "training_curve.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.5), constrained_layout=True)
    ax.hist(id_receiver, bins=20, alpha=0.65, label="ID")
    ax.hist(ood_receiver, bins=20, alpha=0.65, label="OOD")
    ax.set_xlabel("receiver relative L2")
    ax.set_ylabel("examples")
    ax.set_title("SubsurfaceGen ID vs OOD forward error")
    ax.legend()
    fig.savefig(out / "id_vs_ood_receiver_error.png", dpi=220)
    plt.close(fig)
    return summary
