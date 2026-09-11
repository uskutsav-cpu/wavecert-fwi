from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from wavecert.data.wavefields import WavefieldDatasetArrays, build_wavefield_dataset
from wavecert.physics.helmholtz import Helmholtz2D
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


def _receiver_complex(wavefields: np.ndarray, receiver_indices: np.ndarray) -> np.ndarray:
    n = wavefields.shape[0]
    flat = wavefields.reshape(n, -1, 2)
    v = flat[:, receiver_indices]
    return v[..., 0] + 1j * v[..., 1]


def _predict_all(model, encoder, arrays: WavefieldDatasetArrays, device: str, batch_size: int = 64):
    models = torch.as_tensor(arrays.models, dtype=torch.float32)
    sources = torch.as_tensor(arrays.source_indices, dtype=torch.long)
    freqs = torch.as_tensor(arrays.frequencies_hz, dtype=torch.float32)
    loader = DataLoader(TensorDataset(models, sources, freqs), batch_size=batch_size, shuffle=False)
    out = []
    model.eval()
    with torch.no_grad():
        for m, s, f in loader:
            m, s, f = m.to(device), s.to(device), f.to(device)
            x = encoder.encode_batch(m, s, f)
            y = encoder.decode_output(model(x))
            out.append(y.cpu().numpy())
    return np.concatenate(out, axis=0)


def _evaluate_forward(model, encoder, arrays: WavefieldDatasetArrays, device: str) -> dict:
    pred = _predict_all(model, encoder, arrays, device)
    truth = arrays.wavefields
    wave_rel = _relative_l2_batch(pred, truth)
    pred_rec = _receiver_complex(pred, arrays.receiver_indices)
    true_rec = _receiver_complex(truth, arrays.receiver_indices)
    rec_rel = _relative_l2_batch(pred_rec, true_rec)
    mse = float(np.mean((pred - truth) ** 2))
    return {
        "wavefield_relative_l2": wave_rel,
        "receiver_relative_l2": rec_rel,
        "mse": mse,
        "predictions": pred,
    }


def _benchmark_inference(model, encoder, arrays: WavefieldDatasetArrays, device: str) -> dict:
    n = min(64, len(arrays.models))
    m = torch.as_tensor(arrays.models[:n], dtype=torch.float32, device=device)
    s = torch.as_tensor(arrays.source_indices[:n], dtype=torch.long, device=device)
    f = torch.as_tensor(arrays.frequencies_hz[:n], dtype=torch.float32, device=device)
    x = encoder.encode_batch(m, s, f)
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
        t0 = time.perf_counter()
        reps = 20
        for _ in range(reps):
            _ = model(x)
        neural_batch_seconds = (time.perf_counter() - t0) / reps

    physics = Helmholtz2D(
        shape=arrays.shape,
        spacing=arrays.spacing,
        damping_width=max(3, min(arrays.shape) // 8),
        damping_strength=2.0,
    )
    for i in range(min(3, n)):
        q = physics.source(int(arrays.source_indices[i]))
        _ = physics.solve_state(arrays.models[i].reshape(-1), q, float(arrays.frequencies_hz[i]))
    t0 = time.perf_counter()
    for i in range(n):
        q = physics.source(int(arrays.source_indices[i]))
        _ = physics.solve_state(arrays.models[i].reshape(-1), q, float(arrays.frequencies_hz[i]))
    exact_batch_seconds = time.perf_counter() - t0
    return {
        "batch_size": n,
        "neural_batch_seconds": neural_batch_seconds,
        "exact_batch_seconds": exact_batch_seconds,
        "neural_seconds_per_sample": neural_batch_seconds / n,
        "exact_seconds_per_sample": exact_batch_seconds / n,
        "throughput_speedup_exact_over_neural": exact_batch_seconds / neural_batch_seconds,
    }


def run_training(
    *,
    output_dir: str | Path = "results/phase2",
    checkpoint_path: str | Path = "checkpoints/fno_phase2.pt",
    shape: tuple[int, int] = (24, 24),
    n_train: int = 384,
    n_val: int = 72,
    n_test: int = 96,
    epochs: int = 60,
    batch_size: int = 16,
    seed: int = 20260910,
    device: str = "cpu",
    config: FNOConfig = FNOConfig(),
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cpu":
        torch.set_num_threads(min(8, max(1, torch.get_num_threads())))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    common = dict(shape=shape, spacing=0.05, frequencies_hz=(2.5, 3.0, 3.5, 4.0), n_source_positions=6, n_receivers=min(20, shape[1] - 4))
    train = build_wavefield_dataset(n_samples=n_train, seed=seed, **common)
    val = build_wavefield_dataset(n_samples=n_val, seed=seed + 1, **common)
    test = build_wavefield_dataset(n_samples=n_test, seed=seed + 2, **common)

    norm = WavefieldNormalization.from_arrays(train.models, train.frequencies_hz, train.wavefields)
    encoder = FNOInputEncoder(shape, norm)
    model = FNO2dWavefield(config).to(device)

    m_train = torch.as_tensor(train.models, dtype=torch.float32)
    s_train = torch.as_tensor(train.source_indices, dtype=torch.long)
    f_train = torch.as_tensor(train.frequencies_hz, dtype=torch.float32)
    y_train = torch.as_tensor(train.wavefields, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(m_train, s_train, f_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    val_tensors = (
        torch.as_tensor(val.models, dtype=torch.float32, device=device),
        torch.as_tensor(val.source_indices, dtype=torch.long, device=device),
        torch.as_tensor(val.frequencies_hz, dtype=torch.float32, device=device),
        torch.as_tensor(val.wavefields, dtype=torch.float32, device=device),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=2e-4)
    history: list[dict] = []
    best_val = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        seen = 0
        for m, s, f, y in train_loader:
            m, s, f, y = m.to(device), s.to(device), f.to(device), y.to(device)
            x = encoder.encode_batch(m, s, f)
            target = encoder.normalize_target(y).permute(0, 3, 1, 2)
            pred = model(x)
            loss_full = torch.mean((pred - target) ** 2)

            # Slightly emphasize the acquisition surface while preserving full-wavefield training.
            pred_dec = encoder.decode_output(pred)
            target_dec = y
            flat_p = pred_dec.reshape(pred_dec.shape[0], -1, 2)
            flat_t = target_dec.reshape(target_dec.shape[0], -1, 2)
            idx = torch.as_tensor(train.receiver_indices, dtype=torch.long, device=device)
            scale = 0.5 * (norm.output_std_real + norm.output_std_imag)
            loss_rec = torch.mean(((flat_p[:, idx] - flat_t[:, idx]) / max(scale, 1e-8)) ** 2)
            loss = loss_full + 0.20 * loss_rec

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.detach()) * len(m)
            seen += len(m)

        model.eval()
        with torch.no_grad():
            vm, vs, vf, vy = val_tensors
            val_pred = model(encoder.encode_batch(vm, vs, vf))
            val_target = encoder.normalize_target(vy).permute(0, 3, 1, 2)
            val_loss = float(torch.mean((val_pred - val_target) ** 2).cpu())
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(seen, 1),
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)

    test_metrics = _evaluate_forward(model, encoder, test, device)
    timing = _benchmark_inference(model, encoder, test, device)
    wave_rel = test_metrics.pop("wavefield_relative_l2")
    rec_rel = test_metrics.pop("receiver_relative_l2")
    test_metrics.pop("predictions")

    summary = {
        "seed": seed,
        "shape": list(shape),
        "data": {"n_train": n_train, "n_val": n_val, "n_test": n_test},
        "model_config": asdict(config),
        "normalization": asdict(norm),
        "best_epoch": best_epoch,
        "best_validation_normalized_mse": best_val,
        "test": {
            "wavefield_relative_l2_median": float(np.median(wave_rel)),
            "wavefield_relative_l2_mean": float(np.mean(wave_rel)),
            "wavefield_relative_l2_p90": float(np.quantile(wave_rel, 0.9)),
            "receiver_relative_l2_median": float(np.median(rec_rel)),
            "receiver_relative_l2_mean": float(np.mean(rec_rel)),
            "receiver_relative_l2_p90": float(np.quantile(rec_rel, 0.9)),
            "mse": float(test_metrics["mse"]),
        },
        "timing": timing,
    }
    summary["exit_criteria"] = {
        "real_trainable_neural_operator": True,
        "validation_selected_checkpoint": True,
        "median_receiver_relative_l2_le_0_20": summary["test"]["receiver_relative_l2_median"] <= 0.20,
        "finite_autodiff_gradient": None,
    }

    torch.save(
        checkpoint_payload(
            model,
            config,
            norm,
            shape,
            seed=seed,
            best_epoch=best_epoch,
            history=history,
            receiver_indices=test.receiver_indices.tolist(),
            data_config=common,
        ),
        checkpoint_path,
    )

    (out / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.semilogy([r["epoch"] for r in history], [r["train_loss"] for r in history], label="train")
    ax.semilogy([r["epoch"] for r in history], [r["val_loss"] for r in history], label="validation")
    ax.axvline(best_epoch, linestyle="--", linewidth=1, label=f"best epoch {best_epoch}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("normalized MSE")
    ax.set_title("Forward-only FNO training")
    ax.legend()
    fig.savefig(out / "training_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.hist(rec_rel, bins=20, alpha=0.7, label="receiver")
    ax.hist(wave_rel, bins=20, alpha=0.6, label="full wavefield")
    ax.axvline(0.20, linestyle="--", linewidth=1, label="Phase-2 receiver target")
    ax.set_xlabel("relative L2 error")
    ax.set_ylabel("test samples")
    ax.set_title("Held-out forward errors")
    ax.legend()
    fig.savefig(out / "forward_error_histogram.png", dpi=200)
    plt.close(fig)

    return summary
