from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wavecert.data.wavefields import build_wavefield_dataset
from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate


def main() -> None:
    checkpoint = Path("checkpoints/fno_phase2.pt")
    summary_path = Path("results/phase2/summary.json")
    surrogate = TrainedFNOWavefieldSurrogate.load(checkpoint)
    shape = surrogate.shape
    data = build_wavefield_dataset(
        n_samples=4,
        shape=shape,
        spacing=0.05,
        frequencies_hz=(2.5, 3.0, 3.5, 4.0),
        n_source_positions=6,
        n_receivers=min(20, shape[1] - 4),
        seed=20260912,
    )
    physics = Helmholtz2D(
        shape=shape,
        spacing=0.05,
        damping_width=max(3, min(shape) // 8),
        damping_strength=2.0,
    )
    i = 0
    m = data.models[i].reshape(-1).astype(float)
    q = physics.source(int(data.source_indices[i]))
    frequency = float(data.frequencies_hz[i])
    observed = physics.restrict(
        data.wavefields[(i + 1) % len(data.models), ..., 0].reshape(-1)
        + 1j * data.wavefields[(i + 1) % len(data.models), ..., 1].reshape(-1),
        data.receiver_indices,
    )
    f0, g, _, _ = surrogate.objective_and_gradient(
        m, q, observed, tuple(data.receiver_indices.tolist()), frequency
    )
    rng = np.random.default_rng(77)
    v = rng.normal(size=m.size)
    v /= np.linalg.norm(v)
    eps = 5e-4
    fp, _, _, _ = surrogate.objective_and_gradient(
        m + eps * v, q, observed, tuple(data.receiver_indices.tolist()), frequency
    )
    fm, _, _, _ = surrogate.objective_and_gradient(
        m - eps * v, q, observed, tuple(data.receiver_indices.tolist()), frequency
    )
    fd = (fp - fm) / (2 * eps)
    ad = float(np.dot(g, v))
    rel = abs(fd - ad) / max(abs(fd), abs(ad), 1e-12)
    validation = {
        "objective": float(f0),
        "gradient_norm": float(np.linalg.norm(g)),
        "gradient_all_finite": bool(np.all(np.isfinite(g))),
        "autodiff_directional": ad,
        "finite_difference_directional": float(fd),
        "surrogate_gradient_fd_relative_error": float(rel),
        "passed": bool(np.all(np.isfinite(g)) and rel < 2e-2),
    }
    Path("results/phase2/autodiff_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    summary = json.loads(summary_path.read_text())
    summary["autodiff_validation"] = validation
    summary["exit_criteria"]["finite_autodiff_gradient"] = validation["passed"]
    summary["passed"] = all(bool(v) for v in summary["exit_criteria"].values())
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit("Phase 2 exit criteria failed")


if __name__ == "__main__":
    main()
