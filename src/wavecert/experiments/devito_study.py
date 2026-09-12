from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

from wavecert.benchmarks.marmousi import downsample_velocity, load_marmousi_binary
from wavecert.physics.devito_backend import DevitoAcousticConfig, DevitoAcousticFWI


def run_devito_study(
    *,
    marmousi_path: str | Path,
    output_dir: str | Path = "results/devito_real",
    shape: tuple[int, int] = (64, 128),
    receivers: int = 32,
    optimizer_iterations: int = 5,
) -> dict:
    """Actually execute Devito forward, adjoint-gradient, and L-BFGS-B on Marmousi."""

    path = Path(marmousi_path)
    if not path.exists():
        raise FileNotFoundError(path)
    velocity = load_marmousi_binary(path)
    if float(np.median(velocity)) > 20.0:
        velocity = velocity / 1000.0
    vp = downsample_velocity(velocity, shape).T.astype(np.float32)
    nx, nz = vp.shape
    dx = dz = 10.0

    src = np.array([[0.50 * (nx - 1) * dx, 20.0]], dtype=np.float32)
    rec = np.zeros((receivers, 2), dtype=np.float32)
    rec[:, 0] = np.linspace(20.0, (nx - 3) * dx, receivers)
    rec[:, 1] = 20.0

    config = DevitoAcousticConfig(
        spacing=(dx, dz),
        nbl=20,
        tn=500.0,
        f0=0.010,
        space_order=4,
    )
    truth = DevitoAcousticFWI(vp, source_coordinates=src, receiver_coordinates=rec, config=config)
    observed, _ = truth.forward(save=False)

    vp0 = ndi.gaussian_filter(vp, sigma=2.0)
    inverse = DevitoAcousticFWI(
        vp0, source_coordinates=src, receiver_coordinates=rec, config=config
    )
    initial_objective, gradient = inverse.objective_and_gradient(observed)
    result = inverse.run_lbfgsb(
        observed,
        maxiter=optimizer_iterations,
        vmin=max(1.2, float(np.min(vp)) - 0.2),
        vmax=float(np.max(vp)) + 0.3,
        ftol=1e-6,
    )

    summary = {
        "marmousi_path": str(path),
        "shape": list(vp.shape),
        "observed_shape": list(observed.shape),
        "forward_executed": bool(np.isfinite(observed).all()),
        "adjoint_gradient_executed": bool(np.isfinite(gradient).all()),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "initial_objective": float(initial_objective),
        "final_objective": float(result.fun),
        "objective_reduction_fraction": float(
            1.0 - float(result.fun) / max(float(initial_objective), 1e-15)
        ),
        "optimizer_iterations_limit": optimizer_iterations,
        "optimizer_status": int(result.status),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "full_external_validation": bool(
            np.isfinite(observed).all()
            and np.isfinite(gradient).all()
            and np.isfinite(result.fun)
            and float(result.fun) < float(initial_objective)
        ),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
