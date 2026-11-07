from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

from wavecert.benchmarks.marmousi import downsample_velocity, load_marmousi_binary
from wavecert.physics.devito_backend import DevitoAcousticConfig, DevitoAcousticFWI
from wavecert.provenance import environment_record, sha256_file


def run_devito_study(
    *,
    marmousi_path: str | Path,
    output_dir: str | Path = "results/devito_real",
    shape: tuple[int, int] = (64, 128),
    receivers: int = 32,
    optimizer_iterations: int = 5,
) -> dict:
    """Verify a fixed-dt Devito baseline on downsampled public Marmousi.

    This exercises independent time-domain forward/adjoint/optimization code.
    It does not implement or validate WaveCert's time-domain certificate.
    """
    path = Path(marmousi_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if min(shape) < 8 or receivers < 1 or optimizer_iterations < 1:
        raise ValueError("shape dimensions must be >=8 and receiver/iteration counts positive")
    velocity = load_marmousi_binary(path)
    if float(np.median(velocity)) > 20.0:
        velocity = velocity / 1000.0
    vp = downsample_velocity(velocity, shape).T.astype(np.float64)
    nx, _ = vp.shape
    dx = dz = 10.0
    src = np.array([[0.50 * (nx - 1) * dx, 20.0]], dtype=np.float64)
    rec = np.zeros((receivers, 2), dtype=np.float64)
    rec[:, 0] = np.linspace(20.0, (nx - 3) * dx, receivers)
    rec[:, 1] = 20.0
    vmin = max(1.2, float(np.min(vp)) - 0.2)
    vmax = float(np.max(vp)) + 0.3
    config = DevitoAcousticConfig(
        spacing=(dx, dz), nbl=20, tn=500.0, f0=0.010,
        space_order=4, max_velocity=vmax, dtype="float64",
    )
    truth = DevitoAcousticFWI(vp, source_coordinates=src, receiver_coordinates=rec, config=config)
    observed, _ = truth.forward(save=False)
    times = truth.time_values
    vp0 = ndi.gaussian_filter(vp, sigma=2.0)
    inverse = DevitoAcousticFWI(
        vp0, source_coordinates=src, receiver_coordinates=rec,
        config=replace(config, dt=truth.dt),
    )
    initial_objective, gradient = inverse.objective_and_gradient(observed, observed_time_values=times)
    gradient_check = inverse.check_gradient(observed, observed_time_values=times)
    result = inverse.run_lbfgsb(
        observed, observed_time_values=times, maxiter=optimizer_iterations,
        vmin=vmin, vmax=vmax, ftol=1e-6,
    )
    baseline_validated = bool(
        np.isfinite(observed).all() and np.isfinite(gradient).all()
        and np.isfinite(result.fun) and float(result.fun) < float(initial_objective)
        and gradient_check["passed"] and np.array_equal(times, inverse.time_values)
    )
    summary = {
        "marmousi_path": str(path), "marmousi_sha256": sha256_file(path),
        "scope": "Independent time-domain exact-physics baseline on downsampled public Marmousi velocity with synthetic observations; no neural surrogate or WaveCert certificate is exercised.",
        "shape": list(vp.shape), "spacing_m": [dx, dz], "velocity_units": "km/s",
        "time_units": "ms", "dtype": "float64", "fixed_dt": truth.dt,
        "time_samples": len(times), "observed_shape": list(observed.shape),
        "truth_and_inverse_sampling_identical": bool(np.array_equal(times, inverse.time_values)),
        "source_coordinates_m": src.tolist(), "receiver_coordinates_m": rec.tolist(),
        "forward_executed": bool(np.isfinite(observed).all()),
        "adjoint_gradient_executed": bool(np.isfinite(gradient).all()),
        "gradient_check": gradient_check, "gradient_norm": float(np.linalg.norm(gradient)),
        "initial_objective": float(initial_objective), "final_objective": float(result.fun),
        "objective_reduction_fraction": float(1.0 - float(result.fun) / max(float(initial_objective), 1e-15)),
        "optimizer_iterations_limit": optimizer_iterations, "optimizer_iterations": int(result.nit),
        "optimizer_evaluations": int(result.nfev), "optimizer_status": int(result.status),
        "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
        "independent_baseline_validated": baseline_validated,
        "wavecert_time_domain_certificate_validated": False,
        "full_external_validation": False,
        "limitations": [
            "One source geometry and one downsampled velocity model; this is not a broad benchmark.",
            "Observed data are synthesized by Devito, not measured field data.",
            "Downsampling and 10m spacing define a smaller discrete model, not the original full physical extent.",
            "No surrogate speedup, selective-fallback advantage, or time-domain certification is established.",
            "A finite-difference gradient check is numerical evidence, not a formal continuous-PDE guarantee.",
        ],
        "provenance": environment_record(Path(__file__).resolve().parents[3]),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
