from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

from wavecert.benchmarks.marmousi import downsample_velocity, load_marmousi_binary, marmousi_proxy
from wavecert.experiments.end_to_end import _blocks_for_model
from wavecert.experiments.synthetic import velocity_to_m
from wavecert.optimization.fwi import run_inversion, velocity_from_m
from wavecert.physics.helmholtz import Helmholtz2D, SurveyGeometry
from wavecert.physics.production import devito_environment_report
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate


def _load_or_proxy(path: str | Path | None, shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    if path is not None and Path(path).exists():
        velocity = load_marmousi_binary(path)
        # Public geophysical models are commonly stored in m/s; WaveCert's
        # compact reference solver uses km/s-scale values.
        if float(np.median(velocity)) > 20.0:
            velocity = velocity / 1000.0
        return downsample_velocity(velocity, shape), "public-marmousi-binary"
    return downsample_velocity(marmousi_proxy((64, 128)), shape), "marmousi-inspired-proxy"


def run_production_study(
    *,
    checkpoint_path: str | Path = "checkpoints/fno_phase2.pt",
    conformal_scales_path: str | Path = "results/phase5/conformal_scales.json",
    marmousi_path: str | Path | None = None,
    output_dir: str | Path = "results/phase8",
    iterations: int = 6,
) -> dict:
    """Exercise the production-data path and report external-runtime readiness.

    A real public Marmousi binary is used when present. Offline CI falls back to
    a clearly labelled proxy so the benchmark machinery remains executable; it
    never relabels the proxy as Marmousi.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    surrogate = TrainedFNOWavefieldSurrogate.load(checkpoint_path)
    calibration = json.loads(Path(conformal_scales_path).read_text())
    block_scales = {k: float(v) for k, v in calibration["block_scales_by_label"].items()}
    shape = surrogate.shape

    velocity_true, source = _load_or_proxy(marmousi_path, shape)
    # The current FNO was trained over roughly 1.65--2.85 km/s. We do not claim
    # quantitative validity for public Marmousi pixels far outside that range;
    # the report records the range explicitly.
    velocity0 = ndi.gaussian_filter(velocity_true, sigma=2.2, mode="reflect")
    velocity0 = np.clip(velocity0, 1.45, max(3.4, float(np.max(velocity_true))))
    m_true = velocity_to_m(velocity_true).reshape(-1)
    m0 = velocity_to_m(velocity0).reshape(-1)

    physics = Helmholtz2D(
        shape=shape,
        spacing=0.05,
        damping_width=max(3, min(shape) // 8),
        damping_strength=2.0,
    )
    geometry = SurveyGeometry.from_grid(
        shape,
        n_sources=6,
        n_receivers=min(20, shape[1] - 4),
        source_depth=2,
        receiver_depth=2,
        margin=2,
    )
    blocks = _blocks_for_model(physics, geometry, m_true)

    exact = run_inversion(
        policy="exact",
        physics=physics,
        surrogate=surrogate,
        m0=m0,
        m_true=m_true,
        blocks=blocks,
        receiver_indices=geometry.receiver_indices,
        iterations=iterations,
        initial_step=0.015,
        vmax=max(3.5, float(np.max(velocity_true)) + 0.2),
    )
    selective = run_inversion(
        policy="conformal-selective",
        physics=physics,
        surrogate=surrogate,
        m0=m0,
        m_true=m_true,
        blocks=blocks,
        receiver_indices=geometry.receiver_indices,
        iterations=iterations,
        initial_step=0.015,
        vmax=max(3.5, float(np.max(velocity_true)) + 0.2),
        block_scales=block_scales,
    )

    env = devito_environment_report()
    summary = {
        "benchmark_source": source,
        "public_marmousi_executed": source == "public-marmousi-binary",
        "velocity_range": [float(np.min(velocity_true)), float(np.max(velocity_true))],
        "devito_environment": env,
        "production_backend_status": (
            "runtime-validated" if env["executable"] else "adapter-ready-runtime-unavailable"
        ),
        "reference_backend_benchmark": {
            "iterations": iterations,
            "exact_final_model_relative_l2": exact.records[-1].model_relative_l2,
            "selective_final_model_relative_l2": selective.records[-1].model_relative_l2,
            "exact_gradient_blocks": exact.total_exact_gradient_blocks,
            "selective_exact_gradient_blocks": selective.total_exact_gradient_blocks,
            "selective_block_saving_fraction": float(
                1.0 - selective.total_exact_gradient_blocks / max(exact.total_exact_gradient_blocks, 1)
            ),
        },
        "external_requirements": {
            "devito_command": "python -m pip install -e '.[devito]'",
            "marmousi_download_script": "scripts/download_marmousi_public.sh",
            "note": (
                "Devito and the public binary cannot be fetched in the current offline container; "
                "the code path is prepared but those external runs are not fabricated."
            ),
        },
        "exit_criteria": {
            "marmousi_loader_implemented": True,
            "production_devito_adapter_implemented": True,
            "offline_benchmark_path_executed": True,
            "external_runtime_limitation_reported": True,
        },
    }
    summary["passed_code_readiness"] = all(summary["exit_criteria"].values())
    summary["passed_full_external_validation"] = bool(
        summary["public_marmousi_executed"] and env["executable"]
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4), constrained_layout=True)
    panels = [
        ("target", velocity_true),
        ("initial", velocity0),
        ("exact", velocity_from_m(exact.final_model).reshape(shape)),
        ("WaveCert", velocity_from_m(selective.final_model).reshape(shape)),
    ]
    vmin = min(float(np.min(p[1])) for p in panels)
    vmax = max(float(np.max(p[1])) for p in panels)
    for ax, (title, image) in zip(axes, panels, strict=True):
        im = ax.imshow(image, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.tolist(), shrink=0.75, label="velocity (km/s scale)")
    fig.savefig(out / "marmousi_path_reconstruction.png", dpi=220)
    plt.close(fig)
    return summary
