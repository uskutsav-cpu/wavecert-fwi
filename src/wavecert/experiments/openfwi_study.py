from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

from wavecert.benchmarks.openfwi import load_openfwi_velocity_batch
from wavecert.experiments.synthetic import velocity_to_m
from wavecert.metrics import relative_l2
from wavecert.physics.helmholtz import Helmholtz2D, SurveyGeometry
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate


def _resize_velocity(v: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    zoom = (shape[0] / v.shape[0], shape[1] / v.shape[1])
    out = ndi.zoom(v, zoom, order=1)
    out = out[: shape[0], : shape[1]]
    if float(np.median(out)) > 20.0:
        out /= 1000.0
    return np.clip(out, 1.35, 5.5)


def run_openfwi_study(
    *,
    velocity_file: str | Path,
    checkpoint_path: str | Path = "checkpoints/fno_subsurfacegen_v2.pt",
    output_dir: str | Path = "results/openfwi",
    max_models: int = 100,
    frequency_hz: float = 3.0,
) -> dict:
    """Evaluate the learned operator on an official OpenFWI velocity batch.

    Only the velocity batch is required. WaveCert regenerates reference fields
    with its own Helmholtz solver, which keeps the certificate test consistent
    across datasets.
    """

    batch = load_openfwi_velocity_batch(velocity_file)
    surrogate = TrainedFNOWavefieldSurrogate.load(checkpoint_path)
    shape = surrogate.shape
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
    source = int(geometry.source_indices[len(geometry.source_indices) // 2])
    q = physics.source(source)

    wave_errors: list[float] = []
    receiver_errors: list[float] = []
    used = min(max_models, len(batch))
    for i in range(used):
        velocity = _resize_velocity(np.asarray(batch[i]).squeeze(), shape)
        m = velocity_to_m(velocity).reshape(-1)
        exact = physics.solve_state(m, q, frequency_hz)
        neural = surrogate.state(m, q, frequency_hz)
        wave_errors.append(float(relative_l2(exact, neural)))
        exact_rec = physics.restrict(exact, geometry.receiver_indices)
        neural_rec = physics.restrict(neural, geometry.receiver_indices)
        receiver_errors.append(float(relative_l2(exact_rec, neural_rec)))

    summary = {
        "velocity_file": str(velocity_file),
        "official_format_expected": "(N,1,70,70)",
        "models_evaluated": used,
        "frequency_hz": frequency_hz,
        "wavefield_relative_l2_median": float(np.median(wave_errors)),
        "wavefield_relative_l2_mean": float(np.mean(wave_errors)),
        "receiver_relative_l2_median": float(np.median(receiver_errors)),
        "receiver_relative_l2_mean": float(np.mean(receiver_errors)),
        "exit_criteria": {
            "official_velocity_batch_loaded": True,
            "reference_wavefields_generated": True,
            "surrogate_evaluated": True,
        },
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
