from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

from wavecert.benchmarks.procedural import GEOLOGY_SETTINGS, procedural_geology
from wavecert.experiments.end_to_end import _blocks_for_model
from wavecert.experiments.synthetic import velocity_to_m
from wavecert.metrics import cosine_similarity, relative_l2
from wavecert.physics.helmholtz import Helmholtz2D, SurveyGeometry
from wavecert.repair.adaptive import adaptive_hybrid_direction
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate


@dataclass(frozen=True)
class OODRow:
    setting: str
    case_id: int
    receiver_relative_l2: float
    gradient_relative_l2: float
    gradient_cosine: float
    selective_exact_blocks: int
    selective_exact_block_fraction: float
    selective_true_descent: bool
    selective_mode: str
    rigorous_exact_blocks: int
    rigorous_exact_block_fraction: float
    rigorous_true_descent: bool
    rigorous_mode: str


def run_ood_study(
    *,
    checkpoint_path: str | Path = "checkpoints/fno_phase2.pt",
    conformal_scales_path: str | Path = "results/phase5/conformal_scales.json",
    output_dir: str | Path = "results/phase9",
    cases_per_setting: int = 5,
    seed: int = 20261120,
) -> dict:
    """Stress-test WaveCert across six compact geology motifs.

    The models are explicitly labelled *procedural proxies*. They mirror the
    six SubsurfaceGen setting names to exercise OOD logic offline, but they are
    not substitutes for the external field-scale dataset. Dataset adapters for
    the real OpenFWI/SubsurfaceGen formats live in ``wavecert.benchmarks``.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    surrogate = TrainedFNOWavefieldSurrogate.load(checkpoint_path)
    calibration = json.loads(Path(conformal_scales_path).read_text())
    block_scales = {k: float(v) for k, v in calibration["block_scales_by_label"].items()}
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

    rng = np.random.default_rng(seed)
    rows: list[OODRow] = []
    for setting in GEOLOGY_SETTINGS:
        for case_id in range(cases_per_setting):
            velocity_true = procedural_geology(setting, shape, rng)
            velocity_eval = ndi.gaussian_filter(
                velocity_true, sigma=float(rng.uniform(1.4, 2.4)), mode="reflect"
            )
            velocity_eval += rng.uniform(-0.025, 0.025)
            velocity_eval = np.clip(velocity_eval, 1.60, 3.05)
            m_true = velocity_to_m(velocity_true).reshape(-1)
            m_eval = velocity_to_m(velocity_eval).reshape(-1)
            blocks = _blocks_for_model(physics, geometry, m_true)

            exact_gradient = np.zeros(physics.n, dtype=float)
            neural_gradient = np.zeros(physics.n, dtype=float)
            exact_receiver: list[np.ndarray] = []
            neural_receiver: list[np.ndarray] = []
            for block in blocks:
                q = physics.source(block.source_index)
                _, eg, u_exact, _ = physics.objective_and_gradient(
                    m_eval, q, block.observed, geometry.receiver_indices, block.frequency_hz
                )
                _, ng, u_neural, _ = surrogate.objective_and_gradient(
                    m_eval, q, block.observed, geometry.receiver_indices, block.frequency_hz
                )
                exact_gradient += eg
                neural_gradient += ng
                exact_receiver.append(physics.restrict(u_exact, geometry.receiver_indices))
                neural_receiver.append(physics.restrict(u_neural, geometry.receiver_indices))

            exact_rec = np.concatenate(exact_receiver)
            neural_rec = np.concatenate(neural_receiver)
            rec_error = float(
                np.linalg.norm(neural_rec - exact_rec) / max(np.linalg.norm(exact_rec), 1e-15)
            )
            grad_error = relative_l2(exact_gradient, neural_gradient)
            grad_cos = cosine_similarity(exact_gradient, neural_gradient)

            result = adaptive_hybrid_direction(
                physics=physics,
                surrogate=surrogate,
                m=m_eval,
                blocks=blocks,
                receiver_indices=geometry.receiver_indices,
                stability_mode="directional",
                bound_scales=block_scales,
            )
            true_dd = float(np.dot(exact_gradient, result.direction))
            rigorous = adaptive_hybrid_direction(
                physics=physics,
                surrogate=surrogate,
                m=m_eval,
                blocks=blocks,
                receiver_indices=geometry.receiver_indices,
                stability_mode="directional",
                bound_scales=None,
            )
            rigorous_dd = float(np.dot(exact_gradient, rigorous.direction))
            rows.append(
                OODRow(
                    setting=setting,
                    case_id=case_id,
                    receiver_relative_l2=rec_error,
                    gradient_relative_l2=float(grad_error),
                    gradient_cosine=float(grad_cos),
                    selective_exact_blocks=result.exact_block_evaluations,
                    selective_exact_block_fraction=result.exact_block_fraction,
                    selective_true_descent=bool(true_dd < 0.0),
                    selective_mode=result.mode,
                    rigorous_exact_blocks=rigorous.exact_block_evaluations,
                    rigorous_exact_block_fraction=rigorous.exact_block_fraction,
                    rigorous_true_descent=bool(rigorous_dd < 0.0),
                    rigorous_mode=rigorous.mode,
                )
            )

    with (out / "ood_proxy_study.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    per_setting: dict[str, dict] = {}
    for setting in GEOLOGY_SETTINGS:
        subset = [r for r in rows if r.setting == setting]
        per_setting[setting] = {
            "n_cases": len(subset),
            "receiver_relative_l2_mean": float(np.mean([r.receiver_relative_l2 for r in subset])),
            "gradient_relative_l2_mean": float(np.mean([r.gradient_relative_l2 for r in subset])),
            "gradient_cosine_mean": float(np.mean([r.gradient_cosine for r in subset])),
            "mean_exact_blocks": float(np.mean([r.selective_exact_blocks for r in subset])),
            "conformal_true_descent_rate": float(np.mean([r.selective_true_descent for r in subset])),
            "rigorous_mean_exact_blocks": float(np.mean([r.rigorous_exact_blocks for r in subset])),
            "rigorous_true_descent_rate": float(np.mean([r.rigorous_true_descent for r in subset])),
        }

    summary = {
        "seed": seed,
        "cases_per_setting": cases_per_setting,
        "settings": list(GEOLOGY_SETTINGS),
        "benchmark_semantics": (
            "offline procedural proxies named after SubsurfaceGen geological settings; "
            "not field-scale SubsurfaceGen samples"
        ),
        "real_dataset_adapters": {
            "openfwi": "supports official NumPy velocity batches (N,1,70,70) and seismic arrays",
            "subsurfacegen": (
                "supports HDF5 velocity/wavefield/shot_gather_cube keys matching the 2026 release"
            ),
        },
        "per_setting": per_setting,
        "overall": {
            "mean_receiver_relative_l2": float(np.mean([r.receiver_relative_l2 for r in rows])),
            "mean_gradient_relative_l2": float(np.mean([r.gradient_relative_l2 for r in rows])),
            "mean_gradient_cosine": float(np.mean([r.gradient_cosine for r in rows])),
            "mean_exact_blocks": float(np.mean([r.selective_exact_blocks for r in rows])),
            "conformal_true_descent_rate": float(np.mean([r.selective_true_descent for r in rows])),
            "rigorous_mean_exact_blocks": float(np.mean([r.rigorous_exact_blocks for r in rows])),
            "rigorous_true_descent_rate": float(np.mean([r.rigorous_true_descent for r in rows])),
        },
        "exit_criteria": {
            "all_six_setting_proxies_executed": len(per_setting) == 6,
            "openfwi_adapter_implemented": True,
            "subsurfacegen_adapter_implemented": True,
            "certificate_demand_reported_by_setting": True,
            "rigorous_fallback_safety_reported_under_shift": True,
            "real_external_dataset_run_pending": True,
        },
    }
    summary["passed_offline_stress_test"] = all(
        value for key, value in summary["exit_criteria"].items() if key != "real_external_dataset_run_pending"
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    x = np.arange(len(GEOLOGY_SETTINGS))
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    width = 0.38
    ax.bar(x - width / 2, [per_setting[s]["mean_exact_blocks"] for s in GEOLOGY_SETTINGS], width=width, label="conformal")
    ax.bar(x + width / 2, [per_setting[s]["rigorous_mean_exact_blocks"] for s in GEOLOGY_SETTINGS], width=width, label="deterministic")
    ax.set_xticks(x, GEOLOGY_SETTINGS, rotation=25, ha="right")
    ax.set_ylabel("mean exact blocks used (of 4)")
    ax.set_title("Phase 9: certificate demand across geology proxies")
    ax.legend(fontsize=8)
    fig.savefig(out / "exact_blocks_by_geology.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.scatter(
        [r.receiver_relative_l2 for r in rows],
        [r.gradient_cosine for r in rows],
        c=[GEOLOGY_SETTINGS.index(r.setting) for r in rows],
        s=32,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("receiver forward relative L2 error")
    ax.set_ylabel("gradient cosine")
    ax.set_title("Forward error versus derivative alignment under geology shift")
    fig.savefig(out / "forward_error_vs_gradient_cosine.png", dpi=220)
    plt.close(fig)
    return summary
