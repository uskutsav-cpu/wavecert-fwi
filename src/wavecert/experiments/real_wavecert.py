from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

from wavecert.certificates.directional import certify_direction
from wavecert.data.wavefields import WavefieldDatasetArrays
from wavecert.metrics import cosine_similarity, relative_l2
from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate


@dataclass(frozen=True)
class RealWaveCertRow:
    split: str
    case_id: int
    source_index: int
    frequency_hz: float
    forward_receiver_relative_l2: float
    gradient_relative_l2: float
    gradient_cosine: float
    exact_directional_derivative: float
    surrogate_directional_derivative: float
    deterministic_bound: float
    realized_directional_error: float
    conformal_bound: float
    deterministic_covers: bool
    conformal_covers: bool
    neural_true_descent: bool
    deterministic_certified: bool
    conformal_certified: bool
    conformal_false_certification: bool


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(scores, dtype=float))
    if values.size == 0:
        raise ValueError("no calibration scores")
    rank = int(np.ceil((values.size + 1) * (1.0 - alpha))) - 1
    rank = min(max(rank, 0), values.size - 1)
    return float(values[rank])


def _velocity_from_m(m: np.ndarray) -> np.ndarray:
    return 1.0 / np.sqrt(np.asarray(m, dtype=float))


def _evaluate_case(
    arrays: WavefieldDatasetArrays,
    index: int,
    *,
    physics: Helmholtz2D,
    surrogate: TrainedFNOWavefieldSurrogate,
    conformal_scale: float,
    split: str,
) -> RealWaveCertRow:
    m_true = arrays.models[index].astype(np.float64).reshape(-1)
    shape = arrays.shape
    velocity = _velocity_from_m(m_true).reshape(shape)
    velocity_eval = ndi.gaussian_filter(velocity, sigma=1.6, mode="reflect")
    m_eval = (1.0 / np.maximum(velocity_eval, 1e-8) ** 2).reshape(-1)
    source_index = int(arrays.source_indices[index])
    frequency = float(arrays.frequencies_hz[index])
    q = physics.source(source_index)
    stored = arrays.wavefields[index]
    state_true = stored[..., 0] + 1j * stored[..., 1]
    observed = physics.restrict(state_true.reshape(-1), tuple(arrays.receiver_indices.tolist()))

    _, exact_gradient, exact_state, _ = physics.objective_and_gradient(
        m_eval, q, observed, tuple(arrays.receiver_indices.tolist()), frequency
    )
    _, neural_gradient, neural_state, _ = surrogate.objective_and_gradient(
        m_eval, q, observed, tuple(arrays.receiver_indices.tolist()), frequency
    )
    norm = float(np.linalg.norm(neural_gradient))
    if norm <= 1e-14:
        direction = np.zeros_like(neural_gradient)
    else:
        direction = -np.asarray(neural_gradient, dtype=float) / norm

    cert = certify_direction(
        physics=physics,
        surrogate=surrogate,
        m=m_eval,
        q=q,
        observed=observed,
        receiver_indices=tuple(arrays.receiver_indices.tolist()),
        frequency_hz=frequency,
        direction=direction,
        stability_mode="directional",
        validate_exact=True,
    )

    exact_rec = physics.restrict(exact_state, tuple(arrays.receiver_indices.tolist()))
    neural_rec = physics.restrict(neural_state, tuple(arrays.receiver_indices.tolist()))
    forward_error = float(
        np.linalg.norm(neural_rec - exact_rec) / max(np.linalg.norm(exact_rec), 1e-15)
    )
    deterministic_bound = float(cert.directional_error_bound)
    conformal_bound = conformal_scale * deterministic_bound
    exact_dd = float(cert.exact_directional_derivative)
    surrogate_dd = float(cert.surrogate_directional_derivative)
    realized = float(abs(exact_dd - surrogate_dd))
    conformal_certified = bool(surrogate_dd + conformal_bound < 0.0)
    return RealWaveCertRow(
        split=split,
        case_id=index,
        source_index=source_index,
        frequency_hz=frequency,
        forward_receiver_relative_l2=forward_error,
        gradient_relative_l2=float(relative_l2(exact_gradient, neural_gradient)),
        gradient_cosine=float(cosine_similarity(exact_gradient, neural_gradient)),
        exact_directional_derivative=exact_dd,
        surrogate_directional_derivative=surrogate_dd,
        deterministic_bound=deterministic_bound,
        realized_directional_error=realized,
        conformal_bound=conformal_bound,
        deterministic_covers=realized <= deterministic_bound + 1e-12,
        conformal_covers=realized <= conformal_bound + 1e-12,
        neural_true_descent=exact_dd < 0.0,
        deterministic_certified=bool(cert.certified_descent),
        conformal_certified=conformal_certified,
        conformal_false_certification=bool(conformal_certified and exact_dd >= 0.0),
    )


def run_real_wavecert_study(
    *,
    dataset_dir: str | Path = "data/external/subsurfacegen-wavecert",
    checkpoint_path: str | Path = "checkpoints/fno_subsurfacegen_v2.pt",
    output_dir: str | Path = "results/real_wavecert",
    alpha: float = 0.10,
    calibration_cases: int = 120,
    evaluation_cases: int = 100,
) -> dict:
    """Calibrate on real-geology validation data, evaluate untouched ID and OOD data."""

    dataset_dir = Path(dataset_dir)
    validation = WavefieldDatasetArrays.load(dataset_dir / "validation.npz")
    test_id = WavefieldDatasetArrays.load(dataset_dir / "test_id.npz")
    test_ood = WavefieldDatasetArrays.load(dataset_dir / "test_ood.npz")
    surrogate = TrainedFNOWavefieldSurrogate.load(checkpoint_path)
    physics = Helmholtz2D(
        shape=surrogate.shape,
        spacing=validation.spacing,
        damping_width=max(3, min(surrogate.shape) // 8),
        damping_strength=2.0,
    )

    # Calibration pass with scale=1 solely to compute score ratios.
    calibration_rows = [
        _evaluate_case(
            validation,
            i,
            physics=physics,
            surrogate=surrogate,
            conformal_scale=1.0,
            split="calibration",
        )
        for i in range(min(calibration_cases, len(validation.models)))
    ]
    scores = np.asarray(
        [r.realized_directional_error / max(r.deterministic_bound, 1e-15) for r in calibration_rows]
    )
    scale = _conformal_quantile(scores, alpha)

    id_rows = [
        _evaluate_case(
            test_id,
            i,
            physics=physics,
            surrogate=surrogate,
            conformal_scale=scale,
            split="id",
        )
        for i in range(min(evaluation_cases, len(test_id.models)))
    ]
    ood_rows = [
        _evaluate_case(
            test_ood,
            i,
            physics=physics,
            surrogate=surrogate,
            conformal_scale=scale,
            split="ood",
        )
        for i in range(min(evaluation_cases, len(test_ood.models)))
    ]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cases.csv").open("w", newline="") as f:
        rows = [*id_rows, *ood_rows]
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    def aggregate(rows: list[RealWaveCertRow]) -> dict:
        return {
            "n_cases": len(rows),
            "receiver_forward_relative_l2_median": float(
                np.median([r.forward_receiver_relative_l2 for r in rows])
            ),
            "gradient_relative_l2_median": float(np.median([r.gradient_relative_l2 for r in rows])),
            "gradient_cosine_median": float(np.median([r.gradient_cosine for r in rows])),
            "neural_true_descent_rate": float(np.mean([r.neural_true_descent for r in rows])),
            "deterministic_coverage": float(np.mean([r.deterministic_covers for r in rows])),
            "conformal_coverage": float(np.mean([r.conformal_covers for r in rows])),
            "deterministic_certified_fraction": float(
                np.mean([r.deterministic_certified for r in rows])
            ),
            "conformal_certified_fraction": float(np.mean([r.conformal_certified for r in rows])),
            "conformal_false_certifications": int(
                np.sum([r.conformal_false_certification for r in rows])
            ),
        }

    summary = {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "calibration_cases": len(calibration_rows),
        "conformal_scale": scale,
        "id": aggregate(id_rows),
        "ood": aggregate(ood_rows),
        "coverage_semantics": (
            "split-conformal marginal coverage is expected only under exchangeability; "
            "OOD coverage is reported diagnostically and is not guaranteed"
        ),
        "exit_criteria": {
            "deterministic_id_full_coverage": all(r.deterministic_covers for r in id_rows),
            "deterministic_ood_full_coverage": all(r.deterministic_covers for r in ood_rows),
            "id_ood_reported_separately": True,
            "false_certifications_reported": True,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(6.8, 4.5), constrained_layout=True)
    ax.scatter(
        [r.forward_receiver_relative_l2 for r in id_rows],
        [r.gradient_cosine for r in id_rows],
        label="ID",
        alpha=0.7,
    )
    ax.scatter(
        [r.forward_receiver_relative_l2 for r in ood_rows],
        [r.gradient_cosine for r in ood_rows],
        label="OOD",
        alpha=0.7,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("receiver forward relative L2")
    ax.set_ylabel("gradient cosine")
    ax.set_title("Real geology: forward error vs derivative alignment")
    ax.legend()
    fig.savefig(out / "forward_error_vs_gradient_cosine.png", dpi=220)
    plt.close(fig)

    return summary
