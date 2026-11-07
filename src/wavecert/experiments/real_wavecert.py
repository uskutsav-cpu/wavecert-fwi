from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

from wavecert.certificates.directional import certify_direction
from wavecert.data.splits import (
    audit_dataset_roles,
    verify_calibration_checkpoint,
    verify_dataset_provenance,
)
from wavecert.data.wavefields import WavefieldDatasetArrays
from wavecert.metrics import cosine_similarity, relative_l2
from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.statistics import (
    binomial_failure_summary,
    cluster_bootstrap_mean,
    error_ratios,
    scale_error_bounds,
    split_conformal_quantile,
)
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
    slice_id: str = ""
    model_id: str = ""


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Backward-compatible alias for the correct shared finite-sample quantile."""
    return split_conformal_quantile(scores, alpha)


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
    slice_id: str = "",
    model_id: str = "",
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
    conformal_bound = float(scale_error_bounds(np.asarray(deterministic_bound), conformal_scale))
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
        slice_id=slice_id,
        model_id=model_id,
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
    """Calibrate parent-level maxima on an independent, dedicated calibration role.

    ``calibration_cases`` and ``evaluation_cases`` limit complete slices, keeping
    all their source/frequency blocks together. A parent-level maximum is the
    conformal score; correlated slices/blocks never inflate calibration ``n``.
    OOD results remain descriptive. A new run requires all five role metadata
    files and parent isolation; legacy saved result files are left unchanged.
    """

    dataset_dir = Path(dataset_dir)
    if calibration_cases < 1 or evaluation_cases < 1:
        raise ValueError("calibration_cases and evaluation_cases must be positive")
    split_audit = audit_dataset_roles(dataset_dir)
    dataset_provenance = verify_dataset_provenance(dataset_dir)
    checkpoint_provenance = verify_calibration_checkpoint(checkpoint_path, dataset_provenance)
    calibration = WavefieldDatasetArrays.load(dataset_dir / "calibration.npz")
    test_id = WavefieldDatasetArrays.load(dataset_dir / "test_id.npz")
    test_ood = WavefieldDatasetArrays.load(dataset_dir / "test_ood.npz")
    surrogate = TrainedFNOWavefieldSurrogate.load(checkpoint_path)
    physics = Helmholtz2D(
        shape=surrogate.shape,
        spacing=calibration.spacing,
        damping_width=max(3, min(surrogate.shape) // 8),
        damping_strength=2.0,
    )

    def evaluate_role(arrays, role, limit, scale):
        with (dataset_dir / f"{role}_metadata.csv").open(newline="") as stream:
            metadata = list(csv.DictReader(stream))
        if len(metadata) != len(arrays.models):
            raise ValueError(f"{role}: metadata and wavefield example counts differ")
        selected = sorted({row["slice_id"] for row in metadata})[:limit]
        selected_set = set(selected)
        return [
            _evaluate_case(
                arrays, i, physics=physics, surrogate=surrogate, conformal_scale=scale,
                split=role, slice_id=row["slice_id"], model_id=row["model_id"],
            )
            for i, row in enumerate(metadata) if row["slice_id"] in selected_set
        ]

    # Exact validation labels are used only on the independent calibration role.
    calibration_rows = evaluate_role(calibration, "calibration", calibration_cases, 1.0)
    scores_by_parent = defaultdict(list)
    for row in calibration_rows:
        ratio = float(error_ratios(
            np.asarray(row.realized_directional_error), np.asarray(row.deterministic_bound)
        ))
        scores_by_parent[row.model_id].append(ratio)
    scores = np.asarray([max(values) for _, values in sorted(scores_by_parent.items())])
    scale = _conformal_quantile(scores, alpha)
    if not np.isfinite(scale):
        raise ValueError(
            "requested coverage has no finite bound at the available number of independent "
            "calibration parents; increase calibration parents or predeclare a larger alpha"
        )
    id_rows = evaluate_role(test_id, "test_id", evaluation_cases, scale)
    ood_rows = evaluate_role(test_ood, "test_ood", evaluation_cases, scale)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cases.csv").open("w", newline="") as f:
        rows = [*id_rows, *ood_rows]
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    def aggregate(rows: list[RealWaveCertRow]) -> dict:
        by_parent = defaultdict(list)
        for row in rows:
            by_parent[row.model_id].append(row)
        parent_failures = [any(row.conformal_false_certification for row in group) for group in by_parent.values()]
        parent_covered = [all(row.conformal_covers for row in group) for group in by_parent.values()]
        details = {}
        for parent, group in sorted(by_parent.items()):
            max_ratio = max(float(error_ratios(
                np.asarray(row.realized_directional_error), np.asarray(row.deterministic_bound)
            )) for row in group)
            details[parent] = {
                "n_slices": len({row.slice_id for row in group}),
                "n_blocks": len(group),
                "max_normalized_error": max_ratio if np.isfinite(max_ratio) else None,
                "zero_bound_positive_error": bool(np.isposinf(max_ratio)),
                "simultaneously_covered": all(row.conformal_covers for row in group),
            }
        return {
            "n_cases": len(rows),
            "observation_unit": "one slice/source/frequency block",
            "n_slices": len({row.slice_id for row in rows}),
            "n_parent_models": len(by_parent),
            "receiver_forward_relative_l2_median": float(
                np.median([r.forward_receiver_relative_l2 for r in rows])
            ),
            "gradient_relative_l2_median": float(np.median([r.gradient_relative_l2 for r in rows])),
            "gradient_cosine_median": float(np.median([r.gradient_cosine for r in rows])),
            "neural_true_descent_rate": float(np.mean([r.neural_true_descent for r in rows])),
            "deterministic_coverage": float(np.mean([r.deterministic_covers for r in rows])),
            "conformal_coverage": float(np.mean([r.conformal_covers for r in rows])),
            "parent_simultaneous_conformal_coverage": float(np.mean(parent_covered)),
            "parent_coverage_failure_uncertainty": binomial_failure_summary(np.asarray([not value for value in parent_covered])),
            "parent_selected_counts_and_scores": details,
            "deterministic_certified_fraction": float(
                np.mean([r.deterministic_certified for r in rows])
            ),
            "conformal_certified_fraction": float(np.mean([r.conformal_certified for r in rows])),
            "conformal_false_certifications": int(
                np.sum([r.conformal_false_certification for r in rows])
            ),
            "parent_any_false_acceptance": binomial_failure_summary(np.asarray(parent_failures)),
            "parent_bootstrap_gradient_cosine": cluster_bootstrap_mean(
                np.asarray([row.gradient_cosine for row in rows]),
                np.asarray([row.model_id for row in rows]),
            ),
        }

    def sampling_shapes(rows):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row.model_id].append(row)
        return {
            (tuple(sorted(Counter(row.slice_id for row in group).values())),
             tuple(sorted({(row.source_index, row.frequency_hz) for row in group})))
            for group in grouped.values()
        }

    sampling_matches = sampling_shapes(calibration_rows) == sampling_shapes(id_rows)
    summary = {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "calibration_cases": len(calibration_rows),
        "calibration_unit": "maximum normalized error across selected slices and blocks of one parent model",
        "calibration_parent_models": len(scores),
        "calibration_score_by_parent": {parent: max(values) for parent, values in sorted(scores_by_parent.items())},
        "calibration_selected_counts_by_parent": {
            parent: {
                "n_slices": len({row.slice_id for row in calibration_rows if row.model_id == parent}),
                "n_blocks": sum(row.model_id == parent for row in calibration_rows),
            }
            for parent in sorted(scores_by_parent)
        },
        "conformal_scale": scale,
        "finite_calibration_bound_supported": bool(np.isfinite(scale)),
        "calibration_id_within_parent_sampling_shapes_match": sampling_matches,
        "sampling_scope": (
            "Observed slice counts and source/frequency block sets match across calibration and ID parents; parent exchangeability still requires the declared sampling design."
            if sampling_matches else
            "Descriptive evaluation only: selected within-parent slice/block counts differ between calibration and ID. Nominal pooled parent coverage is not established."
        ),
        "split_audit": split_audit,
        "dataset_provenance": dataset_provenance,
        "checkpoint_provenance": checkpoint_provenance,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "calibration_metadata_sha256": hashlib.sha256((dataset_dir / "calibration_metadata.csv").read_bytes()).hexdigest(),
        "id": aggregate(id_rows),
        "ood": aggregate(ood_rows),
        "coverage_semantics": (
            "parent-level split-conformal coverage requires exchangeable independent parents "
            "and the same predeclared within-parent sampling protocol; coverage is not "
            "conditional on acceptance, not an entire inversion-trajectory guarantee, and "
            "does not apply to OOD; interval assumptions are stated alongside each summary"
        ),
        "exit_criteria": {
            "deterministic_id_full_coverage": all(r.deterministic_covers for r in id_rows),
            "deterministic_ood_full_coverage": all(r.deterministic_covers for r in ood_rows),
            "id_ood_reported_separately": True,
            "false_certifications_reported": True,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

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
