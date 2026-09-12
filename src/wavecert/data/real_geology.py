from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from wavecert.benchmarks.subsurfacegen import load_subsurfacegen_velocity_slice
from wavecert.data.wavefields import WavefieldDatasetArrays
from wavecert.experiments.synthetic import velocity_to_m
from wavecert.physics.helmholtz import Helmholtz2D, SurveyGeometry


@dataclass(frozen=True)
class RealGeologyBuildConfig:
    shape: tuple[int, int] = (24, 24)
    spacing: float = 0.05
    frequencies_hz: tuple[float, ...] = (2.5, 3.0, 3.5, 4.0)
    n_source_positions: int = 6
    n_receivers: int = 20
    all_source_frequency_pairs: bool = True
    examples_per_model: int = 4
    seed: int = 20260911


def _prepare_velocity(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    velocity = load_subsurfacegen_velocity_slice(path).astype(np.float64)
    if float(np.median(velocity)) > 20.0:
        velocity /= 1000.0
    zoom = (shape[0] / velocity.shape[0], shape[1] / velocity.shape[1])
    velocity = ndi.zoom(velocity, zoom, order=1)
    velocity = velocity[: shape[0], : shape[1]]
    if velocity.shape != shape:
        raise ValueError(f"downsampled velocity has shape {velocity.shape}, expected {shape}")
    return np.clip(velocity, 1.35, 5.5)


def _pair_schedule(
    geometry: SurveyGeometry,
    frequencies_hz: tuple[float, ...],
    *,
    all_pairs: bool,
    examples_per_model: int,
    rng: np.random.Generator,
) -> list[tuple[int, float]]:
    pairs = [
        (int(source), float(freq)) for source in geometry.source_indices for freq in frequencies_hz
    ]
    if all_pairs:
        return pairs
    if examples_per_model <= 0:
        raise ValueError("examples_per_model must be positive")
    take = min(examples_per_model, len(pairs))
    indices = rng.choice(len(pairs), size=take, replace=False)
    return [pairs[int(i)] for i in indices]


def build_subsurfacegen_wavecert_dataset(
    *,
    selected_root: str | Path,
    output_root: str | Path,
    manifest_path: str | Path | None = None,
    train_models: int = 800,
    validation_models: int = 200,
    id_models: int | None = None,
    ood_models: int | None = None,
    config: RealGeologyBuildConfig | None = None,
) -> dict:
    """Build exact Helmholtz operator-learning data from real SubsurfaceGen geology.

    The geological velocity slices are real SubsurfaceGen samples. Wavefields are
    regenerated with WaveCert's reference Helmholtz solver so certificate
    identities and training labels are numerically consistent.
    """

    if config is None:
        config = RealGeologyBuildConfig()

    selected_root = Path(selected_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        manifest_path = selected_root / "selected_manifest.csv"

    manifest = pd.read_csv(manifest_path)
    required = {"split", "file_path", "slice_id", "model_id", "model_type"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")

    rng = np.random.default_rng(config.seed)
    physics = Helmholtz2D(
        shape=config.shape,
        spacing=config.spacing,
        damping_width=max(3, min(config.shape) // 8),
        damping_strength=2.0,
    )
    geometry = SurveyGeometry.from_grid(
        config.shape,
        n_sources=config.n_source_positions,
        n_receivers=min(config.n_receivers, config.shape[1] - 4),
        source_depth=2,
        receiver_depth=2,
        margin=2,
    )

    train_pool = (
        manifest[manifest["split"].eq("train")]
        .drop_duplicates("slice_id")
        .sample(frac=1.0, random_state=config.seed)
        .reset_index(drop=True)
    )
    if len(train_pool) < train_models + validation_models:
        raise ValueError(
            f"need {train_models + validation_models} unique train slices, found {len(train_pool)}"
        )

    splits = {
        "train": train_pool.iloc[:train_models],
        "validation": train_pool.iloc[train_models : train_models + validation_models],
        "test_id": manifest[manifest["split"].eq("test_in_dist")].drop_duplicates("slice_id"),
        "test_ood": manifest[manifest["split"].eq("test_out_dist")].drop_duplicates("slice_id"),
    }
    if id_models is not None:
        splits["test_id"] = splits["test_id"].iloc[:id_models]
    if ood_models is not None:
        splits["test_ood"] = splits["test_ood"].iloc[:ood_models]

    summaries: dict[str, dict] = {}
    for split_name, rows in splits.items():
        models: list[np.ndarray] = []
        sources: list[int] = []
        frequencies: list[float] = []
        wavefields: list[np.ndarray] = []
        metadata: list[dict] = []

        for _, row in rows.iterrows():
            velocity = _prepare_velocity(selected_root / row["file_path"], config.shape)
            m = velocity_to_m(velocity).astype(np.float64).reshape(-1)
            schedule = _pair_schedule(
                geometry,
                config.frequencies_hz,
                all_pairs=config.all_source_frequency_pairs,
                examples_per_model=config.examples_per_model,
                rng=rng,
            )
            for source_index, frequency in schedule:
                q = physics.source(source_index)
                state = physics.solve_state(m, q, frequency).reshape(config.shape)
                wf = np.empty((*config.shape, 2), dtype=np.float32)
                wf[..., 0] = state.real.astype(np.float32)
                wf[..., 1] = state.imag.astype(np.float32)
                models.append(m.reshape(config.shape).astype(np.float32))
                sources.append(source_index)
                frequencies.append(frequency)
                wavefields.append(wf)
                metadata.append(
                    {
                        "split": split_name,
                        "slice_id": row["slice_id"],
                        "model_id": row["model_id"],
                        "model_type": row["model_type"],
                        "source_index": source_index,
                        "frequency_hz": frequency,
                    }
                )

        arrays = WavefieldDatasetArrays(
            models=np.stack(models),
            source_indices=np.asarray(sources, dtype=np.int64),
            frequencies_hz=np.asarray(frequencies, dtype=np.float32),
            wavefields=np.stack(wavefields),
            receiver_indices=np.asarray(geometry.receiver_indices, dtype=np.int64),
            shape=config.shape,
            spacing=float(config.spacing),
        )
        arrays.save(output_root / f"{split_name}.npz")
        pd.DataFrame(metadata).to_csv(output_root / f"{split_name}_metadata.csv", index=False)
        summaries[split_name] = {
            "unique_models": int(len(rows)),
            "examples": int(len(models)),
            "model_types": {
                str(k): int(v) for k, v in rows["model_type"].value_counts().to_dict().items()
            },
        }

    return {
        "shape": list(config.shape),
        "all_source_frequency_pairs": config.all_source_frequency_pairs,
        "sources": int(config.n_source_positions),
        "frequencies_hz": list(config.frequencies_hz),
        "splits": summaries,
    }
