from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from wavecert.benchmarks.subsurfacegen import load_subsurfacegen_velocity_slice
from wavecert.data.splits import (
    audit_split_manifest,
    grouped_split_manifest,
    verify_dataset_provenance,
)
from wavecert.data.wavefields import WavefieldDatasetArrays
from wavecert.experiments.synthetic import velocity_to_m
from wavecert.physics.helmholtz import Helmholtz2D, SurveyGeometry
from wavecert.provenance import sha256_file


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
    split_mode: str = "grouped",
    calibration_models: int | None = None,
) -> dict:
    """Build exact Helmholtz operator-learning data from real SubsurfaceGen geology.

    The geological velocity slices are real SubsurfaceGen samples. Wavefields are
    regenerated with WaveCert's reference Helmholtz solver so certificate
    identities and training labels are numerically consistent.

    New builds use all five parent-isolated roles. If an explicit ``role``
    manifest is not supplied, a deterministic parent-level research re-split
    is made, preserving original benchmark labels in ``source_split``. This
    is not the official benchmark split. Historical count parameters cap
    selected *slices*, not parent models, within preassigned roles.

    ``split_mode="legacy-uncalibrated"`` reproduces the former slice split
    only for descriptive, uncalibrated workflows; it creates no calibration
    role and cannot be consumed by new calibrated experiments.

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

    if split_mode not in {"grouped", "legacy-uncalibrated"}:
        raise ValueError("split_mode must be grouped or legacy-uncalibrated")
    if split_mode == "grouped":
        if "role" in manifest.columns:
            assigned = manifest.copy()
            assigned["source_split"] = assigned.get("source_split", assigned["split"])
        else:
            assigned = pd.DataFrame(grouped_split_manifest(
                manifest.to_dict("records"), seed=config.seed,
            ))
        roles = ("train", "validation", "calibration", "test_id", "test_ood")
        if set(assigned["role"]) != set(roles):
            raise ValueError("explicit grouped manifest must contain exactly train, validation, calibration, test_id, test_ood roles")
        split_audit = audit_split_manifest(assigned.to_dict("records"))
        if not split_audit["parent_isolation_passed"]:
            raise ValueError("grouped manifest shares slices or parent models between roles")
        splits = {
            role: assigned[assigned["role"].eq(role)].drop_duplicates("slice_id").sort_values("slice_id")
            for role in roles
        }
        limits = {"train": train_models, "validation": validation_models, "calibration": calibration_models, "test_id": id_models, "test_ood": ood_models}
        for role, limit in limits.items():
            if limit is not None:
                if isinstance(limit, bool) or int(limit) != limit or limit < 1:
                    raise ValueError(f"{role} slice limit must be a positive integer")
                splits[role] = splits[role].iloc[:int(limit)]
        protocol = "new parent-isolated research split; source_split retains original benchmark labels"
    else:
        train_pool = (
            manifest[manifest["split"].eq("train")]
            .drop_duplicates("slice_id")
            .sample(frac=1.0, random_state=config.seed)
            .reset_index(drop=True)
        )
        if len(train_pool) < train_models + validation_models:
            raise ValueError(f"need {train_models + validation_models} train slices, found {len(train_pool)}")
        splits = {
            "train": train_pool.iloc[:train_models],
            "validation": train_pool.iloc[train_models:train_models + validation_models],
            "test_id": manifest[manifest["split"].eq("test_in_dist")].drop_duplicates("slice_id"),
            "test_ood": manifest[manifest["split"].eq("test_out_dist")].drop_duplicates("slice_id"),
        }
        if id_models is not None:
            splits["test_id"] = splits["test_id"].iloc[:id_models]
        if ood_models is not None:
            splits["test_ood"] = splits["test_ood"].iloc[:ood_models]
        protocol = "legacy slice split: descriptive only, no independent calibration role"
    if any(rows.empty for rows in splits.values()):
        raise ValueError("every requested dataset role must contain at least one slice")
    selected_manifest = pd.concat([
        rows.assign(role=role, split=role, source_split=rows.get("source_split", rows["split"]))
        for role, rows in splits.items()
    ], ignore_index=True)
    split_audit = audit_split_manifest(selected_manifest.to_dict("records"))
    selected_manifest.to_csv(output_root / "split_manifest.csv", index=False)

    def sha256_file(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    sources_provenance = {}
    for filename in sorted(set(selected_manifest["file_path"])):
        path = selected_root / filename
        sources_provenance[str(filename)] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}

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
                        "source_split": row.get("source_split", row["split"]),
                        "source_file": row["file_path"],
                        "source_sha256": sources_provenance[str(row["file_path"])]["sha256"],
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
            "unique_slices": int(len(rows)),
            "unique_parent_models": int(rows["model_id"].nunique()),
            "arrays_sha256": sha256_file(output_root / f"{split_name}.npz"),
            "metadata_sha256": sha256_file(output_root / f"{split_name}_metadata.csv"),
            "examples": int(len(models)),
            "model_types": {
                str(k): int(v) for k, v in rows["model_type"].value_counts().to_dict().items()
            },
        }

    summary = {
        "schema_version": 1,
        "protocol": protocol,
        "conformal_ready": split_mode == "grouped" and split_audit["parent_isolation_passed"],
        "input_manifest_sha256": sha256_file(manifest_path),
        "role_manifest_sha256": sha256_file(output_root / "split_manifest.csv"),
        "selected_root": str(selected_root),
        "config": asdict(config),
        "preprocessing": {
            "velocity_units": "m/s converted to km/s when median >20",
            "resize": "scipy.ndimage.zoom order=1 to configured shape",
            "velocity_clip_km_s": [1.35, 5.5],
            "wavefields": "regenerated synthetic fields from WaveCert discrete Helmholtz solver; not measured field data",
        },
        "shape": list(config.shape),
        "all_source_frequency_pairs": config.all_source_frequency_pairs,
        "sources": int(config.n_source_positions),
        "frequencies_hz": list(config.frequencies_hz),
        "split_audit": split_audit,
        "raw_source_files": sources_provenance,
        "splits": summaries,
    }
    (output_root / "dataset_manifest.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def reassign_dataset_roles(
    *,
    dataset_dir: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Reassign cached wavefields to new whole-parent roles without recomputation.

    The exact same slices, parent IDs, source files, and block data must be
    present. The destination must be new: it is built in a sibling temporary
    directory and published by directory rename only after all hashes verify.
    Source arrays and their original provenance remain unchanged. This utility
    changes an experimental protocol; it cannot turn previously inspected
    outcomes into an untouched confirmatory test.
    """

    source = Path(dataset_dir).resolve()
    output = Path(output_dir).resolve()
    manifest_path = Path(manifest_path)
    if source == output or output.exists():
        raise ValueError("role reassignment requires a new destination directory")
    source_identity = verify_dataset_provenance(source)
    source_manifest = json.loads((source / "dataset_manifest.json").read_text())
    assigned = pd.read_csv(manifest_path)
    roles = ("train", "validation", "calibration", "test_id", "test_ood")
    required = {"role", "split", "slice_id", "model_id", "model_type", "file_path"}
    if not required.issubset(assigned.columns) or set(assigned["role"]) != set(roles):
        raise ValueError("replacement manifest must contain the five roles and slice/source identity fields")
    if assigned["slice_id"].duplicated().any():
        raise ValueError("replacement manifest must contain exactly one row per slice")
    split_audit = audit_split_manifest(assigned.to_dict("records"))
    if not split_audit["parent_isolation_passed"]:
        raise ValueError("replacement manifest fails whole-parent isolation")
    arrays_by_role = {role: WavefieldDatasetArrays.load(source / f"{role}.npz") for role in roles}
    reference = arrays_by_role[roles[0]]
    if any(
        arrays.shape != reference.shape or arrays.spacing != reference.spacing
        or not np.array_equal(arrays.receiver_indices, reference.receiver_indices)
        for arrays in arrays_by_role.values()
    ):
        raise ValueError("cached roles disagree on the physics grid or receiver geometry")
    metadata = []
    for role, arrays in arrays_by_role.items():
        frame = pd.read_csv(source / f"{role}_metadata.csv")
        if len(frame) != len(arrays.models):
            raise ValueError(f"{role}: cached metadata/example count mismatch")
        if not np.array_equal(frame["source_index"].to_numpy(), arrays.source_indices):
            raise ValueError(f"{role}: cached source metadata disagrees with array values")
        if not np.allclose(frame["frequency_hz"].to_numpy(), arrays.frequencies_hz, rtol=0, atol=1e-7):
            raise ValueError(f"{role}: cached frequency metadata disagrees with array values")
        metadata.extend(frame.to_dict("records"))
    pooled = WavefieldDatasetArrays(
        models=np.concatenate([arrays.models for arrays in arrays_by_role.values()]),
        source_indices=np.concatenate([arrays.source_indices for arrays in arrays_by_role.values()]),
        frequencies_hz=np.concatenate([arrays.frequencies_hz for arrays in arrays_by_role.values()]),
        wavefields=np.concatenate([arrays.wavefields for arrays in arrays_by_role.values()]),
        receiver_indices=reference.receiver_indices, shape=reference.shape, spacing=reference.spacing,
    )
    meta = pd.DataFrame(metadata)
    if set(meta["slice_id"]) != set(assigned["slice_id"]):
        raise ValueError("replacement manifest must retain exactly the cached slices")
    new_for_slice = assigned.set_index("slice_id").to_dict("index")
    for row in metadata:
        replacement = new_for_slice[row["slice_id"]]
        if str(replacement["model_id"]) != str(row["model_id"]) or str(replacement["model_type"]) != str(row["model_type"]):
            raise ValueError("replacement manifest changes a slice's parent or geology")
        if str(replacement["file_path"]) != str(row["source_file"]):
            raise ValueError("replacement manifest changes a cached source file")
        recorded_source = source_manifest.get("raw_source_files", {}).get(str(row["source_file"]), {})
        if row.get("source_sha256") != recorded_source.get("sha256"):
            raise ValueError("cached metadata disagrees with the recorded raw-source identity")
        if (replacement["role"] == "test_ood") != (row["split"] == "test_ood"):
            raise ValueError("replacement manifest must preserve source ID/OOD membership")
        row["split"] = replacement["role"]
        row["source_split"] = replacement.get("source_split", replacement["split"])
    meta = pd.DataFrame(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-building-", dir=output.parent))
    try:
        assigned.sort_values(["role", "slice_id"]).to_csv(temporary / "split_manifest.csv", index=False)
        summaries = {}
        for role in roles:
            indices = np.flatnonzero(meta["split"].eq(role).to_numpy())
            data = WavefieldDatasetArrays(
                models=pooled.models[indices], source_indices=pooled.source_indices[indices],
                frequencies_hz=pooled.frequencies_hz[indices], wavefields=pooled.wavefields[indices],
                receiver_indices=pooled.receiver_indices, shape=pooled.shape, spacing=pooled.spacing,
            )
            data.save(temporary / f"{role}.npz")
            role_meta = meta.iloc[indices]
            role_meta.to_csv(temporary / f"{role}_metadata.csv", index=False)
            slices = role_meta.drop_duplicates("slice_id")
            summaries[role] = {
                "unique_models": int(slices["slice_id"].nunique()),
                "unique_slices": int(slices["slice_id"].nunique()),
                "unique_parent_models": int(slices["model_id"].nunique()),
                "examples": int(len(indices)),
                "model_types": {str(k): int(v) for k, v in slices["model_type"].value_counts().items()},
                "arrays_sha256": sha256_file(temporary / f"{role}.npz"),
                "metadata_sha256": sha256_file(temporary / f"{role}_metadata.csv"),
            }
        summary = {
            **source_manifest,
            "protocol": "development whole-parent role reassignment from cached fields; inspect input selection provenance for sampling design",
            "input_manifest_sha256": sha256_file(manifest_path),
            "role_manifest_sha256": sha256_file(temporary / "split_manifest.csv"),
            "split_audit": split_audit,
            "splits": summaries,
            "reassignment": {
                "source_dataset_manifest_sha256": source_identity["dataset_manifest_sha256"],
                "source_role_manifest_sha256": source_identity["role_manifest_sha256"],
                "wavefield_values_recomputed": False,
                "all_cached_slices_retained": True,
                "confirmatory_or_preregistered": False,
            },
        }
        (temporary / "dataset_manifest.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        verify_dataset_provenance(temporary)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary
