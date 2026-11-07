import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def grouped_dataset(tmp_path):
    pd = pytest.importorskip("pandas")
    h5py = pytest.importorskip("h5py")
    from wavecert.data.real_geology import (
        RealGeologyBuildConfig,
        build_subsurfacegen_wavecert_dataset,
    )

    selected = tmp_path / "selected"
    selected.mkdir()
    records = []
    for role_index, role in enumerate(["train", "validation", "calibration", "test_id", "test_ood"]):
        for parent in range(2):
            name = f"{role}-{parent}"
            filename = f"{name}.h5"
            velocity = np.broadcast_to(np.linspace(1.8, 2.5, 12)[:, None], (12, 12)).copy()
            velocity += 0.01 * role_index + 0.02 * parent
            with h5py.File(selected / filename, "w") as stream:
                stream["velocity"] = velocity.astype(np.float32)
            records.append({
                "role": role, "split": role, "source_split": "external-source",
                "slice_id": name, "model_id": name, "model_type": "mock-geology",
                "file_path": filename,
            })
    pd.DataFrame(records).to_csv(selected / "selected_manifest.csv", index=False)
    dataset = tmp_path / "dataset"
    summary = build_subsurfacegen_wavecert_dataset(
        selected_root=selected, output_root=dataset,
        config=RealGeologyBuildConfig(shape=(8, 8), frequencies_hz=(3.0,), n_source_positions=1, n_receivers=4),
    )
    return selected, dataset, summary


def test_builder_produces_dedicated_parent_isolated_calibration_with_content_identity(grouped_dataset):
    from wavecert.data.splits import audit_dataset_roles, verify_dataset_provenance
    from wavecert.data.wavefields import WavefieldDatasetArrays

    selected, dataset, summary = grouped_dataset
    assert summary["conformal_ready"]
    assert audit_dataset_roles(dataset)["parent_isolation_passed"]
    for role in ["train", "validation", "calibration", "test_id", "test_ood"]:
        arrays = WavefieldDatasetArrays.load(dataset / f"{role}.npz")
        assert len(arrays.models) == 2
        assert summary["splits"][role]["unique_parent_models"] == 2
    for filename, record in summary["raw_source_files"].items():
        assert record["sha256"] == hashlib.sha256((selected / filename).read_bytes()).hexdigest()
    verified = verify_dataset_provenance(dataset)
    assert len(verified["verified_content_files"]) == 11
    assert verified["dataset_manifest_sha256"] == hashlib.sha256((dataset / "dataset_manifest.json").read_bytes()).hexdigest()
    with (dataset / "calibration.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="content changed"):
        verify_dataset_provenance(dataset)


@pytest.mark.parametrize("jacobian_weight", [0.0, 0.01])
def test_training_keeps_calibration_out_of_model_selection_and_links_checkpoint(grouped_dataset, tmp_path, jacobian_weight):
    torch = pytest.importorskip("torch")
    from wavecert.data.splits import verify_calibration_checkpoint, verify_dataset_provenance
    from wavecert.experiments.real_geology_training import run_real_geology_training
    from wavecert.experiments.real_wavecert import run_real_wavecert_study
    from wavecert.surrogates.fno import FNOConfig

    _, dataset, _ = grouped_dataset
    checkpoint = tmp_path / "tiny-fno.pt"
    training = run_real_geology_training(
        dataset_dir=dataset, checkpoint_path=checkpoint, output_dir=tmp_path / "training",
        epochs=1, batch_size=2, patience=1,
        jacobian_weight=jacobian_weight,
        config=FNOConfig(modes_z=2, modes_x=2, width=4, depth=1, padding=1),
    )
    assert "not loaded" in training["calibration_role_use"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["calibration_role_used_for_model_selection"] is False
    assert payload["jacobian_weight"] == jacobian_weight
    if jacobian_weight:
        assert set(payload["jacobian_labels"]) == {"train", "validation"}
        assert (tmp_path / "training/jacobian_train_labels.npz").is_file()
    provenance = verify_dataset_provenance(dataset)
    assert verify_calibration_checkpoint(checkpoint, provenance)["calibration_role_used_for_model_selection"] is False
    result = run_real_wavecert_study(
        dataset_dir=dataset, checkpoint_path=checkpoint, output_dir=tmp_path / "evaluation",
        alpha=0.5, calibration_cases=10, evaluation_cases=1,
    )
    assert result["calibration_parent_models"] == 2
    assert result["finite_calibration_bound_supported"]
    assert result["id"]["n_parent_models"] == 1
    assert result["id"]["parent_bootstrap_gradient_cosine"]["interval"] is None
    assert json.loads((tmp_path / "evaluation/summary.json").read_text())["dataset_provenance"] == provenance
    payload.pop("dataset_manifest_sha256")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="matching dataset-manifest"):
        verify_calibration_checkpoint(checkpoint, provenance)


def test_selector_is_audited_and_does_not_download_by_default(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    from wavecert.data.subsurfacegen_selection import select_subsurfacegen_slices

    frames = {}
    for split in ["train", "test_in_dist", "test_out_dist"]:
        path = tmp_path / "index/data" / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test-index-identity")
        frames[split] = pd.DataFrame([
            {"data_type": "slice", "slice_id": f"{split}-{i}", "model_id": f"{split}-{i}", "model_type": "ood" if split == "test_out_dist" else "id", "file_path": f"{split}-{i}.h5"}
            for i in range(8)
        ])
    monkeypatch.setattr(pd, "read_parquet", lambda path: frames[Path(path).stem])
    manifest = select_subsurfacegen_slices(
        index_root=tmp_path / "index", output_root=tmp_path / "selection",
        train_count=8, id_count=8, ood_count=8, seed=4,
    )
    audit = json.loads((tmp_path / "selection/selected_manifest.audit.json").read_text())
    assert audit["parent_isolation_passed"]
    assert audit["download_requested"] is False
    assert {"source_split", "role"}.issubset(manifest.columns)
    assert not list((tmp_path / "selection").glob("*.h5"))


def test_cached_reassignment_preserves_fields_and_records_new_protocol(grouped_dataset, tmp_path):
    pd = pytest.importorskip("pandas")
    from wavecert.data.real_geology import reassign_dataset_roles
    from wavecert.data.splits import verify_dataset_provenance
    from wavecert.data.wavefields import WavefieldDatasetArrays

    selected, dataset, _ = grouped_dataset
    assigned = pd.read_csv(selected / "selected_manifest.csv")
    mapping = {"train": "validation", "validation": "calibration", "calibration": "test_id", "test_id": "train", "test_ood": "test_ood"}
    assigned["role"] = assigned["role"].map(mapping)
    assigned["split"] = assigned["role"]
    replacement = tmp_path / "new_roles.csv"
    assigned.to_csv(replacement, index=False)
    destination = tmp_path / "new-dataset"
    original_identity = verify_dataset_provenance(dataset)
    result = reassign_dataset_roles(dataset_dir=dataset, manifest_path=replacement, output_dir=destination)
    assert result["reassignment"]["wavefield_values_recomputed"] is False
    assert result["reassignment"]["confirmatory_or_preregistered"] is False
    assert verify_dataset_provenance(dataset) == original_identity
    assert verify_dataset_provenance(destination)["dataset_manifest_sha256"] != original_identity["dataset_manifest_sha256"]
    for old_role, new_role in mapping.items():
        original = WavefieldDatasetArrays.load(dataset / f"{old_role}.npz")
        reassigned = WavefieldDatasetArrays.load(destination / f"{new_role}.npz")
        np.testing.assert_array_equal(original.models, reassigned.models)
        np.testing.assert_array_equal(original.wavefields, reassigned.wavefields)
        np.testing.assert_array_equal(original.source_indices, reassigned.source_indices)
        np.testing.assert_array_equal(original.frequencies_hz, reassigned.frequencies_hz)
    with pytest.raises(ValueError, match="new destination"):
        reassign_dataset_roles(dataset_dir=dataset, manifest_path=replacement, output_dir=destination)
