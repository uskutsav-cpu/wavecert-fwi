from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from wavecert.data.wavefields import build_wavefield_dataset
from wavecert.provenance import sha256_file


@pytest.fixture
def development():
    pytest.importorskip("torch")
    module = importlib.import_module("wavecert.experiments.gradient_development")
    module.torch.set_num_threads(1)
    return module


def _tiny_fixture(root):
    source_protocol = Path(__file__).resolve().parents[1] / "configs/gradient_development_protocol.json"
    protocol = json.loads(source_protocol.read_text())
    protocol["model"] = {"modes_z": 2, "modes_x": 2, "width": 4, "depth": 1, "padding": 1}
    protocol["teacher"].update({"shape": [8, 8], "spacing": .05, "direction_seed": 19})
    protocol["training"].update({"seed": 19, "batch_size": 2, "optimizer_updates": 2,
                                  "validation_interval": 1, "scheduler_T_max": 2, "torch_threads": 1})
    dataset = root / "data"
    dataset.mkdir()
    all_rows, records = [], {}
    for code, role in enumerate(("train", "validation")):
        arrays = build_wavefield_dataset(n_samples=4, shape=(8, 8), seed=190 + code,
                                         n_source_positions=2, n_receivers=4)
        arrays.save(dataset / f"{role}.npz")
        rows = [{"split": role, "slice_id": f"{role}_slice_{index}",
                 "model_id": f"{role}_parent_{index // 2}", "model_type": "tiny_test",
                 "source_index": int(arrays.source_indices[index]),
                 "frequency_hz": float(arrays.frequencies_hz[index])} for index in range(4)]
        with (dataset / f"{role}_metadata.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        all_rows.extend(rows)
        records[role] = {"arrays_sha256": sha256_file(dataset / f"{role}.npz"),
                         "metadata_sha256": sha256_file(dataset / f"{role}_metadata.csv")}
    with (dataset / "split_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    role_hash = sha256_file(dataset / "split_manifest.csv")
    (dataset / "dataset_manifest.json").write_text(json.dumps({
        "role_manifest_sha256": role_hash, "splits": records,
    }))
    protocol["dataset"].update({"path": "data", "roles": records,
                                "dataset_manifest_sha256": sha256_file(dataset / "dataset_manifest.json"),
                                "role_manifest_sha256": role_hash})
    (root / "configs").mkdir()
    path = root / "configs/protocol.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol, path


def test_physical_relative_losses_are_independent_of_output_units_and_parent_row_counts(development):
    torch = development.torch
    truth = torch.tensor([[[[2., -3.], [1., 4.]]], [[[5., 1.], [-2., 3.]]]])
    prediction = truth + .25
    base = development.physical_relative_squared_l2(prediction, truth, 1e-20)
    converted = development.physical_relative_squared_l2(prediction * 1000, truth * 1000, 1e-20)
    torch.testing.assert_close(base, converted)
    assert development.parent_weighted_mean(np.asarray([0., 0., 0., 0., 1.]),
                                            np.asarray(["a", "a", "a", "a", "b"])) == .5


def test_tiny_comparison_finishes_every_candidate_without_any_test_or_calibration_files(development, tmp_path):
    protocol, path = _tiny_fixture(tmp_path)
    # Deliberately no calibration/test files exist: opening or requiring them
    # would fail this end-to-end development-only training/selection check.
    output = tmp_path / "results/tiny_development"
    summary = development.run_gradient_development(path, output, root=tmp_path)
    assert summary["status"] == "complete" and summary["source_unchanged_during_run"]
    assert summary["input_audit"]["opened_array_roles"] == ["train", "validation"]
    assert not summary["calibration_or_test_arrays_loaded"] and not summary["external_test_arrays_loaded"]
    assert len(summary["candidates"]) == 4
    for candidate in summary["candidates"]:
        assert candidate["optimizer_updates_completed"] == 2
        assert candidate["best_optimizer_update"] in {1, 2}
        directory = output / "candidates" / candidate["name"]
        history = list(csv.DictReader((directory / "training_history.csv").open(newline="")))
        assert len(history) == 2 and all(int(row["batch_size"]) == 2 for row in history)
        validation = json.loads((directory / "validation_history.json").read_text())
        assert [row["optimizer_update"] for row in validation] == [0, 1, 2]
        assert validation[0]["checkpoint_eligible"] is False
        selected = validation[candidate["best_optimizer_update"]]
        assert candidate["best_validation_score"] == pytest.approx(
            selected["receiver_relative_squared_l2"] + selected["jvp_relative_squared_l2"] + selected["direction_loss"]
        )
        assert Path(candidate["checkpoint_path"]).is_file()
    assert summary["selection"]["winner"] == min(summary["candidates"], key=lambda row: row["best_validation_score"])["name"]
    assert summary["selection"]["frozen_for_heldout_evaluation"] is False
    with np.load(output / "labels/train.npz") as train:
        norm = development.WavefieldNormalization.from_arrays(train["models"][:4], train["frequencies_hz"][:4], train["fields"][:4])
    assert summary["normalization"] == development.asdict(norm)
    _, audit_after = development.load_development_roles(protocol, tmp_path)
    assert audit_after["input_file_sha256"] == summary["input_audit"]["input_file_sha256"]


def test_teacher_tangent_matches_state_finite_differences_at_smoothed_float32_inputs(development, tmp_path):
    protocol, _ = _tiny_fixture(tmp_path)
    roles, _ = development.load_development_roles(protocol, tmp_path)
    labels, _ = development.make_state_labels(roles["validation"], "validation", protocol)
    index = len(labels["models"]) // 2
    m = labels["models"][index].astype(float).reshape(-1)
    direction = labels["directions"][index].astype(float).reshape(-1)
    physics = development._physics(protocol)
    q = physics.source(int(labels["source_indices"][index]))
    frequency = float(labels["frequencies_hz"][index])
    epsilon = 1e-5
    difference = (physics.solve_state(m + epsilon * direction, q, frequency)
                  - physics.solve_state(m - epsilon * direction, q, frequency)) / (2 * epsilon)
    stored = labels["tangents"][index]
    tangent = (stored[..., 0] + 1j * stored[..., 1]).reshape(-1)
    assert np.linalg.norm(tangent - difference) / np.linalg.norm(difference) < 5e-5
    with pytest.raises(ValueError, match="restricted"):
        development.make_state_labels(roles["validation"], "test_id", protocol)


def test_aggregate_input_gradient_matches_multisource_objective_finite_difference(development, tmp_path):
    protocol, _ = _tiny_fixture(tmp_path)
    roles, _ = development.load_development_roles(protocol, tmp_path)
    labels, _ = development.make_state_labels(roles["validation"], "validation", protocol)
    cases, _ = development.make_inverse_cases(labels, protocol)
    # Repeat the first case with another acquisition block so the same input
    # gradient must accumulate across separate evaluation chunks.
    for key in ("block_case_indices", "observations", "source_indices", "frequencies_hz"):
        cases[key] = np.concatenate([cases[key][:1], cases[key][:1]])
    cases["source_indices"][1] = int(roles["validation"].arrays.source_indices[-1])
    cases["frequencies_hz"][1] += .5
    torch = development.torch
    torch.manual_seed(3)
    model = development.FNO2dWavefield(development.FNOConfig(**protocol["model"]))
    norm = development.WavefieldNormalization.from_arrays(labels["models"][:4], labels["frequencies_hz"][:4], labels["fields"][:4])
    encoder = development.FNOInputEncoder((8, 8), norm)
    gradient = development._neural_inverse_gradients(model, encoder, cases, batch_size=1)
    direction = np.random.default_rng(99).normal(size=cases["models"].shape)
    direction /= np.linalg.norm(direction)

    def objective(models):
        with torch.no_grad():
            inputs = encoder.encode_batch(torch.as_tensor(models[cases["block_case_indices"]], dtype=torch.float32),
                                          torch.as_tensor(cases["source_indices"]), torch.as_tensor(cases["frequencies_hz"]))
            prediction = encoder.decode_output(model(inputs)).numpy().reshape(2, -1, 2)[:, cases["receiver_indices"]]
        return .5 * float(np.sum((prediction - cases["observations"]) ** 2))

    derivative = float(np.sum(gradient.reshape(direction.shape) * direction))
    errors = []
    for epsilon in (1e-2, 3e-3, 1e-3):
        difference = (objective(cases["models"] + epsilon * direction) - objective(cases["models"] - epsilon * direction)) / (2 * epsilon)
        errors.append(abs(difference - derivative) / max(abs(derivative), 1e-12))
    assert min(errors) < 5e-3


def test_selection_requires_every_frozen_candidate_and_uses_common_score_only(development):
    names = ["a", "b", "c", "d"]
    candidates = [{"name": name, "status": "complete", "best_validation_score": score,
                   "checkpoint_path": f"{name}.pt", "checkpoint_sha256": name,
                   "best_optimizer_update": 100, "unused_external_score": -index}
                  for index, (name, score) in enumerate(zip(names, [3., 1., 1., 2.], strict=True))]
    assert development.select_winner(candidates, names)["winner"] == "b"
    with pytest.raises(ValueError, match="all frozen"):
        development.select_winner(candidates[:3], names)
    candidates[0]["status"] = "failed"
    with pytest.raises(ValueError, match="every frozen"):
        development.select_winner(candidates, names)
