from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wavecert.experiments.pod_development import (
    fit_normalized_real_pod,
    load_teacher_inputs,
    normalized_real_snapshots,
    run_pod_development,
)


def test_snapshot_normalization_preserves_complex_channel_pair_weights_and_scale():
    fields = np.zeros((2, 2, 3, 2))
    fields[0, ..., 0] = 3.
    fields[0, ..., 1] = 4.
    fields[1, ..., 0] = 10.
    matrix = normalized_real_snapshots(fields, None, 1e-20)
    # Real and imaginary parts share one complex norm (3/5 and 4/5),
    # instead of independently receiving unit weight.
    assert np.linalg.norm(matrix[:, 0]) == pytest.approx(.6)
    assert np.linalg.norm(matrix[:, 2]) == pytest.approx(.8)
    assert np.linalg.norm(matrix[:, 1]) == pytest.approx(1.)
    np.testing.assert_allclose(matrix, normalized_real_snapshots(fields * 1000, None, 1e-20))


def test_one_training_svd_supplies_all_prefixes_and_state_tangent_weights(monkeypatch):
    rng = np.random.default_rng(17)
    fields, tangents = rng.normal(size=(5, 3, 4, 2)), rng.normal(size=(5, 3, 4, 2))
    count = 0
    original = np.linalg.svd

    def counted(*args, **kwargs):
        nonlocal count
        count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counted)
    basis, singular_values, record = fit_normalized_real_pod(fields, tangents, max_rank=4, floor=1e-20)
    assert count == record["svd_evaluations"] == 1
    assert basis.shape == (12, 4) and len(singular_values) == 12
    for rank in (1, 2, 4):
        np.testing.assert_allclose(basis[:, :rank].T @ basis[:, :rank], np.eye(rank), atol=2e-14)
    snapshots = normalized_real_snapshots(fields, tangents, 1e-20)
    assert np.sum(snapshots[:, :10] ** 2) == pytest.approx(5.)
    assert np.sum(snapshots[:, 10:] ** 2) == pytest.approx(5.)


def test_input_identity_failure_stops_before_loading_or_fitting(tmp_path, monkeypatch):
    protocol = {"teacher_inputs": {"protocol_sha256": "not-the-file-hash", "teacher_labels_sha256": "x", "input_audit_sha256": "y"}}
    (tmp_path / "protocol.json").write_text("{}")
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("must not load changed input archives"))
    with pytest.raises(ValueError, match="pinned teacher study input changed"):
        load_teacher_inputs(protocol, tmp_path)


def test_isolated_runner_rejects_any_output_into_read_only_teacher_tree(tmp_path):
    isolated, main = tmp_path / "isolated", tmp_path / "main"
    isolated.mkdir()
    main.mkdir()
    protocol = isolated / "protocol.json"
    protocol.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ValueError, match="inside this isolated repository"):
        run_pod_development(protocol, main / "results/new", main / "results/overwrite", root=isolated)
    assert not (main / "results").exists()


def test_protocol_is_frozen_to_all_eight_candidates_without_heldout_roles():
    path = Path(__file__).resolve().parents[1] / "configs/pod_development_protocol.json"
    protocol = json.loads(path.read_text())
    assert len(protocol["basis_families"]) * len(protocol["ranks"]) == 8
    assert protocol["teacher_inputs"]["roles_opened"] == ["train", "validation", "validation_inverse_cases"]
    assert protocol["accounting"]["timing_claims"] is False


def test_generic_validation_matches_fno_common_criterion_on_an_independent_tiny_problem():
    torch = pytest.importorskip("torch")
    from wavecert.experiments.gradient_development import evaluate_validation
    from wavecert.experiments.pod_development import evaluate_surrogate_validation
    from wavecert.physics.helmholtz import Helmholtz2D
    from wavecert.surrogates.fno import (
        FNO2dWavefield,
        FNOConfig,
        TrainedFNOWavefieldSurrogate,
        WavefieldNormalization,
    )

    torch.set_num_threads(1)
    torch.manual_seed(71)
    rng = np.random.default_rng(71)
    shape = (4, 5)
    physics = Helmholtz2D(shape=shape, spacing=.1, damping_width=2, damping_strength=1.5)
    models = (.18 + .04 * rng.random((4, *shape))).astype(np.float32)
    directions = rng.normal(size=models.shape)
    directions /= np.linalg.norm(directions.reshape(4, -1), axis=1)[:, None, None]
    sources, frequencies = np.asarray([6, 7, 6, 7]), np.asarray([1.7, 2.1, 1.7, 2.1], dtype=np.float32)
    fields, tangents = [], []
    for index in range(4):
        m = models[index].astype(float).reshape(-1)
        state = physics.solve_state(m, physics.source(int(sources[index])), float(frequencies[index]))
        tangent = physics.solve_tangent(m, state, directions[index].reshape(-1), float(frequencies[index]))
        fields.append(np.stack([state.real, state.imag], axis=-1).reshape(*shape, 2))
        tangents.append(np.stack([tangent.real, tangent.imag], axis=-1).reshape(*shape, 2))
    receivers = np.asarray([2, 6, 6, 11])
    labels = {"models": models, "directions": directions.astype(np.float32),
              "source_indices": sources, "frequencies_hz": frequencies,
              "fields": np.asarray(fields, dtype=np.float32), "tangents": np.asarray(tangents, dtype=np.float32),
              "parent_ids": np.asarray(["p0", "p1", "p0", "p1"]),
              "slice_ids": np.asarray(["s0", "s1", "s0", "s1"]),
              "state_types": np.asarray(["raw", "raw", "smoothed", "smoothed"]),
              "receiver_indices": receivers}
    observations = labels["fields"][:2].reshape(2, -1, 2)[:, receivers]
    gradients = []
    for index in range(2):
        obs = observations[index, :, 0] + 1j * observations[index, :, 1]
        _, gradient, _, _ = physics.objective_and_gradient(models[index + 2].astype(float).reshape(-1),
            physics.source(int(sources[index])), obs, tuple(receivers.tolist()), float(frequencies[index]))
        gradients.append(gradient)
    cases = {"models": models[2:], "parent_ids": np.asarray(["p0", "p1"]),
             "slice_ids": np.asarray(["s0", "s1"]), "block_case_indices": np.asarray([0, 1]),
             "source_indices": sources[:2], "frequencies_hz": frequencies[:2], "receiver_indices": receivers,
             "observations": observations, "exact_gradients": np.asarray(gradients)}
    norm = WavefieldNormalization.from_arrays(models[:2], frequencies[:2], labels["fields"][:2])
    surrogate = TrainedFNOWavefieldSurrogate(FNO2dWavefield(FNOConfig(modes_z=1, modes_x=1, width=4, depth=1, padding=1)),
                                            shape=shape, normalization=norm)
    teacher = {"teacher": {"shape": list(shape), "spacing": .1, "damping_width": 2, "damping_strength": 1.5},
               "training": {"batch_size": 2}, "loss": {"squared_norm_denominator_floor": 1e-20},
               "validation_inverse_cases": {"gradient_norm_zero_threshold": 1e-14}}
    neural = evaluate_validation(surrogate.model, surrogate.encoder, labels, cases, teacher)
    generic = evaluate_surrogate_validation(surrogate, labels, cases, teacher)
    for key in ("selection_score", "receiver_relative_squared_l2", "jvp_relative_squared_l2", "direction_loss",
                "aggregate_gradient_cosine", "aggregate_neural_true_descent"):
        assert generic[key] == pytest.approx(neural[key], rel=3e-5, abs=3e-5)
