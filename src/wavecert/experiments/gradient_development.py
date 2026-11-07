"""Bounded train/validation-only augmentation and derivative-supervision study.

This is an exploratory comparison of established training choices. It does not
load calibration/test arrays, tune on external outcomes, or certify a new model.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

from wavecert.data.splits import audit_split_manifest
from wavecert.data.wavefields import WavefieldDatasetArrays
from wavecert.experiments.derivative_training import normalized_value_and_tangent
from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.provenance import environment_record, sha256_file, source_digest
from wavecert.surrogates.fno import (
    FNO2dWavefield,
    FNOConfig,
    FNOInputEncoder,
    WavefieldNormalization,
    checkpoint_payload,
)

try:
    import torch
except ImportError:  # pragma: no cover - optional training dependency
    torch = None


@dataclass
class DevelopmentRole:
    arrays: WavefieldDatasetArrays
    metadata: list[dict]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _digest(str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes())


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".gradient-dev-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json(path: Path, value) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def _write_npz(path: Path, arrays: dict) -> None:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    _atomic_bytes(path, stream.getvalue())


def _write_csv(path: Path, rows: list[dict]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def validate_protocol(protocol: dict) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported gradient-development protocol schema")
    expected = {("raw", 0.0), ("raw", 1.0), ("raw+smoothed", 0.0), ("raw+smoothed", 1.0)}
    candidates = protocol["candidates"]
    if len(candidates) != 4 or {(row["augmentation"], row["derivative_weight"]) for row in candidates} != expected:
        raise ValueError("the development comparison must retain all four frozen 2x2 candidates")
    names = [row["name"] for row in candidates]
    if len(set(names)) != 4 or any(not name.replace("_", "").isalnum() for name in names):
        raise ValueError("candidate names must be unique safe path components")
    training = protocol["training"]
    for key in ("seed", "batch_size", "optimizer_updates", "validation_interval", "torch_threads"):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "seed" else 1):
            raise ValueError(f"invalid training {key}")
    if training["optimizer_updates"] % training["validation_interval"]:
        raise ValueError("update budget must end at a scheduled validation")
    if training["scheduler_T_max"] != training["optimizer_updates"] or training["device"] != "cpu":
        raise ValueError("this bounded comparison requires CPU and scheduler T_max equal to update budget")
    if protocol["dataset"]["allowed_array_roles"] != ["train", "validation"]:
        raise ValueError("only train and validation arrays are permitted")
    if not np.isfinite(protocol["loss"]["squared_norm_denominator_floor"]) or protocol["loss"]["squared_norm_denominator_floor"] <= 0:
        raise ValueError("relative loss denominator floor must be positive and finite")


def load_development_roles(protocol: dict, root: Path) -> tuple[dict, dict]:
    """Verify/open only train and validation data, never other role arrays."""
    config = protocol["dataset"]
    dataset = root / config["path"]
    manifest_path = dataset / "dataset_manifest.json"
    if sha256_file(manifest_path) != config["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest differs from frozen development protocol")
    manifest = json.loads(manifest_path.read_text())
    if sha256_file(dataset / "split_manifest.csv") != config["role_manifest_sha256"]:
        raise ValueError("parent-role manifest differs from frozen development protocol")
    loaded, records, hashes = {}, [], {}
    for role in ("train", "validation"):
        paths = {"arrays": dataset / f"{role}.npz", "metadata": dataset / f"{role}_metadata.csv"}
        for kind, path in paths.items():
            actual = sha256_file(path)
            expected = config["roles"][role][f"{kind}_sha256"]
            if actual != expected or actual != manifest["splits"][role][f"{kind}_sha256"]:
                raise ValueError(f"{role} {kind} differs from frozen input identity")
            hashes[str(path.relative_to(root))] = actual
        arrays = WavefieldDatasetArrays.load(paths["arrays"])
        with paths["metadata"].open(newline="") as stream:
            metadata = list(csv.DictReader(stream))
        if len(metadata) != len(arrays.models) or not metadata:
            raise ValueError(f"{role} array and metadata counts disagree or are empty")
        for index, row in enumerate(metadata):
            if row["split"] != role or int(row["source_index"]) != int(arrays.source_indices[index]) or float(row["frequency_hz"]) != float(arrays.frequencies_hz[index]):
                raise ValueError(f"{role} metadata and array acquisition identities disagree")
        if not np.all(np.isfinite(arrays.models)) or np.any(arrays.models <= 0):
            raise ValueError("development models must be finite positive squared slowness")
        if arrays.models.dtype != np.float32:
            raise ValueError("development model inputs must already be stored float32")
        if np.any(arrays.source_indices < 0) or np.any(arrays.source_indices >= np.prod(arrays.shape)) or not np.all(np.isfinite(arrays.frequencies_hz)) or np.any(arrays.frequencies_hz <= 0):
            raise ValueError("development acquisition inputs must be finite and valid")
        loaded[role] = DevelopmentRole(arrays, metadata)
        records.extend({**row, "role": role} for row in metadata)
    audit = audit_split_manifest(records)
    if not audit["parent_isolation_passed"]:
        raise ValueError("training and validation parents/slices must remain disjoint")
    train = loaded["train"].arrays
    validation = loaded["validation"].arrays
    teacher = protocol["teacher"]
    if train.shape != validation.shape or list(train.shape) != teacher["shape"] or train.spacing != teacher["spacing"] or validation.spacing != train.spacing:
        raise ValueError("development grid does not match the frozen reference teacher")
    if not np.array_equal(train.receiver_indices, validation.receiver_indices):
        raise ValueError("training and validation receiver geometry differs")
    return loaded, {
        "dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "role_manifest_sha256": config["role_manifest_sha256"],
        "opened_array_roles": ["train", "validation"], "input_file_sha256": hashes,
        "parent_and_slice_audit": audit, "calibration_or_test_arrays_loaded": False,
        "external_test_arrays_loaded": False,
    }


def _physics(protocol: dict) -> Helmholtz2D:
    config = protocol["teacher"]
    return Helmholtz2D(shape=tuple(config["shape"]), spacing=config["spacing"],
                       damping_width=config["damping_width"], damping_strength=config["damping_strength"])


def make_state_labels(role: DevelopmentRole, role_name: str, protocol: dict) -> tuple[dict, dict]:
    """Recompute raw/smoothed state and tangent labels at stored float32 inputs."""
    if role_name not in {"train", "validation"}:
        raise ValueError("teacher labels are restricted to train and validation")
    arrays = role.arrays
    raw = arrays.models.copy()
    sigma = protocol["states"]["smoothing"]["sigma_grid_cells"]
    velocity = 1.0 / np.sqrt(raw.astype(np.float64))
    smooth_velocity = ndi.gaussian_filter(velocity, sigma=(0, sigma, sigma), mode="reflect")
    smooth = (1.0 / smooth_velocity**2).astype(np.float32)
    models = np.concatenate([raw, smooth])
    n = len(raw)
    directions = np.empty_like(models)
    role_code = 0 if role_name == "train" else 1
    for state_code in (0, 1):
        rng = np.random.default_rng(np.random.SeedSequence([
            protocol["teacher"]["direction_seed"], role_code, state_code,
        ]))
        values = rng.normal(size=raw.shape)
        values /= np.linalg.norm(values.reshape(n, -1), axis=1)[:, None, None]
        directions[state_code * n:(state_code + 1) * n] = values.astype(np.float32)
    fields = np.empty((2 * n, *arrays.shape, 2), dtype=np.float32)
    tangents = np.empty_like(fields)
    sources = np.tile(arrays.source_indices, 2)
    frequencies = np.tile(arrays.frequencies_hz, 2)
    physics = _physics(protocol)
    started = time.perf_counter()
    for index, model in enumerate(models):
        m = model.astype(np.float64).reshape(-1)
        q = physics.source(int(sources[index]))
        frequency = float(frequencies[index])
        state = physics.solve_state(m, q, frequency)
        tangent = physics.solve_tangent(m, state, directions[index].astype(np.float64).reshape(-1), frequency)
        for destination, value in ((fields, state), (tangents, tangent)):
            value = value.reshape(arrays.shape)
            destination[index, ..., 0], destination[index, ..., 1] = value.real, value.imag
    result = {
        "models": models, "directions": directions, "fields": fields, "tangents": tangents,
        "source_indices": sources, "frequencies_hz": frequencies,
        "parent_ids": np.tile(np.asarray([row["model_id"] for row in role.metadata]), 2),
        "slice_ids": np.tile(np.asarray([row["slice_id"] for row in role.metadata]), 2),
        "state_types": np.asarray(["raw"] * n + ["smoothed"] * n),
        "original_example_indices": np.tile(np.arange(n), 2),
        "receiver_indices": arrays.receiver_indices,
    }
    return result, {
        "role": role_name, "raw_examples": n, "all_state_examples": 2 * n,
        "n_parent_models": len(set(result["parent_ids"])), "seconds": time.perf_counter() - started,
        "offline_physics_cost": physics.cost_snapshot(),
        "array_sha256": {key: _array_digest(value) for key, value in result.items()},
        "maximum_direction_unit_norm_error": float(np.max(np.abs(
            np.linalg.norm(directions.astype(float).reshape(2 * n, -1), axis=1) - 1
        ))),
        "calibration_or_test_labels_generated": False,
    }


def make_inverse_cases(validation: dict, protocol: dict) -> tuple[dict, dict]:
    """All validation slices, smoothed inputs and raw teacher observations."""
    n = len(validation["models"]) // 2
    slice_ids = sorted(set(validation["slice_ids"][:n]))
    mapping = {name: index for index, name in enumerate(slice_ids)}
    block_case = np.asarray([mapping[name] for name in validation["slice_ids"][:n]])
    shape = tuple(protocol["teacher"]["shape"])
    models, parents = [], []
    for name in slice_ids:
        indices = np.flatnonzero(validation["slice_ids"][:n] == name)
        if len(set(validation["parent_ids"][indices])) != 1:
            raise ValueError("one validation slice maps to multiple parents")
        candidate_models = validation["models"][n + indices]
        if not np.all(candidate_models == candidate_models[0]):
            raise ValueError("validation blocks in a slice do not share the same smoothed model")
        models.append(candidate_models[0])
        parents.append(str(validation["parent_ids"][indices[0]]))
    models = np.asarray(models, dtype=np.float32)
    receivers = validation["receiver_indices"]
    observations = validation["fields"][:n].reshape(n, -1, 2)[:, receivers]
    gradients = np.zeros((len(models), int(np.prod(shape))), dtype=np.float64)
    objectives = np.zeros(len(models), dtype=np.float64)
    physics = _physics(protocol)
    started = time.perf_counter()
    for index in range(n):
        case = block_case[index]
        obs = observations[index, :, 0] + 1j * observations[index, :, 1]
        objective, gradient, _, _ = physics.objective_and_gradient(
            models[case].astype(float).reshape(-1), physics.source(int(validation["source_indices"][index])),
            obs, tuple(receivers.tolist()), float(validation["frequencies_hz"][index]),
        )
        gradients[case] += gradient
        objectives[case] += objective
    result = {
        "models": models, "parent_ids": np.asarray(parents), "slice_ids": np.asarray(slice_ids),
        "block_case_indices": block_case, "observations": observations,
        "source_indices": validation["source_indices"][:n],
        "frequencies_hz": validation["frequencies_hz"][:n], "receiver_indices": receivers,
        "exact_gradients": gradients, "exact_objectives": objectives,
    }
    return result, {
        "n_slices": len(models), "n_parents": len(set(parents)), "n_blocks": n,
        "seconds": time.perf_counter() - started, "offline_physics_cost": physics.cost_snapshot(),
        "array_sha256": {key: _array_digest(value) for key, value in result.items()},
        "case_sampling": "every original validation slice, without subsampling",
    }


def parent_weighted_mean(values: np.ndarray, parent_ids: np.ndarray) -> float:
    values, parent_ids = np.asarray(values, dtype=float), np.asarray(parent_ids)
    if values.ndim != 1 or values.shape != parent_ids.shape or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("finite one-dimensional values and matching parent IDs are required")
    return float(np.mean([np.mean(values[parent_ids == parent]) for parent in sorted(set(parent_ids))]))


def physical_relative_squared_l2(prediction, target, floor: float):
    dimensions = tuple(range(1, prediction.ndim))
    return torch.sum((prediction - target) ** 2, dim=dimensions) / torch.clamp(
        torch.sum(target**2, dim=dimensions), min=floor,
    )


def _physical_predictions(model, encoder, models, sources, frequencies, directions, *, tangent: bool):
    if tangent:
        values, normalized_tangent = normalized_value_and_tangent(
            model, encoder, models, sources, frequencies, directions,
        )
        norm = encoder.normalization
        scale = torch.as_tensor([norm.output_std_real, norm.output_std_imag], dtype=values.dtype).view(1, 2, 1, 1)
        physical_tangent = (normalized_tangent * scale).permute(0, 2, 3, 1)
    else:
        values = model(encoder.encode_batch(models, sources, frequencies))
        physical_tangent = None
    return encoder.decode_output(values), physical_tangent


def _tensor_pool(labels):
    return {key: torch.as_tensor(labels[key], dtype=torch.long if key == "source_indices" else torch.float32)
            for key in ("models", "source_indices", "frequencies_hz", "fields", "directions", "tangents")}


def _neural_inverse_gradients(model, encoder, cases, batch_size: int):
    models = torch.tensor(cases["models"], dtype=torch.float32, requires_grad=True)
    result = np.zeros((len(models), models[0].numel()), dtype=np.float64)
    receivers = torch.as_tensor(cases["receiver_indices"], dtype=torch.long)
    for start in range(0, len(cases["block_case_indices"]), batch_size):
        stop = start + batch_size
        case_indices = torch.as_tensor(cases["block_case_indices"][start:stop], dtype=torch.long)
        sources = torch.as_tensor(cases["source_indices"][start:stop], dtype=torch.long)
        frequencies = torch.as_tensor(cases["frequencies_hz"][start:stop], dtype=torch.float32)
        predicted = encoder.decode_output(model(encoder.encode_batch(models[case_indices], sources, frequencies)))
        predicted = predicted.reshape(len(case_indices), -1, 2)[:, receivers]
        observed = torch.as_tensor(cases["observations"][start:stop], dtype=predicted.dtype)
        objective = .5 * torch.sum((predicted - observed) ** 2)
        (gradient,) = torch.autograd.grad(objective, models)
        result += gradient.detach().numpy().reshape(result.shape)
    return result


def evaluate_validation(model, encoder, labels, cases, protocol: dict) -> dict:
    model.eval()
    pool = _tensor_pool(labels)
    receivers = torch.as_tensor(labels["receiver_indices"], dtype=torch.long)
    floor = protocol["loss"]["squared_norm_denominator_floor"]
    batch_size = protocol["training"]["batch_size"]
    losses = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(labels["models"]), batch_size):
            stop = start + batch_size
            batch = {key: value[start:stop] for key, value in pool.items()}
            predicted, tangent = _physical_predictions(model, encoder, batch["models"], batch["source_indices"],
                                                      batch["frequencies_hz"], batch["directions"], tangent=True)
            for key, prediction, target in (
                ("full_field_relative_squared_l2", predicted, batch["fields"]),
                ("receiver_relative_squared_l2", predicted.reshape(len(predicted), -1, 2)[:, receivers],
                 batch["fields"].reshape(len(predicted), -1, 2)[:, receivers]),
                ("jvp_relative_squared_l2", tangent, batch["tangents"]),
            ):
                losses[key].extend(physical_relative_squared_l2(prediction, target, floor).cpu().tolist())
    mean_losses = {key: parent_weighted_mean(np.asarray(value), labels["parent_ids"]) for key, value in losses.items()}
    neural = _neural_inverse_gradients(model, encoder, cases, batch_size)
    if not np.all(np.isfinite(neural)) or not np.all(np.isfinite(cases["exact_gradients"])):
        raise FloatingPointError("non-finite aggregate validation gradient")
    threshold = protocol["validation_inverse_cases"]["gradient_norm_zero_threshold"]
    inverse_rows = []
    for index, (exact_gradient, neural_gradient) in enumerate(zip(cases["exact_gradients"], neural, strict=True)):
        exact_norm, neural_norm = float(np.linalg.norm(exact_gradient)), float(np.linalg.norm(neural_gradient))
        if not np.isfinite(exact_norm) or not np.isfinite(neural_norm):
            raise FloatingPointError("aggregate validation gradient norm overflowed")
        cosine = (float(np.clip(np.dot(exact_gradient, neural_gradient) / (exact_norm * neural_norm), -1, 1))
                  if min(exact_norm, neural_norm) > threshold else 0.0)
        derivative = float(-np.dot(exact_gradient, neural_gradient) / neural_norm) if neural_norm > threshold else 0.0
        inverse_rows.append({
            "slice_id": str(cases["slice_ids"][index]), "model_id": str(cases["parent_ids"][index]),
            "gradient_cosine": cosine, "exact_gradient_norm": exact_norm, "neural_gradient_norm": neural_norm,
            "exact_gradient_zero": exact_norm <= threshold, "neural_gradient_zero": neural_norm <= threshold,
            "neural_direction_exact_derivative": derivative, "neural_true_descent": derivative < 0,
        })
    for key in ("gradient_cosine", "exact_gradient_norm", "neural_gradient_norm", "neural_true_descent"):
        mean_losses[f"aggregate_{key}"] = parent_weighted_mean(
            np.asarray([row[key] for row in inverse_rows], dtype=float), cases["parent_ids"],
        )
    mean_losses["direction_loss"] = 1 - mean_losses["aggregate_gradient_cosine"]
    mean_losses["selection_score"] = (mean_losses["receiver_relative_squared_l2"]
                                      + mean_losses["jvp_relative_squared_l2"] + mean_losses["direction_loss"])
    mean_losses["inverse_cases"] = inverse_rows
    mean_losses["parent_components"] = {
        str(parent): {key: float(np.mean(np.asarray(value)[labels["parent_ids"] == parent]))
                      for key, value in losses.items()}
        for parent in sorted(set(labels["parent_ids"]))
    }
    for parent, components in mean_losses["parent_components"].items():
        rows = [row for row in inverse_rows if row["model_id"] == parent]
        components.update({f"aggregate_{key}": float(np.mean([row[key] for row in rows]))
                           for key in ("gradient_cosine", "neural_true_descent", "exact_gradient_norm", "neural_gradient_norm")})
    mean_losses["state_type_components"] = {
        str(state): {key: parent_weighted_mean(np.asarray(value)[labels["state_types"] == state],
                                              labels["parent_ids"][labels["state_types"] == state])
                     for key, value in losses.items()}
        for state in ("raw", "smoothed")
    }
    return mean_losses


class PermutationBatches:
    """Concatenate independent shuffled cycles to keep every update full sized."""

    def __init__(self, size: int, batch_size: int, seed: int, stream: int):
        if size < 1 or batch_size < 1:
            raise ValueError("batch stream requires positive data and batch sizes")
        self.size, self.batch_size = size, batch_size
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
        self.pending = np.empty(0, dtype=int)

    def next(self):
        while len(self.pending) < self.batch_size:
            self.pending = np.concatenate([self.pending, self.rng.permutation(self.size)])
        selected, self.pending = self.pending[:self.batch_size], self.pending[self.batch_size:]
        return selected


def train_candidate(
    candidate: dict, training_labels: dict, validation_labels: dict, inverse_cases: dict,
    normalization: WavefieldNormalization, protocol: dict, output: Path, *,
    protocol_sha256: str, input_audit: dict, label_records: dict, source_sha256: str,
) -> dict:
    """Complete one fixed update budget; checkpoint selection uses validation only."""
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for the gradient development study")
    training = protocol["training"]
    torch.manual_seed(training["seed"])
    config = FNOConfig(**protocol["model"])
    shape = tuple(protocol["teacher"]["shape"])
    model = FNO2dWavefield(config)
    encoder = FNOInputEncoder(shape, normalization)
    raw_examples = len(training_labels["models"]) // 2
    count = raw_examples if candidate["augmentation"] == "raw" else 2 * raw_examples
    pool = _tensor_pool({key: value[:count] for key, value in training_labels.items()
                         if key in {"models", "source_indices", "frequencies_hz", "fields", "directions", "tangents"}})
    receivers = torch.as_tensor(training_labels["receiver_indices"], dtype=torch.long)
    batches = PermutationBatches(count, training["batch_size"], training["seed"], training["batch_seed_stream"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training["scheduler_T_max"], eta_min=training["scheduler_eta_min"],
    )
    floor = protocol["loss"]["squared_norm_denominator_floor"]
    weight = candidate["derivative_weight"]
    output.mkdir(parents=True, exist_ok=True)
    best_score, best_update = float("inf"), None
    best_validation = None
    training_history, validation_history = [], []
    training_seconds = validation_seconds = 0.0
    started = time.perf_counter()

    def validate(update: int):
        nonlocal best_score, best_update, best_validation, validation_seconds
        validation_start = time.perf_counter()
        metrics = evaluate_validation(model, encoder, validation_labels, inverse_cases, protocol)
        validation_seconds += time.perf_counter() - validation_start
        row = {"optimizer_update": update, "checkpoint_eligible": update > 0, **metrics}
        validation_history.append(row)
        score = metrics["selection_score"]
        if not np.isfinite(score):
            raise FloatingPointError("non-finite common validation score")
        if update > 0 and score < best_score:
            best_score, best_update, best_validation = score, update, copy.deepcopy(row)
            buffer = io.BytesIO()
            torch.save(checkpoint_payload(
                model, config, normalization, shape,
                seed=training["seed"], best_optimizer_update=update,
                optimizer_updates_budget=training["optimizer_updates"],
                candidate=dict(candidate), development_protocol_sha256=protocol_sha256,
                development_source_sha256=source_sha256,
                dataset="SubsurfaceGen-derived train/validation-only gradient development",
                dataset_manifest_sha256=input_audit["dataset_manifest_sha256"],
                training_manifest_sha256=input_audit["role_manifest_sha256"],
                calibration_role_used_for_model_selection=False,
                calibration_or_test_arrays_loaded=False, external_test_arrays_loaded=False,
                receiver_indices=training_labels["receiver_indices"].tolist(),
                teacher_label_artifacts=label_records,
                normalization_source="recomputed raw training examples only, shared across all candidates",
                best_validation_selection_score=score,
                best_validation_components={key: metrics[key] for key in (
                    "receiver_relative_squared_l2", "jvp_relative_squared_l2", "direction_loss",
                    "aggregate_gradient_cosine", "aggregate_neural_true_descent",
                )},
            ), buffer)
            _atomic_bytes(output / "best.pt", buffer.getvalue())
            _write_json(output / "best_validation.json", best_validation)
        _write_json(output / "validation_history.json", validation_history)
        _write_json(output / "progress.json", {
            "status": "running", "optimizer_update": update,
            "optimizer_update_budget": training["optimizer_updates"],
            "best_update": best_update, "best_score": None if best_update is None else best_score,
            "latest_validation": {key: metrics[key] for key in (
                "selection_score", "receiver_relative_squared_l2", "jvp_relative_squared_l2",
                "direction_loss", "aggregate_gradient_cosine", "aggregate_neural_true_descent",
            )},
        })
        print(json.dumps({"candidate": candidate["name"], "update": update,
                          "score": score, "inverse_gradient_cosine": metrics["aggregate_gradient_cosine"]}), flush=True)

    validate(0)
    for update in range(1, training["optimizer_updates"] + 1):
        training_start = time.perf_counter()
        model.train()
        indices = torch.as_tensor(batches.next(), dtype=torch.long)
        batch = {key: value[indices] for key, value in pool.items()}
        predicted, predicted_tangent = _physical_predictions(
            model, encoder, batch["models"], batch["source_indices"], batch["frequencies_hz"],
            batch["directions"], tangent=weight > 0,
        )
        full = physical_relative_squared_l2(predicted, batch["fields"], floor).mean()
        receiver = physical_relative_squared_l2(
            predicted.reshape(len(indices), -1, 2)[:, receivers],
            batch["fields"].reshape(len(indices), -1, 2)[:, receivers], floor,
        ).mean()
        derivative = (physical_relative_squared_l2(predicted_tangent, batch["tangents"], floor).mean()
                      if weight > 0 else torch.zeros((), dtype=predicted.dtype))
        loss = full + protocol["loss"]["receiver_weight"] * receiver + weight * derivative
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite training loss in {candidate['name']} at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"], error_if_nonfinite=True)
        used_lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        training_seconds += time.perf_counter() - training_start
        training_history.append({
            "optimizer_update": update, "batch_size": len(indices), "total_loss": float(loss.detach()),
            "full_field_relative_squared_l2": float(full.detach()),
            "receiver_relative_squared_l2": float(receiver.detach()),
            "jvp_relative_squared_l2": float(derivative.detach()) if weight > 0 else None,
            "jvp_training_evaluated": weight > 0, "derivative_weight": weight,
            "preclip_parameter_gradient_norm": float(gradient_norm), "learning_rate_used": used_lr,
        })
        if update % training["validation_interval"] == 0:
            _write_csv(output / "training_history.csv", training_history)
            validate(update)
    if best_update is None:
        raise RuntimeError("no eligible trained checkpoint was evaluated")
    result = {
        "name": candidate["name"], "status": "complete", "candidate": dict(candidate),
        "optimizer_updates_completed": training["optimizer_updates"],
        "training_examples": count, "training_parent_counts": dict(sorted(Counter(
            str(parent) for parent in training_labels["parent_ids"][:count]
        ).items())),
        "best_optimizer_update": best_update, "best_validation_score": best_score,
        "best_validation": best_validation, "checkpoint_path": str(output / "best.pt"),
        "checkpoint_sha256": sha256_file(output / "best.pt"),
        "training_seconds": training_seconds, "validation_seconds": validation_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "calibration_or_test_arrays_loaded": False, "external_test_arrays_loaded": False,
    }
    _write_json(output / "summary.json", result)
    _write_json(output / "progress.json", {"status": "complete", "optimizer_update": training["optimizer_updates"],
                                           "best_update": best_update, "best_score": best_score})
    return result


def select_winner(candidates: list[dict], frozen_order: list[str]) -> dict:
    if len(candidates) != len(frozen_order) or set(row["name"] for row in candidates) != set(frozen_order):
        raise ValueError("all frozen candidates must be reported before selecting a winner")
    if any(row["status"] != "complete" or not np.isfinite(row["best_validation_score"]) for row in candidates):
        raise ValueError("every frozen candidate must complete with a finite validation score")
    ranking = sorted(candidates, key=lambda row: (row["best_validation_score"], frozen_order.index(row["name"])))
    return {
        "winner": ranking[0]["name"], "checkpoint_path": ranking[0]["checkpoint_path"],
        "checkpoint_sha256": ranking[0]["checkpoint_sha256"],
        "best_optimizer_update": ranking[0]["best_optimizer_update"],
        "validation_selection_score": ranking[0]["best_validation_score"],
        "ranking": [{"name": row["name"], "score": row["best_validation_score"]} for row in ranking],
        "selection_scope": "common train-development validation rule only; no calibration/test/external outcomes",
        "frozen_for_heldout_evaluation": False,
        "next_step": "Review and separately freeze the selected artifact before any held-out evaluation",
    }


def run_gradient_development(protocol_path: Path, output: Path, *, root: Path | None = None) -> dict:
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for the gradient development study")
    root = Path(__file__).resolve().parents[3] if root is None else root.resolve()
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    validate_protocol(protocol)
    if output.exists() and any(output.iterdir()):
        raise ValueError("development output must be new or empty; existing results/checkpoints are preserved")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(protocol["training"]["torch_threads"])
    provenance = environment_record(root)
    protocol_hash = _digest(protocol_bytes)
    _atomic_bytes(output / "protocol.json", protocol_bytes)
    _write_json(output / "status.json", {"status": "running", "protocol_sha256": protocol_hash,
                                          "started_utc": datetime.now(timezone.utc).isoformat()})
    results = []
    try:
        roles, input_audit = load_development_roles(protocol, root)
        _write_json(output / "input_audit.json", input_audit)
        labels, label_records = {}, {}
        for role in ("train", "validation"):
            labels[role], metadata = make_state_labels(roles[role], role, protocol)
            path = output / "labels" / f"{role}.npz"
            _write_npz(path, labels[role])
            label_records[role] = {**metadata, "artifact": str(path), "artifact_sha256": sha256_file(path)}
            print(json.dumps({"phase": "teacher_labels", "role": role,
                              "examples": metadata["all_state_examples"]}), flush=True)
        inverse, inverse_record = make_inverse_cases(labels["validation"], protocol)
        inverse_path = output / "labels" / "validation_inverse_cases.npz"
        _write_npz(inverse_path, inverse)
        label_records["validation_inverse_cases"] = {**inverse_record, "artifact": str(inverse_path),
                                                     "artifact_sha256": sha256_file(inverse_path)}
        _write_json(output / "teacher_labels.json", label_records)
        n_raw = len(labels["train"]["models"]) // 2
        normalization = WavefieldNormalization.from_arrays(
            labels["train"]["models"][:n_raw], labels["train"]["frequencies_hz"][:n_raw],
            labels["train"]["fields"][:n_raw],
        )
        _write_json(output / "normalization.json", {"source_role": "raw train only", **asdict(normalization)})
        for candidate in protocol["candidates"]:
            result = train_candidate(
                candidate, labels["train"], labels["validation"], inverse, normalization, protocol,
                output / "candidates" / candidate["name"], protocol_sha256=protocol_hash,
                input_audit=input_audit, label_records=label_records, source_sha256=provenance["source_sha256"],
            )
            results.append(result)
            _write_json(output / "completed_candidates.json", results)
        _, end_audit = load_development_roles(protocol, root)
        inputs_unchanged = end_audit["input_file_sha256"] == input_audit["input_file_sha256"]
        source_end = source_digest(root)
        unchanged = source_end == provenance["source_sha256"] and protocol_path.read_bytes() == protocol_bytes
        selection = select_winner(results, [row["name"] for row in protocol["candidates"]])
        selection["source_and_inputs_unchanged"] = unchanged and inputs_unchanged
        _write_json(output / "selection.json", selection)
        summary = {
            "schema_version": 1, "status": "complete" if unchanged and inputs_unchanged else "stale-source-or-inputs",
            "protocol_sha256": protocol_hash, "elapsed_seconds": time.perf_counter() - started,
            "input_audit": input_audit, "normalization": asdict(normalization), "teacher_labels": label_records,
            "candidates": results, "selection": selection, "provenance": provenance,
            "source_sha256_at_end": source_end, "source_unchanged_during_run": unchanged,
            "inputs_unchanged_during_run": inputs_unchanged,
            "calibration_or_test_arrays_loaded": False, "external_test_arrays_loaded": False,
            "limitations": [
                "This is development/model-selection evidence on few validation parents, not independent confirmation.",
                "Smoothing and derivative supervision are established baselines; no new certification theorem is claimed.",
                "Equal update budgets do not imply equal wall time, epochs, or label-generation cost.",
                "All source/receiver/frequency observations are generated by the small reference PDE, not measured field data.",
                "The selected checkpoint requires a separate freeze before held-out calibration/test/external evaluation.",
            ],
        }
        _write_json(output / "summary.json", summary)
        _write_json(output / "status.json", {"status": summary["status"], "protocol_sha256": protocol_hash})
        return summary
    except BaseException as exc:
        _write_json(output / "status.json", {"status": "failed", "protocol_sha256": protocol_hash,
                                              "error_type": type(exc).__name__, "error": str(exc),
                                              "completed_candidate_names": [row["name"] for row in results]})
        raise
