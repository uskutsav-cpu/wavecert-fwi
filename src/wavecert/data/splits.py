"""Parent-model split construction and leakage audits for geological slices.

Splitting slices independently can put correlated cross-sections of the same
Earth model into calibration and evaluation. These utilities retain the source
split as provenance and assign every parent entirely to one analysis role.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

DEFAULT_ROLE_FRACTIONS = {"train": 0.55, "validation": 0.15, "calibration": 0.15, "test_id": 0.15}


def _records(records: Sequence[Mapping], keys: Sequence[str]) -> list[dict]:
    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError("manifest must not be empty")
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value is None or not str(value).strip() or str(value).strip().lower() in {"nan", "none", "null"}:
                raise ValueError(f"manifest requires nonempty {key!r} in every row")
    return rows


def audit_split_manifest(
    records: Sequence[Mapping],
    *,
    role_key: str = "role",
    group_key: str = "model_id",
    unit_key: str = "slice_id",
) -> dict:
    """Report exact unit overlap separately from correlated-parent overlap."""

    rows = _records(records, [role_key, group_key, unit_key])
    by_role: dict[str, list[dict]] = defaultdict(list)
    parents_by_unit: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_role[str(row[role_key])].append(row)
        parents_by_unit[str(row[unit_key])].add(str(row[group_key]))
    conflicting = sorted(unit for unit, parents in parents_by_unit.items() if len(parents) > 1)
    roles = {}
    for role, group in sorted(by_role.items()):
        parent_counts = Counter(str(row[group_key]) for row in group)
        roles[role] = {
            "rows": len(group),
            "unique_slices": len({str(row[unit_key]) for row in group}),
            "unique_parent_models": len(parent_counts),
            "parent_models": sorted(parent_counts),
            "rows_per_parent": dict(sorted(parent_counts.items())),
            "geology_counts": dict(sorted(Counter(str(row.get("model_type", "unknown")) for row in group).items())),
        }
    overlaps = []
    for left, right in itertools.combinations(sorted(by_role), 2):
        units_left = {str(row[unit_key]) for row in by_role[left]}
        units_right = {str(row[unit_key]) for row in by_role[right]}
        parent_left = {str(row[group_key]) for row in by_role[left]}
        parent_right = {str(row[group_key]) for row in by_role[right]}
        overlaps.append({
            "roles": [left, right],
            "overlapping_slices": sorted(units_left & units_right),
            "overlapping_parent_models": sorted(parent_left & parent_right),
        })
    return {
        "roles": roles,
        "pairwise_overlap": overlaps,
        "conflicting_slice_parent_ids": conflicting,
        "slice_isolation_passed": not conflicting and all(not p["overlapping_slices"] for p in overlaps),
        "parent_isolation_passed": not conflicting and all(not p["overlapping_parent_models"] for p in overlaps),
        "exchangeability_established": False,
        "statistical_scope": "Parent isolation is necessary but not sufficient for pooled conformal coverage; fixed geology quotas and unequal within-parent sampling require separate justification.",
        "scope": "Manifest isolation only; does not prove independent sampling, unseen data, or checkpoint provenance.",
    }


def grouped_split_manifest(
    records: Sequence[Mapping],
    *,
    role_fractions: Mapping[str, float] | None = None,
    group_key: str = "model_id",
    unit_key: str = "slice_id",
    strata_key: str | None = None,
    seed: int = 20260912,
    ood_source_split: str = "test_out_dist",
) -> list[dict]:
    """Create a deterministic, whole-parent research re-split with a held-out OOD role.

    Source rows with ``split=ood_source_split`` retain the ``test_ood`` role.
    Other source splits are pooled and re-split by parent without stratification
    by default. An explicit ``strata_key`` enables a stratified research split.
    Original splits are retained in ``source_split``: the result is a *new*
    research protocol and must not be called the official benchmark split.
    Assignment is invariant to row order and uses parent counts, not slice counts.
    Stratified role assignment does not by itself establish exchangeability
    for pooled conformal calibration across geology types.
    """

    rows = _records(records, [group_key, unit_key, "split"] + ([] if strata_key is None else [strata_key]))
    fractions = dict(DEFAULT_ROLE_FRACTIONS if role_fractions is None else role_fractions)
    if not fractions or "test_ood" in fractions:
        raise ValueError("role fractions must be nonempty and exclude reserved test_ood")
    weights = np.asarray(list(fractions.values()), dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("role fractions must be positive finite numbers summing to one")
    roles = list(fractions)
    groups: dict[str, list[dict]] = defaultdict(list)
    unit_parent: dict[str, str] = {}
    for row in rows:
        parent = str(row[group_key])
        unit = str(row[unit_key])
        if unit in unit_parent and unit_parent[unit] != parent:
            raise ValueError(f"slice {unit!r} maps to multiple parents")
        unit_parent[unit] = parent
        groups[parent].append(row)
    assigned: dict[str, str] = {}
    strata: dict[str, list[str]] = defaultdict(list)
    for parent, group in groups.items():
        geologies = {"all-ID"} if strata_key is None else {str(row[strata_key]) for row in group}
        ood_flags = {str(row["split"]) == ood_source_split for row in group}
        if len(geologies) != 1 or len(ood_flags) != 1:
            raise ValueError(f"parent {parent!r} has conflicting geology or ID/OOD source labels")
        if True in ood_flags:
            assigned[parent] = "test_ood"
        else:
            strata[next(iter(geologies))].append(parent)
    if sum(len(parents) for parents in strata.values()) < len(roles):
        raise ValueError("not enough independent ID parent models for all requested roles")
    for geology, parents in sorted(strata.items()):
        parents.sort(key=lambda p: hashlib.sha256(f"{seed}:{geology}:{p}".encode()).digest())
        quotas = len(parents) * weights
        counts = np.floor(quotas).astype(int)
        for i in sorted(range(len(roles)), key=lambda i: (-(quotas[i] - counts[i]), i))[:len(parents) - int(counts.sum())]:
            counts[i] += 1
        if len(parents) >= len(roles):
            for empty in np.flatnonzero(counts == 0):
                donor = int(np.argmax(counts))
                counts[donor] -= 1
                counts[empty] += 1
        start = 0
        for role, count in zip(roles, counts, strict=True):
            for parent in parents[start:start + int(count)]:
                assigned[parent] = role
            start += int(count)
    missing = set(roles) - set(assigned.values())
    if missing:
        raise ValueError(f"strata too small to populate roles {sorted(missing)}; combine strata or add parents")
    output = []
    seen = set()
    for row in sorted(rows, key=lambda r: (str(r[group_key]), str(r[unit_key]))):
        # A slice manifest contains one row per slice, not one row per block.
        unit = str(row[unit_key])
        if unit in seen:
            continue
        seen.add(unit)
        record = dict(row)
        record["source_split"] = row.get("source_split", row["split"])
        record["role"] = assigned[str(row[group_key])]
        record["split"] = record["role"]
        output.append(record)
    return output


def audit_dataset_roles(
    dataset_dir: str | Path,
    *,
    required_roles: tuple[str, ...] = ("train", "validation", "calibration", "test_id", "test_ood"),
) -> dict:
    """Require metadata that can verify whole-parent isolation for a new study.

    A separate calibration file is required: checkpoint-selection validation
    data cannot silently be reused as split-conformal calibration data.
    The caller must also verify the selected checkpoint was trained using the
    recorded training role; metadata alone cannot establish that provenance.
    """

    dataset_dir = Path(dataset_dir)
    records = []
    for role in required_roles:
        metadata = dataset_dir / f"{role}_metadata.csv"
        if not metadata.is_file():
            raise ValueError(
                f"missing {metadata}; new calibrated runs require separate train, validation "
                "(model selection), calibration, and test metadata with slice_id/model_id"
            )
        with metadata.open(newline="") as stream:
            role_rows = list(csv.DictReader(stream))
        if not role_rows:
            raise ValueError(f"empty dataset role {role!r}")
        records.extend({**row, "role": role} for row in role_rows)
    audit = audit_split_manifest(records)
    if not audit["parent_isolation_passed"]:
        overlapping = [p for p in audit["pairwise_overlap"] if p["overlapping_parent_models"]]
        raise ValueError(f"dataset roles share parent models: {overlapping}")
    return audit


def verify_dataset_provenance(dataset_dir: str | Path) -> dict:
    """Verify built role arrays and metadata match the recorded dataset identity."""

    dataset_dir = Path(dataset_dir)
    path = dataset_dir / "dataset_manifest.json"
    if not path.is_file():
        raise ValueError("dataset_manifest.json is required; rebuild with the parent-isolated dataset builder")
    manifest = json.loads(path.read_text())
    if not manifest.get("conformal_ready", False):
        raise ValueError("dataset is explicitly uncalibrated; rebuild with grouped split mode")

    def digest_file(filename):
        digest = hashlib.sha256()
        with Path(filename).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    checks = [("split_manifest.csv", manifest.get("role_manifest_sha256"))]
    for role in ("train", "validation", "calibration", "test_id", "test_ood"):
        metadata = manifest.get("splits", {}).get(role, {})
        checks.extend([
            (f"{role}.npz", metadata.get("arrays_sha256")),
            (f"{role}_metadata.csv", metadata.get("metadata_sha256")),
        ])
    for filename, expected in checks:
        if expected is None or not (dataset_dir / filename).is_file():
            raise ValueError(f"missing dataset provenance or file for {filename}")
        if digest_file(dataset_dir / filename) != expected:
            raise ValueError(f"dataset content changed since manifest creation: {filename}")
    return {
        "dataset_manifest_sha256": digest_file(path),
        "role_manifest_sha256": manifest["role_manifest_sha256"],
        "verified_content_files": [filename for filename, _ in checks],
        "scope": "file identity only; no proof of unseen test data or independent parent sampling",
    }


def verify_calibration_checkpoint(checkpoint_path: str | Path, provenance: Mapping) -> dict:
    """Require an auditable link between a trained checkpoint and grouped roles.

    This checks recorded provenance, not whether a researcher inspected test
    results during development. Legacy checkpoints without this record can
    still be used for descriptive studies, but not this calibrated workflow.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise ImportError("PyTorch is required to verify the checkpoint record") from exc
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("dataset_manifest_sha256") != provenance["dataset_manifest_sha256"]:
        raise ValueError("checkpoint lacks a matching dataset-manifest record; train it on this grouped dataset")
    if payload.get("training_manifest_sha256") != provenance["role_manifest_sha256"]:
        raise ValueError("checkpoint training-role manifest differs from the supplied dataset")
    if payload.get("calibration_role_used_for_model_selection") is not False:
        raise ValueError("checkpoint does not record exclusion of calibration data from model selection")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "checkpoint_sha256": digest.hexdigest(),
        "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
        "training_manifest_sha256": provenance["role_manifest_sha256"],
        "calibration_role_used_for_model_selection": False,
        "scope": "verified checkpoint metadata link; does not establish unseen evaluation or independent sampling",
    }
