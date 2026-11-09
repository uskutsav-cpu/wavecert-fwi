import csv

import pytest

from wavecert.data.splits import audit_dataset_roles, audit_split_manifest, grouped_split_manifest


def _manifest():
    return [
        {
            "slice_id": f"{geology}-{parent}-slice-{section}",
            "model_id": f"{geology}-{parent}",
            "model_type": geology,
            "split": "test_out_dist" if geology == "ood" else "train",
        }
        for geology in ["a", "b", "ood"]
        for parent in range(8)
        for section in range(3)
    ]


def test_grouped_split_is_order_invariant_and_keeps_whole_parents_isolated():
    rows = _manifest()
    forward = grouped_split_manifest(rows, seed=7)
    reverse = grouped_split_manifest(list(reversed(rows)), seed=7)
    assert forward == reverse
    audit = audit_split_manifest(forward)
    assert audit["parent_isolation_passed"]
    assert set(audit["roles"]) == {"train", "validation", "calibration", "test_id", "test_ood"}
    for row in forward:
        assert row["source_split"] == ("test_out_dist" if row["model_type"] == "ood" else "train")
        if row["model_type"] == "ood":
            assert row["role"] == "test_ood"


def test_slice_isolation_does_not_imply_parent_isolation():
    rows = [
        {"role": "calibration", "slice_id": "cross-section-1", "model_id": "same-parent"},
        {"role": "test_id", "slice_id": "cross-section-2", "model_id": "same-parent"},
    ]
    audit = audit_split_manifest(rows)
    assert audit["slice_isolation_passed"]
    assert not audit["parent_isolation_passed"]
    assert audit["pairwise_overlap"][0]["overlapping_parent_models"] == ["same-parent"]


def test_one_parent_cannot_have_conflicting_source_ood_membership():
    rows = _manifest()
    rows[0]["split"] = "test_out_dist"
    with pytest.raises(ValueError, match="conflicting"):
        grouped_split_manifest(rows)


def test_dataset_roles_require_dedicated_calibration_and_reject_parent_overlap(tmp_path):
    rows = grouped_split_manifest(_manifest())
    for role in {row["role"] for row in rows}:
        path = tmp_path / f"{role}_metadata.csv"
        selected = [row for row in rows if row["role"] == role]
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
    assert audit_dataset_roles(tmp_path)["parent_isolation_passed"]
    (tmp_path / "calibration_metadata.csv").unlink()
    with pytest.raises(ValueError, match="separate train, validation"):
        audit_dataset_roles(tmp_path)
