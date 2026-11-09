from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from wavecert.data.splits import audit_split_manifest, grouped_split_manifest


def select_subsurfacegen_slices(
    *,
    index_root: str | Path,
    output_root: str | Path,
    train_count: int = 1000,
    id_count: int = 50,
    ood_count: int = 50,
    seed: int = 20260911,
    download: bool = False,
    repo_id: str = "subsurfacegen/field-scale-dataset",
) -> pd.DataFrame:
    """Select reproducible lightweight SubsurfaceGen velocity slices.

    Only `slice` HDF5 files are selected; no data download occurs by default.
    Source benchmark labels are preserved in ``source_split``. The exported
    ``split``/``role`` labels define a new parent-isolated research re-split,
    not an official benchmark evaluation protocol.
    """

    index_root = Path(index_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    requests = [
        ("train", index_root / "data/train.parquet", train_count, seed),
        ("test_in_dist", index_root / "data/test_in_dist.parquet", id_count, seed + 1),
        ("test_out_dist", index_root / "data/test_out_dist.parquet", ood_count, seed + 2),
    ]

    rows: list[dict] = []
    for split, path, count, random_state in requests:
        frame = pd.read_parquet(path)
        frame = frame[frame["data_type"].eq("slice")].drop_duplicates("slice_id")
        if len(frame) < count:
            raise ValueError(f"{split}: requested {count} slices, only {len(frame)} available")
        selected = frame.sample(n=count, random_state=random_state)
        for _, row in selected.iterrows():
            record = {
                "split": split,
                "slice_id": row["slice_id"],
                "model_id": row["model_id"],
                "model_type": row["model_type"],
                "file_path": row["file_path"],
            }
            rows.append(record)

    manifest = pd.DataFrame(grouped_split_manifest(rows, seed=seed))
    audit = audit_split_manifest(manifest.to_dict("records"))
    if not audit["parent_isolation_passed"]:
        raise RuntimeError("selected manifest failed parent isolation")
    manifest_path = output_root / "selected_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    audit.update({
        "seed": seed,
        "source_repository": repo_id,
        "protocol": "new parent-isolated research re-split; original labels retained as source_split",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_index_sha256": {
            split: hashlib.sha256(path.read_bytes()).hexdigest()
            for split, path, _count, _random_state in requests
        },
        "download_requested": bool(download),
    })
    (output_root / "selected_manifest.audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")

    if download:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("install the 'data' extra and huggingface_hub") from exc
        for filename in manifest["file_path"]:
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=str(filename),
                local_dir=output_root,
            )

    return manifest
