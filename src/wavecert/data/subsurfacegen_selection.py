from __future__ import annotations

from pathlib import Path

import pandas as pd


def select_subsurfacegen_slices(
    *,
    index_root: str | Path,
    output_root: str | Path,
    train_count: int = 1000,
    id_count: int = 50,
    ood_count: int = 50,
    seed: int = 20260911,
    download: bool = True,
    repo_id: str = "subsurfacegen/field-scale-dataset",
) -> pd.DataFrame:
    """Select reproducible lightweight SubsurfaceGen velocity slices.

    Only `slice` HDF5 files are selected; the much larger precomputed wavefields
    and shot gathers are not required for WaveCert's controlled Helmholtz study.
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

    manifest = pd.DataFrame(rows)
    manifest_path = output_root / "selected_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

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
