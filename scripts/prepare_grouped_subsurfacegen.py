"""Prepare a small, revision-pinned, parent-isolated SubsurfaceGen research set.

Only velocity slices are downloaded, never the full 3-D volumes/wavefields.
This creates a new research re-split, not the official benchmark protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path

import pandas as pd

from wavecert.data.splits import audit_split_manifest, grouped_split_manifest
from wavecert.provenance import sha256_file


def _download(url: str, path: Path, maximum_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > maximum_bytes:
                    raise RuntimeError(f"download exceeds {maximum_bytes} byte cap: {url}")
                output.write(chunk)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/external/subsurfacegen-grouped")
    parser.add_argument(
        "--revision",
        help="immutable Hugging Face dataset commit; latest resolved and recorded if omitted",
    )
    parser.add_argument("--slices-per-parent", type=int, default=2)
    parser.add_argument("--ood-slices", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download selected velocity slices (manifest-only otherwise)",
    )
    args = parser.parse_args()
    if args.slices_per_parent < 1 or args.ood_slices < 1:
        parser.error("slice counts must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    repo = "subsurfacegen/field-scale-dataset"
    revision = args.revision
    if revision is None:
        with urllib.request.urlopen(
            f"https://huggingface.co/api/datasets/{repo}", timeout=45
        ) as response:
            revision = json.load(response)["sha"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        parser.error("revision must be a 40-character lowercase commit hash")
    records = []
    for split in ("train", "test_out_dist"):
        index = output / "index" / f"{split}.parquet"
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/data/{split}.parquet"
        _download(url, index, 5_000_000)
        frame = pd.read_parquet(index)
        frame = frame[frame["data_type"].eq("slice")].drop_duplicates("slice_id")
        for parent, group in frame.groupby("model_id", sort=True):
            count = args.ood_slices if split == "test_out_dist" else args.slices_per_parent
            chosen = group.sort_values("slice_id").sample(
                n=min(count, len(group)), random_state=args.seed
            )
            for _, row in chosen.iterrows():
                records.append(
                    {
                        "split": split,
                        "slice_id": str(row["slice_id"]),
                        "model_id": str(parent),
                        "model_type": str(row["model_type"]),
                        "file_path": str(row["file_path"]),
                    }
                )
    for record in records:
        record["split_stratum"] = "common-ID-parent-pool"
    records = grouped_split_manifest(
        records,
        seed=args.seed,
        strata_key="split_stratum",
        role_fractions={"train": 0.4, "validation": 0.1, "calibration": 0.3, "test_id": 0.2},
    )
    for record in records:
        record.pop("split_stratum")
    manifest = output / "selected_manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    sources = []
    if args.download:
        for i, row in enumerate(records):
            path = output / row["file_path"]
            if not path.resolve().is_relative_to(output.resolve()):
                raise ValueError(
                    "remote file_path must remain inside the selected output directory"
                )
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{row['file_path']}"
            _download(url, path, 10_000_000)
            sources.append(
                {
                    "file_path": row["file_path"],
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
            print(f"downloaded {i + 1}/{len(records)}", flush=True)
    provenance = {
        "dataset": repo,
        "revision": revision,
        "license": "CC-BY-4.0",
        "citation": "Stitt et al. (2026), SubsurfaceGen, arXiv:2605.30541",
        "scope": "new whole-parent research re-split of selected synthetic velocity slices; not official benchmark split or field data",
        "assignment": "unstratified parent assignment from the common ID pool; OOD source split reserved",
        "seed": args.seed,
        "manifest_sha256": sha256_file(manifest),
        "index_sha256": {
            p.name: sha256_file(p) for p in sorted((output / "index").glob("*.parquet"))
        },
        "audit": audit_split_manifest(records),
        "downloaded_sources": sources,
    }
    (output / "selection_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"revision": revision, "roles": provenance["audit"]["roles"]}, indent=2))


if __name__ == "__main__":
    main()
