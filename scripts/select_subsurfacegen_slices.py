from __future__ import annotations

import argparse

from wavecert.data.subsurfacegen_selection import select_subsurfacegen_slices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-root", default="data/external/subsurfacegen-index")
    parser.add_argument("--output-root", default="data/external/subsurfacegen-selected-v2")
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--id", type=int, default=50)
    parser.add_argument("--ood", type=int, default=50)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    manifest = select_subsurfacegen_slices(
        index_root=args.index_root,
        output_root=args.output_root,
        train_count=args.train,
        id_count=args.id,
        ood_count=args.ood,
        download=not args.no_download,
    )
    print(manifest.groupby(["split", "model_type"]).size())


if __name__ == "__main__":
    main()
