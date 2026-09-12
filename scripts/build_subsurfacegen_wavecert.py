from __future__ import annotations

import argparse
import json

from wavecert.data.real_geology import (
    RealGeologyBuildConfig,
    build_subsurfacegen_wavecert_dataset,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selected-root", default="data/external/subsurfacegen-selected")
    p.add_argument("--output-root", default="data/external/subsurfacegen-wavecert-v2")
    p.add_argument("--train-models", type=int, default=800)
    p.add_argument("--validation-models", type=int, default=200)
    p.add_argument("--sampled-pairs", action="store_true")
    p.add_argument("--examples-per-model", type=int, default=4)
    args = p.parse_args()

    summary = build_subsurfacegen_wavecert_dataset(
        selected_root=args.selected_root,
        output_root=args.output_root,
        train_models=args.train_models,
        validation_models=args.validation_models,
        config=RealGeologyBuildConfig(
            all_source_frequency_pairs=not args.sampled_pairs,
            examples_per_model=args.examples_per_model,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
