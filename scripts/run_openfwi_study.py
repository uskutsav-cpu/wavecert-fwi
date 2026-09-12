from __future__ import annotations

import argparse
import json

from wavecert.experiments.openfwi_study import run_openfwi_study


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--velocity-file", required=True)
    p.add_argument("--max-models", type=int, default=100)
    args = p.parse_args()
    print(
        json.dumps(
            run_openfwi_study(
                velocity_file=args.velocity_file,
                max_models=args.max_models,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
