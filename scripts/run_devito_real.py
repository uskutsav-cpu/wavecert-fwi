from __future__ import annotations

import argparse
import json

from wavecert.experiments.devito_study import run_devito_study


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--marmousi", required=True)
    p.add_argument("--iterations", type=int, default=5)
    args = p.parse_args()
    print(
        json.dumps(
            run_devito_study(
                marmousi_path=args.marmousi,
                optimizer_iterations=args.iterations,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
