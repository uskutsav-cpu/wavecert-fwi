from __future__ import annotations

import argparse
import json

from wavecert.experiments.failure_study import run_failure_study


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=int, default=80)
    p.add_argument("--seed", type=int, default=20260930)
    args = p.parse_args()
    summary = run_failure_study(n_cases=args.cases, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
