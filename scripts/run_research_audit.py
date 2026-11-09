"""Run without changing archived results: PYTHONPATH=src python scripts/run_research_audit.py."""

from __future__ import annotations

import argparse
import json

from wavecert.statistics.audit import run_research_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="results/research_audit")
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    result = run_research_audit(repo_root=args.repo_root, output_dir=args.output, seed=args.seed)
    print(json.dumps({
        "parent_isolation_passed": result["split_audit"]["parent_isolation_passed"],
        "matched_cost_cosine_effects": result["matched_cost_cosine_effects"],
        "final_dataset": result["final_dataset"],
    }, indent=2))


if __name__ == "__main__":
    main()
