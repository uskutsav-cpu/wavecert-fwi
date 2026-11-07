"""Evaluate the frozen external-stress protocol; no training or tuning options."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wavecert.experiments.openfwi_stress import run_openfwi_stress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/openfwi_external_stress.json")
    parser.add_argument("--data", default="data/external/openfwi_stress")
    parser.add_argument("--output", default="results/openfwi_external_stress")
    args = parser.parse_args()
    summary = run_openfwi_stress(protocol_path=args.protocol, data_dir=args.data,
        output_dir=args.output, repository_root=Path(__file__).resolve().parents[1])
    print(json.dumps({"study_status": summary["study_status"], "n_blocks": summary["n_blocks"],
        "n_model_checkpoint_evaluations": summary["n_model_checkpoint_evaluations"]}, indent=2))


if __name__ == "__main__":
    main()
