from __future__ import annotations

import json

from wavecert.experiments.heuristic_baselines import run_heuristic_baselines

if __name__ == "__main__":
    print(json.dumps(run_heuristic_baselines(), indent=2))
