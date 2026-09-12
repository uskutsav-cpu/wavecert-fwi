from __future__ import annotations

import json

from wavecert.experiments.baseline_study import run_baseline_study

if __name__ == "__main__":
    print(json.dumps(run_baseline_study(), indent=2))
