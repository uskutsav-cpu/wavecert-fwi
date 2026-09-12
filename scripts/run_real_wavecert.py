from __future__ import annotations

import json

from wavecert.experiments.real_wavecert import run_real_wavecert_study

if __name__ == "__main__":
    print(json.dumps(run_real_wavecert_study(), indent=2))
