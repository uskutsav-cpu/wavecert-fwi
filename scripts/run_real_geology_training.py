from __future__ import annotations

import json

from wavecert.experiments.real_geology_training import run_real_geology_training

if __name__ == "__main__":
    print(json.dumps(run_real_geology_training(), indent=2))
