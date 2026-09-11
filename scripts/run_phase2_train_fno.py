from __future__ import annotations

import argparse
import json

from wavecert.experiments.train_fno import run_training
from wavecert.surrogates.fno import FNOConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--train", type=int, default=384)
    p.add_argument("--val", type=int, default=72)
    p.add_argument("--test", type=int, default=96)
    p.add_argument("--shape", type=int, default=24)
    p.add_argument("--width", type=int, default=28)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260910)
    args = p.parse_args()
    summary = run_training(
        shape=(args.shape, args.shape),
        n_train=args.train,
        n_val=args.val,
        n_test=args.test,
        epochs=args.epochs,
        seed=args.seed,
        config=FNOConfig(modes_z=args.modes, modes_x=args.modes, width=args.width, depth=args.depth),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
