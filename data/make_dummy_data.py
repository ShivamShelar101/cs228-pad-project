"""
Creates a tiny fake PAD dataset (random images) in the same ImageFolder layout
the real pipeline uses, so you can test every script end to end in minutes,
before the real CelebA-Spoof download finishes.

    python -m data.make_dummy_data --dst ./data_dummy
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="./data_dummy")
    ap.add_argument("--per_class", type=int, default=24)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    for split, n in [("train", args.per_class), ("val", 8), ("test", 8)]:
        for cls in ("live", "spoof"):
            out = Path(args.dst) / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                base = 90 if cls == "live" else 160  # crude signal so labels are learnable
                arr = np.clip(rng.normal(base, 40, (96, 96, 3)), 0, 255).astype("uint8")
                Image.fromarray(arr).save(out / f"{i}.jpg")
    print("dummy data written to", args.dst)


if __name__ == "__main__":
    main()
