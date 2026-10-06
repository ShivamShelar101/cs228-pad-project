"""
Makes a tiny fake copy of the CelebA-Spoof folder layout (images, *_BB.txt boxes,
metas/intra_test/*.json) so the real prepare script can be tested in seconds.

    python -m data.make_fake_celeba_tree --dst ./fake_celeba
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def build(dst, split, n_subjects, per_class, rng, labels):
    for s in range(n_subjects):
        sid = f"{1000 + s}" if split == "train" else f"{9000 + s}"
        for cls, types in (("live", [0]), ("spoof", [1, 7, 4])):
            d = Path(dst) / "Data" / split / sid / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                st = types[i % len(types)]
                base = 90 if cls == "live" else 160
                arr = np.clip(rng.normal(base, 40, (240, 200, 3)), 0, 255).astype("uint8")
                name = f"{i:06d}"
                Image.fromarray(arr).save(d / f"{name}.jpg")
                (d / f"{name}_BB.txt").write_text("50 40 120 140 0.99\n")
                lab = [0] * 44
                lab[40] = st
                lab[43] = 0 if cls == "live" else 1
                labels[f"Data/{split}/{sid}/{cls}/{name}.jpg"] = lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="./fake_celeba")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for split, n_sub in (("train", 12), ("test", 4)):
        labels = {}
        build(args.dst, split, n_sub, 12, rng, labels)
        meta = Path(args.dst) / "metas" / "intra_test"
        meta.mkdir(parents=True, exist_ok=True)
        for g in range(300):  # labels for images that are NOT on disk, like a partial Kaggle copy
            labels[f"Data/{split}/{50000 + g}/spoof/{g:06d}.jpg"] = [0] * 40 + [7, 0, 0, 1]
        (meta / f"{split}_label.json").write_text(json.dumps(labels))
    print("fake CelebA-Spoof tree written to", args.dst)


if __name__ == "__main__":
    main()
