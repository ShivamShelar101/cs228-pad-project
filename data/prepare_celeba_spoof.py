"""
Builds the ImageFolder dataset this project trains on (train/val/test x live/spoof)
from raw CelebA-Spoof, with every image cropped to the face.

- Only images that really exist on disk are used (some Kaggle copies are partial
  while the label files still list the full dataset).
- Faces are cropped with CelebA-Spoof's *_BB.txt boxes, because the demo app sends
  face crops.
- val comes from different people than train; test comes from the official test split.

Layout (found automatically, or pass --root):
    <root>/metas/intra_test/train_label.json  (+ test_label.json)
    <root>/Data/{train,test}/<subject>/{live,spoof}/<image>.jpg|png  (+ <image>_BB.txt)
Label list per image: index 40 = spoof type (0 live, 1 photo, 2 poster, 3 A4,
4 face mask, 5 upper-body mask, 6 region mask, 7 PC, 8 pad, 9 phone, 10 3D mask).

Usage (Kaggle):
    python -m data.prepare_celeba_spoof --src /kaggle/input --dst ./data_pad \
        --per_class 3000 --test_per_class 600
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

SIZE = 224
PRINT_REPLAY = [1, 2, 3, 7, 8, 9]  # photo, poster, A4 (print) + PC, pad, phone (replay)
IMG_EXT = (".jpg", ".jpeg", ".png")


def find_root(src, max_depth=7):
    """Folder that contains metas/intra_test (Data/ sits next to it). Shallow globbing only."""
    for depth in range(max_depth):
        hits = glob.glob(os.path.join(str(src), *(["*"] * depth), "metas", "intra_test"))
        if hits:
            return Path(hits[0]).parent.parent
    raise SystemExit(f"Could not find metas/intra_test under {src}. Pass --root <folder that holds metas/ and Data/>.")


def crop_face(img, bb_path, bbox_inc=1.5):
    """Square crop around the face box (box coords are given on a 224x224 grid)."""
    W, H = img.size
    try:
        with open(bb_path) as f:
            x, y, w, h = [float(v) for v in f.readline().strip().split(" ")[:4]]
    except Exception:
        return None
    x, w = x * W / 224, w * W / 224
    y, h = y * H / 224, h * H / 224
    side = max(w, h) * bbox_inc
    xc, yc = x + w / 2, y + h / 2
    box = (int(xc - side / 2), int(yc - side / 2), int(xc + side / 2), int(yc + side / 2))
    return img.convert("RGB").crop(box).resize((SIZE, SIZE))  # out-of-frame area becomes black


def scan_split(root, split, label_json, spoof_types, need, rng):
    """Walk subject folders (shuffled) and collect images that exist, until each class has `need`."""
    data_dir = root / "Data" / split
    if not data_dir.is_dir():
        print(f"  (no folder {data_dir})")
        return {}
    labels = None
    subjects = sorted(e.name for e in os.scandir(data_dir) if e.is_dir())
    rng.shuffle(subjects)
    found = defaultdict(lambda: {"live": [], "spoof": []})
    counts = {"live": 0, "spoof": 0}
    for n, s in enumerate(subjects):
        for cls in ("live", "spoof"):
            d = data_dir / s / cls
            if not d.is_dir():
                continue
            for name in os.listdir(d):
                if not name.lower().endswith(IMG_EXT):
                    continue
                key = f"Data/{split}/{s}/{cls}/{name}"
                if cls == "spoof":
                    if labels is None:
                        print(f"  loading {label_json.name} ...")
                        with open(label_json) as f:
                            labels = json.load(f)
                    lab = labels.get(key)
                    if lab is None or int(lab[40]) not in spoof_types:
                        continue
                found[s][cls].append(key)
                counts[cls] += 1
        if n % 500 == 0:
            print(f"  {split}: scanned {n}/{len(subjects)} subjects, live={counts['live']} spoof={counts['spoof']}")
        if counts["live"] >= need and counts["spoof"] >= need:
            break
    print(f"  {split}: found live={counts['live']} spoof={counts['spoof']} images on disk")
    return found


def pick(found, subjects, n, rng):
    """Balanced pick: same number per class, at most n."""
    pools = {}
    for cls in ("live", "spoof"):
        pool = [k for s in subjects for k in found[s][cls]]
        rng.shuffle(pool)
        pools[cls] = pool
    n = min(n, len(pools["live"]), len(pools["spoof"]))
    return {cls: pools[cls][:n] for cls in pools}


def write(root, picks, dst, split):
    stats = {"written": 0, "missing": 0, "bad_box": 0}
    for cls, keys in picks.items():
        out_dir = Path(dst) / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for key in keys:
            img_path = root / key
            try:
                img = crop_face(Image.open(img_path), str(img_path.with_suffix("")) + "_BB.txt")
            except FileNotFoundError:
                stats["missing"] += 1
                continue
            except Exception:
                img = None
            if img is None:
                stats["bad_box"] += 1
                continue
            subject = key.split("/")[2]
            img.save(out_dir / f"{subject}_{img_path.stem}.jpg", quality=95)
            stats["written"] += 1
    print(f"{split}: wrote {stats['written']} images, skipped {stats['missing']} missing, {stats['bad_box']} without a usable face box")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--root", default=None, help="folder holding metas/ and Data/ (auto-detected if omitted)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--per_class", type=int, default=3000, help="train+val images per class")
    ap.add_argument("--test_per_class", type=int, default=600)
    ap.add_argument("--val_frac", type=float, default=0.15, help="fraction of train SUBJECTS used for val")
    ap.add_argument("--spoof_types", type=int, nargs="+", default=PRINT_REPLAY)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.root) if args.root else find_root(args.src)
    meta = root / "metas" / "intra_test"
    print("dataset root:", root)
    types = set(args.spoof_types)

    train_found = scan_split(root, "train", meta / "train_label.json", types, int(args.per_class * 1.2), rng)
    test_found = scan_split(root, "test", meta / "test_label.json", types, args.test_per_class, rng)
    if not train_found:
        raise SystemExit("No training images found on disk. Check --root.")

    subjects = sorted(train_found)
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * args.val_frac))
    val_subjects, train_subjects = subjects[:n_val], subjects[n_val:]
    n_train = int(args.per_class * (1 - args.val_frac))
    n_val_imgs = args.per_class - n_train

    write(root, pick(train_found, train_subjects, n_train, rng), args.dst, "train")
    write(root, pick(train_found, val_subjects, n_val_imgs, rng), args.dst, "val")
    if test_found:
        write(root, pick(test_found, sorted(test_found), args.test_per_class, rng), args.dst, "test")
    else:
        print("WARNING: no test images on disk; hold some train subjects out as test instead (tell me).")
    print("done ->", args.dst)


if __name__ == "__main__":
    main()
