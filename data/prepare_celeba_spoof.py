"""
Builds the ImageFolder dataset this project trains on (train/val/test x live/spoof)
from raw CelebA-Spoof, with every image cropped to the face.

- Only images that really exist on disk are used (some Kaggle copies are partial
  while the label files still list the full dataset).
- Faces are cropped with CelebA-Spoof's *_BB.txt boxes, because the demo app sends
  face crops.
- Crops that would run off the photo are skipped (black bars were 2x as common in spoof images).
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


def crop_face(img, bb_path, bbox_inc=1.5, allow_padding=False):
    """Square crop around the face box (box coords are given on a 224x224 grid).
    Returns (image, None) or (None, reason). Crops that run off the photo are rejected
    unless allow_padding=True, because black padding is far more common in spoof images
    and would let a model cheat."""
    W, H = img.size
    try:
        with open(bb_path) as f:
            x, y, w, h = [float(v) for v in f.readline().strip().split(" ")[:4]]
    except Exception:
        return None, "bad_box"
    x, w = x * W / 224, w * W / 224
    y, h = y * H / 224, h * H / 224
    side = max(w, h) * bbox_inc
    xc, yc = x + w / 2, y + h / 2
    box = (int(xc - side / 2), int(yc - side / 2), int(xc + side / 2), int(yc + side / 2))
    if not allow_padding and (box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H):
        return None, "padding"
    return img.convert("RGB").crop(box).resize((SIZE, SIZE)), None


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


def pools_for(found, subjects, rng):
    """Shuffled list of candidate image keys per class."""
    pools = {}
    for cls in ("live", "spoof"):
        pool = [k for s in subjects for k in found[s][cls]]
        rng.shuffle(pool)
        pools[cls] = pool
    return pools


def write(root, pools, dst, split, n, allow_padding):
    """Write up to n accepted crops per class, then trim so both classes have the same count."""
    stats = {"missing": 0, "bad_box": 0, "padding": 0}
    written = {"live": [], "spoof": []}
    for cls, keys in pools.items():
        out_dir = Path(dst) / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for key in keys:
            if len(written[cls]) >= n:
                break
            img_path = root / key
            try:
                img, why = crop_face(Image.open(img_path), str(img_path.with_suffix("")) + "_BB.txt",
                                     allow_padding=allow_padding)
            except FileNotFoundError:
                stats["missing"] += 1
                continue
            except Exception:
                img, why = None, "bad_box"
            if img is None:
                stats[why] += 1
                continue
            subject = key.split("/")[2]
            out = out_dir / f"{subject}_{img_path.stem}.jpg"
            img.save(out, quality=95)
            written[cls].append(out)
    m = min(len(written["live"]), len(written["spoof"]))
    for cls in written:
        for extra in written[cls][m:]:
            extra.unlink()
    print(f"{split}: kept {m} per class | skipped {stats['padding']} crops that ran off the photo, "
          f"{stats['bad_box']} without a face box, {stats['missing']} missing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--root", default=None, help="folder holding metas/ and Data/ (auto-detected if omitted)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--per_class", type=int, default=3000, help="train+val images per class")
    ap.add_argument("--test_per_class", type=int, default=600)
    ap.add_argument("--val_frac", type=float, default=0.15, help="fraction of train SUBJECTS used for val")
    ap.add_argument("--spoof_types", type=int, nargs="+", default=PRINT_REPLAY)
    ap.add_argument("--oversample", type=float, default=3.5,
                    help="scan this many times more candidates than needed, since many are rejected")
    ap.add_argument("--allow_padding", action="store_true", help="keep crops that run off the photo (black bars)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.root) if args.root else find_root(args.src)
    meta = root / "metas" / "intra_test"
    print("dataset root:", root)
    types = set(args.spoof_types)

    train_found = scan_split(root, "train", meta / "train_label.json", types, int(args.per_class * args.oversample), rng)
    test_found = scan_split(root, "test", meta / "test_label.json", types, int(args.test_per_class * args.oversample), rng)
    if not train_found:
        raise SystemExit("No training images found on disk. Check --root.")

    subjects = sorted(train_found)
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * args.val_frac))
    val_subjects, train_subjects = subjects[:n_val], subjects[n_val:]
    n_train = int(args.per_class * (1 - args.val_frac))
    n_val_imgs = args.per_class - n_train

    write(root, pools_for(train_found, train_subjects, rng), args.dst, "train", n_train, args.allow_padding)
    write(root, pools_for(train_found, val_subjects, rng), args.dst, "val", n_val_imgs, args.allow_padding)
    if test_found:
        write(root, pools_for(test_found, sorted(test_found), rng), args.dst, "test", args.test_per_class,
              args.allow_padding)
    else:
        print("WARNING: no test images on disk; hold some train subjects out as test instead (tell me).")
    print("done ->", args.dst)


if __name__ == "__main__":
    main()
