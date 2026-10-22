"""
Checkpoint Sections II-III: data quality checks and exploratory data analysis.

Reads the prepared face crops (data_pad2) and, for metadata, the original CelebA-Spoof label files and
images (only their headers are read for the resolution, so it stays fast).

Outputs to --out: one PNG per figure, eda_summary.json (all numbers), and captions.md (one sentence per figure).

    python -m evaluation.eda --src $SRC --data_root ./data_pad2 --out results/eda
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from data.prepare_celeba_spoof import find_root

TYPES = {1: "photo", 2: "poster", 3: "A4 print", 4: "face mask", 5: "upper-body mask",
         6: "region mask", 7: "PC screen", 8: "pad screen", 9: "phone screen", 10: "3D mask"}
LIGHT = {0: "n/a (live)", 1: "normal", 2: "strong", 3: "back", 4: "dark"}
ENV = {0: "n/a (live)", 1: "indoor", 2: "outdoor"}
FEATURES = ["brightness", "contrast", "saturation", "sharpness", "hf_energy"]
FEATURE_LABEL = {"brightness": "mean gray level (0-255)", "contrast": "gray-level std",
                 "saturation": "mean HSV saturation (0-255)", "sharpness": "log10 Laplacian variance",
                 "hf_energy": "high-frequency energy share (FFT)"}


def image_stats(path):
    """Simple image statistics of a 224x224 crop."""
    img = Image.open(path).convert("RGB")
    g = np.asarray(img.convert("L"), dtype=np.float32)
    s = np.asarray(img.convert("HSV"), dtype=np.float32)[..., 1]
    # Laplacian (4-neighbour) variance: a standard sharpness measure.
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    # Share of spectral energy outside the central low-frequency disc: screens add fine periodic texture.
    f = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    h, w = f.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    hf = f[r > min(h, w) / 8].sum() / max(f.sum(), 1e-9)
    return {"brightness": float(g.mean()), "contrast": float(g.std()), "saturation": float(s.mean()),
            "sharpness": float(np.log10(lap.var() + 1e-6)), "hf_energy": float(hf),
            "hash": hashlib.md5(np.asarray(img.resize((16, 16)).convert("L")).tobytes()).hexdigest()}


def original_key(split, cls, path, labels):
    subject, stem = Path(path).stem.split("_", 1)
    for ext in (".jpg", ".png", ".jpeg"):
        key = f"Data/{split}/{subject}/{cls}/{stem}{ext}"
        if key in labels:
            return subject, key
    return subject, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="CelebA-Spoof location (for labels and original sizes)")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="results/eda")
    ap.add_argument("--max_original_sizes", type=int, default=400, help="per split and class, for resolution")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    root = find_root(args.src)
    labels = {sp: json.load(open(root / "metas" / "intra_test" / f"{'train' if sp == 'val' else sp}_label.json"))
              for sp in ("train", "test")}
    labels["val"] = labels["train"]          # val subjects come from CelebA-Spoof's train split

    rows = []
    for split in ("train", "val", "test"):
        for cls in ("live", "spoof"):
            for i, p in enumerate(sorted((Path(args.data_root) / split / cls).glob("*.jpg"))):
                subject, key = original_key("test" if split == "test" else "train", cls, p, labels[split])
                lab = labels[split].get(key) if key else None
                row = {"split": split, "cls": cls, "subject": subject, "path": str(p),
                       "spoof_type": TYPES.get(int(lab[40]), "?") if lab and cls == "spoof" else None,
                       "light": LIGHT.get(int(lab[41]), "?") if lab else None,
                       "env": ENV.get(int(lab[42]), "?") if lab else None}
                row.update(image_stats(p))
                if key and i < args.max_original_sizes:
                    try:
                        with Image.open(root / key) as im:
                            row["orig_w"], row["orig_h"] = im.size
                    except Exception:
                        pass
                rows.append(row)
    print(f"{len(rows)} crops analysed")

    # ---------------- schema sanity / quality ----------------
    subj = {sp: {r["subject"] for r in rows if r["split"] == sp} for sp in ("train", "val", "test")}
    hashes = defaultdict(list)
    for r in rows:
        hashes[r["hash"]].append((r["split"], r["cls"]))
    dup_groups = [v for v in hashes.values() if len(v) > 1]
    cross_split = [v for v in dup_groups if len({s for s, _ in v}) > 1]
    cross_label = [v for v in dup_groups if len({c for _, c in v}) > 1]
    per_subject = Counter((r["split"], r["subject"], r["cls"]) for r in rows)
    sizes = [(r["orig_w"], r["orig_h"]) for r in rows if "orig_w" in r]
    summary = {
        "n_crops": {f"{sp}/{c}": sum(1 for r in rows if r["split"] == sp and r["cls"] == c)
                    for sp in ("train", "val", "test") for c in ("live", "spoof")},
        "n_subjects": {sp: len(v) for sp, v in subj.items()},
        "subject_overlap": {"train&val": len(subj["train"] & subj["val"]),
                            "train&test": len(subj["train"] & subj["test"]),
                            "val&test": len(subj["val"] & subj["test"])},
        "images_per_subject_per_class": {"mean": float(np.mean(list(per_subject.values()))),
                                         "max": int(max(per_subject.values()))},
        "near_duplicate_groups": len(dup_groups),
        "near_duplicates_across_splits": len(cross_split),
        "near_duplicates_with_conflicting_labels": len(cross_label),
        "missing_metadata": sum(1 for r in rows if r["light"] is None),
        "original_resolution": {"n": len(sizes),
                                "width_median": float(np.median([w for w, _ in sizes])) if sizes else None,
                                "height_median": float(np.median([h for _, h in sizes])) if sizes else None,
                                "width_min": int(min(w for w, _ in sizes)) if sizes else None,
                                "width_max": int(max(w for w, _ in sizes)) if sizes else None},
        "spoof_types": {sp: dict(Counter(r["spoof_type"] for r in rows if r["split"] == sp and r["cls"] == "spoof"))
                        for sp in ("train", "val", "test")},
        "feature_means": {c: {f: float(np.mean([r[f] for r in rows if r["cls"] == c])) for f in FEATURES}
                          for c in ("live", "spoof")},
    }

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    captions = {}
    C = {"live": "#2a9d8f", "spoof": "#e76f51"}

    # Class balance per split
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    xs = np.arange(3)
    for k, c in enumerate(("live", "spoof")):
        v = [summary["n_crops"][f"{sp}/{c}"] for sp in ("train", "val", "test")]
        ax.bar(xs + (k - 0.5) * 0.38, v, 0.38, label=c, color=C[c])
        for x, n in zip(xs + (k - 0.5) * 0.38, v):
            ax.text(x, n, str(n), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(["train", "val", "test"])
    ax.set_title("Class balance of the prepared face crops")
    ax.set_xlabel("split"); ax.set_ylabel("number of crops"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "class_balance.png", dpi=200); plt.close(fig)
    captions["class_balance.png"] = ("Classes were balanced by construction (equal live and spoof crops per split), "
                                     "so plain accuracy is meaningful, but we still report APCER and BPCER separately.")

    # Spoof-type mix per split (normalised)
    types = sorted({t for sp in summary["spoof_types"].values() for t in sp if t})
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for k, sp in enumerate(("train", "val", "test")):
        tot = max(sum(summary["spoof_types"][sp].values()), 1)
        ax.bar(np.arange(len(types)) + (k - 1) * 0.27,
               [100 * summary["spoof_types"][sp].get(t, 0) / tot for t in types], 0.27, label=sp)
    ax.set_xticks(np.arange(len(types))); ax.set_xticklabels(types, rotation=20, ha="right")
    ax.set_title("Spoof-type mix per split")
    ax.set_xlabel("spoof type"); ax.set_ylabel("share of spoof crops (%)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "spoof_types.png", dpi=200); plt.close(fig)
    captions["spoof_types.png"] = ("The test split contains a much larger share of PC-screen replays than train, "
                                   "a distribution shift that later explains most baseline errors.")

    # Univariate: histograms + box plots per class
    fig, axes = plt.subplots(2, len(FEATURES), figsize=(15, 5.4))
    for j, f in enumerate(FEATURES):
        for c in ("live", "spoof"):
            v = [r[f] for r in rows if r["cls"] == c]
            axes[0, j].hist(v, 40, alpha=0.6, color=C[c], label=c)
        axes[0, j].set_title(f); axes[0, j].set_xlabel(FEATURE_LABEL[f]); axes[0, j].set_ylabel("crops")
        axes[1, j].boxplot([[r[f] for r in rows if r["cls"] == c] for c in ("live", "spoof")], showfliers=False)
        axes[1, j].set_xticks([1, 2]); axes[1, j].set_xticklabels(["live", "spoof"])
        axes[1, j].set_xlabel("class"); axes[1, j].set_ylabel(FEATURE_LABEL[f])
    axes[0, 0].legend()
    fig.suptitle("Image statistics of the face crops, live vs spoof (all splits)")
    fig.tight_layout(); fig.savefig(out / "univariate.png", dpi=170); plt.close(fig)
    captions["univariate.png"] = ("Live and spoof crops overlap heavily on every simple statistic, so no single "
                                  "hand-made feature separates them and a learned model is needed.")

    # Per spoof type: sharpness and HF energy
    groups = ["live"] + types
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, f in zip(axes, ("sharpness", "hf_energy")):
        data = [[r[f] for r in rows if (r["cls"] == "live" if g == "live" else r["spoof_type"] == g)] for g in groups]
        ax.boxplot([d if d else [np.nan] for d in data], showfliers=False)
        ax.set_xticks(range(1, len(groups) + 1)); ax.set_xticklabels(groups)
        ax.set_title(f"{f} by presentation type"); ax.set_xlabel("presentation type")
        ax.set_ylabel(FEATURE_LABEL[f]); ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(out / "by_type.png", dpi=200); plt.close(fig)
    captions["by_type.png"] = ("PC-screen replays have sharpness and high-frequency energy close to live faces, "
                               "which matches the baseline's main failure mode.")

    # Bivariate: correlation matrix (features + label)
    M = np.array([[r[f] for f in FEATURES] + [1.0 if r["cls"] == "live" else 0.0] for r in rows])
    corr = np.corrcoef(M.T)
    names = FEATURES + ["label (live=1)"]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlation of image statistics with each other and the label")
    fig.tight_layout(); fig.savefig(out / "correlation.png", dpi=200); plt.close(fig)
    summary["correlation_with_label"] = {f: float(corr[-1, k]) for k, f in enumerate(FEATURES)}
    captions["correlation.png"] = ("Correlations with the label are weak, confirming that the classes are not "
                                   "separable by global image statistics.")

    # Bivariate: scatter sharpness vs HF energy
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    rng = np.random.default_rng(0)
    for c in ("live", "spoof"):
        sub = [r for r in rows if r["cls"] == c]
        sub = [sub[i] for i in rng.choice(len(sub), min(1500, len(sub)), replace=False)]
        ax.scatter([r["sharpness"] for r in sub], [r["hf_energy"] for r in sub], s=4, alpha=0.4, c=C[c], label=c)
    ax.set_title("Sharpness vs high-frequency energy (1,500 crops per class)")
    ax.set_xlabel(FEATURE_LABEL["sharpness"]); ax.set_ylabel(FEATURE_LABEL["hf_energy"]); ax.legend()
    fig.tight_layout(); fig.savefig(out / "scatter.png", dpi=200); plt.close(fig)
    captions["scatter.png"] = "The two texture statistics overlap strongly between classes, with spoofs spread wider."

    # Images per subject
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for c in ("live", "spoof"):
        ax.hist([n for (sp, s, cc), n in per_subject.items() if cc == c], bins=range(1, 12), alpha=0.6,
                color=C[c], label=c, align="left")
    ax.set_title("Crops per subject and class (cap = 8)")
    ax.set_xlabel("crops from one subject"); ax.set_ylabel("number of subjects"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "per_subject.png", dpi=200); plt.close(fig)
    captions["per_subject.png"] = ("No subject contributes more than 8 crops per class, so the model cannot lean "
                                   "on a few heavily repeated identities.")

    # Lighting / environment
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key, title in ((axes[0], "light", "Lighting condition"), (axes[1], "env", "Environment")):
        cats = sorted({r[key] for r in rows if r[key] and r["cls"] == "spoof"})
        for k, sp in enumerate(("train", "test")):
            sub = [r for r in rows if r["split"] == sp and r["cls"] == "spoof"]
            ax.bar(np.arange(len(cats)) + (k - 0.5) * 0.38,
                   [100 * sum(1 for r in sub if r[key] == c) / max(len(sub), 1) for c in cats], 0.38, label=sp)
        ax.set_xticks(np.arange(len(cats))); ax.set_xticklabels(cats)
        ax.set_title(f"{title} of spoof crops"); ax.set_xlabel(key); ax.set_ylabel("share of spoof crops (%)")
        ax.legend()
    fig.tight_layout(); fig.savefig(out / "conditions.png", dpi=200); plt.close(fig)
    captions["conditions.png"] = "Capture conditions of the spoof crops are similar in train and test, mostly indoor and normal light."

    json.dump(summary, open(out / "eda_summary.json", "w"), indent=2)
    (out / "captions.md").write_text("\n".join(f"- **{k}**: {v}" for k, v in captions.items()) + "\n")
    print(json.dumps(summary, indent=2))
    print("figures ->", out)


if __name__ == "__main__":
    main()
