"""
Checkpoint Section IV: baseline threshold analysis and error counts.

For every baseline checkpoint (one per seed) it scores the validation and test sets and reports:
  * APCER / BPCER / ACER with raw counts (e.g. "119/600") at the default threshold (p = 0.5)
  * the EER (where APCER = BPCER) on validation and on test, and ROC AUC
  * an operating point chosen on VALIDATION (the validation EER threshold), applied unchanged to test
  * mean +/- std over seeds
Figures (seed 0): score histograms, APCER and BPCER against the threshold, ROC curve, and image grids of
test false accepts (fakes called live) and false rejects (real faces called fake).

    python -m evaluation.threshold_analysis --data_root ./data_pad2 \
        --ckpt_glob "checkpoints/baseline/seed*_best.pt" --out results/threshold
"""
import argparse
import glob
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from data.dataset import get_transforms
from models.baseline import build_model
from utils.device import get_device
from utils.seed import set_seed


LABEL = "Baseline"  # model name used in figure titles and the table heading (--label)


def scores(model, root, split, device):
    """Probability of 'live' for every image, the true labels (1 = live) and the file paths."""
    ds = datasets.ImageFolder(f"{root}/{split}", transform=get_transforms(False))
    live_idx = ds.class_to_idx["live"]
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(torch.sigmoid(model(x.to(device))).cpu())
    p = torch.cat(out).numpy()
    y = np.array([int(t == live_idx) for _, t in ds.samples])
    return p, y, [s for s, _ in ds.samples]


def rates(p, y, thr):
    """APCER = fakes accepted as live; BPCER = real faces rejected. Returns rates and counts."""
    fake, live = y == 0, y == 1
    fa = int((p[fake] >= thr).sum())
    fr = int((p[live] < thr).sum())
    nf, nl = int(fake.sum()), int(live.sum())
    apcer, bpcer = fa / nf, fr / nl
    return {"APCER": apcer, "BPCER": bpcer, "ACER": (apcer + bpcer) / 2,
            "APCER_count": f"{fa}/{nf}", "BPCER_count": f"{fr}/{nl}", "threshold": float(thr)}


def eer(p, y):
    """Threshold where APCER and BPCER are closest, and the error rate there."""
    u = np.unique(np.concatenate([p, [0.0, 1.0]]))
    thrs = np.concatenate([u, (u[:-1] + u[1:]) / 2])      # scores and the midpoints between them
    gaps = np.array([abs(rates(p, y, t)["APCER"] - rates(p, y, t)["BPCER"]) for t in thrs])
    # Several thresholds can tie (e.g. a gap between the classes); take the middle of the tied range.
    tied = thrs[gaps <= gaps.min() + 1e-12]
    best = float(np.median(tied))
    r = rates(p, y, best)
    return float(best), (r["APCER"] + r["BPCER"]) / 2


def auc(p, y):
    """ROC AUC via the rank statistic (probability a random live image scores above a random fake)."""
    pos, neg = p[y == 1], p[y == 0]
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def figures(pv, yv, pt, yt, op_thr, paths_t, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    # 1. score distributions
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(0, 1, 41)
    ax.hist(pt[yt == 1], bins, alpha=0.6, label="live (test)", color="#2a9d8f")
    ax.hist(pt[yt == 0], bins, alpha=0.6, label="spoof (test)", color="#e76f51")
    ax.axvline(0.5, ls="--", c="k", lw=1, label="default threshold 0.5")
    ax.axvline(op_thr, ls=":", c="#264653", lw=1.5, label=f"chosen on val ({op_thr:.3f})")
    ax.set_yscale("log")
    ax.set_title(f"{LABEL} score distribution on the test set")
    ax.set_xlabel("predicted probability of 'live'")
    ax.set_ylabel("number of images (log scale)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "scores_hist.png", dpi=200); plt.close(fig)

    # 2. APCER (FAR) and BPCER (FRR) against the threshold
    thrs = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for p, y, tag, ls in ((pv, yv, "val", "--"), (pt, yt, "test", "-")):
        a = [rates(p, y, t)["APCER"] * 100 for t in thrs]
        b = [rates(p, y, t)["BPCER"] * 100 for t in thrs]
        ax.plot(thrs, a, ls, c="#e76f51", label=f"APCER / FAR ({tag})")
        ax.plot(thrs, b, ls, c="#2a9d8f", label=f"BPCER / FRR ({tag})")
    ax.axvline(op_thr, ls=":", c="#264653", lw=1.5, label="operating point (val EER)")
    ax.set_title(f"{LABEL} error rates against the decision threshold")
    ax.set_xlabel("threshold on P(live)")
    ax.set_ylabel("error rate (%)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "far_frr.png", dpi=200); plt.close(fig)

    # 3. ROC
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for p, y, tag, ls in ((pv, yv, "val", "--"), (pt, yt, "test", "-")):
        apc = [rates(p, y, t)["APCER"] for t in thrs]
        tpr = [1 - rates(p, y, t)["BPCER"] for t in thrs]
        ax.plot(apc, tpr, ls, label=f"{tag} (AUC {auc(p, y):.3f})")
    ax.plot([0, 1], [0, 1], c="0.7", lw=0.8)
    ax.set_title(f"{LABEL} ROC")
    ax.set_xlabel("APCER (fakes accepted)")
    ax.set_ylabel("1 - BPCER (real faces accepted)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "roc.png", dpi=200); plt.close(fig)

    # 4. error grids at the chosen operating point
    paths_t = np.array(paths_t)
    for name, mask, title in (("false_accepts", (yt == 0) & (pt >= op_thr), "Test fakes accepted as live"),
                              ("false_rejects", (yt == 1) & (pt < op_thr), "Test real faces rejected")):
        idx = np.where(mask)[0]
        idx = idx[np.argsort(-pt[idx] if name == "false_accepts" else pt[idx])][:12]
        if len(idx) == 0:
            continue
        fig, axes = plt.subplots(2, 6, figsize=(12, 4.4))
        for ax in axes.flat:
            ax.axis("off")
        for ax, i in zip(axes.flat, idx):
            ax.imshow(Image.open(paths_t[i]))
            ax.set_title(f"P(live)={pt[i]:.2f}", fontsize=8)
        fig.suptitle(f"{title} (most confident first, threshold {op_thr:.3f})")
        fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt_glob", default="checkpoints/baseline/seed*_best.pt")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--out", default="results/threshold")
    ap.add_argument("--label", default="Baseline", help="model name for titles, e.g. \"Defended model\"")
    args = ap.parse_args()
    global LABEL
    LABEL = args.label
    set_seed(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = get_device()

    per_seed, first = [], None
    for ck in sorted(glob.glob(args.ckpt_glob)):
        model = build_model(args.arch, pretrained=False).to(device)
        model.load_state_dict(torch.load(ck, map_location=device)); model.eval()
        pv, yv, _ = scores(model, args.data_root, "val", device)
        pt, yt, paths_t = scores(model, args.data_root, "test", device)
        op_thr, val_eer = eer(pv, yv)
        _, test_eer = eer(pt, yt)
        r = {"checkpoint": ck,
             "default": rates(pt, yt, 0.5),
             "operating_point": rates(pt, yt, op_thr),
             "val_EER": val_eer, "test_EER": test_eer,
             "val_AUC": auc(pv, yv), "test_AUC": auc(pt, yt)}
        per_seed.append(r)
        print(ck, json.dumps({k: v for k, v in r.items() if k != "checkpoint"}))
        if first is None:
            first = (pv, yv, pt, yt, op_thr, paths_t)
    if not per_seed:
        raise SystemExit(f"no checkpoints matched {args.ckpt_glob}")

    def ms(vals):
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    summary = {}
    for key in ("val_EER", "test_EER", "val_AUC", "test_AUC"):
        summary[key] = ms([r[key] for r in per_seed])
    for point in ("default", "operating_point"):
        for m in ("APCER", "BPCER", "ACER", "threshold"):
            summary[f"{point}_{m}"] = ms([r[point][m] for r in per_seed])
    json.dump({"per_seed": per_seed, "summary": summary}, open(out / "threshold.json", "w"), indent=2)

    def f(k, pct=True):
        m, s = summary[k]
        return f"{m*100:.1f}% ± {s*100:.1f}" if pct else f"{m:.3f} ± {s:.3f}"
    lines = [f"**{LABEL} over {len(per_seed)} seed(s)** (mean ± std)", "",
             "| Metric | Default threshold (0.5) | Operating point (val EER threshold) |", "|---|---|---|",
             f"| Threshold | 0.500 | {f('operating_point_threshold', False)} |",
             f"| APCER (fakes accepted) | {f('default_APCER')} | {f('operating_point_APCER')} |",
             f"| BPCER (real rejected) | {f('default_BPCER')} | {f('operating_point_BPCER')} |",
             f"| ACER | {f('default_ACER')} | {f('operating_point_ACER')} |", "",
             f"EER: validation {f('val_EER')}, test {f('test_EER')}. "
             f"AUC: validation {f('val_AUC', False)}, test {f('test_AUC', False)}.", "",
             "| Seed checkpoint | APCER count (0.5) | BPCER count (0.5) | APCER count (op.) | BPCER count (op.) |",
             "|---|---|---|---|---|"]
    for r in per_seed:
        lines.append(f"| {Path(r['checkpoint']).name} | {r['default']['APCER_count']} | {r['default']['BPCER_count']} | "
                     f"{r['operating_point']['APCER_count']} | {r['operating_point']['BPCER_count']} |")
    md = "\n".join(lines) + "\n"
    (out / "threshold.md").write_text(md)
    print(md)
    figures(*first, out)
    print("figures ->", out)


if __name__ == "__main__":
    main()
