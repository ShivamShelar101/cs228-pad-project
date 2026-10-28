"""
Demographic differentials: do error rates differ between groups of people, and does adversarial
training widen or narrow the gap?

Groups come from CelebA-Spoof's 40 CelebA face attributes (crowd-annotated, binary, coarse; treat the
result as an indication, not a definitive fairness audit). If a spoof image carries no face attributes,
it inherits the majority attributes of the same subject's live images.

For each model (baseline, defense), each seed, and each group it reports on the TEST split:
  APCER (fakes accepted), BPCER (real faces rejected), ACER, at the validation-chosen threshold;
  for the defense also the attack success at 2/255 on fakes it rejects (PGD-10).
Then the gap between the two groups of each attribute, as mean +- std over seeds.

    python -m evaluation.fairness --src ./celeba_meta --data_root ./data_pad3 \
        --baseline_glob "checkpoints/baseline3/seed[012]_best.pt" \
        --defense_glob "checkpoints/eps2_ft3/seed[012]_best.pt" \
        --baseline_thr results/data3/threshold_baseline3/threshold.json \
        --defense_thr results/data3/threshold_eps2_ft3/threshold.json --out results/fairness
"""
import argparse
import glob
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from attacks.pgd_attack import pgd_attack
from data.dataset import get_transforms
from data.prepare_celeba_spoof import find_root
from evaluation.eda import original_key
from models.baseline import build_model
from utils.device import get_device
from utils.seed import set_seed

# CelebA attribute order (indices 0-39 of the CelebA-Spoof label vector)
ATTR = {"Male": 20, "Young": 39, "Pale_Skin": 26, "Eyeglasses": 15, "Heavy_Makeup": 18}
NAMES = {"Male": ("male", "female"), "Young": ("young", "older"), "Pale_Skin": ("pale skin", "not pale skin"),
         "Eyeglasses": ("eyeglasses", "no eyeglasses"), "Heavy_Makeup": ("heavy makeup", "no heavy makeup")}


def subject_attrs(labels):
    """Majority value of each attribute over a subject's live images (fallback for spoof images)."""
    votes = defaultdict(lambda: defaultdict(Counter))
    for key, lab in labels.items():
        parts = key.split("/")
        if len(parts) > 3 and parts[3] == "live":
            for a, i in ATTR.items():
                votes[parts[2]][a][int(lab[i] > 0)] += 1
    return {s: {a: c.most_common(1)[0][0] for a, c in d.items()} for s, d in votes.items()}


def thresholds(path):
    return {r["checkpoint"]: r["operating_point"]["threshold"] for r in json.load(open(path))["per_seed"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--baseline_glob", required=True)
    ap.add_argument("--defense_glob", required=True)
    ap.add_argument("--baseline_thr", required=True)
    ap.add_argument("--defense_thr", required=True)
    ap.add_argument("--eps", type=float, default=2 / 255)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--out", default="results/fairness")
    args = ap.parse_args()
    set_seed(0)
    device = get_device()

    root = find_root(args.src)
    labels = json.load(open(root / "metas" / "intra_test" / "test_label.json"))
    fallback = subject_attrs(labels)
    ds = datasets.ImageFolder(f"{args.data_root}/test", transform=get_transforms(False))
    inv = {v: k for k, v in ds.class_to_idx.items()}

    # group membership per test image
    member, from_subject = [], 0
    for path, y in ds.samples:
        cls = inv[y]
        subject, key = original_key("test", cls, path, labels)
        lab = labels.get(key) if key else None
        if lab is not None and (cls == "live" or any(int(lab[i]) for i in range(40))):
            m = {a: int(lab[i] > 0) for a, i in ATTR.items()}
        else:
            m = fallback.get(subject)
            from_subject += 1
        member.append(m)
    print(f"{len(ds)} test images; {from_subject} took attributes from the subject's live images")

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    results = {}  # model -> list over seeds of {group: stats}
    for model_name, pattern, thr_path in (("baseline", args.baseline_glob, args.baseline_thr),
                                          ("defense", args.defense_glob, args.defense_thr)):
        thr = thresholds(thr_path)
        for ck in sorted(glob.glob(pattern)):
            model = build_model(args.arch, pretrained=False).to(device)
            model.load_state_dict(torch.load(ck, map_location=device))
            model.eval()
            t = math.log(thr[ck] / (1 - thr[ck]))        # P(live) > p  <=>  logit > log(p / (1-p))
            accepted, attacked_ok = [], []
            for x, _ in loader:
                x = x.to(device)
                with torch.no_grad():
                    lg = model(x)
                accepted += (lg > t).cpu().tolist()
                if model_name == "defense":
                    adv = pgd_attack(model, x, eps=args.eps, alpha=max(args.eps / 4, 0.5 / 255), steps=10)
                    with torch.no_grad():
                        attacked_ok += (model(adv) > t).cpu().tolist()
            per_group = {}
            for a in ATTR:
                for val in (1, 0):
                    idx = [i for i, m in enumerate(member) if m is not None and m[a] == val]
                    live = [i for i in idx if inv[ds.samples[i][1]] == "live"]
                    spoof = [i for i in idx if inv[ds.samples[i][1]] == "spoof"]
                    g = NAMES[a][0 if val else 1]
                    apcer = sum(accepted[i] for i in spoof) / max(len(spoof), 1)
                    bpcer = sum(not accepted[i] for i in live) / max(len(live), 1)
                    st = {"n_live": len(live), "n_spoof": len(spoof), "APCER": apcer, "BPCER": bpcer,
                          "ACER": (apcer + bpcer) / 2}
                    if model_name == "defense":
                        caught = [i for i in spoof if not accepted[i]]
                        st["ASRc_2"] = sum(attacked_ok[i] for i in caught) / max(len(caught), 1)
                    per_group[(a, g)] = st
            results.setdefault(model_name, []).append(per_group)
            print(f"{model_name} {ck} done")

    def ms(vals):
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    lines = ["**Error rates by demographic group (test set, validation-chosen thresholds, mean ± std over seeds)**", "",
             "| Attribute | Group | Live / spoof images | Baseline APCER | Baseline BPCER | Defense APCER | Defense BPCER | Defense attack success 2/255 |",
             "|---|---|---|---|---|---|---|---|"]
    gaps = []
    for a in ATTR:
        acer_gap = {}
        for g in NAMES[a]:
            b = [r[(a, g)] for r in results["baseline"]]
            d = [r[(a, g)] for r in results["defense"]]
            f = lambda rs, k: "{:.1f}% ± {:.1f}".format(*(100 * v for v in ms([r[k] for r in rs])))
            lines.append(f"| {a} | {g} | {b[0]['n_live']} / {b[0]['n_spoof']} | {f(b, 'APCER')} | {f(b, 'BPCER')} | "
                         f"{f(d, 'APCER')} | {f(d, 'BPCER')} | {f(d, 'ASRc_2')} |")
            acer_gap[g] = ([r["ACER"] for r in b], [r["ACER"] for r in d])
        g1, g2 = NAMES[a]
        bg = [abs(x - y) for x, y in zip(acer_gap[g1][0], acer_gap[g2][0])]
        dg = [abs(x - y) for x, y in zip(acer_gap[g1][1], acer_gap[g2][1])]
        gaps.append((a, ms(bg), ms(dg)))
    lines += ["", "**ACER gap between the two groups of each attribute (percentage points, mean ± std over seeds)**", "",
              "| Attribute | Baseline gap | Defense gap |", "|---|---|---|"]
    for a, (bm, bs), (dm, dsd) in gaps:
        lines.append(f"| {a} | {100*bm:.1f} ± {100*bs:.1f} | {100*dm:.1f} ± {100*dsd:.1f} |")
    lines += ["", "Attributes are CelebA's crowd-annotated binary labels; small groups give noisy estimates, so read the "
              "image counts alongside each rate."]
    os.makedirs(args.out, exist_ok=True)
    md = "\n".join(lines)
    open(os.path.join(args.out, "fairness.md"), "w").write(md + "\n")
    json.dump({m: [{f"{a}|{g}": v for (a, g), v in r.items()} for r in rs] for m, rs in results.items()},
              open(os.path.join(args.out, "fairness.json"), "w"), indent=2)
    print(md)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    xs = range(len(gaps))
    ax.bar([x - 0.2 for x in xs], [100 * g[1][0] for g in gaps], 0.4, yerr=[100 * g[1][1] for g in gaps],
           label="baseline", color="#E8705A", capsize=3)
    ax.bar([x + 0.2 for x in xs], [100 * g[2][0] for g in gaps], 0.4, yerr=[100 * g[2][1] for g in gaps],
           label="defense", color="#2A9D8F", capsize=3)
    ax.set_xticks(list(xs)); ax.set_xticklabels([g[0].replace("_", " ") for g in gaps], fontsize=9)
    ax.set_ylabel("ACER gap between groups (pp)")
    ax.set_title("Demographic differential in ACER, baseline vs defense")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fairness_gap.png"), dpi=200)
    print("chart ->", os.path.join(args.out, "fairness_gap.png"))


if __name__ == "__main__":
    main()
