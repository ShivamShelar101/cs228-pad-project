"""
Phase 4 (proposal timeline): evaluation matrix.

Computes clean metrics (APCER/BPCER/ACER) and adversarial Attack
Success Rate for both baseline and hardened checkpoints, across every
seed you trained, and reports mean +/- std -- not single-run numbers.
This is the "rigor" pass discussed for a stronger-than-class-project
writeup.

Usage:
    python -m evaluation.evaluate --data_root ./data_pad \
        --baseline_glob "checkpoints/baseline/seed*_best.pt" \
        --hardened_glob "checkpoints/hardened/seed*_epoch14.pt"
"""
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

import torch

from data.dataset import IMAGENET_MEAN, IMAGENET_STD, get_dataloaders
from models.baseline import build_model
from utils.device import get_device
from attacks.pgd_attack import pgd_attack
from evaluation.metrics import compute_pad_metrics


def accuracy(model, loader, device, live_idx):
    ok = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            ok += ((model(x) > 0) == (y == live_idx)).sum().item()
            n += len(y)
    return ok / n


def evaluate_checkpoint(ckpt_path, arch, test_loader, val_loader, device, live_idx, spoof_idx,
                        eps=8 / 255, strong_eps=16 / 255, strong_steps=40):
    model = build_model(arch, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    clean = compute_pad_metrics(model, test_loader, device, live_idx)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    lo, hi = (0.0 - mean) / std, (1.0 - mean) / std
    c = defaultdict(int)
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        is_live = y == live_idx
        xs, xl = x[~is_live], x[is_live]
        if len(xs):
            c["n_spoof"] += len(xs)
            adv = pgd_attack(model, xs, eps=eps)                                   # fakes pushed toward live
            with torch.no_grad():
                c["asr"] += (model(adv) > 0).sum().item()
            adv = pgd_attack(model, xs, eps=strong_eps, steps=strong_steps)        # stronger attacker
            with torch.no_grad():
                c["asr_strong"] += (model(adv) > 0).sum().item()
            noise = (torch.randint(0, 2, xs.shape, device=device).float() * 2 - 1) * eps / std
            with torch.no_grad():                                                  # control: random noise, no gradient
                c["apcer_noise"] += (model(torch.max(torch.min(xs + noise, hi), lo)) > 0).sum().item()
        if len(xl):
            c["n_live"] += len(xl)
            adv = pgd_attack(model, xl, eps=eps, target=0.0)                       # real faces pushed toward spoof
            with torch.no_grad():
                c["bpcer_adv"] += (model(adv) <= 0).sum().item()

    return {
        **clean,
        "ASR": c["asr"] / c["n_spoof"],
        "ASR_strong": c["asr_strong"] / c["n_spoof"],
        "APCER_noise": c["apcer_noise"] / c["n_spoof"],
        "BPCER_adv": c["bpcer_adv"] / c["n_live"],
        "ACC_val": accuracy(model, val_loader, device, live_idx),
        "ACC_test": accuracy(model, test_loader, device, live_idx),
    }


def summarize(results):
    keys = results[0].keys()
    return {
        k: {
            "mean": statistics.mean(r[k] for r in results),
            "std": statistics.stdev(r[k] for r in results) if len(results) > 1 else 0.0,
        }
        for k in keys
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--baseline_glob", default="checkpoints/baseline/seed*_best.pt")
    ap.add_argument("--hardened_glob", default="checkpoints/hardened/seed*_epoch14.pt")
    ap.add_argument("--eps", type=float, default=8 / 255, help="attack budget in pixel units (16/255 = 0.0627)")
    ap.add_argument("--strong_eps", type=float, default=16 / 255, help="budget of the stronger attacker")
    ap.add_argument("--strong_steps", type=int, default=40)
    ap.add_argument("--out", default="results/eval_summary.json")
    args = ap.parse_args()

    device = get_device()
    _, val_loader, test_loader, class_to_idx = get_dataloaders(args.data_root)
    live_idx, spoof_idx = class_to_idx["live"], class_to_idx["spoof"]

    report = {}
    for name, pattern in [("baseline", args.baseline_glob), ("hardened", args.hardened_glob)]:
        ckpts = sorted(glob.glob(pattern))
        if not ckpts:
            print(f"warning: no checkpoints matched {pattern}")
            continue
        results = [
            evaluate_checkpoint(c, args.arch, test_loader, val_loader, device, live_idx, spoof_idx,
                                args.eps, args.strong_eps, args.strong_steps)
            for c in ckpts
        ]
        report[name] = {"per_seed": results, "summary": summarize(results)}
        print(name, report[name]["summary"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
