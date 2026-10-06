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

import torch

from data.dataset import get_dataloaders
from models.baseline import build_model
from utils.device import get_device
from attacks.pgd_attack import pgd_attack, attack_success_rate
from evaluation.metrics import compute_pad_metrics


def evaluate_checkpoint(ckpt_path, arch, loader, device, live_idx, spoof_idx, eps=8 / 255):
    model = build_model(arch, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    clean = compute_pad_metrics(model, loader, device, live_idx)

    asrs = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        spoof_mask = labels == spoof_idx
        if spoof_mask.sum() == 0:
            continue
        adv = pgd_attack(model, imgs[spoof_mask], eps=eps)
        asrs.append(attack_success_rate(model, adv))
    asr = sum(asrs) / len(asrs) if asrs else 0.0

    return {**clean, "ASR": asr}


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
    ap.add_argument("--out", default="results/eval_summary.json")
    args = ap.parse_args()

    device = get_device()
    _, _, test_loader, class_to_idx = get_dataloaders(args.data_root)
    live_idx, spoof_idx = class_to_idx["live"], class_to_idx["spoof"]

    report = {}
    for name, pattern in [("baseline", args.baseline_glob), ("hardened", args.hardened_glob)]:
        ckpts = sorted(glob.glob(pattern))
        if not ckpts:
            print(f"warning: no checkpoints matched {pattern}")
            continue
        results = [
            evaluate_checkpoint(c, args.arch, test_loader, device, live_idx, spoof_idx, args.eps)
            for c in ckpts
        ]
        report[name] = {"per_seed": results, "summary": summarize(results)}
        print(name, report[name]["summary"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
