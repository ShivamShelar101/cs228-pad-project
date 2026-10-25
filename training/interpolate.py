"""
Weight interpolation between a clean model and its adversarially fine-tuned version (no training):
    w(alpha) = (1 - alpha) * w_baseline + alpha * w_defended
Both start from the same baseline weights, so the straight line between them usually stays a good model,
and points along it trade clean accuracy for robustness (cf. WiSE-FT, Wortsman et al., CVPR 2022).

    python -m training.interpolate --base checkpoints/baseline3/seed2_best.pt \
        --defended checkpoints/eps2_ft3/seed2_best.pt --alphas 0.25 0.5 0.75 --out_prefix checkpoints/interp3 --seed 2
writes checkpoints/interp3_a025/seed2_best.pt, ...
"""
import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--defended", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    a_sd = torch.load(args.base, map_location="cpu")
    b_sd = torch.load(args.defended, map_location="cpu")
    for alpha in args.alphas:
        mix = {}
        for k, v in b_sd.items():
            if torch.is_floating_point(v):
                mix[k] = (1 - alpha) * a_sd[k] + alpha * v      # weights and batch-norm statistics
            else:
                mix[k] = v                                         # integer counters: keep
        d = f"{args.out_prefix}_a{int(round(alpha * 100)):03d}"
        os.makedirs(d, exist_ok=True)
        torch.save(mix, f"{d}/seed{args.seed}_best.pt")
        print(f"alpha={alpha}: {d}/seed{args.seed}_best.pt")


if __name__ == "__main__":
    main()
