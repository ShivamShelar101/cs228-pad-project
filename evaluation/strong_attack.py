"""
Stronger attack check: is the defense's robustness real, or an artifact of a weak 10-step PGD?

For each model, takes the test fakes it rejects WITHOUT attack and attacks them with
  * PGD-10 (the attack used everywhere else, for reference),
  * PGD-50 with 2 random restarts and two losses (4 runs of 50 steps) (cross-entropy, and the raw logit, which
    cannot saturate). An image counts as broken if ANY restart, loss or step pushes it past
    the threshold while staying inside the budget.
If the strong attack succeeds far more often than PGD-10, the reported robustness was overestimated.

    python -m evaluation.strong_attack --data_root ./data_pad2 \
        --ckpt_glob "checkpoints/eps2_ft/seed*_best.pt" --eps 1 2 4 --out results/strong/eps2_ft
"""
import argparse
import glob
import json
import os
import statistics

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from attacks.pgd_attack import pgd_attack
from data.dataset import IMAGENET_MEAN, IMAGENET_STD, get_transforms
from models.baseline import build_model
from utils.device import get_device
from utils.seed import set_seed


def strong_attack(model, x, eps, steps=50, restarts=2):
    """Returns a bool tensor: broken (accepted as live) at any step of any restart, for either loss."""
    dev = x.device
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    lo, hi = (0 - mean) / std, (1 - mean) / std
    eps_t = eps / std
    broken = torch.zeros(x.size(0), dtype=torch.bool, device=dev)
    target = torch.ones(x.size(0), device=dev)
    for loss_name in ("ce", "logit"):
        for r in range(restarts):
            alpha_t = eps_t / 4
            adv = torch.max(torch.min(x + (torch.rand_like(x) * 2 - 1) * eps_t, hi), lo).detach()
            for t in range(steps):
                if t in (steps // 2, 3 * steps // 4):
                    alpha_t = alpha_t / 2          # step-size decay, as in APGD
                adv.requires_grad_(True)
                logit = model(adv)
                broken |= (logit.detach() > 0)
                loss = F.binary_cross_entropy_with_logits(logit, target) if loss_name == "ce" else -logit.sum()
                g = torch.autograd.grad(loss, adv)[0]
                adv = adv.detach() - alpha_t * g.sign()
                adv = torch.max(torch.min(adv, x + eps_t), x - eps_t)
                adv = torch.max(torch.min(adv, hi), lo).detach()
            with torch.no_grad():
                broken |= model(adv) > 0
    return broken


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt_glob", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--eps", type=float, nargs="+", default=[1, 2, 4], help="budgets in /255")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--out", required=True, help="output prefix (.json and .md are added)")
    args = ap.parse_args()
    set_seed(0)
    device = get_device()

    ds = datasets.ImageFolder(f"{args.data_root}/test", transform=get_transforms(False))
    spoof_idx = ds.class_to_idx["spoof"]
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    per_seed = []
    for ck in sorted(glob.glob(args.ckpt_glob)):
        model = build_model(args.arch, pretrained=False).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        n_caught, weak, strong = 0, {e: 0 for e in args.eps}, {e: 0 for e in args.eps}
        for x, y in loader:
            x = x[y == spoof_idx].to(device)
            if not len(x):
                continue
            with torch.no_grad():
                caught = model(x) <= 0
            x = x[caught]
            if not len(x):
                continue
            n_caught += len(x)
            for e in args.eps:
                adv = pgd_attack(model, x, eps=e / 255, alpha=max(e / 4, 0.5) / 255, steps=10)
                with torch.no_grad():
                    w = model(adv) > 0
                weak[e] += w.sum().item()
                # the strong attack includes PGD-10, so it can only find more successes, never fewer
                strong[e] += (w | strong_attack(model, x, e / 255, args.steps, args.restarts)).sum().item()
        r = {"checkpoint": ck, "n_caught": n_caught,
             **{f"pgd10_eps{e:g}": weak[e] / max(n_caught, 1) for e in args.eps},
             **{f"strong_eps{e:g}": strong[e] / max(n_caught, 1) for e in args.eps}}
        print(json.dumps(r), flush=True)
        per_seed.append(r)

    keys = [k for k in per_seed[0] if k.startswith(("pgd10", "strong"))]
    summary = {k: (statistics.mean(r[k] for r in per_seed),
                   statistics.stdev(r[k] for r in per_seed) if len(per_seed) > 1 else 0.0) for k in keys}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"per_seed": per_seed, "summary": summary}, open(args.out + ".json", "w"), indent=2)
    lines = [f"**Attack success on caught test fakes: PGD-10 vs PGD-{args.steps} x {args.restarts} restarts x 2 losses** "
             f"({len(per_seed)} seed(s), `{args.ckpt_glob}`)", "",
             "| Budget | PGD-10 | Strong attack |", "|---|---|---|"]
    for e in args.eps:
        w, s = summary[f"pgd10_eps{e:g}"], summary[f"strong_eps{e:g}"]
        lines.append(f"| {e:g}/255 | {100*w[0]:.1f}% ± {100*w[1]:.1f} | {100*s[0]:.1f}% ± {100*s[1]:.1f} |")
    md = "\n".join(lines)
    open(args.out + ".md", "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
