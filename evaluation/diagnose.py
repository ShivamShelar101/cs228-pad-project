"""
Diagnostics for a finished run. Answers two questions:
  1. Why does the baseline accept ~45% of TEST fakes when it scored ~99% on VAL?
  2. Is the hardened model's low attack success real robustness, or a shortcut
     (a model trained only on attacked FAKES can learn "has attack noise -> fake")?

    python -m evaluation.diagnose --data_root ./data_pad \
        --baseline checkpoints/baseline/seed0_best.pt \
        --model full=checkpoints/full/seed0_epoch4.pt \
        --model adv_only=checkpoints/adv_only/seed0_epoch4.pt \
        --model reg_only=checkpoints/reg_only/seed0_epoch4.pt
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from attacks.pgd_attack import pgd_attack
from data.dataset import IMAGENET_MEAN, IMAGENET_STD, get_dataloaders
from models.baseline import build_model
from utils.device import get_device


def load(path, arch, device):
    m = build_model(arch, pretrained=False).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval()


def measure(model, loader, device, live_idx, strong_steps):
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    lo, hi = (0.0 - mean) / std, (1.0 - mean) / std
    c = defaultdict(int)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        is_live = y == live_idx
        xs, xl = x[~is_live], x[is_live]
        if len(xs):
            c["n_spoof"] += len(xs)
            with torch.no_grad():
                c["spoof_accepted_clean"] += (model(xs) > 0).sum().item()
            adv = pgd_attack(model, xs, eps=8 / 255, alpha=2 / 255, steps=10)
            with torch.no_grad():
                c["spoof_accepted_pgd8"] += (model(adv) > 0).sum().item()
            adv = pgd_attack(model, xs, eps=16 / 255, alpha=2 / 255, steps=strong_steps)
            with torch.no_grad():
                c["spoof_accepted_pgd16_strong"] += (model(adv) > 0).sum().item()
            # control: random +-8/255 noise, no gradient information at all
            noise = (torch.randint(0, 2, xs.shape, device=device).float() * 2 - 1) * (8 / 255) / std
            xn = torch.max(torch.min(xs + noise, hi), lo)
            with torch.no_grad():
                c["spoof_accepted_random_noise"] += (model(xn) > 0).sum().item()
        if len(xl):
            c["n_live"] += len(xl)
            with torch.no_grad():
                c["live_rejected_clean"] += (model(xl) <= 0).sum().item()
            # same perturbation applied to REAL faces (pushes them further toward 'live')
            adv = pgd_attack(model, xl, eps=8 / 255, alpha=2 / 255, steps=10)
            with torch.no_grad():
                c["live_rejected_pgd8"] += (model(adv) <= 0).sum().item()
    return {
        "fakes accepted, clean": c["spoof_accepted_clean"] / c["n_spoof"],
        "fakes accepted, PGD 8/255 x10": c["spoof_accepted_pgd8"] / c["n_spoof"],
        "fakes accepted, PGD 16/255 strong": c["spoof_accepted_pgd16_strong"] / c["n_spoof"],
        "fakes accepted, RANDOM noise 8/255": c["spoof_accepted_random_noise"] / c["n_spoof"],
        "real rejected, clean": c["live_rejected_clean"] / c["n_live"],
        "real rejected, after PGD 8/255": c["live_rejected_pgd8"] / c["n_live"],
    }


def accuracy(model, loader, device, live_idx):
    ok = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred_live = model(x) > 0
            ok += (pred_live == (y == live_idx)).sum().item()
            n += len(y)
    return ok / n


def missed_fakes_grid(model, loader, device, live_idx, out_png, k=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    shown = []
    with torch.no_grad():
        for x, y in loader:
            xs = x[(y != live_idx)].to(device)
            if not len(xs):
                continue
            for img in xs[model(xs) > 0].cpu():
                shown.append((img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
            if len(shown) >= k:
                break
    if not shown:
        print("no accepted fakes to show")
        return
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for ax, im in zip(axes.flat, shown[:k]):
        ax.imshow(im)
        ax.axis("off")
    fig.suptitle("Test fakes that the baseline accepted as LIVE")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print("saved", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--model", action="append", default=[], help='"name=path/to/checkpoint.pt"')
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--strong_steps", type=int, default=40)
    ap.add_argument("--out", default="results/diagnose")
    args = ap.parse_args()

    device = get_device()
    _, val_loader, test_loader, c2i = get_dataloaders(args.data_root)
    live_idx = c2i["live"]

    models = {"baseline": load(args.baseline, args.arch, device)}
    for spec in args.model:
        name, path = spec.split("=", 1)
        models[name] = load(path, args.arch, device)

    print(f"\nBaseline clean accuracy: val {accuracy(models['baseline'], val_loader, device, live_idx):.3f}"
          f"  vs  test {accuracy(models['baseline'], test_loader, device, live_idx):.3f}")

    results = {}
    for name, m in models.items():
        print(f"measuring {name} ...")
        results[name] = measure(m, test_loader, device, live_idx, args.strong_steps)

    keys = list(next(iter(results.values())))
    width = max(len(k) for k in keys)
    print("\n" + " " * (width + 2) + "".join(f"{n:>12}" for n in results))
    for k in keys:
        print(f"{k:<{width}}  " + "".join(f"{results[n][k]*100:>11.1f}%" for n in results))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out + ".json", "w"), indent=2)
    missed_fakes_grid(models["baseline"], test_loader, device, live_idx, args.out + "_baseline_missed_fakes.png")


if __name__ == "__main__":
    main()
