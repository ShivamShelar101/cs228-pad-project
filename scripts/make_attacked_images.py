"""
Makes demo images for Kiosk Guard: test-set fakes that BOTH models reject, saved clean and after a
PGD attack on the baseline at a small budget (default 2/255, the defense's training budget).

Upload a pair in the app: the clean image should read SPOOF on both models; the attacked one is
built to read LIVE on the baseline, and the hardened model holds on most of them (about 3 in 4 at
2/255 in the evaluation). Every image is kept, whether or not the hardened model holds, and a summary
file records the outcome per image, so the demo cannot be cherry-picked without it being visible.

Saved as PNG (lossless) after 8-bit rounding, and re-scored after rounding, because that is what the
app will receive.

    python -m scripts.make_attacked_images --data_root ./data_pad3 \
        --baseline checkpoints/baseline3/seed2_best.pt --hardened checkpoints/eps2_ft3/seed2_best.pt \
        --thresholds app/backend/thresholds.json --n 8 --out demo_images
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from attacks.pgd_attack import pgd_attack
from data.dataset import IMAGENET_MEAN, IMAGENET_STD, get_dataloaders
from models.baseline import build_model
from utils.device import get_device
from utils.seed import set_seed


def logit(p):
    return float(np.log(p / (1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--hardened", required=True)
    ap.add_argument("--thresholds", default=None,
                    help="JSON {'baseline': p, 'hardened': p} with the decision thresholds the app uses (default 0.5)")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--eps", type=float, default=2 / 255)
    ap.add_argument("--out", default="demo_images")
    args = ap.parse_args()
    set_seed(0)

    thr = {"baseline": 0.5, "hardened": 0.5}
    if args.thresholds:
        thr.update(json.load(open(args.thresholds)))
    t_base, t_hard = logit(thr["baseline"]), logit(thr["hardened"])   # compare logits, same as P(live) > p

    device = get_device()
    _, _, test_loader, c2i = get_dataloaders(args.data_root, batch_size=16, num_workers=0)
    spoof_idx = c2i["spoof"]

    def load(path):
        m = build_model(args.arch, pretrained=False).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        return m.eval()

    base, hard = load(args.baseline), load(args.hardened)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)
    to8 = lambda z: torch.round(torch.clamp(z * std + mean, 0, 1) * 255) / 255
    from8 = lambda p: (p - mean) / std

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, kept = [], 0
    for imgs, labels in test_loader:
        x = imgs[labels == spoof_idx].to(device)
        if not len(x):
            continue
        with torch.no_grad():
            both_reject = (base(x) <= t_base) & (hard(x) <= t_hard)   # start from fakes both models catch
        x = x[both_reject]
        if not len(x):
            continue
        # attack the baseline (pushing past ITS threshold: target the live side with BCE as in pgd_attack)
        adv8 = to8(pgd_attack(base, x, eps=args.eps, alpha=max(args.eps / 4, 0.5 / 255), steps=20))
        clean8 = to8(x)
        with torch.no_grad():
            b_adv, h_adv = base(from8(adv8)), hard(from8(adv8))
        for i in range(len(x)):
            if kept >= args.n:
                break
            fooled_base = bool(b_adv[i] > t_base)
            if not fooled_base:
                continue                               # the attack must work on the baseline to be a demo pair
            for name, img in ((f"clean_{kept:02d}.png", clean8[i]), (f"attacked_{kept:02d}.png", adv8[i])):
                Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(out / name)
            held = bool(h_adv[i] <= t_hard)
            rows.append({"pair": kept, "baseline_fooled": fooled_base, "hardened_held": held,
                         "p_live_baseline": float(torch.sigmoid(b_adv[i])), "p_live_hardened": float(torch.sigmoid(h_adv[i]))})
            print(f"pair {kept:02d}: baseline fooled, hardened {'HELD' if held else 'fooled too'}")
            kept += 1
        if kept >= args.n:
            break
    held = sum(r["hardened_held"] for r in rows)
    summary = {"eps": f"{args.eps * 255:.0f}/255", "thresholds": thr, "pairs": rows,
               "hardened_held": f"{held}/{len(rows)}"}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(f"done: {len(rows)} pairs in {out}/, hardened model held on {held}/{len(rows)} (see summary.json)")


if __name__ == "__main__":
    main()
