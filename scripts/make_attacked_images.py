"""
Makes PGD-attacked spoof images for the live demo. Upload them in Kiosk Guard to
show the baseline being fooled and the hardened model (hopefully) holding.

Saves PNG (lossless). JPEG would wash the perturbation out. Only images that
still fool the BASELINE after 8-bit saving are kept, so the demo is honest.

    python -m scripts.make_attacked_images --data_root ./data_pad \
        --baseline checkpoints/baseline/seed0_best.pt --n 8 --out demo_images
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from attacks.pgd_attack import pgd_attack
from data.dataset import IMAGENET_MEAN, IMAGENET_STD, get_dataloaders
from models.baseline import build_model
from utils.device import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--hardened", default=None, help="optional: also report if it resists")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--out", default="demo_images")
    args = ap.parse_args()

    device = get_device()
    _, _, test_loader, c2i = get_dataloaders(args.data_root, batch_size=16, num_workers=0)
    spoof_idx = c2i["spoof"]

    def load(path):
        m = build_model(args.arch, pretrained=False).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        return m.eval()

    base = load(args.baseline)
    hard = load(args.hardened) if args.hardened else None
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept, tried = 0, 0
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        mask = labels == spoof_idx
        if not mask.any():
            continue
        x = imgs[mask]
        with torch.no_grad():
            correctly_caught = base(x) < 0          # only attack fakes the baseline rejects
        x = x[correctly_caught]
        if len(x) == 0:
            continue
        adv = pgd_attack(base, x, eps=args.eps)
        # simulate saving as an 8-bit PNG, then re-score
        pix = torch.clamp(adv * std + mean, 0, 1)
        pix8 = torch.round(pix * 255) / 255
        adv_q = (pix8 - mean) / std
        with torch.no_grad():
            fooled_base = base(adv_q) > 0
            fooled_hard = (hard(adv_q) > 0) if hard else None
        for i in range(len(x)):
            tried += 1
            if not fooled_base[i]:
                continue
            arr = (pix8[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(out / f"attacked_{kept:02d}.png")
            note = "" if fooled_hard is None else f" | hardened fooled: {bool(fooled_hard[i])}"
            print(f"saved attacked_{kept:02d}.png (baseline fooled){note}")
            kept += 1
            if kept >= args.n:
                print(f"done: {kept} images in {out}/ (tried {tried})")
                return
    print(f"done: only {kept} images fooled the baseline (tried {tried}). Try a larger --eps.")


if __name__ == "__main__":
    main()
