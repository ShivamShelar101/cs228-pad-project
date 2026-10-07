"""
Phase 3 (proposal timeline): adversarially-hardened training.

Combines adversarial training (mixing in PGD-perturbed images of BOTH classes
each batch) with the trajectory-consistency regularizer from losses.py.

For the seed/ablation rigor pass, run this multiple times with
different --seed and with --adv_weight 0 / --reg_weight 0 to isolate
which component is doing the work -- launch each as a separate
Kaggle session rather than one long job (see README "Rigor pass").

Usage:
    python -m training.train_hardened --data_root ./data_pad --seed 0
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.optim import AdamW

from data.dataset import get_dataloaders
from models.baseline import build_model
from utils.device import get_device
from attacks.pgd_attack import pgd_attack
from training.losses import trajectory_consistency_loss, update_centroids
from training.train_baseline import evaluate_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adv_weight", type=float, default=1.0,
                     help="loss weight of the attacked images (1.0 = same as clean ones, 0 = no adversarial training)")
    ap.add_argument("--reg_weight", type=float, default=0.1,
                     help="0 disables the trajectory regularizer (ablation)")
    ap.add_argument("--eps", type=float, default=8 / 255, help="training attack budget in pixel units")
    ap.add_argument("--no_pretrained", action="store_true", help="skip ImageNet weights (smoke tests)")
    ap.add_argument("--out", default="checkpoints/hardened")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    os.makedirs(args.out, exist_ok=True)

    train_loader, val_loader, _, class_to_idx = get_dataloaders(args.data_root, args.batch_size)
    live_idx, spoof_idx = class_to_idx["live"], class_to_idx["spoof"]

    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
    opt = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    centroids = {
        "live": torch.zeros(model.feat_dim, device=device),
        "spoof": torch.zeros(model.feat_dim, device=device),
    }

    for epoch in range(args.epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            live_labels = (labels == live_idx).float()
            weights = torch.ones(imgs.size(0), device=device)

            # Attack BOTH classes: fakes are pushed toward "live" (label stays fake), real faces are
            # pushed toward "spoof" (label stays real). If only fakes were attacked, the model could
            # learn the shortcut "attack noise = fake" instead of becoming robust.
            clean_imgs = imgs
            extra_imgs, extra_labels = [], []
            if args.adv_weight > 0:
                for mask, aim, true_live in ((labels == spoof_idx, 1.0, 0.0), (labels == live_idx, 0.0, 1.0)):
                    if mask.any():
                        adv = pgd_attack(model, clean_imgs[mask], eps=args.eps, target=aim)
                        extra_imgs.append(adv)
                        extra_labels.append(torch.full((adv.size(0),), true_live, device=device))
            if extra_imgs:
                imgs = torch.cat([clean_imgs] + extra_imgs, dim=0)
                live_labels = torch.cat([live_labels] + extra_labels)
                weights = torch.cat([weights] + [torch.full_like(l, args.adv_weight) for l in extra_labels])

            opt.zero_grad()
            logits, feats = model(imgs, return_features=True)
            loss = (criterion(logits, live_labels) * weights).sum() / weights.sum()
            if args.reg_weight > 0:
                loss = loss + args.reg_weight * trajectory_consistency_loss(
                    feats, live_labels.long(), centroids
                )
                update_centroids(centroids, feats.detach(), live_labels.long())
            loss.backward()
            opt.step()

        val_acc = evaluate_acc(model, val_loader, device, live_idx)
        print(f"epoch {epoch} done: clean val_acc={val_acc:.4f}")
        torch.save(model.state_dict(), f"{args.out}/seed{args.seed}_epoch{epoch}.pt")


if __name__ == "__main__":
    main()
