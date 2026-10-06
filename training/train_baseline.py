"""
Phase 1 (proposal timeline): train the clean baseline PAD classifier.

Run on Colab/Kaggle GPU. Saves a checkpoint every epoch so a
disconnect doesn't cost you the whole run -- copy checkpoints/ to
Drive or a Kaggle Dataset after each session.

Usage:
    python -m training.train_baseline --data_root ./data_pad --seed 0
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.optim import AdamW

from data.dataset import get_dataloaders
from models.baseline import build_model
from utils.device import get_device


def evaluate_acc(model, loader, device, live_idx):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            live_labels = (labels == live_idx).float()
            preds = (torch.sigmoid(model(imgs)) > 0.5).float()
            correct += (preds == live_labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_pretrained", action="store_true", help="skip ImageNet weights (smoke tests)")
    ap.add_argument("--out", default="checkpoints/baseline")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    os.makedirs(args.out, exist_ok=True)

    train_loader, val_loader, _, class_to_idx = get_dataloaders(args.data_root, args.batch_size)
    live_idx = class_to_idx["live"]

    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
    opt = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            live_labels = (labels == live_idx).float()
            opt.zero_grad()
            loss = criterion(model(imgs), live_labels)
            loss.backward()
            opt.step()

        val_acc = evaluate_acc(model, val_loader, device, live_idx)
        print(f"epoch {epoch}: val_acc={val_acc:.4f}")
        torch.save(model.state_dict(), f"{args.out}/seed{args.seed}_epoch{epoch}.pt")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{args.out}/seed{args.seed}_best.pt")


if __name__ == "__main__":
    main()
