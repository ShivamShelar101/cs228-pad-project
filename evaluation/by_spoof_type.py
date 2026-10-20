"""
Where does the baseline fail on the test set? Breaks the TEST fakes down by what kind of
fake they are (spoof type, lighting, indoor/outdoor) and prints the share the model accepts as
live for each group. Also compares with the same breakdown of the training fakes.

    python -m evaluation.by_spoof_type --src $SRC --data_root ./data_pad2 \
        --ckpt checkpoints/baseline2/seed0_best.pt
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from data.dataset import get_transforms
from data.prepare_celeba_spoof import find_root
from models.baseline import build_model
from utils.device import get_device

TYPES = {1: "photo", 2: "poster", 3: "A4 print", 4: "face mask", 5: "upper-body mask",
         6: "region mask", 7: "PC screen", 8: "pad screen", 9: "phone screen", 10: "3D mask"}
LIGHT = {1: "normal", 2: "strong", 3: "back", 4: "dark"}
ENV = {1: "indoor", 2: "outdoor"}


def predict(model, ds, device):
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out += (model(x.to(device)) > 0).cpu().tolist()
    return out


def original_key(split, path, labels):
    subject, stem = Path(path).stem.split("_", 1)
    for ext in (".jpg", ".png", ".jpeg"):
        key = f"Data/{split}/{subject}/spoof/{stem}{ext}"
        if key in labels:
            return key
    return None


def report(title, rows):
    print(f"\n{title}")
    print(f"  {'group':<18}{'images':>8}{'accepted as live':>20}")
    for name, (n, acc) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name:<18}{n:>8}{100 * acc / n:>19.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="resnet18")
    args = ap.parse_args()

    device = get_device()
    model = build_model(args.arch, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    root = find_root(args.src)

    for split, label_file in (("test", "test_label.json"), ("train", "train_label.json")):
        ds = datasets.ImageFolder(f"{args.data_root}/{split}", transform=get_transforms(False))
        spoof_idx = ds.class_to_idx["spoof"]
        with open(root / "metas" / "intra_test" / label_file) as f:
            labels = json.load(f)
        pred_live = predict(model, ds, device)
        groups = {"spoof type": defaultdict(lambda: [0, 0]), "lighting": defaultdict(lambda: [0, 0]),
                  "environment": defaultdict(lambda: [0, 0]), "all fakes": defaultdict(lambda: [0, 0])}
        for (path, target), live in zip(ds.samples, pred_live):
            if target != spoof_idx:
                continue
            key = original_key(split, path, labels)
            if key is None:
                continue
            lab = labels[key]
            for g, name in (("spoof type", TYPES.get(int(lab[40]), str(lab[40]))),
                            ("lighting", LIGHT.get(int(lab[41]), str(lab[41]))),
                            ("environment", ENV.get(int(lab[42]), str(lab[42]))),
                            ("all fakes", "all")):
                groups[g][name][0] += 1
                groups[g][name][1] += int(live)
        print(f"\n===== {split.upper()} fakes: share {Path(args.ckpt).parent.name} accepts as live =====")
        for g, rows in groups.items():
            report(g, {k: tuple(v) for k, v in rows.items()})


if __name__ == "__main__":
    main()
