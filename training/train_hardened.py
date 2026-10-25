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
from utils.seed import set_seed
from attacks.pgd_attack import pgd_attack
from training.losses import trajectory_consistency_loss, update_centroids
from training.train_baseline import evaluate_acc
from data.dataset import IMAGENET_MEAN, IMAGENET_STD


def bernoulli_kl(logit_p, logit_q):
    """KL(Bern(sigmoid(logit_p)) || Bern(sigmoid(logit_q))), per image, numerically stable."""
    import torch.nn.functional as F
    p = torch.sigmoid(logit_p)
    return p * (F.logsigmoid(logit_p) - F.logsigmoid(logit_q)) + \
        (1 - p) * (F.logsigmoid(-logit_p) - F.logsigmoid(-logit_q))


def trades_adversary(model, x, eps, steps):
    """TRADES inner maximization (Zhang et al., ICML 2019): find x' in the eps-ball that maximizes
    KL(f(x) || f(x')), i.e. the point where the prediction CHANGES most, whatever the label."""
    was_training = model.training
    model.eval()
    dev = x.device
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    lo, hi = (0 - mean) / std, (1 - mean) / std
    eps_t, alpha_t = eps / std, max(eps / 4, 0.5 / 255) / std
    with torch.no_grad():
        clean_logit = model(x)
    adv = (x + 0.001 * torch.randn_like(x)).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        kl = bernoulli_kl(clean_logit, model(adv)).sum()
        g = torch.autograd.grad(kl, adv)[0]
        adv = adv.detach() + alpha_t * g.sign()
        adv = torch.max(torch.min(adv, x + eps_t), x - eps_t)
        adv = torch.max(torch.min(adv, hi), lo).detach()
    model.train(was_training)
    return adv


def robust_val_acc(model, val_loader, device, live_idx, eps, max_batches=8):
    """Accuracy on validation images attacked in the harmful direction (fakes toward live, real toward spoof).
    Returns (overall, attacked-fake accuracy, attacked-real accuracy)."""
    ok = {True: 0, False: 0}
    n = {True: 0, False: 0}
    # The val loader is NOT shuffled and sorted by class (all live, then all spoof), so take evenly
    # spaced batches across the whole set instead of the first few (which would be one class only).
    step = max(1, len(val_loader) // max_batches)
    for i, (x, y) in enumerate(val_loader):
        if i % step != 0:
            continue
        x, y = x.to(device), y.to(device)
        is_live = y == live_idx
        adv = x.clone()
        if (~is_live).any():
            adv[~is_live] = pgd_attack(model, x[~is_live], eps=eps, alpha=max(eps / 4, 0.5 / 255), steps=5, target=1.0)
        if is_live.any():
            adv[is_live] = pgd_attack(model, x[is_live], eps=eps, alpha=max(eps / 4, 0.5 / 255), steps=5, target=0.0)
        model.eval()
        with torch.no_grad():
            correct = (model(adv) > 0) == is_live
        for cls in (True, False):
            ok[cls] += correct[is_live == cls].sum().item()
            n[cls] += (is_live == cls).sum().item()
    fake = ok[False] / max(n[False], 1)
    real = ok[True] / max(n[True], 1)
    return (ok[True] + ok[False]) / max(n[True] + n[False], 1), fake, real


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
    ap.add_argument("--noise_weight", type=float, default=1.0,
                     help="weight of randomly-noised copies (true labels). Teaches 'noise is not an attack signal'; 0 disables")
    ap.add_argument("--fake_adv_weight", type=float, default=None,
                    help="weight of attacked FAKES (default: --adv_weight). Raise it if fakes stay easy to attack")
    ap.add_argument("--real_adv_weight", type=float, default=None,
                    help="weight of attacked REAL faces (default: --adv_weight)")
    ap.add_argument("--attack_steps", type=int, default=10, help="PGD steps used during training")
    ap.add_argument("--split_bn", action="store_true",
                    help="forward clean / attacked / noisy images in separate passes (separate batch-norm statistics)")
    ap.add_argument("--trades_beta", type=float, default=0.0,
                    help="> 0 switches to the TRADES loss: BCE(clean) + beta * KL(f(x) || f(x_adv)). Lower beta "
                         "favours clean accuracy, higher favours robustness. Replaces the adv/noise/reg terms")
    ap.add_argument("--init", default=None,
                    help="start from this checkpoint (e.g. the trained baseline) instead of ImageNet weights")
    ap.add_argument("--no_pretrained", action="store_true", help="skip ImageNet weights (smoke tests)")
    ap.add_argument("--out", default="checkpoints/hardened")
    args = ap.parse_args()

    set_seed(args.seed)
    fake_w = args.adv_weight if args.fake_adv_weight is None else args.fake_adv_weight
    real_w = args.adv_weight if args.real_adv_weight is None else args.real_adv_weight
    print(f"hardened training: lr={args.lr} eps={args.eps:.4f} fake_adv_weight={fake_w} real_adv_weight={real_w} "
          f"noise_weight={args.noise_weight} reg_weight={args.reg_weight} split_bn={args.split_bn} "
          f"trades_beta={args.trades_beta}")
    device = get_device()
    os.makedirs(args.out, exist_ok=True)

    train_loader, val_loader, _, class_to_idx = get_dataloaders(args.data_root, args.batch_size)
    live_idx, spoof_idx = class_to_idx["live"], class_to_idx["spoof"]

    model = build_model(args.arch, pretrained=not args.no_pretrained and not args.init).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"initialized from {args.init}")
    opt = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    centroids = {
        "live": torch.zeros(model.feat_dim, device=device),
        "spoof": torch.zeros(model.feat_dim, device=device),
    }

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            live_labels = (labels == live_idx).float()
            if args.trades_beta > 0:
                adv = trades_adversary(model, imgs, args.eps, args.attack_steps)
                opt.zero_grad()
                clean_logit, adv_logit = model(imgs), model(adv)
                loss = criterion(clean_logit, live_labels).mean() + \
                    args.trades_beta * bernoulli_kl(clean_logit, adv_logit).mean()
                loss.backward()
                opt.step()
                continue

            # Each group = (images, true live-labels, per-image loss weights). Every group mixes BOTH classes:
            # a batch-norm pass over a single-class batch normalizes away the live/spoof difference itself.
            ones = torch.ones_like(live_labels)
            groups = [(imgs, live_labels, ones)]
            if args.adv_weight > 0:
                # Attack BOTH classes: fakes pushed toward "live" (label stays fake), real faces toward "spoof"
                # (label stays real). Attacking only fakes lets the model learn "attack noise = fake".
                adv_x, adv_y, adv_w = [], [], []
                for mask, aim, true_live, w in ((labels == spoof_idx, 1.0, 0.0, fake_w),
                                                (labels == live_idx, 0.0, 1.0, real_w)):
                    if mask.any() and w > 0:
                        adv = pgd_attack(model, imgs[mask], eps=args.eps, alpha=max(args.eps / 4, 0.5 / 255),
                                         steps=args.attack_steps, target=aim)
                        adv_x.append(adv)
                        adv_y.append(torch.full((adv.size(0),), true_live, device=device))
                        adv_w.append(torch.full((adv.size(0),), float(w), device=device))
                if adv_x:
                    groups.append((torch.cat(adv_x), torch.cat(adv_y), torch.cat(adv_w)))
            if args.noise_weight > 0 and args.adv_weight > 0:
                # Control against the shortcut "perturbed = fake": random +-eps noise on BOTH classes, true labels.
                m = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
                sd = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
                noise = (torch.randint(0, 2, imgs.shape, device=device).float() * 2 - 1) * args.eps / sd
                noisy = torch.max(torch.min(imgs + noise, (1 - m) / sd), (0 - m) / sd)
                groups.append((noisy, live_labels, ones * args.noise_weight))

            live_labels = torch.cat([g[1] for g in groups])
            weights = torch.cat([g[2] for g in groups])

            opt.zero_grad()
            if args.split_bn:
                # One forward pass per group (clean / attacked / noisy), each containing both classes, so
                # batch-norm statistics are never computed over a mix of clean and attacked images.
                outs = [model(g[0], return_features=True) for g in groups]
                logits = torch.cat([o[0] for o in outs])
                feats = torch.cat([o[1] for o in outs])
            else:
                logits, feats = model(torch.cat([g[0] for g in groups]), return_features=True)
            loss = (criterion(logits, live_labels) * weights).sum() / weights.sum()
            if args.reg_weight > 0:
                loss = loss + args.reg_weight * trajectory_consistency_loss(
                    feats, live_labels.long(), centroids
                )
                update_centroids(centroids, feats.detach(), live_labels.long())
            loss.backward()
            opt.step()

        val_acc = evaluate_acc(model, val_loader, device, live_idx)
        if args.adv_weight > 0 or args.trades_beta > 0:
            rob, rob_fake, rob_real = robust_val_acc(model, val_loader, device, live_idx, args.eps)
        else:
            rob = rob_fake = rob_real = float("nan")
        # Keep the epoch that is good on BOTH clean and attacked validation images (weaker of the two).
        score = min(val_acc, rob) if (args.adv_weight > 0 or args.trades_beta > 0) else val_acc
        print(f"epoch {epoch} done: clean val_acc={val_acc:.4f}  attacked val_acc={rob:.4f} "
              f"(fakes {rob_fake:.4f}, real {rob_real:.4f})  score={score:.4f}", flush=True)
        torch.save(model.state_dict(), f"{args.out}/seed{args.seed}_epoch{epoch}.pt")
        if score > best:
            best = score
            torch.save(model.state_dict(), f"{args.out}/seed{args.seed}_best.pt")
            print(f"  -> new best, saved seed{args.seed}_best.pt")


if __name__ == "__main__":
    main()
