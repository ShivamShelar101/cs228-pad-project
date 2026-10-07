"""
Digital-domain PGD attack against the PAD classifier.

This is the digital stand-in for AdvGen described in the proposal (Module 3):
given SPOOF images the classifier correctly rejects, perturb them under an
L_inf budget until the classifier calls them 'live'. There is no print or
replay step here, which is the explicit scope cut noted in the proposal.

The budget `eps` is in real pixel units (8/255 = 8 gray levels out of 255).
Images arrive ImageNet-normalized, so the budget is converted per channel and
the result is kept inside the valid [0, 1] pixel range.
"""
import torch
import torch.nn.functional as F

from data.dataset import IMAGENET_MEAN, IMAGENET_STD


def pgd_attack(model, images, eps=8 / 255, alpha=2 / 255, steps=10, target=1.0):
    """images: (B,3,H,W) normalized. target=1.0 pushes toward 'live' (attack on a fake),
    target=0.0 pushes toward 'spoof' (attack on a real face). Returns adversarial images."""
    was_training = model.training
    model.eval()

    device = images.device
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    lo, hi = (0.0 - mean) / std, (1.0 - mean) / std      # valid pixel range, normalized
    eps_t, alpha_t = eps / std, alpha / std               # budget in normalized units

    images = images.clone().detach()
    adv = images.clone().detach().requires_grad_(True)
    target_live = torch.full((images.size(0),), float(target), device=device)

    for _ in range(steps):
        logits = model(adv)
        loss = F.binary_cross_entropy_with_logits(logits, target_live)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() - alpha_t * grad.sign()        # descend loss -> push toward 'live'
        adv = torch.max(torch.min(adv, images + eps_t), images - eps_t)  # stay inside eps-ball
        adv = torch.max(torch.min(adv, hi), lo).detach()                 # stay a valid image
        adv.requires_grad_(True)

    model.train(was_training)  # leave the model how we found it
    return adv.detach()


def attack_success_rate(model, adv_images, threshold: float = 0.0) -> float:
    """Fraction of adversarial spoof images now classified 'live' (logit > threshold)."""
    model.eval()
    with torch.no_grad():
        logits = model(adv_images)
        preds_live = (logits > threshold).float()
    return preds_live.mean().item()
