"""
Simplified 'live-to-spoof trajectory alignment' regularizer, inspired by
Sun et al., CVPR 2023 ("Rethinking Domain Generalization for Face
Anti-Spoofing: Separability and Alignment").

IMPORTANT SCOPE NOTE (say this in the report too): the original paper
aligns trajectories across multiple labeled SOURCE DOMAINS. This
project has no multi-domain labels, so this is a single-domain
approximation -- it penalizes batch-to-batch drift in the live-centroid
-> spoof-centroid direction using an EMA reference, which is NOT the
full method. Report it as an inspired simplification, not a
reproduction.
"""
import torch


def trajectory_consistency_loss(features, live_labels, momentum_centroids):
    """
    features: (B, D) penultimate embeddings
    live_labels: (B,) 1=live, 0=spoof
    momentum_centroids: dict with running 'live' / 'spoof' centroid (D,) tensors
    """
    live_feats = features[live_labels == 1]
    spoof_feats = features[live_labels == 0]
    if live_feats.numel() == 0 or spoof_feats.numel() == 0:
        return torch.tensor(0.0, device=features.device)

    batch_dir = live_feats.mean(0) - spoof_feats.mean(0)
    batch_dir = batch_dir / (batch_dir.norm() + 1e-8)

    ref_dir = momentum_centroids["live"] - momentum_centroids["spoof"]
    ref_dir = ref_dir / (ref_dir.norm() + 1e-8)

    return 1 - torch.dot(batch_dir, ref_dir.detach())


def update_centroids(centroids, features, live_labels, momentum: float = 0.9):
    with torch.no_grad():
        if (live_labels == 1).any():
            centroids["live"] = momentum * centroids["live"] + (1 - momentum) * features[live_labels == 1].mean(0)
        if (live_labels == 0).any():
            centroids["spoof"] = momentum * centroids["spoof"] + (1 - momentum) * features[live_labels == 0].mean(0)
