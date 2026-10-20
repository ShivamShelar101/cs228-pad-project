import torch.nn as nn
from torchvision import models


class PADClassifier(nn.Module):
    """
    Binary live-vs-spoof classifier. forward() returns a single logit
    (sigmoid -> P(live)); pass return_features=True to also get the
    penultimate embedding, needed by the trajectory-consistency
    regularizer in training/losses.py.
    """

    def __init__(self, arch: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if arch == "resnet18":
            backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            self.feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif arch == "mobilenet_v2":
            backbone = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
            self.feat_dim = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"unknown arch {arch}")
        self.backbone = backbone
        self.head = nn.Linear(self.feat_dim, 1)

    def forward(self, x, return_features: bool = False):
        feats = self.backbone(x)
        logits = self.head(feats).squeeze(1)
        if return_features:
            return logits, feats
        return logits


def build_model(arch: str = "resnet18", pretrained: bool = True) -> PADClassifier:
    return PADClassifier(arch, pretrained)
