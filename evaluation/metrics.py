"""
Standard FAS metrics.
APCER: Attack Presentation Classification Error Rate (spoof called live)
BPCER: Bona-fide Presentation Classification Error Rate (live called spoof)
ACER:  Average Classification Error Rate = (APCER + BPCER) / 2
"""
import torch


def compute_pad_metrics(model, loader, device, live_idx, threshold: float = 0.5):
    model.eval()
    spoof_wrong, spoof_total = 0, 0
    live_wrong, live_total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            live_labels = (labels == live_idx).float()
            probs = torch.sigmoid(model(imgs))
            preds_live = (probs > threshold).float()

            spoof_mask = live_labels == 0
            live_mask = live_labels == 1
            spoof_wrong += (preds_live[spoof_mask] == 1).sum().item()
            spoof_total += spoof_mask.sum().item()
            live_wrong += (preds_live[live_mask] == 0).sum().item()
            live_total += live_mask.sum().item()

    apcer = spoof_wrong / max(spoof_total, 1)
    bpcer = live_wrong / max(live_total, 1)
    acer = (apcer + bpcer) / 2
    return {"APCER": apcer, "BPCER": bpcer, "ACER": acer}
