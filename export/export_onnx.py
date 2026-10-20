"""
Phase 5 (proposal timeline): export the best hardened checkpoint to
ONNX for local CPU inference on the M1 via onnxruntime
(app/backend/main.py).

Usage:
    python -m export.export_onnx \
        --ckpt checkpoints/hardened/seed0_epoch14.pt \
        --out app/backend/model.onnx
"""
import argparse

import torch

from models.baseline import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--out", default="app/backend/model.onnx")
    args = ap.parse_args()

    model = build_model(args.arch, pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    kwargs = dict(
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
    )
    try:
        # classic exporter: one self-contained .onnx file (what the backend expects)
        torch.onnx.export(model, dummy, args.out, opset_version=17, dynamo=False, **kwargs)
    except Exception as e:  # newer torch may drop the classic exporter
        print("classic exporter unavailable, using the new one:", type(e).__name__)
        torch.onnx.export(model, dummy, args.out, external_data=False, **kwargs)
    print(f"exported to {args.out}")


if __name__ == "__main__":
    main()
