#!/usr/bin/env bash
# One-time Kiosk Guard setup with the final models (data v3, seed 2: the seed chosen on validation).
#   bash scripts/prepare_demo_mac.sh
# Writes app/backend/model_baseline.onnx, model_hardened.onnx, thresholds.json (validation operating
# points), and demo_images/ (clean + attacked pairs with summary.json). Then start the app with:
#   cd app/backend && uvicorn main:app --port 8000          (terminal 1)
#   cd app/frontend && npm install && npm run dev            (terminal 2)
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${SEED:-2}
BASE=checkpoints/baseline3/seed${SEED}_best.pt
HARD=checkpoints/eps2_ft3/seed${SEED}_best.pt
[ -d .venv ] && source .venv/bin/activate
pip install -q -r app/backend/requirements.txt
python -m export.export_onnx --ckpt "$BASE" --out app/backend/model_baseline.onnx
python -m export.export_onnx --ckpt "$HARD" --out app/backend/model_hardened.onnx
python - "$BASE" "$HARD" <<'PY'
import json, sys
base, hard = sys.argv[1], sys.argv[2]
def op(path, ck):
    for r in json.load(open(path))["per_seed"]:
        if r["checkpoint"] == ck:
            return round(r["operating_point"]["threshold"], 4)
    raise SystemExit(f"no threshold for {ck} in {path}")
t = {"baseline": op("results/data3/threshold_baseline3/threshold.json", base),
     "hardened": op("results/data3/threshold_eps2_ft3/threshold.json", hard)}
json.dump(t, open("app/backend/thresholds.json", "w"), indent=2)
print("thresholds (chosen on validation):", t)
PY
python -m scripts.make_attacked_images --data_root ./data_pad3 --baseline "$BASE" --hardened "$HARD" \
    --thresholds app/backend/thresholds.json --n 8 --out demo_images
echo
echo "Ready. Terminal 1:  cd app/backend && uvicorn main:app --port 8000"
echo "       Terminal 2:  cd app/frontend && npm install && npm run dev   (then open the address it prints)"
