#!/usr/bin/env bash
# Runs the whole checkpoint (3 seeded baselines, threshold analysis, EDA, error analysis) on a Mac.
# Uses the Apple GPU (MPS) automatically. Output: bundle_checkpoint.zip in the repo root.
#
# Usage, from the repo root:
#   bash scripts/run_checkpoint_mac.sh [path/to/mac_data.zip]     (default: ~/Downloads/mac_data.zip)
set -euo pipefail
cd "$(dirname "$0")/.."
ZIP="${1:-$HOME/Downloads/mac_data.zip}"
log() { printf '\n==== %s  %s\n' "$(date +%H:%M:%S)" "$*"; }

log "1/7 Python environment"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Need Python 3.10+ (brew install python@3.12)"'
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
python -c "from utils.device import get_device; print('device:', get_device())"

log "2/7 Data"
if [ ! -d data_pad2 ] || [ ! -d celeba_meta ]; then
  [ -f "$ZIP" ] || { echo "mac_data.zip not found at $ZIP (pass its path as the first argument)"; exit 1; }
  unzip -q -o "$ZIP" -d .
fi
for s in train val test; do for c in live spoof; do echo "$s/$c $(ls data_pad2/$s/$c | wc -l)"; done; done

log "3/7 Baseline: ResNet-18, 5 epochs, seeds 0 1 2"
for s in 0 1 2; do
  if [ -f "checkpoints/baseline/seed${s}_best.pt" ] && [ -f "checkpoints/baseline/.seed${s}_done" ]; then
    echo "seed $s already trained, skipping"; continue
  fi
  python -m training.train_baseline --data_root ./data_pad2 --epochs 5 --seed "$s" --out checkpoints/baseline
  touch "checkpoints/baseline/.seed${s}_done"
done

log "4/7 Threshold analysis"
python -m evaluation.threshold_analysis --data_root ./data_pad2 --ckpt_glob "checkpoints/baseline/seed*_best.pt" --out results/threshold

log "5/7 EDA and data-quality checks"
python -m evaluation.eda --src ./celeba_meta --data_root ./data_pad2 --out results/eda

log "6/7 Error analysis by spoof type (seed 0)"
python -m evaluation.by_spoof_type --src ./celeba_meta --data_root ./data_pad2 \
  --ckpt checkpoints/baseline/seed0_best.pt | tee results/by_spoof_type_seed0.txt

log "7/7 Bundle"
{ git rev-parse HEAD 2>/dev/null || echo "not a git checkout"; echo "device: $(python -c 'from utils.device import get_device; print(get_device())')"; } | tee results/COMMIT.txt
rm -f bundle_checkpoint.zip
zip -qr bundle_checkpoint.zip results checkpoints/baseline/seed0_best.pt checkpoints/baseline/seed1_best.pt checkpoints/baseline/seed2_best.pt
ls -lh bundle_checkpoint.zip
log "Done. Upload bundle_checkpoint.zip to Claude."
