#!/usr/bin/env bash
# Retrain on data v4 (3x more training data, more PC-screen replays in train/val, SAME test set as data_pad2)
# and compare with the data_pad2 models.
#   caffeinate -i bash scripts/run_data3_mac.sh [path/to/mac_data4.zip]
# Defaults: baselines on seeds 0 1 2, defense on seed 0 (about 5 hours). Then, for the other defense seeds:
#   DEF_SEEDS="0 1 2" caffeinate -i bash scripts/run_data3_mac.sh
# Rerunning skips finished work. Output: results_data4.zip
set -euo pipefail
cd "$(dirname "$0")/.."
ZIP="${1:-$HOME/Downloads/mac_data4.zip}"
SEEDS=${SEEDS:-"0 1 2"}
DEF_SEEDS=${DEF_SEEDS:-"0"}
EXTRA=${EXTRA:-}
OUT=results/data4
log() { printf '\n==== %s  %s\n' "$(date +%H:%M:%S)" "$*"; }
[ -d .venv ] && source .venv/bin/activate
python -c "from utils.device import get_device; print('device:', get_device())"
if [ ! -d data_pad4 ]; then
  [ -f "$ZIP" ] || { echo "mac_data4.zip not found at $ZIP"; exit 1; }
  unzip -q -o "$ZIP" -d .
fi
for s in train val test; do for c in live spoof; do echo "$s/$c $(ls data_pad4/$s/$c | wc -l)"; done; done
mkdir -p "$OUT"

for s in $SEEDS; do
  if [ -f "checkpoints/baseline4/.seed${s}_done" ]; then echo "baseline4 seed $s done, skipping"; continue; fi
  log "Baseline on data v4, seed $s"
  python -m training.train_baseline --data_root ./data_pad4 --epochs 5 --seed "$s" --out checkpoints/baseline4 $EXTRA
  rm -f checkpoints/*/seed*_epoch*.pt; touch "checkpoints/baseline4/.seed${s}_done"
done
SG="[$(echo $SEEDS | tr -d ' ')]"
log "Threshold analysis, data v4 baselines"
python -m evaluation.threshold_analysis --data_root ./data_pad4 --ckpt_glob "checkpoints/baseline4/seed${SG}_best.pt" --out "$OUT/threshold_baseline4"

for s in $DEF_SEEDS; do
  if [ -f "checkpoints/eps2_ft4/.seed${s}_done" ]; then echo "eps2_ft4 seed $s done, skipping"; continue; fi
  log "Defense on data v4 (adv. fine-tuning 2/255), seed $s"
  python -m training.train_hardened --data_root ./data_pad4 --epochs 8 --lr 2e-5 --seed "$s" --eps 0.00784 \
      --init "checkpoints/baseline4/seed${s}_best.pt" --out checkpoints/eps2_ft4 $EXTRA
  rm -f checkpoints/*/seed*_epoch*.pt; touch "checkpoints/eps2_ft4/.seed${s}_done"
done
DG="[$(echo $DEF_SEEDS | tr -d ' ')]"

log "Test evaluation (same test images as data_pad2)"
python -m evaluation.evaluate --data_root ./data_pad4 --sweep 1 2 4 8 16 \
    --baseline_glob "checkpoints/baseline4/seed${DG}_best.pt" --hardened_glob "checkpoints/eps2_ft4/seed${DG}_best.pt" \
    --out "$OUT/test_eps2_ft4.json"
python -m evaluation.make_report --run "Data v4: adv. fine-tuning 2/255=$OUT/test_eps2_ft4.json" --out "$OUT/report"
python -m evaluation.threshold_analysis --data_root ./data_pad4 --ckpt_glob "checkpoints/eps2_ft4/seed${DG}_best.pt" --label "Defended model" --out "$OUT/threshold_eps2_ft4"
for v in baseline4 eps2_ft4; do for s in $DEF_SEEDS; do
  python -m evaluation.by_spoof_type --src ./celeba_meta --data_root ./data_pad4 --ckpt "checkpoints/$v/seed${s}_best.pt" > "$OUT/by_spoof_type_${v}_seed${s}.txt"
done; done
python -m evaluation.eda --src ./celeba_meta --data_root ./data_pad4 --out "$OUT/eda" > "$OUT/eda_log.txt"

log "Bundle"
{ git rev-parse HEAD 2>/dev/null || echo "not a git checkout"; echo "device: $(python -c 'from utils.device import get_device; print(get_device())')"; echo "baseline seeds: $SEEDS; defense seeds: $DEF_SEEDS"; } | tee "$OUT/COMMIT.txt"
rm -f results_data4.zip && zip -qr results_data4.zip "$OUT" && ls -lh results_data4.zip
cat "$OUT/report.md"
log "Done. Upload results_data4.zip to Claude."
