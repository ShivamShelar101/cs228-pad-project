#!/usr/bin/env bash
# Defense run on a Mac: adversarial fine-tuning at 2/255 from each baseline seed (the E2 configuration),
# then the full robustness evaluation (budget sweep 1-16/255, conditional and worst-case attack success),
# threshold analysis and per-type error analysis. Needs the checkpoint run first (baselines + data).
#
#   caffeinate -i bash scripts/run_defense_mac.sh            # seeds 0 1 2 (long: run overnight)
#   SEEDS="0" caffeinate -i bash scripts/run_defense_mac.sh  # one seed first
# Rerunning skips seeds already trained. Output: results_defense.zip (small, no checkpoints).
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2"}
EPOCHS=${EPOCHS:-8}
EXTRA=${EXTRA:-}
log() { printf '\n==== %s  %s\n' "$(date +%H:%M:%S)" "$*"; }

[ -d .venv ] && source .venv/bin/activate
python -c "from utils.device import get_device; print('device:', get_device())"
[ -d data_pad2 ] || { echo "data_pad2 missing: run scripts/run_checkpoint_mac.sh first"; exit 1; }

for s in $SEEDS; do
  [ -f "checkpoints/baseline/seed${s}_best.pt" ] || { echo "baseline seed $s missing: run the checkpoint script first"; exit 1; }
  if [ -f "checkpoints/eps2_ft/.seed${s}_done" ]; then echo "defense seed $s already trained, skipping"; continue; fi
  log "Defense training, seed $s: PGD adversarial fine-tuning at 2/255, $EPOCHS epochs"
  python -m training.train_hardened --data_root ./data_pad2 --epochs "$EPOCHS" --lr 2e-5 --seed "$s" \
      --eps 0.00784 --init "checkpoints/baseline/seed${s}_best.pt" --out checkpoints/eps2_ft $EXTRA
  touch "checkpoints/eps2_ft/.seed${s}_done"
done

mkdir -p results/defense
log "Robustness evaluation on test: budget sweep 1, 2, 4, 8, 16 / 255"
python -m evaluation.evaluate --data_root ./data_pad2 --sweep 1 2 4 8 16 \
    --baseline_glob "checkpoints/baseline/seed[$(echo $SEEDS | tr -d ' ')]_best.pt" \
    --hardened_glob "checkpoints/eps2_ft/seed[$(echo $SEEDS | tr -d ' ')]_best.pt" \
    --out results/defense/eval_eps2_ft.json
python -m evaluation.make_report --run "Adversarial fine-tuning at 2/255=results/defense/eval_eps2_ft.json" \
    --out results/defense/report

log "Threshold analysis of the defended models (operating point chosen on validation)"
python -m evaluation.threshold_analysis --data_root ./data_pad2 \
    --ckpt_glob "checkpoints/eps2_ft/seed[$(echo $SEEDS | tr -d ' ')]_best.pt" --label "Defended model" --out results/defense/threshold

log "Error analysis by spoof type, defended model"
FIRST=$(echo $SEEDS | awk '{print $1}')
python -m evaluation.by_spoof_type --src ./celeba_meta --data_root ./data_pad2 \
    --ckpt "checkpoints/eps2_ft/seed${FIRST}_best.pt" | tee results/defense/by_spoof_type_seed${FIRST}.txt

log "Bundle"
{ git rev-parse HEAD 2>/dev/null || echo "not a git checkout"; echo "device: $(python -c 'from utils.device import get_device; print(get_device())')"; echo "seeds: $SEEDS"; } | tee results/defense/COMMIT.txt
rm -f results_defense.zip && zip -qr results_defense.zip results/defense && ls -lh results_defense.zip
cat results/defense/report.md
log "Done. Upload results_defense.zip to Claude."
