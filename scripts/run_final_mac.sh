#!/usr/bin/env bash
# Final-milestone experiments on a Mac (run after run_checkpoint_mac.sh and run_defense_mac.sh).
#   1. Two variants aimed at the defense's clean-accuracy cost, plus an ablation of the regularizer,
#      each fine-tuned from baseline seed 0:
#        light    attacked images weighted 0.5 instead of 1.0
#        eps1     training budget 1/255 instead of 2/255
#        advonly  ablation: no trajectory regularizer (reg_weight 0)
#   2. Picks the best defense on VALIDATION only (never test): lowest mean of clean ACER and
#      attack success at 2/255 on caught fakes, among eps2_ft (the current defense), light and eps1.
#   3. Trains the winner on seeds 1 and 2 (skipped if the winner is eps2_ft, already done).
#   4. Test evaluation of every model, threshold analysis and per-type errors for all seeds,
#      final comparison table and trade-off chart. Output: results_final.zip
#
#   caffeinate -i bash scripts/run_final_mac.sh        (about 7-9 hours; rerunning skips finished work)
set -euo pipefail
cd "$(dirname "$0")/.."
EPOCHS=${EPOCHS:-8}
EXTRA=${EXTRA:-}
OUT=results/final
log() { printf '\n==== %s  %s\n' "$(date +%H:%M:%S)" "$*"; }
[ -d .venv ] && source .venv/bin/activate
python -c "from utils.device import get_device; print('device:', get_device())"
for s in 0 1 2; do
  [ -f "checkpoints/baseline/seed${s}_best.pt" ] || { echo "baseline seed $s missing: run run_checkpoint_mac.sh"; exit 1; }
  [ -f "checkpoints/eps2_ft/seed${s}_best.pt" ] || { echo "defense seed $s missing: run run_defense_mac.sh"; exit 1; }
done
mkdir -p "$OUT"

variant_args() {
  case "$1" in
    eps2_ft) echo "" ;;
    light)   echo "--adv_weight 0.5" ;;
    eps1)    echo "--eps 0.00392" ;;
    advonly) echo "--reg_weight 0" ;;
  esac
}
train() {  # variant seed
  local v=$1 s=$2
  if [ -f "checkpoints/$v/.seed${s}_done" ]; then echo "$v seed $s already trained, skipping"; return; fi
  log "Train $v, seed $s ($EPOCHS epochs)"
  python -m training.train_hardened --data_root ./data_pad2 --epochs "$EPOCHS" --lr 2e-5 --seed "$s" \
      --eps 0.00784 --init "checkpoints/baseline/seed${s}_best.pt" --out "checkpoints/$v" $(variant_args "$v") $EXTRA
  touch "checkpoints/$v/.seed${s}_done"
}
evaluate() {  # variant split out [seeds-glob]
  local v=$1 split=$2 out=$3 sg=${4:-"[012]"}
  [ -f "$out" ] && { echo "$out exists, skipping"; return; }
  python -m evaluation.evaluate --data_root ./data_pad2 --split "$split" --sweep 1 2 4 8 16 \
      --baseline_glob "checkpoints/baseline/seed${sg}_best.pt" --hardened_glob "checkpoints/$v/seed${sg}_best.pt" --out "$out"
}

for v in light eps1 advonly; do train "$v" 0; done

log "Select the defense on validation (seed 0)"
for v in eps2_ft light eps1; do evaluate "$v" val "$OUT/val_$v.json" 0; done
WIN=$(python - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]; best = None
for v in ("eps2_ft", "light", "eps1"):
    s = json.load(open(f"{out}/val_{v}.json"))["hardened"]["summary"]
    score = 0.5 * s["ACER"]["mean"] + 0.5 * s["ASRc_eps2"]["mean"]
    print(f"{v}: val ACER {s['ACER']['mean']:.3f}  val attack success 2/255 {s['ASRc_eps2']['mean']:.3f}  score {score:.3f}", file=sys.stderr)
    if best is None or score < best[1]: best = (v, score)
print(best[0])
PY
)
echo "selected on validation: $WIN" | tee "$OUT/selected.txt"

for s in 1 2; do train "$WIN" "$s"; done

log "Test evaluation of every model"
evaluate eps2_ft test "$OUT/test_eps2_ft.json"
for v in light eps1 advonly; do
  if [ "$v" = "$WIN" ]; then evaluate "$v" test "$OUT/test_$v.json"; else evaluate "$v" test "$OUT/test_$v.json" 0; fi
done
python -m evaluation.final_table \
  --run "Adv. fine-tuning 2/255=$OUT/test_eps2_ft.json" \
  --run "Lighter adv. weight (0.5)=$OUT/test_light.json" \
  --run "Training budget 1/255=$OUT/test_eps1.json" \
  --run "No regularizer (ablation)=$OUT/test_advonly.json" \
  --out "$OUT/final"

log "Threshold analysis and per-type errors for the selected defense ($WIN), all seeds"
python -m evaluation.threshold_analysis --data_root ./data_pad2 --ckpt_glob "checkpoints/$WIN/seed[012]_best.pt" --out "$OUT/threshold_$WIN"
for v in eps2_ft "$WIN"; do for s in 0 1 2; do
  f="$OUT/by_spoof_type_${v}_seed${s}.txt"; [ -f "$f" ] && continue
  python -m evaluation.by_spoof_type --src ./celeba_meta --data_root ./data_pad2 --ckpt "checkpoints/$v/seed${s}_best.pt" > "$f"
done; done
for s in 0 1 2; do
  f="$OUT/by_spoof_type_baseline_seed${s}.txt"; [ -f "$f" ] || \
  python -m evaluation.by_spoof_type --src ./celeba_meta --data_root ./data_pad2 --ckpt "checkpoints/baseline/seed${s}_best.pt" > "$f"
done

log "Bundle"
{ git rev-parse HEAD 2>/dev/null || echo "not a git checkout"; echo "device: $(python -c 'from utils.device import get_device; print(get_device())')"; echo "selected: $WIN"; } | tee "$OUT/COMMIT.txt"
rm -f results_final.zip && zip -qr results_final.zip "$OUT" && ls -lh results_final.zip
cat "$OUT/final.md"
log "Done. Upload results_final.zip to Claude."
