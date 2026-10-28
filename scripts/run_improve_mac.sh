#!/usr/bin/env bash
# Improvement experiments on data v3 (run after run_data3_mac.sh):
#   A. Weight interpolation between each baseline3 seed and its defended model (alpha 0.25/0.5/0.75).
#      No training; evaluation only.
#   B. TRADES (beta 1 and 3), fine-tuned from baseline3 seed 0 at 2/255.
#   C. Selection on VALIDATION among: defense (eps2_ft3), interpolations, TRADES variants, by the mean
#      of clean ACER and attack success at 2/255. The winner, if trainable and not yet 3 seeds, gets seeds 1, 2.
#   D. Test evaluation of everything and one final table + trade-off chart.
#   caffeinate -i bash scripts/run_improve_mac.sh      (about 8-10 h; rerunning skips finished work)
# Output: results_improve.zip
set -euo pipefail
cd "$(dirname "$0")/.."
EXTRA=${EXTRA:-}
EPOCHS=${EPOCHS:-8}
OUT=results/improve
log() { printf '\n==== %s  %s\n' "$(date +%H:%M:%S)" "$*"; }
[ -d .venv ] && source .venv/bin/activate
python -c "from utils.device import get_device; print('device:', get_device())"
for s in 0 1 2; do
  [ -f "checkpoints/eps2_ft3/seed${s}_best.pt" ] || { echo "run scripts/run_data3_mac.sh first (eps2_ft3 seed $s missing)"; exit 1; }
done
mkdir -p "$OUT"

log "A. Weight interpolation, seeds 0 1 2"
for s in 0 1 2; do
  python -m training.interpolate --base "checkpoints/baseline3/seed${s}_best.pt" \
      --defended "checkpoints/eps2_ft3/seed${s}_best.pt" --alphas 0.25 0.5 0.75 --out_prefix checkpoints/interp3 --seed "$s"
done

train_trades() {  # beta seed
  local b=$1 s=$2 d="checkpoints/trades3_b$1"
  if [ -f "$d/.seed${s}_done" ]; then echo "TRADES beta=$b seed $s done, skipping"; return; fi
  log "B. TRADES beta=$b, seed $s"
  python -m training.train_hardened --data_root ./data_pad3 --epochs "$EPOCHS" --lr 2e-5 --seed "$s" --eps 0.00784 \
      --trades_beta "$b" --init "checkpoints/baseline3/seed${s}_best.pt" --out "$d" $EXTRA
  touch "$d/.seed${s}_done"
}
for b in 1 3; do train_trades "$b" 0; done

evaluate() {  # dir split out seeds-glob
  [ -f "$3" ] && { echo "$3 exists, skipping"; return; }
  python -m evaluation.evaluate --data_root ./data_pad3 --split "$2" --sweep 1 2 4 8 \
      --baseline_glob "checkpoints/baseline3/seed${4}_best.pt" --hardened_glob "checkpoints/$1/seed${4}_best.pt" --out "$3"
}
CANDS="eps2_ft3 interp3_a025 interp3_a050 interp3_a075 trades3_b1 trades3_b3"
log "C. Selection on validation (seed 0)"
for c in $CANDS; do evaluate "$c" val "$OUT/val_$c.json" 0; done
WIN=$(python - "$OUT" $CANDS <<'PY'
import json, sys
out, cands = sys.argv[1], sys.argv[2:]
best = None
for c in cands:
    s = json.load(open(f"{out}/val_{c}.json"))["hardened"]["summary"]
    score = 0.5 * s["ACER"]["mean"] + 0.5 * s["ASRc_eps2"]["mean"]
    print(f"{c}: val ACER {s['ACER']['mean']:.3f}  val attack success 2/255 {s['ASRc_eps2']['mean']:.3f}  score {score:.3f}", file=sys.stderr)
    if best is None or score < best[1]:
        best = (c, score)
print(best[0])
PY
)
echo "selected on validation: $WIN" | tee "$OUT/selected.txt"
case "$WIN" in trades3_b*) b=${WIN#trades3_b}; for s in 1 2; do train_trades "$b" "$s"; done ;; esac

log "D. Test evaluation"
for c in $CANDS; do
  if [ -f "checkpoints/$c/seed1_best.pt" ] && [ -f "checkpoints/$c/seed2_best.pt" ]; then g="[012]"; else g="0"; fi
  evaluate "$c" test "$OUT/test_$c.json" "$g"
done
python -m evaluation.final_table \
  --run "Adv. fine-tuning 2/255 (data v3)=$OUT/test_eps2_ft3.json" \
  --run "Interpolation alpha 0.25=$OUT/test_interp3_a025.json" \
  --run "Interpolation alpha 0.50=$OUT/test_interp3_a050.json" \
  --run "Interpolation alpha 0.75=$OUT/test_interp3_a075.json" \
  --run "TRADES beta 1=$OUT/test_trades3_b1.json" \
  --run "TRADES beta 3=$OUT/test_trades3_b3.json" \
  --out "$OUT/improve"
if [ -f "checkpoints/$WIN/seed1_best.pt" ] && [ -f "checkpoints/$WIN/seed2_best.pt" ]; then
  python -m evaluation.threshold_analysis --data_root ./data_pad3 --ckpt_glob "checkpoints/$WIN/seed[012]_best.pt" \
      --label "Selected model ($WIN)" --out "$OUT/threshold_$WIN"
  for s in 0 1 2; do
    python -m evaluation.by_spoof_type --src ./celeba_meta --data_root ./data_pad3 --ckpt "checkpoints/$WIN/seed${s}_best.pt" > "$OUT/by_spoof_type_${WIN}_seed${s}.txt"
  done
fi
log "Bundle"
{ git rev-parse HEAD 2>/dev/null || echo "not a git checkout"; echo "device: $(python -c 'from utils.device import get_device; print(get_device())')"; echo "selected: $WIN"; } | tee "$OUT/COMMIT.txt"
# Collect into a temp file first: writing straight into $OUT would make the *.txt glob include the output
# file itself, and cat would copy it into itself until the disk is full.
TMP=$(mktemp)
for f in "$OUT"/*.md "$OUT"/*.txt "$OUT"/threshold_*/threshold.md; do
  [ -f "$f" ] && [ "$(basename "$f")" != all_results.txt ] && { echo "===== $f"; cat "$f"; echo; }
done > "$TMP"
mv "$TMP" "$OUT/all_results.txt"
rm -f results_improve.zip && zip -qr results_improve.zip "$OUT" && ls -lh results_improve.zip
cat "$OUT/improve.md"
log "Done. Upload results/improve/all_results.txt and results/improve/improve_tradeoff.png to Claude."
