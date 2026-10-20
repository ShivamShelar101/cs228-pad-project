#!/usr/bin/env bash
# The whole real experiment: baseline + 3 hardened variants + evaluation + report.
#   bash scripts/run_experiment.sh ./data_pad
# Defaults: 1 seed, 5 epochs (the pilot). The full run:
#   SEEDS="0 1 2" EPOCHS=10 bash scripts/run_experiment.sh ./data_pad
# (EXTRA="--no_pretrained" is only for offline smoke tests.)
set -e
cd "$(dirname "$0")/.."
DATA=${1:-./data_pad}
EPOCHS=${EPOCHS:-5}
SEEDS=${SEEDS:-"0"}
EXTRA=${EXTRA:-}
LAST=$((EPOCHS-1))

for s in $SEEDS; do
  python -m training.train_baseline --data_root $DATA --epochs $EPOCHS --seed $s $EXTRA --out checkpoints/baseline
  python -m training.train_hardened --data_root $DATA --epochs $EPOCHS --seed $s $EXTRA --out checkpoints/full
  python -m training.train_hardened --data_root $DATA --epochs $EPOCHS --seed $s $EXTRA --reg_weight 0 --out checkpoints/adv_only
  python -m training.train_hardened --data_root $DATA --epochs $EPOCHS --seed $s $EXTRA --adv_weight 0 --out checkpoints/reg_only
done

for cfg in full adv_only reg_only; do
  python -m evaluation.evaluate --data_root $DATA \
      --baseline_glob "checkpoints/baseline/seed*_best.pt" \
      --hardened_glob "checkpoints/${cfg}/seed*_best.pt" \
      --out results/eval_${cfg}.json
done

python -m evaluation.make_report \
    --run "Full method (adv. training + regularizer)=results/eval_full.json" \
    --run "Adversarial training only=results/eval_adv_only.json" \
    --run "Regularizer only=results/eval_reg_only.json" \
    --out results/report
echo "RUN COMPLETE: see results/report.md and results/report.png"
