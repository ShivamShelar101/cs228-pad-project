#!/usr/bin/env bash
# Demographic differentials of the final models (data v3, 3 seeds), about 30-60 min.
#   caffeinate -i bash scripts/run_fairness_mac.sh      -> results/fairness/fairness.md and fairness_gap.png
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate
python -m evaluation.fairness --src ./celeba_meta --data_root ./data_pad3 \
    --baseline_glob "checkpoints/baseline3/seed[012]_best.pt" \
    --defense_glob "checkpoints/eps2_ft3/seed[012]_best.pt" \
    --baseline_thr results/data3/threshold_baseline3/threshold.json \
    --defense_thr results/data3/threshold_eps2_ft3/threshold.json --out results/fairness
echo "Done. Upload results/fairness/fairness.md and results/fairness/fairness_gap.png to Claude."
