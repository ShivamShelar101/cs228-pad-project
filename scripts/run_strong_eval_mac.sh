#!/usr/bin/env bash
# Stronger-attack check of the selected defense (PGD-50, 2 restarts, 2 losses) at 1, 2, 4 / 255.
#   caffeinate -i bash scripts/run_strong_eval_mac.sh      (about 1-3 hours)  -> results_strong.zip
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate
mkdir -p results/strong
python -m evaluation.strong_attack --data_root ./data_pad2 --ckpt_glob "checkpoints/eps2_ft/seed[012]_best.pt" \
    --eps 1 2 4 --out results/strong/eps2_ft
git rev-parse HEAD > results/strong/COMMIT.txt 2>/dev/null || true
rm -f results_strong.zip && zip -qr results_strong.zip results/strong && ls -lh results_strong.zip
echo "Done. Upload results_strong.zip to Claude."
