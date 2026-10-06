#!/usr/bin/env bash
# Packs what you need to bring back from Kaggle into one zip (all seeds).
#   EPOCHS=5 bash scripts/bundle_outputs.sh      (EPOCHS must match the run)
set -e
cd "$(dirname "$0")/.."
LAST=$(( ${EPOCHS:-5} - 1 ))
rm -rf bundle bundle.zip && mkdir -p bundle/checkpoints
for f in checkpoints/baseline/seed*_best.pt; do cp "$f" "bundle/checkpoints/baseline_$(basename "$f")"; done
for cfg in full adv_only reg_only; do
  for f in checkpoints/$cfg/seed*_epoch${LAST}.pt; do
    [ -f "$f" ] && cp "$f" "bundle/checkpoints/${cfg}_$(basename "$f")"
  done
done
cp -r results bundle/results
[ -d demo_images ] && cp -r demo_images bundle/demo_images
zip -qr bundle.zip bundle
echo "bundle.zip ready ($(du -h bundle.zip | cut -f1)). Files:"
ls bundle/checkpoints
