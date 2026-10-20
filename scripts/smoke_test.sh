#!/usr/bin/env bash
# Runs the REAL pipeline on fake data in a few minutes: fake CelebA-Spoof tree -> the real
# prepare script -> scripts/run_experiment.sh (baseline, 3 hardened variants, evaluation,
# report) -> bundle -> ONNX export -> attacked demo images.
# If this prints SMOKE TEST PASSED, your setup is good.
set -e
cd "$(dirname "$0")/.."
# Safety: the smoke test deletes checkpoints/, results/ and bundle/ when it finishes.
if [ -z "$FORCE" ] && { [ -e results/report.md ] || [ -d bundle ] || [ -d checkpoints/baseline ]; }; then
  echo "STOP: found checkpoints/, results/ or bundle/ from a previous run."
  echo "The smoke test would DELETE them. Run it from a fresh copy of the project."
  echo "If they are only leftovers from an earlier smoke test, run:  FORCE=1 ./scripts/smoke_test.sh"
  exit 1
fi
rm -rf fake_celeba data_dummy checkpoints results demo_images demo_smoke bundle bundle.zip smoke_*.onnx

python -m data.make_fake_celeba_tree --dst ./fake_celeba
python -m data.prepare_celeba_spoof --src ./fake_celeba --dst ./data_dummy --per_class 40 --test_per_class 16

EXTRA="--no_pretrained" EPOCHS=1 bash scripts/run_experiment.sh ./data_dummy

python -m scripts.make_attacked_images --data_root ./data_dummy \
    --baseline checkpoints/baseline/seed0_best.pt \
    --hardened checkpoints/full/seed0_best.pt --n 3 --out demo_images
EPOCHS=1 bash scripts/bundle_outputs.sh

python -m export.export_onnx --ckpt bundle/checkpoints/baseline_seed0_best.pt --out smoke_baseline.onnx
python -m export.export_onnx --ckpt bundle/checkpoints/full_seed0_best.pt --out smoke_hardened.onnx
# clean up so the fake results never mix with your real ones
rm -rf fake_celeba data_dummy checkpoints results demo_images bundle bundle.zip smoke_*.onnx
echo "SMOKE TEST PASSED"
