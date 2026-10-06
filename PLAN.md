# Plan

Work in stages. Each one ends with something you can check before moving on.

## Stage 1: setup and smoke test (30 min)
`START_HERE.md` steps 1-3. Done when the smoke test prints SMOKE TEST PASSED and the repo is on GitHub.

## Stage 2: pilot run on Kaggle (1-2 hours)
One seed, 5 epochs, up to 3000 images per class. The point is to learn whether the data, labels and attack work
before spending more time. Note how long it took.
Done when you have `results/report.md` and a downloaded `bundle.zip`.

## Stage 3: read the pilot and decide
- Baseline clean accuracy should be well above 50% after the first epoch. If it stays near 50%, the data is wrong.
- If the attack succeeds under about 10% of the time on the baseline, it is too weak to show anything: rerun the
  evaluation with a bigger budget, `--eps 0.0627` (16/255), and report both.
- If the hardened model does not beat the baseline, that is still a result. Try `--adv_weight 2.0` (attacked images
  count double) or more epochs once. If it still does not help, say so honestly.
- If clean accuracy drops while attack success also drops, that is the normal trade-off: report both numbers.

## Stage 4: the full run (as long as you like)
More seeds and epochs, and more data if this Kaggle copy has it:
```bash
SEEDS="0 1 2" EPOCHS=10 bash scripts/run_experiment.sh ./data_pad
```
This is about 6x the pilot's work. If it does not fit in one session, run one seed per session
(`SEEDS="1"` and so on), keep each session's `checkpoints/`, and evaluate them together at the end.
Update `LAST` (= EPOCHS - 1) in the notebook's attacked-images cell.

## Stage 5: the demo app (a couple of hours)
```bash
cd ~/Downloads/cs228-pad-project && source venv/bin/activate
unzip -o ~/Downloads/bundle.zip
ls bundle/checkpoints                      # exact names; the commands below assume EPOCHS=5, seed 0
python -m export.export_onnx --ckpt bundle/checkpoints/baseline_seed0_best.pt --out app/backend/model_baseline.onnx
python -m export.export_onnx --ckpt bundle/checkpoints/full_seed0_epoch4.pt   --out app/backend/model_hardened.onnx

# terminal 1
cd app/backend && uvicorn main:app --port 8000
# terminal 2
cd app/frontend && npm install && npm run dev
```
Open the localhost link. Show your face (both models should say LIVE), a photo or phone replay (SPOOF), and then
upload one `bundle/demo_images/attacked_*.png`: the baseline should be fooled and, hopefully, the hardened model not.
Use the **upload** for the attacked image. Do not print it and hold it up: PGD perturbations mostly stop working
once printed.

## Stage 6: report and slides
Report skeleton (about 4 pages):
1. Problem and motivation (reuse slides 2-3)
2. Related work (the three papers, one paragraph each)
3. Method: data, baseline, PGD attack (8/255, 10 steps), adversarial training, stability rule, what was simplified
4. Results: paste `results/report.md`, the chart, and 3-4 sentences on what it shows
5. Demo: a Kiosk Guard screenshot
6. Limits (below)

Send me the numbers when you have them and I will add a results slide and fill in the report.

## Limits to state honestly
- The attack is digital only (no printing or re-photographing). AdvGen's paper shows PGD loses most of its power
  once printed, so these results are about digital robustness.
- One dataset (a partial copy of it) and, if you stay with one seed, a single training run, so differences of a
  point or two may be noise.
- The stability rule is a single-dataset simplification of the CVPR paper's method, not a reproduction.
- The attacker sees the model's gradients (white-box), which is an easier setting than the real world.
- Spoof images are print and replay types only (photo, poster, A4, PC, pad, phone).
- In CelebA-Spoof the live photos come from the CelebA celebrity set and the spoof photos were captured separately,
  so a model can pick up on photo style as well as real spoof cues. Face crops that would have run off the photo are
  skipped for both classes, because black padding was twice as common in spoof images (60% vs 33% in a check).
