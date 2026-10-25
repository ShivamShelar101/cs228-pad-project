# CS 228: adversarially robust face anti-spoofing + Kiosk Guard

Start with `START_HERE.md`, then `PLAN.md`.

## Reproducing the checkpoint report
Every number and figure in the checkpoint report comes from `notebooks/kaggle_checkpoint.ipynb`
(seeds 0, 1, 2; the commit hash is written to `results/COMMIT.txt`).

**Environment** (Python 3.10+): `pip install -r requirements.txt` (on a Mac: `bash scripts/setup_mac.sh`).

**Data (not in this repo).** CelebA-Spoof is used under its non-commercial research license and is not
redistributed here. Get the Kaggle copy `attentionlayer241/celeba-spoof-for-face-antispoofing` (attach it to a
Kaggle notebook, or `kagglehub.dataset_download(...)`), then build the face crops:
```bash
python -m data.prepare_celeba_spoof --src <dataset folder> --dst ./data_pad2 \
    --per_class 3000 --test_per_class 600 --max_per_subject 8      # deterministic: same images every time
```
**Baseline, threshold analysis, EDA, error analysis**
```bash
for s in 0 1 2; do python -m training.train_baseline --data_root ./data_pad2 --epochs 5 --seed $s --out checkpoints/baseline; done
python -m evaluation.threshold_analysis --data_root ./data_pad2 --ckpt_glob "checkpoints/baseline/seed*_best.pt" --out results/threshold
python -m evaluation.eda --src <dataset folder> --data_root ./data_pad2 --out results/eda
python -m evaluation.by_spoof_type --src <dataset folder> --data_root ./data_pad2 --ckpt checkpoints/baseline/seed0_best.pt
```
| Script | Report section |
|---|---|
| `data/prepare_celeba_spoof.py` | II. data loading, cropping, subject-disjoint split, balancing |
| `evaluation/eda.py` | II. quality checks (leakage, duplicates); III. all EDA figures |
| `training/train_baseline.py` | IV. baseline training (seeded via `utils/seed.py`) |
| `evaluation/threshold_analysis.py` | IV. rates with counts, EER, FAR/FRR, ROC, operating point, error grids |
| `evaluation/by_spoof_type.py` | IV. error analysis by spoof type, lighting, environment |


### Running the checkpoint on a Mac (Apple GPU, no Kaggle quota)

1. On Kaggle, run `notebooks/kaggle_export_mac.ipynb` as a CPU commit (inputs: your `cs228-data` notebook and the
   CelebA-Spoof dataset). Download `mac_data.zip` from its Output. It holds licensed data: never commit it.
2. From the repo root: `caffeinate -i bash scripts/run_checkpoint_mac.sh ~/Downloads/mac_data.zip`
   (creates `.venv`, installs `requirements.txt`, trains seeds 0-2, runs threshold analysis, EDA and error analysis,
   writes `bundle_checkpoint.zip`). Rerunning skips seeds already trained. Results on MPS differ slightly from CUDA.


### Defense run on a Mac (after the checkpoint run)

`caffeinate -i bash scripts/run_defense_mac.sh` adversarially fine-tunes each baseline seed at 2/255 (8 epochs),
then evaluates the budget sweep (1-16/255), conditional and worst-case attack success, the validation-chosen
operating point, and per-type errors. Several hours on an M-series Mac; rerunning skips finished seeds.
Output: `results_defense.zip`.


### Final experiments on a Mac (after the defense run)

`caffeinate -i bash scripts/run_final_mac.sh` trains two variants aimed at the defense's clean-accuracy cost
(lighter adversarial weight; 1/255 budget) and an ablation without the regularizer, selects the best defense on
validation only, trains it on 3 seeds, and writes the final comparison table and trade-off chart
(`evaluation/final_table.py`). About 7-9 hours; rerunning skips finished work. Output: `results_final.zip`.


### Extra experiments (after the final run)

* **Stronger attack check:** `caffeinate -i bash scripts/run_strong_eval_mac.sh` attacks the caught test fakes of the
  selected defense with PGD-50 x 2 restarts x 2 losses (cross-entropy and raw logit) at 1, 2, 4/255 and compares with
  PGD-10 (`evaluation/strong_attack.py`). Output: `results_strong.zip`.
* **Data v3 (distribution-shift fix):** run `notebooks/kaggle_data3.ipynb` as a Kaggle CPU commit. It builds
  `data_pad3` (9,000 crops per class for train+val, PC-screen spoofs drawn 3x as readily via
  `prepare_celeba_spoof.py --type_boost 7=3`, test split copied unchanged from `data_pad2`) and outputs `mac_data3.zip`.
  Then `caffeinate -i bash scripts/run_data3_mac.sh ~/Downloads/mac_data3.zip` retrains the baselines and the defense on
  it. Output: `results_data3.zip`.

* **Improvement experiments:** `caffeinate -i bash scripts/run_improve_mac.sh` (after the data v3 run) tries weight
  interpolation between each baseline and its defended model (`training/interpolate.py`, no training) and the TRADES
  loss (`train_hardened.py --trades_beta`), selects on validation, and writes `results/improve/`.
* **Data v4:** `notebooks/kaggle_data4.ipynb` (Kaggle CPU) boosts PC-screen spoofs further
  (`--type_boost 7=8 --oversample 8`); then `scripts/run_data4_mac.sh ~/Downloads/mac_data4.zip`.

## What it does
1. Trains a ResNet-18 live-vs-spoof detector on face crops from CelebA-Spoof (baseline).
2. Attacks it with a digital PGD attack (budget 8/255 pixel levels, 10 steps).
3. Hardens it with adversarial training (attacking both fakes and real faces) and a simplified "live-to-spoof direction" regularizer.
4. Compares APCER / BPCER / ACER and attack success rate, with ablations and seeds.
5. Serves baseline and hardened models side by side in a webcam app (FastAPI + React + MediaPipe).

## Layout
| Folder | What |
|---|---|
| `data/` | dataset prep (face crops, only images that exist), fake-data tools for tests |
| `models/` | ResNet-18 wrapper |
| `attacks/` | PGD attack (pixel-unit budget) |
| `training/` | baseline and hardened training, regularizer |
| `evaluation/` | metrics, evaluation across seeds, results table and chart |
| `export/` | PyTorch to ONNX |
| `scripts/` | setup, smoke test, full experiment, bundle, attacked demo images |
| `app/` | Kiosk Guard backend and frontend |
| `notebooks/` | the Kaggle notebook |

## Things to know
- `pgd_attack` works in real pixel units and keeps images valid. Tune `--eps` in `evaluate.py` /
  `train_hardened.py` if the attack is too weak or strong.
- The regularizer is a single-dataset simplification of the CVPR 2023 method, not a reproduction.
- The attack is digital only, so results speak to digital robustness.
- The dataset is used under CelebA-Spoof's non-commercial research license: do not redistribute it.
