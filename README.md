# CS 228: adversarially robust face anti-spoofing + Kiosk Guard

Start with `START_HERE.md`, then `PLAN.md`.

## What it does
1. Trains a ResNet-18 live-vs-spoof detector on face crops from CelebA-Spoof (baseline).
2. Attacks it with a digital PGD attack (budget 8/255 pixel levels, 10 steps).
3. Hardens it with adversarial training and a simplified "live-to-spoof direction" regularizer.
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
