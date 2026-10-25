**Baseline over 3 seed(s)** (mean ± std)

| Metric | Default threshold (0.5) | Operating point (val EER threshold) |
|---|---|---|
| Threshold | 0.500 | 0.464 ± 0.133 |
| APCER (fakes accepted) | 19.4% ± 6.5 | 19.6% ± 2.7 |
| BPCER (real rejected) | 0.8% ± 0.2 | 0.8% ± 0.3 |
| ACER | 10.1% ± 3.2 | 10.2% ± 1.5 |

EER: validation 1.2% ± 0.1, test 6.7% ± 1.2. AUC: validation 0.999 ± 0.000, test 0.982 ± 0.005.

| Seed checkpoint | APCER count (0.5) | BPCER count (0.5) | APCER count (op.) | BPCER count (op.) |
|---|---|---|---|---|
| seed0_best.pt | 160/600 | 4/600 | 133/600 | 7/600 |
| seed1_best.pt | 105/600 | 5/600 | 118/600 | 4/600 |
| seed2_best.pt | 85/600 | 6/600 | 101/600 | 3/600 |
