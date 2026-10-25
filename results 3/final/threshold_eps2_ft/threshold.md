**Baseline over 3 seed(s)** (mean ± std)

| Metric | Default threshold (0.5) | Operating point (val EER threshold) |
|---|---|---|
| Threshold | 0.500 | 0.635 ± 0.027 |
| APCER (fakes accepted) | 45.0% ± 0.8 | 36.2% ± 1.8 |
| BPCER (real rejected) | 6.4% ± 1.5 | 10.9% ± 1.3 |
| ACER | 25.7% ± 0.7 | 23.6% ± 1.5 |

EER: validation 7.0% ± 0.8, test 23.0% ± 1.4. AUC: validation 0.983 ± 0.003, test 0.852 ± 0.016.

| Seed checkpoint | APCER count (0.5) | BPCER count (0.5) | APCER count (op.) | BPCER count (op.) |
|---|---|---|---|---|
| seed0_best.pt | 265/600 | 38/600 | 206/600 | 61/600 |
| seed1_best.pt | 271/600 | 48/600 | 228/600 | 74/600 |
| seed2_best.pt | 274/600 | 30/600 | 218/600 | 61/600 |
