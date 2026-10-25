**Baseline over 3 seed(s)** (mean ± std)

| Metric | Default threshold (0.5) | Operating point (val EER threshold) |
|---|---|---|
| Threshold | 0.500 | 0.723 ± 0.064 |
| APCER (fakes accepted) | 40.0% ± 2.2 | 31.5% ± 0.7 |
| BPCER (real rejected) | 4.8% ± 1.4 | 9.5% ± 0.2 |
| ACER | 22.4% ± 0.4 | 20.5% ± 0.3 |

EER: validation 5.2% ± 0.3, test 19.3% ± 0.8. AUC: validation 0.990 ± 0.000, test 0.899 ± 0.003.

| Seed checkpoint | APCER count (0.5) | BPCER count (0.5) | APCER count (op.) | BPCER count (op.) |
|---|---|---|---|---|
| seed0_best.pt | 251/600 | 21/600 | 185/600 | 57/600 |
| seed1_best.pt | 225/600 | 38/600 | 189/600 | 58/600 |
| seed2_best.pt | 244/600 | 27/600 | 193/600 | 56/600 |
