| Model | Seeds | Clean APCER | Clean BPCER | Clean ACER | Attack success 1/255* | Attack success 2/255* | Attack success 4/255* | Attack success 8/255* |
|---|---|---|---|---|---|---|---|---|
| Baseline (no defense) | 3 | 28.8% ± 8.7 | 1.3% ± 0.9 | 15.1% ± 3.9 | 100.0% | 100.0% | 100.0% | 100.0% |
| Adv. fine-tuning 2/255 | 3 | 45.0% ± 0.8 | 6.4% ± 1.5 | 25.7% ± 0.7 | 13.4% ± 2.1 | 32.0% ± 3.4 | 64.6% ± 1.9 | 96.0% ± 0.9 |
| Lighter adv. weight (0.5) | 1 | 42.3% | 4.0% | 23.2% | 15.0% | 36.4% | 71.1% | 99.4% |
| Training budget 1/255 | 1 | 39.7% | 3.0% | 21.3% | 25.7% | 55.5% | 92.0% | 100.0% |
| No regularizer (ablation) | 1 | 44.3% | 6.5% | 25.4% | 12.3% | 29.6% | 62.3% | 95.2% |

*Share of the fakes a model rejects without attack that a 10-step PGD attack then flips (lower is better). Mean ± std over seeds; test set, threshold 0.5.
