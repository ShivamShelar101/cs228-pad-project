**Clean performance on the test set**

| Model | APCER (fakes accepted) | BPCER (real rejected) | ACER |
|---|---|---|---|
| Baseline (no defense) | 19.4% ± 6.5 | 0.8% ± 0.2 | 10.1% ± 3.2 |
| Data v3: adv. fine-tuning 2/255 | 40.0% ± 2.2 | 4.8% ± 1.4 | 22.4% ± 0.4 |

**Robustness (lower is better in every column)**

| Model | Attack success (8/255, 10 steps) | Attack success (stronger: 16/255, 40 steps) | Fakes accepted under random noise (control) | Real faces rejected after attack | Worst case (fake fools it clean OR under any attack) |
|---|---|---|---|---|---|
| Baseline (no defense) | 100.0% ± 0.0 | 100.0% ± 0.0 | 41.3% ± 15.9 | 100.0% ± 0.0 | 100.0% ± 0.0 |
| Data v3: adv. fine-tuning 2/255 | 94.4% ± 0.9 | 100.0% ± 0.0 | 42.3% ± 2.0 | 99.4% ± 0.4 | 100.0% ± 0.0 |

**Validation vs test accuracy**

| Model | Accuracy on validation | Accuracy on test |
|---|---|---|
| Baseline (no defense) | 98.7% ± 0.2 | 89.9% ± 3.2 |
| Data v3: adv. fine-tuning 2/255 | 94.1% ± 0.3 | 77.6% ± 0.4 |

**Attack success vs attack budget (10 steps)**

| Model | 1/255 | 2/255 | 4/255 | 8/255 | 16/255 |
|---|---|---|---|---|---|
| Baseline (no defense) | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 |
| Data v3: adv. fine-tuning 2/255 | 48.4% ± 1.1 | 55.1% ± 1.9 | 72.0% ± 1.4 | 94.2% ± 0.9 | 100.0% ± 0.0 |

**Attack success on fakes the model caught WITHOUT an attack (10 steps)**

| Model | 1/255 | 2/255 | 4/255 | 8/255 | 16/255 |
|---|---|---|---|---|---|
| Baseline (no defense) | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 |
| Data v3: adv. fine-tuning 2/255 | 13.9% ± 2.4 | 25.1% ± 0.7 | 53.3% ± 1.0 | 90.4% ± 1.3 | 100.0% ± 0.0 |
