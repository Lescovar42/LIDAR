# Positive-class-weight ablation summary

The two evaluation fingerprints match, so the threshold sweeps use the same manifest rows.

| Run | Best threshold | Dice | IoU | Precision | Recall | Specificity | FP | FN | Predicted positive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Auto | 0.70 | 0.2032 | 0.1131 | 0.1401 | 0.3699 | 0.8817 | 333,144 | 92,442 | 13.08% |
| pos_weight=1.0 | 0.55 | 0.2042 | 0.1137 | 0.1391 | 0.3837 | 0.8762 | 348,412 | 90,409 | 13.66% |

**Rule-based phase recommendation:** `retain_weight_but_improve_diversity_first`.

The weight-only ablation did not provide a decisive precision/FP improvement; prioritize measured diversity limitations before architecture changes.

This conclusion is measured on frozen source-region validation only. It must not be described as cross-region generalization.
