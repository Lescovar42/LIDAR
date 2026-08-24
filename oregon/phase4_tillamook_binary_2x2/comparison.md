# Tillamook strict-binary 2×2 feature/depth comparison

All model selection and threshold selection below use the frozen Tillamook validation split only. The internal test split was not evaluated.

| Run | Architecture | Features | Params | Epoch | Thr | Dice | IoU | Precision | Recall | Specificity | FP | FN | Pred + |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deep_7ch | Deep | 7ch | 7,767,137 | 21 | 0.45 | 0.4085 | 0.2567 | 0.2961 | 0.6584 | 0.7396 | 13,034,704 | 2,844,173 | 31.71% |
| deep_3ch | Deep | 3ch | 7,765,985 | 26 | 0.50 | 0.3434 | 0.2073 | 0.2327 | 0.6543 | 0.6412 | 17,962,608 | 2,878,302 | 40.09% |
| shallow_7ch | Shallow | 7ch | 468,961 | 21 | 0.50 | 0.3228 | 0.1924 | 0.2105 | 0.6919 | 0.5683 | 21,611,982 | 2,565,382 | 46.88% |
| shallow_3ch | Shallow | 3ch | 467,809 | 4 | 0.45 | 0.2901 | 0.1696 | 0.1826 | 0.7051 | 0.4750 | 26,286,768 | 2,455,430 | 55.07% |

**Validation winner:** `deep_7ch` (Dice=0.4085, threshold=0.45).

**Do not evaluate the internal test until this configuration and threshold are explicitly frozen.**
