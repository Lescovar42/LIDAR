# Landslide4Sense reproduction comparison

Official reported validation reference:

- Precision: 0.5175
- Recall: 0.6550
- F1: 0.5782

## Our independent reproduction

P=0.5829, R=0.4320, F1=0.4962, IoU=0.3300

Absolute F1 delta from report: 0.0820

## Interpretation rule

A large mismatch does not by itself prove the Tillamook pipeline is wrong. First determine whether the official checkpoint itself reproduces the reported validation metric on these validation labels. Then compare the independent reproduction under the same data, architecture, normalization, optimizer, loss, iteration budget, and metric definition.
