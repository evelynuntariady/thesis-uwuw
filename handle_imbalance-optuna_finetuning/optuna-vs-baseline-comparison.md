# Optuna Hyperparameter Tuning — Results vs Baseline

Comparison of the four Optuna result notebooks against the corresponding baseline runs.

**Source notebooks (Optuna):**
- `thesis-3-label-optuna_stage1.ipynb`
- `thesis-3-label-optuna-stage2.ipynb`
- `stage1-weighted-stage2-oversampling-threshold-optu_stage1.ipynb`
- `stage1-weighted-stage2-oversampling-threshold-optu_stage2.ipynb`

**Baselines:** `thesis-3-label.ipynb`, `stage1-weighted-stage2-oversampling+threshold.ipynb`

Splits match the baselines exactly (3-label: test 5253 / Stage 2 test 457; 5-label: test
5276 / Stage 2 test 480; same per-label support), so all comparisons are apples-to-apples.

---

## Headline answer: is it good enough?

**Optuna produced real but modest gains — and its biggest contribution was making the
models better calibrated rather than more accurate.** Two findings dominate:

1. **Most of the improvement shows up at the default 0.5 threshold.** After per-label
   threshold tuning, baseline and Optuna models converge to nearly the same scores. In
   other words, Optuna got you *threshold-tuned-level performance without needing the
   threshold hack*.
2. **One result got actively worse:** on the 5-label Stage 2, per-label threshold tuning
   dropped macro F1 from 0.56 to **0.46** because the `sexually_explicit` threshold
   (0.9730) overfit a tiny validation set. That run's best configuration is now the
   **default 0.5 threshold**, not the tuned one.

Net: gains of **+0.01 to +0.06 F1**. Worth keeping, but the performance ceiling is clearly
**data-limited, not hyperparameter-limited**.

---

## Best hyperparameters found

| Run | learning_rate | epochs | batch | weight_decay | Best val score |
|---|---|---|---|---|---|
| 3-label Stage 1 | 2.513e-05 | 6 | 16 | 0.0161 | F1 **0.4873** |
| 3-label Stage 2 | 4.832e-05 | 6 | 8 | 0.0909 | macro F1 **0.6516** |
| Weighted Stage 1 | 3.334e-05 | 3 | 8 | 0.0049 | F1 **0.5269** |
| Oversampled Stage 2 | *(trial 0; params not retained in saved log)* | — | — | — | macro F1 **0.5165** |

All searches used 10 trials. Every best learning rate landed **2.5–5× higher than the
baseline's fixed 1e-5**, which is the single most consistent signal from the search — the
baselines were training too conservatively.

---

## Option A — 3-label

### Stage 1 (binary toxicity, test: 4797 / 456)

| Config | Toxic P | Toxic R | Toxic F1 | Macro F1 | Acc |
|---|---|---|---|---|---|
| Baseline @0.5 | 0.52 | 0.40 | 0.45 | 0.70 | 0.92 |
| Baseline @tuned 0.3709 | 0.49 | 0.48 | **0.48** | **0.72** | 0.91 |
| Optuna @0.5 | 0.49 | 0.43 | 0.46 | 0.70 | 0.91 |
| Optuna @tuned 0.2556 | 0.44 | 0.50 | 0.47 | 0.71 | 0.90 |

**Verdict: no meaningful gain.** Optuna is +0.01 at the default threshold but −0.01 after
tuning (0.48 → 0.47), and macro F1 slips 0.72 → 0.71. Validation F1 was 0.4873, essentially
the same neighbourhood as the baseline. This stage did not benefit from the search.

### Stage 2 (3-label, test: insults 202 / identity_attack 200 / profanity 71)

| Config | F1 micro | F1 macro | P micro | R micro |
|---|---|---|---|---|
| Baseline @0.5 | 0.58 | 0.54 | 0.62 | 0.55 |
| Baseline @tuned | 0.63 | 0.60 | 0.50 | 0.86 |
| **Optuna @0.5** | 0.61 | **0.60** | 0.59 | 0.64 |
| Optuna @tuned | 0.63 | 0.60 | 0.51 | 0.84 |

Per-label F1 (positive class):

| Label | Base @0.5 | Base @tuned | Optuna @0.5 | Optuna @tuned |
|---|---|---|---|---|
| insults | 0.55 | 0.63 | 0.63 | **0.64** |
| identity_attack | 0.61 | **0.68** | 0.61 | 0.66 |
| profanity_obscenity | 0.44 | 0.50 | **0.55** | 0.52 |

Tuned thresholds: insults 0.1612, identity_attack 0.0354, profanity 0.1308.

**Verdict: the clearest win, but only at the default threshold.** Macro F1 at 0.5 jumped
**0.54 → 0.60 (+0.06)** and `profanity_obscenity` improved 0.44 → 0.55. After threshold
tuning both land at macro 0.60 / micro 0.63 — identical. So Optuna's real value here is that
the model no longer *needs* aggressive threshold correction to perform well, which also means
far better precision (0.59 vs 0.50) at equal macro F1.

---

## Option B — Stage 1 weighted + Stage 2 oversampling

### Stage 1 (binary toxicity, weighted CE, test: 4796 / 480)

Class weights: `[0.5500, 5.4988]`.

| Config | Toxic P | Toxic R | Toxic F1 | Macro F1 | Acc |
|---|---|---|---|---|---|
| Baseline @0.5 | 0.37 | 0.65 | 0.48 | 0.70 | 0.87 |
| Baseline @tuned 0.9314 | 0.51 | 0.48 | 0.49 | 0.72 | 0.91 |
| **Optuna @0.5** | 0.46 | 0.52 | **0.49** | **0.72** | 0.90 |
| Optuna @tuned 0.5101 | 0.46 | 0.52 | 0.49 | 0.72 | 0.90 |

**Verdict: same F1, much better calibration.** The baseline needed an extreme threshold of
**0.9314** to reach macro F1 0.72; Optuna reaches the identical 0.72 at essentially the
natural cutoff (tuned threshold = 0.5101, i.e. no correction needed — the @0.5 and @tuned
reports are byte-identical). A model that performs best near 0.5 is far more trustworthy
than one that only works at 0.93, which was a red flag for overfit calibration. Validation
F1 also rose to 0.5269 (baseline val eval_f1 was 0.4931).

Also notable: Optuna chose only **3 epochs**, consistent with the baseline's observation that
validation loss bottomed out around epoch 1–2.

### Stage 2 (5-label, oversampled, test: 480)

Oversampling identical to baseline: insults 729→1028, identity_attack 698→1161,
threat 78→740, profanity 258→977, sexually_explicit 47→744.

| Config | F1 micro | F1 macro | P micro | R micro |
|---|---|---|---|---|
| Baseline @0.5 | 0.59 | 0.53 | 0.59 | 0.59 |
| Baseline @tuned | 0.62 | 0.54 | 0.50 | 0.82 |
| **Optuna @0.5** | 0.62 | **0.56** | 0.59 | 0.65 |
| Optuna @tuned | 0.63 | **0.46** ⚠️ | 0.52 | 0.80 |

Per-label F1 (positive class):

| Label (support) | Base @0.5 | Base @tuned | Optuna @0.5 | Optuna @tuned |
|---|---|---|---|---|
| insults (218) | 0.60 | 0.67 | 0.62 | **0.68** |
| identity_attack (199) | 0.62 | 0.64 | **0.65** | 0.64 |
| threat_incitement (19) | 0.25 | 0.27 | 0.27 | **0.29** |
| profanity_obscenity (65) | **0.60** | 0.50 | 0.58 | 0.53 |
| sexually_explicit (11) | 0.60 | 0.62 | **0.67** | 0.17 ⚠️ |

Tuned thresholds: insults 0.1717, identity_attack 0.2668, threat 0.6108, profanity 0.1780,
**sexually_explicit 0.9730**.

**Verdict: best-ever result at 0.5, but threshold tuning backfired.**

- At the default threshold, macro F1 improved **0.53 → 0.56** and `sexually_explicit` hit
  **0.67 — the highest score that label has achieved in any experiment.**
- At tuned thresholds, macro F1 **collapsed to 0.46**. The cause is isolated and clear:
  the `sexually_explicit` threshold was tuned to **0.9730**, which scored P 1.00 / R 0.33 /
  F1 0.50 on validation but only **P 1.00 / R 0.09 / F1 0.17** on test.
- Root cause: with just 11 positives in test and roughly 5–6 in validation, the
  precision-recall curve for that label is built from a handful of points. Picking an
  extreme cutoff on so few samples is **threshold overfitting**, not genuine calibration.

---

## What to take forward

| Stage | Use this configuration | Score |
|---|---|---|
| Stage 1 (3-label) | Baseline @tuned 0.3709 *or* Optuna @0.5 | macro F1 0.72 / 0.70 |
| Stage 2 (3-label) | **Optuna @0.5** (preferred) or either @tuned | macro F1 0.60, micro 0.61, precision 0.59 |
| Stage 1 (weighted) | **Optuna @0.5** | macro F1 0.72, toxic F1 0.49, no threshold hack |
| Stage 2 (5-label) | **Optuna @0.5 — do NOT use tuned thresholds** | macro F1 0.56, `sexually_explicit` 0.67 |

### Concrete fixes worth applying

1. **Stop tuning thresholds for ultra-rare labels.** Gate it on validation positive count —
   e.g. only tune a label's threshold if it has ≥ 30 validation positives; otherwise keep
   0.5. This single guard would have prevented the macro F1 0.56 → 0.46 regression.
2. **Prefer the default-threshold Optuna models.** They deliver equal-or-better macro F1
   with substantially higher precision (0.59 vs ~0.50), which is a better operating point
   for moderation than the recall-maximising tuned variants.
3. **Raise the learning rate in any future baseline.** Every search moved away from 1e-5
   toward 2.5e-5–4.8e-5; the original fixed 1e-5 was underfitting.
4. If you keep threshold tuning, **select thresholds by cross-validated F1** rather than a
   single validation split, so rare-label cutoffs are less noisy.

### Is it good enough?

For the thesis, yes — the search is a legitimate, documented contribution and it improved
Stage 2 macro F1 meaningfully at the default threshold (3-label: +0.06; 5-label: +0.03),
plus it removed the baseline's suspicious 0.93 threshold dependence.

As an absolute result, no — toxic-class F1 is still ~0.49 and Stage 2 macro F1 ~0.56–0.60.
Ten trials over four hyperparameters moved things by hundredths, which is the signature of a
**data-bound** rather than **tuning-bound** problem. The rare labels (`threat` F1 0.27–0.29,
`sexually_explicit` unstable between 0.17 and 0.67 depending purely on threshold) remain
limited by having 19 and 11 test positives. More annotated data for those categories would
move the numbers far more than any further hyperparameter search.
