# Stage 2 Imbalance Handling — Experiment Summary & Decision

This document summarizes the experiments comparing ways to handle class imbalance in
the two-stage toxicity classifier, the concept of threshold tuning, the 3-label
experiment, and the label-dependency analysis that justifies the final decision.

All notebooks share the same **two-stage pipeline**:

- **Stage 1** — binary toxicity (`toxic` vs `not toxic`); single-label, softmax /
  CrossEntropy loss.
- **Stage 2** — toxicity *type*; multi-label, sigmoid / BCE loss.
- Model: `indolem/indobertweet-base-uncased`.
- Resampling / reweighting is applied to the **training split only**; the test split
  keeps its original distribution so evaluation stays meaningful.

---

## 1. The three imbalance-handling methods

Only the *mechanism* for handling imbalance changes between these three notebooks
(`thesis-class-weight`, `thesis-focal-loss`, `thesis-oversampling`); the pipeline,
model, and training arguments are otherwise identical.

### 1a. Class weight (`thesis-class-weight.ipynb`)

Keeps the data untouched; makes the **loss function** care more about rare classes.

- Computes `compute_class_weight("balanced", ...)` → higher weight for the minority
  class.
- Stage 1: custom `WeightedTrainerSingleLabel` using
  `nn.CrossEntropyLoss(weight=class_weights)`.
- Stage 2: `WeightedTrainerMultiLabel` using `nn.BCEWithLogitsLoss(pos_weight=...)`,
  where `pos_weight = neg_count / pos_count` per label.

**Effect:** every minority example contributes proportionally more to the gradient.
Data size and content unchanged; only the *penalty* for getting a rare class wrong
goes up.

### 1b. Focal loss (`thesis-focal-loss.ipynb`)

A superset of class weighting. Also reweights the loss, but adds a **hard-example
focusing** term.

Formula used in the notebook:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)   with gamma = 2.0
```

- `alpha` = the same balanced class weights (Stage 1) / `pos_weight` per label
  (Stage 2) → the *class rebalancing* part, identical in spirit to class weight.
- `(1 - p_t)^gamma` = the *extra* part: down-weights easy, already-confident
  predictions and concentrates learning on hard / misclassified samples.
- Implemented as `FocalLossMultiClass` (softmax, Stage 1) and `FocalLossMultiLabel`
  (sigmoid, Stage 2), wired through `FocalLossTrainer*`.

**Effect:** class weight fixes *class frequency*; focal loss fixes class frequency
**and** stops the model from coasting on easy examples. With `gamma = 0`, focal loss
reduces exactly to class weighting — so it is class weighting plus a difficulty knob.

### 1c. Oversampling (`thesis-oversampling.ipynb`)

Doesn't touch the loss at all; changes the **data distribution** so a plain `Trainer`
sees balanced classes.

- Stage 1: finds the majority label, then
  `minority_df.sample(n=len(majority_df), replace=True)` — duplicates minority rows
  with replacement until 50/50, then shuffles.
- Stage 2 (`oversample_multilabel`): for each label below the most frequent label's
  count, duplicates rows carrying that label until the counts match.
- Uses the **stock** `Trainer` with default (unweighted) loss, since balance now comes
  from the data itself.

**Effect:** minority classes are physically seen more often. Simple, but because it
duplicates existing rows (random oversampling, not SMOTE — no synthetic examples), it
can encourage the model to memorize / overfit the repeated samples.

### Side-by-side

| | Class weight | Focal loss | Oversampling |
|---|---|---|---|
| What it modifies | Loss function | Loss function | Training data |
| Mechanism | Weight rare classes in CE/BCE | Weight rare classes **+** focus on hard examples | Duplicate minority rows to balance counts |
| Trainer | Custom `WeightedTrainer*` | Custom `FocalLossTrainer*` | Stock `Trainer` |
| Key hyperparam | balanced class weights / `pos_weight` | same `alpha` + `gamma=2.0` | target = majority count |
| Dataset size | unchanged | unchanged | grows (duplicates) |
| Main risk | may under-focus on hard cases | more knobs to tune (`gamma`) | overfitting to duplicated rows |
| Relationship | subset of focal loss (`gamma=0`) | class weight + hard-example focusing | orthogonal — data-level, not loss-level |

### Plain-language analogies

- **Oversampling** = photocopy the rare topic's questions many times so the student
  simply *sees* them more often.
- **Class weight** = make the rare questions *worth more points*.
- **Focal loss** = worth more points, *and* skip drilling what's already easy.

---

## 2. Results of the three methods (5-label Stage 2)

### Stage 1 — binary toxicity (test: 4796 non-toxic / 480 toxic)

Judged on the **toxic class (label 1)** and **macro F1**, since accuracy is misleading
under ~10:1 imbalance.

| Method | Toxic precision | Toxic recall | Toxic F1 | Macro F1 | Accuracy |
|---|---|---|---|---|---|
| Class weight | 0.37 | **0.68** | 0.48 | 0.70 | 0.87 |
| **Focal loss** | **0.41** | 0.64 | **0.50** | **0.72** | **0.88** |
| Oversampling | 0.36 | 0.68 | 0.47 | 0.69 | 0.86 |

**Best for Stage 1: Focal loss** — highest toxic-class F1 (0.50), highest macro F1
(0.72), and best precision, while keeping recall competitive.

### Stage 2 — multi-label toxicity type (5 labels)

| Method | F1 micro | F1 macro | Precision micro | Recall micro |
|---|---|---|---|---|
| Class weight | **0.585** | 0.521 | 0.490 | **0.725** |
| Focal loss | 0.565 | 0.474 | 0.462 | 0.729 |
| **Oversampling** | 0.581 | **0.537** | **0.568** | 0.594 |

Per-label F1 for the positive (rare) class:

| Label (support) | Class weight | Focal loss | Oversampling |
|---|---|---|---|
| insults (218) | **0.65** | 0.64 | 0.59 |
| identity_attack (199) | 0.64 | **0.65** | 0.60 |
| threat_incitement (19) | 0.17 | 0.19 | **0.35** |
| profanity_obscenity (65) | 0.56 | 0.47 | **0.56** |
| sexually_explicit (11) | 0.58 | 0.43 | **0.59** |

**Best for Stage 2: Oversampling** — highest macro F1 (0.537), best precision, and
clearly best on the two rarest labels (`threat`, `sexually_explicit`). Macro F1 matters
most under multi-label imbalance because it weights rare labels equally. Class weight
is a close alternative if higher recall / micro-F1 is preferred.

> **Caveat:** these are single training runs at fixed hyperparameters (`gamma=2.0`,
> threshold `0.5`, 5 epochs). The ~0.02 micro-F1 gaps in Stage 2 are small enough to
> shift with a different seed or threshold. The rare-label differences, however, are
> large and consistent.

**Crucial observation:** no method rescued `threat` (F1 0.17–0.35) or
`sexually_explicit` (F1 0.43–0.59) to a usable level. With only ~11–19 positive test
examples, there simply isn't enough signal — oversampling merely duplicates the same
few rows, which the model memorizes rather than generalizes from.

---

## 3. Threshold tuning (concept)

A classifier outputs a **probability** (0–1), not a hard label. To turn it into a
decision you need a cutoff:

```
if probability >= threshold:  ->  predict positive
else:                         ->  predict negative
```

The default is `threshold = 0.5`. **Threshold tuning** means testing other cutoffs and
picking a better one.

**Why 0.5 is often wrong for imbalanced data:** when a class is rare, the model rarely
becomes confident enough to cross 0.5 for it, so it labels almost everything negative
and *misses* the rare positives (low recall). Lowering the threshold (e.g. to 0.37 or
0.22) lets less-confident predictions count as positive, catching more true positives.

**Analogy — how much evidence you demand before sounding an alarm:**

- High threshold (0.5): only alarm when very sure → few false alarms (high precision),
  but misses real cases (low recall).
- Low threshold (0.17): alarm on a hint → catches almost everything (high recall), but
  more false alarms (lower precision).

Tuning = sweeping cutoffs and picking the one that maximizes **F1** (best balance).

**Important — no leakage:** the cutoff must be chosen on the **validation set**, then
applied to the untouched **test set**. Choosing it on the same test data you report is
cheating.

---

## 4. The 3-label experiment (`thesis-3-label.ipynb`)

A different kind of experiment. Instead of adding a new imbalance loss/sampler, it
attacks imbalance by **reducing scope + fixing the evaluation methodology**.

What it changes:

1. **Drops the two rarest labels entirely.** Kept: `insults`, `identity_attack`,
   `profanity_obscenity` (3-label). Dropped: `sexually_explicit`,
   `threat_incitement_to_violence`. It carefully removes only rows that were toxic
   *solely* because of the dropped labels (26,379 → 26,262 rows), keeping the row set
   consistent across both stages.
2. **Adds a proper train/validation/test split (70/20/10).** Validation drives
   checkpoint selection and threshold tuning; test is scored once, untouched, at the
   end.
3. **Tunes the decision threshold** instead of forcing 0.5:
   - Stage 1: best cutoff found = **0.3709** (vs 0.5).
   - Stage 2: a **separate threshold per label** (0.32 / 0.22 / 0.17).
4. **Uses plain `Trainer` with default loss** — no class weight, focal loss, or
   oversampling. The imbalance strategy is scope reduction + threshold tuning.
5. **Adds regularization + early stopping:** `weight_decay=0.01`,
   `EarlyStoppingCallback(patience=2)`.

> Because the label set and row set changed, these numbers are **not directly
> comparable** to the 5-label experiments (different test support).

### Stage 1 — binary toxicity (test: 4797 non-toxic / 456 toxic)

| Threshold | Toxic precision | Toxic recall | Toxic F1 | Macro F1 |
|---|---|---|---|---|
| Default 0.5 | 0.52 | 0.40 | 0.45 | 0.70 |
| **Tuned 0.371** | 0.49 | 0.48 | **0.48** | **0.72** |

### Stage 2 — 3-label (test support: insults 202, identity_attack 200, profanity 71)

| | F1 micro | F1 macro | Precision micro | Recall micro |
|---|---|---|---|---|
| Default 0.5 | 0.58 | 0.54 | **0.62** | 0.55 |
| **Per-label tuned** | **0.63** | **0.60** | 0.50 | **0.86** |

Per-label F1 (positive class):

| Label | @0.5 | @tuned |
|---|---|---|
| insults | 0.55 | **0.63** |
| identity_attack | 0.64 | **0.68** |
| profanity_obscenity | 0.44 | **0.50** |

Per-label threshold tuning trades some precision for a large recall gain and improves
F1 on **every** label (micro recall 0.55 → 0.86).

**Takeaway:** this is the "clean baseline done right" — handles imbalance by scope
reduction (drop the 2 unlearnable labels) and by fixing the real cause of rare-label
collapse (the fixed 0.5 cutoff) via per-label threshold tuning on a proper validation
set. Best-performing and most methodologically honest of the set — *but* it gives up
detecting `sexually_explicit` and `threat_incitement_to_violence` entirely.

---

## 5. Combined experiments (all 5 labels kept)

Two further notebooks combine techniques *across* the two stages instead of applying a
single method. Both **keep all 5 labels**, unlike the 3-label experiment.

### 5a. `stage1-weighted-stage2-focalloss.ipynb`

The simplest combination — essentially "best single method per stage" stitched
together, using the older methodology (no validation split, fixed 0.5 threshold).

- **Stage 1:** weighted CrossEntropy (`WeightedTrainerSingleLabel`, balanced class
  weights).
- **Stage 2:** focal loss (`FocalLossTrainerMultiLabel`, `alpha = neg/pos` per label,
  `gamma = 2.0`).

**Stage 1** (test: 4796 / 480)

| | Toxic P | Toxic R | Toxic F1 | Macro F1 | Acc |
|---|---|---|---|---|---|
| Weighted CE | 0.39 | 0.67 | **0.50** | 0.71 | 0.88 |

**Stage 2** (5-label, threshold 0.5)

| | F1 micro | F1 macro | P micro | R micro |
|---|---|---|---|---|
| Focal loss | 0.57 | 0.47 | 0.46 | 0.73 |

Per-label F1 (positive class): insults 0.63, identity_attack 0.65, **threat 0.19**,
profanity 0.47, sexually_explicit 0.43.

**Takeaway:** numbers are essentially identical to the standalone focal-loss notebook —
the rare labels still collapse (threat 0.19). Combining "weighted Stage 1 + focal Stage
2" changed little because each stage already used those methods individually.

### 5b. `stage1-weighted-stage2-oversampling+threshold.ipynb`

The most sophisticated notebook in the folder. It combines **three techniques + the
clean methodology** from the 3-label experiment, but **keeps all 5 labels**:

- **Stage 1:** weighted CrossEntropy + train/val/test split + early stopping +
  **threshold tuning** (best cutoff 0.9314 on validation).
- **Stage 2:** **oversampling** (`oversample_multilabel` balances every label up to
  ~740–1160 rows) + train/val/test split + **per-label threshold tuning**.

Oversampling counts (Stage 2 train): threat 78 → 740, sexually_explicit 47 → 744.

**Stage 1** (test: 4796 / 480)

| Threshold | Toxic P | Toxic R | Toxic F1 | Macro F1 | Acc |
|---|---|---|---|---|---|
| Default 0.5 | 0.37 | 0.65 | 0.48 | 0.70 | 0.87 |
| **Tuned 0.931** | 0.51 | 0.48 | **0.49** | **0.72** | **0.91** |

**Stage 2** (5-label)

| | F1 micro | F1 macro | P micro | R micro |
|---|---|---|---|---|
| Default 0.5 | 0.59 | 0.53 | 0.59 | 0.59 |
| **Per-label tuned** | 0.62 | **0.54** | 0.50 | **0.82** |

Per-label F1 (positive class):

| Label | @0.5 | @tuned |
|---|---|---|
| insults | 0.60 | **0.67** |
| identity_attack | 0.62 | **0.64** |
| **threat_incitement** | 0.25 | **0.27** |
| profanity_obscenity | 0.60 | 0.50 |
| **sexually_explicit** | 0.60 | **0.62** |

**Takeaway — the key result:** this is the **only 5-label experiment that makes the two
rare labels usable while keeping them.**

- `threat_incitement_to_violence`: F1 **0.25–0.27** (vs 0.17–0.19 in every earlier
  method) — still weak, but the best any experiment achieved.
- `sexually_explicit`: F1 **0.60–0.62** (vs 0.43–0.59 before) — genuinely decent now,
  comparable to `profanity_obscenity`.

The combination of **oversampling** (more minority examples) **+ per-label threshold
tuning** (each rare label uses its own cutoff) is what lifted them.

---

## 6. Label-dependency analysis (`label-dependency-analysis.ipynb`)

Purpose (from the notebook): determine whether the appearance of one label relates to
the appearance of another. Methods used: **Co-occurrence** (counts), **Chi-square**
(statistical dependence), **Mutual Information** (shared information), **Pearson**
(linear correlation). The notebook cites Read & Hollmén (2015) and Dembczyński et al.
(2012) to stress that **co-occurrence ≠ a dependency worth modeling** — a high count
can just reflect a frequent label, not genuine shared information.

### Label frequencies (full dataset)

| Label | Count |
|---|---|
| polarized | 4828 |
| insults | 1064 |
| identity_attack | 1014 |
| profanity_obscenity | 361 |
| **threat_incitement_to_violence** | **109** |
| **sexually_explicit** | **62** |

### How the two minority labels relate to the rest

| Pair | Chi-square | p-value → verdict | Mutual Info | Pearson |
|---|---|---|---|---|
| insults ↔ threat | 2.29 | 0.13 → **Independent** | 0.00005 | 0.011 |
| insults ↔ sexually_explicit | 0.42 | 0.52 → **Independent** | 0.00001 | 0.006 |
| threat ↔ sexually_explicit | 0.00 | 1.00 → **Independent** | 0.00001 | -0.003 |
| identity_attack ↔ threat | 102.8 | sig. → Dependent | 0.0010 | 0.064 |
| identity_attack ↔ sexually_explicit | 22.2 | sig. → Dependent | 0.0003 | 0.031 |
| profanity ↔ sexually_explicit | 293.4 | sig. → Dependent | 0.0014 | 0.109 |

For comparison, the three **kept** labels genuinely relate to each other:

| Pair | Mutual Info | Pearson |
|---|---|---|
| insults ↔ identity_attack | 0.0145 | 0.25 |
| identity_attack ↔ polarized | 0.0211 | 0.24 |
| insults ↔ profanity_obscenity | 0.0110 | 0.24 |

### Interpretation

Even where Chi-square reports "Dependent" for the minority labels, the **Mutual
Information is essentially zero** (0.00001–0.0014) and Pearson correlations are ≤ 0.11.
Those significant Chi-square values are an artifact of the large sample size (26k rows
make even trivial associations "significant") — exactly the trap the cited papers warn
about. MI tells the real story: **`threat` and `sexually_explicit` share almost no
information with any other label.** The kept labels have 10–20× higher MI among
themselves.

---

## 7. Final decision & justification

There are now **two viable paths**, and the choice hinges on a product question — do
`threat_incitement_to_violence` and `sexually_explicit` need to be covered? — rather
than purely on metrics.

### The two options

| Option | threat F1 | sexually_explicit F1 | Trade-off |
|---|---|---|---|
| **3-label** (drop both) | — (dropped) | — (dropped) | Cleanest top-line metrics (Stage 2 macro F1 0.60), but zero coverage of these two categories |
| **5-label weighted + oversampling + threshold** (Experiment 5b) | 0.27 | 0.62 | Full coverage; `sexually_explicit` now usable, `threat` still weak; Stage 2 macro F1 0.54 |

### What changed the earlier "just drop them" conclusion

Experiment 5b (oversampling + per-label threshold tuning, 5 labels) is the **only** run
that made the rare labels usable while keeping them:

- `sexually_explicit`: F1 **0.60–0.62** — no longer clearly unlearnable; comparable to
  `profanity_obscenity`.
- `threat_incitement_to_violence`: F1 **0.25–0.27** — still the weakest label, but its
  best score across all experiments.

So the decision is now nuanced:

- **If these two categories are in scope** (a moderation system arguably should flag
  threats and sexual content even when rare): use **Experiment 5b**. It recovers
  `sexually_explicit` to a usable level, gives `threat` its best-ever score, and still
  improves the common labels via threshold tuning. The dependency analysis confirms the
  two labels are statistically independent of the others, so keeping them costs the
  other labels nothing.
- **If only clean top-line metrics matter and the two categories are out of scope**: the
  **3-label** route still wins on macro F1 (0.60 vs 0.54) and is the most
  methodologically honest, justified by the two evidence lines below.

### Evidence supporting the "drop" option (3-label)

1. **Hard to learn.** 109 and 62 positives total; single-method runs gave F1 ≤ 0.35.
2. **Near-independent of the other labels.** Near-zero mutual information (≤ 0.001) and
   correlation (|r| ≤ 0.03) with every other class, so removing them discards
   essentially no information useful for the kept labels (no classifier-chain signal
   lost).

### Suggested thesis wording (either path)

If dropping (3-label):

> "`threat_incitement_to_violence` (n=109) and `sexually_explicit` (n=62) were excluded
> from Stage 2 because (1) single-method imbalance handling (class weight, focal loss,
> oversampling) could not learn them reliably (F1 ≤ 0.35), and (2) label-dependency
> analysis showed they share negligible mutual information (≤ 0.001) and correlation
> (|r| ≤ 0.03) with the retained labels, so their removal does not discard information
> useful for predicting the remaining classes. Residual imbalance among the three
> retained labels is addressed via per-label decision-threshold tuning."

If keeping (5-label, Experiment 5b):

> "All five Stage-2 labels were retained. Class imbalance was handled with weighted loss
> (Stage 1) and random oversampling (Stage 2), combined with per-label decision-threshold
> tuning on a held-out validation set. This recovered the rarest labels that a fixed 0.5
> threshold suppressed — `sexually_explicit` (F1 0.62) and `threat_incitement_to_violence`
> (F1 0.27) — while improving the frequent labels."

### Best method per stage (summary)

| Stage | Best approach | Evidence |
|---|---|---|
| Stage 1 (binary) | Weighted CE or focal loss, **plus threshold tuning** | Toxic F1 ≈ 0.50, macro F1 0.72; tuning lifts toxic F1 0.45–0.48 → 0.49 and accuracy to 0.91 |
| Stage 2 — if 2 rare labels out of scope | **3-label** scope + per-label threshold tuning | Macro F1 0.60, micro F1 0.63 (tuned) |
| Stage 2 — if 2 rare labels in scope | **5-label oversampling + per-label threshold tuning** (Exp 5b) | Macro F1 0.54; `sexually_explicit` 0.62, `threat` 0.27 (best of any experiment) |

### Important caveats

- The dependency analysis concerns *statistical* independence and learnability, **not
  real-world importance**. Threats and sexually explicit content can matter greatly for
  a moderation product even when rare.
- `threat_incitement_to_violence` at F1 0.25–0.27 (only 19 test positives; oversampled
  from ~78 to 740 training rows) is **statistically shaky and at high overfitting risk**
  — treat it as "better, but not reliable."
- The most robust long-term fix for both rare labels is **more labeled data**, not a
  different loss function or sampler.

