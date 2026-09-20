# Why string-only ZeroER collapses on the product data — diagnosis (2026-09-20)

Step 2 of `NEXT-STEPS-matching.md`, first half ("first, look" + the mean-imputation
proxy). Produced by `collapse_diagnosis.py`; tables in `blocks.csv`,
`presence_confusion.csv`, `treatments.csv`; fits in the `<dataset>_collapse` studies of
`outputs/optuna_matching.db` (80 trials, 6 FAIL). Cells: the 8 `abt_buy` /
`amazon_googleproducts` rungs + two working controls (`dblp_acm` left/2,
`dblp_scholar` left/5), under B-paper and B-code. Gold scores and diagnoses only.
One run per cell, no uncertainty estimate.

Checks: zero-fill reproduces the ladder on all 20 (cell, config) rows (max |ΔF1| 2e-16);
the presence mask rebuilt from `reference_features._tables` agrees with the matrix on
100% of pairs (absent ⇔ all-zero block); per-block log-likelihood ratios sum to the
model's own `Q_M − Q_U` within 1e-7. All diagnostics are read after one extra E-step,
so parameters and labels belong to the same state (≤ 6 labels change, except
`dblp_scholar` left/5 B-code mean-impute, which stopped unsettled at the 40-iteration cap).

## 1. The presence split is observed, not inferred

φ = phi coefficient of the fitted match labels. Zero-fill, B-paper:

| cell | φ vs gold | φ vs sparse attribute present | predicted that have it | pairs having it that are predicted | base rate |
|---|---:|---:|---:|---:|---:|
| abt_buy left/13 | −0.00 | 0.38 (`price`) | 100% | 19% | 26% |
| abt_buy right/5 | 0.00 | 0.76 | 100% | 64% | 24% |
| abt_buy left/2 | −0.04 | 0.79 | 100% | 69% | 26% |
| abt_buy left/1 | −0.13 | **1.00** | 100% | **100%** | 24% |
| amazon right/5 | 0.10 | 0.82 (`manufacturer`) | 70% | 99% | 7% |
| amazon right/2 | 0.10 | 0.78 | 64% | 99% | 7% |
| amazon left/2 | 0.11 | 0.70 | 52% | 100% | 6% |
| amazon left/1 | 0.08 | 0.61 | 43% | 96% | 5% |
| dblp_acm left/2 (control) | **0.96** | 0.11 | | | |
| dblp_scholar left/5 (control) | **0.77** | ≤ 0.20 | | | |

* `abt_buy` left/1: the "match" component **is** "price present on both sides" — all 264
  predicted matches are the 264 price-present pairs (independently re-checked).
* B-code lands on the same attribute, more narrowly (100% of predicted matches have the
  sparse attribute on all eight cells; φ vs gold 0.03–0.05 on amazon).
* The sparse block carries the posterior; the key attribute contributes ~nothing. Mean
  block LLR, B-paper: `abt_buy` left/1 `price` **+286.7** on present pairs, −14.5 on absent,
  `name` 0.00; amazon left/2 `manufacturer` **+2282.7** present, −9.9 absent, `title` −0.03.
  On the controls every block separates the gold classes by sign.
* **amazon `price` is a second problem missingness does not explain**: never missing, yet
  its block votes "match" for both gold classes (σ_M 0.33–0.38 vs σ_U 0.15–0.17; 14.6% of
  Amazon prices are the value `0`). Not examined further.

## 2. The initialisation already points at presence; the EM sharpens it

The scaled feature sum rewards pairs with more non-zero blocks: under B-paper the *initial*
labels have φ 0.36–0.55 vs the sparse attribute and ≤ 0.16 vs gold on the product cells
(controls: 0.89 / 0.69 vs gold). The EM takes that to 0.8–1.0. Under B-code the seed is
tiny (`init_frac` 0.001–0.012) and the EM still ends on the sparse attribute — the init
contributes but is not the whole cause, consistent with step 0's ε sweep.

## 3. A second mechanism the plan has no arm for: the κ clip erases the key block

Share of columns whose regulariser κ sits at `m_step`'s clip of 1 (variance +1 on a [0, 1]
feature → σ_M ≈ σ_U ≈ 1, LLR ≈ 0):

| | key block (`name`/`title`) | other blocks | controls |
|---|---:|---:|---:|
| zero-fill, B-code | **88%** | 47% | 0% |
| zero-fill, B-paper | **50%** | 10% | 4% |
| key-attribute only | 0% | | 0% |

Cause (verified): the released `bay_coeff` equals **BC⁴/4** of the paper's Eq. 10, so its
maximum is **0.25**; `m_step`'s cap rule tests `target_bay >= 1` and can never fire. Once M
and U overlap in a column (`bay_ori` 0.245–0.249 for `name` after the split is on price)
the target `bay_ori + c_bay` has no root; the single secant step extrapolates and κ is
clipped to 1. `c_bay` 0.1 makes 40% of the coefficient's range unreachable, 0.01 makes 4% —
why B-code flattens more and why larger `c_bay` is worse on `abt_buy`. Note the unit
mismatch this implies: the paper's κ′ is a change in BC, the code's `c_bay` a change in
BC⁴/4. The loop: presence split → key block's components coincide → κ = 1 flattens it →
nothing pulls the split back. Causal order is read off the final state only; settling it
needs an iteration trace (not run).

## 4. The mean-imputation proxy (F1, B-paper; B-code in parentheses)

| cell | zero-fill | mean-impute | key attribute | accept-all | oracle |
|---|---:|---:|---:|---:|---:|
| abt_buy left/13 | 0.059 (0.035) | 0.089 (0.035) | 0.334 (0.105) | 0.144 | 0.999 |
| abt_buy right/5 | 0.173 (0.036) | 0.175 (0.036) | 0.443 (crash) | 0.331 | 0.995 |
| abt_buy left/2 | 0.220 (0.034) | 0.244 (0.034) | 0.638 (crash) | 0.621 | 0.960 |
| abt_buy left/1 | 0.292 (0.036) | 0.292 (0.034) | 0.784 (crash) | 0.831 | 0.904 |
| amazon right/5 | 0.176 (0.096) | 0.176 (0.178) | 0.445 (0.494) | 0.147 | 0.993 |
| amazon right/2 | 0.215 (0.095) | 0.213 (0.182) | 0.496 (0.538) | 0.309 | 0.960 |
| amazon left/2 | 0.198 (0.074) | 0.510 (0.198) | 0.479 (0.594) | 0.525 | 0.896 |
| amazon left/1 | 0.158 (0.055) | 0.609 (0.102) | 0.392 (0.631) | 0.628 | 0.783 |
| dblp_acm left/2 | 0.975 (0.944) | 0.976 (0.949) | 0.938 (0.660) | 0.595 | 0.999 |
| dblp_scholar left/5 | 0.794 (0.537) | **0.864 (0.778)** | 0.835 (0.713) | 0.522 | 0.947 |

* **Mean imputation does not repair the product data because it does not remove the
  presence signal**: imputed rows sit at one point, present rows are spread, and the
  mixture separates "spike" from "spread" (φ vs `price` rises to 0.94–1.00). On amazon
  left/2, left/1 the split *inverts* (π_M 0.94 / 0.95) and F1 ≈ accept-all — a rescue to the
  floor, not a repair. The proxy is a poor stand-in for marginalising absent blocks, so its
  failure is **no evidence against the missing-aware likelihood**. Where it helps, the
  plan's caveat holds: it also changes the init and the covariances.
* It helps where the informative blocks are healthy (`dblp_scholar` left/5 0.794 → 0.864):
  zero-fill treats an absent value as extreme disagreement even in working cells (absent-row
  LLR for `authors` −68.6 on `dblp_scholar`, −147.9 on `dblp_acm`).
* **Dropping the other blocks is what moves the product data**, yet key-attribute-only
  stays far from the oracle and its fitted φ vs gold is only 0.10–0.41 (controls 0.81–0.89):
  the `name`/`title` *string* features separate the classes weakly. That is the gap the
  embedding columns are meant to fill; a missingness repair will not close it.

## 5. What this means for the repair arms

1. **Missing-aware likelihood stays the first arm, now on observed evidence** — paired with
   the init-over-present-blocks ablation, since the seed is itself a presence split. Expect
   it to fix `abt_buy price`, amazon `manufacturer` and the absent-value penalties on
   `dblp_scholar`; it will not fix amazon `price`.
2. **Drop-by-missingness**: key-attribute-only is its extreme case. The intermediate arm
   (drop only blocks below a presence rate: `price` 24–26%, `manufacturer` 5–7%) was not run.
3. **`abt_buy description` does not drive the collapse** (present on 60–66% of pairs, LLR
   within ±0.75, fitted φ 0.00–0.05). Re-inspect once the `price` contrast is gone.
4. **The κ clip is a separate lever.** The unreachable-target condition
   `bay_ori + c_bay > 0.25` is label-free and cheap to log per iteration. Candidate fixes — a
   cap rule at the true maximum, or leaving such columns unregularised — are model changes
   that belong behind a flag. Decide before evaluating the missing-aware likelihood: a
   repaired likelihood may still have its key block flattened.

## Unresolved

Causal order of the presence split and the κ clip; why `abt_buy` left/13 under B-paper
looks different (no κ at the clip, predicted component a 19% subset of price-present
pairs); amazon `price`; the intermediate drop-by-missingness arm, the 1-D gate on
attributes and the likelihood change itself are not run.
