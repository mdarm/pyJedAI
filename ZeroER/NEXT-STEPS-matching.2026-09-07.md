# Next: matching — three axes on one feature matrix

Thesis refocused on the **matching stage** itself: per-column embedding features
and other ML algorithms that fit that stage, with ZeroER staying in the pipeline.
The three-axis design (features × matcher × operating point) replaced an earlier
two-arm structure that had only one matcher and one feature family — see §"What
changed and why".

Revised 2026-09-07. This pass records that **the ladder is pinned, materialised
and measured** (§"The ladder"), corrects two claims in §"Status" that did not
survive checking, and folds the trailing notes blob into
§"The operating points are chosen". Completed work is stripped from the run order and
compressed into §"Status".

Wiki refs are page slugs in `msc-thesis/wiki/wiki/pages/`.

---

## Status — what exists, so it is not re-planned

**Phase 0 is built, verified and committed** (`9048d27`, 2026-08-28, on
`michail-dev`; the tree is clean apart from tooling files). It shipped:

* **`data.py`** — the embedding tree split into siblings `RECORD_DIR`
  (`embeddings/record/`, `data.py:37`) and `ATTR_DIR` (`embeddings/attribute/`,
  `data.py:38`); `embedding_models` / `load_embeddings` default to `RECORD_DIR`.
  This is what keeps a per-attribute file from registering as a model named
  `<model>_<attr>` in the blocking sweep. Confirmed: `embedding_models()`
  returns exactly 20.
* **`embedding_features.py`** — `build_embedding_features(pairs, dataset, model,
  attributes, data, metrics=("cos",))` emits `{attr}_emb_{metric}` columns
  row-aligned with `pairs`; `embedding_feature_builder(...)` binds everything but
  `pairs`. `METRICS` carries `cos` and `linf` only (§"One metric per attribute").
* **`extra_features` hook** on `BlockFeatureCache`, `ZeroEREstimator`
  (`estimator.py:109`, `:221`) and `ZeroERMatcher` (`matcher.py:56`). Passing it
  to both an estimator and its cache raises; a row-count mismatch raises.
* **`tests/test_embedding_features.py`** — six tests pinning block naming,
  absent → exactly 0, partition re-selection, cache-slice exactness, and both
  guards. All six **run and pass** — the vector tree is present locally.

**Encoding is complete.** 20 record models and 20 complete per-attribute model
sets on all five datasets — every `(model, attr, side)` file present, verified by
file count (13G under `record/`, 51G under `attribute/`). Nothing below is blocked
on a GPU job.

**The blocking sweep is the clean reference grid.** Five `<dataset>_blocking`
studies × 4000 trials, the only studies in the live `outputs/optuna.db`. Every
trial stores `tp` / `candset_size` / `n_gold` / PC / PQ, so the operating points
in §"The ladder" needed no re-run. The ten inferior
`block-filter[_full]` studies were deleted from the live DB and survive in
`outputs/optuna.db.bak-preaugust` — quarantine done.

→ **One live provenance risk.** `embedding_times.txt` records a `phase=record`
re-encode of the ten sentence-transformer baselines on 2026-08-11/12 whose
outputs were **never copied down** — every `.npy` under `record/` still carries
mtime 2026-08-06. Harmless today. But if a newer `record/` is ever pulled off
Kaggle, the model *names* are identical, so a join on
`(embedding_model, top_k, similarity_distance)` keys perfectly and silently pairs
**different vectors** — the exact failure that killed the `block-filter` studies.
Checksum `record/` before replacing anything in it.

**The test suite runs — but the replication test is not among what it verifies.**
`pytest` is in `data-env`; `pytest tests/` gives **8 passed, 2 skipped**. An
earlier revision recorded the two skips as the embedding-feature tests awaiting an
absent vector tree. **That was wrong**: the tree is present, all six embedding
tests pass, and the two skips are `tests/test_paper_replication.py`, gated behind
`RUN_SLOW=1`.

→ **The replication test is excluded by design, not by accident.** Its two slow
params replay upstream on the pinned Magellan candsets — `dblp_acm` 46,456 pairs and
`abt_buy` **164,072** — through pure-Python per-pair string similarity, and
`_load_case` calls `build_zeroer_features` without `n_jobs`, so it runs
single-threaded. The docstring's "tens of minutes" is accurate; a run started
2026-09-07 was still going at 33 minutes on one core. Treat it as a deliberate
pre-release check, not a per-commit gate. If it is ever wanted routinely, pass
`n_jobs=-1` in `_load_case` — the same parallelisation `BlockFeatureCache` already
uses — rather than trimming the fixtures, which would break the exact-F1 assertions.

**Magellan reference features are built for every ladder cell, and the parity test
is gone.** `reference_features.py` runs upstream's own
`data_loading_helper.feature_extraction.extract_features` — genuine
`py_entitymatching`, not this repo's port — over the ladder's candidate sets.
**All 19 cells are complete** (`datasets/magellan-features/<dataset>__<side>_k<k>.csv.gz`,
last write 2026-09-07 01:44). Nothing in step 1 or 2 waits on this.

Two stages, two interpreters — kept here only because a re-run needs it:

```
data-env/bin/python                  reference_features.py pairs
~/micromamba/envs/ZeroER/bin/python  reference_features.py features
```

The py3.6 env is `~/micromamba/envs/ZeroER` (micromamba 2.9.0 at `~/bin/micromamba`,
`MAMBA_ROOT_PREFIX=$HOME/micromamba`), built from upstream's `environment.yml`
with the `setuptools` pin dropped — that version is not on conda-forge and is
not load-bearing. `py_entitymatching` 0.4.0 / numpy 1.19.5 / pandas 1.1.5.

This replaced `tests/test_feature_parity.py`, `tests/generate_magellan_fixtures.py`
and the 2% `tests/fixtures/magellan/` samples, all removed. The port is not
unverified by the removal: on `fodors_zagats` right k=1, our
`build_zeroer_features` and upstream's `extract_features` produce **identical
column sets (73 each) and values agreeing to 1.1e-16** — one ULP, on real ladder
pairs rather than a 2% sample. Re-run that check against any cell's stored matrix
whenever the port changes.

→ **One real port discrepancy, found by the full-cell check.** On
`dblp_scholar`, exactly **one feature of 37** disagrees with Magellan:
`year_year_lev_sim` (max |diff| 0.333; every other column, and every column on
the other four datasets, agrees to 1.1e-16).

*Cause.* Magellan never concatenates the tables — it resolves each pair through
the fk columns of separate `ltable` / `rtable`, so `left.year` stays `int64`
(`'1995'`) while `right.year`, 54% missing, is `float64` (`'1992.0'`). Our port
takes pyJedAI's single concatenated `entities` frame, and `pd.concat` of
int64 + float64 yields float64 — so the left side becomes `'1995.0'` and the
Levenshtein similarity differs. It bites only where the two tables carry
different numeric dtypes for one attribute, i.e. where one side has NaNs and the
other does not; among the five benchmarks that is `dblp_scholar.year` alone.

*Impact: nil in practice, but do not let it drift.* The screening table below
measures `dblp_scholar year` at separation **−0.02** — the feature carries no
signal either way. **Not fixed** — the concatenated frame is pyJedAI's interface,
so a fix means per-side dtype preservation in `build_zeroer_features`, which is a
design change, not a patch. Left as a known, characterised difference; the note
in `zeroer_features.py`'s docstring claiming verification against upstream lists
`fodors_zagats`, `dblp_acm`, `abt_buy` and `fodors_zagats_single` — **not**
`dblp_scholar` or `amazon_googleproducts`, which is consistent with this only
surfacing now.

→ **Budget for `amazon_googleproducts`.** Its left `description` reaches **19,263
characters**, which Magellan types `str_gt_10w` and then featurises with
`jaccard qgm_3` and `cosine dlm_dc0` — q-gram sets over 19k-character strings,
per pair. Measured: the 16,130-pair cell burned **>11 cpu-hours across 15
workers**, against ~4 wall-minutes for `abt_buy`'s comparable 14,053-pair cell.
The same feature set runs in `zeroer_features`, so **this cost lands on step 1
too**, and it scales with `BlockFeatureCache`'s `k_max`, not with the trial's
`top_k`. Pick `k_max` deliberately on that dataset, or accept a multi-hour
one-time build per (model, distance).

→ **The posterior is now retained — step 0a is done.** `match_pairs` used to keep
only the `P_M >= 0.5` edges, discarding the rest (on `fodors_zagats` at 662 pairs
that is **570 of 662 posteriors, 86%**). It now stores `candidate_pairs` and the
full `posteriors` vector, exposed as `ZeroERMatcher.posterior_frame()` and
`ZeroEREstimator.posterior_frame()` → `(id1, id2, p_match)` on pyJedAI internal
ids. Verified continuous, not rounded: 658 distinct values over 662 pairs, 646
strictly interior. A length mismatch between pairs and posteriors now raises
instead of being silently truncated by `zip`. Covered by
`tests/test_posteriors.py` (six tests, including the `BlockFeatureCache` route the
ladder runs on); suite is **14 passed, 2 skipped**.

---

## The ladder — pinned, materialised, measured

**This is done and must not be re-searched.** `reference_features.LADDER` fixes the
cells; both the candidate pairs (`outputs/ladder_pairs/`) and the Magellan
reference features exist for all 19. Blocking model
`tencent_KaLM-Embedding-Gemma3-12B-2511`, `cosine`, throughout.

Recomputed from the stored pairs against full gold, 2026-09-07 — `side` is the
**queried** side, so `|C| = n_queried × k` exactly:

| dataset | side/k | \|C\| | tp | gold | PC | PQ | oracle F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fodors_zagats | right/4 | 1,324 | 112 | 112 | 1.000 | 0.085 | 1.000 |
| fodors_zagats | left/1 | 533 | 112 | 112 | 1.000 | 0.210 | 1.000 |
| fodors_zagats | right/1 | 331 | 112 | 112 | 1.000 | 0.338 | 1.000 |
| dblp_acm | right/12 | 27,528 | 2,224 | 2,224 | 1.000 | 0.081 | 1.000 |
| dblp_acm | right/5 | 11,470 | 2,221 | 2,224 | 0.999 | 0.194 | 0.999 |
| dblp_acm | left/2 | 5,232 | 2,220 | 2,224 | 0.998 | 0.424 | 0.999 |
| dblp_acm | left/1 | 2,616 | 2,210 | 2,224 | 0.994 | 0.845 | 0.997 |
| abt_buy | left/13 | 14,053 | 1,094 | 1,097 | 0.997 | 0.078 | 0.999 |
| abt_buy | right/5 | 5,460 | 1,085 | 1,097 | 0.989 | 0.199 | 0.995 |
| abt_buy | left/2 | 2,162 | 1,012 | 1,097 | 0.923 | 0.468 | 0.960 |
| abt_buy | left/1 | 1,081 | 905 | 1,097 | 0.825 | 0.837 | 0.904 |
| amazon_googleproducts | right/5 | 16,130 | 1,282 | 1,300 | 0.986 | 0.079 | 0.993 |
| amazon_googleproducts | right/2 | 6,452 | 1,199 | 1,300 | 0.922 | 0.186 | 0.960 |
| amazon_googleproducts | left/2 | 2,726 | 1,056 | 1,300 | 0.812 | 0.387 | 0.896 |
| amazon_googleproducts | left/1 | 1,363 | 836 | 1,300 | 0.643 | 0.613 | 0.783 |
| dblp_scholar | right/1 | 64,263 | 5,178 | 5,347 | 0.968 | 0.081 | 0.984 |
| dblp_scholar | left/10 | 26,160 | 5,219 | 5,347 | 0.976 | 0.200 | 0.988 |
| dblp_scholar | left/5 | 13,080 | 4,812 | 5,347 | 0.900 | 0.368 | 0.947 |
| dblp_scholar | left/1 | 2,616 | 2,374 | 5,347 | 0.444 | 0.907 | 0.615 |

`oracle F1 = 2·PC/(1+PC)` — a perfect matcher keeps exactly the `tp` gold pairs in
`C`, so precision 1.0 at recall PC. **Report it in every results row**; without it,
0.60 at `dblp_scholar` left/1 (ceiling 0.615) and 0.60 at right/1 (ceiling 0.984)
look identical when one is near-perfect and the other is poor.

**Why `fodors_zagats` has three rungs and not four.** At k=1 each queried record
contributes one pair, so `PQ ≤ min(n_gold, n_queried)/n_queried` — a hard cap over
the whole grid:

| dataset | n_left | n_right | gold | cap query-left | cap query-right |
| --- | ---: | ---: | ---: | ---: | ---: |
| fodors_zagats | 533 | 331 | 112 | 0.210 | **0.338** |
| dblp_acm | 2,616 | 2,294 | 2,224 | 0.850 | 0.969 |
| abt_buy | 1,081 | 1,092 | 1,097 | 1.000 | 1.000 |
| amazon_googleproducts | 1,363 | 3,226 | 1,300 | 0.954 | 0.403 |
| dblp_scholar | 2,616 | 64,263 | 5,347 | 1.000 | 0.083 |

`fodors_zagats` tops out at **0.338** on either side, so a PQ≈0.8 rung does not
exist at any k. Being saturated, it lands *on* its caps exactly (112/331, 112/533).

→ **Do not read the two short top rungs the same way.** `fodors_zagats` at 0.338 is
**cardinality** — no configuration can do better, a property of the dataset.
`amazon_googleproducts` at 0.613 is **retrieval** — its cap is 0.954, so a 0.8 rung
was available and the blocker simply missed it (836 of 1,300 gold at k=1). The
first is not a blocker result; the second is.

---

## What changed and why

The wiki confirms the plan's feature reasoning and contradicts its *scope*.

**Confirmed, unchanged:** per-attribute is the right shape
(`deeper-distributed-representations` is exactly this — per-dimension cosine over
averaged attribute DRs gives "an *m*-dimensional similarity vector (one entry per
attribute)", `distributed-representations-er`); Arm A must run first (*"reporting
ZeroER without stating the blocking configuration is close to meaningless"*,
`zeroer-entity-resolution-zero-labels` / `overview.md` #6); report the spread
across 20 models rather than the best (`ensemblelink-record-linkage`'s
53-points-vs-1–2 ablation); raw vectors and the Zeakis whole-record scalar stay
rejected as ZeroER features.

**Four changes.**

### 1. A second unsupervised matcher the plan had missed

`maximum-entropy-record-linkage` (Lee, Zhang & Kim 2021). **Unsupervised MEC
consumes the same similarity vectors, needs zero labels, and fixes two things
this file already lists as open problems:**

* **The fixed 0.5 threshold across a 12.5× prevalence swing.** MEC has no
  threshold. It builds the linked set `M̂` by maximising the entropy of the
  density ratio `r(γ) = m(γ)/u(γ)` — greedy in descending `r̂`, skipping any pair
  whose endpoint is already used.
* **The bipartite constraint.** All five benchmarks are Clean-Clean. MEC enforces
  "each record appears at most once" *inside* estimation; this plan currently
  bolts it on afterwards as UMC. That is Problem I of that paper, stated against
  F&S directly. Problem II is that Winkler-Jaro EM's "log-likelihood" assumes IID
  comparison vectors, impossible under bipartite constraints.

It also proves WJ-EM is a special case of unsupervised MEC (the updates differ
only in the support of the agreement sum), so it is a principled generalisation
of the family ZeroER descends from, not a rival. And it brings **FLR/MMR** error
semantics, which `overview.md` names as unclaimed: *"Can maximum-entropy-record-linkage's
FLR/MMR semantics be ported to the embedding-based DB-ER thread? … No source
ports this."*

Same features, same zero-label claim, a second matcher, an open question closed.

A fourth option sits beside it and is also unclaimed — `overview.md`:
*"Cross-thread integration of bayesian-record-linkage and bi-encoder ER … running
Bayesian inference on top of bi-encoder similarities as the comparison-vector
feature."* Optional; MEC is the cheaper of the two.

### 2. The screening table points at a model class, not just a feature filter

Three measured facts below — `abt_buy price` 0-inflated bimodal at 76.5% exact
zeros, `dblp_acm year` at σ_M = 0.000, `dblp_acm venue` at separation −1.02 — are
things ZeroER's Gaussian structurally cannot represent. The previous plan's
answer was to **drop those columns**.

`fellegi-sunter-framework` Step 1 handles all three natively: comparison vectors
are categorical, *"missing values get their own category"*, and continuous
similarities *"can be discretised into ordinal categories"*. Missing becomes a
level rather than a zero, so the misspecification disappears instead of costing a
column. That is why `moretti-shlomo-prediction-models` uses deciles — F&S is
multinomial. **Discretised F&S / WJ-EM on the same per-attribute cosines is a
third matcher, and the only one that can use the columns ZeroER must discard.**

### 3. Run a matcher tournament and read the winner as a result

`er-in-practice-self-serve` is the wiki's strongest operational claim about
matching: no architecture wins everywhere, so train several families and let a
tournament pick — DeepMatcher 3/6, LightGBM 3/6, GAT 0/6 across six benchmarks
(864 → 5M records, ≤10K labels). *"The tournament costs nothing extra because all
families share the same training pairs and embeddings."* The winner is a
**diagnostic**:

| Winner | What it says about the data |
| --- | --- |
| **LightGBM** (GBDT on attribute features) | small, or identifier-heavy — "the embedding tower lacks signal to fine-tune meaningfully" |
| **DeepMatcher** (MLP over embeddings + attributes) | soft semantic similarity matters — sparse attributes, OCR noise, paraphrase equivalence |
| **GAT** (GNN over the candidate graph) | large and densely co-referent; also suffers train/test graph mismatch |

*"None of these conditions can be reliably predicted from schema inspection
alone — only the data reveals which signal type dominates."*

Our feature matrix is already built by the time step 3 runs, so a GBDT over the
same columns is nearly free. **Label-freedom is preserved through the routes the
wiki documents** — `knowledge-distillation-er` (LLM teacher annotates, student
learns; avg F1 0.85 with zero gold labels) or `cross-dataset-transfer-er` (donor
picked by KL over post-filtering similarity distributions). State which route is
used; do not quietly train on gold.

### 4. Sparse lexical features are a second feature family, not an addendum item

The screening table's actual finding is that only `name`/`title` carries signal,
and the two product datasets need identifier-shaped evidence —
`entity-matching-7b-llms` §5.3 measured that **model number alone beats a
four-attribute prompt** on Abt-Buy. `ensemblelink-record-linkage`'s rationale is a
*feature* argument: dense catches `NYC` / `New York City`, sparse character
n-grams catch `Montegomery` / `Montgomery` — the families fail on disjoint inputs.

Character-n-gram TF-IDF cosine per attribute is cheap, is orthogonal to both
string-edit sims and embedding cosines, and lands in the *same* attribute group
under `module.py:168`'s `name.split("_")[0]` rule, so Innovation 1 models it
correctly with no naming trick. If "features per column" is the focus, one family
is not a study.

### And one demotion

**Drop the "reproduce ZeroER Table 3" framing.** The wiki records that the 2020
headline *"does not survive contact with 2025-26 pipelines"*: mean F1 **0.58**
over 8 datasets (`distiller-knowledge-distillation-er`), **0.520 / 0.644** on
Abt-Buy / Walmart-Amazon (`entity-matching-7b-llms`), and 35.4–70.9% → **0.00% /
0.15% / 5.23%** between fine-tuned and default filtering
(`end-to-end-deep-er-without-labels`).

Keep the run, change the headline to the **PQ ladder as a continuous curve**. That
70.93% → 0.00% is a two-point anecdote on three datasets; we can make it a
controlled curve on five, with `tp` / `candset_size` / `n_gold` already stored per
cell. Two consequences: it is a stronger claim than "the reference every later
delta subtracts from". The candsets that used to block this are now all pinned
and materialised (§"The ladder").

### The framing problem this refocus has to answer

Four independent 2026 sources converge on *"the matcher is no longer where the
headroom is"* — `opensanctions-pairs` (98.95 F1 zero-shot on 755,540 pairs),
`er-in-practice-self-serve`, `agentic-er-vision`, `ensemblelink-record-linkage`. A
matcher-focused thesis must meet that head-on. Two defences the wiki supports:

1. **"Saturation is a property of the data, not of the task."** OpenSanctions
   entities carry up to 131 fields; on 3–8-attribute Magellan/WDC benchmarks
   `ce-rag4em-blocking-rag` still recovers 2–24 F1 points from external evidence,
   and EnsembleLink sits ~5 pair-F1 below supervised Ditto on DBLP-Scholar.
2. **Move the question from accuracy to probability quality.** *"Is any DL ER
   matcher's output calibrated?"* — no paper in the corpus reports intercept /
   slope / flexible curve / Net Benefit, and ZeroER's posterior is named as
   unaudited across four pages (`calibration-prediction-models`,
   `class-imbalance-er`, `zeroer-entity-resolution-zero-labels`, `overview.md`
   #17). This is now a **headline deliverable, not a rider** — see step 3b. It
   also makes the singleton-group over-confidence prediction falsifiable, and it
   registers a result even where F1 does not move, which the measurement below
   says is the likely case.

---

## The design — three axes, one feature matrix

| Axis | Levels |
| --- | --- |
| **Features** | string-only (reference) · + per-attribute embedding cosine · + per-attribute char-n-gram TF-IDF cosine · embedding-only |
| **Matcher** | ZeroER GMM · unsupervised MEC · discretised F&S / WJ-EM · GBDT (label-free-supervised) · Zeakis `1/(1+dist)` + UMC (control) |
| **Operating point** | the PQ ladder — 19 pinned cells (§"The ladder") |

Reported with pair-P/R/F1 **and** calibration (intercept, slope), spread across
the 20 embedding models rather than the best.

This crossing is, in `overview.md`'s own words, *"the most directly actionable gap
the corpus now contains"*: **"Nobody has run a controlled ablation attributing
end-to-end F1 across the three stages on a common benchmark."** Say so in the
writeup — the previous revision had the ablation and never claimed it.

---

## What the measurements already constrain

### ZeroER's covariance decomposition dictates feature shape

`module.py:168` derives groups from `name.split("_")[0]`, and `module.py:447-457`
rebuilds `Cov_M` / `Cov_U` *from* `group_name_2_col_indices` — within-group
entries get `corr[i,j]·std[i]·std[j]`, everything outside stays zero. Four
consequences, still load-bearing:

1. **Per-attribute is the only shape that fits; a whole-record scalar
   misspecifies rather than corrupts.** A column named `emb_cos` opens a
   **singleton group**, and the code then *forces* it independent of every
   attribute block. Since a whole-record cosine really is correlated with all of
   them, the model double-counts and should be expected to be **over-confident**,
   not unstable. That makes DeepER — not Zeakis — the precedent to port, and it
   gives the over-confidence prediction an exact metric (step 3b).
2. **Grouping is by column name.** `{attr}_emb_{metric}` joins the `{attr}` group.
   **The name is a modelling decision, not a label.**
3. **Low variance is the known failure mode.** Innovation 2 (adaptive
   Bhattacharyya regularisation) exists because near-zero-variance features
   dominate and cause the singularity problem. Cosines *within a k-NN candidate
   set* are concentrated in a narrow high band by construction. `c_bay` is fixed
   at 0.1 and has never been tuned against this profile.
4. **Raw vectors are out for three independent reasons.** *(i)* `4d+1` was
   designed for d≈68; at d=4096 the pooled correlation matrix `R`
   (`module.py:158`) is 16.8M entries estimated from 331 pairs at `fodors_zagats`
   right k=1 — guaranteed singular, and `multivariate_normal.logpdf(...,
   allow_singular=True)` (`module.py:236`) will pseudo-invert it and return
   numbers that mean nothing. *(ii)* `get_y_init_given_threshold` min-max scales,
   sums and thresholds at 0.8; coordinates have no orientation and element-wise
   `|u−v|` has the inverted one, so the EM *initialisation* would be actively
   wrong. *(iii)* A per-coordinate Gaussian cannot represent an inner product.

### Innovation 1's premise survives on learned features

`cosine(R_M, R_U)` — the empirical fact the paper's Table 1 reports at 0.94–0.99.
Measured 2026-08-15 with `tencent_KaLM-Embedding-Gemma3-12B-2511`, at each
dataset's cheapest cell reaching PC ≥ 0.90, gold from `matches.csv`:

| dataset | side/k | pairs | M | string | emb | combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fodors_zagats | right/1 | 331 | 112 | 0.872 | 0.785 | 0.860 |
| dblp_acm | right/1 | 2,294 | 2,208 | 0.746 | 0.967 | 0.748 |
| abt_buy | left/2 | 2,162 | 1,012 | 0.990 | 0.996 | 0.990 |
| amazon_googleproducts | left/3 | 4,089 | 1,145 | 0.971 | 0.980 | 0.969 |
| dblp_scholar | left/5 | 13,080 | 4,812 | 0.927 | 0.909 | 0.914 |

Embedding cosines sit in the same band as string features on the same candidate
sets and are *higher* on three of five. **Correlation sharing is not what will
break.** (Our string column does not reproduce the paper's numbers exactly —
different candidate sets, reduced attribute allow-list — so read it as an internal
control, not a replication.)

### Innovation 2's failure mode is what bites

Standardised separation between the M and U means, per attribute:

| dataset | feature | μ_M | σ_M | μ_U | σ_U | sep |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fodors_zagats | `name` | 0.972 | 0.059 | 0.729 | 0.083 | **3.37** |
| fodors_zagats | `addr` | 0.976 | 0.041 | 0.854 | 0.071 | **2.12** |
| fodors_zagats | `phone` | 0.991 | 0.010 | 0.939 | 0.042 | 1.69 |
| fodors_zagats | `city` | 0.986 | 0.041 | 0.939 | 0.084 | 0.72 |
| dblp_acm | `title` | 0.990 | 0.020 | 0.816 | 0.151 | 1.61 |
| dblp_acm | `year` | 1.000 | **0.000** | 0.948 | 0.076 | 0.97 |
| dblp_acm | `authors` | 0.968 | 0.057 | 0.802 | 0.294 | 0.79 |
| dblp_acm | `venue` | 0.911 | 0.055 | 0.959 | 0.037 | **−1.02** |
| dblp_scholar | `title` | 0.972 | 0.039 | 0.802 | 0.083 | **2.62** |
| dblp_scholar | `authors` | 0.912 | 0.191 | 0.714 | 0.264 | 0.86 |
| dblp_scholar | `venue` | 0.745 | 0.358 | 0.687 | 0.379 | 0.16 |
| dblp_scholar | `year` | 0.506 | 0.482 | 0.515 | 0.435 | −0.02 |
| abt_buy | `name` | 0.908 | 0.047 | 0.837 | 0.077 | 1.12 |
| abt_buy | `description` | 0.441 | 0.365 | 0.452 | 0.348 | −0.03 |
| abt_buy | `price` | 0.200 | 0.383 | 0.266 | 0.417 | −0.16 |
| amazon_googleproducts | `title` | 0.878 | 0.076 | 0.788 | 0.111 | 0.94 |
| amazon_googleproducts | `price` | 0.911 | 0.056 | 0.891 | 0.057 | 0.35 |
| amazon_googleproducts | `manufacturer` | 0.072 | 0.250 | 0.047 | 0.204 | 0.11 |
| amazon_googleproducts | `description` | 0.669 | 0.278 | 0.648 | 0.245 | 0.08 |

Three things to act on:

* **Only `name`/`title` carries real signal** on every dataset — one usable column
  per dataset, plus `addr`/`phone` on `fodors_zagats`. **This is the hole the
  sparse-lexical family is aimed at** (§change 4).
* **Negative separation is real.** `dblp_acm venue` at **−1.02**: matches score
  *lower* than non-matches, because at right k=1 the candidate set is already
  2,208/2,294 matches and the non-matches are within-venue confusions. Since
  `getScaledSum` sums all columns assuming higher = match, such a column pushes
  the EM initialisation the wrong way. Screen on the *sign*, not the magnitude.
* **The high-missingness columns are 0-inflated bimodal, not Gaussian.**
  `abt_buy price` sits at σ≈0.38/0.42 around means 0.20/0.27 with 76.5% exact
  zeros. **This is what the discretised-F&S matcher is for** (§change 2) — it is a
  distributional misspecification no covariance structure fixes.

**First observed effect, for calibration of expectations.** `fodors_zagats` at
right k=5: string-only F1 **0.982** → string + 4 cosines **0.941** (114 → 107
edges). A saturated dataset plus two near-zero-separation columns is exactly where
a loss is predicted. Not a verdict on step 3 — and precisely why step 3b matters.

### Missingness is severe and must not be embedded

Absent rows are exact zero rows; the feature is forced to 0 when either side is
absent (the Magellan convention: NaN in either value → NaN feature → 0 after
`fillna`). `embedding_features.py` does this from the vectors' zero rows.

| dataset | side | attribute | missing |
| --- | --- | --- | ---: |
| dblp_scholar | right | `year` | 54% |
| amazon_googleproducts | right | `manufacturer` | 93% |
| abt_buy | left | `price` | 61% |
| abt_buy | right | `price` | 46% |
| abt_buy | right | `description` | 40% |
| dblp_scholar | right | `venue` | 23% |

`drop_zero_variance=True` does **not** catch these: `abt_buy`'s `price_emb_cos` is
0 for 76.5% of pairs, `nunique() > 1` keeps it, and `BlockFeatureCache.get`
re-applies the same weak test per slice.

Two further Phase 0 findings that still bind:

* **Absence comes from the vectors, not `presence.npz`.** That mask is indexed by
  *full*-table rows while `load_embeddings` re-selects rows by id, so on a tuning
  partition it silently mis-indexes (`abt_buy`: 326 vector rows against a
  1,081-row mask). Use `~vec.any(axis=1)`. `presence.npz` stays useful for
  model-independent full-table analysis only.
* **Rank and margin features cannot go through the cache.**
  `BlockFeatureCache.get` serves smaller `top_k` as a row slice, exact only for
  pair-local values. Blocking rank, margin-to-next-neighbour and reciprocal rank
  all depend on the other pairs in the set, so they need their own build per
  `top_k` and must not be added to `embedding_features.py`.

### One metric per attribute, not several

For L2-normalised vectors `‖u−v‖₂² = 2−2cos`, so a Euclidean-derived similarity is
a strictly monotone function of the cosine — Spearman against `cos` is exactly
**+1.0000** — and dot *is* the cosine. `1/(1+‖u−v‖₁)` reaches **+0.9994** while
collapsing into a near-constant range (0.018–0.071 on `abt_buy`'s `description`),
i.e. straight into the singularity condition. Only `‖u−v‖∞` carries distinct rank
information (Spearman 0.86–0.94) and is offered as `linf`. The same measurement
voids `low-resource-deep-er`'s summed element-wise absolute difference as "one
extra column" — summed `|u−v|` *is* the L1. Its actual contribution is
schema-independent dimensionality, which matters for transfer, not here.

### Which blockers

`efficiency_leaderboard()` over the three non-saturated datasets:

| model | mean rank | mean PQ |
| --- | ---: | ---: |
| `tencent_KaLM-Embedding-Gemma3-12B-2511` | 2.00 | 0.334 |
| `codefuse-ai_F2LLM-v2-14B` | 4.33 | 0.334 |
| `codefuse-ai_F2LLM-v2-8B` | 4.67 | 0.331 |

Independently matches `llm-embedder-blocking-benchmark`. Use these three for
blocking. Note the feature-source model need not be the blocking model —
`pretrained-embeddings-er-experimental` finds SBERT wins blocking *and*
unsupervised matching while fine-tuned RoBERTa wins *supervised* matching.

---

## Run order

Steps 0, 0a and the ladder selection are done (§"Status", §"The ladder"), so step 1
is the live front. Nothing below is blocked on data, code or GPU.

| # | step | new code | gate it opens |
| --- | --- | --- | --- |
| ~~0a~~ | ~~**Retain the posterior**~~ | — | **done** 2026-09-07 — `posterior_frame()`, `tests/test_posteriors.py` |
| 1 | **Reference ladder** — string-only ZeroER at the 19 pinned cells | a runner + results view | the curve, and the baseline every delta subtracts from |
| 2 | **Full-width screening** — 20 models × the ladder, no EM | small script | which columns step 3 may use |
| 3 | **The grid** — features × matchers at the screened columns | MEC + F&S + GBDT + TF-IDF builders — **none of the four exist** | the contribution |
| 3b | **Calibration** — intercept/slope on step 1 and 3 posteriors | scoring only — 0a landed | the over-confidence prediction, falsifiable |
| 4 | **Controls and topline** — Zeakis + UMC; B0 reranker | UMC + a reranker harness | "do you even need a generative matcher", and the headroom |

### Step 1 — the reference ladder

Inputs are all in place: the cells are pinned, `outputs/ladder_pairs/` and the
Magellan reference matrices are built, and `BlockFeatureCache` +
`ZeroEREstimator(blocker='precomputed', cache=...)` already does the run. What does
*not* exist is a driver — `hyperparameter-tuning.ipynb` is the blocking notebook
(25 cells, all blocking), so step 1 needs its own runner and a results view.

Per (dataset, cell): string-only pair-P/R/F1, `|C|`, PC, PQ, realised class
balance, **and the oracle F1 ceiling** from §"The ladder".

**Persist `posterior_frame()` per cell as well.** It is free at run time and it is
the whole input to step 3b; skipping it means re-running the ladder later to
recover a number step 1 already had.

Do **not** vary the attribute allow-list or `c_bay` here — upstream defaults
`c_bay=0.1`, `max_iter=40`, `run_trans=False`.

Report a **default column beside the tuned ladder** — one configuration for all
five datasets. Per `end-to-end-deep-er-without-labels`' methodology
(`overview.md` #24), that is the number a practitioner without labels actually
experiences, and the gap between the two columns is itself the result.

Three reasons it still comes first: every later number is a delta against it; the
prevalence coupling means screening must be redone per operating point; and the
curve is a deliverable in its own right.

### Step 2 — full-width feature screening

Extend the measurement above from one model to all 20 at the step-1 cells. Per
(dataset, cell, model, attribute) record separation, per-class sd, zero-inflation
fraction, `cosine(R_M, R_U)`, and post-`c_bay` Bhattacharyya overlap. No EM, so
this is `build_embedding_features` plus gold.

**The gate must be label-free.** Separation magnitude and sign both come from
`matches.csv`, so gating on them makes step 3 "ZeroER, given a gold-label-selected
feature set" — not the method `zeroer-entity-resolution-zero-labels` describes,
and not like-for-like against step 1. The wiki carries the same pattern as a named
failure: Avito's cluster features held *"50% of XGB importance"* in validation and
were a leak (`er-competitions` §9a, the feature graveyard).

**Use the label-free proxy.** Fit a **1-D two-component GMM per candidate feature**
and screen on its *own* fitted separation and post-`c_bay` Bhattacharyya overlap.
It is literally ZeroER's model at `d = 1`, so a feature that fails it is one the
full model cannot use either — and it resolves the sign without gold: the match
component is the one the fit gives the smaller mixing weight. Precedent for
distributional-geometry-as-label-substitute: `distiller-knowledge-distillation-er`
ranks training tuples by blocking similarity; `cross-dataset-transfer-er` selects a
donor by KL over the post-filtering similarity distribution alone.

The measured separation table stays valuable as **ground truth for validating the
proxy**, and the screening study is publishable standalone: no source
characterises embedding-similarity features against ZeroER's two premises, and
this is `zeroer-entity-resolution-zero-labels`' Table 1 measurement extended to
learned features.

### Step 3 — the grid

Cross the three axes. Notes per axis:

**Features.** Per-attribute cosine through the Phase 0 hook; char-n-gram TF-IDF
cosine per attribute as a new builder emitting `{attr}_ngram_cos` (same grouping
rule, no naming trick needed). Both string+emb and emb-only.

**Matchers.**
* *ZeroER* — **sweep `c_bay`**. It is the knob deciding whether concentrated
  cosines dominate the GMM and has never been tuned against this variance profile;
  `dblp_acm year` at σ_M = 0.000 is the singularity condition verbatim.
* *Unsupervised MEC* — greedy descending `r̂(γ)` with endpoint exclusion; iterate
  θ, ξ, `n_M` to the fixed point. Report FLR/MMR beside pair-F1.
* *Discretised F&S / WJ-EM* — bin each per-attribute similarity into ordinal
  categories with **missing as its own level**. This is the arm that keeps
  `abt_buy price`, `amazon_googleproducts manufacturer` and `dblp_scholar year`
  instead of discarding them.
* *GBDT* — label-free supervision only; name the route
  (`knowledge-distillation-er` teacher labels, or `cross-dataset-transfer-er`
  donor). The tournament reading is pre-registered: which family wins is a
  statement about the dataset, not a horse race.

**Reporting.** Spread across the 20 models, not the best, so
`ensemblelink-record-linkage`'s "embedder choice moves 1–2 points" null is
detectable rather than hidden by a max. `er-in-practice-self-serve`'s matcher
tournament makes a mixed result a **diagnostic, not a wash** — pre-register that
reading.

### Step 3b — calibration

Intercept and slope on the posteriors computed in steps 1 and 3, plus the flexible
curve and Net Benefit where cheap. Slope `b < 1` means predictions are too extreme;
intercept `a < 0` means systematic overestimation — which is exactly the
singleton-group double-counting prediction, made falsifiable. It registers a result
even where F1 does not move. Four wiki pages record that no DL ER paper reports this.

**The inputs now exist** (step 0a, 2026-09-07): read them off
`ZeroEREstimator.posterior_frame()`, which returns every candidate pair with its
continuous `p_match`. Step 1 must **persist that frame per cell**, not just the
pair-F1 row — otherwise 3b re-runs the whole ladder to recover what step 1 already
computed. Join to gold through `data._ids_mapping_1/2`, the mapping `evaluate` uses.

### Step 4 — controls and topline

* **Zeakis baseline** — `sim = 1/(1+dist)` + `unique-mapping-clustering` over a
  swept threshold, on vectors already cached. This is the *"do you even need a
  generative matcher"* control. Without it no claim about the generative model
  survives review.
* **B0 reranker topline** — zero-shot cross-encoder over the candidate sets we
  already have. `comprehensive-survey-dl-er` §9.2 names bi-encoder-retrieve →
  cross-encoder-rerank as the recommended architecture;
  `ensemblelink-record-linkage` reaches 0.90–0.99 top-1 with no training.

  Run it on **the ladder** (§"The ladder"), not on a bespoke cell set — an earlier
  revision listed cells that are no longer rungs (`amazon` left/3, `dblp_acm`
  right/1). Cost, summing `|C|` over all 19 cells:

  | dataset | cells | pairs |
  | --- | ---: | ---: |
  | fodors_zagats | 3 | 2,188 |
  | dblp_acm | 4 | 46,846 |
  | abt_buy | 4 | 22,756 |
  | amazon_googleproducts | 4 | 26,671 |
  | dblp_scholar | 4 | 106,119 |
  | **total** | **19** | **204,580** |

  At EnsembleLink's measured 748 pairs/sec for a 278M reranker on a 4090 that is
  **~4.6 minutes for the whole ladder** — inference only; the harness (model pull,
  batching, threshold sweep, pair-F1 scoring) is not written. If that is still too
  much, the top rung alone is 8,007 pairs (~11 s).

  **Do not use a rerank score as a matcher feature** — a single pair-level scalar
  lands in a singleton group and double-counts every attribute block. Standalone
  or topline only. Report pair-F1 at a swept threshold, never top-1.

  **The upgrade worth taking:** run it as a **two-tier escalation policy** rather
  than a topline — ZeroER's posterior for the confident band, rerank only on the
  uncertain band (e.g. `P_M ∈ [0.2, 0.8]`), reported as accuracy against
  escalation fraction. `overview.md` names this as the tractable reduction of
  `agentic-er-vision`: *"a two-tier escalation policy … has never been evaluated as
  a policy with a stated cost model. That is a thesis-sized experiment."* It does
  not violate the objection above — the score is never a ZeroER feature, only a
  second-tier decision.

### Where to draw conclusions

Report all five datasets, conclude from **`abt_buy`, `amazon_googleproducts`,
`dblp_scholar`** only. `dblp_acm`'s top rung is 2,210 matches in 2,616 pairs (PQ
0.845) and its bottom rung still has PC 1.000 — the class balance alone makes the
GMM's job trivial, and `fodors_zagats` is perfectly recalled at every rung.
`reporting.py:500` already encodes
`SATURATED_DATASETS = ("dblp_acm", "fodors_zagats")`.

---

## Cross-cutting

### The prevalence coupling (governs every axis)

**The blocker's PQ *is* the class balance the matcher trains on, and the clean
sweep moves it 12.5×** (at k=1, PQ spans 0.077 → 0.964; `dblp_acm` right k=1 →
0.964, `dblp_scholar` right k=1 → 0.081, and within `dblp_scholar` alone left k=5
→ 0.369 vs right k=1 → 0.081). ZeroER is a GMM fitted on that distribution, so
blocking config and matcher config are **not separable**.

The screening table is a second instance: `dblp_acm`'s `venue` separation is
negative *because* right k=1 delivers a 96%-match candidate set. Feature screening
therefore has to be redone per operating point, not once per dataset.

### The operating points are chosen — and which regime that is

**Settled: the PQ ladder in §"The ladder".** The sweep stores `tp` /
`candset_size` / `n_gold` for every (model, k, side) cell, so PC and PQ were known
before any matcher ran. **Do not re-search the blocking axis.** Run every arm on
those same 19 points; that makes "F1 vs blocking PQ" a designed experiment rather
than a scatter. If F1 tracks PQ more than PC, that is the headline: PC is an
oracle-matcher upper bound (`blocking-kpis`), not a prediction.

**But the PQ ladder is a third regime, and reporting one regime is
under-determined.** `sc-block-supervised-contrastive-blocking` §6.3–6.4 runs both
standard regimes on the same nine blockers:

| | Fixed k | Recall-target k |
| --- | --- | --- |
| Protocol | one k everywhere | tune k per dataset until recall ≥ target |
| Reveals | intrinsic quality gaps | the cost of reaching a fixed quality bar |
| Hides | how expensive high recall is | the quality gaps — they migrate into k and \|C\| |

> "By fixing the hyperparameter k differences in recall and precision become
> visible that are not visible if k is tuned."

`blocking-kpis` calls reporting only one "how blocker comparisons get quietly
rigged". Two target conventions coexist and are not interchangeable: recall ≥ 0.90
(Papadakis group, `filtering-verification-framework`) and 99.5% (SC-Block).

**So do both, which is cheap because both read off the stored sweep:**

1. **PQ ladder stays primary.** It is justified by a different argument than
   blocker comparison — PQ *is* the class prior the GMM fits (`class-imbalance-er`)
   — but report PC and the oracle F1 ceiling in every row so movement along the
   curve cannot be misread as matcher behaviour.
2. **Add a PC-matched ladder at recall ≥ 0.90.** That is the regime the literature
   expects for a cross-dataset matcher claim, and it makes the numbers comparable
   to the Papadakis-group line. It also removes the confound directly: equalising
   PC by construction means a cross-dataset matcher comparison is not
   contaminated by differing retrieval headroom.

`filtering-verification-framework` names the confound outright — *"the filter's
configuration is a confound in every matcher comparison"* — with ZeroER's 70.93% →
0.00% as the worked case. And per `blocking-kpis` checklist item 7, name the matcher
consuming each candidate set: PC's oracle assumption guarantees the real pipeline
recovers less.

**The queried side is a hidden parameter, and it moves PC more than model choice
does.** `blocking-kpis`: on `dblp_scholar` (2,616 × 64,263, 5,347 matches — 2.22
gold duplicates per left record against 1.02 per right), querying left at k=1 is
capped at PC ≤ 0.489 and the best of 20 models reaches 0.444; querying right at the
same k reaches 0.970. **That gap exceeds the entire spread between the best and
worst of the 20 models** — which is the answer to "but we chose the best LLM".
Checklist item 8 therefore makes reporting the queried side and the gold cardinality
mandatory: without them a PC is not interpretable on non-1:1 data. §"The ladder"
records the side for every rung.

### Clustering: raw graph primary, UMC beside it

`match_pairs` thresholds at `P_M >= 0.5` off the edge set — no clustering,
`run_trans=False`. Keep it primary for the like-for-like column. But all five
benchmarks are Clean-Clean, where `unique-mapping-clustering` is the Zeakis default
several comparison baselines ran through — and it is the same clustering the Zeakis
control needs, so it costs nothing. `er-in-practice-self-serve` gives the decision
rule: **above ~0.9 pairwise precision transitivity helps; below ~0.5 it destroys
F1** (0.540 → 0.000 and 0.667 → 0.002 on its two largest benchmarks). ZeroER's
`run_trans` is a third option, already exercised by the `fodors_zagats`
replication test — and MEC's bipartite constraint is a fourth, applied *during*
estimation rather than after.

### Matcher-side guardrails worth a separate arm

From `er-in-practice-self-serve` §V, all outside the like-for-like column:

* **Sparsity-aware thresholds.** Bin candidate pairs by number of shared populated
  fields, learn a threshold per bin with monotonicity enforced (`θ_b ≥ θ_{b+1}`):
  **+8.4pp cluster purity** on MusicBrainz, +1.5pp on NCV. **Do not add the count
  as a feature column** — it is a whole-pair scalar and lands in a singleton group.
  Use it to stratify. This is also the architecturally correct home for the
  0-inflated columns: stratifying keeps them as a conditioning variable.
* **Per-field hard vetoes.** `ŷ = 0 if ∃f ∈ V : both_have_f ∧ sim_f < τ_f` —
  **+32.8pp purity on Restaurants**. On the two product datasets this is where the
  identifier signal the embedding features demonstrably miss actually lives.

### Matcher search space (a later arm)

Actual knobs: `c_bay`, `max_iter`, `run_trans`, `LR_dup_free`, `LR_identical`, and
the attribute allow-list / similarity function set feeding `build_zeroer_features`.
Varying the attribute set breaks step 1's like-for-like, so this runs after the
grid. Sampler: **GPSampler** (`auto-configuring-er-pipelines`: the only one
matching grid search across all datasets within 100 trials).

---

## Traps

**Two of five datasets cannot show anything.** `comprehensive-survey-dl-er` §10
flags Fodors-Zagats and DBLP-ACM as near-linearly-separable; the ladder confirms it
— `dblp_acm` holds PC ≥ 0.994 at *every* rung with all 20 models tied, reaching PQ
0.845 at left/1, and `fodors_zagats` is at PC 1.000 at all three rungs. Neither has
a rung where blocking loses anything for a matcher to be measured against.

**Do not compare a top-1 accuracy against a pair-level F1.** This binds B0
directly: take EnsembleLink's reranker but not its metric. Its headline is top-1
under 1:N (precision = recall = F1 by construction) while its own Table 6 reports
89.0 pair-F1 against Ditto's 94.3. `dblp_scholar` has 5,347 gold pairs over 2,616
left records, so argmax-per-query caps recall near 49% while scoring almost
perfectly — the ladder shows exactly this, left/1 at PQ 0.907 with PC 0.444.
**Threshold, and report pair-F1.**

**A negative-separation feature is worse than a useless one.** `getScaledSum` sums
every column to initialise the EM, so `dblp_acm venue` at −1.02 actively degrades
the initialisation. Screening on |separation| alone would keep it.

**Output must not sit beside the whole-record `.npy`s.** `data.py:embedding_models()`
globs `*_left.npy` in the record directory, so `<model>_title_left.npy` there would
register as a model named `<model>_title` and pollute the blocking sweep. The two
trees are separate siblings; neither glob sees the other.

---

## Still open, not folded in

Held separately so they can be adopted item by item, or not at all.

* **Meta-blocking vocabulary for rank features.** `meta-blocking` defines pruning ×
  weighting (WEP / CEP / WNP / CNP × ARCS / CBS / ECBS / JS / EJS / χ²), and its own
  framing is that *"an ANN index plus a top-`k` cutoff is structurally CNP over a
  similarity-weighted graph."* Rank / margin / reciprocal-rank features are the dense
  re-invention of Reciprocal CNP; naming them against that family costs nothing and
  supplies the discounting variants (ECBS, EJS) we do not have. Needs a per-`top_k`
  build; must not go through the cache.
* **Cross-attribute cosines.** `er-competitions` §4a records cross-field matching
  (title₁ vs description₂) among Avito's 587 features. The `attribute/` tree makes
  `title_left × description_right` free. Such a column opens its own group, which
  here is the *correct* choice — it genuinely is a different attribute pair, not a
  whole-record summary. Natural attack on `abt_buy`, where `name` (sep 1.12) and
  `description` (sep −0.03) carry wildly asymmetric information.
* **Predicted-attribute imputation.** `moretti-shlomo-prediction-models` Strategy C
  predicts an attribute observed on only one file and adds the prediction as an extra
  matching variable — cutting downstream regression bias 60× (ARB 0.601 → 0.010).
  Exactly the setting of `amazon_googleproducts`'s 93%-missing `manufacturer`. If
  tried, use the **continuous** predicted similarity, not the decile — unless it goes
  into the discretised-F&S arm, where the decile is native.
* **Bayesian FS on embedding similarities.** `overview.md`'s unclaimed cross-thread
  question. Strictly more work than MEC; same features.
* **Replicate the transitivity precision threshold.** `transitivity-constraint-er`
  states "above 0.9 safe, below 0.5 mega-clusters" from single-run point estimates,
  one seed, six benchmarks, no variance — and says verbatim *"It is cheap to
  replicate on the ZeroER suite."* We additionally have `run_trans` and MEC's
  bipartite constraint as third and fourth treatments never compared on a common
  benchmark. Report baseline pairwise precision, which the page notes the standard
  pipeline never does. Costs almost nothing beyond step 1.
* **INB / `min2`.** `er-competitions` §6a gives ~15 lines of Iterative Neighborhood
  Blending; both that page and `overview.md` flag that soft transitivity — applied to
  the *representation* rather than the *decision* — has never been evaluated on a
  standard ER benchmark. `min2` is a one-line cardinality prior worth +0.019 F1 at
  Shopee, the cheapest thing in the wiki.
* **Blocking-side leftovers.** Say that exact faiss search is a strength
  (`er-in-practice-self-serve` shows HNSW's `M` costs real recall at scale); union
  dense ∪ sparse retrievers rather than only ranking them (`abt_buy` at k=2 is where
  it shows); ΔPC on `amazon_googleproducts`, where `overview.md` carries a signed
  prediction that dense blocking should be *worse* for minority groups; `F_max` /
  `k_F` is near-degenerate (RR 0.997–0.9999 everywhere).

## Wiki gaps worth filing

* **ZeroER's Magellan feature-*generation* path** has no page.
* **How a GMM behaves when fed correlated dense-similarity features.**
  `class-imbalance-er` handles the prior/calibration side but not the covariance
  side. The screening tables above are the start of that page, and the claim is
  sharper than either half of `overview.md`'s open question: `cosine(R_M, R_U)`
  holds for learned features while the variance/separation profile does not.

Four *named* `overview.md` open questions land on this design:

| Open question | What answers it |
| --- | --- |
| "does the generative model still hold up when features are learned embeddings rather than engineered similarity vectors?" | the grid + the Zeakis control |
| "Can maximum-entropy-record-linkage's FLR/MMR semantics be ported to the embedding-based DB-ER thread?" | the MEC arm |
| "Is *any* DL ER matcher's output calibrated?" | step 3b |
| "Where is the *pipeline-level* headroom, concretely?" — flagged as the most actionable gap in the corpus | the full ladder × features × matcher × clustering ablation |


notes:
 The immediate obstacle

  nn_search(v1, v2, ...) indexes left and queries right, and both callers — _precomputed_nn_blocks:78 and
  BlockFeatureCache._built:125 — hardcode that orientation. There is no side parameter anywhere in the
  blocking path. The ladder is 11 left-queried rungs out of 19:

  ┌───────────────────────┬─────────────────────────┐
  │ dataset               │ left-queried rungs      │
  ├───────────────────────┼─────────────────────────┤
  │ fodors_zagats         │ left/1                  │
  ├───────────────────────┼─────────────────────────┤
  │ dblp_acm              │ left/2, left/1          │
  ├───────────────────────┼─────────────────────────┤
  │ abt_buy               │ left/13, left/2, left/1 │
  ├───────────────────────┼─────────────────────────┤
  │ amazon_googleproducts │ left/2, left/1          │
  ├───────────────────────┼─────────────────────────┤
  │ dblp_scholar          │ left/10, left/5, left/1 │
  └───────────────────────┴─────────────────────────┘

  So the plan's "Zero new code: BlockFeatureCache + ZeroEREstimator(blocker='precomputed', cache=...)
  already does it" is wrong — as written that path can only serve 8 of the 19 cells.

  The good news is that the fix is probably not to add a side parameter. The pairs for all 19 cells are
  already materialised in outputs/ladder_pairs/, so step 1 needs no k-NN at all: load pairs → build
  features → match_pairs → score. That also sidesteps the plan's worry about k_max driving the amazon
  feature cost, since there's no k_max in that path.

  The decision that gates it

  Which matrix is step 1's baseline, because the cost differs by hours:

  • Stored Magellan features (datasets/magellan-features/) — already built for all 19 cells, genuine
  py_entitymatching. Step 1 becomes essentially free.
  • Our port recomputed — the matrix later arms extend with {attr}_emb_cos. Extrapolating from measured
  timings at n_jobs=-1: fodors ~3s, dblp_acm ~3min, abt_buy ~12min, dblp_scholar ~13min, amazon ~4h (0.53
  s/pair × 26,671 pairs). Amazon is ~90% of the total.

  Parity is now proven to 1 ULP on four of five datasets, so the two are interchangeable except
  dblp_scholar.year_year_lev_sim — which the screening table measures at separation −0.02, i.e. no signal
  either way.

  There's also a sub-decision buried here: the stored matrices use all shared attributes (73 columns on
  fodors, including type and class), whereas DATASETS[ds]["attributes"] is a reduced allow-list. The plan
  says "do not vary the attribute allow-list" in step 1 but never says which one is the baseline. That
  choice propagates through every later delta.

  After step 1

  Unchanged from the plan, and all of it still unwritten:

  • Step 2 — 20 models × ladder, no EM. The gate must be the label-free 1-D two-component GMM proxy, not
  measured separation; gating on gold makes step 3 "ZeroER given a gold-selected feature set" and breaks
  like-for-like against step 1.
  • Step 3 — the grid. Four builders that exist nowhere in the code: unsupervised MEC, discretised
  F&S/WJ-EM, GBDT, char-n-gram TF-IDF. Plus a c_bay sweep.
  • Step 3b — calibration. Now genuinely unblocked by 0a, but step 1 must persist posterior_frame() per
  cell or 3b re-runs the ladder.
  • Step 4 — Zeakis + UMC control, reranker harness. UMC also doesn't exist.

  The cheapest addition remains the PC-matched ladder at recall ≥ 0.90 — reads off the stored sweep, no
  new blocking runs, and makes the numbers comparable to the line the work will be judged against.


clarification:
 The immediate obstacle

  nn_search(v1, v2, ...) indexes left and queries right, and both callers — _precomputed_nn_blocks:78 and
  BlockFeatureCache._built:125 — hardcode that orientation. There is no side parameter anywhere in the
  blocking path. The ladder is 11 left-queried rungs out of 19:

  ┌───────────────────────┬─────────────────────────┐
  │ dataset               │ left-queried rungs      │
  ├───────────────────────┼─────────────────────────┤
  │ fodors_zagats         │ left/1                  │
  ├───────────────────────┼─────────────────────────┤
  │ dblp_acm              │ left/2, left/1          │
  ├───────────────────────┼─────────────────────────┤
  │ abt_buy               │ left/13, left/2, left/1 │
  ├───────────────────────┼─────────────────────────┤
  │ amazon_googleproducts │ left/2, left/1          │
  ├───────────────────────┼─────────────────────────┤
  │ dblp_scholar          │ left/10, left/5, left/1 │
  └───────────────────────┴─────────────────────────┘

  So the plan's "Zero new code: BlockFeatureCache + ZeroEREstimator(blocker='precomputed', cache=...)
  already does it" is wrong — as written that path can only serve 8 of the 19 cells.

  The good news is that the fix is probably not to add a side parameter. The pairs for all 19 cells are
  already materialised in outputs/ladder_pairs/, so step 1 needs no k-NN at all: load pairs → build
  features → match_pairs → score. That also sidesteps the plan's worry about k_max driving the amazon
  feature cost, since there's no k_max in that path.

  The decision that gates it

  Which matrix is step 1's baseline, because the cost differs by hours:

  • Stored Magellan features (datasets/magellan-features/) — already built for all 19 cells, genuine
  py_entitymatching. Step 1 becomes essentially free.
  • Our port recomputed — the matrix later arms extend with {attr}_emb_cos. Extrapolating from measured
  timings at n_jobs=-1: fodors ~3s, dblp_acm ~3min, abt_buy ~12min, dblp_scholar ~13min, amazon ~4h (0.53
  s/pair × 26,671 pairs). Amazon is ~90% of the total.

  Parity is now proven to 1 ULP on four of five datasets, so the two are interchangeable except
  dblp_scholar.year_year_lev_sim — which the screening table measures at separation −0.02, i.e. no signal
  either way.

  There's also a sub-decision buried here: the stored matrices use all shared attributes (73 columns on
  fodors, including type and class), whereas DATASETS[ds]["attributes"] is a reduced allow-list. The plan
  says "do not vary the attribute allow-list" in step 1 but never says which one is the baseline. That
  choice propagates through every later delta.

  After step 1

  Unchanged from the plan, and all of it still unwritten:

  • Step 2 — 20 models × ladder, no EM. The gate must be the label-free 1-D two-component GMM proxy, not
  measured separation; gating on gold makes step 3 "ZeroER given a gold-selected feature set" and breaks
  like-for-like against step 1.
  • Step 3 — the grid. Four builders that exist nowhere in the code: unsupervised MEC, discretised
  F&S/WJ-EM, GBDT, char-n-gram TF-IDF. Plus a c_bay sweep.
  • Step 3b — calibration. Now genuinely unblocked by 0a, but step 1 must persist posterior_frame() per
  cell or 3b re-runs the ladder.
  • Step 4 — Zeakis + UMC control, reranker harness. UMC also doesn't exist.

  The cheapest addition remains the PC-matched ladder at recall ≥ 0.90 — reads off the stored sweep, no
  new blocking runs, and makes the numbers comparable to the line the work will be judged against.
