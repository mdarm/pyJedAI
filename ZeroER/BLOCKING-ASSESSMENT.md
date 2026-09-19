# How the blocker was assessed — protocol, KPIs, and the model-selection query

Companion to `NEXT-STEPS-matching.md`. That file takes the blocking axis as
settled and says *do not re-search it*; this one records **why it is settled and
on what evidence**, so the choice of
`tencent_KaLM-Embedding-Gemma3-12B-2511` is defensible without re-running
anything.

Wiki refs are page slugs in `msc-thesis/wiki/wiki/pages/`. Footnotes cite
**source** pages only — per `SCHEMA.md`, concept and analysis pages
(`blocking-kpis`, `llm-embedder-blocking-benchmark`) are syntheses and are
referenced in prose, never as citation targets.

---

## 1. What was measured

One sweep, reused by every protocol below. No blocker is trained or fine-tuned.

| | |
| --- | --- |
| Models | 20 off-the-shelf embedders, 384–5376 dims, 22M–14B params |
| `k` | 1–100 |
| Queried side | both directions, as a swept parameter |
| Datasets | the five ZeroER benchmarks, **full tables, no train/test split** |
| Trials | 4,000 per `<dataset>_blocking` study; 5 studies |
| Storage | `outputs/optuna.db` (the only live studies; the ten inferior `block-filter[_full]` studies were quarantined to `outputs/optuna.db.bak-preaugust`) |
| Per-trial record | `tp`, `candset_size`, `n_gold`, `dim`, `faiss_seconds` as trial user-attrs |
| Sampler | Optuna |

Storing `tp` / `candset_size` / `n_gold` per trial is what makes every protocol
below a *re-read* of one sweep rather than five separate experiments — PC and PQ
are derivable for any cell without touching faiss again. It is also what let the
19-cell matching ladder be pinned without a re-run.

The Optuna choice follows the ER-specific HPO evidence: off-the-shelf samplers
reach grid-search F1 in 30–100 trials at two to three orders of magnitude lower
cost,[^ac-speed] with GPSampler the only one matching or exceeding grid search on
every dataset within 100 trials.[^ac-gp]

**Not measured, and it bounds every efficiency claim.** `faiss_seconds` covers
index build + query only. Encode time is excluded, and it is the dominant term —
the models span 22M to 14B parameters against UniBlocker's 110M MPNet, and
`dblp_scholar` alone is 11 GB of stored vectors. The figure of merit the
literature actually asks for is *end-to-end pipeline runtime at fixed recall with
blocker training time charged in*;[^scb-training-rule] we have neither the encode
half nor the matcher half. So: report `faiss_s`, never rank on it. The query
below is written to enforce exactly that.

## 2. The KPIs, and which way each one lies

Definitions from the canonical survey:[^bfs-pc][^bfs-pqrr]

```
PC = |D(B)| / |D(E)|   pairs completeness ≡ recall      = tp / n_gold
PQ = |D(B)| / ‖B‖      pairs quality      ≡ precision   = tp / candset_size
RR = 1 − ‖B‖ / ‖E‖     reduction ratio
‖E‖ = |E₁| · |E₂|      (Record Linkage — all five benchmarks)
```

Three properties govern how they may be reported:

- **PC is optimistic.** It assumes an oracle matcher, so it is an *upper bound on
  what the pipeline can achieve, never a prediction of it*.[^bfs-oracle] The
  purpose of the assumption is to let a blocker be scored independently of
  whichever matcher follows — which is precisely why a blocker result must name
  the matcher that consumes the candidate set.[^bfs-oracle]
- **PQ is pessimistic** — repeated comparisons between the same duplicate pair
  count as false positives[^bfs-bias] — **and unstable**: it is hypersensitive to
  RR whenever non-matches dominate, which is why the field converged on PC, RR and
  their harmonic mean as the headline triple.[^bbe-pqsensitive] A 1.4-point RR
  difference has moved a published PQ by an order of magnitude (16.06 → 1.52 on
  Amazon-Google).[^bbe-pqexample]
- **There is no optimum to converge on.** The survey defines blocking as
  simultaneous maximisation of all three and then concedes the definition is not
  operational, because it presupposes knowing `D(E)` — "which is actually what ER
  tries to compute".[^bfs-engineering] So any KPI is a joint property of *blocker
  and configuration*, and the reported number is a chosen point on a trade-off
  curve, not a measurement of the blocker alone.

`F_PC,RR` is available[^bbe-fmean] but is not used here: RR sits at 0.997–0.9999
across the whole sweep, so it is near-degenerate and `F_PC,RR` would just mirror
PC.[^bbe-fmirrors]

## 3. Three protocols, because one is under-determined

Reporting a single regime is how blocker comparisons get quietly rigged. The two
standard regimes rank the same blockers differently — "By fixing the
hyperparameter k differences in recall and precision become visible that are not
visible if k is tuned"[^scb-fixedk] — because tuning `k` to a recall target
converts a *quality* difference into a *cost* difference, migrating it into `k`
and `|C|`.[^scb-ktradeoff] Both are run.

| | **A — recall-threshold** | **B — cost-at-recall** | **C — leaderboard (this query)** |
| --- | --- | --- | --- |
| Question | how do we compare to published numbers? | what does the bar cost, per direction? | which model do we ship? |
| Protocol | PC/PQ at smallest `k` reaching PC ≥ 0.90, **querying TableA** | smallest `\|C\|` reaching PC ≥ 0.90, **per side** | smallest `\|C\|` reaching PC ≥ 0.90, **side free**, ranked and aggregated |
| Unit | one row per dataset | one row per (dataset, side) | one row per model |
| Source | UniBlocker[^ub-protocol] | SC-Block[^scb-995] | ours |
| Code | `reporting.uniblocker_comparison` | `reporting.cost_at_recall` | the SQL below / `reporting.efficiency_leaderboard` |

Results for A and B live on `llm-embedder-blocking-benchmark`; a fourth
(NLSHBlock's per-dataset targets, on 3:1:1 test splits) is tabulated there too and
is explicitly *not* commensurable, since its PQ sits behind a different `‖E‖`.

### On `PC_TARGET = 0.90`

The docstring attributes it to UniBlocker, and that is right: its comparison table
is read at a "preset PC threshold (90%)", with `K` the neighbour count needed to
reach it.[^ub-protocol] Worth adding that the same number arrives independently as
a *pipeline contract* from the Papadakis-group line — filtering maximises
precision subject to recall ≥ 0.90, else maximises recall[^e2e-objective] — which
is the stronger justification, because it makes 0.90 the bar the downstream
matcher was designed against rather than a table convention.

It is a choice, not a constant. SC-Block uses **99.5%**[^scb-995] and NLSHBlock
uses per-dataset targets of 89–99%.[^nlsh-targets] The target *is* the experiment;
state it in every row.

## 4. What the query does

Reading top to bottom. Five CTEs, then two SELECTs sharing them.

**`param` — resolve categorical parameters.** Optuna stores `param_value` as a
float. For a `CategoricalDistribution` that float is the *index* into
`$.attributes.choices`, not the value, so `embedding_model` comes back as `7.0`
unless resolved through `distribution_json`. The `CASE` does that; everything else
is cast to INT.

**`trial` — pick the blocking studies.** `state = 'COMPLETE'` drops pruned and
failed trials. The suffix test is `SUBSTR(study_name, -9) = '_blocking'` rather
than `LIKE '%_blocking'` because `_` is a single-character wildcard in `LIKE` —
the `LIKE` form would also match a hypothetical `Xblocking`. Strips the suffix to
recover `dataset`.

**`cell` — one row per (dataset, trial).** A `MAX(CASE WHEN …)` pivot over both
`trial_params` and `trial_user_attributes`. The `CAST(… AS INTEGER)` on the
user-attrs is load-bearing: they are stored as JSON *text*, so an uncast
`ORDER BY` sorts lexicographically and `'100006' < '2162'` — the cheapest
candidate set would come out wrong.

**`grid` — the denominator.** `DISTINCT dataset, model, dim`, i.e. every model
tried on every dataset, independent of whether it ever cleared the bar. This
exists solely so the LEFT JOIN downstream has something to fail against.

**`op` — the operating point.** Filters `tp/n_gold >= target` (that is PC ≥ 0.90),
then takes `ROW_NUMBER() … ORDER BY c ASC, tp/c DESC` and keeps `rn = 1`: **the
cheapest candidate set that reaches the bar, per (dataset, model), with the
queried side chosen freely.** `|C| = k × n_queried` is quantised by integer `k`,
so the cheapest row is rarely unique; PQ breaks that tie so a best-of-many
selection can never be beaten by one of its own members.

**`ranked` — rank on cost, keep the failures.** `grid LEFT JOIN op`, so a model
that never reaches 0.90 survives with `c IS NULL`. `RANK() OVER (… ORDER BY
(o.c IS NULL), o.c ASC)` puts the boolean first, so NULLs sort last — a missing
model ranks **last** rather than vanishing from its own mean. `RANK` (not
`ROW_NUMBER`) so tied costs share a rank.

**`scope` — decide which datasets are allowed to vote.** Counts models tied at
rank 1; if more than half of them tie, the dataset `discriminates = 0`. Every
model reaching the bar at the same `k` means the dataset carries no ordering
information, and including it would add a near-constant to every model's mean rank
while diluting the datasets that do separate.

Then:

- **SELECT 1** prints the scope decision — `n_models`, `tied_at_rank1`,
  `pct_tied`, and the verdict — so the exclusion is auditable rather than
  asserted.
- **SELECT 2** is the leaderboard: `mean_rank` / `worst_rank` / `wins` computed
  **only over discriminating datasets** via `CASE WHEN sc.discriminates`, plus
  `mean_PQ`, `mean_C`, `misses`, `dim` and `faiss_s`. Ordered by mean rank, then
  worst rank, then mean PQ.

### Three things it gets right, worth keeping

1. **The saturation exclusion is derived, not declared.** `reporting.py:500`
   hardcodes `SATURATED_DATASETS = ("dblp_acm", "fodors_zagats")`. The query
   instead measures tie density and lets the data name them — and it reproduces
   the constant exactly (100% and 90% tied against ≤20% elsewhere). That converts
   a magic constant into a result, and it independently corroborates the
   published benchmark-difficulty critique: standard benchmarks are
   near-linearly-separable, Fodors-Zagat named
   explicitly.[^survey-benchmarks]
2. **Failures rank last instead of disappearing.** `reporting.efficiency_leaderboard`
   drops any model missing the target on any dataset
   (`groupby("model").filter(...)`) — it returns 19 rows, not 20. Dropping is
   survivorship bias: a model that fails the hardest dataset should be penalised,
   not excused. The LEFT JOIN is the better design.
3. **`faiss_s` and `dim` are shown but never sorted on**, with the reason stated
   inline. Correct — an efficiency ranking that omits encode time and index size
   would be an unsupported claim.[^scb-training-rule]

### Four things to fix or report

**(a) `mean_PQ` and `mean_C` re-introduce the bias the LEFT JOIN removed.**
They are plain `AVG(r.pq)` / `AVG(r.c)` with no `discriminates` guard, and
SQLite's `AVG` skips NULLs. So a model with `misses > 0` gets those two columns
averaged over *fewer datasets* than its neighbours, while `mean_rank` beside it is
computed over all three. Live case: `nvidia_llama-embed-nemotron-8b` misses
`abt_buy`, so its `mean_C = 18,426` is a mean over four datasets against everyone
else's five. Either guard both columns with `discriminates` (matching `mean_rank`)
or add a count column; do not leave them mixed.

**(b) `mean_C` averages across incommensurable comparison spaces.** `‖E‖` ranges
from 176,423 (`fodors_zagats`) to 168,112,008 (`dblp_scholar`) — a 950× spread —
and the per-dataset `|C|` at the bar ranges 331 to 64,263. A mean over those is
not a quantity. The reporting rule that an RR without its `‖E‖` is unfalsifiable
applies here in the same form. Rank-based aggregation is the right call precisely
because it is scale-free; `mean_C` should be read as a rough tie-breaker or split
per dataset.

**(c) The queried side is chosen freely and then dropped from the output.**
This is the one that matters. `op` lets each model pick whichever direction is
cheapest, and the leaderboard's SELECT list carries no `side` column. On a 1:N
gold standard the queried side caps PC by construction, and our own measurement on
`dblp_scholar` (2,616 × 64,263; 2.22 gold duplicates per left record against 1.02
per right) puts querying left at `k=1` under a hard `PC ≤ 0.489` ceiling while
right reaches 0.970 — **a gap larger than the entire spread between the best and
worst of the 20 models**. And it is not hypothetical here: at the 0.90 bar,
`abt_buy` splits **11 models on left against 8 on right**, so a third of that
dataset's ranking compares models across structurally different retrieval
problems at near-equal cost (left `k=2` → 2,162 pairs; right `k=2` → 2,184).
Free-side selection is a defensible reading of cost-at-recall — the bar is the
bar, however you reach it — but the side must appear in the output, per the
reporting checklist's requirement to name the indexed and queried collections
along with duplicates-per-record on each side.

**(d) The SQL and Python leaderboards are different statistics with the same
name.** Both are live, they disagree, and nothing says so. Verified against
`outputs/optuna.db` at `target = 0.90`:

| | SQL | `reporting.efficiency_leaderboard` |
| --- | --- | --- |
| Rank ordering | `c` only | `(c, −PQ)` tuples, `rank(method="min")` |
| Models that miss | ranked last, `misses` reported | dropped (19 rows, not 20) |
| `mean_PQ` / `mean_C` over | all 5 datasets | the 3 discriminating ones |
| KaLM-Gemma3-12B | **1.67**, `mean_PQ` 0.4608 | **2.00**, `mean_PQ` 0.3344 |
| F2LLM-v2-14B / -8B | 1.67 / 1.67 (tied with KaLM) | 4.33 / 4.67 |
| F2LLM-v2-4B | 2.67 (4th) | 6.33 (6th) |
| e5_instruct | 4.33 (7th) | 4.67 (4th) |

The rank difference is the substantive one. SQL ties every model sharing a `|C|`;
Python separates them on PQ. At fixed `|C|`, `PQ = tp/|C|`, so ranking on PQ there
is ranking on *recall above the bar* — a quality axis that the recall-target regime
has deliberately converted into cost.[^scb-ktradeoff] The SQL is the more faithful
implementation of the protocol; Python's rewards overshoot. Pick one, and note in
the writeup that the top three are a genuine tie under the strict reading.

## 5. The result, at `PC_TARGET = 0.90`

Scope — which datasets are permitted to rank:

| dataset | models | tied at rank 1 | % tied | role |
| --- | ---: | ---: | ---: | --- |
| abt_buy | 20 | 4 | 20.0 | ranks |
| amazon_googleproducts | 20 | 1 | 5.0 | ranks |
| dblp_scholar | 20 | 1 | 5.0 | ranks |
| dblp_acm | 20 | 20 | 100.0 | saturated — excluded |
| fodors_zagats | 20 | 18 | 90.0 | saturated — excluded |

Leaderboard, top of the table (mean rank over the three discriminating datasets):

| model | mean rank | worst | wins | mean PQ | misses | dim | faiss_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tencent_KaLM-Embedding-Gemma3-12B-2511` | 1.67 | 2 | 1 | 0.4608 | 0 | 3840 | 10.8 |
| `codefuse-ai_F2LLM-v2-14B` | 1.67 | 2 | 1 | 0.4595 | 0 | 5120 | 14.6 |
| `codefuse-ai_F2LLM-v2-8B` | 1.67 | 2 | 1 | 0.4584 | 0 | 4096 | 11.6 |
| `codefuse-ai_F2LLM-v2-4B` | 2.67 | 5 | 1 | 0.4714 | 0 | 2560 | 7.0 |
| `Qwen_Qwen3-Embedding-4B` | 3.00 | 5 | 0 | 0.4572 | 0 | 2560 | 7.2 |
| … | | | | | | | |
| `nvidia_llama-embed-nemotron-8b` | 16.33 | 20 | 0 | 0.3873 | **1** | 4096 | 11.8 |
| `distilroberta` | 18.33 | 20 | 0 | 0.2814 | 0 | 768 | 2.4 |

**How to read it.** The top three are tied, not ordered — `1.67` is the same
`{1,2,2}` rank vector for all of them. KaLM's lead over F2LLM-14B is 0.0013 mean
PQ, which is a third-order tie-break on an unstable
metric,[^bbe-pqsensitive] not a result. The honest statement is *a leading group of
three-to-four models, indistinguishable at this target*, and the effect worth
reporting is the **bottom** of the table: the spread from 0.46 to 0.28 mean PQ
across model families is real, and `distilroberta` / `mpnet` / `minilm6` cost 3–9×
more candidate pairs to clear the same bar.

That reading also matches the external evidence. Swapping embedders moves
end-to-end accuracy by 1–2 points where swapping rerankers moves it by up to
53,[^el-ablation] so a near-tie at the top is the expected outcome, and a sweep
that reported only its winner would be overclaiming. Note too that the winner is
protocol-dependent: the model with the single best *cost* here is not the model
protocol A selects as the uniform one, and `codefuse-ai_F2LLM-v2-4B` has the
highest mean PQ of all 20 while ranking 4th on cost.

**What this does not establish.** Nothing about efficiency in the sense the
literature means — encode time is unmeasured, and the leading group runs 2.5–14B
parameters against the 22M–110M models at the bottom. On a
runtime-at-fixed-recall accounting with training and encoding charged
in,[^scb-training-rule] the ranking could invert entirely. State the KPI as *cost
in candidate pairs at PC ≥ 0.90*, which is what was actually measured.

## 6. Reporting checklist for any blocking number that leaves this repo

Condensed from `blocking-kpis`, in the order it usually gets violated:

1. PC and RR, with `‖E‖` stated. `|C|` in absolute terms.
2. The regime — fixed `k` (give `k`) or recall-target (give the target).
3. PQ only alongside `‖E‖`, with the sensitivity caveat.
4. The **queried side**, plus duplicates-per-record on each side. Without it a PC
   is not interpretable on non-1:1 data. ← currently missing from the leaderboard
5. The **matcher** that consumes the candidate set — PC's oracle assumption
   guarantees the real pipeline recovers less.[^bfs-oracle]
6. Runtime as end-to-end pipeline time at fixed recall, including blocker training
   — or no efficiency claim at all.[^scb-training-rule]
7. No borrowed AP/mAP. UniBlocker's mAP does not reproduce from its own stated
   formula (its `fodors-zagats` row implies 21.01 against a printed 60.51), so PC
   and PQ at a stated `k` are the only cross-paper-safe surface.
8. No cross-family metric comparison — a top-1 / recall@k figure never goes
   against a pair-level F1.

---

## Footnotes

[^bfs-pc]: [[blocking-filtering-survey]] §2 — "(1) Pair Completeness (PC) corresponds to recall, estimating the portion of the detectable duplicates in B with respect to those in E: PC(B) = |D(B)|/|D(E)| ∈ [0, 1]."
[^bfs-pqrr]: [[blocking-filtering-survey]] §2 — "(2) Pairs Quality (PQ) corresponds to precision… PQ(B) = |D(B)|/‖B‖ ∈ [0, 1]. (3) Reduction Ratio (RR) measures the reduction in the number of pairwise comparisons in B with respect to the brute-force approach: RR(B, E) = 1 − ‖B‖/‖E‖ ∈ [0, 1]."
[^bfs-oracle]: [[blocking-filtering-survey]] §2 — "A common assumption in the literature is the oracle, i.e., a perfect matching function that, for each pair of entity profiles, decides correctly whether they match or not. Using an oracle, a pair of duplicates is detected as long as they share at least one block. This allows for reasoning about the performance of blocking methods independently of matching methods".
[^bfs-bias]: [[blocking-filtering-survey]] §2 — "Note that PC provides an optimistic estimation of recall, presuming the existence of an oracle, while PQ provides a pessimistic estimation of precision, treating as false positives the repeated comparisons between duplicates (i.e., only the non-repeated duplicate pairs are considered as true positives)."
[^bfs-engineering]: [[blocking-filtering-survey]] §2 — "Conceptually, Blocking can be viewed as an optimization task, but this implies that the real duplicate collection D(E) is known, which is actually what ER tries to compute. Hence, Blocking is typically treated as an engineering task that provides an approximate solution for the data at hand."
[^bbe-fmean]: [[blocking-biases-evaluation]] §III eq. 2 — "The harmonic mean, which balances the trade-off between PC and RR, is defined as: F_PC,RR = 2 × PC × RR / (PC + RR)."
[^bbe-pqsensitive]: [[blocking-biases-evaluation]] §III — "Among these metrics, PQ is typically sensitive to small changes in RR… Therefore, in most studies, PC, RR, and their harmonic mean are used as the key metrics for evaluating blocking methods."
[^bbe-pqexample]: [[blocking-biases-evaluation]] §V-B2 — "in the AMZ-GOO dataset, Suffix achieves a PQ of 16.06, while AUTO, with a slightly lower RR (99.85 for Suffix versus 98.45 for AUTO), sees a drastic drop in PQ to 1.52."
[^bbe-fmirrors]: [[blocking-biases-evaluation]] §V-B3 — "F_PC,RR mainly mirrors PC, also in their biases. This, again, can be explained by the fact that RR is often close to 1, and its biases are insignificant."
[^ub-protocol]: [[uniblocker-universal-dense-blocking]] §5.2 / Table 3 caption — "if both methods exceed the preset PC threshold (90%), we evaluate their PQ, otherwise, we compare their PC. K is the number of nearest neighbors needed to reach the reported PC".
[^e2e-objective]: [[end-to-end-deep-er-without-labels]] §4 — "maximize precision for recall at least 0.90; otherwise maximize recall."
[^scb-fixedk]: [[sc-block-supervised-contrastive-blocking]] §6.3 — "We use recall also referred to as pair completeness and precision also referred to as pair quality to evaluate the candidate sets with respect to the test sets of the datasets. By fixing the hyperparameter k differences in recall and precision become visible that are not visible if k is tuned."
[^scb-ktradeoff]: [[sc-block-supervised-contrastive-blocking]] §6.4 — "Increasing the k of the nearest neighbour blockers increases the probability of finding a matching pair resulting in a higher recall. Higher values of k produce larger candidate sets because the matcher has to compare more candidate pairs. This prolongs the matching phase of the entity resolution pipeline."
[^scb-995]: [[sc-block-supervised-contrastive-blocking]] Abstract — "For measuring the overall runtime, we determine the candidate sets that need to be passed to the matcher for maintaining 99.5% recall."
[^scb-training-rule]: [[sc-block-supervised-contrastive-blocking]] §7.2 — "Training of a blocker is only reasonable if the runtime of the pipeline including training time is shorter than the runtime of a pipeline with a blocker that is not trained."
[^nlsh-targets]: [[neural-lsh-block]] §4.2 — "we set a target recall and compare accuracy and size of candidate pairs. We set the target recalls of the five datasets as 89%, 97%, 99%, 97%, and 94% resepectively for AB, AG, DA, DS, and WA. These target recalls are selected from DL-Block [43], which represent the best performance in its framework for each dataset."
[^survey-benchmarks]: [[comprehensive-survey-dl-er]] §10 — "A recent analysis [68] indicates that existing benchmarks may pose rather easy classification tasks. Theoretical measures of linearity suggest that many of these datasets (e.g., Fodors-Zagat) involve almost linearly separable candidate pairs … we categorize the datasets … according to their determinism criterion."
[^ac-speed]: [[auto-configuring-er-pipelines]] §5.2 RQ3 — "the run-time is reduced by 2 or even 3 orders of magnitude. … the value is well below 0.35%. This means sampling-based search is consistently faster than grid search by more than 285 times."
[^ac-gp]: [[auto-configuring-er-pipelines]] §5.2 [synthesis] — of RandomSampler / TPESampler / QMCSampler / GPSampler, GPSampler is the only one to match or exceed grid-search F1 across all datasets within 100 trials, and has the highest mean F1 per trial.
[^el-ablation]: [[ensemblelink-record-linkage]] §5.1 / Table 3 — "The choice of reranker has substantial impact: Jina v2 Multilingual and Jina v3 achieve 84–90% accuracy on all tasks, while the English-only MiniLM reranker fails catastrophically on multilingual party matching (31% vs 84%)… Embedding model choice has minimal impact—accuracy varies by only 1–2 points across embeddings with the same reranker." (the 53-point figure is 84 − 31.)
