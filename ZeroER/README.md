# Blocking assessment & next steps — presentation notes

---

# Slide 1 — What was measured, and under whose protocol

**One sweep, reused by every protocol.** 20 off-the-shelf embedding models ×
k ∈ [1,100] × both query directions × 5 ZeroER benchmarks. Full tables, no
train/test split, 4,000 Optuna trials per dataset. *No blocker trained or
fine-tuned.*

Per trial we store `tp`, `|C|`, `n_gold`, `dim`, `faiss_s` → every KPI (Key
Performance Indicator) is derivable afterwards, so each protocol is a **re-read,
not a re-run**.

**KPIs** (Papadakis, Skoutas, Thanos & Palpanas, *ACM CSUR* 2020, §2):

```
PC = |D(B)| / |D(E)| = tp / n_gold        Pairs Completeness ≡ recall
PQ = |D(B)| / ‖B‖    = tp / |C|           Pairs Quality      ≡ precision
RR = 1 − ‖B‖ / ‖E‖ ,   ‖E‖ = |E₁| · |E₂|  Reduction Ratio (Record Linkage case)
|C| = k × n_queried                        quantised by integer k
```

**Symbols and abbreviations**

| Symbol / term | Meaning |
| --- | --- |
| **ER** | Entity Resolution |
| **PC** | Pairs Completeness — the blocking-stage name for recall |
| **PQ** | Pairs Quality — the blocking-stage name for precision |
| **RR** | Reduction Ratio — fraction of the brute-force comparisons avoided |
| **F_PC,RR** | Harmonic mean of PC and RR |
| **KPI** | Key Performance Indicator |
| `B` | the block collection (the candidate pairs the blocker emits) |
| `E`, `E₁`, `E₂` | the entity collection(s); `E₁` left table, `E₂` right table |
| `D(B)`, `D(E)` | detectable duplicates in `B` / in `E` |
| `‖B‖`, `‖E‖` | comparison counts in `B` / in the full Cartesian product |
| `\|C\|` | candidate-set size (`‖B‖` in absolute terms) |
| `tp` | true positives — gold matching pairs retained by the blocker |
| `n_gold` | number of gold-standard matching pairs |
| `k` | neighbours retrieved per queried record (k-NN = k-Nearest Neighbours) |
| `n_queried` | records on the queried side of the k-NN search |
| `dim` | embedding dimensionality |
| `faiss_s` | seconds spent in FAISS (Facebook AI Similarity Search) index + query |
| **1:N** | a gold standard where one record may match many (not one-to-one) |

**Each number lies in a known direction — state it or do not state the number:**

* **PC is optimistic** — computed under an *oracle matcher* (a hypothetical
  perfect classifier), so it is an upper bound on the pipeline, never a
  prediction of it (ACM CSUR 2020 §2).
* **PQ is pessimistic and unstable** — hypersensitive to RR; a published
  1.4-point RR change moved PQ 16.06 → 1.52 (Moslemi, Balamurugan & Milani 2024,
  §III, §V-B2).
* **RR is 0.997–0.9999 across our sweep** → near-degenerate, so F_PC,RR would
  only mirror PC.

**Why the bar is PC ≥ 0.90 — two independent justifications:**

* UniBlocker's "preset PC threshold (90%)", which puts our numbers beside their
  published table unmodified (Wang, Lin, Han et al. 2024, §5.2).
* The same 0.90 as a *pipeline contract*: "maximize precision for recall at least
  0.90" (Neuhof, Fisichella & Papadakis, DOLAP 2026, §4).

> **The target is the experiment.** SC-Block uses 99.5%; NLSHBlock uses 89–99%
> per dataset. A different bar gives a different ranking.

---

# Slide 2 — How the model is chosen: cost at equal recall

**Selection rule**, per (dataset, model):

```
operating point = argmin |C|  subject to  PC ≥ 0.90
                  (queried side free; integer-k ties broken on PQ)

then rank models on |C| within each dataset, aggregate by mean rank
```

Why rank rather than a mean `|C|`: ‖E‖ spans 176,423 → 168,112,008 across the
five datasets (950×), so an average `|C|` is not a quantity. Rank is scale-free.

**Two design choices worth defending:**

* Models that never reach 0.90 **rank last, not dropped** — dropping them is
  survivorship bias.
* Datasets where >50% of models tie at rank 1 are **excluded as saturated** —
  *derived from the data, not declared*.

**The saturation test reproduces the literature:**

| dataset | tied at rank 1 | role |
| --- | ---: | --- |
| abt_buy | 4/20 (20%) | ranks |
| amazon_googleproducts | 1/20 (5%) | ranks |
| dblp_scholar | 1/20 (5%) | ranks |
| dblp_acm | **20/20 (100%)** | saturated — excluded |
| fodors_zagats | **18/20 (90%)** | saturated — excluded |

→ independently confirms the benchmark-difficulty critique: standard benchmarks
are near-linearly separable, **Fodors-Zagat named explicitly** (Karapiperis,
Tjortjis & Verykios, *ACM CSUR* 2026, §10).

**Leaderboard** (mean rank over the three discriminating datasets):

```
KaLM-Gemma3-12B  1.67   F2LLM-v2-14B  1.67   F2LLM-v2-8B  1.67
F2LLM-v2-4B      2.67   …              …     distilroberta 18.33
```

*(Sampler choice follows Nikoletos, Efthymiou, Papadakis & Stefanidis 2025 —
off-the-shelf HPO (HyperParameter Optimisation) samplers reach grid-search F1 in
30–100 trials at 2–3 orders of magnitude lower cost.)*

### Cross-check — nAP, the whole-sweep criterion

**Abbreviations.** AP = Average Precision · **nAP = normalised Average
Precision** · mAP = mean Average Precision · IR = Information Retrieval.

The rule above reads the sweep at **one bar** (PC ≥ 0.90). A second criterion
reads the **whole curve** instead, and was used to rank models during tuning:

```
AP  = Σ_k (PC_k − PC_{k−1}) · PQ_k        anchored at PC_0 = 0
nAP = AP / min(1, n_gold / n_queried)
```

`AP` is the area under the PC-vs-PQ curve traced by sweeping k ∈ [1,100] — the
standard IR form, and the same quantity UniBlocker reports as mAP. Anchoring at
`PC_0 = 0` matters: integrating only between the swept endpoints would score a
blocker that already reaches PC = 1 at k = 1 as **zero**, because its PC never
moves afterwards.

**Why it must be normalised.** `AP` is capped at `min(1, n_gold/n_queried)`, and
that ceiling is a property of **dataset cardinality × queried side**, not of the
model — it is the same quantity as the k = 1 cap on PQ. On `dblp_scholar` the
ceilings are **1.000 querying left against 0.083 querying right**, so a raw-AP
comparison always prefers left *even though right is the better blocker*
(PC 0.970 vs 0.444 at k = 1). Dividing by the ceiling removes exactly that
artefact and makes the number averageable across datasets and sides.

**The two criteria agree on the winner — but only one is decisive:**

| Criterion | Reads | `KaLM-Gemma3-12B` | Decisiveness |
| --- | --- | --- | --- |
| **nAP** | whole k-sweep, all 5 datasets | mean rank 3.4; **highest mean nAP 0.8961** | weak — ties on rank with `e5_instruct`, which has *more* wins (3 vs 2); decided on the mean-nAP tie-break |
| **Cost at PC ≥ 0.90** | one bar, 3 discriminating datasets | **mean rank 2.00**, clear first | strong — this is the reported criterion |

Two independent criteria selecting the same model is the useful statement; nAP
alone would not have settled it.

**nAP is internal-only, deliberately.** It is never quoted against a published
mAP, because mAP is definition-sensitive: UniBlocker's printed values do not
reproduce from its own stated formula (its `fodors-zagats` row implies 21.01
against a printed 60.51; `dblp-acm` implies 83.65 against 91.88). PC and PQ at a
stated `k`, or cost at a stated recall bar, are the only cross-paper-safe
surfaces — which is why the three reported protocols use those and not this.

---

# Slide 3 — What this establishes, and what it does not

**The honest reading of the result:**

* **The top three are tied, not ordered.** `1.67` is the same `{1,2,2}` rank
  vector for all three. KaLM's lead over F2LLM-14B is 0.0013 mean PQ — a
  third-order tie-break on an unstable metric.
* **The reportable effect is at the bottom of the table:** mean PQ 0.46 → 0.28,
  and `distilroberta` / `mpnet` / `minilm6` need **3–9× more candidate pairs** to
  clear the same bar.
* Consistent with external evidence: swapping embedding models moves end-to-end
  accuracy **1–2 points**, swapping rerankers up to **53** (Dasanaike 2026, §5.1
  / Table 3). A near-tie at the top is the *expected* outcome — reporting only a
  winner would overclaim.

**Not established — efficiency in the sense the literature means.** `faiss_s`
covers index + query only. Encode time is excluded and dominates (22M–14B
parameters; 11 GB of vectors on `dblp_scholar` alone). The required figure of
merit is end-to-end pipeline runtime at fixed recall with blocker training
charged in (Brinkmann, Shraga & Bizer 2023, §7.2). **Our claim is precisely: cost
in candidate pairs at PC ≥ 0.90.**

**One open reporting gap.** The queried side is a free parameter and is currently
dropped from the leaderboard output. On 1:N data it caps PC by construction — on
`dblp_scholar`, querying left at k=1 is capped at **PC ≤ 0.489** while right
reaches **0.970**, *a gap wider than the entire 20-model spread*. At the 0.90
bar, `abt_buy` splits **11 models left / 8 right**.

**Decision this supports:** pin `tencent_KaLM-Embedding-Gemma3-12B-2511` +
cosine for the 19-cell matching ladder. Blocking axis closed; matching is the
live front.

---

# Slide 4 — Next steps: the matching stage, three axes on one feature matrix

**Where we are.** Blocking is closed. The 19-cell PQ ladder is pinned,
materialised and measured on KaLM-Gemma3-12B + cosine; candidate pairs *and*
genuine `py_entitymatching` reference features exist for all 19 cells. Posterior
retention landed. **Step 1 is the live front and is unblocked — no GPU, no new
data, ~0 compute.**

### The design

| Axis | Levels |
| --- | --- |
| **Features** | string-only (reference) · + per-attribute embedding cosine · + character-n-gram TF-IDF · embedding-only |
| **Matcher** | ZeroER GMM · unsupervised MEC · discretised F&S / WJ-EM · GBDT · Zeakis `1/(1+d)` + UMC *(control)* |
| **Operating point** | the 19-cell PQ ladder |

**Abbreviations on this slide**

| Term | Meaning |
| --- | --- |
| **GMM** | Gaussian Mixture Model — ZeroER's two-component match/non-match model |
| **EM** | Expectation-Maximisation — the algorithm that fits it, without labels |
| **MEC** | Maximum Entropy Classification — a threshold-free unsupervised alternative |
| **F&S** | Fellegi–Sunter — the 1969 probabilistic record-linkage framework |
| **WJ-EM** | Winkler–Jaro EM — the standard F&S parameter-estimation procedure |
| **GBDT** | Gradient-Boosted Decision Trees |
| **TF-IDF** | Term Frequency–Inverse Document Frequency (sparse lexical features) |
| **UMC** | Unique Mapping Clustering — greedy one-to-one assignment |
| `Cov_M`, `Cov_U` | ZeroER's covariance matrices for the Match / Unmatch components |
| **FLR / MMR** | False Link Rate / Missing Match Rate — MEC's error semantics |
| **LP** | Linear Predictor — `logit(π̂)`, the model's score on the log-odds scale |
| `π̂` | the model's predicted probability that a pair is a match |

*Why per-attribute is the only admissible feature shape:* ZeroER rebuilds
`Cov_M`/`Cov_U` from groups derived by `name.split("_")[0]`, so a whole-record
scalar opens a **singleton group** and is then *forced* independent of every
attribute block — it double-counts and should be structurally over-confident
(Wu et al., SIGMOD 2020).

### Run order

| # | Step | Status |
| --- | --- | --- |
| 1 | **Reference ladder** — string-only ZeroER at the 19 cells | unblocked; needs a runner only |
| 2 | **Full-width screening** — 20 models × ladder, no EM | label-free gate required |
| 3 | **The grid** — features × matchers | 4 builders do not exist yet |
| 3b | **Calibration** — intercept/slope on step 1 + 3 posteriors | inputs exist |
| 4 | **Controls** — Zeakis + UMC; zero-shot reranker topline | ~4.6 min inference, whole ladder |

Report the ceiling in every row:

```
oracle F1 = 2·PC / (1 + PC)
```

Without it, F1 = 0.60 at `dblp_scholar` left/1 (ceiling **0.615**) and right/1
(ceiling **0.984**) look identical when one is near-perfect and the other poor.

### The headline deliverable is calibration, not F1

Four 2026 sources converge on *"the matcher is no longer where the headroom is."*
The defence is to **move the question from accuracy to probability quality** — no
DL-ER (Deep-Learning Entity Resolution) paper reports calibration, and ZeroER's
posterior is unaudited. Fit on held-out pairs (van den Goorbergh, van Smeden,
Timmerman & Van Calster, *JAMIA* 2022):

```
intercept:  logit(P(Y=1)) = a  + LP          (coefficient of LP fixed at 1)
slope:      logit(P(Y=1)) = a* + b · LP
            LP = logit(π̂)

a < 0  ⇒ systematic overestimation
b < 1  ⇒ predictions too extreme / over-confident
```

This makes the singleton-group prediction **falsifiable**, and registers a result
even where F1 does not move — which the measurements say is the likely case.

### Two constraints that shape every arm

* **Prevalence coupling.** The blocker's PQ *is* the class prior the GMM fits, and
  the ladder moves it **12.5×** (0.077 → 0.964). Blocking and matcher
  configuration are not separable → feature screening must be redone **per
  operating point**, not once per dataset.
* **Same exclusion as Slide 2.** Conclude from `abt_buy` /
  `amazon_googleproducts` / `dblp_scholar` only; `dblp_acm` and `fodors_zagats`
  are saturated on *both* stages.

### Three decisions for review

1. **Scope of the matcher axis** — four new matchers is the bulk of the work.
   **MEC** (Lee, Zhang & Kim 2021) closes a named open question: no threshold, and
   it enforces the bipartite (one-to-one) constraint *during* estimation rather
   than bolting it on afterwards. **Discretised F&S** (Fellegi & Sunter, *JASA*
   1969) is the only arm that can keep the zero-inflated columns ZeroER must
   discard — missing becomes its own category. GBDT + TF-IDF are cheaper but less
   novel. **Cut to two?**
2. **The screening gate must be label-free** — a 1-D two-component GMM proxy,
   *not* measured separation, or step 3 becomes "ZeroER given a
   gold-label-selected feature set" and breaks like-for-like against step 1.
   Confirm.
3. **Add the PC-matched ladder at recall ≥ 0.90?** Cheapest item on the list —
   reads straight off the stored sweep, no new runs — and it is the regime the
   work will be judged against (Neuhof, Fisichella & Papadakis, DOLAP 2026, whose
   worked case is ZeroER at **70.93% → 0.00%** under a filter change alone).

**The "so what" line if novelty is questioned:** *nobody has run a controlled
ablation attributing end-to-end F1 across the three stages on a common
benchmark.* The ladder × features × matcher crossing is exactly that, and the
blocking work is already half of it.

---

# Slide 5 (backup) — Why one metric per attribute: cosine and Euclidean are the same feature

**The question this answers.** The feature axis offers *one* similarity per
attribute per model. The obvious objection is "why not add Euclidean, dot product
and L1 as extra columns — they are free." The answer is that on L2-normalised
vectors the Euclidean distance is a deterministic, strictly decreasing function
of the cosine, so such a column adds no information and actively harms the
covariance estimate.

**The hypothesis holds here, and was checked.** Every stored vector is
L2-normalised: measured row norms lie in [0.99999988, 1.00000012] across the
embedding tree, i.e. unit to within float32 precision. Absent attribute values
are stored as exact zero rows and are excluded by the presence mask before any
metric is applied, so the proposition below is applied only where it is valid.

```latex
% requires \usepackage{amsmath, amssymb}

\textbf{Proposition.}
Let $u,v \in \mathbb{S}^{d-1} = \{x \in \mathbb{R}^{d} : \|x\|_{2} = 1\}$. Then
\[
  \|u-v\|_{2}^{2} \;=\; 2 - 2\cos(u,v),
  \qquad\text{where}\qquad
  \cos(u,v) \;:=\; \frac{\langle u,v\rangle}{\|u\|_{2}\,\|v\|_{2}} .
\]

\textbf{Proof.}
The Euclidean norm is induced by the inner product, so expanding bilinearly,
\[
  \|u-v\|_{2}^{2}
    = \langle u-v,\, u-v\rangle
    = \langle u,u\rangle - 2\langle u,v\rangle + \langle v,v\rangle
    = \|u\|_{2}^{2} - 2\langle u,v\rangle + \|v\|_{2}^{2}.
\]
By hypothesis $\|u\|_{2} = \|v\|_{2} = 1$, so the cosine's denominator is $1$ and
$\cos(u,v) = \langle u,v\rangle$. Substituting,
\[
  \|u-v\|_{2}^{2} \;=\; 1 - 2\cos(u,v) + 1 \;=\; 2 - 2\cos(u,v).
  \qquad\blacksquare
\]

\textbf{Corollary 1 (order equivalence).}
Put $\varphi(c) = \sqrt{2-2c}$, so that $\|u-v\|_{2} = \varphi\!\left(\cos(u,v)\right)$.
On $[-1,1)$,
\[
  \varphi'(c) \;=\; -\,\frac{1}{\sqrt{2-2c}} \;<\; 0,
\]
hence $\varphi$ is strictly decreasing and, for any two pairs on the unit sphere,
\[
  \cos(u,v) \;>\; \cos(u',v')
  \quad\Longleftrightarrow\quad
  \|u-v\|_{2} \;<\; \|u'-v'\|_{2}.
\]
Cosine and Euclidean distance therefore induce the same ordering, the same
$k$-nearest-neighbour sets, and a Spearman rank correlation of exactly $+1$.
The inner product coincides with the cosine outright.

\textbf{Corollary 2 (feature redundancy).}
Let $s = f\!\left(\|u-v\|_{2}\right)$ for any strictly decreasing $f$, for instance
$f(x) = 1/(1+x)$. Then $s = (f \circ \varphi)(\cos)$ is a strictly increasing,
deterministic transform of the cosine. As a column of the ZeroER feature matrix it
carries no rank information beyond \texttt{\{attr\}\_emb\_cos}, while entering the
same within-attribute covariance block at correlation $\approx 1$ --- the
near-singular condition that the adaptive regularisation exists to suppress.

\textbf{Remark (the identity is specific to $p=2$).}
Only $\|\cdot\|_{2}$ is induced by an inner product, so only it can be rewritten
through $\langle u,v\rangle$. For $p \neq 2$ the quantity $\|u-v\|_{p}$ is not a
function of $\langle u,v\rangle$ alone, and no such identity is available.
```

**What the remark buys, measured.** Because the identity fails for `p ≠ 2`, the
other norms had to be tested rather than derived:

| Candidate metric | Spearman vs cosine | Verdict |
| --- | ---: | --- |
| Euclidean `‖u−v‖₂`, dot product | **+1.0000** | excluded **by proof** — identical ordering |
| `1/(1+‖u−v‖₁)` | +0.9994 | excluded by measurement — near-redundant, and collapses to a near-constant range (0.018–0.071 on `abt_buy` `description`), i.e. straight into the singularity condition |
| `‖u−v‖∞` | 0.86–0.94 | **kept** as `linf` — the only norm carrying distinct rank information |

**Conclusion.** `METRICS = ("cos", "linf")`. One metric per attribute is a
consequence of the geometry, not a simplification: adding Euclidean would
duplicate a column at correlation 1 inside a covariance block that ZeroER already
struggles to keep non-singular.

---

# References

**Venue abbreviations.** ACM CSUR = ACM Computing Surveys · SIGMOD = ACM
International Conference on Management of Data · JASA = Journal of the American
Statistical Association · JAMIA = Journal of the American Medical Informatics
Association · DOLAP = International Workshop on Design, Optimization, Languages
and Analytical Processing of Big Data · SDM = SIAM International Conference on
Data Mining · arXiv = open-access preprint server.

**Blocking stage**

* **Papadakis, Skoutas, Thanos & Palpanas** — *Blocking and Filtering Techniques
  for Entity Resolution: A Survey.* ACM CSUR 2020. — canonical PC / PQ / RR, the
  oracle assumption, the optimistic/pessimistic biases.
* **Brinkmann, Shraga & Bizer** — *SC-Block: Supervised Contrastive Blocking
  within Entity Resolution Pipelines.* arXiv:2303.03132, 2023. — fixed-`k` vs
  recall-target regimes; end-to-end runtime as the figure of merit.
* **Moslemi, Balamurugan & Milani** — evaluating biases in blocking. arXiv 2024. —
  F_PC,RR; PQ's RR-sensitivity.
* **Wang, Lin, Han et al.** — *UniBlocker*, universal dense blocking. 2024. — the
  90% PC threshold protocol; note its mAP (mean Average Precision) does not
  reproduce from its own stated formula, so only PC and PQ at a stated `k` are
  cross-paper-safe.
* **Wang et al.** — *NLSHBlock*, neuralised LSH (Locality-Sensitive Hashing) for
  blocking. SDM 2024. — per-dataset recall targets on 3:1:1 splits.
* **Nikoletos, Efthymiou, Papadakis & Stefanidis** — auto-configuring ER
  pipelines. 2025. — Optuna samplers; GPSampler (Gaussian-Process sampler) as the
  universal recommendation.
* **Karapiperis, Tjortjis & Verykios** — comprehensive survey of deep learning for
  ER. ACM CSUR 2026. — the benchmark-determinism critique.
* **Dasanaike** — *EnsembleLink.* Harvard Government, 2026. — embedding-model swap
  moves accuracy 1–2 points vs reranker swap up to 53.

**Matching stage**

* **Wu, Chu, Talwar et al.** — *ZeroER: Entity Resolution using Zero Labelled
  Examples.* SIGMOD 2020. — the GMM/EM matcher, feature grouping, adaptive
  regularisation.
* **Fellegi & Sunter** — *A Theory for Record Linkage.* JASA 1969. — comparison
  vectors as categorical; missing values get their own category.
* **Lee, Zhang & Kim** — maximum-entropy record linkage. 2021. — FLR/MMR error
  semantics; WJ-EM shown to be a special case; bipartite constraint enforced
  during estimation.
* **Neuhof, Fisichella & Papadakis** — end-to-end deep ER without labels. DOLAP
  2026. — the recall ≥ 0.90 contract; ZeroER 70.93% → 0.00% under a filter change.
* **van den Goorbergh, van Smeden, Timmerman & Van Calster** — JAMIA 2022. — the
  calibration suite: intercept, slope, flexible curve, Net Benefit.
* **Pavani, Aluri, Jadhav, Prasad & Sanka (Walmart)** — ER in practice. 2026. —
  the matcher tournament; transitivity precision thresholds.
