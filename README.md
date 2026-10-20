## Overview

This project extends ZeroER to improve:

* **Blocking & Filtering**
  Support for modern blocking techniques from pyJedAI, including semantic nearest-neighbor search using pre-trained language models.

* **Semantic Similarity**
  Replace traditional string similarities (e.g. Jaccard, Jaro) with embedding-based semantic similarity.

* **Performance & Scalability**
  Runtime analysis and optimization of ZeroER bottlenecks for faster large-scale ER workflows.

## Goals

* Combine syntactic and semantic blocking strategies
* Explore the interaction between semantic matching and syntactic filtering
* Deliver a fast, unsupervised ER pipeline without labeled data
* Integrate ZeroER++ into pyJedAI (last step)

## Experimental Setup

Experiments were conducted on either one of the following instances:

- [AWS EC2 g5.4xlarge](https://aws.amazon.com/ec2/instance-types/g5/): 16 vCPUs, Intel Xeon Platinum 8259CL processor (2.50 GHz), 64 GB system memory, and a single NVIDIA A10G Tensor Core GPU with 24 GB of GDDR6 VRAM, running Ubuntu 20.04.6 LTS (Focal Fossa)
- [AWS EC2 r6i.8xlarge](https://aws.amazon.com/ec2/instance-types/r6i/): 32 vCPUs, Intel Xeon Platinum 8375C processor (2.90 GHz), 256 GB of main memory, running Ubuntu 20.04.6 LTS (Focal Fossa)

Embeddings were computed on:

- Custom KVM virtual machine: 48 vCPUs on an AMD EPYC 9B45 processor (24 cores, 2 threads per core), 181 GB of system memory, and a single NVIDIA RTX PRO 6000 Blackwell GPU with 96 GB of VRAM

## Precomputed Embeddings

To compare semantic-similarity backends without re-encoding on every run, each
record of the five benchmarks was embedded ahead of time with 20 text-embedding
models, from classic sentence-transformer baselines to recent LLM-based
embedders. Datasets are discovered by walking `ZeroER/datasets/` for any folder
holding both a `left.csv` and a `right.csv`; each row is serialised into a
single string by joining its non-empty fields as `column: value` segments
separated by ` | `, and that text is fed to the encoder. The resulting matrices
are saved as `<model>_left.npy` / `<model>_right.npy` under
`ZeroER/datasets/embeddings/<dataset>/`, row-aligned with the source CSVs. See
[`datasets/embeddings/README.md`](ZeroER/datasets/embeddings/README.md) for the
full model list and filename-to-model mapping.

## Based On

* ZeroER — *Entity Resolution using Zero Labeled Examples* (SIGMOD 2021)
* pyJedAI — Open-source entity resolution framework

## References

Wu, R., Chaba, S., Sawlani, S., Chu, X., & Thirumuruganathan, S.
**ZeroER: Entity Resolution using Zero Labeled Examples**
SIGMOD Conference 2021, pp. 1149–1164.


## To-do (updated 20 October 2026)

### Completed by 22 September 2026

- [x] Rebuild record and attribute embeddings: 20 models, all five benchmarks; correct left/right-padding pooling. [Embedding inventory](ZeroER/datasets/embeddings/README.md)
- [x] Port Magellan feature generation; add cached-reference parity tests for all five benchmarks. The DBLP–Scholar `year` difference remains open below. [Tests](ZeroER/tests/test_feature_parity.py)
- [x] Add upstream-result regression tests for FZ, DA and AB, plus FZ transitivity. These do not establish full paper replication. [Tests](ZeroER/tests/test_paper_replication.py)
- [x] Integrate the local ZeroER model, matcher and estimator with pyJedAI. [Code layout](ZeroER/README.md)
- [x] Assess blocking on all five benchmarks: 20 models × k 1–100 × both directions; candidate counts and PC/PQ; aggregate model ranking at PC ≥ 90%.
- [x] Run matching on all five benchmarks, including AB/AG; compare string baselines, transitivity, missingness and name/title-only features.
- [x] Diagnose AB/AG collapse: sparse-attribute presence, initialization and regularization. [Diagnosis](ZeroER/outputs/collapse_diagnosis/REPORT.md)
- [x] Tune five ZeroER parameters for zero-fill and missing-aware string features: 1,500 trials, followed by full-data refits.
- [x] Compare strings + embedding cosines (1,340 trials) and all-attribute cosine-only features (1,320 trials).
- [x] Jointly tune cosine-only model combinations and nine ZeroER parameters: 1,000 trials, followed by full-data refits.
- [x] Record Brier scores; compare retrieval cosine + greedy one-to-one, accept-all and the blocking oracle.
- [x] Organise experiments in [tuning](ZeroER/hyperparameter-tuning.ipynb) and [full-data results](ZeroER/zeroer.ipynb) notebooks, with Optuna databases, SQL tables and a [results summary](ZeroER/MATCHING-PROGRESS.md).

### Completed 19–20 October 2026

- [x] **By 19 October:** Prepare publication notes and a manuscript draft in the [thesis repository](../msc-thesis/transcripts/acm-twc-2027/sections/notes.tex).
- [x] **20 October:** Repeat embedding-only fitting with 20 single encoders and seven combinations. Select settings within each feature set using marginal negative log-likelihood (NLL) on validation pairs, with three train/validation splits that share no records within each split.
- [x] **20 October:** Finish 594 searches × 10 configurations over 22 distinct candidate sets (five primary points and 19 sensitivity points, with two shared). Complete 593 selected full-data refits, including all 135 primary refits; retain the one failed sensitivity refit in reporting.
- [x] **20 October:** Compare posterior-probability, log-likelihood-ratio (LLR) and direct-cosine thresholds. Evaluate threshold-only, threshold + greedy one-to-one, and greedy without a cutoff on frozen scores; report gold-selected oracle thresholds separately. [Decision rules](ZeroER/decisions.py)
- [x] **20 October:** Run 44 original ZeroER baseline fits on the same candidates and 44 hard-EM/zero-fill diagnostic refits. Reuse Magellan caches and extract only the 868 missing Amazon–Google pairs.
- [x] **20 October:** Time all 593 successful selected fits and 22 direct-cosine candidate sets, with three serial repeats and fresh similarity calculations. Separate loading, feature preparation, fitting, scoring and matching decisions; fresh string-feature timing currently covers FZ only.
- [x] **20 October:** Organise the new runs in [tuning](ZeroER/unsupervised-tuning.ipynb) and [results](ZeroER/unsupervised-results.ipynb) notebooks with a separate experiment database; audit all 67,158 decision rows and preserve the earlier results. Commit the experiment code and ignore local research notes and planning drafts.

### In progress (20 October 2026)

- [ ] Add posterior-probability threshold curves beside LLR and record cosine for F2LLM-v2-4B, and clarify the experiment descriptions in the thesis README.

### Remaining

- [ ] Define how to choose the embedding model, decision score, threshold and assignment rule without target gold labels. The current best shared rule (F2LLM-v2-4B, LLR ≥ −5 + greedy, mean F1 .81936) was identified using gold F1; only its per-dataset fit settings were selected without labels.
- [ ] Evaluate the frozen selection procedure on untouched records or datasets, with uncertainty estimates and probability-calibration checks. Full-data refits remain development results.
- [ ] Resolve the DBLP–Scholar `year` dtype/string-conversion difference in the local Magellan feature port; the parity test currently records the known discrepancy.
- [ ] Create Walmart and IMDb embeddings for the planned pyJedAI D5–D9 generalisation experiments, then evaluate the frozen pipeline on new datasets.
- [ ] Extend fresh string-feature timing beyond FZ and measure embedding generation, blocking and end-to-end runtime. Use the measurements to decide whether vectorisation, Polars or PyTorch would help.
- [ ] Document the complete workflow and package ZeroER++ for integration into the main pyJedAI library; the local model/matcher/estimator adapter is implemented.
- [ ] Turn the publication notes into methods and results sections, with reproducible settings and clear distinctions between unsupervised fitting, gold-selected comparisons and oracle bounds.
