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


## To-do (22/09/2026)

### Completed

- [x] Rebuild record and attribute embeddings: 20 models, all five benchmarks; correct left/right-padding pooling. [Embedding inventory](ZeroER/datasets/embeddings/README.md)
- [x] Port Magellan feature generation; add cached-reference parity tests for all five benchmarks. One known DBLP–Scholar difference remains below. [Tests](ZeroER/tests/test_feature_parity.py)
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

### Remaining

- [ ] Resolve the DBLP–Scholar `year` dtype/string-conversion difference in feature parity.
- [ ] Complete single-column/single-table benchmark support and tests; cached embedding blocking currently requires two tables.
- [ ] Compare single models, model combinations and string features with equal tuning budgets and the same attributes/assignment rules.
- [ ] Use consistent, disjoint evaluation splits and uncertainty estimates; current full-data results include tuning pairs.
- [ ] Define matching selection across datasets; blocking already has aggregate ranking, but matching tuning selects per dataset. Validate donor transfer.
- [ ] Evaluate proper-scoring-rule selection and calibration; Brier is recorded, while tuning still maximises F1.
- [ ] Test decision thresholds beyond .5; initialization thresholds and transitivity on/off have already been explored. Review findings with George.
- [ ] Run hybrid joint tuning and L∞ matching experiments; feature caches exist.
- [ ] Profile encoding, feature generation and EM; compare Pandas/vectorised or Polars implementations. Consider PyTorch if profiling justifies it.
- [ ] Consider applying final architecture into new dataset(s), and get KPIs from that.
- [ ] Document the pyJedAI workflow and package ZeroER++ for integration into the main library; the local adapter is already implemented.
