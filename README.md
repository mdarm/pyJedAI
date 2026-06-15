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


## To-do (16/06/2026)

- [ ] Fully reproduce ZeroER's feature generation and test it (only way to for results to be comparable)
- [ ] Create unit tests for replicating ZeroER results (on all benchmarks)
- [ ] Generalise to integrate one-column benchmarks (all possible pairs from a table)

### ideal target: 16/07/2026

- [ ] Fully understand the reasoning behind ZeroER and replicate it from scratch (potential PyTorch implementation)
- [ ] Experiment with the feature generation used (Pandas vectorisation & Polars implementation)
- [ ] Fully understand PyJedAI's flow
- [x] Integrate ZeroER as an unsupervised matcher into PyJedAI's flow
- [ ] Organise experiment tracking/results via Optuna DB and a dedicated README
- [x] Run experiments on all benchmark datasets
  - [ ] Abt-Buy & Amazon-Google Products (challenging ones according to George)
  - [ ] Maybe use proper scoring rules while assessing best parameters (and not maximising KPIs such as F2)
