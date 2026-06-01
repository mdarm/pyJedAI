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

## Based On

* ZeroER — *Entity Resolution using Zero Labeled Examples* (SIGMOD 2020)
* pyJedAI — Open-source entity resolution framework

## References

Wu, R., Chaba, S., Sawlani, S., Chu, X., & Thirumuruganathan, S.
**ZeroER: Entity Resolution using Zero Labeled Examples**
SIGMOD Conference 2020, pp. 1149–1164.


## To-do (ideal target: 30/06/2026)

- [ ] Fully understand the reasoning behind ZeroER and replicate it from scratch (potential PyTorch implementation)
- [ ] Experiment with the feature generation used (Pandas vectorisation & Polars implementation)
- [ ] Fully understand PyJedAI's flow
- [x] Integrate ZeroER as an unsupervised matcher into PyJedAI's flow
- [ ] Create unit tests for replicating ZeroER results (LATER)
- [ ] Organise experiment tracking/results via Optuna DB and a dedicated README
- [x] Run experiments on all benchmark datasets
  - [ ] Abt-Buy & Amazon-Google Products (challenging ones according to George)
  - [x] Evaluate experiments on a Pareto front (minimise number of blocks while maximising KPIs)
  - [ ] Maybe use proper scoring rules while assessing best parameters (and not maximising KPIs such as F1)
  
