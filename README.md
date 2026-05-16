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

