"""Wrapper layer — ZeroER inside the pyJedAI framework.

Named ``pyjedai_module`` (not ``pyjedai``) on purpose: a module literally
called ``pyjedai.py`` would shadow the installed ``pyjedai`` package and break
the ``from pyjedai... import`` lines below.

This is the wrapper concern of the three-layer split. It depends on both:

* the **model** (``zeroer.ZeroerModel`` / ``get_y_init_given_threshold``), which
  knows nothing about pyJedAI, and
* the **pyJedAI framework** (``Data``, ``PYJEDAIFeature``, the blockers, the
  string matchers), which knows nothing about ZeroER.

Two framework-facing concerns live here:

* ``ZeroERMatcher`` — the low-level pyJedAI stage (``PYJEDAIFeature``) that
  turns a block structure into a match graph: it builds the exact
  ZeroER/Magellan similarity-feature matrix (``zeroer_features``) from the
  candidate pairs and runs the ``ZeroerModel`` from ``zeroer.py`` on it.

* ``ZeroEREstimator`` — the orchestrator / public entry point. It owns the
  end-to-end pipeline (blocking -> matching) behind a single flat, typed
  constructor, which is exactly the surface an Optuna sweep tunes. Most
  experiment code should only ever touch this class.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from networkx import Graph

from pyjedai.block_building import StandardBlocking
from pyjedai.block_cleaning import BlockFiltering, BlockPurging
from pyjedai.comparison_cleaning import WeightedEdgePruning
from pyjedai.datamodel import Data, PYJEDAIFeature
from pyjedai.evaluation import Evaluation
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding

from module import ZeroerModel, get_y_init_given_threshold
from zeroer_features import build_zeroer_features

__all__ = ["ZeroERMatcher", "ZeroEREstimator"]


def _precomputed_nn_blocks(vectors, data, top_k: int,
                           similarity_distance: str) -> dict:
    """k-NN blocking over precomputed record embeddings.

    Replaces ``EmbeddingsNNBlockBuilding``'s vectorise-then-search with a plain
    faiss search over vectors embedded offline: index the left table, query the
    right, key blocks by the right entity's global id — the same
    ``{entity_id: neighbour set}`` shape the other blockers emit.
    """
    if data.is_dirty_er:
        raise ValueError("precomputed blocking supports clean-clean ER only")
    v1, v2 = (np.asarray(v).astype(np.float32) for v in vectors)  # own copies
    if len(v1) != data.num_of_entities_1 or len(v2) != data.num_of_entities_2:
        raise ValueError(
            f"embedding rows ({len(v1)}, {len(v2)}) do not match table sizes "
            f"({data.num_of_entities_1}, {data.num_of_entities_2}) — "
            "vectors not row-aligned with this Data")
    if similarity_distance == "cosine":
        faiss.normalize_L2(v1)
        faiss.normalize_L2(v2)
        index = faiss.IndexFlatIP(v1.shape[1])
    elif similarity_distance == "euclidean":
        index = faiss.IndexFlatL2(v1.shape[1])
    else:
        raise ValueError("similarity_distance must be 'cosine' or 'euclidean'")
    index.add(v1)
    _, neighbours = index.search(v2, min(top_k, len(v1)))
    limit = data.dataset_limit
    return {limit + j: set(row.tolist()) for j, row in enumerate(neighbours)}


class ZeroERMatcher(PYJEDAIFeature):
    """Unsupervised matching via ZeroER's EM over string-similarity features.

    Consumes any pyJedAI block structure (cleaned candidate-pair dict),
    builds the exact ZeroER/Magellan similarity-feature matrix
    (``zeroer_features.build_zeroer_features``), and runs the ``ZeroerModel``
    from ``zeroer.py``. Predicted matches (P_M >= 0.5) become weighted edges.
    """

    _method_name = "ZeroER Unsupervised Matching"
    _method_info = (
        "Unsupervised EM over per-attribute string similarity features "
        "(ported from chu-data-lab/zeroer)."
    )
    _method_short_name = "ZeroER"

    def __init__(self, attributes=None, c_bay: float = 0.1, max_iter: int = 40):
        super().__init__()
        self.attributes = attributes
        self.c_bay = c_bay
        self.max_iter = max_iter
        self.feature_matrix: pd.DataFrame = None
        self.pairs: Graph = None
        self.execution_time = 0.0
        self.features_time = 0.0
        self.em_time = 0.0

    def predict(self, blocks: dict, data: Data, tqdm_disable: bool = True) -> Graph:
        if not blocks:
            raise ValueError("Empty blocks structure")
        self.data = data

        if self.attributes is None:
            attrs_2 = data.attributes_2 if not data.is_dirty_er else data.attributes_1
            attrs = [a for a in data.attributes_1 if a in attrs_2]
            if not attrs:
                raise ValueError("No shared attributes between datasets; pass attributes= explicitly.")
        else:
            attrs = list(self.attributes)

        start = time.time()
        candidate_pairs = []
        for entity_id, candidates in blocks.items():
            for cand in candidates:
                candidate_pairs.append((entity_id, cand))
        if not data.is_dirty_er:
            # orient as (left-table row, right-table row) — the feature
            # functions are left/right-asymmetric (e.g. monge_elkan)
            limit = data.dataset_limit
            candidate_pairs = [(a, b) if a < limit else (b, a)
                               for a, b in candidate_pairs]

        feat_t0 = time.time()
        self.feature_matrix = build_zeroer_features(
            candidate_pairs, data.entities, attrs,
            dataset_limit=None if data.is_dirty_er else data.dataset_limit)
        self.features_time = time.time() - feat_t0

        if self.feature_matrix.shape[1] == 0:
            raise RuntimeError("Feature matrix is empty after dropping constant columns.")

        em_t0 = time.time()
        y_init = get_y_init_given_threshold(self.feature_matrix)
        _, P_M = ZeroerModel.run_em(
            similarity_matrixs=(self.feature_matrix.values, None, None),
            feature_names=self.feature_matrix.columns.tolist(),
            y_inits=(y_init, None, None),
            id_dfs=(None, None, None),
            LR_dup_free=False,
            LR_identical=False,
            run_trans=False,
            c_bay=self.c_bay,
            max_iter=self.max_iter,
        )
        self.em_time = time.time() - em_t0

        self.pairs = Graph()
        for (id1, id2), p in zip(candidate_pairs, P_M):
            if p >= 0.5:
                self.pairs.add_edge(id1, id2, weight=float(p))

        self.execution_time = time.time() - start
        return self.pairs

    def evaluate(self,
                 prediction,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True):
        if self.data is None:
            raise AttributeError("Cannot evaluate without data object.")
        if self.data.ground_truth is None:
            raise AttributeError("Cannot evaluate without ground truth.")

        eval_obj = Evaluation(self.data)
        true_positives = 0
        total_matching_pairs = prediction.number_of_edges()
        for _, (id1, id2) in self.data.ground_truth.iterrows():
            id1 = self.data._ids_mapping_1[id1]
            id2 = self.data._ids_mapping_1[id2] if self.data.is_dirty_er \
                else self.data._ids_mapping_2[id2]
            if (id1 in prediction and id2 in prediction[id1]) or \
                    (id2 in prediction and id1 in prediction[id2]):
                true_positives += 1

        eval_obj.calculate_scores(true_positives=true_positives,
                                  total_matching_pairs=total_matching_pairs)
        return eval_obj.report(self.method_configuration(),
                               export_to_df,
                               export_to_dict,
                               with_classification_report,
                               verbose)

    def _configuration(self) -> dict:
        return {
            "Attributes": self.attributes if self.attributes is not None else "auto",
            "c_bay": self.c_bay,
            "max_iter": self.max_iter,
        }

    def stats(self) -> None:
        pass


class ZeroEREstimator:
    """End-to-end unsupervised ER: a pyJedAI blocker feeding ZeroER's EM.

    The constructor is the full hyperparameter / tuning surface — every knob is
    a flat, typed argument, so an Optuna ``suggest_*`` maps one-to-one onto it.

    Parameters
    ----------
    blocker:
        ``'embeddings'`` -> ``EmbeddingsNNBlockBuilding`` (PLM nearest-neighbour
        candidate generation); ``'standard'`` -> ``StandardBlocking`` +
        ``BlockPurging`` + ``BlockFiltering`` + ``WeightedEdgePruning``;
        ``'precomputed'`` -> faiss k-NN over injected record embeddings
        (``vectors``), nothing embedded live.
    vectorizer, top_k, similarity_distance:
        ``EmbeddingsNNBlockBuilding`` knobs (ignored when ``blocker='standard'``).
        Note ``EmbeddingsNNBlockBuilding`` supports only FAISS for the NN search.
        ``top_k``/``similarity_distance`` also drive the precomputed branch.
    vectors:
        ``(left, right)`` record-embedding matrices, row-aligned with the two
        tables (``utils.load_embeddings``). Required when
        ``blocker='precomputed'``; injected rather than loaded here so the
        estimator stays ignorant of the on-disk embedding layout.
    smoothing_factor, block_filtering_ratio, weighting_scheme:
        Standard-branch cleaning knobs (ignored when ``blocker='embeddings'``):
        ``BlockPurging`` purge threshold, ``BlockFiltering`` keep-ratio, and the
        ``WeightedEdgePruning`` meta-blocking weight scheme.
    attributes:
        Columns the matcher compares; ``None`` -> shared-attribute auto-detect.
    c_bay, max_iter:
        ZeroER EM knobs, forwarded to ``ZeroerModel`` via ``ZeroERMatcher``.
    """

    def __init__(
        self,
        blocker: str = "embeddings",
        vectorizer: str = "sminilm",
        top_k: int = 1,
        similarity_distance: str = "cosine",
        weighting_scheme: str = "EJS",
        smoothing_factor: float = 1.025,
        block_filtering_ratio: float = 0.8,
        attributes: Optional[List[str]] = None,
        c_bay: float = 0.1,
        max_iter: int = 40,
        vectors: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        if blocker not in ("embeddings", "standard", "precomputed"):
            raise ValueError(
                "blocker must be 'embeddings', 'standard' or 'precomputed'")
        if blocker == "precomputed" and vectors is None:
            raise ValueError("blocker='precomputed' needs vectors=(left, right)")
        self.blocker = blocker
        self.vectors = vectors
        self.vectorizer = vectorizer
        self.top_k = top_k
        self.similarity_distance = similarity_distance
        self.weighting_scheme = weighting_scheme
        self.smoothing_factor = smoothing_factor
        self.block_filtering_ratio = block_filtering_ratio
        self.attributes = attributes
        self.c_bay = c_bay
        self.max_iter = max_iter
        # populated by build_blocks / fit_predict
        self.blocking_seconds: float = 0.0
        self.candset_size: int = 0
        self.blocks: Optional[dict] = None
        self.matcher: Optional[ZeroERMatcher] = None
        self.pairs: Optional[Graph] = None

    def build_blocks(self, data: Data) -> dict:
        """Run the configured blocker; record candset size + wall time."""
        t0 = time.time()
        if self.blocker == "embeddings":
            blocks = EmbeddingsNNBlockBuilding(
                vectorizer=self.vectorizer, similarity_search="faiss",
            ).build_blocks(
                data,
                top_k=self.top_k,
                similarity_distance=self.similarity_distance,
                load_embeddings_if_exist=True,
                save_embeddings=True,
                tqdm_disable=True,
            )
        elif self.blocker == "precomputed":
            blocks = _precomputed_nn_blocks(
                self.vectors, data, self.top_k, self.similarity_distance)
        else:
            blocks = StandardBlocking(disable_ray=True).build_blocks(data, tqdm_disable=True)
            blocks = BlockPurging(
                smoothing_factor=self.smoothing_factor).process(blocks, data, tqdm_disable=True)
            blocks = BlockFiltering(
                ratio=self.block_filtering_ratio).process(blocks, data, tqdm_disable=True)
            blocks = WeightedEdgePruning(
                weighting_scheme=self.weighting_scheme).process(blocks, data, tqdm_disable=True)
        if isinstance(blocks, tuple):       # some blockers return (blocks, info)
            blocks = blocks[0]
        self.blocking_seconds = time.time() - t0
        self.candset_size = sum(len(s) for s in blocks.values())
        self.blocks = blocks
        return blocks

    def fit_predict(self, data: Data) -> Graph:
        """Block, then run the ZeroER matcher; return the predicted match graph.

        Single end-to-end entry point (the sklearn-style ``fit_predict`` name is
        the ml-project-style orchestrator convention, not ZeroER's own API).

        It deliberately stops at the raw *pairwise* match graph and does **not**
        cluster: the original ZeroER reports pairwise match predictions
        (``pred.csv``), so scoring these edges directly keeps end-to-end P/R/F1
        at the same granularity as the paper — i.e. a direct comparison against
        the original ZeroER experiments. Wiring a clustering stage in here would
        change the estimand and break that comparison.
        """
        blocks = self.build_blocks(data)
        self.matcher = ZeroERMatcher(
            attributes=self.attributes, c_bay=self.c_bay, max_iter=self.max_iter)
        self.pairs = self.matcher.predict(blocks, data, tqdm_disable=True)
        return self.pairs
