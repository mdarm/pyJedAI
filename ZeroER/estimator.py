"""Orchestration layer — the end-to-end ZeroER pipeline and its blocking.

This is the top of the three-layer split: it owns the candidate-generation
(blocking) stage and wires it to the matcher in ``matcher.py``, which in turn
drives the pure model in ``module.py``.

* ``nn_search`` / ``_precomputed_nn_blocks`` — faiss k-NN candidate generation
  over record embeddings computed offline.
* ``BlockFeatureCache`` — blocking + feature reuse across a sweep's trials.
* ``ZeroEREstimator`` — the public entry point. Every knob is a flat, typed
  constructor argument, so an Optuna ``suggest_*`` maps one-to-one onto it.
  Experiment code should only ever touch this class.
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
from pyjedai.datamodel import Data
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding

from data import load_embeddings
from matcher import ZeroERMatcher
from zeroer_features import build_zeroer_features

__all__ = ["ZeroERMatcher", "ZeroEREstimator", "BlockFeatureCache", "nn_search"]


def nn_search(v1, v2, k: int, similarity_distance: str) -> np.ndarray:
    """Ordered k-NN of every v2 row among v1 rows: an (n2, k) row-index
    matrix, nearest first — so the top-j neighbours are its first j columns."""
    v1, v2 = (np.asarray(v).astype(np.float32) for v in (v1, v2))  # own copies
    if similarity_distance == "cosine":
        faiss.normalize_L2(v1)
        faiss.normalize_L2(v2)
        index = faiss.IndexFlatIP(v1.shape[1])
    elif similarity_distance == "euclidean":
        index = faiss.IndexFlatL2(v1.shape[1])
    else:
        raise ValueError("similarity_distance must be 'cosine' or 'euclidean'")
    index.add(v1)
    _, neighbours = index.search(v2, min(k, len(v1)))
    return neighbours


def _blocks_from_neighbours(neighbours, limit: int) -> dict:
    """k-NN index matrix -> the ``{entity_id: neighbour set}`` block dict the
    pyJedAI stages and the metrics expect, keyed by the right entity's global
    id (its row offset by ``limit``)."""
    return {limit + j: set(row.tolist()) for j, row in enumerate(neighbours)}


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
    v1, v2 = vectors
    if len(v1) != data.num_of_entities_1 or len(v2) != data.num_of_entities_2:
        raise ValueError(
            f"embedding rows ({len(v1)}, {len(v2)}) do not match table sizes "
            f"({data.num_of_entities_1}, {data.num_of_entities_2}) — "
            "vectors not row-aligned with this Data")
    neighbours = nn_search(v1, v2, top_k, similarity_distance)
    return _blocks_from_neighbours(neighbours, data.dataset_limit)


class BlockFeatureCache:
    """Blocking + exact-feature reuse across a sweep's trials.

    k-NN candidate lists are nested in ``top_k`` (the top-3 neighbours are a
    prefix of the top-10 list), so blocking and ZeroER feature generation run
    **once** per (embedding model, similarity distance), at ``k_max``; every
    trial with ``top_k <= k_max`` is served as a row slice. Feature values are
    pair-local (no cross-pair statistics), so slices are exact — the constant-
    column drop, the one subset-dependent step, is re-applied per slice,
    matching ``build_zeroer_features(drop_zero_variance=True)`` on the same
    pairs. Builds are parallelised over pairs (``n_jobs``).

    Entries stay in memory for the cache's lifetime: at most
    (#models x #distances) matrices of ``k_max x n_right`` feature rows.

    Pass an instance as ``ZeroEREstimator(blocker='precomputed', cache=...)``
    to run the normal end-to-end pipeline off the cache.

    ``extra_features`` appends matcher features that ``build_zeroer_features``
    does not produce — Arm B's embedding similarities
    (``embedding_features.embedding_feature_builder``). It is called once per
    build, with the ``k_max`` pair array, and must return a frame of that many
    rows. Only pair-local features are admissible: the ``top_k`` slice above is
    exact precisely because no feature depends on the other pairs in the set.
    """

    def __init__(self, dataset, data, attributes, k_max, n_jobs=-1,
                 extra_features=None):
        self.dataset = dataset
        self.data = data
        self.attributes = attributes
        self.k_max = k_max
        self.n_jobs = n_jobs
        self.extra_features = extra_features
        self.limit = data.dataset_limit
        self._store = {}          # (model, distance) -> (nbrs, pairs, features)
        self.build_seconds = {}   # same key -> one-time blocking+feature cost

    def _built(self, model, distance):
        key = (model, distance)
        if key not in self._store:
            t0 = time.time()
            v1, v2 = load_embeddings(self.dataset, model, self.data)
            nbrs = nn_search(v1, v2, self.k_max, distance)
            pairs = np.column_stack([nbrs.ravel(),
                                     self.limit + np.repeat(np.arange(len(nbrs)),
                                                            nbrs.shape[1])])
            features = build_zeroer_features(
                pairs, self.data.entities, self.attributes,
                dataset_limit=self.limit, drop_zero_variance=False,
                n_jobs=self.n_jobs)
            if self.extra_features is not None:
                extra = self.extra_features(pairs)
                if len(extra) != len(pairs):
                    raise ValueError(
                        f"extra_features returned {len(extra)} rows for "
                        f"{len(pairs)} pairs — must be row-aligned")
                features = pd.concat(
                    [features.reset_index(drop=True),
                     extra.reset_index(drop=True)], axis=1)
            self._store[key] = (nbrs, pairs, features)
            self.build_seconds[key] = round(time.time() - t0, 3)
        return self._store[key]

    def get(self, model, distance, top_k):
        """(pairs, features, blocks) for one trial, rows aligned; ``blocks``
        is the ``{right_global_id: neighbour set}`` dict the metrics expect."""
        nbrs, pairs, features = self._built(model, distance)
        if top_k > nbrs.shape[1]:
            raise ValueError(f"top_k={top_k} exceeds cached k_max={nbrs.shape[1]}")
        keep = np.tile(np.arange(nbrs.shape[1]), len(nbrs)) < top_k
        sliced = features.loc[keep]
        sliced = sliced.loc[:, sliced.nunique() > 1].reset_index(drop=True)
        blocks = _blocks_from_neighbours(nbrs[:, :top_k], self.limit)
        return pairs[keep], sliced, blocks


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
        ``'precomputed'`` -> faiss k-NN over record embeddings computed offline,
        supplied either as ``vectors`` (one-shot) or as a ``cache`` (reused
        across trials); nothing is embedded live.
    vectorizer, top_k, similarity_distance:
        ``EmbeddingsNNBlockBuilding`` knobs (ignored when ``blocker='standard'``).
        Note ``EmbeddingsNNBlockBuilding`` supports only FAISS for the NN search.
        ``top_k``/``similarity_distance`` also drive the precomputed branch, where
        ``vectorizer`` names the offline embedding model (the ``cache`` key).
    vectors:
        ``(left, right)`` record-embedding matrices, row-aligned with the two
        tables (``data.load_embeddings``). Injected rather than loaded here so
        the estimator stays ignorant of the on-disk embedding layout.
    cache:
        A ``BlockFeatureCache``. Serves blocking *and* the ZeroER feature matrix
        from one build per (embedding model, distance), so a sweep's trials cost
        only the EM. Takes precedence over ``vectors``. ``blocking_seconds`` then
        reports the cache's one-time build cost rather than the ~0s of a hit, so
        the figure stays comparable across trials.
    smoothing_factor, block_filtering_ratio, weighting_scheme:
        Standard-branch cleaning knobs (ignored when ``blocker='embeddings'``):
        ``BlockPurging`` purge threshold, ``BlockFiltering`` keep-ratio, and the
        ``WeightedEdgePruning`` meta-blocking weight scheme.
    attributes:
        Columns the matcher compares; ``None`` -> shared-attribute auto-detect.
    extra_features:
        Optional ``pairs -> DataFrame`` appended to the ZeroER string-similarity
        matrix — Arm B's per-attribute embedding similarities, built by
        ``embedding_features.embedding_feature_builder``. Injected rather than
        loaded here, like ``vectors``, so the estimator stays ignorant of the
        on-disk embedding layout. On the ``cache`` path the cache owns feature
        generation, so pass it to the ``BlockFeatureCache`` instead; supplying
        both is an error rather than a silent no-op.
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
        cache: Optional[BlockFeatureCache] = None,
        extra_features=None,
    ) -> None:
        if blocker not in ("embeddings", "standard", "precomputed"):
            raise ValueError(
                "blocker must be 'embeddings', 'standard' or 'precomputed'")
        if blocker == "precomputed" and vectors is None and cache is None:
            raise ValueError(
                "blocker='precomputed' needs vectors=(left, right) or cache=")
        if cache is not None and extra_features is not None:
            raise ValueError(
                "pass extra_features to the BlockFeatureCache, not the estimator "
                "— the cache builds and slices the feature matrix on that path")
        self.blocker = blocker
        self.vectors = vectors
        self.cache = cache
        self.extra_features = extra_features
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
        self._cached_pairs = None       # cache branch only: rows aligned with
        self._cached_features = None    # the feature matrix, so EM skips rebuild

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
            if self.cache is not None:
                self._cached_pairs, self._cached_features, blocks = self.cache.get(
                    self.vectorizer, self.similarity_distance, self.top_k)
            else:
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
        # a cache hit costs ~0s, so report the one-time build instead — that is
        # the figure a trial's blocking cost should be compared on
        self.blocking_seconds = (
            self.cache.build_seconds[(self.vectorizer, self.similarity_distance)]
            if self.cache is not None else time.time() - t0)
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
            attributes=self.attributes, c_bay=self.c_bay, max_iter=self.max_iter,
            extra_features=self.extra_features)
        if self._cached_features is not None:
            # features already built by the cache — go straight to the EM
            self.matcher.data = data      # predict() would have set this
            self.pairs = self.matcher.match_pairs(
                self._cached_pairs, self._cached_features)
        else:
            self.pairs = self.matcher.predict(blocks, data, tqdm_disable=True)
        return self.pairs
