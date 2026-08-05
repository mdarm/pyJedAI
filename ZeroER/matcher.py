"""Adapter layer — ZeroER as a pyJedAI matching stage.

This is the wrapper concern of the three-layer split. It depends on both:

* the **model** (``module.ZeroerModel`` / ``get_y_init_given_threshold``), which
  knows nothing about pyJedAI, and
* the **pyJedAI framework** (``Data``, ``PYJEDAIFeature``, ``Evaluation``),
  which knows nothing about ZeroER.

``ZeroERMatcher`` is the low-level pyJedAI stage (``PYJEDAIFeature``) that turns
a block structure into a match graph: it builds the exact ZeroER/Magellan
similarity-feature matrix (``zeroer_features``) from the candidate pairs and
runs the ``ZeroerModel`` from ``module.py`` on it. Orchestration (blocking ->
matching) lives one layer up, in ``estimator.py``.
"""
from __future__ import annotations

import time

import pandas as pd
from networkx import Graph

from pyjedai.datamodel import Data, PYJEDAIFeature
from pyjedai.evaluation import Evaluation

from module import ZeroerModel, get_y_init_given_threshold
from zeroer_features import build_zeroer_features

__all__ = ["ZeroERMatcher"]


class ZeroERMatcher(PYJEDAIFeature):
    """Unsupervised matching via ZeroER's EM over string-similarity features.

    Consumes any pyJedAI block structure (cleaned candidate-pair dict),
    builds the exact ZeroER/Magellan similarity-feature matrix
    (``zeroer_features.build_zeroer_features``), and runs the ``ZeroerModel``
    from ``module.py``. Predicted matches (P_M >= 0.5) become weighted edges.
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
        feature_matrix = build_zeroer_features(
            candidate_pairs, data.entities, attrs,
            dataset_limit=None if data.is_dirty_er else data.dataset_limit)
        self.features_time = time.time() - feat_t0

        self.match_pairs(candidate_pairs, feature_matrix)
        self.execution_time = time.time() - start
        return self.pairs

    def match_pairs(self, candidate_pairs, feature_matrix: pd.DataFrame) -> Graph:
        """EM + thresholding on an externally built feature matrix (rows
        aligned with ``candidate_pairs``). ``predict`` lands here after
        building the matrix itself; sweeps that cache features across trials
        (``estimator.BlockFeatureCache``) reach it through
        ``ZeroEREstimator.fit_predict``."""
        if feature_matrix.shape[1] == 0:
            raise RuntimeError("Feature matrix is empty after dropping constant columns.")
        self.feature_matrix = feature_matrix

        em_t0 = time.time()
        y_init = get_y_init_given_threshold(feature_matrix)
        _, P_M = ZeroerModel.run_em(
            similarity_matrixs=(feature_matrix.values, None, None),
            feature_names=feature_matrix.columns.tolist(),
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
                self.pairs.add_edge(int(id1), int(id2), weight=float(p))
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
