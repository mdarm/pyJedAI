"""Adapter layer — ZeroER as a pyJedAI matching stage.

This is the wrapper concern of the three-layer split. It depends on both:

* the **model** (``zeroer.model.ZeroerModel`` / ``get_y_init_given_threshold``), which
  knows nothing about pyJedAI, and
* the **pyJedAI framework** (``Data``, ``PYJEDAIFeature``, ``Evaluation``),
  which knows nothing about ZeroER.

``ZeroERMatcher`` is the low-level pyJedAI stage (``PYJEDAIFeature``) that turns
a block structure into a match graph: it builds the exact ZeroER/Magellan
similarity-feature matrix (``zeroer.features.strings``) from the candidate pairs and
runs the ``ZeroerModel`` from ``zeroer.model`` on it. Orchestration (blocking ->
matching) lives one layer up, in ``zeroer.estimator``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from networkx import Graph

from pyjedai.datamodel import Data, PYJEDAIFeature
from pyjedai.evaluation import Evaluation

from zeroer.model import ZeroerModel, initial_labels
from zeroer.features.strings import build_zeroer_features

__all__ = ["ZeroERMatcher"]


class ZeroERMatcher(PYJEDAIFeature):
    """Unsupervised matching via ZeroER's EM over string-similarity features.

    Consumes any pyJedAI block structure (cleaned candidate-pair dict),
    builds the exact ZeroER/Magellan similarity-feature matrix
    (``zeroer.features.strings.build_zeroer_features``), and runs the ``ZeroerModel``
    from ``zeroer.model``. Predicted matches (P_M >= 0.5) become weighted edges.

    ``extra_features`` is an optional ``pairs -> DataFrame`` whose columns are
    appended to that matrix — Arm B's per-attribute embedding similarities
    (``zeroer.features.embeddings.embedding_feature_builder``). It applies to
    ``predict``, which builds the matrix itself; callers that build their own and
    go straight to ``match_pairs`` (the ``BlockFeatureCache`` path) have already
    appended whatever they wanted.

    The match graph keeps only the edges above the 0.5 threshold, so it is lossy:
    the EM's posterior for a pair it rejected is not recoverable from it. The full
    posterior is therefore retained alongside, as ``posteriors`` (row-aligned with
    ``candidate_pairs``) and via ``posterior_frame()``. Calibration needs the
    negatives as much as the positives — a reliability curve fitted on the
    ``P_M >= 0.5`` survivors alone is fitted on a truncated sample and its
    intercept/slope mean nothing.
    """

    _method_name = "ZeroER Unsupervised Matching"
    _method_info = (
        "Unsupervised EM over per-attribute string similarity features "
        "(ported from chu-data-lab/zeroer)."
    )
    _method_short_name = "ZeroER"

    def __init__(self, attributes=None, c_bay: float = 0.1, max_iter: int = 40,
                 extra_features=None, init_threshold: float = 0.8, tol=None,
                 transitivity: bool = False, pi_M=None, hard: bool = False,
                 init_statistic: str = "sum", missing: str = "zero"):
        if missing not in ("zero", "ignore"):
            raise ValueError("missing must be 'zero' or 'ignore'")
        super().__init__()
        self.attributes = attributes
        self.c_bay = c_bay
        self.max_iter = max_iter
        self.extra_features = extra_features
        # The defaults above and below are the released code's. The paper's
        # stated configuration (sections 3.3, 5.1) is c_bay=0.01,
        # init_threshold=0.5, tol=1e-5, max_iter=200.
        self.init_threshold = init_threshold
        self.tol = tol
        # ZeroER's in-model transitivity for duplicate-free tables — the only
        # variant one cross-table candidate set can serve; the paper's full
        # Algorithm 2 also needs within-table candidate sets and features.
        self.transitivity = transitivity
        # pi_M: the initial match prior (None = the share of initial labels);
        # hard: hard-assignment EM. Both are upstream run_em arguments.
        self.pi_M = pi_M
        self.hard = hard
        # Not upstream — see zeroer.model.initial_labels and ZeroerModel.__init__.
        # init_statistic 'present_mean' seeds EM from the blocks a pair has;
        # missing='ignore' leaves absent attributes out of the likelihood instead
        # of reading them as zero similarity. Both need match_pairs(present=).
        self.init_statistic = init_statistic
        self.missing = missing
        self.model: ZeroerModel = None
        self.feature_matrix: pd.DataFrame = None
        self.pairs: Graph = None
        # row-aligned pair list and the EM's full posterior over it — the graph
        # above drops everything below 0.5, these do not
        self.candidate_pairs: list | None = None
        self.posteriors: np.ndarray | None = None
        self.execution_time = 0.0
        self.features_time = 0.0
        self.em_time = 0.0

    def predict(self, blocks: dict, data: Data, tqdm_disable: bool = True) -> Graph:
        if not blocks:
            raise ValueError("Empty blocks structure")
        self.data = data

        if self.missing != "zero":
            raise ValueError("missing='ignore' needs a presence mask: build the "
                             "matrix yourself and call match_pairs(present=...)")
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
        if self.extra_features is not None:
            extra = self.extra_features(candidate_pairs)
            if len(extra) != len(candidate_pairs):
                raise ValueError(
                    f"extra_features returned {len(extra)} rows for "
                    f"{len(candidate_pairs)} pairs — must be row-aligned")
            feature_matrix = pd.concat(
                [feature_matrix.reset_index(drop=True),
                 extra.reset_index(drop=True)], axis=1)
        self.features_time = time.time() - feat_t0

        self.match_pairs(candidate_pairs, feature_matrix)
        self.execution_time = time.time() - start
        return self.pairs

    def match_pairs(self, candidate_pairs, feature_matrix: pd.DataFrame,
                    present=None, trace: bool = False, y_init=None) -> Graph:
        """EM + thresholding on an externally built feature matrix (rows
        aligned with ``candidate_pairs``). ``predict`` lands here after
        building the matrix itself; sweeps that cache features across trials
        (``zeroer.estimator.BlockFeatureCache``) reach it through
        ``ZeroEREstimator.fit_predict``.

        ``present``: {attribute block: boolean array, True where both records
        have the value}, row-aligned — what ``init_statistic='present_mean'`` and
        ``missing='ignore'`` read. ``trace`` keeps the per-iteration state in
        ``self.model.history``. ``y_init`` optionally supplies binary initial
        labels; otherwise they are computed once using the selected statistic."""
        if self.missing == "ignore" and present is None:
            raise ValueError("missing='ignore' needs present=")
        if feature_matrix.shape[1] == 0:
            raise RuntimeError("Feature matrix is empty after dropping constant columns.")
        self.feature_matrix = feature_matrix

        em_t0 = time.time()
        names = feature_matrix.columns.tolist()
        if y_init is None:
            y_init = initial_labels(feature_matrix.values, names, self.init_threshold,
                                    self.init_statistic, present)
        # transitivity looks pairs up by id, so it needs the real ones
        id_df = pd.DataFrame(list(candidate_pairs)) if self.transitivity else None
        # P_M is the last E-step's posterior; the model's parameters are one
        # M-step ahead of it (run_em ends on an M-step, as upstream does)
        self.model, P_M = ZeroerModel.run_em(
            similarity_matrixs=(feature_matrix.values, None, None),
            feature_names=names,
            y_inits=(y_init, None, None),
            id_dfs=(id_df, None, None),
            LR_dup_free=self.transitivity,
            LR_identical=False,
            run_trans=self.transitivity,
            c_bay=self.c_bay,
            max_iter=self.max_iter,
            init_threshold=self.init_threshold,
            tol=self.tol,
            pi_M=self.pi_M,
            hard=self.hard,
            init_statistic=self.init_statistic,
            present=present if self.missing == "ignore" else None,
            trace=trace,
        )
        self.em_time = time.time() - em_t0

        candidate_pairs = list(candidate_pairs)
        P_M = np.asarray(P_M, dtype=float).ravel()
        # zip() below would silently truncate to the shorter of the two, hiding a
        # misalignment as a plausible-looking edge set — the same failure mode the
        # extra_features row-count guard exists for.
        if len(P_M) != len(candidate_pairs):
            raise RuntimeError(
                f"EM returned {len(P_M)} posteriors for {len(candidate_pairs)} "
                "candidate pairs — feature matrix and pair list are misaligned")
        self.candidate_pairs = candidate_pairs
        self.posteriors = P_M

        self.pairs = Graph()
        for (id1, id2), p in zip(candidate_pairs, P_M):
            if p >= 0.5:
                self.pairs.add_edge(int(id1), int(id2), weight=float(p))
        return self.pairs

    def posterior_frame(self) -> pd.DataFrame:
        """Every candidate pair with its EM posterior — the input calibration
        (intercept/slope, reliability curve, Net Benefit) needs.

        ``id1`` / ``id2`` are pyJedAI *internal* ids, the same space ``evaluate``
        maps the gold standard into via ``data._ids_mapping_1/2``; join through
        that mapping to attach labels. No labels are attached here — screening or
        scoring on gold is the caller's decision, not the matcher's.
        """
        if self.posteriors is None:
            raise RuntimeError("no posteriors yet — run predict() or match_pairs()")
        return pd.DataFrame({
            "id1": [int(a) for a, _ in self.candidate_pairs],
            "id2": [int(b) for _, b in self.candidate_pairs],
            "p_match": self.posteriors,
        })

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
            "init_threshold": self.init_threshold,
            "tol": self.tol,
            "transitivity": self.transitivity,
            "pi_M": self.pi_M,
            "hard": self.hard,
            "init_statistic": self.init_statistic,
            "missing": self.missing,
        }

    def stats(self) -> None:
        pass
