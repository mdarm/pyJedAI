"""The EM's posterior survives ``match_pairs``, for every candidate pair.

The match graph is thresholded at ``P_M >= 0.5``, so it cannot answer "what did
the model think of the pairs it rejected?" — and that is precisely the half a
calibration curve needs. Fitting intercept/slope on the survivors alone fits a
truncated sample and reports numbers that look fine and mean nothing.

Six invariants: the posterior is kept for *all* pairs, it is a continuous
probability rather than the rounded 0/1 the graph implies, the graph is exactly
its ``>= 0.5`` subset (so nothing is being retained inconsistently), a
pair/posterior length mismatch raises instead of being silently truncated by
``zip``, the accessor refuses before the EM has run, and the
``BlockFeatureCache`` route — the one the ladder uses — retains it too.

Pairs come from a deterministic stride over the left table rather than k-NN, so
most of the suite needs no embeddings: 662 pairs on ``fodors_zagats``, both
classes present, EM in a few seconds. Only the cache test needs `minilm6`, and
it skips without it.

Run:  pytest tests/test_posteriors.py -v
"""
import os

import numpy as np
import pandas as pd
import pytest

from data import DATASETS, RECORD_DIR, load_data
from estimator import BlockFeatureCache, ZeroEREstimator
from matcher import ZeroERMatcher
from zeroer_features import build_zeroer_features

DATASET = "fodors_zagats"
K = 2


@pytest.fixture(scope="module")
def fitted():
    """One EM run over a real candidate set, shared by the assertions below."""
    data = load_data(DATASET)
    attrs = DATASETS[DATASET]["attributes"]
    limit = data.dataset_limit

    # a deterministic stride over the left table, not k-NN: these invariants are
    # about plumbing, not blocking, and this keeps the test embedding-free
    n_right = len(data.dataset_2)
    pairs = [(i % limit, limit + i) for i in range(n_right)]
    pairs += [((i + 7) % limit, limit + i) for i in range(n_right)]

    features = build_zeroer_features(pairs, data.entities, attrs,
                                     dataset_limit=limit)
    matcher = ZeroERMatcher(attributes=attrs)
    matcher.data = data
    graph = matcher.match_pairs(pairs, features)
    return matcher, graph, pairs


def test_posterior_is_kept_for_every_candidate_pair(fitted):
    """Not just the ones that cleared 0.5 — the rejects are the point."""
    matcher, graph, pairs = fitted

    assert matcher.posteriors is not None
    assert len(matcher.posteriors) == len(pairs)
    assert matcher.candidate_pairs == list(pairs)
    # the graph is lossy, so this is a strict gain and not a restatement
    assert graph.number_of_edges() < len(pairs)


def test_posterior_is_continuous_not_thresholded(fitted):
    """A 0/1 vector would score as perfectly calibrated-or-broken and hide both."""
    matcher, _, _ = fitted
    p = matcher.posteriors

    assert p.dtype == np.float64
    assert p.min() >= 0.0 and p.max() <= 1.0
    # ``_hard`` rounds; ``match_pairs`` must not be taking that path
    assert not np.all(np.isin(p, (0.0, 1.0))), "posterior looks rounded"


def test_graph_is_exactly_the_half_above_the_threshold(fitted):
    """Pins the two representations against each other, in both directions."""
    matcher, graph, pairs = fitted
    frame = matcher.posterior_frame()

    assert list(frame.columns) == ["id1", "id2", "p_match"]
    assert len(frame) == len(pairs)
    np.testing.assert_array_equal(frame["p_match"].to_numpy(), matcher.posteriors)

    above = {(int(a), int(b)) for (a, b), p
             in zip(matcher.candidate_pairs, matcher.posteriors) if p >= 0.5}
    edges = {tuple(sorted(e)) for e in graph.edges()}
    assert {tuple(sorted(e)) for e in above} == edges


def test_misaligned_posterior_raises_instead_of_truncating():
    """``zip`` would have paired the first n and dropped the rest, silently."""
    matcher = ZeroERMatcher(attributes=["a"])
    features = pd.DataFrame({"a_a_lev_sim": [0.1, 0.9, 0.5]})

    # stub the EM: return one posterior too few for the three feature rows
    import matcher as matcher_module
    original = matcher_module.ZeroerModel.run_em
    matcher_module.ZeroerModel.run_em = staticmethod(
        lambda *a, **k: (None, np.array([0.2, 0.8])))
    try:
        with pytest.raises(RuntimeError, match="misaligned"):
            matcher.match_pairs([(0, 3), (1, 4), (2, 5)], features)
    finally:
        matcher_module.ZeroerModel.run_em = original


def test_posterior_frame_refuses_before_the_em_has_run():
    with pytest.raises(RuntimeError, match="no posteriors yet"):
        ZeroERMatcher().posterior_frame()


def test_cache_path_keeps_the_posterior_too():
    """The ``BlockFeatureCache`` route skips ``predict`` and goes straight to
    ``match_pairs`` — and it is the route the ladder runs on, so cover it."""
    model = "minilm6"
    if not os.path.exists(os.path.join(RECORD_DIR, DATASET, f"{model}_left.npy")):
        pytest.skip(f"no {model} record embeddings for {DATASET}")

    data = load_data(DATASET)
    attrs = DATASETS[DATASET]["attributes"]
    cache = BlockFeatureCache(DATASET, data, attrs, k_max=K, n_jobs=1)
    est = ZeroEREstimator(blocker="precomputed", cache=cache, vectorizer=model,
                          top_k=K, similarity_distance="cosine", attributes=attrs)
    graph = est.fit_predict(data)
    frame = est.posterior_frame()

    assert len(frame) == est.candset_size
    assert len(frame) > graph.number_of_edges()
    assert (frame["p_match"] < 0.5).sum() > 0, "no negatives retained"
