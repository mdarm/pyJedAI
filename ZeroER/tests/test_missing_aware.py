"""The missing-aware fit and the present-block initialisation.

Both are departures from upstream, behind ``ZeroERMatcher(missing="ignore")`` and
``init_statistic="present_mean"``. What has to hold for them to be trusted:

* with nothing missing, the missing-aware fit *is* the plain fit;
* whatever sits in an absent attribute's feature block does not matter;
* a pair with no attribute present gets the prior, and every output is finite;
* the present-block seed does not count columns.

Stored Magellan matrices only — ``fodors_zagats`` (nothing missing) and
``abt_buy`` left/1 (``price`` present on 24% of pairs): no embeddings, seconds.

Run:  pytest tests/test_missing_aware.py -v
"""
import numpy as np
import pandas as pd
import pytest

from zeroer.experiments.evaluation import informative, load_side
from zeroer.matcher import ZeroERMatcher
from zeroer.model import initial_labels

PAPER = dict(c_bay=0.01, init_threshold=0.5, tol=1e-5, max_iter=200, transitivity=False)


def _cell(dataset, side, k):
    s = load_side(dataset, side, cosine=False)
    keep = s["rank"] < k
    present = s["present"][keep].reset_index(drop=True)
    features = informative(s["features"][keep].reset_index(drop=True), present)
    return list(zip(s["left"][keep], s["right"][keep])), features, present


def _posteriors(pairs, features, present, **config):
    matcher = ZeroERMatcher(**dict(PAPER, **config))
    matcher.match_pairs(pairs, features, present=present)
    return matcher


def test_with_nothing_missing_it_is_the_plain_fit():
    pairs, features, present = _cell("fodors_zagats", "right", 1)
    assert present.to_numpy().all()
    plain = _posteriors(pairs, features, present)
    aware = _posteriors(pairs, features, present, missing="ignore")
    np.testing.assert_allclose(aware.posteriors, plain.posteriors, atol=1e-8)


@pytest.fixture(scope="module")
def sparse():
    return _cell("abt_buy", "left", 1)


def test_what_sits_in_an_absent_block_does_not_matter(sparse):
    pairs, features, present = sparse
    assert 0.1 < present["price"].mean() < 0.5
    filled = features.copy()
    for column in features.columns:
        absent = ~present[column.split("_")[0]].to_numpy()
        filled.loc[absent, column] = features.loc[~absent, column].mean()
    config = dict(missing="ignore", init_statistic="present_mean")
    zeros = _posteriors(pairs, features, present, **config).posteriors
    means = _posteriors(pairs, filled, present, **config).posteriors
    assert np.isfinite(zeros).all()
    np.testing.assert_allclose(means, zeros, atol=1e-8)


def test_a_pair_with_nothing_present_gets_the_prior(sparse):
    pairs, features, present = sparse
    present = present.copy()
    present.iloc[:5] = False
    fitted = _posteriors(pairs, features, present, missing="ignore",
                         init_statistic="present_mean")
    blind = fitted.posteriors[:5]
    assert np.allclose(blind, blind[0]) and 0 < blind[0] < 1
    assert abs(blind[0] - fitted.model.pi_M) < 1e-3     # pi_M is one M-step ahead


def test_present_mean_does_not_count_columns():
    rng = np.random.default_rng(0)
    a, b = rng.random(200), rng.random(200)
    once = initial_labels(np.column_stack([a, b]), ["a_x", "b_x"], 0.5, "present_mean")
    thrice = initial_labels(np.column_stack([a, b, b, b]), ["a_x", "b_x", "b_y", "b_z"],
                            0.5, "present_mean")
    assert once == thrice
    assert once != initial_labels(np.column_stack([a, b, b, b]),
                                  ["a_x", "b_x", "b_y", "b_z"], 0.5, "sum")
