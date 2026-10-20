"""The baseline-configuration knobs reach the EM, and do what they say.

Upstream hard-wires the initialisation threshold (0.8) and the stopping rule, and
``match_pairs`` used to hard-code transitivity off and throw the fitted model
away. The paper's stated configuration differs on all of them (κ′ = 0.01,
ε = 0.5, likelihood change < 1e-5, <= 200 iterations), so they are now arguments
whose defaults stay upstream's — ``test_paper_replication.py`` and
``test_posteriors.py`` are what holds the defaults still; this file covers the
non-default side.

Runs on the stored Magellan matrix of the smallest ladder cell
(``fodors_zagats`` right/1, 331 pairs): no feature generation, EM in seconds.

Run:  pytest tests/test_baseline_config.py -v
"""
from os.path import join

import numpy as np
import pandas as pd
import pytest

from matcher import ZeroERMatcher
from module import get_y_init_given_threshold
from config import FEATURES_DIR, cell_name


def _load(k):
    """(pairs, features) — pairs as row numbers on disjoint left/right ranges."""
    stored = pd.read_csv(join(FEATURES_DIR, cell_name("fodors_zagats", "right", k) + ".csv.gz"),
                         dtype={"ltable_id": str, "rtable_id": str})
    left = pd.factorize(stored["ltable_id"])[0]
    right = pd.factorize(stored["rtable_id"])[0] + left.max() + 1
    features = stored.drop(columns=["ltable_id", "rtable_id"])
    # 'type' / 'class' are outside the attribute allow-list ('class' leaks gold)
    features = features.loc[:, ~features.columns.str.split("_").str[0].isin(["type", "class"])]
    features = features.loc[:, features.nunique() > 1]
    return list(zip(left, right)), features


@pytest.fixture(scope="module")
def cell():
    return _load(1)


def test_init_threshold_reaches_the_model(cell):
    """The matcher computes the threshold seed once and the model honours it."""
    pairs, features = cell
    matcher = ZeroERMatcher(init_threshold=0.5, max_iter=1)
    matcher.match_pairs(pairs, features)
    at_half = get_y_init_given_threshold(features, 0.5)
    assert list(matcher.model.y) == at_half
    assert at_half != get_y_init_given_threshold(features, 0.8)


def test_tol_stops_on_the_expected_log_likelihood(cell):
    """A tolerance nothing can miss stops at the first comparison — iteration 2."""
    pairs, features = cell
    matcher = ZeroERMatcher(tol=np.inf, max_iter=50)
    matcher.match_pairs(pairs, features)
    assert matcher.model.converged and matcher.model.n_iter == 2


def test_iteration_limit_is_reported_as_not_converged(cell):
    pairs, features = cell
    matcher = ZeroERMatcher(tol=0.0, max_iter=3)
    matcher.match_pairs(pairs, features)
    assert not matcher.model.converged and matcher.model.n_iter == 3


def test_transitivity_leaves_each_record_at_most_one_match():
    """Under the duplicate-free assumption two predicted matches may not share a
    record. Needs right/4: at right/1 the plain fit already has none to repair."""
    pairs, features = _load(4)

    def shared_endpoints(matcher):
        graph = matcher.match_pairs(pairs, features)
        return sum(degree > 1 for _, degree in graph.degree())

    assert shared_endpoints(ZeroERMatcher()) > 0
    assert shared_endpoints(ZeroERMatcher(transitivity=True)) == 0
