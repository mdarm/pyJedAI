"""Explicit EM seeds and held-out mixture densities, independent of gold labels."""
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from scipy.stats import multivariate_normal

from matcher import ZeroERMatcher
from module import ZeroerModel, initial_labels


@pytest.fixture
def model():
    """Known block-diagonal parameters, without an EM fit in scorer tests."""
    fitted = ZeroerModel.__new__(ZeroerModel)
    fitted._num_cols = 3
    fitted.feature_names = ["a_x", "a_y", "b_x"]
    fitted.group_name_2_col_indices = {"a": [0, 1], "b": [2]}
    fitted.observed = None
    fitted.pi_M = .3
    fitted.Mu_M = np.array([.8, .7, .9])
    fitted.Mu_U = np.array([.1, .2, .3])
    fitted.Cov_M = np.array([[.2, .03, 0.], [.03, .1, 0.], [0., 0., .4]])
    fitted.Cov_U = np.array([[.3, -.02, 0.], [-.02, .2, 0.], [0., 0., .1]])
    return fitted


def test_final_mixture_scores_match_scipy(model):
    x = np.array([[.8, .6, .9], [.1, .2, .3], [.3, .7, .1], [-.2, .5, 1.2]])
    actual = model.score_components(x)
    log_m = multivariate_normal.logpdf(x, model.Mu_M, model.Cov_M + np.eye(3) * 1e-8)
    log_u = multivariate_normal.logpdf(x, model.Mu_U, model.Cov_U + np.eye(3) * 1e-8)
    density = model.pi_M * np.exp(log_m) + (1 - model.pi_M) * np.exp(log_u)
    np.testing.assert_allclose(actual["log_match"], log_m)
    np.testing.assert_allclose(actual["log_nonmatch"], log_u)
    np.testing.assert_allclose(actual["log_density"], np.log(density))
    np.testing.assert_allclose(actual["llr"], log_m - log_u)
    np.testing.assert_allclose(actual["p_match"], model.pi_M * np.exp(log_m) / density)
    np.testing.assert_allclose(logit(actual["p_match"]), logit(actual["pi"]) + actual["llr"])
    np.testing.assert_array_equal(actual["n_blocks"], np.full(len(x), 2))
    np.testing.assert_allclose(model.score_samples(x), actual["log_density"])
    assert actual["pi"] == model.pi_M


def test_missing_blocks_are_marginalised_and_absent_values_do_not_matter(model):
    model.observed = np.ones((1, 3), dtype=bool)
    x = np.array([[.8, .6, .9], [.1, .2, .3], [.3, .7, .1], [-.2, .5, 1.2]])
    present = {"a": np.array([True, False, True, False]),
               "b": np.array([True, True, False, False])}
    expected = []
    for mu, cov in ((model.Mu_M, model.Cov_M), (model.Mu_U, model.Cov_U)):
        logs = []
        for i, row in enumerate(x):
            cols = [j for block, indices in model.group_name_2_col_indices.items()
                    if present[block][i] for j in indices]
            logs.append(multivariate_normal.logpdf(
                row[cols], mu[cols], cov[np.ix_(cols, cols)] + np.eye(len(cols)) * 1e-8)
                if cols else 0.)
        expected.append(np.array(logs))
    actual = model.score_components(x, present)
    np.testing.assert_allclose(actual["log_match"], expected[0])
    np.testing.assert_allclose(actual["log_nonmatch"], expected[1])
    np.testing.assert_allclose(actual["log_density"], np.logaddexp(
        np.log(model.pi_M) + expected[0], np.log1p(-model.pi_M) + expected[1]), atol=1e-15)
    np.testing.assert_array_equal(actual["n_blocks"], [2, 1, 1, 0])
    assert actual["llr"][-1] == actual["log_density"][-1] == 0.
    assert actual["p_match"][-1] == model.pi_M
    changed = x.copy()
    for block, cols in model.group_name_2_col_indices.items():
        changed[np.ix_(~present[block], cols)] = 999.
    other = model.score_components(changed, present)
    for key in actual:
        np.testing.assert_array_equal(actual[key], other[key])


def test_scoring_is_repeatable_and_does_not_change_fitted_state(model):
    # Also exercise the E-step's repair for a negative covariance eigenvalue.
    model.Cov_M[0, 0] = -.1
    before = pickle.dumps(model)
    x = np.ones((5, 3))
    first = model.score_components(x)
    second = model.score_components(x)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert pickle.dumps(model) == before


def test_extreme_evidence_keeps_raw_llr_and_stable_probabilities(model):
    model.Mu_M, model.Mu_U = np.ones(3), -np.ones(3)
    model.Cov_M = model.Cov_U = np.eye(3)
    x = np.array([[1e3] * 3, [-1e3] * 3])
    scored = model.score_components(x)
    assert scored["llr"][0] > 500 and scored["llr"][1] < -500
    np.testing.assert_array_equal(scored["p_match"], [1., 0.])
    np.testing.assert_array_equal(scored["p_match"], expit(logit(scored["pi"]) + scored["llr"]))
    assert np.isfinite(scored["log_density"]).all()


@pytest.mark.parametrize("prior", [0., 1.])
def test_boundary_priors_score_as_a_single_component(model, prior):
    model.pi_M = prior
    scored = model.score_components(np.ones((2, 3)))
    np.testing.assert_array_equal(scored["p_match"], np.full(2, prior))
    np.testing.assert_allclose(scored["log_density"],
                               scored["log_match" if prior else "log_nonmatch"])


def test_scoring_shapes_include_one_row_and_empty_batches(model):
    for count in (0, 1, 7):
        scored = model.score_components(np.ones((count, 3)))
        assert all(value.shape == (count,) for key, value in scored.items() if key != "pi")
    model.observed = np.ones((1, 3), dtype=bool)
    assert model.score_samples(np.empty((0, 3)), {"a": [], "b": []}).shape == (0,)


def test_zero_fill_scoring_retains_absent_feature_evidence(model):
    x = np.zeros((2, 3))
    plain = model.score_components(x)
    masked = model.score_components(x, {"a": [False, False], "b": [False, False]})
    for key in plain:
        np.testing.assert_array_equal(plain[key], masked[key])


@pytest.mark.parametrize("x", [np.zeros((2, 2)), np.zeros(3), [[np.nan, 0., 1.]],
                                [[np.inf, 0., 1.]]])
def test_scoring_rejects_wrong_width_and_nonfinite_features(model, x):
    with pytest.raises(ValueError, match="feature width|finite"):
        model.score_components(x)


@pytest.mark.parametrize("present", [None, {"a": [True, True]},
    {"a": [True], "b": [True, True]}, {"a": [True, True], "b": [True, np.nan]}])
def test_missing_aware_scoring_requires_aligned_boolean_masks(model, present):
    model.observed = np.ones((1, 3), dtype=bool)
    with pytest.raises(ValueError, match="present=|presence mask"):
        model.score_components(np.ones((2, 3)), present)


def _seed_features():
    return pd.DataFrame({"a_x": [.9, .8, .2, .1, .9, .8, .2, .1],
                         "b_x": [0., 0., 0., 0., .9, .8, .2, .1]})


def test_supplied_seed_controls_both_class_parameters_and_initial_prior():
    features = _seed_features()
    seed = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    model = ZeroerModel(features.values, features.columns, seed, None, .01)
    np.testing.assert_array_equal(model.y, seed)
    np.testing.assert_array_equal(model.y_step, seed)
    assert model.pi_M == seed.mean() == .25
    np.testing.assert_allclose(model.Mu_M, features.values[seed == 1].mean(axis=0))
    np.testing.assert_allclose(model.Mu_U, features.values[seed == 0].mean(axis=0))
    seed[:] = 0
    assert model.y.sum() == 2                    # caller mutations cannot change the fit


def test_omitted_seed_uses_the_selected_initializer():
    features = _seed_features()
    model = ZeroerModel(features.values, features.columns, None, None, .01, init_threshold=.5)
    expected = initial_labels(features.values, features.columns, threshold=.5)
    np.testing.assert_array_equal(model.y, expected)
    assert model.pi_M == np.mean(expected)


def test_masked_seed_is_honoured_even_when_the_likelihood_uses_zero_fill():
    features = _seed_features()
    present = {"a": np.ones(8, dtype=bool), "b": np.array([False] * 4 + [True] * 4)}
    expected = initial_labels(features.values, features.columns, .5, "present_mean", present)
    assert expected != initial_labels(features.values, features.columns, .5, "present_mean")
    matcher = ZeroERMatcher(init_threshold=.5, init_statistic="present_mean", max_iter=1)
    matcher.match_pairs([(i, 8 + i) for i in range(8)], features, present=present)
    np.testing.assert_array_equal(matcher.model.y, expected)
    assert matcher.model.observed is None


def test_matcher_passes_explicit_seed_without_reinitializing(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("an explicit seed must not be recomputed")
    monkeypatch.setattr("matcher.initial_labels", forbidden)
    monkeypatch.setattr("module.initial_labels", forbidden)
    seed = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    matcher = ZeroERMatcher(max_iter=1)
    matcher.match_pairs([(i, 8 + i) for i in range(8)], _seed_features(), y_init=seed)
    np.testing.assert_array_equal(matcher.model.y, seed)


@pytest.mark.parametrize("seed, message", [([0, 1], "one value per"),
    ([0, 0, 0, 0, 1, 1, 1, 2], "binary"), ([0] * 8, "two rows"),
    ([0] * 7 + [1], "two rows")])
def test_invalid_seeds_fail_with_an_explanation(seed, message):
    features = _seed_features()
    with pytest.raises(ValueError, match=message):
        ZeroerModel(features.values, features.columns, seed, None, .01)
