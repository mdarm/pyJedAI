"""Original baseline replay retains cached candidates and legacy EM scores."""
from types import SimpleNamespace
import sqlite3

import numpy as np
import optuna
import pandas as pd
import pytest

from zeroer.config import MATCHING_STORAGE
from zeroer.experiments.unsupervised.runner import load_threshold_curve
import zeroer.experiments.unsupervised.baselines as baseline


@pytest.fixture
def setup(monkeypatch, tmp_path):
    ids = pd.DataFrame({"ltable_id": ["l0", "l1", "l0", "l1"],
                        "rtable_id": ["r0", "r0", "r1", "r1"]})
    features = pd.DataFrame({"name_name_lev_sim": [.1, .2, .8, .9],
                             "constant_x": [1.] * 4, "class_class_exm": [0, 1, 1, 0]})
    pd.concat([ids, features], axis=1).to_csv(tmp_path / "fixture__right_k2.csv.gz", index=False)
    pairs = np.array([[0, 2], [1, 2], [0, 3], [1, 3]])
    monkeypatch.setattr(baseline, "FEATURES_DIR", str(tmp_path))
    monkeypatch.setattr(baseline, "LADDER", {"fixture": [("right", 2), ("right", 1)]})
    monkeypatch.setattr(baseline, "DROP_GROUPS", {"fixture": ["class"]})
    monkeypatch.setattr(baseline, "_code_hash", lambda: "baseline-code")

    def candidates(dataset, side, k):
        keep = np.arange(4) % 2 < k
        return ids[keep].reset_index(drop=True), pairs[keep], None, 2

    monkeypatch.setattr(baseline, "_candidate_input", candidates)
    fitted, gold_reads, mode = [], [], {"fail_code": False}
    posterior = np.array([.1, .5, .8, .4])

    class Matcher:
        def __init__(self, **config):
            self.config = config

        def match_pairs(self, passed_pairs, passed_features):
            fitted.append(dict(self.config))
            np.testing.assert_array_equal(passed_pairs, pairs)
            assert list(passed_features) == ["name_name_lev_sim"]
            if mode["fail_code"] and self.config["max_iter"] == 40:
                raise ValueError("synthetic empty seed class")
            self.posteriors = posterior.copy()
            self.model = SimpleNamespace(y=np.array([0, 0, 1, 1]), n_iter=4,
                converged=True, pi_M=.99,
                score_components=lambda *a, **k: pytest.fail("baseline refreshed its legacy scores"))

    monkeypatch.setattr(baseline, "ZeroERMatcher", Matcher)

    def gold(dataset, passed_ids):
        assert len(fitted) > len(gold_reads), "gold must follow fitting"
        pd.testing.assert_frame_equal(passed_ids, ids)
        gold_reads.append(dataset)
        return np.array([False, True, True, False]), 3  # one match was blocked out

    monkeypatch.setattr(baseline, "_load_gold", gold)
    args = dict(dataset="fixture", side="right", k=2,
                storage="sqlite:///" + str(tmp_path / "new.db"), output_dir=tmp_path / "scores",
                thresholds=[.1, .25, .5, .75, .9])
    return args, fitted, gold_reads, posterior, mode


def test_cached_loader_uses_same_pair_order_and_original_feature_policy(setup, monkeypatch):
    cell = baseline._load_cell("fixture", "right", 1)
    assert cell["ids"].to_dict("list") == {"ltable_id": ["l0", "l0"], "rtable_id": ["r0", "r1"]}
    np.testing.assert_allclose(cell["features"].to_numpy(), [[.1], [.8]])
    monkeypatch.setattr(baseline, "_candidate_input", lambda *a: (
        cell["ids"].iloc[::-1].reset_index(drop=True), cell["pairs"], None, 2))
    with pytest.raises(ValueError, match="identical candidate"):
        baseline._load_cell("fixture", "right", 1)


def test_larger_cell_requires_its_exact_cache_and_does_not_change_old_slices(setup, monkeypatch, tmp_path):
    original_candidates = baseline._candidate_input
    with pytest.raises(ValueError, match="cached candidate range"):
        baseline._load_cell("fixture", "right", 3)
    ids = pd.DataFrame({"ltable_id": ["l0", "l1", "l2"] * 2,
                        "rtable_id": ["r0"] * 3 + ["r1"] * 3})
    stored = ids.assign(name_name_lev_sim=[.1, .2, .3, .8, .9, .7])
    stored.to_csv(tmp_path / "fixture__right_k3.csv.gz", index=False)
    pairs = np.array([[i % 3, 3 + i // 3] for i in range(6)])
    monkeypatch.setattr(baseline, "_candidate_input", lambda dataset, side, k:
                        (ids, pairs, None, 3) if k == 3 else original_candidates(dataset, side, k))
    larger = baseline._load_cell("fixture", "right", 3)
    pd.testing.assert_frame_equal(larger["ids"], ids)
    np.testing.assert_allclose(larger["features"].iloc[:, 0], stored.name_name_lev_sim)
    smaller = baseline._load_cell("fixture", "right", 1)
    assert smaller["source"].endswith("fixture__right_k2.csv.gz")
    np.testing.assert_allclose(smaller["features"].iloc[:, 0], [.1, .8])


def test_fixed_baselines_keep_legacy_probabilities_and_report_exact_decisions(setup):
    args, fitted, gold_reads, posterior, _ = setup
    study = baseline.run_baselines(**args)
    assert study.user_attrs["phase"] == "baseline"
    assert study.user_attrs["objective"] == "fixed_configuration_no_selection"
    assert fitted == [baseline.BASELINES["B-paper"], baseline.BASELINES["B-code"]]
    assert len(gold_reads) == 2
    for trial in study.trials:
        assert trial.value == 0 and trial.params == baseline.BASELINES[trial.user_attrs["baseline"]]
        attrs = trial.user_attrs
        assert attrs["primary_decision"]["threshold"] == .5
        assert not attrs["primary_decision"]["uses_gold"]
        assert attrs["predicted"] == attrs["tp"] == 2
        assert attrs["recall"] == pytest.approx(2 / 3)
        assert attrs["f1"] == pytest.approx(.8)
        assert attrs["score_state"] == "legacy_last_e_step"
        assert attrs["feature_generation_seconds"] is None
        assert attrs["code_hash"] == "baseline-code"
        saved = pd.read_parquet(attrs["scores_path"])
        np.testing.assert_array_equal(saved["p_match"], posterior)
        np.testing.assert_array_equal(saved["original_match"], posterior >= .5)
        assert {row["rule"] for row in attrs["decisions"]} == {
            "fixed", "pure_transitivity", "oracle_threshold"}
        assert all(row["uses_gold"] == (row["rule"] == "oracle_threshold")
                   for row in attrs["decisions"])
        for decoder in ("threshold", "greedy"):
            curve = load_threshold_curve(args["storage"], study.study_name, trial.number,
                                         "legacy_posterior", decoder)
            assert curve.iloc[0].threshold == "+inf"
            assert len(curve) == len(np.unique(posterior)) + 1
    with sqlite3.connect(baseline._database(args["storage"])) as conn:
        count = conn.execute("SELECT COUNT(*) FROM unsupervised_decisions").fetchone()[0]
        assert count == sum(len(t.user_attrs["decisions"]) for t in study.trials)


def test_resume_skips_fits_and_gold_but_changed_policy_is_rejected(setup):
    args, fitted, gold_reads, _, _ = setup
    first = baseline.run_baselines(**args)
    second = baseline.run_baselines(**args)
    assert len(second.trials) == len(first.trials) == len(fitted) == len(gold_reads) == 2
    with pytest.raises(ValueError, match="changed"):
        baseline.run_baselines(**dict(args, thresholds=[.5]))


def test_failed_baseline_is_not_silently_completed_and_retry_is_explicit(setup):
    args, fitted, gold_reads, _, mode = setup
    mode["fail_code"] = True
    first = baseline.run_baselines(**args)
    assert [t.state for t in first.trials] == [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL]
    assert "synthetic empty seed class" in first.trials[1].user_attrs["error"]
    assert len(gold_reads) == 1
    baseline.run_baselines(**args)
    assert len(fitted) == 2
    mode["fail_code"] = False
    retried = baseline.run_baselines(**args, retry_failed=True)
    assert len(retried.trials) == 3 and retried.trials[-1].state == optuna.trial.TrialState.COMPLETE
    assert len(fitted) == 3 and len(gold_reads) == 2


def test_legacy_database_and_unknown_baseline_are_rejected(setup):
    args, *_ = setup
    with pytest.raises(ValueError, match="legacy"):
        baseline.run_baselines(**dict(args, storage=MATCHING_STORAGE))
    with pytest.raises(ValueError, match="B-paper"):
        baseline.run_baselines(**args, baselines=["tuned"])
