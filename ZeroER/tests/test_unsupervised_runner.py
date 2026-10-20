"""Selection never reads gold; replay preserves settings and auditable results."""
from types import SimpleNamespace
import sqlite3

import numpy as np
import optuna
import pandas as pd
import pytest

import unsupervised as runner


@pytest.fixture
def setup(monkeypatch, tmp_path):
    ids = pd.DataFrame({"ltable_id": ["l0", "l0", "l1", "l1", "l2", "l2", "l3", "l3"],
                        "rtable_id": ["r0", "r1", "r0", "r1", "r2", "r3", "r2", "r3"]})
    record = np.array([.9, .8, .95, .1, .7, .2, .8, .6])
    bank = pd.DataFrame({"name_emb_cos__tiny": [.1, .2, .3, .4, .6, .7, .8, .9],
                         "phone_emb_cos__tiny": [.2, .4, 0, .8, .2, .4, 0, .9],
                         "record_emb_cos__tiny": record})
    present = pd.DataFrame({"name": [True] * 8, "phone": bank.phone_emb_cos__tiny != 0,
                            "record": [True] * 8})
    pairs = np.array([[int(a[1:]), 4 + int(b[1:])] for a, b in ids.to_numpy()])
    cell = dict(ids=ids, pairs=pairs, bank=bank, present=present,
                record_scores=pd.DataFrame({"tiny": record}), attributes=["name", "phone"],
                input_hash="fixture", timings={"loading_seconds": .001})
    monkeypatch.setattr(runner, "load_cell", lambda *args, **kwargs: cell)
    monkeypatch.setattr(runner, "_code_hash", lambda: "code-fixture")
    splits = [dict(train=np.arange(4), validation=np.arange(4, 8), excluded=np.array([], dtype=int),
                   metadata={"fold": 0}),
              dict(train=np.arange(4, 8), validation=np.arange(4), excluded=np.array([], dtype=int),
                   metadata={"fold": 1})]
    monkeypatch.setattr(runner, "record_disjoint_splits", lambda *args, **kwargs: splits)
    fitted = []

    class Matcher:
        def __init__(self, **config):
            self.config = config

        def match_pairs(self, pairs, features, present=None):
            fitted.append(dict(self.config))
            if self.config["c_bay"] == -.5 and features.iloc[:, 0].mean() > .5:
                raise RuntimeError("bad second fold")
            width = features.shape[1]
            self.model = SimpleNamespace(y=np.array([0, 1]), converged=True, n_iter=3,
                kappas=np.zeros(width), Mu_M=np.ones(width), Mu_U=np.zeros(width),
                score_components=self.score)

        def score(self, features, present=None):
            p = np.clip(features[:, 0], .01, .99)
            return dict(log_match=np.log(p), log_nonmatch=np.log1p(-p),
                        p_match=p, llr=np.log(p) - np.log1p(-p),
                        log_density=np.full(len(p), -self.config["c_bay"]),
                        n_blocks=present.sum(axis=1).to_numpy(), pi=.3)

    monkeypatch.setattr(runner, "ZeroERMatcher", Matcher)
    common = dict(init_threshold=.5, init_statistic="sum", missing="ignore", hard=False,
                  transitivity=False, max_iter=20, tol=1e-5, pi_M=None)
    configs = {"good": dict(common, c_bay=.1), "worse": dict(common, c_bay=.3),
               "partial-failure": dict(common, c_bay=-.5)}
    args = dict(dataset="fixture", side="right", k=2,
                panel=dict(name="single:tiny", models=["tiny"], fusion="separate"),
                configs=configs, split_spec=dict(n_splits=2, seed=0),
                storage="sqlite:///" + str(tmp_path / "new.db"))
    return args, cell, fitted


def test_search_is_label_free_failed_folds_cannot_win_and_resume_skips_fits(setup, monkeypatch):
    args, cell, fitted = setup
    monkeypatch.setattr(runner, "_load_gold", lambda *args: pytest.fail("search opened gold"))
    study = runner.run_search(**args)
    assert len(study.trials) == 3
    assert study.best_value == pytest.approx(.1)
    failed = study.trials[2]
    assert failed.state == optuna.trial.TrialState.FAIL
    assert "error" not in failed.user_attrs["folds"][0]
    assert "error" in failed.user_attrs["folds"][1]
    assert failed.user_attrs["n_errors"] == 1
    assert study.best_trial.params == args["configs"]["good"]
    count = len(fitted)
    resumed = runner.run_search(**args)
    assert len(resumed.trials) == 3 and len(fitted) == count


def test_changed_protocol_and_mixed_missingness_are_rejected(setup, monkeypatch):
    args, _, _ = setup
    runner.run_search(**args)
    with pytest.raises(ValueError, match="protocol"):
        runner.run_search(**dict(args, split_spec=dict(n_splits=2, seed=1)))
    configs = {"zero": dict(args["configs"]["good"], missing="zero"),
               "ignore": args["configs"]["good"]}
    with pytest.raises(ValueError, match="missingness"):
        runner.run_search(**dict(args, configs=configs))
    monkeypatch.setattr(runner, "_code_hash", lambda: "changed-code")
    with pytest.raises(ValueError, match="protocol"):
        runner.run_search(**args)


def test_refit_preserves_selected_parameters_and_persists_curves_and_baselines(setup, monkeypatch, tmp_path):
    args, cell, fitted = setup
    study = runner.run_search(**args)
    gold = np.array([False, True, True, False, False, True, True, False])
    gold_reads = []

    def load_gold(dataset, ids):
        gold_reads.append(dataset)
        assert fitted[-1] == study.best_trial.params
        return gold, 5  # one true pair was missed by blocking

    monkeypatch.setattr(runner, "_load_gold", load_gold)
    thresholds = dict(llr=[-1, 0, 1], posterior=[.5], cosine=[.5, .8])
    full = runner.run_refit(study.study_name, args["storage"], thresholds, tmp_path)
    trial = full.trials[0]
    assert trial.params == study.best_trial.params
    assert trial.value == study.best_value
    assert trial.user_attrs["source_trial"] == study.best_trial.number
    assert trial.user_attrs["n_gold"] == 5 and trial.user_attrs["tp_blocking"] == 4
    scores = pd.read_parquet(trial.user_attrs["scores_path"])
    assert {"llr", "p_match", "log_density", "eligible"}.issubset(scores)
    baseline = next(row for row in trial.user_attrs["decisions"]
                    if row["score"] == "record_cos:tiny" and row["rule"] == "pure_transitivity")
    assert baseline["predicted"] == 4 and baseline["tp"] == 4
    assert baseline["recall"] == .8
    curve = runner.load_threshold_curve(args["storage"], full.study_name, 0, "llr", "threshold")
    assert curve.iloc[0].threshold == "+inf" and curve.iloc[0].predicted == 0
    with sqlite3.connect(runner._database(args["storage"])) as connection:
        count = connection.execute("SELECT COUNT(*) FROM unsupervised_decisions").fetchone()[0]
        assert count == len(trial.user_attrs["decisions"])
        assert connection.execute("SELECT encoding FROM unsupervised_thresholds LIMIT 1").fetchone()[0] == "json+zlib"
    fit_count = len(fitted)
    rerun = runner.run_refit(study, args["storage"], thresholds, tmp_path)
    assert len(rerun.trials) == 1 and len(fitted) == fit_count and len(gold_reads) == 1
    with pytest.raises(ValueError, match="threshold policy"):
        runner.run_refit(study, args["storage"], dict(thresholds, llr=[0]), tmp_path)


def test_legacy_databases_are_rejected_and_no_finite_trial_cannot_refit(setup, tmp_path):
    args, _, _ = setup
    with pytest.raises(ValueError, match="legacy"):
        runner.run_search(**dict(args, storage=runner.MATCHING_STORAGE))
    study = optuna.create_study(study_name="empty", storage=args["storage"], direction="minimize")
    with pytest.raises(ValueError, match="No complete"):
        runner.run_refit(study, args["storage"], {}, tmp_path)


def test_ablation_is_separate_from_selection(setup, monkeypatch, tmp_path):
    args, cell, fitted = setup
    study = runner.run_search(**args)
    monkeypatch.setattr(runner, "_load_gold", lambda *args: (np.zeros(len(cell["ids"]), dtype=bool), 0))
    full = runner.run_refit(study, args["storage"], dict(llr=[0], posterior=[.5], cosine=[.5]),
                           tmp_path, config_override={"missing": "zero"}, selection="zero_fill")
    assert full.user_attrs["role"] == "ablation"
    assert full.trials[0].params["missing"] == "zero"
    assert study.best_trial.params["missing"] == "ignore"
    assert full.trials[0].value == study.best_value


def test_empty_fold_is_recorded_as_failure_and_not_selected(setup, monkeypatch):
    args, cell, _ = setup
    monkeypatch.setattr(runner, "record_disjoint_splits", lambda *args, **kwargs: [
        dict(train=np.arange(8), validation=np.array([], dtype=int), excluded=np.array([], dtype=int),
             metadata={"fold": 0})])
    study = runner.run_search(**args)
    assert all(t.state == optuna.trial.TrialState.FAIL for t in study.trials)
    assert all("Empty training or validation" in t.user_attrs["folds"][0]["error"] for t in study.trials)


def test_no_feature_rows_abstain_even_with_positive_record_scores(setup, monkeypatch, tmp_path):
    args, cell, _ = setup
    cell["present"].iloc[0] = False
    study = runner.run_search(**args)
    monkeypatch.setattr(runner, "_load_gold", lambda *args: (np.zeros(len(cell["ids"]), dtype=bool), 0))
    full = runner.run_refit(study, args["storage"], dict(llr=[-100], posterior=[0], cosine=[-1]), tmp_path)
    rows = full.trials[0].user_attrs["decisions"]
    assert full.trials[0].user_attrs["n_no_features"] == 1
    for score in ("llr", "posterior", "record_cos:tiny", "attribute_cos:mean"):
        fixed = next(row for row in rows if row["score"] == score and row["rule"] == "fixed"
                     and row["decoder"] == "threshold")
        assert fixed["predicted"] == 7
