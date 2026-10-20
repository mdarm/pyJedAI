"""Selected fits are shared across cuts and completed timing repeats resume."""
import json
import sqlite3

import numpy as np
import optuna
import pandas as pd
import pytest

import unsupervised_timing as timing


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    storage = "sqlite:///" + str(tmp_path / "timing.db")
    study = optuna.create_study(study_name="selected_full", storage=storage)
    panel = dict(models=["minilm6"], attributes=["name"], metrics=["cos"], fusion="separate")
    spec = dict(dataset="fixture", side="right", k=2, panel=panel,
                input_hash="inputs", code_hash="code")
    for key, value in dict(phase="full", spec=spec, refit_hash="refit",
        policy=dict(thresholds=dict(llr=[-1, 0], posterior=[.5], cosine=[.5]))).items():
        study.set_user_attr(key, value)
    study.add_trial(optuna.trial.create_trial(value=1, user_attrs={"source_study": "source", "source_trial": 2}))
    cell = dict(ids=pd.DataFrame({"ltable_id": ["a", "a", "b", "b"],
                                  "rtable_id": ["c", "d", "c", "d"]}),
                pairs=np.array([[0, 2], [0, 3], [1, 2], [1, 3]]),
                bank=pd.DataFrame({"name_emb_cos__minilm6": [.1, .3, .6, .9],
                                   "record_emb_cos__minilm6": [.2, .4, .5, .8]}),
                present=pd.DataFrame({"name": [True] * 4, "record": [True] * 4}),
                record_scores=pd.DataFrame({"minilm6": [.2, .4, .5, .8]}),
                attributes=["name"], input_hash="inputs")
    calls = dict(load=0, fit=0, fresh=[], prepare=0)

    def load(**kwargs):
        calls["load"] += 1
        return cell

    def fresh(repeats, **kwargs):
        calls["fresh"].append(repeats)
        return dict(input_hash="inputs", max_absolute_difference=0,
                    repeats=[dict(loading_seconds=1, similarity_seconds=2,
                                  preprocessing_seconds=3) for _ in range(repeats)])

    def prepare(*args):
        calls["prepare"] += 1
        return cell["bank"], cell["bank"], 11

    def fit(*args):
        calls["fit"] += 1
        return None, dict(llr=np.array([-2, 0, 1, 3]), p_match=np.array([.1, .5, .7, .9])), \
            dict(fit_seconds=5, score_seconds=7)

    monkeypatch.setattr(timing, "_code_hash", lambda: "code")
    monkeypatch.setattr(timing, "_timing_code_hash", lambda: "timing-code")
    monkeypatch.setattr(timing, "load_cell", load)
    monkeypatch.setattr(timing, "benchmark_similarity", fresh)
    monkeypatch.setattr(timing, "_prepare", prepare)
    monkeypatch.setattr(timing, "_fit", fit)
    return dict(storage=storage, study_name=study.study_name), calls


def test_repeat_timings_are_queryable_share_fits_and_resume(benchmark):
    args, calls = benchmark
    rows = timing.benchmark_study(**args, repeats=2)
    assert len(rows) == 34
    assert calls == dict(load=1, fit=2, fresh=[2], prepare=2)
    assert set(rows.score_name) == {"llr", "posterior", "record_cos:minilm6",
                                     "record_cos:mean", "attribute_cos:mean"}
    matching = rows[["similarity_seconds", "preprocessing_seconds", "fit_seconds",
                     "score_seconds", "decode_seconds"]].sum(axis=1)
    np.testing.assert_allclose(rows.matching_seconds, matching)
    np.testing.assert_allclose(rows.total_seconds, matching + rows.loading_seconds)
    baseline = rows[rows.score_name.str.contains("cos")]
    assert (baseline.fit_seconds == 0).all()
    assert (baseline.preprocessing_seconds == 3).all()
    for value in baseline.metadata_json:
        metadata = json.loads(value)
        assert metadata["cost_scope"] == "panel-wide upper-bound, shared feature work"
        assert metadata["uses_gold"] is False
    again = timing.benchmark_study(**args, repeats=2)
    pd.testing.assert_frame_equal(rows, again)
    assert calls == dict(load=1, fit=2, fresh=[2], prepare=2)
    extended = timing.benchmark_study(**args, repeats=3)
    assert len(extended) == 51
    assert calls == dict(load=2, fit=3, fresh=[2, 1], prepare=3)
    with sqlite3.connect(timing._database(args["storage"])) as connection:
        assert connection.execute("SELECT COUNT(*) FROM unsupervised_timing_runs").fetchone()[0] == 3


def test_changed_timing_code_cannot_mix_with_previous_measurements(benchmark, monkeypatch):
    args, calls = benchmark
    timing.benchmark_study(**args, repeats=1)
    monkeypatch.setattr(timing, "_timing_code_hash", lambda: "different")
    with pytest.raises(ValueError, match="protocol or source refit changed"):
        timing.benchmark_study(**args, repeats=2)
    assert calls["fit"] == 1


def test_interrupted_repeat_leaves_completed_repeats_resumable(benchmark, monkeypatch):
    args, calls = benchmark
    original = timing._fit

    def fail_second(*args):
        if calls["fit"] == 1:
            raise RuntimeError("interrupted")
        return original(*args)

    monkeypatch.setattr(timing, "_fit", fail_second)
    with pytest.raises(RuntimeError, match="interrupted"):
        timing.benchmark_study(**args, repeats=2)
    with sqlite3.connect(timing._database(args["storage"])) as connection:
        assert connection.execute("SELECT COUNT(*) FROM unsupervised_timing_runs").fetchone()[0] == 1
    monkeypatch.setattr(timing, "_fit", original)
    assert len(timing.benchmark_study(**args, repeats=2)) == 34
    assert calls["fit"] == 2 and calls["fresh"] == [2, 1]
