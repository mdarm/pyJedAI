"""Search and replay retain the recorded results and selected parameters."""
from pathlib import Path

import optuna
import pytest

from config import BANK_DIR, MATCHING_STORAGE, STORAGE
from tuning import (EmbeddingObjective, MatchingObjective, replay_selected,
                    run_blocking_sweep, run_diagnosis)

DATASET = "fodors_zagats"
FIXED = dict(tol=1e-5, max_iter=200)
SPACE = {
    "c_bay": (1e-4, 0.2),
    "init_threshold": (0.2, 0.8),
    "init_statistic": ["sum", "present_mean"],
    "pi_M": [None, 0.02, 0.05, 0.1, 0.2, 0.4],
    "hard": [False, True],
}


def assert_same_rungs(actual, expected):
    assert len(actual) == len(expected)
    for current, recorded in zip(actual, expected):
        current = {key: value for key, value in current.items() if key != "seconds"}
        recorded = {key: value for key, value in recorded.items() if key != "seconds"}
        # Older string studies predate the stored feature count.
        assert set(current) - set(recorded) <= {"n_features"}
        assert {key: current[key] for key in recorded} == pytest.approx(recorded)


@pytest.mark.parametrize("experiment,selection", [
    ("match-zero", "target"),
    ("match-missing", "target"),
    ("emb-sweep", "target"),
    ("emb-sweep", "all-20:family"),
])
def test_search_and_replay_match_recorded_fits(experiment, selection, tmp_path, monkeypatch):
    embedding = experiment == "emb-sweep"
    if embedding and not (Path(BANK_DIR) / f"{DATASET}__right_k4").exists():
        pytest.skip("no cached embedding bank")
    tune_name = f"{DATASET}_{experiment}"
    tune = optuna.load_study(study_name=tune_name, storage=MATCHING_STORAGE)
    full = optuna.load_study(study_name=tune_name + "_full", storage=MATCHING_STORAGE)
    recorded = next(trial for trial in full.trials if trial.user_attrs["selection"] == selection)
    source = tune.trials[recorded.user_attrs["source_trial"]]
    missing = None if embedding else ("ignore" if experiment == "match-missing" else "zero")
    attributes = ["name"] if embedding else None

    if embedding:
        objective = EmbeddingObjective(
            DATASET, experiment, SPACE, FIXED,
            models=sorted(key[4:] for key in source.params if key.startswith("use_")),
            attributes=attributes, design={})
    else:
        objective = MatchingObjective(DATASET, experiment, SPACE, FIXED, missing=missing)
    trial = optuna.trial.FixedTrial(source.params, number=source.number)
    assert objective(trial) == pytest.approx(source.value)
    assert trial.params == source.params
    assert trial.user_attrs["config_hash"] == source.user_attrs["config_hash"]
    assert_same_rungs(trial.user_attrs["rungs"], source.user_attrs["rungs"])

    # Only the selected trial is copied. Replay cannot rely on a search design.
    storage = "sqlite:///" + str(tmp_path / "matching.db")
    copied = optuna.create_study(study_name=tune_name, storage=storage, direction="maximize")
    copied.add_trial(source)
    result = replay_selected(DATASET, experiment, {selection: 0}, fixed=FIXED,
                              missing=missing, attributes=attributes, storage=storage)
    replay = result.trials[0]
    assert replay.value == pytest.approx(recorded.value)
    assert replay.params == source.params
    assert replay.distributions == source.distributions
    assert replay.user_attrs["source_trial"] == 0
    assert replay.user_attrs["selection"] == selection
    for scope in ("rungs", "holdout_rungs"):
        assert_same_rungs(replay.user_attrs[scope], recorded.user_attrs[scope])

    def unexpected_load(*args, **kwargs):
        pytest.fail("A completed replay must not load or fit candidates again")

    monkeypatch.setattr("tuning.load_sides", unexpected_load)
    result = replay_selected(DATASET, experiment, {selection: 0}, fixed=FIXED,
                              missing=missing, attributes=attributes, storage=storage)
    assert len(result.trials) == 1


def test_blocking_sweep_retains_counts_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr("tuning.embedding_models", lambda dataset: ["minilm6"])
    storage = "sqlite:///" + str(tmp_path / "blocking.db")
    study = run_blocking_sweep(DATASET, 2, ["right", "left"], storage=storage)
    original = optuna.load_study(study_name=f"{DATASET}_blocking", storage=STORAGE)
    recorded = {(trial.params["queried_side"], trial.params["top_k"]): trial
                for trial in original.trials if trial.params["embedding_model"] == "minilm6"}
    assert len(study.trials) == 4
    for trial in study.trials:
        expected = recorded[trial.params["queried_side"], trial.params["top_k"]]
        assert trial.values == expected.values
        for key in ("tp", "n_gold", "candset_size"):
            assert trial.user_attrs[key] == expected.user_attrs[key]
    assert len(run_blocking_sweep(DATASET, 2, ["right", "left"], storage=storage).trials) == 4


def test_cosine_replay_restores_features_and_all_tuned_settings(tmp_path):
    from data import DATASETS
    from evaluation import evaluate_config, load_sides

    if not (Path(BANK_DIR) / f"{DATASET}__right_k4").exists():
        pytest.skip("no cached embedding bank")
    attributes = DATASETS[DATASET]["attributes"]
    models = ["minilm6", "mpnet"]
    space = dict(SPACE, missing=["zero", "ignore"], transitivity=[False, True],
                 max_iter=[200, 500], tol=[1e-4, 1e-5, 1e-6])
    params = dict(c_bay=.01, init_threshold=.5, init_statistic="present_mean", pi_M=.2,
                  hard=False, missing="ignore", transitivity=True, max_iter=200, tol=1e-4,
                  fusion="separate", use_minilm6=True, use_mpnet=True)
    objective = EmbeddingObjective(DATASET, "emb-cosine-tune", space, {}, models,
                                    attributes, {}, include_strings=False)
    trial = optuna.trial.FixedTrial(params)
    value = objective(trial)
    assert not trial.user_attrs["include_strings"]
    assert trial.user_attrs["attributes"] == attributes
    assert trial.user_attrs["n_errors"] == 0
    assert all(row["n_features"] == len(models) * len(attributes)
               for row in trial.user_attrs["rungs"])

    storage = "sqlite:///" + str(tmp_path / "cosine.db")
    study = optuna.create_study(study_name=f"{DATASET}_emb-cosine-tune", storage=storage,
                                direction="maximize")
    study.add_trial(optuna.trial.create_trial(params=trial.params,
        distributions=trial.distributions, value=value, user_attrs=trial.user_attrs))
    # No feature flags or attribute list supplied: replay must recover them.
    full = replay_selected(DATASET, "emb-cosine-tune", {"target": 0}, {}, storage=storage)
    actual = full.trials[0]
    config = {k: v for k, v in params.items() if k != "fusion" and not k.startswith("use_")}
    expected, attrs = evaluate_config(
        DATASET, load_sides(DATASET, bank=True), config, attributes=attributes,
        panel=dict(models=models, fusion="separate"), include_strings=False,
        holdout=load_sides(DATASET, partition="holdout", bank=True))
    assert actual.value == pytest.approx(expected)
    assert actual.params == params
    assert not actual.user_attrs["include_strings"]
    for scope in ("rungs", "holdout_rungs"):
        assert_same_rungs(actual.user_attrs[scope], attrs[scope])


def test_diagnosis_retains_counts_and_resumes(tmp_path):
    import pandas as pd

    configs = {"B-paper": dict(c_bay=0.01, init_threshold=0.5, tol=1e-5,
                                max_iter=200, transitivity=False)}
    cells = {"dblp_acm": [("left", 2)]}
    storage = "sqlite:///" + str(tmp_path / "diagnosis.db")
    run_diagnosis(configs, cells, ["zero_fill"], {}, storage=storage, out_dir=str(tmp_path))
    original = optuna.load_study(study_name="dblp_acm_collapse", storage=MATCHING_STORAGE)
    expected = next(trial for trial in original.trials
                    if trial.user_attrs["config"] == "B-paper"
                    and trial.params["treatment"] == "zero_fill")
    study = optuna.load_study(study_name="dblp_acm_collapse", storage=storage)
    assert study.trials[0].value == expected.value
    for key in ("tp", "predicted", "tp_blocking", "candset_size", "n_gold"):
        assert study.trials[0].user_attrs[key] == expected.user_attrs[key]
    paths = [tmp_path / name for name in ("blocks.csv", "presence_confusion.csv")]
    before = [pd.read_csv(path) for path in paths]
    run_diagnosis(configs, cells, ["zero_fill"], {}, storage=storage, out_dir=str(tmp_path))
    assert len(study.trials) == 1
    for path, frame in zip(paths, before):
        pd.testing.assert_frame_equal(pd.read_csv(path), frame)
