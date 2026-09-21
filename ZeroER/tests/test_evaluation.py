"""The reference-ladder runner: its slicing premise, its floor, its resume.

``evaluation`` reads every cell of a (dataset, queried side) curve off one
stored matrix, as the rows with neighbour rank < k. That is only right if the
stored pairs really are grouped by queried record in rank order and k-NN lists
are nested — checked here against the smaller cells that were stored
independently. The rest covers the cosine + one-to-one floor and that a re-run
adds nothing to the study.

Needs the blocker's record vectors for ``fodors_zagats`` (the floor is its
cosine); skips without them.

Run:  pytest tests/test_evaluation.py -v
"""
import os
from os.path import join

import numpy as np
import optuna
import pandas as pd
import pytest

import evaluation
from tuning import run_baselines
from data import RECORD_DIR
from evaluation import load_side, one_to_one, one_to_one_oracle
from config import FEATURES_DIR, MODEL, cell_name

DATASET = "fodors_zagats"
needs_vectors = pytest.mark.skipif(
    not os.path.exists(join(RECORD_DIR, DATASET, MODEL + "_left.npy")),
    reason="no record embeddings for " + MODEL)


@needs_vectors
def test_rank_slice_is_the_smaller_stored_cell():
    side = load_side(DATASET, "right")                      # stored at k = 4
    stored = pd.read_csv(join(FEATURES_DIR, cell_name(DATASET, "right", 1) + ".csv.gz"),
                         dtype={"ltable_id": str, "rtable_id": str})
    keep = side["rank"] < 1
    assert side["ids"][keep].reset_index(drop=True).equals(stored[["ltable_id", "rtable_id"]])
    np.testing.assert_array_equal(side["features"][keep].to_numpy(),
                                  stored[side["features"].columns].to_numpy())


def test_one_to_one_takes_the_best_pair_and_uses_every_record_once():
    left = np.array([0, 0, 1, 1])
    right = np.array([10, 11, 10, 11])
    keep = one_to_one(left, right, np.array([0.9, 0.8, 0.95, 0.1]))
    # (1, 10) wins first and blocks both (0, 10) and (1, 11); (0, 11) is left
    assert keep.tolist() == [False, True, True, False]


def test_one_to_one_oracle_is_capped_where_gold_shares_a_record():
    left = np.array([0, 0, 1])
    right = np.array([10, 11, 12])
    assert one_to_one_oracle(left, right, np.array([True, True, True])) == 2


@needs_vectors
def test_rerun_adds_nothing(tmp_path):
    storage = "sqlite:///" + str(tmp_path / "matching.db")
    configs = {"B-code": dict(c_bay=0.1, init_threshold=0.8, tol=None, max_iter=40,
                              transitivity=False)}

    def trials():
        return optuna.load_study(study_name=DATASET + "_ladder", storage=storage).trials

    run_baselines(configs, [DATASET], rungs_only=True, storage=storage,
                  posteriors_dir=str(tmp_path))
    first = trials()
    assert len(first) == 3 and all(t.value is not None for t in first)   # the 3 rungs
    run_baselines(configs, [DATASET], rungs_only=True, storage=storage,
                  posteriors_dir=str(tmp_path))
    assert len(trials()) == 3


@needs_vectors
def test_the_two_halves_partition_the_queried_records():
    full = load_side(DATASET, "right", cosine=False)
    tune = load_side(DATASET, "right", partition="tune", cosine=False)
    hold = load_side(DATASET, "right", partition="holdout", cosine=False)
    assert len(tune["ids"]) + len(hold["ids"]) == len(full["ids"])
    assert not set(tune["ids"]["rtable_id"]) & set(hold["ids"]["rtable_id"])
    assert tune["n_gold"] + hold["n_gold"] == full["n_gold"]
    assert len(tune["ids"]) % full["k_max"] == 0          # whole candidate lists


def test_a_panel_adds_its_columns_to_the_key_attribute_block():
    from os.path import isdir
    from config import BANK_DIR
    if not isdir(join(BANK_DIR, cell_name(DATASET, "right", 4))):
        pytest.skip("no embedding bank — run python -m prepare.build_embedding_bank")
    loaded = {side: load_side(DATASET, side, partition="tune", cosine=False, bank=True)
              for side in ("right", "left")}
    config = dict(c_bay=0.01, init_threshold=0.5, tol=1e-5, max_iter=200)
    strings = evaluation.fit_rungs(DATASET, loaded, config, attributes=["name"])
    panel = evaluation.fit_rungs(DATASET, loaded, config, attributes=["name"],
                                      panel=dict(models=["minilm6", "mpnet"], fusion="separate"))
    assert [p["n_features"] - s["n_features"] for s, p in zip(strings, panel)] == [2, 2, 2]
    assert all("error" not in row and 0 < row["f1"] <= 1 for row in strings + panel)


def test_cosine_only_uses_exactly_the_requested_bank_columns(monkeypatch):
    from config import BANK_DIR, LADDER
    if not os.path.isdir(join(BANK_DIR, cell_name(DATASET, "right", 4))):
        pytest.skip("no embedding bank")
    loaded = {side: load_side(DATASET, side, partition="tune", cosine=False, bank=True)
              for side in ("right", "left")}
    columns = [f"{attr}_emb_cos__{model}" for attr in ("name", "phone")
               for model in ("minilm6", "mpnet")]
    expected = []
    for side, k in LADDER[DATASET]:
        stored = loaded[side]
        keep = stored["rank"] < k
        expected.append((stored["bank"].loc[keep, columns].reset_index(drop=True),
                         stored["bank_present"][keep].reset_index(drop=True)))
        # Neither Magellan values nor its missing-value masks may enter this fit.
        stored["features"].loc[:, :] = -999
        stored["present"].loc[:, :] = False

    original = evaluation.ZeroERMatcher.match_pairs
    seen = []

    def check_inputs(self, pairs, features, present=None, **kwargs):
        wanted_features, wanted_presence = expected[len(seen)]
        pd.testing.assert_frame_equal(features, wanted_features)
        pd.testing.assert_frame_equal(present, wanted_presence)
        seen.append(features.shape)
        return original(self, pairs, features, present=present, **kwargs)

    monkeypatch.setattr(evaluation.ZeroERMatcher, "match_pairs", check_inputs)
    rows = evaluation.fit_rungs(
        DATASET, loaded, dict(c_bay=.01, init_threshold=.5, tol=1e-5, max_iter=200),
        attributes=["name", "phone"],
        panel=dict(models=["minilm6", "mpnet"], fusion="separate"), include_strings=False)
    assert len(seen) == len(rows) == 3
    assert all(row["n_features"] == 4 and "error" not in row for row in rows)


def test_cosine_only_rejects_an_empty_panel():
    with pytest.raises(ValueError, match="nonempty embedding panel"):
        evaluation.fit_rungs(DATASET, {}, {}, include_strings=False)
