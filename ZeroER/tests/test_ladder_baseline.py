"""The reference-ladder runner: its slicing premise, its floor, its resume.

``ladder_baseline`` reads every cell of a (dataset, queried side) curve off one
stored matrix, as the rows with neighbour rank < k. That is only right if the
stored pairs really are grouped by queried record in rank order and k-NN lists
are nested — checked here against the smaller cells that were stored
independently. The rest covers the cosine + one-to-one floor and that a re-run
adds nothing to the study.

Needs the blocker's record vectors for ``fodors_zagats`` (the floor is its
cosine); skips without them.

Run:  pytest tests/test_ladder_baseline.py -v
"""
import os
from os.path import join

import numpy as np
import optuna
import pandas as pd
import pytest

import ladder_baseline
from data import RECORD_DIR
from ladder_baseline import load_side, one_to_one, one_to_one_oracle
from reference_features import FEATURES_DIR, MODEL, cell_name

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
def test_rerun_adds_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder_baseline, "POSTERIORS_DIR", str(tmp_path))
    storage = "sqlite:///" + str(tmp_path / "matching.db")
    configs = {"B-code": ladder_baseline.CONFIGS["B-code"]}

    def trials():
        return optuna.load_study(study_name=DATASET + "_ladder", storage=storage).trials

    ladder_baseline.run(configs, [DATASET], rungs_only=True, storage=storage)
    first = trials()
    assert len(first) == 3 and all(t.value is not None for t in first)   # the 3 rungs
    ladder_baseline.run(configs, [DATASET], rungs_only=True, storage=storage)
    assert len(trials()) == 3
