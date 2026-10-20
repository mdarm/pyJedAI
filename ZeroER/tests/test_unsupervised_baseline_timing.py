"""Fresh baseline timing preserves per-table types and refuses feature drift."""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

import zeroer.experiments.unsupervised.baseline_timing as timing


@pytest.fixture
def setup(monkeypatch, tmp_path):
    left = pd.DataFrame({"id": ["l0", "l1"], "year": [1995, 1996]})
    right = pd.DataFrame({"id": ["r0", "r1"], "year": [1995., np.nan]})
    ids = pd.DataFrame({"ltable_id": ["l0", "l1", "l0", "l1"],
                        "rtable_id": ["r0", "r0", "r1", "r1"]})
    pairs = np.array([[0, 2], [1, 2], [0, 3], [1, 3]])
    # Genuine separate-table values: str(1995) differs from str(1995.0).
    features = pd.DataFrame({"year_year_exm": [1., 0., 0., 0.],
                             "year_year_anm": [1., 1 - 1 / 1996, 0., 0.],
                             "year_year_lev_sim": [2 / 3, .5, 0., 0.]})
    source = tmp_path / "reference.csv.gz"
    pd.concat([ids, features], axis=1).to_csv(source, index=False)
    cell = dict(ids=ids, pairs=pairs, features=features,
                input_hash="cached-input", source=str(source))
    monkeypatch.setattr(timing, "_load_cell", lambda *args: cell)
    monkeypatch.setattr(timing, "load_reference_tables", lambda *args: (left.copy(), right.copy()))
    monkeypatch.setattr(timing, "LADDER", {"fixture": [("right", 2)]})
    monkeypatch.setattr(timing, "_code_hash", lambda: "baseline-timing-code")
    monkeypatch.setattr("zeroer.experiments.unsupervised.runner._load_gold", lambda *a: pytest.fail("timing read gold"))
    calls = dict(build=0, fit=[])
    original = timing.build_zeroer_features

    def build(passed_pairs, entities, attributes, **kwargs):
        calls["build"] += 1
        np.testing.assert_array_equal(passed_pairs, pairs)
        assert kwargs["n_jobs"] == 1 and not kwargs["drop_zero_variance"]
        assert isinstance(entities.year.iloc[0], int)
        assert isinstance(entities.year.iloc[2], float)
        return original(passed_pairs, entities, attributes, **kwargs)

    monkeypatch.setattr(timing, "build_zeroer_features", build)

    class Matcher:
        def __init__(self, **config):
            self.config = config

        def match_pairs(self, passed_pairs, passed_features):
            calls["fit"].append(self.config)
            np.testing.assert_allclose(passed_features.to_numpy(), features.to_numpy(), atol=1e-15)
            self.posteriors = np.array([.2, .5, .9, .4])

    monkeypatch.setattr(timing, "ZeroERMatcher", Matcher)
    return dict(dataset="fixture", side="right", k=2,
                storage="sqlite:///" + str(tmp_path / "new.db")), calls


def test_fresh_features_and_two_fits_are_shared_across_decisions_and_resume(setup):
    args, calls = setup
    rows = timing.benchmark_baseline_cell(**args)
    assert len(rows) == 22 and calls["build"] == 1
    assert calls["fit"] == list(timing.BASELINES.values())
    assert set(rows.baseline) == {"B-code", "B-paper"}
    np.testing.assert_allclose(rows.matching_seconds, rows[["similarity_seconds",
        "preprocessing_seconds", "fit_seconds", "decode_seconds"]].sum(axis=1))
    np.testing.assert_allclose(rows.total_seconds, rows.matching_seconds + rows.loading_seconds)
    for encoded in rows.metadata_json:
        metadata = json.loads(encoded)
        assert metadata["max_absolute_difference"] <= 1e-15
        assert metadata["uses_gold"] is False
        assert metadata["dtype_policy"] == "preserve_per_side_scalars"
        assert metadata["score_state"] == "legacy_last_e_step"
        assert metadata["shared_similarity_across_baselines"]
    again = timing.benchmark_baseline_cell(**args)
    pd.testing.assert_frame_equal(rows, again)
    assert calls["build"] == 1 and len(calls["fit"]) == 2
    extended = timing.benchmark_baseline_cell(**args, repeats=2)
    assert len(extended) == 44 and calls["build"] == 2 and len(calls["fit"]) == 4
    with sqlite3.connect(timing._database(args["storage"])) as connection:
        assert connection.execute("SELECT COUNT(*) FROM unsupervised_baseline_timing_runs").fetchone()[0] == 2


@pytest.mark.parametrize("change", ["schema", "values"])
def test_feature_drift_fails_before_fitting_or_recording(setup, monkeypatch, change):
    args, calls = setup
    original = timing.build_zeroer_features

    def changed(*args, **kwargs):
        features = original(*args, **kwargs)
        if change == "schema":
            return features.drop(columns=["year_year_lev_sim"])
        features.iloc[0, 0] += .01
        return features

    monkeypatch.setattr(timing, "build_zeroer_features", changed)
    with pytest.raises(ValueError, match="schemas differ|values differ"):
        timing.benchmark_baseline_cell(**args)
    assert not calls["fit"]
    with sqlite3.connect(timing._database(args["storage"])) as connection:
        assert connection.execute("SELECT COUNT(*) FROM unsupervised_baseline_timings").fetchone()[0] == 0


def test_changed_code_cannot_append_to_existing_timings(setup, monkeypatch):
    args, calls = setup
    timing.benchmark_baseline_cell(**args)
    monkeypatch.setattr(timing, "_code_hash", lambda: "new-code")
    with pytest.raises(ValueError, match="changed"):
        timing.benchmark_baseline_cell(**args, repeats=2)
    assert calls["build"] == 1


def test_amazon_google_requires_explicit_expensive_opt_in(setup):
    args, calls = setup
    with pytest.raises(ValueError, match="allow_expensive=True"):
        timing.benchmark_baseline_cell(**dict(args, dataset="amazon_googleproducts"))
    assert calls["build"] == 0 and not calls["fit"]
