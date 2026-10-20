"""Record-only timing measures each model once and charges pools by membership."""
import json

import numpy as np
import pandas as pd
import pytest

import zeroer.experiments.unsupervised.cosine_timing as timing


@pytest.fixture
def cosine_inputs(tmp_path, monkeypatch):
    record_dir = tmp_path / "record"
    directory = record_dir / "toy"
    directory.mkdir(parents=True)
    models = ["minilm6", "mpnet"]
    rng = np.random.default_rng(4)
    for model in models:
        for side in ("left", "right"):
            values = rng.normal(size=(4, 3)).astype(np.float16)
            if side == "left":
                values[3] = 0
            np.save(directory / (model + "_" + side + ".npy"), values)
    ids = pd.DataFrame({"ltable_id": ["l0", "l1", "l2", "l3"],
                        "rtable_id": ["r0", "r1", "r2", "r3"]})
    pairs = np.column_stack([np.arange(4), np.arange(4) + 4])
    monkeypatch.setattr(timing.feature_inputs, "RECORD_DIR", str(record_dir))
    monkeypatch.setattr(timing.feature_inputs, "ATTR_DIR", str(tmp_path / "forbidden-attributes"))
    monkeypatch.setattr(timing, "_candidate_input", lambda *args: (ids, pairs, None, 4))
    original = timing._raw_scores
    calls = []

    def raw(dataset, model, candidates, limit):
        calls.append(model)
        # No attribute or metric argument is forwarded: this is record cosine.
        return original(dataset, model, candidates, limit)

    monkeypatch.setattr(timing, "_raw_scores", raw)
    args = dict(dataset="toy", side="right", k=1, models=models,
                storage="sqlite:///" + str(tmp_path / "cosine.db"), thresholds=[.3, .7])
    return args, calls, directory


def test_member_costs_are_exact_and_completed_repeats_resume(cosine_inputs):
    args, calls, _ = cosine_inputs
    rows = timing.benchmark_cosine_cell(**args, repeats=2)
    assert len(rows) == 40
    assert calls == ["minilm6", "mpnet"] * 2
    assert set(rows.score_name) == {"record_cos:minilm6", "record_cos:mpnet", "all:mean", "classic:pair"}
    one = rows[(rows["repeat"] == 0) & (rows.decoder == "greedy") & (rows.threshold_json == "null")]
    singles = one[one.score_name.str.startswith("record_cos:")]
    mean = one[one.score_name == "all:mean"].iloc[0]
    common = json.loads(mean.metadata_json)["common_loading_seconds"]
    assert mean.loading_seconds == pytest.approx(singles.loading_seconds.sum() - common)
    assert mean.similarity_seconds == pytest.approx(singles.similarity_seconds.sum())
    np.testing.assert_allclose(rows.matching_seconds,
        rows.similarity_seconds + rows.preprocessing_seconds + rows.aggregation_seconds + rows.decode_seconds)
    np.testing.assert_allclose(rows.total_seconds, rows.loading_seconds + rows.matching_seconds)
    for text in rows.metadata_json:
        metadata = json.loads(text)
        assert metadata["uses_gold"] is False
        assert metadata["cost_scope"] == "record-only selected models"
    pd.testing.assert_frame_equal(rows, timing.benchmark_cosine_cell(**args, repeats=2))
    assert len(calls) == 4
    assert len(timing.benchmark_cosine_cell(**args, repeats=3)) == 60
    assert len(calls) == 6


def test_changed_inputs_and_cuts_cannot_mix_with_existing_measurements(cosine_inputs):
    args, calls, directory = cosine_inputs
    timing.benchmark_cosine_cell(**args, repeats=1)
    with pytest.raises(ValueError, match="protocol or inputs changed"):
        timing.benchmark_cosine_cell(**dict(args, thresholds=[.5]), repeats=2)
    path = directory / "minilm6_left.npy"
    values = np.load(path)
    values[0] *= 2
    np.save(path, values)
    with pytest.raises(ValueError, match="protocol or inputs changed"):
        timing.benchmark_cosine_cell(**args, repeats=2)
    assert len(calls) == 2


def test_models_must_agree_on_record_presence(cosine_inputs):
    args, _, directory = cosine_inputs
    path = directory / "mpnet_left.npy"
    values = np.load(path)
    values[0] = 0
    np.save(path, values)
    with pytest.raises(ValueError, match="record presence differs"):
        timing.benchmark_cosine_cell(**args, repeats=1)
