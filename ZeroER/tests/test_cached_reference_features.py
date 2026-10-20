"""Feature-cache assembly never recalculates known pairs or hides conflicts."""
import hashlib
import json

import pandas as pd
import pytest

import prepare.cached_reference_features as cached


@pytest.fixture
def setup(tmp_path):
    features, pairs = tmp_path / "features", tmp_path / "pairs"
    features.mkdir()
    pairs.mkdir()
    columns = ["ltable_id", "rtable_id", "name_name_lev_sim"]
    first = features / "fixture__left_k1.csv.gz"
    second = features / "fixture__right_k1.csv.gz"
    pd.DataFrame([["a", "x", .125]], columns=columns).to_csv(first, index=False)
    pd.DataFrame([["b", "y", .75], ["a", "x", .125]], columns=columns).to_csv(second, index=False)
    target = pairs / "fixture__left_k2.csv"
    pd.DataFrame([["b", "y"], ["a", "x"], ["c", "z"]], columns=columns[:2]).to_csv(target, index=False)
    args = dict(dataset="fixture", side="left", k=2, features_dir=str(features), pairs_dir=str(pairs))
    return args, columns, first, second, target


def test_cached_only_never_calls_extractor_and_preserves_existing_files(setup, monkeypatch):
    args, columns, first, second, target = setup
    pd.read_csv(target).iloc[:2].to_csv(target, index=False)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (first, second)}
    monkeypatch.setattr(cached, "extract_pair_features", lambda *a, **k: pytest.fail("recomputed cached pairs"))
    report = cached.build_cached_features(**args, expected_missing=0)
    assert report["reused_rows"] == 2 and report["extracted_rows"] == 0
    result = pd.read_csv(report["target"])
    assert result[columns[2]].tolist() == [.75, .125]
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before} == before
    with open(report["target"] + ".reuse.json") as handle:
        saved = json.load(handle)
    assert saved["sha256"] == hashlib.sha256(open(report["target"], "rb").read()).hexdigest()
    with pytest.raises(FileExistsError, match="overwrite"):
        cached.build_cached_features(**args)


def test_partial_reuse_extracts_only_missing_pairs_and_restores_target_order(setup, monkeypatch):
    args, columns, _, _, _ = setup
    calls = []

    def extract(dataset, pairs, bench_dir, n_jobs):
        calls.append((dataset, pairs.copy(), n_jobs))
        return pairs.assign(name_name_lev_sim=.625)

    monkeypatch.setattr(cached, "extract_pair_features", extract)
    plan = cached.build_cached_features(**args, expected_missing=1, dry_run=True)
    assert not calls and plan["reused_rows"] == 2 and plan["extracted_rows"] == 1
    report = cached.build_cached_features(**args, expected_missing=1, n_jobs=4)
    assert len(calls) == 1 and calls[0][0] == "fixture" and calls[0][2] == 4
    assert calls[0][1].to_dict("list") == {"ltable_id": ["c"], "rtable_id": ["z"]}
    result = pd.read_csv(report["target"])
    assert result.to_dict("list") == {"ltable_id": ["b", "a", "c"],
        "rtable_id": ["y", "x", "z"], columns[2]: [.75, .125, .625]}


def test_conflicting_duplicate_cache_rows_are_rejected_before_extraction(setup, monkeypatch):
    args, columns, _, second, _ = setup
    pd.DataFrame([["b", "y", .75], ["a", "x", .5]], columns=columns).to_csv(second, index=False)
    monkeypatch.setattr(cached, "extract_pair_features", lambda *a, **k: pytest.fail("extracted after conflict"))
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        cached.build_cached_features(**args)


def test_unexpected_missing_scope_and_different_schema_fail_before_extraction(setup, monkeypatch):
    args, _, _, second, _ = setup
    monkeypatch.setattr(cached, "extract_pair_features", lambda *a, **k: pytest.fail("unexpected extraction"))
    with pytest.raises(ValueError, match="Expected 0 missing pairs, found 1"):
        cached.build_cached_features(**args, expected_missing=0)
    pd.read_csv(second).rename(columns={"name_name_lev_sim": "different"}).to_csv(second, index=False)
    with pytest.raises(ValueError, match="schemas differ"):
        cached.build_cached_features(**args)
