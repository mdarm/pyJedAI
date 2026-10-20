"""Larger primary cells are explicit; existing ladder slices keep their paths."""
import json
import numpy as np
import pandas as pd
import pytest

import zeroer.features.bank as bank
from prepare import primary_candidates as primary


@pytest.fixture
def caches(tmp_path, monkeypatch):
    monkeypatch.setattr(bank, "LADDER", {"toy": [("left", 2)]})
    for k in (1, 2, 4):
        directory = tmp_path / ("toy__left_k" + str(k))
        directory.mkdir()
        pd.DataFrame({"ltable_id": np.repeat(["a", "b"], k),
                      "rtable_id": np.tile(["r" + str(i) for i in range(k)], 2),
                      "rank": np.tile(np.arange(k), 2),
                      "present_name": True}).to_parquet(directory / "pairs.parquet", index=False)
        pd.DataFrame({"name_emb_cos__tiny": np.arange(2 * k) + 100 * k}).to_parquet(
            directory / "tiny.parquet", index=False)
    return str(tmp_path)


def test_existing_ladder_slices_ignore_smaller_explicit_directories(caches):
    assert bank.cell_dir("toy", "left", caches).endswith("toy__left_k2")
    frame = bank.load_bank("toy", "left", 1, ["tiny"], attributes=["name"], bank_dir=caches)
    assert frame.iloc[:, 0].tolist() == [200, 202]
    assert len(bank.load_presence("toy", "left", 1, bank_dir=caches)) == 2


def test_larger_cell_requires_its_own_cache_instead_of_silently_truncating(caches):
    frame = bank.load_bank("toy", "left", 4, ["tiny"], attributes=["name"], bank_dir=caches)
    assert frame.iloc[:, 0].tolist() == list(range(400, 408))
    assert len(bank.load_presence("toy", "left", 4, bank_dir=caches)) == 8
    with pytest.raises(FileNotFoundError):
        bank.load_presence("toy", "left", 3, bank_dir=caches)
    with pytest.raises(FileNotFoundError):
        bank.load_bank("toy", "left", 3, ["tiny"], attributes=["name"], bank_dir=caches)


def test_fixed_candidate_export_preserves_prefix_and_never_reads_labels(tmp_path, monkeypatch):
    dataset = primary.DATASET
    tables, records, pairs = [tmp_path / name for name in ("tables", "record", "pairs")]
    for root in (tables, records):
        (root / dataset).mkdir(parents=True)
    pairs.mkdir()
    for side, count in (("left", 2), ("right", 4)):
        pd.DataFrame({"id": [side + str(i) for i in range(count)]}).to_csv(
            tables / dataset / (side + ".csv"), index=False)
        np.save(records / dataset / (primary.MODEL + "_" + side + ".npy"),
                np.arange(count * 3, dtype=np.float32).reshape(count, 3))
    previous = pd.DataFrame({"ltable_id": ["left0", "left0", "left1", "left1"],
                             "rtable_id": ["right0", "right1", "right1", "right2"]})
    old_path = pairs / (dataset + "__left_k2.csv")
    previous.to_csv(old_path, index=False)
    old_bytes = old_path.read_bytes()
    monkeypatch.setattr(primary, "BENCH_DIR", str(tables))
    monkeypatch.setattr(primary, "RECORD_DIR", str(records))
    monkeypatch.setattr(primary, "PAIRS_DIR", str(pairs))
    monkeypatch.setattr(primary, "nn_search", lambda *args: np.array([[0, 1, 2, 3], [1, 2, 3, 0]]))
    path = primary.export_pairs()
    assert len(pd.read_csv(path)) == 8
    assert old_path.read_bytes() == old_bytes
    assert primary.export_pairs() == path
    monkeypatch.setattr(primary, "nn_search", lambda *args: np.array([[1, 0, 2, 3], [1, 2, 3, 0]]))
    with pytest.raises(ValueError, match="non-identical-vector"):
        primary.export_pairs()
    assert old_path.read_bytes() == old_bytes
    right_path = records / dataset / (primary.MODEL + "_right.npy")
    vectors = np.load(right_path)
    vectors[1] = vectors[0]
    np.save(right_path, vectors)
    assert primary.export_pairs() == path
    retrieval = json.loads(path.with_suffix(".retrieval.json").read_text())
    assert len(retrieval["identical_vector_tie_adjustments"]) == 1
    assert old_path.read_bytes() == old_bytes
