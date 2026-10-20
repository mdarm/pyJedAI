"""Label-free loading, fixed train transforms and record-disjoint validation."""
import builtins
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
import pytest

import embedding_bank
import unsupervised_features as uf


@pytest.fixture
def tiny_cell(tmp_path, monkeypatch):
    dataset, models, attributes = "toy", ["minilm6", "mpnet"], ["name", "price"]
    bench, records, attrs, bank = [tmp_path / name for name in ("tables", "record", "attribute", "bank")]
    for root in (bench, records, attrs):
        (root / dataset).mkdir(parents=True)
    directory = bank / "toy__right_k3"
    directory.mkdir(parents=True)
    for side in ("left", "right"):
        pd.DataFrame({"id": [side + str(i) for i in range(6)],
                      "name": ["text"] * 6}).to_csv(bench / dataset / (side + ".csv"), index=False)
    (bench / dataset / "matches.csv").write_text("DO NOT OPEN")
    li = np.tile(np.arange(3), 6)
    ri = np.repeat(np.arange(6), 3)
    metadata = pd.DataFrame({"ltable_id": ["left" + str(i) for i in li],
                             "rtable_id": ["right" + str(i) for i in ri],
                             "rank": np.tile(np.arange(3), 6)})
    rng = np.random.default_rng(45)
    raw = {}
    for model in models:
        columns = {}
        for attribute in [None] + attributes:
            vectors = []
            for side in ("left", "right"):
                vector = rng.normal(size=(6, 4)).astype(np.float32)
                vector /= np.linalg.norm(vector, axis=1, keepdims=True)
                if attribute == "price":
                    vector[1] = 0
                if attribute is None:
                    vector = (vector * 2.3).astype(np.float16)
                root = records if attribute is None else attrs
                stem = model if attribute is None else model + "_" + attribute
                np.save(root / dataset / (stem + "_" + side + ".npy"), vector)
                raw[model, attribute, side] = vector
                vectors.append(vector.astype(np.float32))
            if attribute is None:
                continue
            u, v = vectors[0][li], vectors[1][ri]
            present = u.any(axis=1) & v.any(axis=1)
            metadata["present_" + attribute] = present
            columns[attribute + "_emb_cos__" + model] = np.where(
                present, np.einsum("ij,ij->i", u, v), 0).astype(np.float32)
            columns[attribute + "_emb_linf__" + model] = np.where(
                present, 1 / (1 + np.abs(u - v).max(axis=1)), 0).astype(np.float32)
        pd.DataFrame(columns).to_parquet(directory / (model + ".parquet"), index=False)
    metadata.to_parquet(directory / "pairs.parquet", index=False)
    monkeypatch.setattr(uf, "BENCH_DIR", str(bench))
    monkeypatch.setattr(uf, "RECORD_DIR", str(records))
    monkeypatch.setattr(uf, "ATTR_DIR", str(attrs))
    monkeypatch.setattr(uf, "DATASETS", {dataset: {"attributes": attributes}})
    monkeypatch.setattr(embedding_bank, "LADDER", {dataset: [("right", 3)]})
    monkeypatch.setattr(uf, "load_bank", partial(embedding_bank.load_bank, bank_dir=str(bank)))
    monkeypatch.setattr(uf, "load_presence", partial(embedding_bank.load_presence, bank_dir=str(bank)))
    original_read = pd.read_csv
    original_open = builtins.open

    def forbid_labels(path, *args, **kwargs):
        assert not str(path).endswith("matches.csv"), "labels must not be opened"
        return original_open(path, *args, **kwargs)

    def only_ids(path, *args, **kwargs):
        assert not str(path).endswith("matches.csv"), "labels must not be read"
        assert kwargs.get("usecols") == ["id"], "only table IDs may be loaded"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", only_ids)
    monkeypatch.setattr(builtins, "open", forbid_labels)
    return dict(dataset=dataset, models=models, attributes=attributes, raw=raw, directory=directory)


def test_loader_is_label_free_and_record_cosine_is_normalized(tiny_cell):
    cell = uf.load_cell("toy", "right", 2, tiny_cell["models"], metrics=("cos", "linf"))
    assert len(cell["ids"]) == 12
    np.testing.assert_array_equal(cell["pairs"][:, 0], np.tile([0, 1], 6))
    np.testing.assert_array_equal(cell["pairs"][:, 1], 6 + np.repeat(np.arange(6), 2))
    assert list(cell["present"]) == ["name", "price", "record"]
    assert list(cell["record_scores"]) == tiny_cell["models"]
    assert "record_emb_linf__minilm6" not in cell["bank"]
    for model in tiny_cell["models"]:
        left = tiny_cell["raw"][model, None, "left"].astype(float)[cell["pairs"][:, 0]]
        right = tiny_cell["raw"][model, None, "right"].astype(float)[cell["pairs"][:, 1] - 6]
        expected = (left * right).sum(axis=1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1))
        np.testing.assert_allclose(cell["record_scores"][model], expected, atol=1e-14)
    assert all(value >= 0 for value in cell["timings"].values())
    again = uf.load_cell("toy", "right", 2, tiny_cell["models"], metrics=("cos", "linf"))
    assert cell["input_hash"] == again["input_hash"]
    path = tiny_cell["directory"] / "minilm6.parquet"
    changed = pd.read_parquet(path)
    changed.loc[0, "name_emb_cos__minilm6"] = np.float32(
        changed.loc[0, "name_emb_cos__minilm6"] + 0.01)
    changed.to_parquet(path, index=False)
    assert uf.load_cell("toy", "right", 2, tiny_cell["models"], metrics=("cos", "linf"))["input_hash"] != cell["input_hash"]


def test_fresh_repeats_equal_the_cached_attribute_bank(tiny_cell):
    report = uf.benchmark_similarity("toy", "right", 2, tiny_cell["models"],
                                     metrics=("cos", "linf"), repeats=2)
    assert len(report["repeats"]) == 2
    assert report["max_absolute_difference"] <= 1e-6
    assert report["n_pairs"] == 12
    assert report["n_features"] == 10
    assert report["cache_state"] == "OS cache unspecified"
    for timing in report["repeats"]:
        assert set(timing) == {"loading_seconds", "similarity_seconds", "preprocessing_seconds"}
        assert all(value >= 0 for value in timing.values())


def test_loader_can_omit_record_vectors(tiny_cell, monkeypatch):
    def unexpected_vectors(*args, **kwargs):
        pytest.fail("cached-only loader should not touch raw vectors")

    monkeypatch.setattr(uf, "_raw_scores", unexpected_vectors)
    cell = uf.load_cell("toy", "right", 1, tiny_cell["models"], include_record=False)
    assert list(cell["present"]) == tiny_cell["attributes"]
    assert cell["record_scores"].shape == (6, 0)


@pytest.mark.parametrize("fusion", ["separate", "mean", "zmean", "family"])
def test_transform_reuses_training_schema_and_statistics(fusion):
    models = ["minilm6", "mpnet"]
    bank = pd.DataFrame({"name_emb_cos__minilm6": [0., 1., 2., 3., 100.],
                         "name_emb_cos__mpnet": [4., 6., 8., 10., -100.],
                         "price_emb_cos__minilm6": [5., 5., 5., 5., 9.],
                         "price_emb_cos__mpnet": [5., 5., 5., 5., 10.],
                         "record_emb_cos__minilm6": [.1, .2, .3, .4, .9],
                         "record_emb_cos__mpnet": [.2, .4, .6, .8, .9]})
    present = pd.DataFrame({"name": [True] * 5, "price": [True] * 5,
                            "record": [True] * 5})
    transform = uf.PanelTransform(models, ["name", "price"], fusion=fusion)
    transform.fit(bank.iloc[:4], present.iloc[:4])
    train = transform.transform(bank.iloc[:4], present.iloc[:4])
    result = transform.transform(bank.iloc[4:], present.iloc[4:])
    assert list(result) == transform.columns_ == list(train)
    assert not any(column.startswith("price_") for column in result)
    assert any(column.startswith("record_") for column in result)
    if fusion in ("zmean", "family"):
        assert train.iloc[0, 0] == pytest.approx(0)
        assert train.iloc[3, 0] == pytest.approx(1)
        # A validation outlier is scaled using the training range, never clipped.
        assert result.iloc[0, 0] == pytest.approx(8)
    pd.testing.assert_frame_equal(train, transform.transform(bank.iloc[:4], present.iloc[:4]))
    missing = present.iloc[4:].copy()
    missing["name"] = False
    assert (transform.transform(bank.iloc[4:], missing).filter(like="name_") == 0).all().all()


def test_record_features_always_use_cosine_with_linf_attribute_panel():
    bank = pd.DataFrame({"name_emb_linf__minilm6": [.3, .4, .8],
                         "record_emb_cos__minilm6": [-.2, .4, .9]})
    present = pd.DataFrame({"name": [True] * 3, "record": [True] * 3})
    panel = uf.PanelTransform(["minilm6"], ["name"], metrics=("linf",)).fit(bank, present)
    assert panel.columns_ == list(bank)


def test_empty_observed_training_features_are_rejected():
    bank = pd.DataFrame({"name_emb_cos__minilm6": [0., 0., 1.]})
    panel = uf.PanelTransform(["minilm6"], ["name"])
    for mask in ([False] * 3, [True, True, False]):
        with pytest.raises(ValueError, match="no nonconstant observed"):
            panel.fit(bank, pd.DataFrame({"name": mask}))


def test_record_splits_are_disjoint_exhaustive_and_stable_across_cells():
    ids = pd.DataFrame(product(["L" + str(i) for i in range(25)],
                               ["R" + str(i) for i in range(25)]),
                       columns=["ltable_id", "rtable_id"])
    splits = uf.record_disjoint_splits(ids, n_splits=3, seed=17)
    subset = ids.sample(frac=.5, random_state=2).reset_index(drop=True)
    smaller = uf.record_disjoint_splits(subset, n_splits=3, seed=17)
    for split, small in zip(splits, smaller):
        np.testing.assert_array_equal(np.sort(np.concatenate([split[name] for name in
                                                              ("train", "validation", "excluded")])),
                                      np.arange(len(ids)))
        train, validation = ids.iloc[split["train"]], ids.iloc[split["validation"]]
        for column in ids:
            assert set(train[column]).isdisjoint(validation[column])
        for name in ("train", "validation", "excluded"):
            full_pairs = set(map(tuple, ids.iloc[split[name]].to_numpy()))
            small_pairs = set(map(tuple, subset.iloc[small[name]].to_numpy()))
            assert small_pairs <= full_pairs
        metadata = split["metadata"]
        assert metadata["n_train"] > 0 and metadata["n_validation"] > 0
        assert metadata["n_pairs"] == 625
        assert metadata["retained_fraction"] + metadata["excluded_fraction"] == pytest.approx(1)
