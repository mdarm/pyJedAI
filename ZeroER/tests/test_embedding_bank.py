"""The cosine bank holds what ``build_embedding_features`` would compute.

The bank exists so that no pool trial reads raw vectors; that is only safe if a
bank column is the same number the single-model API produces for the same pair.
Checked on the smallest cell with the smallest model, including the ``rank < k``
slice and the presence flags.

Needs ``minilm6``'s attribute vectors for ``fodors_zagats``; skips without them.

Run:  pytest tests/test_embedding_bank.py -v
"""
import os
from os.path import join

import numpy as np
import pytest

import zeroer.features.bank as embedding_bank
from prepare.build_embedding_bank import build
from zeroer.config import ATTR_DIR
from zeroer.data.benchmarks import DATASETS, FAMILIES, PARAMETERS_M, load_data
from zeroer.features.embeddings import build_embedding_features, fuse

DATASET, MODEL = "fodors_zagats", "minilm6"
pytestmark = pytest.mark.skipif(
    not os.path.exists(join(ATTR_DIR, DATASET, MODEL + "_name_left.npy")),
    reason="no attribute embeddings for " + MODEL)


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    bank_dir = str(tmp_path_factory.mktemp("bank"))
    build(DATASET, bank_dir=bank_dir, models=[MODEL])
    return bank_dir


def test_bank_equals_the_single_model_features(bank):
    data = load_data(DATASET)
    attributes = DATASETS[DATASET]["attributes"]
    for k in (4, 1):                                  # the stored cell, and a slice of it
        pairs = embedding_bank.load_presence(DATASET, "right", k, bank_dir=bank)
        left = data.dataset_1["id"].astype(str).tolist()
        right = data.dataset_2["id"].astype(str).tolist()
        positions = [(left.index(a), data.dataset_limit + right.index(b))
                     for a, b in zip(pairs["ltable_id"], pairs["rtable_id"])]
        expected = build_embedding_features(positions, DATASET, MODEL, attributes, data,
                                            metrics=("cos", "linf"))
        banked = embedding_bank.load_bank(DATASET, "right", k, [MODEL],
                                       metrics=("cos", "linf"), bank_dir=bank)
        assert list(banked.columns) == [c + "__" + MODEL for c in expected.columns]
        np.testing.assert_allclose(banked.to_numpy(), expected.to_numpy(), atol=1e-6)


def test_absent_values_are_flagged_and_scored_zero(bank):
    pairs = embedding_bank.load_presence(DATASET, "right", 4, bank_dir=bank)
    banked = embedding_bank.load_bank(DATASET, "right", 4, [MODEL], bank_dir=bank)
    for attr in DATASETS[DATASET]["attributes"]:
        absent = ~pairs["present_" + attr].to_numpy()
        assert (banked.loc[absent, "{}_emb_cos__{}".format(attr, MODEL)] == 0).all()


def test_rebuild_is_skipped(bank, capsys):
    build(DATASET, bank_dir=bank, models=[MODEL])
    assert capsys.readouterr().out == ""


def test_every_pool_model_has_a_family_and_a_size():
    from zeroer.data.benchmarks import embedding_models
    pool = embedding_models(DATASET)
    in_families = [m for members in FAMILIES.values() for m in members]
    assert sorted(in_families) == sorted(pool) == sorted(PARAMETERS_M)
    assert [len(m) for m in FAMILIES.values()] == [4, 2, 4, 10]


def test_fusion_rules():
    import pandas as pd
    rng = np.random.default_rng(0)
    models = ["minilm6", "mpnet", "bge_large"]
    present = {"name": np.array([True] * 50 + [False] * 10)}
    bank = pd.DataFrame({"name_emb_cos__" + m: np.where(present["name"], rng.random(60), 0.0)
                         for m in models})
    separate = fuse(bank, models, ["name"], present, "separate")
    assert list(separate.columns) == list(bank.columns)
    mean = fuse(bank, models, ["name"], present, "mean")
    np.testing.assert_allclose(mean["name_emb_cos__mean"], bank.mean(axis=1))
    family = fuse(bank, models, ["name"], present, "family")
    assert list(family.columns) == ["name_emb_cos__classic", "name_emb_cos__retriever"]
    for column in family:             # back on [0, 1], and 0 where the value is absent
        assert family[column][:50].min() == 0 and family[column][:50].max() == 1
        assert (family[column][50:] == 0).all()
    # every column still files into the attribute's block
    assert {c.split("_")[0] for c in list(separate) + list(mean) + list(family)} == {"name"}
