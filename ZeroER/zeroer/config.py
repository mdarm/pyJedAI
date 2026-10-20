"""Shared paths and frozen candidate sets; usable from Python 3.6."""
import os
from os.path import abspath, dirname, join

# Artifacts stay beside the notebooks, outside the Python package.
ROOT_DIR = dirname(dirname(abspath(__file__)))
BENCH_DIR = join(ROOT_DIR, "datasets")
EMB_DIR = join(BENCH_DIR, "embeddings")
RECORD_DIR = join(EMB_DIR, "record")
ATTR_DIR = join(EMB_DIR, "attribute")
OUTPUT_DIR = join(ROOT_DIR, "outputs")
PAIRS_DIR = join(OUTPUT_DIR, "ladder_pairs")
FEATURES_DIR = join(BENCH_DIR, "magellan-features")
# Keep the existing cache location: it contains cosine and L-infinity features.
BANK_DIR = join(OUTPUT_DIR, "cosine_bank")
STORAGE = "sqlite:///" + join(OUTPUT_DIR, "optuna.db")
MATCHING_STORAGE = "sqlite:///" + join(OUTPUT_DIR, "optuna_matching.db")
UNSUPERVISED_STORAGE = "sqlite:///" + join(OUTPUT_DIR, "optuna_unsupervised.db")
UPSTREAM = os.environ.get("ZEROER_UPSTREAM", join(dirname(dirname(ROOT_DIR)), "zeroer"))

# Fixed retrieval model and PQ operating points used by the matching experiments.
MODEL = "tencent_KaLM-Embedding-Gemma3-12B-2511"

DISTANCE = "cosine"

LADDER = {
    "fodors_zagats":         [("right", 4), ("left", 1), ("right", 1)],
    "dblp_acm":              [("right", 12), ("right", 5), ("left", 2), ("left", 1)],
    "abt_buy":               [("left", 13), ("right", 5), ("left", 2), ("left", 1)],
    "amazon_googleproducts": [("right", 5), ("right", 2), ("left", 2), ("left", 1)],
    "dblp_scholar":          [("right", 1), ("left", 10), ("left", 5), ("left", 1)],
}


def cell_name(dataset, side, k):
    return "{}__{}_k{}".format(dataset, side, k)
