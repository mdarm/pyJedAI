"""Build pair-local cosine and L-infinity caches, resuming per dataset/model.

    python -m prepare.build_embedding_bank [dataset ...]

Outputs stay in ``outputs/cosine_bank/``. Presence flags live in pairs.parquet,
separate from similarity columns; every model must agree on pairs and presence.
"""
import hashlib
import json
import os
import sys
import time
from os.path import exists, join

import numpy as np
import pandas as pd

from config import ATTR_DIR, BANK_DIR, BENCH_DIR, FEATURES_DIR, LADDER, cell_name
from data import DATASETS, embedding_models
from embedding_bank import cell_dir, sides
from embedding_features import METRICS

def _manifest(path, update=None):
    manifest = json.load(open(path)) if exists(path) else {}
    if update:
        manifest.update(update)
        json.dump(manifest, open(path, "w"), indent=1, sort_keys=True)
    return manifest


def _pairs(dataset, side):
    """The stored pairs of one side's largest cell, as table row numbers."""
    k = sides(dataset)[side]
    ids = pd.read_csv(join(FEATURES_DIR, cell_name(dataset, side, k) + ".csv.gz"),
                      usecols=["ltable_id", "rtable_id"], dtype=str)
    rows = {}
    for column, table in (("ltable_id", "left"), ("rtable_id", "right")):
        table_ids = pd.read_csv(join(BENCH_DIR, dataset, table + ".csv"),
                                usecols=["id"], dtype=str)["id"]
        rows[column] = pd.Series(np.arange(len(table_ids)), index=table_ids)[ids[column]].to_numpy()
    return ids.assign(rank=np.arange(len(ids)) % k), rows["ltable_id"], rows["rtable_id"]


def build(dataset, bank_dir=BANK_DIR, models=None, attr_dir=ATTR_DIR):
    attributes = DATASETS[dataset]["attributes"]
    cells = {side: _pairs(dataset, side) for side in sides(dataset)}
    for side in cells:
        os.makedirs(cell_dir(dataset, side, bank_dir), exist_ok=True)

    for model in models or embedding_models(dataset):
        if all(exists(join(cell_dir(dataset, side, bank_dir), model + ".parquet"))
               for side in cells):
            continue
        t0 = time.time()
        columns = {side: {} for side in cells}
        present = {side: {} for side in cells}
        sources = {}
        for attr in attributes:
            vectors = {}
            for table in ("left", "right"):
                path = join(attr_dir, dataset, "{}_{}_{}.npy".format(model, attr, table))
                vectors[table] = np.load(path).astype(np.float32, copy=False)
                sources[os.path.basename(path)] = dict(
                    bytes=os.path.getsize(path), mtime=int(os.path.getmtime(path)))
            for side, (_, li, ri) in cells.items():
                u, v = vectors["left"][li], vectors["right"][ri]
                # an exact zero row is the encoders' "value absent" marker
                both = u.any(axis=1) & v.any(axis=1)
                present[side]["present_" + attr] = both
                for metric, fn in METRICS.items():
                    columns[side]["{}_emb_{}__{}".format(attr, metric, model)] = \
                        np.where(both, fn(u, v), 0.0).astype(np.float32)
                del u, v
            del vectors

        for side, (ids, _, _) in cells.items():
            directory = cell_dir(dataset, side, bank_dir)
            pairs_path = join(directory, "pairs.parquet")
            pairs = ids.assign(**present[side])
            if exists(pairs_path):
                if not pd.read_parquet(pairs_path).equals(pairs):
                    raise ValueError(
                        "{}: {}'s pairs or presence differ from the bank's — its "
                        "vectors were not encoded from the same tables".format(directory, model))
            else:
                pairs.to_parquet(pairs_path, index=False)
                _manifest(join(directory, "manifest.json"), dict(
                    pairs_md5=hashlib.md5("\n".join(
                        ids["ltable_id"] + "\t" + ids["rtable_id"]).encode()).hexdigest(),
                    n_pairs=len(ids), metrics=sorted(METRICS), attributes=attributes,
                    models={}))
            frame = pd.DataFrame(columns[side])
            frame.to_parquet(join(directory, model + ".parquet"), index=False)
            manifest = _manifest(join(directory, "manifest.json"))
            manifest["models"][model] = dict(
                sources=sources, dtype="float32",
                md5=hashlib.md5(frame.to_numpy().tobytes()).hexdigest())
            _manifest(join(directory, "manifest.json"), manifest)
        print("{} {}: {:.0f}s".format(dataset, model, time.time() - t0), flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or LADDER:
        build(name)
