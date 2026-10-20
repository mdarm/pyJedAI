"""Materialize only the fixed KaLM Amazon/Google left-k4 primary candidate cell.

    python -m prepare.primary_candidates pairs
    python -m prepare.primary_candidates bank

This uses existing record/attribute vectors and the frozen retrieval settings.
It performs no blocker search, encoding or label reads. The original left-k2
pair CSV and cache remain unchanged; the first two neighbors must match them.
Genuine Magellan features are generated separately by reference_features.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from zeroer.config import (ATTR_DIR, BANK_DIR, BENCH_DIR, DISTANCE, MODEL, PAIRS_DIR,
                    RECORD_DIR, cell_name)
from zeroer.data.benchmarks import DATASETS, embedding_models
from zeroer.estimator import nn_search
from zeroer.features.panels import _raw_scores

DATASET, SIDE, K = "amazon_googleproducts", "left", 4
METRICS = ("cos", "linf")


def _tables():
    return [pd.read_csv(Path(BENCH_DIR) / DATASET / (side + ".csv"),
                        usecols=["id"], dtype=str)["id"].to_numpy()
            for side in ("left", "right")]


def _check_prefix(ids):
    previous = pd.read_csv(Path(PAIRS_DIR) / (cell_name(DATASET, SIDE, 2) + ".csv"), dtype=str)
    prefix = ids.loc[np.arange(len(ids)) % K < 2].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix, previous)


def export_pairs():
    """Write the new CSV only after exact prefix parity with the existing k2."""
    left_ids, right_ids = _tables()
    vectors = [np.load(Path(RECORD_DIR) / DATASET / (MODEL + "_" + side + ".npy"),
                       mmap_mode="r") for side in ("left", "right")]
    if len(vectors[0]) != len(left_ids) or len(vectors[1]) != len(right_ids):
        raise ValueError("retrieval embeddings and table IDs have different row counts")
    neighbors = nn_search(vectors[1], vectors[0], K, DISTANCE)
    if neighbors.shape != (len(left_ids), K):
        raise ValueError("fixed retrieval did not return four neighbors per left record")
    # FAISS can reverse exact ties when k changes. Preserve the frozen prefix
    # only by swapping already-returned neighbors with identical raw vectors.
    previous = pd.read_csv(Path(PAIRS_DIR) / (cell_name(DATASET, SIDE, 2) + ".csv"), dtype=str)
    right_positions = pd.Series(np.arange(len(right_ids)), index=right_ids)
    old_neighbors = right_positions.loc[previous.rtable_id].to_numpy().reshape(len(left_ids), 2)
    tie_adjustments = []
    for row in np.unique(np.where(neighbors[:, :2] != old_neighbors)[0]):
        for rank, expected in enumerate(old_neighbors[row]):
            actual = neighbors[row, rank]
            if actual == expected:
                continue
            positions = np.flatnonzero(neighbors[row] == expected)
            if not len(positions) or not np.array_equal(vectors[1][actual], vectors[1][expected]):
                raise ValueError("non-identical-vector prefix difference at row {}, rank {}".format(row, rank))
            position = positions[0]
            neighbors[row, rank], neighbors[row, position] = expected, actual
            tie_adjustments.append(dict(left_row=int(row), rank=int(rank),
                                        from_right_row=int(actual), to_right_row=int(expected)))
    ids = pd.DataFrame({"ltable_id": np.repeat(left_ids, K),
                        "rtable_id": right_ids[neighbors.ravel()]})
    _check_prefix(ids)
    path = Path(PAIRS_DIR) / (cell_name(DATASET, SIDE, K) + ".csv")
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path, dtype=str), ids)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        ids.to_csv(path, index=False)
    retrieval = dict(model=MODEL, distance=DISTANCE, k=K, n_pairs=len(ids),
                     identical_vector_tie_adjustments=tie_adjustments)
    path.with_suffix(".retrieval.json").write_text(json.dumps(retrieval, indent=2, sort_keys=True))
    print("pairs ready: {} ({} pairs, k2 prefix identical, {} exact-tie reorderings)".format(
        path, len(ids), len(tie_adjustments)), flush=True)
    return path


def _write_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(path)


def build_bank(models=None):
    """Build cosine/L-infinity columns in bounded vector chunks, model by model."""
    pair_path = Path(PAIRS_DIR) / (cell_name(DATASET, SIDE, K) + ".csv")
    if not pair_path.exists():
        raise FileNotFoundError("run `python -m prepare.primary_candidates pairs` first")
    ids = pd.read_csv(pair_path, dtype=str)
    _check_prefix(ids)
    left_ids, right_ids = _tables()
    if len(ids) != len(left_ids) * K:
        raise ValueError("candidate CSV has the wrong size")
    index_l = pd.Series(np.arange(len(left_ids)), index=left_ids)
    index_r = pd.Series(np.arange(len(right_ids)), index=right_ids)
    pairs = np.column_stack([index_l.loc[ids.ltable_id].to_numpy(),
                             len(left_ids) + index_r.loc[ids.rtable_id].to_numpy()])
    attributes = DATASETS[DATASET]["attributes"]
    directory = Path(BANK_DIR) / cell_name(DATASET, SIDE, K)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    pair_hash = hashlib.sha256(ids.to_csv(index=False).encode()).hexdigest()
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists() else
                dict(dataset=DATASET, side=SIDE, k=K, model=MODEL, distance=DISTANCE,
                     n_pairs=len(ids), pairs_sha256=pair_hash, attributes=attributes,
                     metrics=list(METRICS), models={}))
    if manifest["pairs_sha256"] != pair_hash:
        raise ValueError("candidate IDs changed since this bank was started")
    retrieval_path = pair_path.with_suffix(".retrieval.json")
    if retrieval_path.exists():
        manifest["retrieval"] = json.loads(retrieval_path.read_text())
    _write_manifest(manifest_path, manifest)
    for model in models or embedding_models(DATASET):
        output = directory / (model + ".parquet")
        sources = {}
        for attr in attributes:
            for table in ("left", "right"):
                path = Path(ATTR_DIR) / DATASET / (model + "_" + attr + "_" + table + ".npy")
                stat = path.stat()
                sources[path.name] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if output.exists() and model in manifest["models"]:
            if manifest["models"][model]["sources"] != sources:
                raise ValueError("attribute vector sources changed for " + model)
            continue
        columns, presence, timings = {}, {}, {}
        for attr in attributes:
            values, there, measured = _raw_scores(DATASET, model, pairs, len(left_ids),
                                                 attribute=attr, metrics=METRICS)
            presence["present_" + attr] = there
            for metric, score in values.items():
                columns["{}_emb_{}__{}".format(attr, metric, model)] = score.astype(np.float32)
            for key, value in measured.items():
                timings[key] = timings.get(key, 0.0) + value
        metadata = ids.assign(rank=np.arange(len(ids)) % K, **presence)
        metadata_path = directory / "pairs.parquet"
        if metadata_path.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(metadata_path), metadata)
        else:
            metadata.to_parquet(metadata_path, index=False)
        frame = pd.DataFrame(columns)
        temporary = output.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(output)
        manifest["models"][model] = dict(sources=sources, dtype="float32", timings=timings,
            sha256=hashlib.sha256(frame.to_numpy().tobytes()).hexdigest())
        _write_manifest(manifest_path, manifest)
        print("cached {}: {} pairs x {} columns".format(model, len(frame), len(frame.columns)), flush=True)
    print("bank ready: {} ({} models)".format(directory, len(manifest["models"])), flush=True)
    return directory


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "pairs":
        export_pairs()
    elif stage == "bank":
        build_bank()
    else:
        sys.exit(__doc__)
