"""Assemble genuine Magellan caches by pair ID, extracting only missing pairs.

Usable under the Python 3.6 upstream environment. Existing feature artifacts
are read-only; the new cache is published only after assembly validates.
"""
import glob
import hashlib
import json
import os
import sys
import tempfile
import time
from os.path import exists, join

import numpy as np
import pandas as pd

from config import BENCH_DIR, FEATURES_DIR, PAIRS_DIR, cell_name
from prepare.reference_features import extract_pair_features

IDS = ["ltable_id", "rtable_id"]


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cached_features(dataset, side, k, bench_dir=BENCH_DIR,
                          features_dir=FEATURES_DIR, pairs_dir=PAIRS_DIR,
                          n_jobs=4, expected_missing=None, dry_run=False):
    """Reuse all compatible dataset caches and extract only unknown target pairs.

    ``dry_run`` reports coverage and validates overlapping caches without any
    extraction or writes. ``expected_missing`` guards a reviewed extraction
    scope. Raw cached numeric text is retained when assembling reused rows.
    The ``.reuse.json`` sidecar records source hashes, coverage and extraction
    time. Existing target files are never overwritten.
    """
    started = time.time()
    name = cell_name(dataset, side, k)
    target = join(features_dir, name + ".csv.gz")
    if exists(target):
        raise FileExistsError("Refusing to overwrite existing feature cache: " + target)
    pair_path = join(pairs_dir, name + ".csv")
    pairs = pd.read_csv(pair_path, dtype=str)
    if list(pairs.columns) != IDS or pairs.isnull().values.any() or pairs.duplicated(IDS).any():
        raise ValueError("Target candidate IDs must be nonmissing, unique and ordered in two columns")
    source_paths = sorted(glob.glob(join(features_dir, dataset + "__*.csv.gz")))
    if not source_paths:
        raise ValueError("No existing dataset feature cache is available for schema validation")
    sources, frames, columns = [], [], None
    for path in source_paths:
        frame = pd.read_csv(path, dtype=str)
        if columns is None:
            columns = list(frame.columns)
        if list(frame.columns) != columns or columns[:2] != IDS:
            raise ValueError("Cached feature schemas differ: " + path)
        sources.append(dict(path=os.path.abspath(path), sha256=_sha256(path), rows=len(frame)))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    cache = combined.drop_duplicates(IDS).set_index(IDS)
    values = combined.iloc[:, 2:].values.astype(float)
    expected = cache.reindex(pd.MultiIndex.from_frame(combined[IDS])).values.astype(float)
    difference = np.abs(values - expected)
    if not np.isfinite(difference).all() or (difference > 1e-15).any():
        raise ValueError("Conflicting duplicate feature rows in existing caches")
    index = pd.MultiIndex.from_frame(pairs)
    reused = index.isin(cache.index)
    missing = pairs.loc[~reused].reset_index(drop=True)
    if expected_missing is not None and len(missing) != expected_missing:
        raise ValueError("Expected {} missing pairs, found {}".format(expected_missing, len(missing)))
    report = dict(dataset=dataset, side=side, k=int(k), target=os.path.abspath(target),
                  target_pairs_sha256=_sha256(pair_path), rows=len(pairs),
                  reused_rows=int(reused.sum()), extracted_rows=len(missing),
                  n_features=len(columns) - 2, sources=sources, workers=n_jobs,
                  duplicate_max_absolute_difference=float(difference.max()) if difference.size else 0.,
                  loading_validation_seconds=time.time() - started,
                  interpreter=sys.executable, source="genuine upstream Magellan; cached pair reuse")
    if dry_run:
        return report
    extract_started = time.time()
    fresh = None
    if len(missing):
        fresh = extract_pair_features(dataset, missing, bench_dir=bench_dir, n_jobs=n_jobs)
        if list(fresh.columns) != columns or not fresh[IDS].astype(str).reset_index(drop=True).equals(missing):
            raise ValueError("Newly extracted features differ in schema or candidate order")
        if not np.isfinite(fresh.iloc[:, 2:].values.astype(float)).all():
            raise ValueError("Newly extracted features contain nonfinite values")
    report["extraction_seconds"] = time.time() - extract_started
    assembled = pd.concat([pairs, cache.reindex(index).reset_index(drop=True)], axis=1)
    if fresh is not None:
        assembled.loc[~reused, columns[2:]] = fresh.iloc[:, 2:].values
    if assembled.isnull().values.any():
        raise ValueError("Assembled feature cache contains unfilled rows")
    np.testing.assert_array_equal(assembled.loc[reused, columns[2:]].values.astype(float),
                                  cache.reindex(index[reused]).values.astype(float))
    if _sha256(pair_path) != report["target_pairs_sha256"]:
        raise ValueError("Target candidate pairs changed during extraction")
    for source in sources:
        if _sha256(source["path"]) != source["sha256"]:
            raise ValueError("Source cache changed during extraction: " + source["path"])
    os.makedirs(features_dir, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + name, suffix=".csv.gz", dir=features_dir)
    os.close(descriptor)
    try:
        assembled.to_csv(temporary, index=False, compression="gzip")
        report["sha256"] = _sha256(temporary)
        # Hard-link publication is atomic and cannot overwrite a concurrent cache.
        os.link(temporary, target)
    finally:
        os.unlink(temporary)
    report["seconds"] = time.time() - started
    with open(target + ".reuse.json", "w") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
    return report
