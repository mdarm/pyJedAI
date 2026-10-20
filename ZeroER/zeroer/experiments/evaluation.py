"""Load fixed candidates, fit ZeroER, and score against the full gold standard."""
import hashlib
import json
import time
from os.path import join

import networkx as nx
import numpy as np
import pandas as pd

from zeroer.config import BENCH_DIR, FEATURES_DIR, LADDER, MODEL, cell_name
from zeroer.data.benchmarks import embedding_models, load_data, load_embeddings
from zeroer.features.bank import load_bank, load_presence
from zeroer.features.embeddings import fuse
from zeroer.matcher import ZeroERMatcher
from zeroer.data.reference import load_reference_tables

# These stored feature groups are excluded from matching; class leaks gold.
DROP_GROUPS = {"fodors_zagats": ["type", "class"]}


def _f1(tp, predicted, n_gold):
    return 2 * tp / (predicted + n_gold) if predicted + n_gold else 0.0


def config_hash(params):
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


def blocking_cosine(dataset, stored):
    """The blocker's whole-record cosine of every stored pair."""
    data = load_data(dataset)
    v1, v2 = load_embeddings(dataset, MODEL, data)
    rows_l = pd.Series(np.arange(len(v1)), index=data.dataset_1[data.id_column_name_1])
    rows_r = pd.Series(np.arange(len(v2)), index=data.dataset_2[data.id_column_name_2])
    u = v1[rows_l[stored["ltable_id"]].to_numpy()]
    v = v2[rows_r[stored["rtable_id"]].to_numpy()]
    return np.einsum("ij,ij->i", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))


def presence(dataset, ids, attributes):
    """Per pair and attribute: is the value there on *both* sides? Read off
    ``zeroer.data.reference.load_reference_tables`` — the tables the stored matrices were built
    from (``abt_buy``'s right ``description`` is upstream's ``description +
    manufacturer`` concat, which propagates NaN). An absent value is an all-zero
    feature block in the stored matrix; this is what tells it from a present
    value that merely disagrees."""
    left, right = load_reference_tables(dataset, BENCH_DIR)
    left, right = (t.set_index(t["id"].astype(str)) for t in (left, right))
    return pd.DataFrame({
        attr: left[attr].reindex(ids["ltable_id"]).notna().to_numpy()
        & right[attr].reindex(ids["rtable_id"]).notna().to_numpy()
        for attr in attributes})


def load_side(dataset, side, partition=None, seed=0, cosine=True, bank=False):
    """Everything the cells of one (dataset, queried side) share, read off the
    stored matrix at that side's largest ``k``.

    ``partition`` — None is every queried record. ``"tune"`` / ``"holdout"`` are
    the two halves of a seeded split of the *queried records*, each kept with its
    whole stored candidate list: a row slice of the stored matrix (no re-blocking,
    no new features), the same PQ as the full cell, and two halves that share no
    queried record. ``n_gold`` is then the gold pairs of the kept queried records —
    including those the blocker missed. ``cosine=False`` skips the blocker's cosine
    (and the record vectors it needs) when no floor is wanted. ``bank=True`` adds
    the cosine bank's columns and presence masks for these pairs — every model,
    ``cos`` — after checking that the bank holds the same pairs in the same order.
    """
    k_max = max(k for s, k in LADDER[dataset] if s == side)
    stored = pd.read_csv(join(FEATURES_DIR, cell_name(dataset, side, k_max) + ".csv.gz"),
                         dtype={"ltable_id": str, "rtable_id": str})
    gt = pd.read_csv(join(BENCH_DIR, dataset, "matches.csv"), dtype=str)
    gold = set(zip(gt.iloc[:, 0], gt.iloc[:, 1]))
    banked = bank_present = None
    if bank:
        bank_rows = load_presence(dataset, side, k_max)
        ids = bank_rows[["ltable_id", "rtable_id"]]
        if not ids.equals(stored[["ltable_id", "rtable_id"]]):
            raise ValueError("{} {}: the cosine bank's pairs are not the stored "
                             "matrix's — rebuild it (python -m prepare.build_embedding_bank)".format(dataset, side))
        banked = load_bank(dataset, side, k_max, embedding_models(dataset))
        bank_present = bank_rows.filter(regex="^present_").rename(
            columns=lambda name: name[len("present_"):])
    if partition is not None:
        queried = 0 if side == "left" else 1          # column of the queried table
        n_queried = len(stored) // k_max
        tune = np.random.default_rng(seed).permutation(n_queried) < n_queried // 2
        kept = tune if partition == "tune" else ~tune
        if bank:
            banked = banked[np.repeat(kept, k_max)].reset_index(drop=True)
            bank_present = bank_present[np.repeat(kept, k_max)].reset_index(drop=True)
        stored = stored[np.repeat(kept, k_max)].reset_index(drop=True)
        kept_ids = set(stored.iloc[:, queried])
        gold = {pair for pair in gold if pair[queried] in kept_ids}

    features = stored.drop(columns=["ltable_id", "rtable_id"])
    group = features.columns.str.split("_").str[0]
    features = features.loc[:, ~group.isin(DROP_GROUPS.get(dataset, []))]

    cos = blocking_cosine(dataset, stored) if cosine else None
    # each queried record's pairs must come nearest-first, or the rank slices
    # below are not the smaller-k candidate sets — and a re-encoded record/ tree
    # under unchanged model names would show up exactly here
    if cosine and (np.diff(cos.reshape(-1, k_max), axis=1) > 1e-4).any():
        raise ValueError(
            "{} {}/{}: stored pairs are not in the blocker's neighbour order — "
            "have the record vectors changed since the pairs were exported?"
            .format(dataset, side, k_max))

    # ids are strings on some datasets and the matcher wants ints: number the
    # records, left and right on disjoint ranges
    left = pd.factorize(stored["ltable_id"])[0]
    right = pd.factorize(stored["rtable_id"])[0] + left.max() + 1
    return dict(k_max=k_max, ids=stored[["ltable_id", "rtable_id"]], left=left,
                right=right, features=features, blocking_cos=cos, n_gold=len(gold),
                bank=banked, bank_present=bank_present,
                present=presence(dataset, stored, dict.fromkeys(
                    features.columns.str.split("_").str[0])),
                rank=np.arange(len(stored)) % k_max,
                is_gold=np.array([pair in gold for pair in
                                  zip(stored["ltable_id"], stored["rtable_id"])]))


def one_to_one(left, right, score):
    """Greedy unique mapping: best-scoring pair first, every record used once."""
    keep = np.zeros(len(score), dtype=bool)
    used = set()
    for i in np.argsort(-score, kind="stable"):
        if left[i] not in used and right[i] not in used:
            keep[i] = True
            used.update((left[i], right[i]))
    return keep


def one_to_one_oracle(left, right, is_gold):
    """The most gold pairs any one-to-one mapping of these candidates can keep —
    below ``is_gold.sum()`` wherever the gold standard is not one-to-one."""
    graph = nx.Graph(zip(left[is_gold], right[is_gold]))
    return len(nx.bipartite.hopcroft_karp_matching(graph, set(left[is_gold]))) // 2


def fit(config, pairs, features):
    """One EM fit -> (posteriors, fit attributes); raises what the EM raises."""
    matcher = ZeroERMatcher(**config)
    matcher.match_pairs(pairs, features)
    model = matcher.model
    return matcher.posteriors, dict(
        init_frac=float(np.mean(model.y)), pi_M=float(model.pi_M),
        n_iter=int(model.n_iter), converged=bool(model.converged))


def informative(features, present=None):
    """Drop the columns that cannot separate anything: constant over the rows
    where their attribute is present (over every row, without ``present``). With
    absent values left out of the fit, a column that only varies between "absent"
    (0) and one present value would be a zero-variance column in disguise."""
    if present is None:
        return features.loc[:, features.nunique() > 1]
    block = features.columns.str.split("_").str[0]
    return features.loc[:, [features.loc[present[b].to_numpy(), c].nunique() > 1
                            for c, b in zip(features.columns, block)]]


def fit_rungs(dataset, loaded, config, attributes=None, panel=None, include_strings=True):
    """One matcher configuration on every named rung of a dataset.

    ``loaded``: {side: ``load_side(...)``}, so a search loads each side once.
    ``attributes`` restricts the matrix to those attribute blocks (None = all).
    ``panel`` — ``dict(models=[...], fusion=...)`` — adds the panel's embedding
    columns (``zeroer.features.embeddings.fuse``) for the same attributes; it needs
    ``load_side(bank=True)``. ``include_strings=False`` uses only those embedding
    columns and their cached presence masks; an empty panel is invalid.
    Returns one dict of raw counts per rung; a fit that raises scores nothing
    (``tp`` = ``predicted`` = 0) and carries its ``error`` — a configuration that
    cannot be fitted is a bad configuration, not a missing value.
    """
    if not include_strings and (panel is None or not panel["models"]):
        raise ValueError("Cosine-only matching requires a nonempty embedding panel")
    rows = []
    for side, k in LADDER[dataset]:
        s = loaded[side]
        keep = s["rank"] < k
        is_gold = s["is_gold"][keep]
        present = s["present" if include_strings else "bank_present"][keep].reset_index(drop=True)
        features = (s["features"][keep].reset_index(drop=True) if include_strings
                    else pd.DataFrame(index=present.index))
        if include_strings and attributes is not None:
            features = features.loc[:, features.columns.str.split("_").str[0].isin(attributes)]
        if panel is not None:
            features = pd.concat([features, fuse(
                s["bank"][keep].reset_index(drop=True), panel["models"],
                attributes or list(present.columns), present, panel["fusion"])], axis=1)
        features = informative(features, present if config.get("missing") == "ignore" else None)
        row = dict(queried_side=side, top_k=k, candset_size=int(keep.sum()),
                   tp_blocking=int(is_gold.sum()), n_gold=s["n_gold"], tp=0, predicted=0,
                   n_features=features.shape[1])
        t0 = time.time()
        try:
            matcher = ZeroERMatcher(**config)
            matcher.match_pairs(list(zip(s["left"][keep], s["right"][keep])), features,
                                present=present)
        except Exception as error:      # newton, an empty seed class, a singular fit
            row.update(error=repr(error))
        else:
            predicted = matcher.posteriors >= 0.5
            row.update(predicted=int(predicted.sum()), tp=int((predicted & is_gold).sum()),
                       brier=float(np.mean((matcher.posteriors - is_gold) ** 2)),
                       pi_M=float(matcher.model.pi_M), n_iter=int(matcher.model.n_iter),
                       converged=bool(matcher.model.converged),
                       # share of columns whose regulariser hit m_step's clip of 1,
                       # i.e. were flattened out of the likelihood ratio
                       kappa_capped=float((matcher.model.kappas >= 1).mean()))
        row.update(f1=_f1(row["tp"], row["predicted"], row["n_gold"]),
                   seconds=round(time.time() - t0, 1))
        rows.append(row)
    return rows


def load_sides(dataset, partition=None, seed=0, bank=False):
    """Load each query direction once for a search or replay."""
    return {side: load_side(dataset, side, partition=partition, seed=seed,
                            cosine=False, bank=bank)
            for side in dict.fromkeys(side for side, _ in LADDER[dataset])}


def evaluate_config(dataset, loaded, config, attributes=None, panel=None, holdout=None,
                    include_strings=True):
    """Mean F1 and stored counts, using the same fit for searches and replays."""
    rungs = fit_rungs(dataset, loaded, config, attributes=attributes, panel=panel,
                      include_strings=include_strings)
    attrs = dict(
        rungs=rungs,
        n_errors=sum("error" in row for row in rungs),
        seconds=round(sum(row["seconds"] for row in rungs), 1),
        candset_size=sum(row["candset_size"] for row in rungs),
        pair_completeness=float(np.mean([row["tp_blocking"] / row["n_gold"] for row in rungs])))
    if holdout is not None:
        attrs["holdout_rungs"] = fit_rungs(
            dataset, holdout, config, attributes=attributes, panel=panel,
            include_strings=include_strings)
    return float(np.mean([row["f1"] for row in rungs])), attrs
