"""Record-space export for assessing blocking quality.

``datasets/tsne_dashboard.py`` plots *pair* space — ``|u-v|`` and the ZeroER
similarity vector — which is the geometry a **matcher** sees. This module builds
the complementary view: **record** space, where k-NN blocking actually operates.
A gold pair is retrievable iff one endpoint falls inside the other's ``k``
nearest neighbours, and that is a statement about where records sit, not about
pair differences. A pair can be cleanly separable in ``|u-v|`` space and still
never be retrieved; only the record-space view shows why.

Per (dataset, embedding model) it exports four tidy frames:

``records``     one row per record — 2-D coordinates plus the source attributes
``gold``        one row per gold pair — the rank at which each side retrieves it
``neighbours``  one row per (query, rank) — what the blocker actually returned
``meta``        one row per (dataset, model) — collection sizes, for the KPIs

Ranks are always computed on the **full** vectors. Only the *plotted* records are
subsampled (large right-hand collections are unplottable), so subsampling can
never change a rank or a KPI — it only hides points.

Run ``python blocking_space.py`` to build the default export, then
``streamlit run datasets/blocking_dashboard.py``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data import DATASETS, load_data, load_embeddings
from estimator import nn_search
from tuning import OUTPUT_DIR

__all__ = ["build", "export", "gold_ranks", "OUT_DIR"]

OUT_DIR = os.path.join(OUTPUT_DIR, "blocking_space")

# Rank far enough out to cover the sweep's k range; the neighbour list kept for
# the "what did it retrieve instead" panel is much shorter.
K_RANK = 100
K_NEIGHBOURS = 10
# Records drawn per collection. t-SNE on 64k points is neither fast nor legible;
# gold endpoints are always kept, the remainder is a seeded sample.
PLOT_MAX = 6000
DEFAULT_MODELS = ("tencent_KaLM-Embedding-Gemma3-12B-2511",
                  "codefuse-ai_F2LLM-v2-8B",
                  "nvidia_llama-embed-nemotron-8b")   # two strong + one anomaly


def _tsne_2d(X, seed=0):
    """PCA-then-t-SNE, as in the pair-space notebook.

    No feature standardisation here: these vectors are L2-normalised and live on
    a sphere, so rescaling per dimension would distort the very geometry the
    k-NN search uses.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=seed).fit_transform(X)
    perp = max(5.0, min(30.0, (len(X) - 1) / 3.0))
    return TSNE(n_components=2, perplexity=perp, init="pca",
                learning_rate="auto", random_state=seed).fit_transform(X)


def gold_ranks(nbrs, data, queried_side, limit):
    """1-based rank at which each gold pair enters the candidate set.

    ``nbrs.shape[1] + 1`` means "never retrieved within the searched range".
    k-NN lists are nested in k, so a pair is in the candidate set at every
    ``k >= its rank`` — one pass therefore yields PC at *every* k.

    Same computation as the blocking sweep's ``_gold_ranks``; the export asserts
    the two agree before writing.
    """
    n_k = nbrs.shape[1]
    ranks = np.full(len(data.ground_truth), n_k + 1, dtype=np.int64)
    for r, (_, (id1, id2)) in enumerate(data.ground_truth.iterrows()):
        i1 = data._ids_mapping_1[id1]              # left-table row index
        i2 = data._ids_mapping_2[id2] - limit      # right-table row index
        row, target = (i2, i1) if queried_side == "right" else (i1, i2)
        hit = np.flatnonzero(nbrs[row] == target)
        if hit.size:
            ranks[r] = hit[0] + 1
    return ranks


def _plot_rows(n, keep, plot_max, seed):
    """Row indices to draw: every gold endpoint, then a seeded fill to ``plot_max``."""
    keep = np.asarray(sorted(keep), dtype=np.int64)
    if n <= plot_max:
        return np.arange(n)
    room = max(plot_max - len(keep), 0)
    rest = np.setdiff1d(np.arange(n), keep, assume_unique=False)
    if room and len(rest):
        rng = np.random.default_rng(seed)
        rest = rng.choice(rest, min(room, len(rest)), replace=False)
    else:
        rest = np.empty(0, dtype=np.int64)
    return np.union1d(keep, rest)


def build(dataset, model, distance="cosine", k_rank=K_RANK,
          k_neighbours=K_NEIGHBOURS, plot_max=PLOT_MAX, seed=0):
    """The four frames for one (dataset, model). See the module docstring."""
    data = load_data(dataset)
    attrs = DATASETS[dataset]["attributes"]
    limit = data.dataset_limit
    v1, v2 = load_embeddings(dataset, model, data)

    # --- ranks, on the full vectors, both directions ------------------------
    # nn_search indexes its first argument and queries its second, so the
    # queried side is the second one.
    ranks, nbrs = {}, {}
    for side, (idx, qry) in (("right", (v1, v2)), ("left", (v2, v1))):
        nbrs[side] = nn_search(idx, qry, k_rank, distance)
        ranks[side] = gold_ranks(nbrs[side], data, side, limit)

    gt = data.ground_truth
    gold = pd.DataFrame({
        "left_row":  [data._ids_mapping_1[i] for i in gt.iloc[:, 0]],
        "right_row": [data._ids_mapping_2[i] - limit for i in gt.iloc[:, 1]],
        "left_id":   gt.iloc[:, 0].to_numpy(),
        "right_id":  gt.iloc[:, 1].to_numpy(),
        "rank_left":  ranks["left"],      # rank when querying the left table
        "rank_right": ranks["right"],
    })

    # --- which records to draw ----------------------------------------------
    plot = {"left":  _plot_rows(len(v1), set(gold.left_row), plot_max, seed),
            "right": _plot_rows(len(v2), set(gold.right_row), plot_max, seed)}

    # One t-SNE over both collections together: the two tables must share a
    # space, or "is b near a" is not a question the picture can answer.
    stacked = np.vstack([np.asarray(v1)[plot["left"]],
                         np.asarray(v2)[plot["right"]]])
    coords = _tsne_2d(stacked, seed)
    n_left_plotted = len(plot["left"])

    frames = []
    for side, table, id_col, off in (
            ("left", data.dataset_1, data.id_column_name_1, 0),
            ("right", data.dataset_2, data.id_column_name_2, n_left_plotted)):
        n = len(table)
        f = pd.DataFrame({"side": side, "row": np.arange(n),
                          "id": table[id_col].astype(str).to_numpy()})
        for a in attrs:
            f[a] = table[a].astype(str).to_numpy() if a in table else ""
        f["x"] = np.nan
        f["y"] = np.nan
        rows = plot[side]
        f.loc[rows, "x"] = coords[off:off + len(rows), 0]
        f.loc[rows, "y"] = coords[off:off + len(rows), 1]
        f["plotted"] = f.index.isin(rows)
        frames.append(f)
    records = pd.concat(frames, ignore_index=True)

    # --- what each plotted query actually retrieved -------------------------
    nb = []
    for side in ("left", "right"):
        m = nbrs[side][:, :k_neighbours]
        q = plot[side]                       # queries are rows of the queried side
        nb.append(pd.DataFrame({
            "side": side,
            "query_row": np.repeat(q, m.shape[1]),
            "rank": np.tile(np.arange(1, m.shape[1] + 1), len(q)),
            "target_row": m[q].ravel(),
        }))
    neighbours = pd.concat(nb, ignore_index=True)

    meta = pd.DataFrame([dict(
        dataset=dataset, model=model, distance=distance,
        n_left=len(v1), n_right=len(v2), n_gold=len(gold),
        k_rank=k_rank, plotted_left=len(plot["left"]),
        plotted_right=len(plot["right"]),
    )])

    for f in (records, gold, neighbours):
        f.insert(0, "model", model)
        f.insert(0, "dataset", dataset)
    return records, gold, neighbours, meta


def export(datasets=None, models=DEFAULT_MODELS, out_dir=OUT_DIR, **kw):
    """Build every (dataset, model) pair and write the four parquets.

    Ranks are cross-checked against the blocking sweep's stored ``tp`` counts
    where the study exists, so a silent misalignment between the embeddings and
    the ground truth cannot reach the dashboard.
    """
    os.makedirs(out_dir, exist_ok=True)
    parts = {"records": [], "gold": [], "neighbours": [], "meta": []}
    for dataset in (datasets or list(DATASETS)):
        for model in models:
            print(f"  {dataset} / {model} ...", flush=True)
            r, g, n, m = build(dataset, model, **kw)
            for key, frame in zip(parts, (r, g, n, m)):
                parts[key].append(frame)
    for key, frames in parts.items():
        path = os.path.join(out_dir, f"{key}.parquet")
        pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    export()
