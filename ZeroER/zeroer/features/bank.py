"""Read cached per-attribute embedding similarities and pair metadata.

Build with ``python -m prepare.build_embedding_bank``. Existing files remain in
``outputs/cosine_bank/``; rows are sliced by neighbour rank for smaller cells.
"""
from os.path import join

import pandas as pd

from zeroer.config import BANK_DIR, LADDER, cell_name
from zeroer.data.benchmarks import DATASETS


def sides(dataset):
    """{queried side: its largest ladder k} — the cells the bank is built at."""
    out = {}
    for side, k in LADDER[dataset]:
        out[side] = max(k, out.get(side, 0))
    return out


def cell_dir(dataset, side, bank_dir):
    return join(bank_dir, cell_name(dataset, side, sides(dataset)[side]))


def _read_dir(dataset, side, k, bank_dir):
    """Keep ladder slices unchanged; larger cells require their explicit cache."""
    if k <= sides(dataset)[side]:
        return cell_dir(dataset, side, bank_dir)
    return join(bank_dir, cell_name(dataset, side, k))


def load_bank(dataset, side, k, models, attributes=None, metrics=("cos",),
              bank_dir=BANK_DIR):
    """Read a ladder slice, or an explicitly materialized larger cell.

    A missing larger cell raises instead of silently returning fewer neighbors.
    """
    directory = _read_dir(dataset, side, k, bank_dir)
    keep = (pd.read_parquet(join(directory, "pairs.parquet"), columns=["rank"])["rank"] < k).to_numpy()
    attributes = attributes or DATASETS[dataset]["attributes"]
    frames = [pd.read_parquet(
        join(directory, model + ".parquet"),
        columns=["{}_emb_{}__{}".format(a, m, model) for a in attributes for m in metrics])
        for model in models]
    return pd.concat(frames, axis=1)[keep].reset_index(drop=True).astype(float)


def load_presence(dataset, side, k, bank_dir=BANK_DIR):
    """ids, rank and the embedding inputs' ``present_<attr>`` flags of one cell."""
    pairs = pd.read_parquet(join(_read_dir(dataset, side, k, bank_dir), "pairs.parquet"))
    return pairs[pairs["rank"] < k].reset_index(drop=True)
