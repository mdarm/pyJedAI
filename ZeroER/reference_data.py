"""Benchmark tables with upstream Magellan preparation, also under Python 3.6."""
from os.path import join

import pandas as pd

from config import BENCH_DIR

PREP = {
    "fodors_zagats": {},
    "dblp_acm": {},
    "dblp_scholar": {},
    "abt_buy": {"concat": ("description", "manufacturer")},
    "amazon_googleproducts": {"rename": {"name": "title"}},
}


def load_reference_tables(dataset, bench_dir=BENCH_DIR):
    """This repo's two tables, UTF-8, with upstream's prep applied."""
    spec = PREP[dataset]
    left = pd.read_csv(join(bench_dir, dataset, "left.csv"), encoding="utf-8")
    right = pd.read_csv(join(bench_dir, dataset, "right.csv"), encoding="utf-8")
    if "rename" in spec:
        right = right.rename(columns=spec["rename"])
    if "concat" in spec:
        target, source = spec["concat"]
        right[target] = right[target] + " " + right[source]
    return left, right
