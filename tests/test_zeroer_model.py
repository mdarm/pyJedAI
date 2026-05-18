"""Unit test for the ported ZeroerModel.

Builds a synthetic similarity matrix with two well-separated Gaussian
clusters (matches near 1.0, non-matches near 0.0) and asserts that
run_em recovers the cluster assignment with high F1.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from pyjedai.unsupervised_matching import ZeroerModel, get_y_init_given_threshold


def zeroer_sanity_check():
    rng = np.random.default_rng(0)
    n_match, n_unmatch, n_feat = 200, 200, 4

    match = np.clip(rng.normal(loc=0.9, scale=0.05, size=(n_match, n_feat)), 0, 1)
    unmatch = np.clip(rng.normal(loc=0.1, scale=0.05, size=(n_unmatch, n_feat)), 0, 1)
    X = np.vstack([match, unmatch])
    y_true = np.concatenate([np.ones(n_match, dtype=int), np.zeros(n_unmatch, dtype=int)])

    perm = rng.permutation(X.shape[0])
    X, y_true = X[perm], y_true[perm]

    feature_names = [f"attr{i}_score" for i in range(n_feat)]
    y_init = get_y_init_given_threshold(pd.DataFrame(X))

    _, P_M = ZeroerModel.run_em(
        similarity_matrixs=(X, None, None),
        feature_names=feature_names,
        y_inits=(y_init, None, None),
        id_dfs=(None, None, None),
        LR_dup_free=False,
        LR_identical=False,
        run_trans=False,
        c_bay=0.1,
        max_iter=40,
    )

    y_pred = (P_M >= 0.5).astype(int)
    assert f1_score(y_true, y_pred) >= 0.95
