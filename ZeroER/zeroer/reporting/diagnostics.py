"""Feature treatments and component diagnostics for a fitted ZeroER model.

``diagnose`` refreshes the final E-step so parameters, log-ratios and labels refer
to the same state. It mutates the fitted model; scores should be recorded first.
"""
import numpy as np
from scipy.stats import multivariate_normal

def _blocks(columns):
    """{attribute: its column names}, grouped the way ``zeroer.model`` groups them."""
    blocks = {}
    for name in columns:
        blocks.setdefault(name.split("_")[0], []).append(name)
    return blocks


def treat(features, present, treatment, key):
    """The feature matrix one treatment fits, constant columns dropped."""
    features = features.astype(float)
    if treatment.startswith("key_attribute"):
        features = features[_blocks(features.columns)[key]]
    if treatment.endswith("mean_impute"):
        features = features.copy()
        for attr, columns in _blocks(features.columns).items():
            absent = ~present[attr].to_numpy()
            if absent.any() and not absent.all():
                features.loc[absent, columns] = features.loc[~absent, columns].mean().to_numpy()
    return features.loc[:, features.nunique() > 1]


def _phi(a, b):
    """2x2 counts of two boolean arrays and their phi coefficient."""
    n11, n10 = int((a & b).sum()), int((a & ~b).sum())
    n01, n00 = int((~a & b).sum()), int((~a & ~b).sum())
    denominator = np.sqrt(float(n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return dict(n11=n11, n10=n10, n01=n01, n00=n00,
                phi=(n11 * n00 - n10 * n01) / denominator if denominator else np.nan)


def diagnose(matcher, features, is_gold, present):
    """(per-block rows, confusion rows) for one fitted model."""
    model = matcher.model
    model.e_step()      # refresh: Q_M / Q_U / P_M now follow the final parameters
    predicted = model.P_M >= 0.5
    total = model.Q_M - model.Q_U
    blocks, summed = [], np.zeros(len(total))
    for attr, names in _blocks(features.columns).items():
        cols = [features.columns.get_loc(name) for name in names]
        x, here = model.X[:, cols], present[attr].to_numpy()
        llr = (multivariate_normal.logpdf(x, model.Mu_M[cols], model.Cov_M[np.ix_(cols, cols)],
                                          allow_singular=True)
               - multivariate_normal.logpdf(x, model.Mu_U[cols], model.Cov_U[np.ix_(cols, cols)],
                                            allow_singular=True))
        summed += llr
        zero_row = ~x.any(axis=1)
        blocks.append(dict(
            block=attr, n_columns=len(cols), present_share=here.mean(),
            zero_row_share=zero_row.mean(),
            absent_rows_all_zero=zero_row[~here].mean() if (~here).any() else np.nan,
            present_rows_all_zero=zero_row[here].mean() if here.any() else np.nan,
            mu_M=model.Mu_M[cols].mean(), mu_U=model.Mu_U[cols].mean(),
            std_M=np.sqrt(np.diag(model.Cov_M)[cols]).mean(),
            std_U=np.sqrt(np.diag(model.Cov_U)[cols]).mean(),
            kappa=model.kappas[cols].mean(), kappa_at_cap=(model.kappas[cols] >= 1).mean(),
            llr_all=llr.mean(), llr_gold_match=llr[is_gold].mean(),
            llr_gold_nonmatch=llr[~is_gold].mean(),
            llr_present=llr[here].mean() if here.any() else np.nan,
            llr_absent=llr[~here].mean() if (~here).any() else np.nan))
    cell = dict(pi_M=float(model.pi_M), predicted=int(predicted.sum()),
                labels_changed_by_refresh=int((predicted != (matcher.posteriors >= 0.5)).sum()),
                llr_blocks_vs_total_max_abs=float(np.abs(summed - total).max()))
    # the initial labels too: is the split already there before the first M-step?
    against = dict({"gold": is_gold}, **{attr: present[attr].to_numpy() for attr in present},
                   all_attributes=present.all(axis=1).to_numpy())
    confusion = [dict(cell, labels=name, against=what, **_phi(labels, truth))
                 for name, labels in (("initial", np.asarray(model.y, dtype=bool)),
                                      ("fitted", predicted))
                 for what, truth in against.items()]
    return blocks, confusion
