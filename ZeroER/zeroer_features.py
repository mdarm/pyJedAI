"""
Faithful reconstruction of ZeroER's *pre-matching* similarity feature matrix.

Reproduces the path:
    get_features (magellan_modified_feature_generation)
      -> em.extract_feature_vecs
      -> gather_similarity_features
without depending on py_entitymatching. Sim primitives come from
py_stringmatching, which is exactly what Magellan wraps internally, so the
numbers match.

Differences from the 6-feature hand-rolled version are intentional and are the
whole point: type-aware feature lists, both q2 and q3 grams, dice/monge_elkan/
overlap_coeff retained, lev kept as lev_SIM (not 1-sim), numeric attrs get
abs_norm instead of string sims, and the type-upgrade rule for mismatched
attribute types.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import py_stringmatching as sm

# ---- tokenizers (Magellan's defaults: q-grams ARE padded with #/$) --------
_QG2 = sm.QgramTokenizer(qval=2, return_set=True, padding=True)
_QG3 = sm.QgramTokenizer(qval=3, return_set=True, padding=True)
_DLM = sm.DelimiterTokenizer(delim_set=[' '], return_set=True)  # dlm_dc0
_WS  = sm.WhitespaceTokenizer(return_set=False)                 # for type inference

# ---- sim functions --------------------------------------------------------
_COS = sm.Cosine()
_DICE = sm.Dice()
_JAC = sm.Jaccard()
_OC = sm.OverlapCoefficient()
_MEL = sm.MongeElkan()
_LEV = sm.Levenshtein()
_JARO = sm.Jaro()
_JW = sm.JaroWinkler()

# ---------------------------------------------------------------------------
# Attribute type inference, mirroring py_entitymatching.get_attr_types.
# Magellan rule: look at the average number of whitespace tokens per non-null
# value. <=1 word -> str_eq_1w; (1,5] -> str_bt_1w_5w; (5,10] -> str_bt_5w_10w;
# >10 -> str_gt_10w. Numeric dtype -> numeric. Bool -> boolean.
# ---------------------------------------------------------------------------
def infer_attr_type(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return 'un_determined'
    if pd.api.types.is_bool_dtype(s):
        return 'boolean'
    if pd.api.types.is_numeric_dtype(s):
        return 'numeric'
    # try numeric coercion
    coerced = pd.to_numeric(s, errors='coerce')
    if coerced.notna().mean() > 0.5:
        return 'numeric'
    avg_tokens = s.astype(str).map(lambda x: len(_WS.tokenize(x))).mean()
    if avg_tokens <= 1:
        return 'str_eq_1w'
    elif avg_tokens <= 5:
        return 'str_bt_1w_5w'
    elif avg_tokens <= 10:
        return 'str_bt_5w_10w'
    return 'str_gt_10w'


# Type rank for the mismatch-upgrade rule in extract_features.
_TYPE_RANK = {'boolean': 1, 'numeric': 2, 'str_eq_1w': 3, 'str_bt_1w_5w': 4,
              'str_bt_5w_10w': 5, 'str_gt_10w': 6, 'un_determined': 7}


# ---------------------------------------------------------------------------
# Survivor feature lists per type (post distance/non-normalized filter).
# Each entry: (feature_suffix, kind, tokenizer-or-None)
#   kind in {set_sim, mel, lev_sim, jaro, jw, exact, abs_norm}
# Tokenizer key in {'qg2','qg3','dlm', None}
# ---------------------------------------------------------------------------
_SET_SIMS = {'cos': _COS, 'dice': _DICE, 'jac': _JAC, 'oc': _OC}
_TOKENIZERS = {'qg2': _QG2, 'qg3': _QG3, 'dlm': _DLM}

_FEATURES = {
    'str_eq_1w': [
        ('cos_qgm_2', 'set_sim', 'cos', 'qg2'), ('cos_qgm_3', 'set_sim', 'cos', 'qg3'),
        ('dice_qgm_2', 'set_sim', 'dice', 'qg2'), ('dice_qgm_3', 'set_sim', 'dice', 'qg3'),
        ('lev_sim', 'lev_sim', None, None),
        ('jar', 'jaro', None, None), ('jwn', 'jw', None, None),
        ('exm', 'exact', None, None),
        ('mel_qgm_2', 'mel', None, 'qg2'), ('mel_qgm_3', 'mel', None, 'qg3'),
        ('oc_qgm_2', 'set_sim', 'oc', 'qg2'), ('oc_qgm_3', 'set_sim', 'oc', 'qg3'),
        ('jac_qgm_2', 'set_sim', 'jac', 'qg2'), ('jac_qgm_3', 'set_sim', 'jac', 'qg3'),
    ],
    'str_bt_1w_5w': [
        ('cos_dlm', 'set_sim', 'cos', 'dlm'), ('cos_qgm_3', 'set_sim', 'cos', 'qg3'),
        ('dice_dlm', 'set_sim', 'dice', 'dlm'), ('dice_qgm_3', 'set_sim', 'dice', 'qg3'),
        ('lev_sim', 'lev_sim', None, None),
        ('jar', 'jaro', None, None), ('jwn', 'jw', None, None),
        ('exm', 'exact', None, None),
        ('mel_dlm', 'mel', None, 'dlm'), ('mel_qgm_3', 'mel', None, 'qg3'),
        ('oc_dlm', 'set_sim', 'oc', 'dlm'), ('oc_qgm_3', 'set_sim', 'oc', 'qg3'),
        ('jac_dlm', 'set_sim', 'jac', 'dlm'), ('jac_qgm_3', 'set_sim', 'jac', 'qg3'),
    ],
    'str_bt_5w_10w': [
        ('cos_dlm', 'set_sim', 'cos', 'dlm'), ('cos_qgm_3', 'set_sim', 'cos', 'qg3'),
        ('dice_dlm', 'set_sim', 'dice', 'dlm'), ('dice_qgm_3', 'set_sim', 'dice', 'qg3'),
        ('mel_dlm', 'mel', None, 'dlm'), ('mel_qgm_3', 'mel', None, 'qg3'),
        ('oc_dlm', 'set_sim', 'oc', 'dlm'), ('oc_qgm_3', 'set_sim', 'oc', 'qg3'),
        ('jac_dlm', 'set_sim', 'jac', 'dlm'), ('jac_qgm_3', 'set_sim', 'jac', 'qg3'),
    ],
    'str_gt_10w': [
        ('cos_dlm', 'set_sim', 'cos', 'dlm'), ('cos_qgm_3', 'set_sim', 'cos', 'qg3'),
        ('dice_dlm', 'set_sim', 'dice', 'dlm'), ('dice_qgm_3', 'set_sim', 'dice', 'qg3'),
        ('mel_dlm', 'mel', None, 'dlm'), ('mel_qgm_3', 'mel', None, 'qg3'),
        ('oc_dlm', 'set_sim', 'oc', 'dlm'), ('oc_qgm_3', 'set_sim', 'oc', 'qg3'),
        ('jac_dlm', 'set_sim', 'jac', 'dlm'), ('jac_qgm_3', 'set_sim', 'jac', 'qg3'),
    ],
    'numeric': [
        ('exm', 'exact', None, None),
        ('anm', 'abs_norm', None, None),
        ('lev_sim', 'lev_sim', None, None),
    ],
    'boolean': [
        ('exm', 'exact', None, None),
    ],
    'un_determined': [],
}


def _abs_norm(x: str, y: str) -> float:
    try:
        d, e = float(x), float(y)
    except (ValueError, TypeError):
        return 0.0
    if d == 0 and e == 0:
        return 1.0
    denom = max(abs(d), abs(e))
    return 1.0 - abs(d - e) / denom if denom else 0.0


def _compute(kind, sim_key, tok_key, a, b, toks_cache):
    """Vectorized over arrays a,b (raw lowercased strings)."""
    n = len(a)
    out = np.zeros(n)
    if kind == 'exact':
        return np.where((a == b) & (a != ''), 1.0, 0.0)
    if kind == 'abs_norm':
        return np.array([_abs_norm(x, y) for x, y in zip(a, b)])
    if kind == 'lev_sim':
        return np.array([_LEV.get_sim_score(x, y) if (x or y) else 0.0
                         for x, y in zip(a, b)])
    if kind == 'jaro':
        return np.array([_JARO.get_sim_score(x, y) if (x and y) else 0.0
                         for x, y in zip(a, b)])
    if kind == 'jw':
        return np.array([_JW.get_sim_score(x, y) if (x and y) else 0.0
                         for x, y in zip(a, b)])
    # token/ngram based
    ta, tb = toks_cache[tok_key]
    if kind == 'set_sim':
        fn = _SET_SIMS[sim_key]
        return np.array([fn.get_sim_score(x, y) if (x and y) else 0.0
                         for x, y in zip(ta, tb)])
    if kind == 'mel':
        return np.array([_MEL.get_raw_score(x, y) if (x and y) else 0.0
                         for x, y in zip(ta, tb)])
    raise ValueError(kind)


def build_zeroer_features(pairs, entities: pd.DataFrame, attributes,
                          drop_zero_variance: bool = True) -> pd.DataFrame:
    """
    pairs: list of (left_idx, right_idx) row-index tuples into `entities`.
    entities: DataFrame of records (single-table dedup style, as ZeroER uses).
    attributes: shared attribute names to featurize (id-like cols excluded by caller).
    """
    if not pairs:
        return pd.DataFrame()
    li, ri = map(np.asarray, zip(*pairs))
    blocks = []
    for attr in attributes:
        if attr == 'id':            # ZeroER drops id-based features
            continue
        col = entities[attr].astype(str).to_numpy()
        atype = infer_attr_type(entities[attr])
        feats = _FEATURES.get(atype, [])
        if not feats:
            continue

        a, b = col[li], col[ri]

        # Precompute tokenizations once per unique value, per needed tokenizer.
        needed_toks = {f[3] for f in feats if f[3] is not None}
        uniq, inv = np.unique(col, return_inverse=True)
        toks_cache = {}
        for tk in needed_toks:
            tokz = _TOKENIZERS[tk]
            tok_u = [tokz.tokenize(u) if u else [] for u in uniq]
            toks_cache[tk] = ([tok_u[i] for i in inv[li]],
                              [tok_u[i] for i in inv[ri]])

        data = {}
        for suffix, kind, sim_key, tok_key in feats:
            data[f'{attr}_{suffix}'] = _compute(kind, sim_key, tok_key, a, b, toks_cache)
        blocks.append(pd.DataFrame(data))

    if not blocks:
        return pd.DataFrame()
    fm = pd.concat(blocks, axis=1).fillna(0.0)
    if drop_zero_variance:
        fm = fm.loc[:, fm.nunique(dropna=False) > 1]
    return fm


if __name__ == '__main__':
    entities = pd.DataFrame({
        'id':    [0, 1, 2, 3, 4, 5],
        'name':  ['Sony Vaio Laptop', 'Sony Viao Laptop', 'Dell XPS 13',
                  'Apple MacBook Pro 16 inch retina display model',
                  'apple macbook pro 16 inch retina display model',
                  'Dell XPS 13'],
        'brand': ['Sony', 'Sony', 'Dell', 'Apple', 'Apple', 'Dell'],
        'price': ['999', '999', '1200', '1999', '1999', '1200'],
    })
    pairs = [(0, 1), (0, 2), (2, 5), (3, 4), (1, 1)]
    attrs = ['name', 'brand', 'price']

    for a in attrs:
        print(f'{a:6s} -> inferred type: {infer_attr_type(entities[a])}')
    print()

    fm = build_zeroer_features(pairs, entities, attrs)
    pd.set_option('display.width', 240, 'display.max_columns', 60)
    print('shape:', fm.shape)
    print('columns:')
    for c in fm.columns:
        print('   ', c)
    print()
    print(fm.round(3).T)
