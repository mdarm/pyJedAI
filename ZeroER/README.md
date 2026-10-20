# ZeroER experiments

Open [hyperparameter-tuning.ipynb](hyperparameter-tuning.ipynb) for experiment settings,
searches and results. Open [zeroer.ipynb](zeroer.ipynb) for selected configurations
refitted on the full candidate sets. Both query stored results by default;
enable the relevant `RUN_*` flag to fit models.

The embedding-only experiments use [unsupervised-tuning.ipynb](unsupervised-tuning.ipynb)
and [unsupervised-results.ipynb](unsupervised-results.ipynb), with a separate
`outputs/optuna_unsupervised.db`. They reuse completed blocking, select configurations
by held-out marginal NLL within each panel, and report fixed and oracle decisions
separately. Original ZeroER baselines use the same candidate sets.
The fixed KaLM blocker's five previously selected operating points are primary;
the 19-point ladder is sensitivity analysis. Both notebooks show the original
selection evidence and distinguish these scopes.
Distribution fitting uses no transitivity constraints; thresholds and greedy
one-to-one matching operate only on frozen scores.

| Code | Responsibility |
| --- | --- |
| `module.py` → `matcher.py` → `estimator.py` | Numerical model, pyJedAI matching adapter, then blocking/matching pipeline |
| `data.py`, `reference_data.py` | Dataset/encoder metadata and loading; upstream table preparation |
| `zeroer_features.py`, `embedding_features.py` | String features, embedding similarities and fusion |
| `embedding_bank.py` | Read cached embedding features and presence flags |
| `evaluation.py`, `metrics.py`, `diagnostics.py` | Fit fixed candidates, score predictions and inspect fitted components |
| `tuning.py` | Shared objectives, resumable sweeps and replays of stored trial parameters |
| `reporting.py` | Database queries, result tables and plots |
| `unsupervised*.py`, `decisions.py` | Label-free selection, fixed-candidate baselines, timing and threshold/greedy decisions |
| `config.py` | Paths and frozen retrieval settings; Python 3.6 compatible |
| `prepare/` | Commands that build cached experiment inputs |
| `analysis/` | Exploratory notebooks, exports and Streamlit dashboards |

Experiment configurations and selection rules belong in the notebooks. Shared
fitting code belongs in `evaluation.py`; an Optuna run uses `tuning.py`.
Cached-candidate experiments call the matcher directly. Use `ZeroEREstimator`
when the experiment also generates candidates.

The existing databases and caches keep their paths: blocking results in
`outputs/optuna.db`, matching results in `outputs/optuna_matching.db`, Magellan
features in `datasets/magellan-features/`, and cosine/L∞ features in
`outputs/cosine_bank/`. Full-data results include tuning records and are
development results.

Run preparation commands from this directory:

```bash
../../data-env/bin/python -m prepare.reference_features pairs
~/micromamba/envs/ZeroER/bin/python -m prepare.reference_features features
../../data-env/bin/python -m prepare.build_embedding_bank
```

The first command retrieves pairs; the second uses the separate Python 3.6
Magellan environment. Set `ZEROER_UPSTREAM` if the upstream checkout is elsewhere.
Each command accepts dataset names after its stage or module name.

For exploration, run `python -m analysis.blocking_space`, then
`streamlit run analysis/blocking_dashboard.py` in the data environment.
[analysis/embedding_space.ipynb](analysis/embedding_space.ipynb) produces the
pair-space exports for `streamlit run analysis/tsne_dashboard.py`.

Run checks with `../../data-env/bin/python -m pytest -q`.
