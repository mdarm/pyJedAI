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

Reusable Python code lives in the `zeroer` package. Notebooks remain at the
root as experiment entrypoints; preparation and exploration have separate folders.

| Location | Responsibility |
| --- | --- |
| [zeroer/model.py](zeroer/model.py) | Numerical mixture model, EM fitting loop and final-parameter scoring |
| [zeroer/matcher.py](zeroer/matcher.py) | Training and prediction adapter for pyJedAI; feature validation, initialization and match graphs |
| [zeroer/estimator.py](zeroer/estimator.py) | Assemble blocking, feature generation and matching |
| [zeroer/decisions.py](zeroer/decisions.py) | Threshold and greedy one-to-one decisions on frozen scores |
| [zeroer/data/](zeroer/data/) | `benchmarks.py`: dataset/encoder metadata and loading; `reference.py`: upstream table preparation |
| [zeroer/features/](zeroer/features/) | `strings.py`, `embeddings.py`: similarities; `bank.py`: cached inputs; `panels.py`: fitted feature transforms and record-disjoint splits |
| [zeroer/experiments/](zeroer/experiments/) | `evaluation.py`, `tuning.py`: fixed-candidate evaluation, earlier searches and replays |
| [zeroer/experiments/unsupervised/](zeroer/experiments/unsupervised/) | `runner.py`: label-free selection/refits; `baselines.py`, `refinement.py`; separate fitted, cosine and string-baseline timing modules |
| [zeroer/reporting/](zeroer/reporting/) | `tables.py`: stored-result queries/plots; `metrics.py`: retrieval metrics; `diagnostics.py`: component inspection |
| [zeroer/config.py](zeroer/config.py) | Shared paths and frozen retrieval settings; Python 3.6 compatible |
| [prepare/](prepare/) | Commands that build cached experiment inputs |
| [analysis/](analysis/) | Exploratory notebooks, exports and Streamlit dashboards |

Experiment configurations and selection rules belong in the notebooks. Shared
experiment orchestration belongs in `zeroer.experiments`; numerical computation
stays in `zeroer.model`. Cached-candidate experiments call the matcher directly.
Use `ZeroEREstimator` when the experiment also generates candidates.

```python
from zeroer.matcher import ZeroERMatcher
from zeroer.estimator import ZeroEREstimator
from zeroer.experiments.unsupervised.runner import run_search, run_refit
```

The fitting path is candidate pairs → feature preparation → `ZeroERMatcher` →
`ZeroerModel.run_em`. For label-free selection, `PanelTransform` learns feature
statistics on training rows and reuses them on validation rows. Prediction from
that fitted state is `PanelTransform.transform` → `model.score_components` →
`decisions.decode`; it needs no gold labels and does not run EM again.

In the clean-clean experiments, pairs have shape `(n_pairs, 2)`, with right-side
row IDs offset by the left-table size. Feature rows and boolean presence masks retain the same pair order.
Feature names identify attribute blocks; an absent block is ignored only when
`missing="ignore"`. Final scoring returns row-aligned LLRs, posteriors and marginal
log densities. The legacy matcher graph retains last-E-step posteriors ≥ .5;
`matcher.posteriors` retains all candidate probabilities. Gold is used separately
for evaluation.

The existing databases and caches keep their paths: blocking results in
`outputs/optuna.db`, matching results in `outputs/optuna_matching.db`, Magellan
features in `datasets/magellan-features/`, and cosine/L∞ features in
`outputs/cosine_bank/`. Full-data results include tuning records and are
development results.

The move preserves artifact paths and stored notebook outputs. Code fingerprints
now read the packaged sources; existing provenance checks still require a new
experiment name when source code differs from a recorded run.

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

Run checks with `../../data-env/bin/python -m pytest -q`. The full suite includes
benchmark refits and feature extraction. To check the package without replaying
those experiments, use the synthetic/mocked suites:

```bash
../../data-env/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_unsupervised_*.py tests/test_decisions.py \
  tests/test_primary_candidates.py tests/test_cached_reference_features.py
```
