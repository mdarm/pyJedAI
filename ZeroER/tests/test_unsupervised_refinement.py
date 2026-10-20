"""Refinement reuses the verified grid and runs only genuinely new settings."""
import optuna
import pytest

import unsupervised as core
from unsupervised_refinement import run_refinement
from test_unsupervised_runner import setup


def grids(args):
    base = args["configs"]["good"]
    original = {"{}:{}".format(c, init): dict(base, c_bay=c, init_statistic=init)
                for c in (.001, .01, .1) for init in ("sum", "present_mean")}
    extra = {"{}:{}".format(c, init): dict(base, c_bay=c, init_statistic=init)
             for c in (0, .0001) for init in ("sum", "present_mean")}
    return original, extra


def test_six_copied_four_fitted_selected_and_resume_is_idempotent(setup, monkeypatch):
    args, _, fitted = setup
    original, extra = grids(args)
    source = core.run_search(**dict(args, configs=original))
    before = len(fitted)
    monkeypatch.setattr(core, "_load_gold", lambda *args: pytest.fail("refinement read gold"))
    refined = run_refinement(source.study_name, args["storage"], extra)
    assert len(refined.trials) == 10
    assert len(fitted) - before == 4 * 2  # four new configurations, two fixture folds
    assert refined.best_trial.params["c_bay"] == 0
    assert refined.best_value == 0
    assert len(refined.user_attrs["spec"]["configs"]) == 10
    for old, copied in zip(source.trials, refined.trials[:6]):
        assert copied.params == old.params and copied.value == old.value
        assert copied.user_attrs["folds"] == old.user_attrs["folds"]
        assert copied.user_attrs["copied"] is True
        assert copied.user_attrs["source_study"] == source.study_name
        assert copied.user_attrs["source_trial"] == old.number
        assert set(copied.distributions["c_bay"].choices) == {0, .0001, .001, .01, .1}
    before = len(fitted)
    resumed = run_refinement(source, args["storage"], extra)
    assert len(resumed.trials) == 10 and len(fitted) == before
    assert len(source.trials) == 6


@pytest.mark.parametrize("changed", ["code", "input", "split"])
def test_changed_source_evidence_is_rejected_before_copying(setup, monkeypatch, changed):
    args, cell, fitted = setup
    original, extra = grids(args)
    source = core.run_search(**dict(args, configs=original))
    if changed == "code":
        monkeypatch.setattr(core, "_code_hash", lambda: "different")
    elif changed == "input":
        cell["input_hash"] = "different"
    else:
        source.set_user_attr("splits", [])
    before = len(fitted)
    with pytest.raises(ValueError, match="changed"):
        run_refinement(source, args["storage"], extra)
    assert len(fitted) == before
    assert len(optuna.study.get_all_study_summaries(storage=args["storage"])) == 1


def test_unfinished_grid_and_mismatched_keys_are_rejected(setup):
    args, _, _ = setup
    original, extra = grids(args)
    source = core.run_search(**dict(args, configs=original))
    malformed = {name: dict(config, unexpected=1) for name, config in extra.items()}
    with pytest.raises(ValueError, match="same parameter keys"):
        run_refinement(source, args["storage"], malformed)
    source.enqueue_trial(original[next(iter(original))])
    with pytest.raises(ValueError, match="six finished"):
        run_refinement(source, args["storage"], extra)


def test_failed_original_is_preserved_and_never_retried(setup):
    args, _, fitted = setup
    original, extra = grids(args)
    original[next(iter(original))]["c_bay"] = -.5
    source = core.run_search(**dict(args, configs=original))
    assert source.trials[0].state == optuna.trial.TrialState.FAIL
    before = len(fitted)
    refined = run_refinement(source, args["storage"], extra)
    assert refined.trials[0].state == optuna.trial.TrialState.FAIL
    assert refined.trials[0].user_attrs["copied"]
    assert len(fitted) - before == 8
