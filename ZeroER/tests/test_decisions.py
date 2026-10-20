"""Post-fit decisions: endpoint constraints, exact curves and oracle isolation."""
import json

import numpy as np
import pytest

from zeroer.decisions import decode, decision_rows, threshold_curve


def test_no_cut_includes_negative_scores_and_separates_id_spaces():
    pairs = [(0, 1), (1, 0), (0, 0), (2, 2)]
    scores = [-0.1, -0.2, -0.3, -4.0]
    assert decode(pairs, scores, "greedy", None).tolist() == [True, True, False, True]
    assert not decode(pairs, scores, "greedy", 0.0).any()


def test_greedy_does_not_force_maximum_cardinality_or_use_an_endpoint_twice():
    # The best edge blocks both alternatives. There is no augmenting step.
    pairs = [(0, 0), (0, 1), (1, 0)]
    assert decode(pairs, [0.9, 0.8, 0.7], "greedy", None).tolist() == [True, False, False]


def test_ties_are_deterministic_by_ids_and_inclusive():
    pairs = [(1, 1), (0, 1), (1, 0), (0, 0)]
    expected = {(0, 0), (1, 1)}
    for order in ([0, 1, 2, 3], [2, 3, 1, 0]):
        shuffled = [pairs[i] for i in order]
        keep = decode(shuffled, [0.5] * 4, "greedy", 0.5)
        assert {pair for pair, take in zip(shuffled, keep) if take} == expected
    assert decode(pairs, [0.5] * 4, "threshold", 0.5).all()


def test_eligibility_abstains_before_greedy_and_threshold_supports_partial_overlap():
    pairs = [(0, 0), (0, 1), (1, 2), (2, 3)]
    scores = [float("nan"), 0.8, 0.2, -0.1]
    eligible = [False, True, True, False]
    assert decode(pairs, scores, "greedy", None, eligible).tolist() == [False, True, True, False]
    assert decode(pairs, scores, "greedy", 0.5, eligible).tolist() == [False, True, False, False]


@pytest.mark.parametrize("decoder", ["threshold", "greedy"])
def test_exact_curve_matches_brute_force_including_ties_and_missing_gold(decoder):
    rng = np.random.default_rng(17)
    pairs = [(left, right) for left in range(6) for right in range(5)]
    scores = rng.choice([-1.0, 0.0, 0.25, 0.75, 1.0], size=len(pairs))
    gold = np.array([left == right for left, right in pairs])
    eligible = rng.random(len(pairs)) > 0.15
    n_gold = int(gold.sum()) + 2  # two matches were lost during blocking
    curve = threshold_curve(pairs, scores, gold, n_gold, decoder, eligible)
    assert len(curve) == len(set(scores[eligible])) + 1
    assert curve[0]["threshold"] == "+inf"
    for row in curve:
        keep = decode(pairs, scores, decoder, row["threshold"], eligible)
        tp, predicted = int((keep & gold).sum()), int(keep.sum())
        assert row["tp"] == tp
        assert row["predicted"] == predicted
        assert row["f1"] == pytest.approx(2 * tp / (predicted + n_gold))
    json.dumps(curve, allow_nan=False)


def test_oracle_is_labeled_and_fixed_rules_do_not_change_with_gold():
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    scores = np.array([0.9, 0.8, 0.7, -0.1])
    first = decision_rows(pairs, scores, [True, False, False, True], 2, "cosine", [0.0, 0.8])
    second = decision_rows(pairs, scores, [False, True, True, False], 2, "cosine", [0.0, 0.8])
    assert len(first) == 7
    for rows in (first, second):
        assert sum(row["uses_gold"] for row in rows) == 2
        assert all(row["uses_gold"] == (row["rule"] == "oracle_threshold") for row in rows)
        assert all(row["decode_seconds"] >= 0 for row in rows)
        json.dumps(rows, allow_nan=False)
    assert [(r["rule"], r["threshold"], r["predicted"]) for r in first if not r["uses_gold"]] == [
        (r["rule"], r["threshold"], r["predicted"]) for r in second if not r["uses_gold"]]
    pure = next(row for row in first if row["rule"] == "pure_transitivity")
    assert pure["threshold"] is None and pure["predicted"] == 2


def test_oracle_prefers_reject_all_when_all_f1_are_zero():
    rows = decision_rows([(0, 0)], [0.5], [False], 0, "cosine", [0.0])
    oracles = [row for row in rows if row["uses_gold"]]
    assert all(row["threshold"] == "+inf" and row["predicted"] == 0 for row in oracles)
    json.dumps(rows, allow_nan=False)


def test_empty_inputs_and_all_ineligible():
    assert decode([], [], "greedy", None).size == 0
    curve = threshold_curve([], [], [], 2)
    assert len(curve) == 1 and curve[0]["recall"] == 0
    rows = decision_rows([(0, 0)], [float("nan")], [True], 1, "llr", [], [False])
    assert all(row["predicted"] == 0 for row in rows)


@pytest.mark.parametrize("kwargs", [
    dict(pairs=[(0, 0)], scores=[]),
    dict(pairs=[(0, 0)], scores=[0.5], eligible=[]),
    dict(pairs=[(0, 0)], scores=[float("nan")]),
    dict(pairs=[(0, 0)], scores=[0.5], decoder="unknown"),
    dict(pairs=[(0, 0)], scores=[0.5], threshold=None),
    dict(pairs=[(0, 0)], scores=[0.5], threshold=float("nan")),
])
def test_invalid_decoder_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        decode(**kwargs)
