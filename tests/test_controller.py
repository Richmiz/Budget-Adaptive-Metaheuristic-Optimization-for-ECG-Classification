"""Decision rules of the marginal-utility controller (protocol §3.3-3.4)."""

import math

import pytest

from bamo.search.controller import ControllerSettings, MarginalUtilityPolicy


def policy(**kw):
    return MarginalUtilityPolicy(ControllerSettings(**kw))


def finish_warmup(p, reward=0.0):
    while p.in_warmup:
        _, block, _ = p.choose()
        p.record(block, reward)


def test_reward_normalizes_by_reference_cost():
    p = policy()
    r, dc = p.reward(delta_m=0.001, round_seconds=120.0, n_evals=2, t_ref=60.0)
    assert dc == pytest.approx(2.0) and r == pytest.approx(0.0005)
    r, _ = p.reward(delta_m=-0.01, round_seconds=120.0, n_evals=2, t_ref=60.0)
    assert r == 0.0


def test_reward_without_cost_normalization_uses_evaluation_count():
    p = policy(cost_normalization=False)
    r, dc = p.reward(0.001, round_seconds=999.0, n_evals=2, t_ref=60.0)
    assert dc == 2.0 and r == pytest.approx(0.0005)


def test_proposal_example_block_b_has_higher_rate():
    # Proposal §3.3.4: A gains 0.001 in 300 s, B gains 0.0008 in 100 s.
    p = policy()
    ra, _ = p.reward(0.001, 300.0, 2, 60.0)
    rb, _ = p.reward(0.0008, 100.0, 2, 60.0)
    assert rb > ra


def test_warmup_is_interleaved_three_rounds_each():
    p = policy()
    seq = []
    while p.in_warmup:
        action, block, _ = p.choose()
        assert action == "warmup"
        seq.append(block)
        p.record(block, 0.0)
    assert seq == ["A", "B", "C", "J"] * 3


def test_score_formula_and_sigma_floor():
    p = policy()
    finish_warmup(p, 0.0)
    expected = 1.0 * 1e-4 * math.sqrt(2 * math.log(1 + 12) / (1 + 3))
    assert p.score("A") == pytest.approx(expected)


def test_switch_respects_hysteresis():
    p = policy(mandatory_probes=False)
    finish_warmup(p)
    p.current = "A"
    p.rewards["B"].extend([2e-5, 2e-5])  # B slightly better, within h = 5e-5
    assert p.choose()[:2] == ("continue", "A")
    p.rewards["B"].extend([0.01])
    action, block, _ = p.choose()
    assert (action, block) == ("switch", "B")


def test_mandatory_probe_picks_longest_unvisited_block():
    p = policy()
    finish_warmup(p)
    for _ in range(9):
        p.record("A", 0.01)
    action, block, _ = p.choose()
    assert action == "probe" and block == "B"


def test_stop_needs_streak_rounds_and_min_budget_fraction():
    p = policy()
    finish_warmup(p, 0.0)
    for _ in range(3):
        _, block, _ = p.choose()
        p.record(block, 0.0)
    assert p.streak >= 3
    assert not p.stop_proposed(elapsed_s=0.39 * 1000, budget_s=1000)
    assert p.stop_proposed(elapsed_s=0.40 * 1000, budget_s=1000)


def test_stop_not_proposed_while_a_block_is_productive():
    p = policy()
    finish_warmup(p, 0.0)
    p.record("C", 0.01)
    for _ in range(3):
        p.record("A", 0.0)
    assert not p.stop_proposed(1000, 1000)


def test_round_robin_cycles_after_warmup():
    p = policy(adaptive=False)
    finish_warmup(p)
    blocks = []
    for _ in range(8):
        action, block, _ = p.choose()
        assert action == "round_robin"
        blocks.append(block)
        p.record(block, 0.0)
    assert blocks == ["A", "B", "C", "J"] * 2
