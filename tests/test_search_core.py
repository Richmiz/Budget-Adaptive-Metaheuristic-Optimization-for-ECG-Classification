"""Search space, budget clock, DE engine and tracker."""

import math

import numpy as np
import pytest

from bamo.config import load_config
from bamo.search.budget import BudgetClock, BudgetExhausted, SimulatedTime
from bamo.search.de import DifferentialEvolution
from bamo.search.evaluator import SharedInitCache, SyntheticEvaluator
from bamo.search.space import SearchSpace, latin_hypercube
from bamo.search.tracker import SearchTracker


def space():
    return SearchSpace.from_config(load_config()["search_space"])


def test_space_blocks_and_roundtrip():
    s = space()
    assert s.dim == 9
    assert list(s.block_indices("A")) == [0, 1]
    assert list(s.block_indices("B")) == [2, 3]
    assert list(s.block_indices("C")) == [4, 5, 6, 7, 8]
    assert list(s.block_indices("J")) == list(range(9))
    rng = np.random.default_rng(0)
    for x in rng.random((50, 9)):
        c = s.decode(x)
        assert s.config_key(s.decode(s.encode(c))) == s.config_key(c)
        assert 1e-5 <= c["learning_rate"] <= 3e-3
        assert c["batch_size"] in (32, 64, 128)


def test_choice_bins_are_equal_width():
    s = space()
    p = s.params[s.names.index("kernel_size")]
    assert [p.decode(u) for u in (0.0, 0.32, 0.34, 0.66, 0.67, 1.0)] == [5, 5, 9, 9, 15, 15]


def test_latin_hypercube_strata():
    u = latin_hypercube(8, 9, np.random.default_rng(1))
    for j in range(9):
        assert sorted(np.floor(u[:, j] * 8).astype(int)) == list(range(8))


def test_budget_clock_and_charge():
    ts = SimulatedTime()
    clock = BudgetClock(100.0, ts).start()
    ts.advance(40)
    clock.charge(30)
    assert clock.elapsed() == pytest.approx(70)
    with pytest.raises(BudgetExhausted):
        clock.charge(31)
    assert clock.exhausted()
    with pytest.raises(BudgetExhausted):
        clock.check()


def test_de_trial_only_changes_active_block():
    rng = np.random.default_rng(3)
    de = DifferentialEvolution(9, 8, rng=rng)
    de.initialize(rng.random((8, 9)), rng.random(8))
    active = np.array([4, 5, 6, 7, 8])
    for _ in range(100):
        t = de.next_target()
        trial = de.make_trial(t, active)
        inactive = np.setdiff1d(np.arange(9), active)
        assert np.array_equal(trial[inactive], de.X[t][inactive])
        assert ((trial >= 0) & (trial <= 1)).all()


def test_de_selection_is_greedy_and_handles_invalid():
    rng = np.random.default_rng(4)
    de = DifferentialEvolution(9, 8, rng=rng)
    de.initialize(rng.random((8, 9)), [0.8] * 8)
    assert not de.select(0, np.zeros(9), None)
    assert not de.select(0, np.zeros(9), 0.7)
    assert de.select(0, np.zeros(9), 0.9)
    assert de.fitness[0] == 0.9


def test_tracker_incumbent_cache_and_shared_charge(tmp_path):
    s = space()
    ts = SimulatedTime()
    clock = BudgetClock(10_000, ts).start()
    ev = SyntheticEvaluator(ts, base_cost_s=60)
    cache = SharedInitCache(tmp_path / "seed.json")
    tr = SearchTracker(s, ev, clock, train_seed=1, max_fidelity=20, shared_cache=cache)
    x = np.full(9, 0.5)
    r1 = tr.evaluate(x=x, tag="init")
    assert tr.incumbent["fitness"] == r1.fitness
    t_after = clock.elapsed()
    tr.evaluate(x=x, tag="search")                       # identical config -> run cache, no cost
    assert clock.elapsed() == pytest.approx(t_after)
    tr.evaluate(x=x, seed=99, tag="reeval")              # other seed never becomes the incumbent
    assert tr.incumbent["index"] == 0
    tr.evaluate(x=x, fidelity=5, tag="hb")               # low fidelity never becomes the incumbent
    assert tr.incumbent["index"] == 0

    ts2 = SimulatedTime()
    clock2 = BudgetClock(10_000, ts2).start()
    tr2 = SearchTracker(s, SyntheticEvaluator(ts2), clock2, 1, 20, SharedInitCache(tmp_path / "seed.json"))
    r2 = tr2.evaluate(x=x, tag="init")                   # served from the shared cache, cost charged
    assert r2.fitness == r1.fitness
    assert ts2.now() == 0.0 and clock2.elapsed() == pytest.approx(r1.cost_s)
    assert tr2.records[0]["source"] == "shared_cache"


def test_evaluator_interrupts_at_cap():
    s = space()
    ts = SimulatedTime()
    clock = BudgetClock(30, ts).start()
    tr = SearchTracker(s, SyntheticEvaluator(ts, base_cost_s=60), clock, 1, 20)
    with pytest.raises(BudgetExhausted):
        tr.evaluate(x=np.full(9, 0.5))
    assert tr.incumbent is None and not tr.records
    assert math.isclose(clock.elapsed(), 30)
