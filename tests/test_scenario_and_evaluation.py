"""Szenario (wortgleich aus der Hill-Climbing-Demo, eingefrorene Werte) und Auswertung (Kennzahlen, Urteil, Chained
LK gegen Tiefe 1 und gegen unabhängige Neustarts, Sweeps, Streuung)."""

from dataclasses import replace

import numpy as np
import pytest

import lk_constants as C
import lk_evaluation as ev
import lk_scenario as S
import lk_tour as A


# --- Szenario ---------------------------------------------------------------------------------------------------------------------------------


def test_instance_shape_depot_and_area():
    inst = S.generate(60, 0, 3)
    assert inst.xy.shape == (61, 2) and inst.n == 60 and inst.n_nodes == 61
    assert inst.xy[0].tolist() == [50.0, 50.0]
    assert inst.xy.min() >= 0.0 and inst.xy.max() <= C.AREA


def test_instance_is_deterministic_seed_dependent_and_matches_the_frozen_bound():
    a, b, c = S.generate(40, 25, 5), S.generate(40, 25, 5), S.generate(40, 25, 6)
    assert np.array_equal(a.xy, b.xy) and not np.array_equal(a.xy, c.xy)
    inst, D = ev.instance(60, 0, 100000)
    assert inst.xy[0].tolist() == [50.0, 50.0] and float(inst.xy[1:].sum()) == pytest.approx(float(S.generate(60, 0, 100000).xy[1:].sum()))
    assert ev.reference_bound(60, 0, 100000) == pytest.approx(618.76, abs=0.05)


def test_grouped_stops_lie_closer_together_than_uniform_ones():
    def mean_nn(share):
        vals = []
        for seed in range(10):
            xy = S.generate(80, share, seed).xy[1:]
            d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            vals.append(d.min(axis=1).mean())
        return float(np.mean(vals))
    assert mean_nn(100) < 0.7 * mean_nn(0)


# --- Analyse ------------------------------------------------------------------------------------------------------------------------------------


def test_analysis_fields_are_consistent():
    a = ev.analyse(ev.Settings(budget=20000))
    run = a.run
    assert a.bound < run.best_length <= run.final_length + 1e-9
    assert a.gap == pytest.approx(100 * (run.best_length - a.bound) / a.bound) and a.final_gap >= a.gap - 1e-9
    assert a.start_gap > a.single_gap - 5.0                         # die Startlösung ist meist deutlich schlechter als ein Abstieg
    assert run.evaluations >= a.settings.budget
    assert 0.0 <= a.accept_rate <= 1.0


def test_analysis_is_deterministic_and_chain_seed_matters():
    s = ev.Settings(n=30, budget=20000)
    a, b, c = ev.analyse(s), ev.analyse(s), ev.analyse(replace(s, chain_seed=1))
    assert np.array_equal(a.run.best_tour, b.run.best_tour) and a.gap == b.gap
    assert a.gap != c.gap or not np.array_equal(a.run.final_tour, c.run.final_tour)


def test_depth_one_settings_make_run_and_chain1_identical():
    """Bei max_depth=1 fallen `run` und die interne Tiefe-1-Vergleichsgröße `chain1` zusammen (beide rechnen exakt
    denselben Chained-LK-Lauf) - ein struktureller Selbst-Test, kein Zufall."""
    a = ev.analyse(ev.Settings(n=25, budget=15000, max_depth=1))
    assert a.gap == pytest.approx(a.chain1_gap, abs=1e-9)
    assert np.array_equal(a.run.best_tour, a.chain1.best_tour)


def test_without_baselines_the_comparison_fields_are_empty_and_fast():
    a = ev.analyse(ev.Settings(n=20, budget=2000), with_baselines=False)
    assert a.chain1 is None and a.single is None


def test_lk_restarts_uses_at_least_one_descent_and_stays_near_the_budget():
    inst, D = ev.instance(40, 0, 100000)
    cand = ev.candidate_lists(40, 0, 100000)
    best, starts, used = ev.lk_restarts(D, cand, 1, 1000, 0)
    assert starts >= 1 and used >= 1000
    best2, starts2, used2 = ev.lk_restarts(D, cand, 1, 100000, 0)
    assert starts2 >= 3 and best2 <= best + 1e-6


# --- Urteil -------------------------------------------------------------------------------------------------------------------------------------


def _fake(gap, chain1_gap):
    class F:
        pass
    f = F()
    f.gap, f.chain1_gap = gap, chain1_gap
    return f


def test_verdict_codes():
    assert ev.verdict(_fake(1.0, 1.0 + ev.WIN_MARGIN + 0.1)) == "beats_hc"
    assert ev.verdict(_fake(1.0 + ev.LOSE_MARGIN + 0.1, 1.0)) == "hc_wins"
    assert ev.verdict(_fake(1.0, 1.05)) == "comparable"


def test_verdict_of_a_real_run_at_the_default_settings():
    assert ev.verdict(ev.analyse(ev.Settings())) in ("beats_hc", "comparable", "hc_wins")


# --- Sweeps und Tabellen ------------------------------------------------------------------------------------------------------------------


def test_run_config_counts_runs_and_aggregates():
    r = ev.run_config(ev.Settings(n=20, budget=5000))
    assert r["n_runs"] == len(C.SWEEP_SEEDS) * C.SWEEP_CHAINS
    assert r["gap_min"] <= r["gap"] <= r["gap_max"] and r["gap_sd"] >= 0 and r["seconds"] > 0


def test_run_config_ignores_the_seeds_of_the_base_settings():
    a = ev.run_config(ev.Settings(n=15, budget=3000, seed=1, chain_seed=5))
    b = ev.run_config(ev.Settings(n=15, budget=3000, seed=999, chain_seed=0))
    assert all(a[k] == b[k] for k in a if not k.endswith("seconds"))


def test_sweep_values_labels_and_ordering():
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    rows = ev.sweep("budget", ev.Settings(n=20), (2000, 20000))
    assert rows[1]["gap"] <= rows[0]["gap"] + 1.0


def test_restart_sweep_never_uses_a_budget_smaller_than_one_descent():
    rows = ev.restart_sweep("max_depth", ev.Settings(n=20, budget=5000), (1, 2))
    assert len(rows) == 2 and all(r["starts"] >= 1 for r in rows)


def test_scaling_table_structure(monkeypatch):
    monkeypatch.setattr(ev, "SCALING_N", (10, 20))
    tab = ev.scaling_table(ev.Settings(budget=3000))
    assert len(tab) == 2 and all([r["value"] for r in blk["rows"]] == [10, 20] for blk in tab) and tab[0]["label"] != tab[1]["label"]


def test_chain_spread_returns_one_value_per_chain_and_is_deterministic():
    a = ev.chain_spread(ev.Settings(n=15, budget=3000), 5)
    b = ev.chain_spread(ev.Settings(n=15, budget=3000), 5)
    assert len(a["lk"]) == len(a["chain1"]) == 5 and np.array_equal(a["lk"], b["lk"]) and np.array_equal(a["chain1"], b["chain1"])
