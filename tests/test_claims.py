"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Instanzen (je drei Ketten-Seeds) belegt,
mit denselben Auswertungsfunktionen wie die App selbst (`ev.sweep`/`ev.restart_sweep`/`ev.scaling_table`) - ein eigenständiges Ad-hoc-Skript hätte
bei der Erstmessung eine andere (nicht-gepaarte) Zufallsfolge benutzt und eine FALSCHE Geschichte nahegelegt (Tiefe 2 gewinnt angeblich schon ab
50 Tausend) - erst der Test gegen die echte `chained_run`/`analyse`-Implementierung deckte auf, dass der reale Umschlagpunkt bei 500 Tausend liegt.
Positive UND negative Aussagen: beim Standardbudget/der Standardgröße hilft Tiefe 2 NICHT klar - das steht hier ebenso als Test wie die Stellen,
an denen sie klar gewinnt (großes Budget, große Instanz). Rechenzeiten sind nur als Größenordnung geprüft."""

import itertools
from functools import lru_cache

import numpy as np
import pytest

import lk_algorithm as LK
import lk_constants as C
import lk_dlb as D
import lk_evaluation as ev
import lk_tour as A


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Standardfall: bei 60 Stopps/200 Tausend sind Tiefe 1 und 2 praktisch gleichauf ----------------------------------------------------------------


def test_standard_case_numbers():
    std = cfg()
    near(std["gap"], 0.66, 0.35)
    near(std["chain1"], 0.64, 0.35)
    near(std["single"], 6.51, 1.5)
    assert abs(std["gap"] - std["chain1"]) < 0.3                          # praktisch gleichauf, kein klarer Sieg beim Standardfall


# --- Tiefen-Sweep: ein einzelner Abstieg saettigt ab Tiefe 3 -------------------------------------------------------------------------------------


@pytest.mark.parametrize("depth,gap,tol", [(1, 7.05, 1.2), (2, 6.51, 1.2), (3, 6.45, 1.2), (5, 6.45, 1.2), (8, 6.45, 1.2)])
def test_single_descent_depth_numbers(depth, gap, tol):
    near(cfg(max_depth=depth)["single"], gap, tol)


def test_single_descent_saturates_by_depth_three():
    d3 = cfg(max_depth=3)["single"]
    d8 = cfg(max_depth=8)["single"]
    assert abs(d3 - d8) < 0.3                                            # keine weitere Verbesserung ab Tiefe 3


# --- Unabhängige Neustarts: ehrlicher Negativbefund, bei JEDEM Budget --------------------------------------------------------------------------


def test_independent_restarts_never_clearly_benefit_from_more_depth_at_two_hundred_thousand():
    rows = ev.restart_sweep("max_depth", ev.Settings(), (1, 2, 3, 5))
    gaps = {r["value"]: r["gap"] for r in rows}
    near(gaps[1], 0.68, 0.3)
    assert gaps[2] >= gaps[1] - 0.2                                       # Tiefe 2 ist NICHT klar besser als Tiefe 1 bei reinen Neustarts
    assert gaps[3] >= gaps[1] - 0.2 and gaps[5] >= gaps[1] - 0.2          # ebenso wenig Tiefe 3 oder 5


# --- Chained LK: der Budget-Sweep - der reale Umschlagpunkt liegt bei 500 Tausend, nicht bei 50 Tausend ---------------------------------------------


@pytest.mark.parametrize("budget,chain1,chain2,tol", [
    (10000, 1.36, 1.77, 0.5), (25000, 1.03, 1.02, 0.4), (50000, 0.79, 1.02, 0.4), (100000, 0.70, 0.68, 0.3),
    (200000, 0.64, 0.66, 0.3), (500000, 0.60, 0.51, 0.25), (1000000, 0.58, 0.47, 0.25), (2000000, 0.58, 0.47, 0.25),
])
def test_budget_sweep_numbers(budget, chain1, chain2, tol):
    row = cfg(budget=budget)
    near(row["chain1"], chain1, tol)
    near(row["gap"], chain2, tol)


def test_depth_two_loses_clearly_at_ten_thousand_and_wins_clearly_from_five_hundred_thousand_on():
    low = cfg(budget=10000)
    assert low["gap"] > low["chain1"] + 0.2                              # Tiefe 2 verliert klar bei sehr knappem Budget
    high = cfg(budget=500000)
    assert high["gap"] < high["chain1"] - 0.05                           # und gewinnt ab 500 Tausend
    higher = cfg(budget=1000000)
    assert higher["gap"] < higher["chain1"] - 0.05


def test_depth_two_is_not_a_clear_winner_at_the_standard_budget_and_size():
    """Ehrlicher Kernbefund: beim in dieser Linie ueblichen Standardfall (60 Stopps, 200 Tausend) gibt es KEINEN
    klaren Sieger - anders als die Vorab-Vermutung ("tiefere Suche gewinnt immer") nahelegen wuerde."""
    std = cfg(budget=200000)
    assert abs(std["gap"] - std["chain1"]) < 0.3


def test_the_advantage_of_depth_over_chain1_grows_from_five_hundred_thousand_onward():
    mid = cfg(budget=500000)
    large = cfg(budget=2000000)
    mid_edge = mid["chain1"] - mid["gap"]
    large_edge = large["chain1"] - large["gap"]
    assert mid_edge > 0 and large_edge >= mid_edge - 0.05                 # der Vorsprung schrumpft NICHT mehr, sobald er einmal positiv ist


# --- Chained LK skalierung: ab 100 Stopps gewinnt Tiefe 2 schon beim Standardbudget --------------------------------------------------------------


@pytest.mark.parametrize("n,chain1,chain2,tol", [(100, 1.14, 1.06, 0.4), (150, 1.74, 1.46, 0.5), (200, 2.09, 1.82, 0.5)])
def test_scaling_numbers_at_two_hundred_thousand(n, chain1, chain2, tol):
    row = ev.run_config(ev.Settings(), n=n, budget=200000)
    near(row["chain1"], chain1, tol)
    near(row["gap"], chain2, tol)


def test_depth_two_wins_clearly_from_one_hundred_stops_on_at_the_standard_budget():
    for n in (100, 150, 200):
        row = ev.run_config(ev.Settings(), n=n, budget=200000)
        assert row["gap"] < row["chain1"] - 0.05


def test_depth_two_does_not_yet_win_at_sixty_stops_the_standard_budget():
    row = ev.run_config(ev.Settings(), n=60, budget=200000)
    assert row["gap"] >= row["chain1"] - 0.1                              # kein klarer Sieg bei der Standardgröße


# --- Breadth1-Sensitivität ------------------------------------------------------------------------------------------------------------------------


def test_breadth1_one_is_clearly_worse_and_two_and_above_are_close():
    """RCL/Kandidatenzahl auf Ebene 1 ist kein Regler in der App (fix auf CANDIDATE_K=5) - hier direkt gegen
    lk_algorithm.chained_run gemessen, mit demselben start+seed-Muster wie lk_evaluation.analyse()."""
    gaps = {}
    for breadth1 in (1, 2, 5, 8):
        vals = []
        for seed in C.SWEEP_SEEDS:
            inst, Dm = ev.instance(60, 0, seed)
            bound = ev.reference_bound(60, 0, seed)
            cand = D.build_candidate_lists(Dm, max(breadth1, D.CANDIDATE_K))
            for ch in range(C.SWEEP_CHAINS):
                start = A.random_tour(len(Dm), np.random.default_rng(ch))
                r = LK.chained_run(Dm, start, cand, max_depth=2, breadth1=breadth1, budget=200000, seed=ch, keep_snapshots=False)
                vals.append(100 * (r.best_length - bound) / bound)
        gaps[breadth1] = float(np.mean(vals))
    near(gaps[1], 1.29, 0.5)
    assert gaps[1] > gaps[5] + 0.3                                        # breadth1=1 ist klar schlechter als 5
    assert abs(gaps[2] - gaps[8]) < 0.5                                   # ab 2 keine klare weitere Verbesserung


# --- Kleine-Instanz-Demonstration (n=8): 2-opt haengt fest, Tiefe 2 findet das globale Optimum ------------------------------------------------------


def _brute_force_optimum(Dm):
    n = len(Dm)
    best = np.inf
    for perm in itertools.permutations(range(1, n)):
        L = A.tour_length(np.array([0, *perm]), Dm)
        best = min(best, L)
    return best


def test_small_instance_demonstration_two_opt_gets_stuck_and_depth_two_escapes():
    n, seed, start_seed = 8, 1, 6
    rng = np.random.default_rng(seed * 1000 + n)
    xy = rng.random((n, 2)) * 100
    Dm = A.dist_matrix(xy)
    cand = D.build_candidate_lists(Dm, min(D.CANDIDATE_K, n - 1))
    t0 = A.random_tour(n, np.random.default_rng(start_seed))
    opt = _brute_force_optimum(Dm)
    r1 = D.dlb_descend(Dm, t0, cand, seed=start_seed)
    r2 = LK.lk_descend(Dm, t0, cand, max_depth=2, breadth1=D.CANDIDATE_K, seed=start_seed)
    near(opt, 293.21, 0.05)
    near(r1.length, 312.35, 0.05)
    assert r1.length > opt + 1e-6                                        # 2-opt (Tiefe 1) bleibt echt haengen
    assert r2.length == pytest.approx(opt, abs=1e-6)                     # Tiefe 2 findet das globale Optimum
    gap_stuck = 100 * (r1.length - opt) / opt
    near(gap_stuck, 6.53, 0.1)


# --- Sonstiges --------------------------------------------------------------------------------------------------------------------------------


def test_bound_matches_the_frozen_reference():
    near(ev.reference_bound(60, 0, 100000), 618.76, 0.1)


def test_preset_count_matches_the_readme():
    assert len(C.PRESETS) == 5
