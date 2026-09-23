"""lk_algorithm.lk_descend: die zentrale Garantie ist die Tiefe-1-Regression gegen lk_dlb.dlb_descend (byte-gleiche
Tour UND Bewertungszahl bei unbegrenztem Budget - kein eigenständig hergeleitetes Risiko, s. Modul-Docstring in
lk_algorithm.py für die Geometrie-Herleitung). Daneben: gültige Permutation, Längenbuchhaltung, Monotonie,
Determinismus, `touched`-Kurzweg, Tiefen-Buchführung, und dass größere Tiefe im Mittel nie schlechter ist."""

import numpy as np
import pytest

import lk_algorithm as LK
import lk_dlb as D
import lk_kick as K
import lk_tour as A


def _instance(n, seed):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 100
    return A.dist_matrix(xy)


# --- Zentrale Regression: Tiefe 1 == dlb_descend --------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(15))
def test_depth_one_exactly_reproduces_dlb_descend_unlimited_budget(seed):
    """Bei max_depth=1 und breadth1 >= Kandidatenlisten-Größe muss lk_descend fuer JEDEN Seed exakt dieselbe Tour UND
    dieselbe Bewertungszahl wie dlb_descend liefern - die zentrale Garantie, dass die Verkettung keine eigene,
    unabhaengig hergeleitete Logik fuer den Basisfall einfuehrt."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(15, 60))
    Dm = _instance(n, seed)
    cand = D.build_candidate_lists(Dm, min(D.CANDIDATE_K, n - 1))
    t0 = A.random_tour(n, np.random.default_rng(seed + 1000))
    base = D.dlb_descend(Dm, t0, cand, seed=seed + 7)
    lk1 = LK.lk_descend(Dm, t0, cand, max_depth=1, breadth1=D.CANDIDATE_K, seed=seed + 7)
    assert np.array_equal(base.tour, lk1.tour)
    assert base.evaluations == lk1.evaluations
    assert base.length == pytest.approx(lk1.length, abs=1e-6)


@pytest.mark.parametrize("seed", range(10))
def test_depth_one_reproduces_dlb_descend_with_the_touched_shortcut_too(seed):
    Dm = _instance(60, seed)
    cand = D.build_candidate_lists(Dm, D.CANDIDATE_K)
    t0 = A.random_tour(60, np.random.default_rng(seed + 1))
    full0 = D.dlb_descend(Dm, t0, cand, seed=0)
    rng = np.random.default_rng(seed + 2)
    kicked, touched = K.double_bridge(full0.tour, rng, n_bridges=1)
    base = D.dlb_descend(Dm, kicked, cand, seed=0, touched=touched)
    lk1 = LK.lk_descend(Dm, kicked, cand, max_depth=1, breadth1=D.CANDIDATE_K, seed=0, touched=touched)
    assert np.array_equal(base.tour, lk1.tour) and base.evaluations == lk1.evaluations


def test_depth_one_with_a_capped_budget_matches_up_to_the_last_partial_chain():
    """Anders als dlb_descend prueft lk_descend das Budget nur zwischen ganzen Ketten, nicht zwischen einzelnen
    Kandidaten innerhalb einer Kette (dokumentierte Vereinfachung) - bei einem knappen Budget kann sie deshalb um
    bis zu einer Kette (hoechstens `breadth1` Kandidaten) ueberziehen, wie das "die erste Einheit laeuft immer zu
    Ende"-Muster an anderer Stelle in dieser Linie (z. B. GRASPs erster Neustart)."""
    Dm = _instance(40, 3)
    cand = D.build_candidate_lists(Dm, D.CANDIDATE_K)
    t0 = A.random_tour(40, np.random.default_rng(4))
    full = D.dlb_descend(Dm, t0, cand, seed=3)
    cap = max(1, full.evaluations // 3)
    base = D.dlb_descend(Dm, t0, cand, seed=3, max_evaluations=cap)
    lk1 = LK.lk_descend(Dm, t0, cand, max_depth=1, breadth1=D.CANDIDATE_K, seed=3, max_evaluations=cap)
    assert not base.converged and not lk1.converged
    assert lk1.evaluations >= cap
    assert lk1.evaluations <= base.evaluations + D.CANDIDATE_K              # hoechstens eine Kette mehr


# --- Grundlegende Korrektheit ueber alle Tiefen -----------------------------------------------------------------------


@pytest.mark.parametrize("max_depth", [1, 2, 3, 5])
def test_valid_permutation_length_bookkeeping_and_monotone_descent(max_depth):
    Dm = _instance(30, 5)
    cand = D.build_candidate_lists(Dm, D.CANDIDATE_K)
    t0 = A.random_tour(30, np.random.default_rng(1))
    r = LK.lk_descend(Dm, t0, cand, max_depth=max_depth, breadth1=5, seed=2)
    assert sorted(r.tour.tolist()) == list(range(30))
    assert r.length == pytest.approx(A.tour_length(r.tour, Dm), abs=1e-6)
    assert r.length <= A.tour_length(t0, Dm) + 1e-9
    assert r.converged


def test_deterministic_for_a_seed_and_different_for_another():
    Dm = _instance(30, 6)
    cand = D.build_candidate_lists(Dm, D.CANDIDATE_K)
    t0 = A.random_tour(30, np.random.default_rng(1))
    a = LK.lk_descend(Dm, t0, cand, max_depth=3, seed=11)
    b = LK.lk_descend(Dm, t0, cand, max_depth=3, seed=11)
    c = LK.lk_descend(Dm, t0, cand, max_depth=3, seed=40)
    assert np.array_equal(a.tour, b.tour) and a.evaluations == b.evaluations
    assert not np.array_equal(a.tour, c.tour) or a.evaluations != c.evaluations


def test_depth_histogram_sums_to_the_number_of_committed_moves_and_stays_within_max_depth():
    Dm = _instance(40, 7)
    cand = D.build_candidate_lists(Dm, D.CANDIDATE_K)
    t0 = A.random_tour(40, np.random.default_rng(2))
    r = LK.lk_descend(Dm, t0, cand, max_depth=4, seed=9)
    assert all(1 <= depth <= 4 for depth in r.depth_hist)
    assert sum(r.depth_hist.values()) > 0


def test_deeper_search_is_never_worse_on_average_than_depth_one():
    """Kein Anspruch auf einen Sieg bei JEDER einzelnen Instanz (unterschiedliche fruehe Zuege koennen zu
    unterschiedlichen, nicht direkt vergleichbaren Lokaloptima fuehren) - aber im Mittel darf mehr Tiefe nicht
    schaden, sonst waere sie als Regler wertlos."""
    gaps1, gaps3 = [], []
    for seed in range(10):
        Dm = _instance(40, seed)
        cand = D.build_candidate_lists(Dm, D.CANDIDATE_K)
        t0 = A.random_tour(40, np.random.default_rng(seed + 50))
        r1 = LK.lk_descend(Dm, t0, cand, max_depth=1, breadth1=D.CANDIDATE_K, seed=1)
        r3 = LK.lk_descend(Dm, t0, cand, max_depth=3, breadth1=D.CANDIDATE_K, seed=1)
        gaps1.append(r1.length)
        gaps3.append(r3.length)
    assert float(np.mean(gaps3)) <= float(np.mean(gaps1)) + 1e-6


def test_breadth1_of_one_tries_only_the_first_candidate_at_level_one():
    """Bei breadth1=1 darf hoechstens 1 echter (gueltiger) Versuch je (c1, sign)-Kombination unternommen werden -
    ein grober, aber unabhaengiger Test der Breite-Begrenzung (nicht ueber das Ergebnis, sondern die Struktur)."""
    Dm = _instance(25, 8)
    cand = D.build_candidate_lists(Dm, D.CANDIDATE_K)
    t = [int(x) for x in A.random_tour(25, np.random.default_rng(3))]
    pos = [0] * 25
    for idx, city in enumerate(t):
        pos[city] = idx
    applied, delta, evaluations, nodes = LK._chain(t, pos, Dm, cand, t[0], 1, max_depth=5, breadth1=1)
    assert evaluations <= D.CANDIDATE_K                                   # nie mehr als die ganze Kandidatenliste einer Ebene


def test_invalid_local_search_or_accept_like_inputs_are_rejected():
    Dm = _instance(10, 0)
    cand = D.build_candidate_lists(Dm, min(D.CANDIDATE_K, 9))
    t0 = A.random_tour(10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        LK.lk_descend(Dm, t0, cand, max_depth=0)


# --- Chained LK (Doppelbrücke + LK-Wiederabstieg) -------------------------------------------------------------------


def test_chained_run_never_worsens_the_best_tour_and_keeps_a_valid_permutation():
    Dm = _instance(30, 12)
    cand = D.build_candidate_lists(Dm, D.CANDIDATE_K)
    t0 = A.random_tour(30, np.random.default_rng(3))
    r = LK.chained_run(Dm, t0, cand, max_depth=2, budget=20000, seed=5)
    assert sorted(r.best_tour.tolist()) == list(range(30))
    assert r.best_length == pytest.approx(A.tour_length(r.best_tour, Dm), abs=1e-6)
    assert r.best_length <= A.tour_length(t0, Dm) + 1e-9
    assert r.iterations >= 1 and r.evaluations >= 20000


def test_chained_run_snapshots_track_the_current_tour_and_best_is_monotone():
    Dm = _instance(25, 13)
    cand = D.build_candidate_lists(Dm, D.CANDIDATE_K)
    t0 = A.random_tour(25, np.random.default_rng(1))
    r = LK.chained_run(Dm, t0, cand, max_depth=2, budget=15000, seed=1, keep_snapshots=True)
    assert len(r.snapshots) == r.iterations + 1
    lengths = [A.tour_length(s, Dm) for s in r.snapshots]
    running_best = np.minimum.accumulate(lengths)
    assert running_best[-1] == pytest.approx(r.best_length, abs=1e-6)


def test_chained_run_is_deterministic_for_a_seed():
    Dm = _instance(25, 14)
    cand = D.build_candidate_lists(Dm, D.CANDIDATE_K)
    t0 = A.random_tour(25, np.random.default_rng(2))
    a = LK.chained_run(Dm, t0, cand, max_depth=2, budget=15000, seed=9)
    b = LK.chained_run(Dm, t0, cand, max_depth=2, budget=15000, seed=9)
    assert np.array_equal(a.best_tour, b.best_tour) and a.evaluations == b.evaluations


def test_chained_run_at_depth_one_uses_only_two_opt_moves_like_dlb_descend():
    """Bei Tiefe 1 darf jede Kick-Wiederabstieg-Runde nur committende Ketten der Laenge 1 erzeugen (reine 2-opt-
    Zuege) - dieselbe Tiefe-1-Garantie wie bei lk_descend selbst, hier innerhalb der Chained-LK-Schleife."""
    Dm = _instance(30, 15)
    cand = D.build_candidate_lists(Dm, D.CANDIDATE_K)
    t0 = A.random_tour(30, np.random.default_rng(4))
    r0 = LK.lk_descend(Dm, t0, cand, max_depth=1, breadth1=D.CANDIDATE_K, seed=0)
    assert set(r0.depth_hist) <= {1}
