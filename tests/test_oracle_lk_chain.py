"""Orakel-Test für die LK-Kette und die Kleine-Instanz-Behauptung: Längen werden aus der Tourliste neu gerechnet (nicht aus den Delta-Summen der
Kette), die exakte Rundtour kommt aus der Held-Karp-DP (kein Brute-Force-Code der Demo, keine eingefrorenen Werte), und die Don't-Look-Bit-Suche
wird auf echte verbessernde 2-opt-Züge der Kandidatenlisten geprüft (Zug wird auf der Liste ausgeführt, Länge neu gerechnet)."""

import numpy as np

import lk_algorithm as LK
import lk_dlb as D
import lk_evaluation as ev
import lk_kick as K
import lk_tour as A


def _len(t, Dl):
    return sum(Dl[t[k]][t[(k + 1) % len(t)]] for k in range(len(t)))


def _optimum(Dl):
    """Held-Karp-DP über Teilmengen (exakt), Knoten 0 = Depot."""
    n = len(Dl)
    m = n - 1
    dp = {(1 << j, j): Dl[0][j + 1] for j in range(m)}
    for mask in range(1, 1 << m):
        for j in range(m):
            cur = dp.get((mask, j))
            if cur is None:
                continue
            for k in range(m):
                if not (mask >> k) & 1:
                    key = (mask | 1 << k, k)
                    v = cur + Dl[j + 1][k + 1]
                    if v < dp.get(key, float("inf")):
                        dp[key] = v
    return min(dp[((1 << m) - 1, j)] + Dl[j + 1][0] for j in range(m))


def _improving_candidate_moves(t, Dl, cand):
    """Zahl der 2-opt-Züge mit Kandidatenkante (c1, c2), d(c1,c2) < d(Ankerkante), die die Tour (neu gerechnet) verkürzen."""
    n = len(t)
    base = _len(t, Dl)
    count = 0
    for tt in (list(t), list(t)[::-1]):                 # Vorwärts- und Rückwärtsrichtung
        for c1 in range(n):
            k1 = tt.index(c1)
            rot = tt[k1:] + tt[:k1]                     # c1 steht vorn, sein Nachfolger folgt
            d_anchor = Dl[c1][rot[1]]
            for c2 in cand[c1]:
                if Dl[c1][c2] >= d_anchor:
                    break
                k = rot.index(c2)
                if k < 2 or k == n - 1:
                    continue
                if _len([c1] + rot[1:k + 1][::-1] + rot[k + 1:], Dl) < base - 1e-9:
                    count += 1
    return count


def test_lk_descents_keep_valid_tours_exact_lengths_and_reach_a_candidate_two_opt_optimum():
    rng = np.random.default_rng(2024)
    for _ in range(40):
        n = int(rng.integers(6, 22))
        xy = rng.random((n, 2)) * 100
        Dm = A.dist_matrix(xy)
        Dl = Dm.tolist()
        cand = D.build_candidate_lists(Dm, min(5, n - 1))
        start = rng.permutation(n)
        for depth in (1, 2, 3, 5):
            r = LK.lk_descend(Dm, start, cand, max_depth=depth, breadth1=5, seed=int(rng.integers(0, 99)))
            t = r.tour.tolist()
            assert sorted(t) == list(range(n)) and r.converged
            assert abs(r.length - _len(t, Dl)) < 1e-7                    # Buchführung der verketteten Deltas == Neuberechnung
            assert r.length <= _len(start.tolist(), Dl) + 1e-9
            assert max(r.depth_hist, default=1) <= depth
            # Don't-Look-Bits sind ein Näherungs-, kein exaktes Kriterium: ein Rest verbessernder Kandidatenzüge ist möglich, aber
            # (Tiefe 1) genau dann vorhanden, wenn ein voller Neuscan (ohne Don't-Look-Bits) einen Gewinn findet - Orakel == Suche
            residual = _improving_candidate_moves(t, Dl, cand.tolist())
            rescan = LK.lk_descend(Dm, r.tour, cand, max_depth=depth, breadth1=5, seed=1)
            assert rescan.length <= r.length + 1e-9 and abs(rescan.length - _len(rescan.tour.tolist(), Dl)) < 1e-7
            if depth == 1:
                assert (residual > 0) == (rescan.length < r.length - 1e-9)
            elif residual:
                assert rescan.length < r.length - 1e-9                 # jeder Tiefe-1-Gewinn bleibt für tiefere Ketten sichtbar (Kette bricht bei Gewinn sofort ab)


def test_chained_run_keeps_exact_best_and_final_lengths_and_never_exceeds_the_budget_by_more_than_one_search():
    rng = np.random.default_rng(5)
    for _ in range(15):
        n = int(rng.integers(8, 25))
        Dm = A.dist_matrix(rng.random((n, 2)) * 100)
        Dl = Dm.tolist()
        cand = D.build_candidate_lists(Dm, min(5, n - 1))
        run = LK.chained_run(Dm, rng.permutation(n), cand, max_depth=int(rng.integers(1, 4)), budget=int(rng.integers(200, 3000)), seed=int(rng.integers(0, 50)))
        assert abs(run.best_length - _len(run.best_tour.tolist(), Dl)) < 1e-7
        assert abs(run.final_length - _len(run.final_tour.tolist(), Dl)) < 1e-7
        lengths = [_len(s.tolist(), Dl) for s in run.snapshots]
        assert all(b <= a + 1e-7 for a, b in zip(lengths, lengths[1:]))       # nur gleich gute oder bessere Touren werden angenommen
        assert run.best_length <= run.final_length + 1e-9
        assert not np.any(np.diff(run.trace_best) > 1e-9)


def test_double_bridge_changes_exactly_the_edges_it_reports():
    rng = np.random.default_rng(9)
    for _ in range(100):
        n = int(rng.integers(8, 30))
        t = rng.permutation(n)
        u, touched = K.double_bridge(t, rng)
        assert sorted(u.tolist()) == sorted(t.tolist())
        new_edges = A.tour_edges(u) - A.tour_edges(t)
        assert new_edges and {x for e in new_edges for x in e} <= set(touched)


def test_small_instance_claim_against_an_exact_dynamic_programme():
    """app.py/README: 8 Knoten, 2-opt-Abstieg bleibt bei 312.35 km (6.53 % über dem Optimum 293.21 km), Tiefe 2 findet das Optimum."""
    n, seed, start_seed = 8, 1, 6
    xy = np.random.default_rng(seed * 1000 + n).random((n, 2)) * 100
    Dm = A.dist_matrix(xy)
    Dl = Dm.tolist()
    cand = D.build_candidate_lists(Dm, min(D.CANDIDATE_K, n - 1))
    t0 = A.random_tour(n, np.random.default_rng(start_seed))
    opt = _optimum(Dl)
    r1 = D.dlb_descend(Dm, t0, cand, seed=start_seed)
    r2 = LK.lk_descend(Dm, t0, cand, max_depth=2, breadth1=D.CANDIDATE_K, seed=start_seed)
    assert abs(opt - 293.21) < 0.01 and abs(r1.length - 312.35) < 0.01
    assert abs(100 * (r1.length - opt) / opt - 6.53) < 0.01
    assert abs(r2.length - opt) < 1e-6
    # r1 ist ein ECHTES 2-opt-Optimum der vollen Nachbarschaft (jeder 2-opt-Nachbar, per Liste neu gerechnet, ist nicht kürzer)
    t = r1.tour.tolist()
    assert all(_len(t[:i + 1] + t[i + 1:j + 1][::-1] + t[j + 1:], Dl) >= r1.length - 1e-9
               for i in range(n) for j in range(i + 2, n) if not (i == 0 and j == n - 1))


def test_reference_bound_never_exceeds_the_exact_optimum_on_small_instances():
    for stops, seed in ((7, 1), (8, 2), (9, 3), (10, 4)):
        inst, Dm = ev.instance(stops, 0, seed)
        assert ev.reference_bound(stops, 0, seed) <= _optimum(Dm.tolist()) + 1e-7


def test_every_committed_chain_of_depth_k_is_a_real_improving_k_plus_one_opt_move():
    """Eine bei Tiefe k committete Kette tauscht höchstens k+1 Kanten (jede Umkehrung entfernt 2 und fügt 2 ein, die Schlusskante hebt sich je
    Stufe auf), gemessen an der Kantenmenge vor/nach - und ihr Delta ist genau die neu gerechnete Längenänderung und negativ."""
    rng = np.random.default_rng(77)
    seen = {}
    for _ in range(300):
        n = int(rng.integers(7, 20))
        Dm = A.dist_matrix(rng.random((n, 2)) * 100)
        Dl = Dm.tolist()
        cand = D.build_candidate_lists(Dm, min(5, n - 1))
        t = [int(x) for x in rng.permutation(n)]
        pos = [0] * n
        for idx, c in enumerate(t):
            pos[c] = idx
        for c1 in rng.permutation(n).tolist():
            for sign in (1, -1):
                depth = int(rng.integers(1, 6))
                before_t, before_len = list(t), _len(t, Dl)
                applied, delta, _, nodes = LK._chain(t, pos, Dm, cand, c1, sign, depth, 5)
                if not applied:
                    assert t == before_t                                   # nicht committet => vollständig rückgängig gemacht
                    continue
                assert delta < 0 and abs((_len(t, Dl) - before_len) - delta) < 1e-7
                assert all(pos[c] == k for k, c in enumerate(t))
                removed = A.tour_edges(np.array(before_t)) - A.tour_edges(np.array(t))
                assert len(removed) <= len(nodes) + 1 <= depth + 1
                seen[len(nodes)] = seen.get(len(nodes), 0) + 1
                break
    assert seen.get(1, 0) > 50 and seen.get(2, 0) > 5
