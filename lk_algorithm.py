"""Lin-Kernighan (Lin & Kernighan 1973) für eine Rundtour (TSP): eine direkte Verallgemeinerung der Kandidatenlisten+
Don't-Look-Bits-2-opt-Suche aus `lk_dlb.py` (Bentley 1992 / Johnson & McGeoch, wortgleich aus `iterated-local-search-
demo/ils_dlb.py` übernommen). Ein einzelner 2-opt-Zug MIT Kandidatenliste ist bereits die Tiefe-1-Variante eines
Lin-Kernighan-Zugs: Kante (t1,t2) entfernen, Kante (t2,t3) zu einem Kandidaten von t2 hinzufügen, Kante (t3,t4)
entfernen, Kante (t4,t1) schließen. Bei Tiefe 1 MUSS diese Suche deshalb exakt `lk_dlb.dlb_descend` reproduzieren
(siehe tests/test_algorithm.py) - kein eigenständig hergeleitetes Risiko.

Lin-Kernighan verallgemeinert das: schließt ein Zug bei Tiefe 1 nicht mit Gewinn (kumulierter Delta >= 0), wird er
probeweise WEITERGEFÜHRT statt verworfen - der bei Tiefe 1 freigelegte Knoten t4 wird zum neuen Anker (t1 bleibt die
ganze Kette über an derselben Tour-Position fixiert relativ zum jeweils freien Ende, s. u.), ein weiterer Kandidat
wird gesucht, eine weitere Umkehrung probeweise angewandt, der kumulierte Längen-Delta nachgeführt - bis zu `max_depth`
Ebenen oder bis der kumulierte Delta negativ ist (dann wird sofort committet, "first improvement" über die Tiefen).

**Geometrie der Verkettung** (hergeleitet aus `_delta_2opt`/`dlb_descend`s eigener Indexkonvention, nicht aus dem
Original-Paper direkt übernommen): bei Tiefe 1 mit `sign=+1` liegt der Anker (t1, die ganze Kette über fix) an
Position `pos[c1]+1` (= `next(c1)`), das bewegliche Ende (t2) ist `c1` selbst; die Kandidatenliste `cand[c1]` liefert
t3-Kandidaten (kurze NEUE Kante (t2,t3)), t4 = `next(t3)`. Nach der (probeweise angewandten) Umkehrung sitzt t1 IMMER
unmittelbar VOR dem neuen freien Ende (hier: t4) im Array - das ist unabhängig von der Tiefe (die Umkehrung stellt
diese Nachbarschaft jedes Mal neu her), also verwendet JEDE tiefere Ebene dieselbe Konvention wie `sign=-1` bei
`dlb_descend`, verankert am jeweils freien Ende. Das macht die Rekursion trivial: Ebene 2+ ist strukturell identisch
zu Ebene 1 mit `sign=-1` und `anchor_node = freies Ende der vorigen Ebene`.

**Breite/Backtracking** (bewusste Scope-Entscheidung, s. README "Was nicht umgesetzt"): auf Ebene 1 werden bis zu
`breadth1` Kandidaten der Reihe nach probiert (jeder mit voller Tiefen-Rekursion); ab Ebene 2 wird GIERIG nur der
erste gültige Kandidat verfolgt (kein Backtracking dort) - eine Vereinfachung des Original-Papers (das auf Ebene 1
UND 2 mehrere Kandidaten probiert, danach ebenfalls gierig ist). Pruning: dieselbe sortierte-Kandidaten-Abbruch-
Heuristik wie `dlb_descend` auf JEDER Ebene (kein zusätzliches kumuliertes Teilgewinn-Pruning über die lokale
Distanz-Heuristik hinaus - eine weitere dokumentierte Vereinfachung).

Eine Bewertung = ein geprüftes Kandidatenpaar (wie `dlb_descend`), auf jeder Ebene der Kette einzeln gezählt -
dieselbe Budget-Einheit wie in der ganzen Trajektorien-Metaheuristiken-Linie."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

import lk_kick as K
import lk_tour as A

EPS = 1e-9


def _valid(a, b, n):
    return b >= a + 2 and not (a == 0 and b == n - 1)


def _apply_reversal(t, pos, a, b):
    t[a + 1:b + 1] = t[a + 1:b + 1][::-1]
    for q in range(a + 1, b + 1):
        pos[t[q]] = q


def _extend_greedy(t, pos, D, cand, free, cum_before, depth, max_depth):
    """Ebene `depth` (>= 2) der Kette, gierig (nur der erste gültige Kandidat, kein Backtracking). Gibt
    (committed, delta_dieser_teilkette, Bewertungen, betroffene_Knoten_je_Ebene) zurück; committed=True heißt: das
    tentative Ergebnis bleibt im Array stehen (kumulierter Gewinn < 0 irgendwo auf dem Weg nach unten)."""
    if depth > max_depth:
        return False, 0.0, 0, []
    n = len(t)
    i = pos[free]
    anchor_pos = (i - 1) % n
    nxt_pos = i
    d_anchor = D[t[anchor_pos], t[nxt_pos]]
    evaluations = 0
    chosen = None
    for c2 in cand[free]:
        evaluations += 1
        if D[free, c2] >= d_anchor:
            break
        j = pos[c2]
        p2 = (j - 1) % n
        a, b = (anchor_pos, p2) if anchor_pos < p2 else (p2, anchor_pos)
        if not _valid(a, b, n):
            continue
        chosen = (a, b)
        break
    if chosen is None:
        return False, 0.0, evaluations, []
    a, b = chosen
    bp1 = (b + 1) % n
    nodes = (t[a], t[b], t[a + 1], t[bp1])                              # Reihenfolge wie dlb_descend NACH der Umkehrung liest (t[a],t[a+1],t[b],t[bp1]) - vor der Umkehrung sind a+1/b vertauscht
    delta = D[t[a], t[b]] + D[t[a + 1], t[bp1]] - D[t[a], t[a + 1]] - D[t[b], t[bp1]]
    _apply_reversal(t, pos, a, b)
    new_free = t[bp1]
    cum = cum_before + delta
    if cum < -EPS:
        return True, delta, evaluations, [nodes]
    ok, deeper_delta, deeper_ev, deeper_nodes = _extend_greedy(t, pos, D, cand, new_free, cum, depth + 1, max_depth)
    evaluations += deeper_ev
    if ok:
        return True, delta + deeper_delta, evaluations, [nodes] + deeper_nodes
    _apply_reversal(t, pos, a, b)                                          # diese Ebene rückgängig machen (Umkehrung ist selbstinvers)
    return False, 0.0, evaluations, []


def _chain(t, pos, D, cand, c1, sign, max_depth, breadth1):
    """Eine vollständige LK-Kette ab `c1` in Richtung `sign`: Kandidaten der Reihe nach probieren (wie
    `dlb_descend`), jeweils mit voller Tiefen-Rekursion, bis zu `breadth1` echte Versuche (nicht ungültige/
    übersprungene Kandidaten). Gibt (committed, Gesamt-Delta, Bewertungen, betroffene_Knoten_je_Ebene) zurück; bei
    committed=False ist das Array wieder im Ausgangszustand (vollständig rückgängig gemacht). Bricht - wie
    `dlb_descend` - SOFORT beim ersten committenden Kandidaten ab, ohne weitere Kandidaten zu bewerten (zentral
    für die Tiefe-1-Regressionsgleichheit mit `dlb_descend`, s. Modul-Docstring)."""
    n = len(t)
    evaluations = 0
    i = pos[c1]
    anchor_pos = i if sign == 1 else (i - 1) % n
    nxt_pos = (anchor_pos + 1) % n
    d_anchor = D[t[anchor_pos], t[nxt_pos]]
    tried = 0
    for c2 in cand[c1]:
        if tried >= breadth1:
            break
        evaluations += 1
        if D[c1, c2] >= d_anchor:
            break
        j = pos[c2]
        p2 = j if sign == 1 else (j - 1) % n
        a, b = (anchor_pos, p2) if anchor_pos < p2 else (p2, anchor_pos)
        if not _valid(a, b, n):
            continue
        tried += 1
        bp1 = (b + 1) % n
        nodes = (t[a], t[b], t[a + 1], t[bp1])                              # s. _extend_greedy: dlb_descend-Reihenfolge nach der Umkehrung
        delta = D[t[a], t[b]] + D[t[a + 1], t[bp1]] - D[t[a], t[a + 1]] - D[t[b], t[bp1]]
        _apply_reversal(t, pos, a, b)
        if delta < -EPS:
            return True, delta, evaluations, [nodes]
        free = t[bp1]
        ok, deeper_delta, deeper_ev, deeper_nodes = _extend_greedy(t, pos, D, cand, free, delta, 2, max_depth)
        evaluations += deeper_ev
        if ok:
            return True, delta + deeper_delta, evaluations, [nodes] + deeper_nodes
        _apply_reversal(t, pos, a, b)
    return False, 0.0, evaluations, []


@dataclass
class LkResult:
    tour: np.ndarray
    length: float
    evaluations: int
    converged: bool                                                 # False, wenn max_evaluations vor der Konvergenz erreicht wurde
    depth_hist: dict = field(default_factory=dict)                  # {Tiefe: Zahl der Züge, die dort committet haben}


def lk_descend(D, start, cand, max_depth=5, breadth1=5, seed=0, max_evaluations=None, touched=None):
    """Ein Lin-Kernighan-Abstieg (erste Verbesserung über Ketten bis `max_depth`). `touched` (optional): nur diese
    Knoten starten in der Warteschlange (Kurzweg für den Wiederabstieg nach einer Doppelbrücken-Störung, wie
    `lk_dlb.dlb_descend`); `None` = voller Scan. Bei `max_depth=1` (und `breadth1 >=` Kandidatenlisten-Größe)
    reproduziert diese Funktion `dlb_descend` exakt (s. tests/test_algorithm.py)."""
    if max_depth < 1 or breadth1 < 1:
        raise ValueError((max_depth, breadth1))
    n = len(start)
    t = [int(x) for x in start]
    pos = [0] * n
    for idx, city in enumerate(t):
        pos[city] = idx
    rng = np.random.default_rng(seed)
    if touched is None:
        dontlook = [False] * n
        queue = deque(rng.permutation(n).tolist())
        in_queue = [True] * n
    else:
        dontlook = [True] * n
        start_nodes = rng.permutation(np.asarray(touched, dtype=np.int64)).tolist()
        queue = deque(start_nodes)
        in_queue = [False] * n
        for city in start_nodes:
            dontlook[city] = False
            in_queue[city] = True
    evaluations = 0
    depth_hist = {}
    length = A.tour_length(np.array(t), D)

    while queue:
        c1 = queue.popleft()
        in_queue[c1] = False
        if dontlook[c1]:
            continue
        improved = False
        for sign in (1, -1):
            if max_evaluations is not None and evaluations >= max_evaluations:
                return LkResult(np.array(t), length, evaluations, False, depth_hist)
            applied, delta, spent, nodes = _chain(t, pos, D, cand, c1, sign, max_depth, breadth1)
            evaluations += spent
            if applied:
                length += delta
                improved = True
                depth_hist[len(nodes)] = depth_hist.get(len(nodes), 0) + 1
                for level_nodes in nodes:
                    for city in level_nodes:
                        dontlook[city] = False
                        if not in_queue[city]:
                            queue.append(city)
                            in_queue[city] = True
                break
        if not improved:
            dontlook[c1] = True
        if max_evaluations is not None and evaluations >= max_evaluations:
            return LkResult(np.array(t), length, evaluations, False, depth_hist)

    return LkResult(np.array(t), length, evaluations, True, depth_hist)


# --- Chained LK (Applegate, Cook & Rohe 2003): Doppelbrücke + LK-Wiederabstieg statt unabhängiger Neustarts -----
# Wortgleiches Muster wie ils_algorithm.run() (iterated-local-search-demo), hier mit lk_descend statt DLB-2-opt als
# lokaler Suche - der Cross-Edge aus den DAG-Notizen ("wrap it in ILS = chained LK", "LK is the strong one" als
# austauschbare lokale Suche innerhalb der ILS-Kick-Schleife). `n_bridges` ist hier bewusst NICHT neu kalibriert
# (das ist ILS' eigenes Regler-Terrain), fix auf `lk_constants.DEFAULT_N_BRIDGES`.


@dataclass
class ChainRun:
    best_tour: np.ndarray
    best_length: float
    final_tour: np.ndarray
    final_length: float
    evaluations: int = 0
    iterations: int = 0
    accepted: int = 0
    snapshots: list = field(default_factory=list)          # aktuelle Tour nach jeder Iteration (nur bei keep_snapshots=True)
    trace_iter: np.ndarray = None
    trace_length: np.ndarray = None
    trace_best: np.ndarray = None
    max_depth: int = 2
    n_bridges: int = 1


def chained_run(D, start, cand, max_depth=2, breadth1=5, n_bridges=1, budget=100000, seed=0, keep_snapshots=True, trace_points=300):
    """Ein Chained-LK-Lauf: erster Abstieg aus `start`, dann Doppelbrücke + LK-Wiederabstieg (nur die betroffene
    Umgebung, `touched`) wiederholt, bis das Budget erschöpft ist. Nur die neue Tour wird angenommen, wenn sie
    mindestens so kurz wie die aktuelle ist (klassisches ILS/Chained-LK-Muster, "better"). `budget` zählt bewertete
    Kandidatenpaare insgesamt; der Kick selbst zählt nicht (dieselbe Konvention wie überall in der Linie)."""
    rng = np.random.default_rng(seed)
    evaluations = iterations = accepted = 0
    trace_every = max(1, budget // trace_points)

    init_seed = int(rng.integers(0, 2**31 - 1))
    r0 = lk_descend(D, start, cand, max_depth=max_depth, breadth1=breadth1, seed=init_seed, max_evaluations=budget)
    current, current_length = r0.tour, r0.length
    evaluations += r0.evaluations
    best_tour, best_length = current.copy(), current_length
    snapshots = [current.copy()] if keep_snapshots else []
    tr_it, tr_len, tr_best = [evaluations], [current_length], [current_length]

    while evaluations < budget:
        remaining = budget - evaluations
        kicked, touched = K.double_bridge(current, rng, n_bridges=n_bridges)
        descend_seed = int(rng.integers(0, 2**31 - 1))
        r = lk_descend(D, kicked, cand, max_depth=max_depth, breadth1=breadth1, seed=descend_seed, max_evaluations=remaining, touched=touched)
        evaluations += r.evaluations
        iterations += 1
        take = r.length <= current_length + EPS
        if take:
            current, current_length = r.tour, r.length
            accepted += 1
        if current_length < best_length - EPS:
            best_length, best_tour = current_length, current.copy()
        if keep_snapshots:
            snapshots.append(current.copy())
        if iterations % trace_every == 0 or evaluations >= budget:
            tr_it.append(evaluations)
            tr_len.append(current_length)
            tr_best.append(best_length)

    final_length = A.tour_length(current, D)
    best_length = A.tour_length(best_tour, D)
    return ChainRun(best_tour, best_length, current, final_length, evaluations, iterations, accepted, snapshots,
                     np.array(tr_it), np.array(tr_len), np.array(tr_best), max_depth, n_bridges)
