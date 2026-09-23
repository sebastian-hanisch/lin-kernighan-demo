"""Auswertung der Lin-Kernighan-Demo: ein Chained-LK-Lauf (Doppelbrücke + LK-Wiederabstieg, wie iterated-local-
search-demo, hier mit lk_descend statt Kandidatenliste+DLB-2-opt als lokaler Suche) gegen denselben Kick-Mechanismus
bei Tiefe 1 (die faire Vergleichsgröße - "hilft tiefere Suche als ILS-Motor?"), gegen unabhängige LK-Neustarts (der
Kontrast "Kick statt Wegwerfen") und gegen einen einzelnen Abstieg. Sweeps, Vergleichstabellen, Kettenstreuung.

Der Abstand zur Schranke ist der Abstand zu einer *unteren* Schranke der kürzesten Tour (1-Baum, Held-Karp). Ein
Vorschlag ist ein geprüftes Kandidatenpaar (dieselbe Einheit wie `lk_dlb.dlb_descend`); der Kick selbst zählt
nicht zum Budget (dieselbe Konvention wie in der ganzen Trajektorien-Metaheuristiken-Linie)."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import lk_algorithm as LK
import lk_constants as C
import lk_dlb as DLB
import lk_scenario as S
import lk_tour as A


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    cluster_share: int = C.DEFAULT_BALLUNG
    seed: int = C.DEFAULT_SEED
    max_depth: int = C.DEFAULT_DEPTH
    budget: int = C.DEFAULT_BUDGET
    chain_seed: int = C.DEFAULT_CHAIN_SEED


@lru_cache(maxsize=256)
def instance(n, cluster_share, seed):
    inst = S.generate(n, cluster_share, seed)
    return inst, A.dist_matrix(inst.xy)


@lru_cache(maxsize=256)
def reference_bound(n, cluster_share, seed):
    inst, D = instance(n, cluster_share, seed)
    ref = A.descend(D, A.nearest_neighbor_tour(D), "2opt+oropt", "best", keep_steps=False)
    return A.held_karp_bound(D, ref.length, C.BOUND_ITERATIONS)


@lru_cache(maxsize=256)
def candidate_lists(n, cluster_share, seed):
    inst, D = instance(n, cluster_share, seed)
    return DLB.build_candidate_lists(D, min(DLB.CANDIDATE_K, len(D) - 1))


def lk_restarts(D, cand, max_depth, budget, seed, breadth1=C.DEFAULT_BREADTH1):
    """Unabhängige LK-Neustarts (kein Kick, jeder Neustart konstruiert eine frische Zufallstour) - der Kontrast zu
    `lk_algorithm.chained_run`: "wirft man die Tour bei jedem Versuch weg, oder stört man eine gute gezielt?"."""
    rng = np.random.default_rng(seed)
    used, starts, best = 0, 0, None
    while used < budget or best is None:
        cap = None if best is None else budget - used
        r = LK.lk_descend(D, A.random_tour(len(D), rng), cand, max_depth=max_depth, breadth1=breadth1, seed=starts, max_evaluations=cap)
        used += r.evaluations
        starts += 1
        if best is None or r.length < best:
            best = r.length
    return best, starts, used


@dataclass
class Analysis:
    settings: Settings
    inst: object
    D: np.ndarray
    bound: float
    start_tour: np.ndarray
    run: object                     # Chained LK bei der gewählten Tiefe
    seconds: float
    chain1: object                  # Chained LK bei Tiefe 1 (faire Vergleichsgröße, gleiches Budget)
    chain1_seconds: float
    single: object                  # ein einzelner LK-Abstieg (kein Kick), gleiches Budget als Obergrenze
    single_seconds: float
    crossings_end: int

    def gap_of(self, length):
        return 100.0 * (length - self.bound) / self.bound

    @property
    def gap(self):
        return self.gap_of(self.run.best_length)

    @property
    def final_gap(self):
        return self.gap_of(self.run.final_length)

    @property
    def chain1_gap(self):
        return self.gap_of(self.chain1.best_length)

    @property
    def single_gap(self):
        return self.gap_of(self.single.length)

    @property
    def start_gap(self):
        return self.gap_of(A.tour_length(self.start_tour, self.D))

    @property
    def accept_rate(self):
        return self.run.accepted / max(self.run.iterations, 1)


def analyse(settings, keep_snapshots=True, with_baselines=True):
    inst, D = instance(settings.n, settings.cluster_share, settings.seed)
    bound = reference_bound(settings.n, settings.cluster_share, settings.seed)
    cand = candidate_lists(settings.n, settings.cluster_share, settings.seed)
    start = A.random_tour(len(D), np.random.default_rng(settings.chain_seed))
    t0 = time.perf_counter()
    run = LK.chained_run(D, start, cand, max_depth=settings.max_depth, breadth1=C.DEFAULT_BREADTH1, n_bridges=C.DEFAULT_N_BRIDGES,
                          budget=settings.budget, seed=settings.chain_seed, keep_snapshots=keep_snapshots)
    seconds = time.perf_counter() - t0
    chain1 = single = None
    chain1_seconds = single_seconds = 0.0
    if with_baselines:
        t0 = time.perf_counter()
        chain1 = LK.chained_run(D, start, cand, max_depth=1, breadth1=C.DEFAULT_BREADTH1, n_bridges=C.DEFAULT_N_BRIDGES,
                                 budget=settings.budget, seed=settings.chain_seed, keep_snapshots=False)
        chain1_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        single = LK.lk_descend(D, start, cand, max_depth=settings.max_depth, breadth1=C.DEFAULT_BREADTH1,
                                seed=settings.chain_seed, max_evaluations=settings.budget)
        single_seconds = time.perf_counter() - t0
    return Analysis(settings, inst, D, bound, start, run, seconds, chain1, chain1_seconds, single, single_seconds,
                     A.count_crossings(inst.xy, run.best_tour))


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------------

WIN_MARGIN = 0.1                  # so viel besser als Chained LK bei Tiefe 1 gilt als Sieg (Prozentpunkte) - kleiner als sonst in der
LOSE_MARGIN = 0.1                 # Linie üblich (0.3), weil die Abstände hier deutlich kleiner sind (0.4-2 %, nicht 2-90 % wie anderswo)


def verdict(a):
    """Code: beats_hc (deutlich besser als Chained LK bei Tiefe 1, gleiches Budget - die faire Vergleichsgröße, da
    beide denselben Kick-Mechanismus nutzen), hc_wins, comparable. Gilt für diesen einen Lauf - die Ketten streuen."""
    if a.gap <= a.chain1_gap - WIN_MARGIN:
        return "beats_hc"
    if a.chain1_gap <= a.gap - LOSE_MARGIN:
        return "hc_wins"
    return "comparable"


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS, **changes):
    """Mittel über die festen Instanzen und je `chains` Ketten-Seeds für die Einstellungen `base` mit `changes`."""
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        for ch in range(chains):
            a = analyse(replace(s0, seed=seed, chain_seed=ch), keep_snapshots=False)
            rows.append({"gap": a.gap, "chain1": a.chain1_gap, "single": a.single_gap, "seconds": a.seconds,
                         "iterations": a.run.iterations, "accept_rate": a.accept_rate, "crossings": a.crossings_end})
    out = {k: _mean(rows, k) for k in rows[0]}
    out.update({"gap_sd": float(np.std([r["gap"] for r in rows])), "gap_min": float(np.min([r["gap"] for r in rows])),
                "gap_max": float(np.max([r["gap"] for r in rows])), "n_runs": len(rows)})
    return out


SWEEP_VALUES = {"budget": (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000), "max_depth": (1, 2, 3, 4, 5, 6, 8),
                "n": (10, 20, 40, 60, 100, 150, 200), "cluster_share": (0, 25, 50, 75, 100)}
SWEEP_LABELS = {"budget": "Budget (bewertete Kandidatenpaare)", "max_depth": "LK-Tiefe", "n": "Stopps", "cluster_share": "Anteil in Gruppen (%)"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def restart_sweep(param="max_depth", base=Settings(), values=None):
    """Wie `sweep`, aber mit unabhängigen Neustarts (`lk_restarts`) statt Chained LK - der Kontrast-Experiment."""
    values = SWEEP_VALUES[param] if values is None else values
    rows = []
    for v in values:
        s = replace(base, **{param: v})
        gaps, starts_list = [], []
        for seed in C.SWEEP_SEEDS:
            inst, D = instance(s.n, s.cluster_share, seed)
            bound = reference_bound(s.n, s.cluster_share, seed)
            cand = candidate_lists(s.n, s.cluster_share, seed)
            for ch in range(C.SWEEP_CHAINS):
                length, starts, used = lk_restarts(D, cand, s.max_depth if param != "max_depth" else v, s.budget, ch)
                gaps.append(100 * (length - bound) / bound)
                starts_list.append(starts)
        rows.append({"value": v, "gap": float(np.mean(gaps)), "starts": float(np.mean(starts_list))})
    return rows


SCALING_N = C.SCALING_N
SCALING_POLICIES = (("Budget 200 Tausend", lambda n: 200000), ("Budget 5 000 · Stopps", lambda n: 5000 * n))


def scaling_table(base=Settings()):
    return [{"label": label, "rows": [{"value": n, **run_config(base, n=n, budget=fn(n))} for n in SCALING_N]} for label, fn in SCALING_POLICIES]


def chain_spread(settings, k=C.SPREAD_CHAINS):
    """k Ketten-Seeds auf derselben Instanz: Chained LK bei der gewählten Tiefe gegen Tiefe 1 (gleicher Kick-Mechanismus)."""
    gaps, gaps1 = [], []
    for ch in range(k):
        a = analyse(replace(settings, chain_seed=ch), keep_snapshots=False)
        gaps.append(a.gap)
        gaps1.append(a.chain1_gap)
    return {"lk": np.array(gaps), "chain1": np.array(gaps1)}
