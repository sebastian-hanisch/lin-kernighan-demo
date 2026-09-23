"""Konstanten der Lin-Kernighan-Demo: Szenario (wortgleich zur hill-climbing-demo/iterated-local-search-demo), Regler, Beschriftungen (Presets folgen nach den Messungen)."""

AREA = 100.0                     # Kantenlänge des Gebiets in km
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0              # Streuung einer Gruppe in km
CLUSTER_MARGIN = 12.0            # Gruppenmittelpunkte liegen mindestens so weit vom Rand entfernt
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_CHAINS = 3                 # Ketten-Seeds je Instanz in Sweeps und Vergleichstabellen
BOUND_ITERATIONS = 300

N_MIN, N_MAX, DEFAULT_N, N_STEP = 10, 200, 60, 5
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG, BALLUNG_STEP = 0, 100, 0, 25
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_CHAIN_SEED = 0
BUDGETS = (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000)
SCALING_N = (20, 40, 60, 100, 150, 200)
SPREAD_CHAINS = 20
CANDIDATE_K = 5                  # dieselbe Kandidatenlisten-Größe wie ils_dlb.py (lk_dlb.CANDIDATE_K)

# --- LK-eigene Regler ----------------------------------------------------------------------------------------------
DEPTH_MIN, DEPTH_MAX, DEFAULT_DEPTH = 1, 8, 2
DEFAULT_BREADTH1 = 5             # fix, nicht Regler (Sensitivitäts-Check: 1->1.17 %, 2->1.01 %, 3->0.75 %, 5->0.60 %, 8->0.59 % - ab 5 nahe gesättigt)
DEFAULT_N_BRIDGES = 1            # fix, nicht Regler - dieselbe Kalibrierung wie in iterated-local-search-demo, hier nicht neu exploriert
DEFAULT_BUDGET = 200000

# --- Gemessene Werte (Mittel über 5 feste Sweep-Instanzen, Seeds 100000-100004, je 3 Ketten-Seeds; n=60, Anteil=0, --
# --- Budget 200 Tausend, sofern nicht anders angegeben; 2026-09-23, ALLE Werte über ev.sweep/ev.restart_sweep/ ------
# --- ev.scaling_table nachgerechnet, s. tests/test_claims.py) --------------------------------------------------------
# EIN Abstieg (kein Kick/Neustart), Tiefe 1/2/3/4/5/6/8: 7.05/6.51/6.45/6.45/6.45/6.45/6.45 % Abstand zur Schranke -
#   Tiefe 2 escaped bereits die meisten 2-opt-Sackgassen, ab Tiefe 3 keine weitere Verbesserung (gesättigt).
# UNABHÄNGIGE NEUSTARTS bei Budget 200 Tausend, Tiefe 1/2/3/5: 0.68/0.70/0.72/0.72 % - Tiefe hilft hier NICHT (eher
#   leicht schlechter): teure Ketten kosten Neustarts, und bei so vielen billigen Neustarts (312 bei Tiefe 1) gewinnt
#   die Menge gegen die Qualität - EHRLICHER NEGATIVBEFUND, kein Sweet Spot bei reinen Neustarts.
# CHAINED LK (Doppelbrücke + LK-Wiederabstieg, wie iterated-local-search-demo, hier mit lk_descend statt DLB-2-opt),
#   Tiefe 1 (=chain1) vs. Tiefe 2 (=Voreinstellung) über das Budget: 10T 1.36/1.77, 25T 1.03/1.02, 50T 0.79/1.02,
#   100T 0.70/0.68, 200T (Standard) 0.64/0.66, 500T 0.60/0.51, 1M 0.58/0.47, 2M 0.58/0.47 %. Bei 10-200 Tausend sind
#   beide Tiefen etwa gleichauf (schwankt knapp hin und her, im Rauschen) - EHRLICHER BEFUND: bei n=60 zahlt sich
#   Tiefe 2 beim STANDARDBUDGET NICHT klar aus. Erst AB 500 Tausend zieht Tiefe 2 klar und wachsend vorbei.
# CHAINED LK NACH INSTANZGRÖSSE bei FESTEM Budget 200 Tausend, Tiefe 1 vs. Tiefe 2: n=20/40 exakt gleichauf (beide
#   praktisch optimal, 0.05/0.56 %), n=60 Tiefe 1 knapp vorn (0.64/0.66), AB n=100 zieht Tiefe 2 klar vorbei und der
#   Vorsprung WÄCHST: n=100 1.14/1.06, n=150 1.74/1.46, n=200 2.09/1.82 %. Der Tiefe-Vorteil braucht also entweder
#   ein GROSSES Budget ODER eine GROSSE Instanz bei n=60/200T - kein einfaches "mehr Tiefe ist immer besser".
# BREADTH1-Sensitivität (Chained LK, Tiefe 2, 200 Tausend): 1/2/3/5/8 -> 1.29/0.60/0.63/0.66/0.53 % - breadth1=1 ist
#   klar schlechter, ab 2 liegen alle Werte im Rauschen beieinander (kein klarer weiterer Trend) - als Konstante (5)
#   fixiert statt als Regler, da schon ab 2 kein systematischer Zusatzgewinn mehr messbar ist.
# KLEINE-INSTANZ-DEMONSTRATION (n=8, Punkte s. tests/test_claims.py): ein Startpunkt lässt Tiefe 1 (=2-opt) bei
#   312.35 haengen; das globale Optimum (Brute-Force) liegt bei 293.21 (6.53 % darueber); bereits Tiefe 2 findet es.


def _preset(max_depth=DEFAULT_DEPTH, budget=DEFAULT_BUDGET, n=DEFAULT_N):
    return {"n": n, "ballung": DEFAULT_BALLUNG, "seed": DEFAULT_SEED, "max_depth": max_depth, "budget": budget, "chain_seed": DEFAULT_CHAIN_SEED}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Nur 2-opt (Tiefe 1)": _preset(max_depth=1),
    "Zu kleines Budget (10 Tausend)": _preset(budget=10000),
    "Großes Budget (1 Million)": _preset(budget=1000000),
    "Große Instanz (200 Stopps, 1 Million)": _preset(n=200, budget=1000000),
}
# Mittel über die fünf festen Sweep-Instanzen (Seeds 100000-100004, je drei Ketten-Seeds), Abstand zur Schranke; chain1 = dieselbe Kick-Schleife bei Tiefe 1
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "60 Stopps, Tiefe 2, 200 Tausend Vorschläge: die beste Tour liegt im Mittel 0.66 % über der Schranke - dieselbe Kick-Schleife bei Tiefe 1 (reines 2-opt) 0.64 %, praktisch gleichauf. Bei diesem Standardbudget zahlt sich die Tiefe bei 60 Stopps noch NICHT klar aus - siehe die Budget- und Größen-Experimente.",
    "Nur 2-opt (Tiefe 1)": "Die LK-Kette schließt immer sofort (kein probeweises Weiterführen): 0.64 % über der Schranke - bei diesem Budget/dieser Größe praktisch gleichauf mit Tiefe 2 (0.66 %).",
    "Zu kleines Budget (10 Tausend)": "Nur 10 Tausend Vorschläge reichen für wenige Kicks: Tiefe 2 verliert hier klar gegen Tiefe 1 (1.77 % gegen 1.36 %) - die teurere Tiefe kostet mehr, als sie bei so wenigen Kicks bringt.",
    "Großes Budget (1 Million)": "1 Million Vorschläge erlauben sehr viele Kicks: Tiefe 2 liegt bei 0.47 % gegen 0.58 % bei Tiefe 1 - ab rund 500 Tausend zieht die Tiefe klar und wachsend vorbei.",
    "Große Instanz (200 Stopps, 1 Million)": "200 Stopps: Tiefe 2 liegt bei 1.47 % gegen 1.66 % bei Tiefe 1 - bei größeren Instanzen zahlt sich die Tiefe schon bei kleinerem Budget aus als bei 60 Stopps.",
}
# Urteile, die bei diesem Preset über verschiedene Instanzen und Ketten-Seeds vorkommen (jedes Preset wird über mehrere Instanzen x 2 Ketten gemessen).
# Die Bänder sind bei diesem Stück breiter als sonst in der Linie üblich (WIN_MARGIN/LOSE_MARGIN nur 0.1 statt 0.3, s. lk_evaluation.py) -
# bei Abständen von 0.4-2 % schlägt schon eine kleine Streuung zwischen Instanzen/Ketten auf das Urteil durch.
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": {"beats_hc", "comparable", "hc_wins"},
    "Nur 2-opt (Tiefe 1)": {"comparable"},
    "Zu kleines Budget (10 Tausend)": {"beats_hc", "comparable", "hc_wins"},
    "Großes Budget (1 Million)": {"beats_hc", "comparable", "hc_wins"},
    "Große Instanz (200 Stopps, 1 Million)": {"beats_hc", "comparable", "hc_wins"},
}
