"""Lin-Kernighan - eine Lieferrunde, deren Suche probeweise mehrere Kanten weit vorausschaut - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Siebtes Stück der Trajektorien-Metaheuristiken-Linie der "Konzepte"-Reihe, erstes Stück des Nachbarschafts-Zweigs
(Kind der Wurzel Hill Climbing, nicht der ILS/VNS/Tabu/GRASP-Kette). Ein einzelner 2-opt-Zug mit Kandidatenliste ist
bereits die Tiefe-1-Variante eines Lin-Kernighan-Zugs; LK verallgemeinert das, indem ein nicht sofort verbessernder
Zug probeweise WEITERGEFÜHRT statt verworfen wird, bis zu einer wählbaren Tiefe. Eingesetzt als austauschbare lokale
Suche innerhalb derselben Doppelbrücken-Kick-Schleife wie iterated-local-search-demo ("Chained LK", Applegate,
Cook & Rohe 2003). Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time
from dataclasses import replace

import numpy as np
import streamlit as st

import lk_constants as C
import lk_tour as A
from lk_evaluation import SWEEP_LABELS, Settings, analyse, chain_spread, restart_sweep, scaling_table, sweep, verdict
from lk_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_chain_seed,
    randomize_seed,
    sync_query_params,
)
from lk_visualization import build_budget, build_instance, build_restart_contrast, build_scaling, build_spread, build_sweep, build_tour, build_trace

st.set_page_config(page_title="Lin-Kernighan – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _restart_sweep(param, base):
    return restart_sweep(param, base)


@st.cache_data(show_spinner=False)
def _spread(base):
    return chain_spread(base)


@st.cache_data(show_spinner=False)
def _scaling(base):
    return scaling_table(base)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


st.title("🔗 Lin-Kernighan – eine Suche, die probeweise mehrere Kanten weit vorausschaut")
st.markdown(
    """
Ein 2-opt-Zug mit Kandidatenliste entfernt eine Kante, fügt eine kurze neue hinzu, entfernt eine zweite Kante und
schließt sofort mit einer dritten - das ist bereits ein Lin-Kernighan-Zug der **Tiefe 1**. **Lin-Kernighan**
(Lin & Kernighan 1973) verallgemeinert das: schließt der Zug bei Tiefe 1 nicht mit Gewinn, wird er **probeweise
weitergeführt** statt verworfen - eine weitere Kante wird entfernt, eine weitere hinzugefügt, bis zu einer wählbaren
**Tiefe** oder bis der kumulierte Gewinn positiv ist. Eingesetzt als lokale Suche innerhalb derselben Doppelbrücken-
Kick-Schleife wie die iterated-local-search-demo ("**Chained LK**", Applegate, Cook & Rohe 2003) - lohnt sich die
tiefere Suche, wenn man sie gezielt wiederverwendet statt sie unabhängig neu zu starten?
"""
)
st.caption(
    "Siebtes Stück der Trajektorien-Metaheuristiken-Linie der \"Konzepte\"-Reihe, erstes Stück des "
    "**Nachbarschafts-Zweigs** (Kind der Wurzel, nicht der ILS/VNS/Tabu/GRASP-Kette): dieselbe Rundtour wie in der "
    "[hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/), der "
    "[simulated-annealing-demo](https://sebastianhanisch-simulated-annealing-demo.streamlit.app/), der "
    "[iterated-local-search-demo](https://github.com/sebastian-hanisch/iterated-local-search-demo), der "
    "[variable-neighborhood-search-demo](https://github.com/sebastian-hanisch/variable-neighborhood-search-demo), der "
    "[tabu-search-demo](https://github.com/sebastian-hanisch/tabu-search-demo) und der "
    "[grasp-demo](https://github.com/sebastian-hanisch/grasp-demo) - ein Depot in der Mitte, n Kundenstopps in einem "
    "100 × 100-km-Gebiet, euklidische Entfernungen. VLSN und VRP-Nachbarschaften setzen später auf diesem Zweig auf."
)

with st.expander("So funktioniert Lin-Kernighan", expanded=True):
    st.markdown(
        """
1. **Kante entfernen, kurze Kante hinzufügen.** An einem Knoten t1 wird die aktuelle Kante entfernt; eine kurze neue
   Kante zu einem Kandidaten (den nächsten Nachbarn) wird hinzugefügt - wie bei 2-opt mit Kandidatenliste.
2. **Zweite Kante entfernen.** Das legt einen neuen freien Endpunkt frei; die dazu passende zweite Kante wird entfernt.
3. **Schließen oder weitermachen.** Verbessert das Schließen (die Tour jetzt wieder zu einem Kreis machen) die
   Gesamtlänge (kumulierter Gewinn negativ), wird committet. Sonst wird PROBEWEISE weitergemacht - Schritt 1-2 mit
   dem neuen freien Endpunkt -, bis zur gewählten **Tiefe** oder bis ein Schließen sich lohnt; gelingt das nicht,
   wird die ganze Kette rückgängig gemacht und der nächste Kandidat probiert.
4. **Tiefe 1 = 2-opt mit Kandidatenliste.** Bei Tiefe 1 entspricht diese Suche exakt Kandidatenliste + Don't-Look-
   Bits-2-opt aus der iterated-local-search-demo (geprüft, siehe Tests) - kein eigenständiges Risiko im Grenzfall.
5. **Chained LK.** Wie bei Iterated Local Search wird eine Doppelbrücke gekickt, nur die betroffene Umgebung neu
   durchsucht (mit Lin-Kernighan statt reinem 2-opt), und nur eine mindestens so kurze Tour übernommen.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stops = st.slider(
        "Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
        help="Anzahl der Kundenstopps (das Depot kommt dazu).",
    )
    cluster_share = st.slider(
        "Anteil der Stopps in Gruppen [%]", *bounds("ballung_slider"), key="ballung_slider", step=C.BALLUNG_STEP,
        help="Wie viele Stopps in fünf Gruppen (Städten) liegen statt gleichverteilt im Gebiet.",
    )
    depth = st.slider(
        "LK-Tiefe", C.DEPTH_MIN, C.DEPTH_MAX, key="depth_slider",
        help="Wie viele Kanten die Suche probeweise weiterverfolgt, bevor sie aufgibt. Tiefe 1 = reines 2-opt. Als Chained LK bei 200 Tausend "
             "Vorschlägen (60 Stopps): Tiefe 1 0.64 %, Tiefe 2 **0.66 %** (Voreinstellung) über der Schranke - bei diesem Standardbudget praktisch "
             "gleichauf. Erst ab rund 500 Tausend Vorschlägen zieht Tiefe 2 klar vorbei; bei größeren Instanzen (100+ Stopps) lohnt sie sich schon "
             "beim Standardbudget (siehe Budget- und Skalierungs-Experiment).",
    )
    budget = st.select_slider(
        "Budget (bewertete Kandidatenpaare)", options=list(C.BUDGETS), key="budget_select", format_func=_fmt_int,
        help="Bei 10 Tausend verliert Tiefe 2 klar gegen Tiefe 1 (1.77 gegen 1.36 %). Von 25 bis 200 Tausend sind beide etwa gleichauf (schwankt "
             "im Rauschen). Erst ab 500 Tausend zieht Tiefe 2 klar und wachsend vorbei (500 Tausend: 0.51 gegen 0.60 %; 2 Millionen: 0.47 gegen 0.58 %).",
    )
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für die Lage der Stopps.")
    chain_seed = st.number_input(
        "Zufalls-Seed der Kette", *bounds("chain_seed_input"), key="chain_seed_input", step=1,
        help="Steuert die zufällige Startlösung, die Doppelbrücken-Schnittpunkte und die Kandidatenwahl innerhalb der LK-Kette.",
    )
    st.button("🎲 Neue Kette würfeln", width="stretch", on_click=randomize_chain_seed, help="Würfelt einen neuen Seed für dieselbe Instanz.")

sync_query_params({
    "n_slider": int(n_stops), "ballung_slider": int(cluster_share), "seed_input": int(seed), "depth_slider": int(depth),
    "budget_select": int(budget), "chain_seed_input": int(chain_seed),
})

settings = Settings(int(n_stops), int(cluster_share), int(seed), int(depth), int(budget), int(chain_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
run = a.run
xy = a.inst.xy
code = verdict(a)
data_key = settings

# --- Lin-Kernighan in Aktion -----------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Lin-Kernighan in Aktion")
STEP_LABELS = {1: "1 · Instanz", 2: "2 · Erster Abstieg", 3: "3 · Kicks", 4: "4 · Ergebnis"}
if "lk_step" not in st.session_state or st.session_state.get("lk_step_owner") != data_key:
    st.session_state["lk_step"] = 1
    st.session_state["lk_step_owner"] = data_key
    st.session_state.pop("lk_iter", None)
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="lk_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

n_snaps = len(run.snapshots)
iteration = n_snaps - 1
play_kicks = False
if step == 3 and n_snaps > 1:
    it_col, itplay_col = st.columns([5, 2])
    with it_col:
        iteration = st.slider("Iteration", 0, n_snaps - 1, value=n_snaps - 1, key="lk_iter", help="Die Tour nach dieser Iteration (0 = nach dem ersten Abstieg, vor dem ersten Kick).")
    with itplay_col:
        play_kicks = st.button("▶️ Kicks abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    if n_snaps <= 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_snaps - 1, min(n_snaps, 40))})


def _render(current_step, it):
    with view_slot.container():
        if current_step == 1:
            st.markdown(f"**{a.inst.n} Kundenstopps und das Depot (Stern)** – {a.inst.cluster_share} % der Stopps in Gruppen")
            st.plotly_chart(build_instance(xy), width="stretch", key="s1_map")
        elif current_step == 2:
            c1, c2 = st.columns([3, 2])
            c1.markdown(f"**Tour nach dem ersten Abstieg** – {100 * (run.trace_length[0] - a.bound) / a.bound:.1f} % über der Schranke")
            c1.plotly_chart(build_tour(xy, run.snapshots[0]), width="stretch", key="s2_map")
            c2.markdown("**Vor den Kicks**")
            c2.metric("Startlösung", f"{a.start_gap:.1f} %", help="Abstand zur Schranke der Startlösung, vor jeder Optimierung.")
            c2.metric("Nach dem ersten LK-Abstieg", f"{100 * (run.trace_length[0] - a.bound) / a.bound:.1f} %", help="Abstand zur Schranke, bevor die erste Doppelbrücke angewendet wird.")
        elif current_step == 3:
            st.markdown("**Länge der aktuellen und der besten Tour über die bewerteten Kandidatenpaare**")
            st.plotly_chart(build_trace(run.trace_iter, run.trace_length, run.trace_best, a.bound, a.single.length, a.chain1.best_length), width="stretch", key=f"s3_trace_{it}")
            st.markdown(f"**Tour nach Iteration {it} von {n_snaps - 1}**")
            st.plotly_chart(build_tour(xy, run.snapshots[it], ghost=run.best_tour if it < n_snaps - 1 else None), width="stretch", key=f"s3_map_{it}")
        else:
            c1, c2 = st.columns(2)
            c1.markdown(f"**Chained LK: beste Tour** – {a.gap:.1f} % über der Schranke")
            c1.plotly_chart(build_tour(xy, run.best_tour), width="stretch", key="s4_lk")
            c2.markdown(f"**Chained LK, Tiefe 1** (gleiches Budget, fairer Vergleich) – {a.chain1_gap:.1f} % über der Schranke")
            c2.plotly_chart(build_tour(xy, a.chain1.best_tour), width="stretch", key="s4_chain1")


if auto_play:
    for s in STEP_LABELS:
        if s == 3:
            for f in _frames():
                _render(3, f)
                time.sleep(0.1)
            time.sleep(0.6)
        else:
            _render(s, iteration)
            time.sleep(1.2)
    step = 4
elif play_kicks:
    for f in _frames():
        _render(3, f)
        time.sleep(0.1)
else:
    _render(step, iteration)

if step == 1:
    st.caption(f"{a.inst.n} Stopps; die untere Schranke der kürzesten Rundtour liegt bei {a.bound:,.0f} km (1-Baum-Schranke, Held-Karp).".replace(",", "."))
elif step == 2:
    st.caption("Der erste Abstieg optimiert lokal (Lin-Kernighan bei der gewählten Tiefe), bis kein Zug mehr verbessert. Ab hier beginnen die Kicks.")
elif step == 3:
    st.caption(f"{_fmt_int(run.evaluations)} bewertete Kandidatenpaare in {run.iterations} Iterationen; angenommen wurden {a.accept_rate:.0%} davon.")
else:
    st.caption(f"Links die beste Tour aus {run.iterations} Iterationen bei Tiefe {settings.max_depth}, rechts dieselbe Kick-Schleife bei Tiefe 1 mit gleichem Bewertungsbudget ({_fmt_int(settings.budget)}).")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Kette gefunden hat")
st.caption(
    "**Abstand zur Schranke:** Länge der Tour gegenüber einer unteren Schranke der kürzesten Rundtour (1-Baum, Held-Karp) in Prozent. "
    "Ein Lauf ist eine Ziehung: die Ketten streuen (siehe Streuung unten), Vergleiche gelten für diesen Lauf."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Chained LK: beste Tour", f"{a.gap:.1f} %", delta=f"letzte Tour {a.final_gap:.1f} %", delta_color="off", help="Abstand zur Schranke der kürzesten je besuchten Tour; im Delta der der letzten Tour der Kette.")
m2.metric("Chained LK, Tiefe 1", f"{a.chain1_gap:.1f} %", delta="fairer Vergleich, gleiches Budget", delta_color="off", help="Dieselbe Kick-Schleife, aber mit reinem 2-opt (Tiefe 1) als lokaler Suche.")
m3.metric("Ein einzelner Abstieg", f"{a.single_gap:.1f} %", delta=f"{_fmt_int(a.single.evaluations)} Bewertungen", delta_color="off", help="Ein LK-Abstieg bei derselben Tiefe, kein Kick - die Obergrenze ohne Wiederverwendung.")
m4.metric("Angenommene Iterationen", f"{a.accept_rate:.0%}", delta=f"{run.accepted} von {run.iterations}", delta_color="off", help="Anteil der Kicks, deren Wiederabstieg mindestens so kurz wie die vorige Tour war.")

if code == "beats_hc":
    st.success(f"✅ Tiefere Suche zahlt sich aus: {a.gap:.1f} % über der Schranke gegen {a.chain1_gap:.1f} % bei Tiefe 1 (gleiches Budget, gleicher Kick-Mechanismus). Andere Ketten streuen um dieses Ergebnis.")
elif code == "comparable":
    st.info(f"ℹ️ Gleichauf: Tiefe {settings.max_depth} {a.gap:.1f} %, Tiefe 1 {a.chain1_gap:.1f} % über der Schranke. Eine andere Kette kann das Bild drehen.")
else:
    st.warning(f"⚠️ Tiefe 1 (reines 2-opt) ist hier besser: {a.chain1_gap:.1f} % gegen {a.gap:.1f} % über der Schranke. Bei knappem Budget passen zu wenige Kicks, um die teurere Tiefe zu amortisieren - siehe README 'Was nicht funktioniert hat'.")

d1, d2 = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    unit_time = lambda sec: f"{sec * 1000:.0f} ms"  # noqa: E731
    st.table({"": ["Länge (km)", "Abstand zur Schranke", "Bewertete Kandidatenpaare", "Rechenzeit"],
              "Chained LK (beste Tour)": [f"{run.best_length:.1f}", f"{a.gap:.2f} %", _fmt_int(run.evaluations), unit_time(a.seconds)],
              "Chained LK (letzte Tour)": [f"{run.final_length:.1f}", f"{a.final_gap:.2f} %", "–", "–"],
              "Chained LK, Tiefe 1": [f"{a.chain1.best_length:.1f}", f"{a.chain1_gap:.2f} %", _fmt_int(settings.budget), unit_time(a.chain1_seconds)],
              "Ein einzelner Abstieg": [f"{a.single.length:.1f}", f"{a.single_gap:.2f} %", _fmt_int(a.single.evaluations), unit_time(a.single_seconds)]})
with d2:
    st.markdown("**Was gerechnet wurde**")
    st.table({"": ["LK-Tiefe", "Budget", "Iterationen", "Kreuzungen der besten Tour"],
              "Einstellung": [f"{settings.max_depth}", _fmt_int(settings.budget), f"{run.iterations}", f"{a.crossings_end}"]})
    st.caption("Ein Vorschlag ist ein geprüftes Kandidatenpaar, dieselbe Einheit wie in der iterated-local-search-demo - der Kick selbst zählt nicht zum Budget. Rechenzeiten hängen vom Rechner ab, nur die Größenordnung zählt.")

st.markdown("---")

# --- Sweeps ------------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Budget, Tiefe und Instanz ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
base_sweep = replace(settings, seed=0, chain_seed=0)
if st.button("Sweep über 5 feste Instanzen berechnen (dauert etwa 20 bis 90 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Instanzen × 3 Ketten..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band) über 5 feste Instanzen (Seeds 100000–100004, getrennt vom Seed oben) mit je drei Ketten; alle anderen Regler wie in der Seitenleiste. "
               "Gepunktet: Chained LK bei Tiefe 1; gestrichelt: ein einzelner Abstieg.")

st.markdown("---")

# --- Experimente --------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Neustarts statt Kicks: hilft Tiefe auch ohne gezielte Wiederverwendung?")
if st.button("LK-Tiefe 1 bis 8 als Chained LK UND als unabhängige Neustarts vergleichen (dauert etwa 60 Sekunden)", key="restart_start"):
    st.session_state["restart_on"] = True
if st.session_state.get("restart_on"):
    with st.spinner("Rechne 7 Tiefen × 5 Instanzen × 3 Ketten, je Chained LK und Neustarts..."):
        rows_chain = _sweep("max_depth", base_sweep)
        rows_restart = _restart_sweep("max_depth", base_sweep)
    st.plotly_chart(build_restart_contrast(rows_chain, rows_restart, SWEEP_LABELS["max_depth"]), width="stretch", key="restart_chart")
    st.table({"Tiefe": [f"{r['value']}" for r in rows_chain], "Chained LK (%)": [f"{r['gap']:.2f}" for r in rows_chain],
              "Neustarts (%)": [f"{r['gap']:.2f}" for r in rows_restart], "Neustarts (Anzahl)": [f"{r['starts']:.1f}" for r in rows_restart]})
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (Budget wie in der Seitenleiste). Bei UNABHÄNGIGEN Neustarts hilft tiefere Suche NIE klar - alle Tiefen liegen bei 200 Tausend Vorschlägen um 0.68-0.72 %, "
               "mit wachsender Tiefe eher leicht SCHLECHTER (teurere Ketten kosten Neustarts, ohne die Güte je Neustart entsprechend zu verbessern). Als CHAINED LK schwankt Tiefe 2 bei diesem Standardbudget noch "
               "ähnlich knapp um Tiefe 1 - der Unterschied zeigt sich erst bei größerem Budget oder größerer Instanz (siehe die Experimente unten), wo Chained LK klar zulegt, während Neustarts es nie tun.")

st.markdown("---")

st.subheader("🔬 Budget: wann amortisiert sich die Tiefe?")
if st.button("Budget von 10 Tausend bis 2 Millionen durchfahren (dauert etwa 90 Sekunden)", key="budget_start"):
    st.session_state["budget_on"] = True
if st.session_state.get("budget_on"):
    with st.spinner("Rechne 8 Budgets × 5 Instanzen × 3 Ketten..."):
        rows_b = _sweep("budget", base_sweep)
    st.plotly_chart(build_budget(rows_b), width="stretch", key="budget_chart")
    st.table({"Budget": [_fmt_int(r["value"]) for r in rows_b], "Chained LK (%)": [f"{r['gap']:.2f}" for r in rows_b],
              "Tiefe 1 (%)": [f"{r['chain1']:.2f}" for r in rows_b], "ein Abstieg (%)": [f"{r['single']:.2f}" for r in rows_b]})
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (60 Stopps, Tiefe 2). Bei 10 Tausend verliert Tiefe 2 klar gegen Tiefe 1 (**1.77 %** gegen 1.36 %) - zu wenige Kicks passen ins Budget, um die teurere Tiefe "
               "zu amortisieren. Von 25 bis 200 Tausend sind beide etwa gleichauf (schwankt knapp hin und her, im Rauschen). Erst AB 500 Tausend zieht Tiefe 2 klar und wachsend vorbei (500 Tausend: **0.51 %** gegen "
               "0.60 %; 2 Millionen: **0.47 %** gegen 0.58 %) - anders als bei GRASP oder Tabu Search verpufft der Vorteil hier nicht bei großem Budget, sondern baut sich erst auf.")

st.markdown("---")

st.subheader("🔬 Streuung: wie verlässlich ist eine Kette?")
if st.button("20 Ketten auf dieser Instanz berechnen (dauert etwa 15 Sekunden)", key="spread_start"):
    st.session_state["spread_on"] = True
if st.session_state.get("spread_on"):
    with st.spinner("Rechne 20 Ketten..."):
        sp = _spread(replace(settings, chain_seed=0))
    st.plotly_chart(build_spread(sp["lk"], sp["chain1"]), width="stretch", key="spread_chart")
    s1, s2 = st.columns(2)
    s1.metric("Chained LK: Mittel ± Streuung", f"{sp['lk'].mean():.2f} ± {sp['lk'].std():.2f} %", help="Mittel und Standardabweichung des Abstands der besten Tour über 20 Ketten.")
    s2.metric("Chained LK, Tiefe 1: Mittel ± Streuung", f"{sp['chain1'].mean():.2f} ± {sp['chain1'].std():.2f} %", help="Dieselben 20 Ketten, aber mit reinem 2-opt (Tiefe 1) als lokaler Suche.")
    st.caption("Dieselbe Instanz, 20 verschiedene Ketten-Seeds (steuern Startlösung, Kicks und Kandidatenwahl).")

st.markdown("---")

st.subheader("🔬 Skalierung: wie viel Budget braucht ein größeres Problem?")
if st.button("Stopps von 20 bis 200 durchfahren (dauert etwa 90 Sekunden)", key="scaling_start"):
    st.session_state["scaling_on"] = True
if st.session_state.get("scaling_on"):
    with st.spinner("Rechne 6 Größen × 2 Budgetregeln × 5 Instanzen × 3 Ketten..."):
        sc = _scaling(replace(base_sweep, n=C.DEFAULT_N))
    st.plotly_chart(build_scaling(sc), width="stretch", key="scaling_chart")
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (Einstellungen wie in der Seitenleiste außer Stopps und Budget); zum Vergleich dieselbe Kick-Schleife bei Tiefe 1.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die tiefere Suche wird gezielt wiederverwendet (Chain), nicht weggeworfen (Neustart)** | Bei unabhängigen Neustarts hilft Tiefe NIE klar (alle Tiefen bei 200 Tausend um 0.68-0.72 %, eher leicht schlechter mit mehr Tiefe) - teurere Ketten kosten Neustarts, ohne die Güte je Neustart entsprechend zu verbessern. Als Chained LK zeigt sich dagegen bei genug Budget oder Instanzgröße ein klarer, wachsender Vorsprung. | Kein Nachfolger nötig - dieselbe Lehre wie Iterated Local Search selbst: teure Suche lohnt sich nur mit billiger Wiederverwendung |
| **Genug Budget ODER genug Instanzgröße, um die Tiefe zu amortisieren** | Bei 60 Stopps und dem Standardbudget (200 Tausend) sind Tiefe 1 und 2 praktisch gleichauf (0.64 gegen 0.66 %) - erst ab 500 Tausend zieht Tiefe 2 klar vorbei (0.51 gegen 0.60 %). Bei größeren Instanzen (ab 100 Stopps) gewinnt Tiefe 2 dagegen schon beim Standardbudget. | Kleineres n, größeres Budget, oder eine flachere Tiefe |
| **Ein einzelner Abstieg profitiert nur begrenzt von mehr Tiefe** | Bei einem einzelnen Abstieg sättigt die Güte bereits bei Tiefe 3 (6.45 % gegen 7.05 % bei Tiefe 1) - tiefer bringt nichts mehr, aber kostet weiter mehr Bewertungen je Neustart. | (kein Nachfolger nötig - der Sättigungspunkt ist strukturell, nicht regler-spezifisch) |
| **Kein volles Backtracking über Ebene 1 hinaus** | Ab Ebene 2 wird gierig nur der erste gültige Kandidat verfolgt (keine Rückverfolgung mehrerer Kandidaten wie im Original-Paper bei Ebene 1 UND 2) - eine bewusste Vereinfachung für die Demo. | **VLSN** (sehr große Nachbarschaften, systematisches Backtracking über eine ganze Klasse von Zügen) |
"""
)
st.caption(
    "Die Nachbarn des Nachbarschafts-Zweigs (noch nicht gebaut): VLSN (cyclic exchange, Dynasearch) und VRP-Nachbarschaften "
    "(inter-route-Züge) setzen auf Lin-Kernighan auf; ALNS ist der Konvergenzpunkt mit VNS, an eine CVRP-Instanz gebunden."
)

st.markdown("---")

with st.expander("🔬 Kleine Instanz: wo 2-opt hängen bleibt und Lin-Kernighan nicht"):
    st.markdown(
        """
Bei 8 Stopps (Brute-Force-Vergleich möglich) bleibt ein 2-opt-Abstieg (Tiefe 1) mit Kandidatenliste bei einer
bestimmten Startlösung bei **312.35 km** hängen - einem echten lokalen Optimum, das aber **6.53 %** über dem wahren
globalen Optimum liegt (**293.21 km**, per Brute-Force über alle 5040 Touren bestätigt). Schon **Tiefe 2** findet
von derselben Startlösung aus das globale Optimum - der probeweise weitergeführte Zug erkennt eine Verbesserung,
die der einstufige 2-opt-Zug nicht sehen kann. Das ist die Kernidee von Lin-Kernighan in Reinform: nicht jede
verbessernde Umstrukturierung ist mit einem einzelnen Kantentausch sichtbar.
        """
    )
    st.caption("Ein einzelnes, gezielt gefundenes Beispiel (Suche über 300 Zufallsinstanzen à 8-10 Stopps) - kein Beleg, dass das bei jeder Instanz oder jedem Start passiert, aber ein Existenzbeweis, dass es passieren kann.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Kürzeste Rundtour über $N = n+1$ Knoten mit euklidischen Entfernungen $d_{ij}$; $L(\pi)$ ist die Länge einer Tour $\pi$.

**LK-Zug (sequentiell, vereinfacht).** Ausgehend von $t_1$: Kante $(t_1, t_2)$ entfernen (Gewinn $g_1 = d(t_1,t_2)$), Kante $(t_2, t_3)$ zu
einem Kandidaten $t_3$ hinzufügen ($d(t_2,t_3) < g_1$, sonst kein Fortschritt möglich), Kante $(t_3, t_4)$ entfernen. Schließen mit
$(t_4, t_1)$ verbessert, wenn der kumulierte Delta $< 0$ ist; sonst wird ab $t_4$ probeweise weitergemacht (bis zur Tiefe $k$).

**Geometrie.** Jede Ebene ist ein 2-opt-artiger Segmentumkehr (wie bei Hill Climbing), aber verkettet: $t_1$ bleibt über die ganze Kette
fix, das jeweils freie Ende wandert. Bei Tiefe 1 ist das exakt ein 2-opt-Zug mit Kandidatenliste (siehe `lk_algorithm.py` für die volle
Herleitung und `tests/test_algorithm.py` für den Regressionsbeweis gegen `lk_dlb.dlb_descend`).

**Chained LK.** $\pi' = \text{Kick}(\pi)$ (Doppelbrücke, wie iterated-local-search-demo), dann $\pi'' = \text{LK}(\pi', k)$ nur um die vom
Kick betroffenen Knoten. Angenommen: $\pi \leftarrow \pi''$ falls $L(\pi'') \le L(\pi)$. Gemerkt wird $\arg\min L$ über alle je besuchten $\pi''$.

**Kennzahl.** Abstand zur Schranke $= 100 \cdot (L - w)/w$ mit der 1-Baum-Schranke $w$. Vergleichsgrößen bei gleichem Budget: dieselbe
Kick-Schleife bei Tiefe 1 (fairer Vergleich) und ein einzelner LK-Abstieg ohne Kick.

**Grenzen.** (1) Ab Ebene 2 kein Backtracking (nur der erste gültige Kandidat, keine Rückverfolgung wie im Original-Paper). (2) Tiefe
hilft nur bei gezielter Wiederverwendung (Kick), nicht bei unabhängigen Neustarts. (3) Ein einzelner Abstieg sättigt bereits bei Tiefe 3.

**Literatur.** Lin, S., & Kernighan, B. W. (1973). *An effective heuristic algorithm for the traveling-salesman problem.* Operations
Research, 21(2), 498-516. Applegate, D., Cook, W., & Rohe, A. (2003). *Chained Lin-Kernighan for large traveling salesman problems.*
INFORMS Journal on Computing, 15(1), 82-92.

Implementiert in `lk_algorithm.py` (die LK-Kette und die Chained-LK-Schleife), `lk_kick.py` (Doppelbrücke, aus iterated-local-search-demo),
`lk_dlb.py` (Kandidatenliste + Don't-Look-Bits-2-opt, die Tiefe-1-Referenz), `lk_tour.py` (Nachbarschaften, Abstieg, Schranke - aus der
Hill-Climbing-Demo), `lk_scenario.py` (Instanzen), `lk_evaluation.py` (Kennzahlen, Sweeps, Experimente, Urteil).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
