# Lin-Kernighan – eine Suche, die probeweise mehrere Kanten weit vorausschaut – Streamlit-Demo

**[→ Demo live ausprobieren](#)** (Deploy offen)

Siebtes Stück der **Trajektorien-Metaheuristiken-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", erstes Stück des **Nachbarschafts-Zweigs**:
dieselbe Rundtour wie in der [hill-climbing-demo](../hill-climbing-demo), der [simulated-annealing-demo](../simulated-annealing-demo), der [iterated-local-search-demo](../iterated-local-search-demo), der [variable-neighborhood-search-demo](../variable-neighborhood-search-demo), der [tabu-search-demo](../tabu-search-demo) und der [grasp-demo](../grasp-demo) (ein Depot, n Kundenstopps in einem 100 × 100-km-Gebiet), dieselbe untere Schranke.

**Einordnung in die Reihe:** **Lin-Kernighan** (Lin & Kernighan 1973) ist ein Kind der Wurzel Hill Climbing, aber KEIN Kind der bisherigen ILS/VNS/Tabu/GRASP-Kette - es beginnt den **Nachbarschafts-Zweig**: statt der Metaheuristik (wann/wie oft suchen) verändert LK die **Nachbarschaft selbst**. Ein einzelner 2-opt-Zug mit Kandidatenliste ist bereits die Tiefe-1-Variante eines LK-Zugs; LK verallgemeinert das, indem ein nicht sofort verbessernder Zug probeweise WEITERGEFÜHRT statt verworfen wird, bis zu einer wählbaren Tiefe. Eingesetzt als lokale Suche innerhalb derselben Doppelbrücken-Kick-Schleife wie die iterated-local-search-demo ("**Chained LK**", Applegate, Cook & Rohe 2003) - der Cross-Edge, den die ILS-Demo bereits als offene Frage benannt hatte.
```
hill-climbing-demo (Wurzel: nur bergab, bleibt im ersten Optimum stecken)        [gebaut]
  ├─ simulated-annealing-demo (nimmt Verschlechterungen an, Abkühlplan)          [gebaut]
  ├─ iterated-local-search-demo (stört ein gutes Optimum mit fester Störstärke)  [gebaut]
  │     └─ variable-neighborhood-search-demo (Störstärke eskaliert + Reset)     [gebaut]
  ├─ tabu-search-demo (immer der beste Zug, Gedächtnis gegen Rückwege)          [gebaut]
  ├─ grasp-demo (randomisierte Konstruktion, viele Neuanfänge)                  [gebaut]
  └─ [Nachbarschafts-Zweig]
        lin-kernighan-demo (variable Tiefe statt fixer 2-opt-Nachbarschaft)     [dieses Stück]
          ^ Querkante zu ILS ("Chained LK" = ILS mit LK statt 2-opt als innerer Suche)
          └─ VLSN, VRP-Nachbarschaften                                          [nicht gebaut]
```

Ergebnis in Kürze: **tiefere Suche hilft - aber nur unter zwei Bedingungen: gezielte Wiederverwendung (Kick statt Neustart) UND genug Budget oder genug Instanzgröße.** Bei 60 Stopps und dem Standardbudget dieser Linie (200 Tausend) sind Chained LK bei Tiefe 2 und Tiefe 1 praktisch gleichauf (0.66 % gegen 0.64 % über der Schranke) - ein ehrlicher Befund: die Tiefe zahlt sich beim üblichen Standardbudget bei dieser Standardgröße noch NICHT aus. Erst **ab 500 Tausend** Vorschlägen zieht Tiefe 2 klar und wachsend vorbei (0.51 % gegen 0.60 %, bei 2 Millionen 0.47 % gegen 0.58 %); bei **größeren Instanzen** (ab 100 Stopps) gewinnt Tiefe 2 dagegen schon beim Standardbudget. Bei **unabhängigen Neustarts** (keine Kicks, jeder Versuch wirft die Tour weg) hilft Tiefe dagegen NIE klar (alle Tiefen ~0.7 %, eher leicht schlechter mit mehr Tiefe) - der Kontrast zeigt, dass es die gezielte Wiederverwendung ist, nicht die Tiefe allein, die den Unterschied macht. Ein einzelner Abstieg profitiert klar von Tiefe (7.05 % → 6.45 %), sättigt aber bereits bei Tiefe 3.

| Frage | Ergebnis (60 gleichverteilte Stopps, Tiefe 2, 200 Tausend Vorschläge, sofern nicht anders angegeben; Mittel über 5 feste Instanzen, Seeds 100000–100004, mit je 3 Ketten-Seeds; Abstand = Prozent über der 1-Baum-Schranke) |
|---|---|
| Standardfall | ➖ Chained LK **0.66 %** über der Schranke gegen **0.64 %** bei Tiefe 1 (dieselbe Kick-Schleife, praktisch gleichauf) und **6.51 %** für einen einzelnen Abstieg |
| **Ein einzelner Abstieg, Tiefe 1-8** | ✅ **7.05/6.51/6.45/6.45/6.45/6.45/6.45 %** - Tiefe 2 escaped die meisten 2-opt-Sackgassen, ab Tiefe 3 keine weitere Verbesserung (gesättigt) |
| **Unabhängige Neustarts, Tiefe 1-5** | ❌ **0.68/0.70/0.72/0.72 %** - Tiefe hilft hier NIE klar (eher leicht schlechter): teure Ketten kosten Neustarts |
| **Chained LK über das Budget (Tiefe 1 gegen 2)** | ⚠️ 10T: **1.36/1.77 %** (Tiefe 2 verliert klar). 25T-200T: etwa gleichauf (1.03/1.02, 0.79/1.02, 0.70/0.68, 0.64/0.66 %). AB 500T dreht sich das Bild klar: 0.60/0.51, 1M **0.58/0.47**, 2M **0.58/0.47 %** |
| **Kleine Instanz (8 Stopps)** | ✅ Ein 2-opt-Abstieg bleibt bei 312.35 km hängen (6.53 % über dem globalen Optimum 293.21 km, Brute-Force bestätigt) - Tiefe 2 findet das Optimum |
| **Größe (Tiefe 1 gegen 2, 200 Tausend)** | ✅ n=100: 1.14/1.06, n=150: 1.74/1.46, n=200: **2.09/1.82 %** - Tiefe 2 gewinnt ab 100 Stopps klar, auch beim Standardbudget |

## Was die Demo zeigt

1. **Lin-Kernighan in Aktion** (Schritt-Slider + Abspielen): **Instanz** → **Erster Abstieg** → **Kicks** (Iterations-Regler + ▶️ Kicks abspielen: Länge der aktuellen/besten Tour über die bewerteten Kandidatenpaare, dazu die Tour nach der gewählten Iteration) → **Ergebnis** (beste Tour neben derselben Kick-Schleife bei Tiefe 1).
2. **Was die Kette gefunden hat:** beste und letzte Tour, Tiefe-1-Vergleich, ein einzelner Abstieg, angenommene Iterationen; Urteil (`beats_hc` → `comparable` → `hc_wins`), Detailtabellen.
3. **📐 Sweeps** über Budget, LK-Tiefe, Stopps und Gruppen (feste Instanzen ab 100000, drei Ketten je Instanz).
4. **🔬 Experimente auf Abruf:** **Neustarts statt Kicks** (der Kontrast - hilft Tiefe auch ohne gezielte Wiederverwendung? Nein); Budget von 10 Tausend bis 2 Millionen (wann amortisiert sich die Tiefe?); Streuung über 20 Ketten; Skalierung von 20 bis 200 Stopps.
5. **🔬 Kleine Instanz** (eingeklappt): das konkrete 8-Stopps-Beispiel, bei dem 2-opt hängen bleibt und Tiefe 2 nicht.
6. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (gezielte Wiederverwendung statt Neustart, genug Budget zur Amortisation, Sättigung bei einem einzelnen Abstieg, kein volles Backtracking ab Ebene 2).

Regler: Stopps (10–200), Anteil der Stopps in Gruppen, **LK-Tiefe** (1–8), **Budget** (10 Tausend bis 2 Millionen bewertete Kandidatenpaare), Seed der Instanz (+ 🎲), Seed der Kette (+ 🎲).
Kein Regler für die Kandidatenbreite auf Ebene 1 (`breadth1`, fix auf 5 - Sensitivitäts-Check gemessen, ab 5 nahe gesättigt) oder die Störstärke (fix auf 1 Doppelbrücke - das ist ILS' eigenes Regler-Terrain, hier nicht neu exploriert).

## Messwerte der Presets (Instanz-Seed 35, Ketten-Seed 0; sie prüfen sich mit Urteil-Bändern selbst)

Die einzelne Standardinstanz landet hier durch Zufall sehr nah am echten Optimum (0.1 % statt der 0.66 % im Mittel über fünf Instanzen); die Mittelwerte oben in der Tabelle sind die belastbaren Zahlen.

| Preset | Urteil (Band über Instanzen × Ketten) |
|---|---|
| Standardfall (Voreinstellung) | beats_hc / comparable / hc_wins |
| Nur 2-opt (Tiefe 1) | comparable |
| Zu kleines Budget (10 Tausend) | beats_hc / comparable / hc_wins |
| Großes Budget (1 Million) | beats_hc / comparable / hc_wins |
| Große Instanz (200 Stopps, 1 Million) | beats_hc / comparable / hc_wins |

## Modell und Verfahren

- **Instanz, Nachbarschaften, Abstieg, Schranke** (`lk_scenario.py`, `lk_tour.py`): wortgleiche Kopien aus der [hill-climbing-demo](../hill-climbing-demo) (per Test gegen eingefrorene Werte), über die [iterated-local-search-demo](../iterated-local-search-demo).
- **Kandidatenliste + Don't-Look-Bits** (`lk_dlb.py`, wortgleiche Kopie aus `ils_dlb.py`): die bewährte 2-opt-Suche mit Kandidatenliste - bleibt eigenständig als die Tiefe-1-Referenz, nicht nur als Unterfall von LK.
- **Lin-Kernighan-Kette** (`lk_algorithm.py`): generalisiert `dlb_descend`s Kern-Schleife um probeweise verkettete Umkehrungen - bei jedem Kandidaten wird ein 2-opt-artiger Zug angewandt; verbessert das sofortige Schließen die Tour, wird committet, sonst probeweise mit dem neu freigelegten Endpunkt weitergemacht (bis zur Tiefe, dann rückgängig gemacht). Auf Ebene 1 bis zu 5 Kandidaten probiert (Breite), ab Ebene 2 gierig (kein Backtracking, dokumentierte Vereinfachung gegenüber dem Original-Paper). **Bei Tiefe 1 exakt gleich `dlb_descend`** (Regressionstest, kein eigenständig hergeleitetes Risiko - siehe Modul-Docstring in `lk_algorithm.py` für die vollständige Geometrie-Herleitung).
- **Chained LK** (`lk_algorithm.chained_run`, `lk_kick.py` wortgleich aus `ils_kick.py`): dieselbe Doppelbrücken-Kick-Schleife wie iterated-local-search-demo, hier mit `lk_descend` statt Kandidatenliste+DLB-2-opt als lokaler Suche - der Cross-Edge aus den DAG-Notizen ("LK is the strong one" als austauschbare lokale Suche).
- **Unabhängige Neustarts** (`lk_evaluation.lk_restarts`): derselbe LK-Abstieg, aber ohne Kick - jeder Versuch konstruiert eine frische Zufallstour. Der Kontrast, der zeigt, dass Tiefe NUR mit gezielter Wiederverwendung hilft.
- **Auswertung** (`lk_evaluation.py`): Kennzahlen, Urteil, Sweeps über feste Instanzen × Ketten, Streuung, Skalierung.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutung: "tiefere Suche schlägt Kandidatenlisten-2-opt bei gleichem Budget"** – **bestätigt, aber NUR unter zwei Bedingungen, nicht beim Standardfall dieser Linie**. Beim üblichen Standardbudget (200 Tausend) UND der Standardgröße (60 Stopps) sind Chained LK bei Tiefe 2 und Tiefe 1 praktisch gleichauf (0.66 % gegen 0.64 %) - kein klarer Sieg, obwohl die Vorab-Vermutung genau das erwartet hätte. Der Vorteil zeigt sich erst mit MEHR Budget (ab 500 Tausend, wachsend bis 0.47 % gegen 0.58 % bei 1-2 Millionen) ODER mit einer GRÖSSEREN Instanz (ab 100 Stopps gewinnt Tiefe 2 schon beim Standardbudget, bei 200 Stopps 1.82 % gegen 2.09 %) - anders als bei GRASP oder Tabu Search verpufft der Vorteil hier NICHT bei großem Budget, sondern baut sich erst auf. Bei UNABHÄNGIGEN Neustarts dagegen verschwindet der Vorteil bei JEDEM Budget (alle Tiefen bei 200 Tausend 0.68-0.72 %, teils leicht schlechter bei mehr Tiefe) - ein ehrlicher Negativbefund: teurere Neustarts kosten Neustart-Anzahl, ohne die Güte je Neustart entsprechend zu verbessern (bei so vielen billigen Neustarts gewinnt oft die Menge gegen die Qualität). Das ist die schärfste, am klarsten differenzierte Erkenntnis der Demo: Tiefe ist kein Selbstzweck, sondern zahlt sich nur mit der richtigen Einbettung UND genug Budget oder Größe aus - und beim in dieser Linie üblichen Standardfall (60 Stopps, 200 Tausend) eben noch nicht.
- **Ein einzelner Abstieg sättigt schnell.** Tiefe 2 escaped bereits die meisten 2-opt-Sackgassen (7.05 % → 6.51 %), ab Tiefe 3 bringt mehr Tiefe nichts mehr (6.45 % bis Tiefe 8) - aber kostet in der Chained-LK-Schleife weiter mehr Bewertungen je Kick. Der kalibrierte Sweet Spot (Tiefe 2) liegt darum NICHT beim Sättigungspunkt eines einzelnen Abstiegs (Tiefe 3), sondern niedriger.
- **Kein volles Backtracking über Ebene 1 hinaus** (dokumentierte Vereinfachung gegenüber dem Original-Paper, das auch auf Ebene 2 mehrere Kandidaten probiert): ab Ebene 2 wird gierig nur der erste gültige Kandidat verfolgt. Die Breite auf Ebene 1 (`breadth1`) ist ab 2 im Rauschen (kein klarer weiterer Trend, nur breadth1=1 ist deutlich schlechter) und deshalb als Konstante fixiert statt als Regler.
- **Synthetische Instanzen:** euklidisch, gleichverteilt oder in fünf Gruppen, ein Fahrzeug, keine Kapazitäten oder Zeitfenster. Zeiten hängen vom Rechner und der Python-Version ab (die Tests prüfen nur Größenordnungen).

## Verifikation

- **Zentrale Regression:** `lk_descend(max_depth=1, breadth1>=Kandidatenlisten-Größe)` liefert für jeden getesteten Seed BYTE-GLEICH dieselbe Tour und Bewertungszahl wie `lk_dlb.dlb_descend` (auch mit dem `touched`-Kurzweg) - kein eigenständig hergeleitetes Risiko im Grenzfall, nur die verkettete Vertiefung ist neu.
- **LK-Kette:** gültige Permutation nach jeder Tiefe, Längenbuchhaltung stimmt mit neu berechneter Tourlänge überein, strikt monotone Verbesserung, Rückgängig-Mechanik bei erfolgloser Kette geprüft (Umkehrung ist selbstinvers), Tiefen-Buchführung (`depth_hist`) summiert korrekt.
- **Kleine-Instanz-Demonstration:** eine konkrete 8-Stopps-Instanz, bei der ein 2-opt-Abstieg (Tiefe 1) nachweislich in einem echten, aber suboptimalen lokalen Optimum hängen bleibt (312.35 km, 6.53 % über dem per Brute-Force bestätigten globalen Optimum 293.21 km) - Tiefe 2 findet von derselben Startlösung aus das Optimum.
- Übernommener Kern: 2-opt gegen Brute-Force, Abstieg strikt monoton und im lokalen Optimum, Bewertungsbudget, 1-Baum-Schranke gegen Brute-Force (n = 8) und CP-SAT (n = 20); Instanz gegen eingefrorene Werte; Doppelbrücke erschöpfend gegen alle 2-opt-Züge (aus der iterated-local-search-demo übernommen).
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, Tiefen-, Budget-, Neustart-Kontrast- und Größen-Aussagen; jeweils Mittel über die festen Sweep-Instanzen × Ketten; positive **und** negative Aussagen; Rechenzeiten nur als Größenordnung); alle 5 Presets über mehrere Instanzen und Ketten in Urteil-Bändern; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt bei 10 und 60 Stopps, Iterations-Regler, ▶️ Abspielen und ▶️ Kicks abspielen ohne doppelte Schlüssel, Würfel-Knöpfe, Permalink-Grenzen, Extremwerte, Experimente auf Abruf, Footer).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Neustart-Kontrast, Budget, Streuung, Skalierung), Kleine-Instanz-Demo, 🚧 Grenzen, Mathe |
| `lk_algorithm.py` | Die LK-Kette (verkettete probeweise Umkehrungen) und die Chained-LK-Schleife |
| `lk_dlb.py` | Kandidatenliste + Don't-Look-Bits-2-opt, die Tiefe-1-Referenz (aus der iterated-local-search-demo) |
| `lk_kick.py` | Doppelbrücken-Zug (aus der iterated-local-search-demo) |
| `lk_tour.py` | Nachbarschaften, Abstieg (mit Bewertungsbudget), Kreuzungen, 1-Baum-Schranke (aus der Hill-Climbing-Demo) |
| `lk_scenario.py`, `lk_constants.py` | Instanzen; Konstanten, Presets |
| `lk_evaluation.py` | Analyse, Urteil, unabhängige Neustarts, Sweeps, Streuung, Skalierung |
| `lk_presets.py`, `lk_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Übernommener Kern, LK-Kette (Tiefe-1-Regression), Chained LK, Szenario und Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
