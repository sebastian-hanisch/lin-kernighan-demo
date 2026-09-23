"""Plotly-Abbildungen der Lin-Kernighan-Demo: Karten, Verlauf, Budget-/Tiefen-Vergleich, Sweeps, Streuung, Skalierung.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import lk_constants as C

TOUR_COLOR = "#4c78a8"
LK_COLOR = "#e45756"
CHAIN1_COLOR = "#54a24b"
SINGLE_COLOR = "#7f7f7f"
RESTART_COLOR = "#f58518"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_layout(fig, height=430):
    fig.update_xaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def _line_trace(xy, edges, color, name, dash=None, width=2.5, showlegend=True):
    x, y = [], []
    for a, b in edges:
        x += [xy[a, 0], xy[b, 0], None]
        y += [xy[a, 1], xy[b, 1], None]
    return go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width, dash=dash), name=name, hoverinfo="skip", showlegend=showlegend)


def _tour_edges_list(tour):
    t = np.asarray(tour)
    return list(zip(t.tolist(), np.roll(t, -1).tolist()))


def build_instance(xy):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=7, color=TOUR_COLOR, line=dict(width=1, color="white")), name="Stopps", hovertemplate="Stopp %{customdata}<extra></extra>", customdata=np.arange(1, len(xy))))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    return _map_layout(fig)


def build_tour(xy, tour, ghost=None):
    """Tour; `ghost` (optional) ist eine zweite Tour, die blass darunter gezeichnet wird (z. B. die beste Tour)."""
    fig = go.Figure()
    if ghost is not None:
        fig.add_trace(_line_trace(xy, _tour_edges_list(ghost), "#c9d6e6", "beste Tour", width=6))
    fig.add_trace(_line_trace(xy, _tour_edges_list(tour), TOUR_COLOR, "aktuelle Tour"))
    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=6, color="white", line=dict(width=1.5, color=TOUR_COLOR)), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    return _map_layout(fig)


def build_trace(trace_iter, trace_length, trace_best, bound, single_length, chain1_length):
    """Länge der aktuellen und der besten Tour über die bewerteten Kandidatenpaare; Schranke, ein einzelner Abstieg und Chained LK bei Tiefe 1 als Linien."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trace_iter, y=trace_length, mode="lines", line=dict(color=TOUR_COLOR, width=1.5), name="aktuelle Tour"))
    fig.add_trace(go.Scatter(x=trace_iter, y=trace_best, mode="lines", line=dict(color=LK_COLOR, width=2.5), name="beste Tour"))
    fig.add_hline(y=bound, line=dict(color=SINGLE_COLOR, dash="dot"), annotation_text="untere Schranke", annotation_position="bottom right")
    fig.add_hline(y=single_length, line=dict(color=SINGLE_COLOR, dash="dash"), annotation_text="ein Abstieg", annotation_position="top right")
    fig.add_hline(y=chain1_length, line=dict(color=CHAIN1_COLOR, dash="dash"), annotation_text="Chained LK, Tiefe 1", annotation_position="top right")
    fig.update_xaxes(title_text="Bewertete Kandidatenpaare", type="log")
    fig.update_yaxes(title_text="Länge (km)", range=[bound * 0.95, max(float(np.percentile(trace_length, 60)), single_length * 1.15)])
    return _base(fig, 340)


def build_budget(rows):
    """Abstand zur Schranke über das Budget: Chained LK bei der gewählten Tiefe gegen Tiefe 1 und einen einzelnen Abstieg."""
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=LK_COLOR, width=2.5), name="Chained LK"))
    fig.add_trace(go.Scatter(x=xs, y=[r["chain1"] for r in rows], mode="lines+markers", line=dict(color=CHAIN1_COLOR, width=2.5), name="Chained LK, Tiefe 1"))
    fig.add_trace(go.Scatter(x=xs, y=[r["single"] for r in rows], mode="lines", line=dict(color=SINGLE_COLOR, width=1.5, dash="dash"), name="ein Abstieg"))
    fig.update_xaxes(title_text="Budget (bewertete Kandidatenpaare)", type="log")
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 380)


def build_sweep(rows, param_label):
    """Abstand zur Schranke (Chained LK, Streuung als Band) und die Vergleichsgrößen über die Werte eines Reglers."""
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    upper = [r["gap"] + r["gap_sd"] for r in rows]
    lower = [max(0.0, r["gap"] - r["gap_sd"]) for r in rows]
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=upper + lower[::-1], fill="toself", fillcolor="rgba(228,87,86,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=LK_COLOR, width=2.5), name="Chained LK"))
    fig.add_trace(go.Scatter(x=xs, y=[r["chain1"] for r in rows], mode="lines+markers", line=dict(color=CHAIN1_COLOR, width=2, dash="dot"), name="Chained LK, Tiefe 1"))
    fig.add_trace(go.Scatter(x=xs, y=[r["single"] for r in rows], mode="lines", line=dict(color=SINGLE_COLOR, width=1.5, dash="dash"), name="ein Abstieg"))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 380)


def build_restart_contrast(chain_rows, restart_rows, param_label):
    """Chained LK gegen unabhängige LK-Neustarts über dieselbe Regler-Achse (`lk_evaluation.sweep` und `.restart_sweep`)."""
    xs = [r["value"] for r in chain_rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in chain_rows], mode="lines+markers", line=dict(color=LK_COLOR, width=2.5), name="Chained LK (Kick)"))
    fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in restart_rows], mode="lines+markers", line=dict(color=RESTART_COLOR, width=2.5), name="Unabhängige Neustarts"))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 380)


def build_spread(lk, chain1):
    fig = go.Figure()
    fig.add_trace(go.Box(y=lk, name="Chained LK", marker_color=LK_COLOR, boxmean=True))
    fig.add_trace(go.Box(y=chain1, name="Chained LK, Tiefe 1", marker_color=CHAIN1_COLOR, boxmean=True))
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    return _base(fig, 360)


def build_scaling(blocks):
    """`blocks`: [{"label": ..., "rows": [...]}, ...] (siehe lk_evaluation.scaling_table)."""
    fig = go.Figure()
    colors = (LK_COLOR, CHAIN1_COLOR)
    for block, color in zip(blocks, colors):
        label, rows = block["label"], block["rows"]
        xs = [r["value"] for r in rows]
        fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=color, width=2.5), name=label))
    fig.update_xaxes(title_text="Stopps")
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 360)
