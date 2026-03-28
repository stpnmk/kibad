"""Stat card and KPI tile components."""
from __future__ import annotations

from dash import html


def stat_card(label: str, value, delta: str | None = None) -> html.Div:
    """KPI tile with optional delta badge.

    Parameters
    ----------
    label : str
        Small uppercase label above the value.
    value : str | int | float
        The main metric value.
    delta : str, optional
        Delta string like "+5.2%" — positive values get green, negative red.
    """
    children = [
        html.Div(label, className="kb-stat-label"),
        html.Div(str(value), className="kb-stat-value"),
    ]
    if delta is not None:
        is_positive = not str(delta).lstrip().startswith("-")
        cls = "kb-stat-delta kb-stat-delta--positive" if is_positive else "kb-stat-delta kb-stat-delta--negative"
        children.append(html.Span(delta, className=cls))

    return html.Div(children, className="kb-stat-card")
