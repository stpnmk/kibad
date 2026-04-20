"""Card / KPI / chip / callout primitives."""
from __future__ import annotations

from typing import Iterable

from dash import html

from app.components.icons import icon


# ---------------------------------------------------------------------------
# Stat / KPI tile
# ---------------------------------------------------------------------------
def stat_card(label: str, value, delta: str | None = None) -> html.Div:
    """KPI tile with optional delta badge.

    Parameters
    ----------
    label : str
        Small uppercase label above the value.
    value : str | int | float
        The main metric value.
    delta : str, optional
        Delta string like ``"+5.2%"`` — positive values get the accent pill,
        negative values get the danger pill.
    """
    children: list = [
        html.Div(label, className="kb-stat-label"),
        html.Div(str(value), className="kb-stat-value"),
    ]
    if delta is not None:
        is_positive = not str(delta).lstrip().startswith("-")
        cls = "kb-stat-delta kb-stat-delta--positive" if is_positive \
            else "kb-stat-delta kb-stat-delta--negative"
        children.append(html.Span(delta, className=cls))

    return html.Div(children, className="kb-stat-card")


def kpi(label: str, value, delta: str | None = None, hint: str | None = None) -> html.Div:
    """Design-spec KPI card: label top, value bottom, delta + optional hint.

    Matches the artboard layout (14×16 padding, value row at bottom with
    trend). Use for dashboard overview metrics.
    """
    bottom_children: list = [html.Div(str(value), className="kb-stat-value")]
    if delta is not None:
        is_positive = not str(delta).lstrip().startswith("-")
        cls = "kb-stat-delta kb-stat-delta--positive" if is_positive \
            else "kb-stat-delta kb-stat-delta--negative"
        bottom_children.append(html.Span(delta, className=cls))

    children: list = [
        html.Div(label, className="kb-stat-label"),
        html.Div(bottom_children, className="kb-stat-row"),
    ]
    if hint:
        children.append(html.Div(hint, className="caption",
                                 style={"color": "var(--text-tertiary)",
                                        "fontSize": "11px"}))
    return html.Div(children, className="kb-stat-card")


# ---------------------------------------------------------------------------
# Chips
# ---------------------------------------------------------------------------
def chip(text: str, variant: str = "neutral") -> html.Span:
    """Uppercase status chip. ``variant`` ∈ {success, warning, danger, info, neutral}."""
    return html.Span(text, className=f"kb-chip kb-chip--{variant}")


def mchip(text: str, removable: bool = False) -> html.Span:
    """Multi-select chip (mono, non-uppercase) matching Dropdown(multi=True) styling.

    Use when rendering a static list of selected values outside of a Dropdown.
    """
    children: list = [html.Span(text)]
    if removable:
        children.append(html.Span(
            icon("x", 10), className="x", role="button",
            **{"aria-label": f"Убрать {text}"},
        ))
    return html.Span(children, className="kb-mchip")


# ---------------------------------------------------------------------------
# How-to callout
# ---------------------------------------------------------------------------
def howto(*content, icon_name: str = "info") -> html.Div:
    """Inline help block — a short "how to read this" note beside a chart/table.

    Accepts a mix of strings / Dash components as positional args.
    """
    return html.Div(
        [
            html.Span(icon(icon_name, 14),
                      style={"color": "var(--accent-500)", "flexShrink": 0,
                             "marginTop": "2px"}),
            html.Div(list(content)),
        ],
        className="kb-howto",
    )


# ---------------------------------------------------------------------------
# Pill-radio group (controlled by a dcc.RadioItems or standalone labels)
# ---------------------------------------------------------------------------
def pill_radio(options: Iterable[dict], active_value: str | None = None,
               id_: str | None = None) -> html.Div:
    """Segmented-control style radio group — use with ``dcc.Input(type='radio')`` pattern,
    or as a pure-presentation group when state is managed elsewhere.

    Parameters
    ----------
    options : iterable of dict
        Each has ``label`` and ``value``.
    active_value : str, optional
        Which option is currently selected.
    id_ : str, optional
        Wrapper id.
    """
    children = []
    for opt in options:
        is_active = opt["value"] == active_value
        cls = "active" if is_active else ""
        children.append(
            html.Label(opt["label"], className=cls,
                       **{"data-value": opt["value"]})
        )
    props = {"className": "kb-pill-radio"}
    if id_:
        props["id"] = id_
    return html.Div(children, **props)


# ---------------------------------------------------------------------------
# Card with head
# ---------------------------------------------------------------------------
def card(title: str | None = None, subtitle: str | None = None,
         *, head_right=None, size: str = "md", children=None) -> html.Div:
    """Styled card with an optional title/subtitle head.

    Parameters
    ----------
    title : str, optional
        Bold title.
    subtitle : str, optional
        Caption below the title.
    head_right : component, optional
        Something to align to the right of the head (e.g. a chip, button).
    size : {'sm', 'md', 'lg'}
        Padding size class.
    children : list
        Body children.
    """
    cls = "kb-card"
    if size == "sm":
        cls += " kb-card--sm"
    elif size == "lg":
        cls += " kb-card--lg"

    kids: list = []
    if title or subtitle or head_right:
        head_left_children: list = []
        if title:
            head_left_children.append(html.H3(title))
        if subtitle:
            head_left_children.append(html.Div(subtitle, className="caption",
                                               style={"color": "var(--text-secondary)",
                                                      "fontSize": "12px"}))
        head_children = [html.Div(head_left_children, className="kb-card-title")]
        if head_right is not None:
            head_children.append(html.Div(head_right))
        kids.append(html.Div(head_children, className="kb-card-head"))

    if children:
        kids.extend(children if isinstance(children, list) else [children])

    return html.Div(kids, className=cls)
