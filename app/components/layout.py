"""Page layout components – headers, sections, wrappers."""
from __future__ import annotations

from dash import html


def section_header(title: str, subtitle: str | None = None) -> html.Div:
    """Page section heading with optional subtitle."""
    children = [html.H2(title)]
    if subtitle:
        children.append(html.P(subtitle))
    return html.Div(children, className="kb-section-header")


def page_header(title: str, subtitle: str | None = None, icon: str | None = None) -> html.Div:
    """Page-level header with icon, title and subtitle."""
    title_children = []
    if icon:
        title_children.append(html.Span(icon, style={"fontSize": "1.5rem"}))
    title_children.append(title)

    children = [html.H1(title_children)]
    if subtitle:
        children.append(html.P(subtitle))
    return html.Div(children, className="kb-page-header")


def empty_state(icon: str = "", title: str = "", description: str = "") -> html.Div:
    """Render a centered empty-state placeholder."""
    return html.Div([
        html.Div(icon, className="kb-empty-state-icon"),
        html.Div(title, className="kb-empty-state-title"),
        html.Div(description, className="kb-empty-state-desc"),
    ], className="kb-empty-state")


def controls_panel(title: str, children) -> html.Div:
    """Wrap form controls in a styled panel."""
    return html.Div([
        html.Div(title, className="kb-controls-title"),
        *children,
    ], className="kb-controls")
