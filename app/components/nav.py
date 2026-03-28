"""Navigation components – sidebar and tab bar."""
from __future__ import annotations

from dash import html, dcc

# Sidebar icon map: short text labels instead of emoji
_ICON_MAP = {
    "house": "~",
    "database": "Д",
    "wrench": "П",
    "diagram-3": "С",
    "table": "Т",
    "search": "И",
    "clipboard-data": "Тс",
    "graph-up": "ВР",
    "pie-chart": "Ф",
    "dice-5": "М",
    "file-earmark-text": "О",
    "question-circle": "?",
    "people": "К",
    "tags": "АВ",
    "funnel": "В",
    "bullseye": "Ц",
    "arrow-left-right": "Ср",
    "bar-chart": "Гр",
    "gear": "Н",
    "card-text": "Тк",
    "grid": "Кт",
    "robot": "АА",
    "layers": "Шб",
    "percent": "%",
}


def sidebar_nav(pages) -> html.Div:
    """Left sidebar with icons for page groups.

    Parameters
    ----------
    pages : iterable
        Values from ``dash.page_registry``, each having ``path``, ``name``,
        ``order``, and optionally ``icon``.
    """
    sorted_pages = sorted(pages, key=lambda p: p.get("order", 99))

    nav_items = []
    for page in sorted_pages:
        icon_name = page.get("icon", "")
        icon_char = _ICON_MAP.get(icon_name, icon_name if len(str(icon_name)) <= 2 else "-")
        name = page.get("name", "")

        nav_items.append(
            dcc.Link(
                html.Div([
                    html.Span(icon_char, style={"fontSize": "1.1rem", "width": "24px", "textAlign": "center"}),
                    html.Span(name, className="kb-sidebar-item-label"),
                ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
                href=page["path"],
                className="kb-sidebar-item",
            )
        )

    return html.Div([
        # Brand
        html.Div([
            html.Span("K", style={"fontWeight": "800", "fontSize": "1rem"}),
        ], className="kb-sidebar-brand"),

        # Navigation
        html.Div(nav_items, className="kb-sidebar-nav"),

    ], className="kb-sidebar", id="sidebar")


def tab_bar(tabs: list[dict], active: str) -> dcc.Tabs:
    """Top-level page tabs.

    Parameters
    ----------
    tabs : list[dict]
        Each dict has ``label`` and ``value`` keys.
    active : str
        Value of the currently active tab.
    """
    return dcc.Tabs(
        id="page-tabs",
        value=active,
        children=[
            dcc.Tab(label=t["label"], value=t["value"], className="tab",
                    selected_className="tab--selected")
            for t in tabs
        ],
        className="kb-tabs",
    )
