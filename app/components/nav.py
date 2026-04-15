"""Navigation components – sidebar and tab bar."""
from __future__ import annotations

from dash import html, dcc
import dash_bootstrap_components as dbc

# Sidebar icon map: short text labels
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

# Section groupings by order range: (min_order, max_order, label)
_SECTIONS = [
    (0, 0, None),          # Home — no section header
    (1, 4, "Данные"),      # Data wrangling pages
    (5, 19, "Анализ"),     # Analytics & modeling
    (20, 99, "Инструменты"),  # Tools & export
]


def _make_nav_item(page: dict) -> list:
    """Build a single sidebar navigation link and its collapsed-mode tooltip.

    Returns a list containing the ``dcc.Link`` and a ``dbc.Tooltip`` so that
    collapsed icons show a right-side tooltip with the page name.
    """
    icon_name = page.get("icon", "")
    icon_char = _ICON_MAP.get(icon_name, icon_name if len(str(icon_name)) <= 2 else "-")
    # Strip the number prefix from names like "5. Исследование" → "Исследование"
    name = page.get("name", "")
    if name and len(name) > 2 and name[0].isdigit() and ". " in name[:5]:
        name = name.split(". ", 1)[1]

    # Unique, stable ID derived from the page path (e.g. "/data" → "sidebar-icon-data").
    # Root path "/" becomes "sidebar-icon-home".
    slug = page["path"].strip("/").replace("/", "-") or "home"
    icon_id = f"sidebar-icon-{slug}"

    link = dcc.Link(
        html.Div([
            html.Span(
                icon_char,
                className="kb-sidebar-icon",
                id=icon_id,
            ),
            html.Span(name, className="kb-sidebar-item-label"),
        ], className="kb-sidebar-item-inner"),
        href=page["path"],
        className="kb-sidebar-item",
    )
    tooltip = dbc.Tooltip(
        name,
        target=icon_id,
        placement="right",
        delay={"show": 400, "hide": 100},
    )
    return [link, tooltip]


def sidebar_nav(pages) -> html.Div:
    """Left sidebar with grouped sections and scroll.

    Parameters
    ----------
    pages : iterable
        Values from ``dash.page_registry``, each having ``path``, ``name``,
        ``order``, and optionally ``icon``.
    """
    sorted_pages = sorted(pages, key=lambda p: p.get("order", 99))

    nav_children = []
    for min_o, max_o, section_label in _SECTIONS:
        group = [p for p in sorted_pages if min_o <= p.get("order", 99) <= max_o]
        if not group:
            continue
        if section_label:
            nav_children.append(
                html.Div(section_label, className="kb-sidebar-section")
            )
        for page in group:
            nav_children.extend(_make_nav_item(page))

    return html.Div([
        # Brand area
        html.Div([
            html.Div([
                html.Span("K", className="kb-sidebar-brand-letter"),
            ], className="kb-sidebar-brand"),
            html.Span("KIBAD", className="kb-sidebar-brand-text"),
        ], className="kb-sidebar-header"),

        # Scrollable navigation area
        html.Div(nav_children, className="kb-sidebar-nav"),

        # Footer: version badge
        html.Div([
            html.Div("v5.0", className="kb-sidebar-version"),
        ], className="kb-sidebar-footer"),
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
