"""Sidebar navigation — grouped sections, active indicator driven by URL."""
from __future__ import annotations

import dash
from dash import Input, Output, clientside_callback, dcc, html

from app.components.icons import icon

# Bootstrap-icon → Lucide-icon mapping (pages declare Bootstrap names in
# ``dash.register_page(icon=...)``; we translate to our Lucide set).
_ICON_MAP = {
    "house":             "home",
    "database":          "database",
    "wrench":            "wrench",
    "diagram-3":         "layers",
    "table":             "table",
    "search":            "search",
    "clipboard-data":    "file-text",
    "graph-up":          "trend",
    "pie-chart":         "pie-chart",
    "dice-5":            "dice",
    "file-earmark-text": "file-text",
    "question-circle":   "help",
    "people":            "users",
    "tags":              "tag",
    "funnel":            "funnel",
    "bullseye":          "target",
    "arrow-left-right":  "arrows-lr",
    "bar-chart":         "bar-chart",
    "gear":              "settings",
    "card-text":         "file-text",
    "grid":              "grid",
    "robot":             "robot",
    "layers":            "layers",
    "percent":           "percent",
}

# Section groupings by order range: (min_order, max_order, label)
_SECTIONS = [
    (0, 0, None),             # Home — no section header
    (1, 4, "Данные"),         # Data wrangling
    (5, 19, "Анализ"),        # Analytics & modeling
    (20, 99, "Инструменты"),  # Tools & export
]


def _strip_order_prefix(name: str) -> str:
    """Remove a leading ``"5. "`` / ``"12. "`` prefix from a page name."""
    if name and len(name) > 2 and name[0].isdigit() and ". " in name[:5]:
        return name.split(". ", 1)[1]
    return name


def _make_nav_item(page: dict) -> dcc.Link:
    """Build one sidebar link. Active-state is toggled by a clientside callback."""
    lucide_name = _ICON_MAP.get(page.get("icon", ""), "dot")
    return dcc.Link(
        html.Div(
            [
                html.Span(icon(lucide_name, 14), className="kb-sidebar-icon"),
                html.Span(_strip_order_prefix(page.get("name", "")),
                          className="kb-sidebar-item-label"),
            ],
            className="kb-sidebar-item-inner",
        ),
        href=page["path"],
        className="kb-sidebar-item",
        refresh=False,
    )


def sidebar_nav(pages) -> html.Div:
    """Left sidebar with grouped sections and the KIBAD brand at top."""
    sorted_pages = sorted(pages, key=lambda p: p.get("order", 99))

    nav_children: list = []
    for min_o, max_o, section_label in _SECTIONS:
        group = [p for p in sorted_pages if min_o <= p.get("order", 99) <= max_o]
        if not group:
            continue
        if section_label:
            nav_children.append(
                html.Div(section_label, className="kb-sidebar-section")
            )
        for page in group:
            nav_children.append(_make_nav_item(page))

    return html.Div([
        # Brand mark + wordmark
        html.Div([
            html.Div(
                html.Span("K", className="kb-sidebar-brand-letter"),
                className="kb-sidebar-brand",
            ),
            html.Span("KIBAD", className="kb-sidebar-brand-text"),
        ], className="kb-sidebar-header"),

        # Scrollable navigation area
        html.Div(nav_children, className="kb-sidebar-nav", id="kb-sidebar-nav"),

        # Version footer
        html.Div(
            html.Span("v 2026.4", className="kb-sidebar-version"),
            className="kb-sidebar-footer",
        ),
    ], className="kb-sidebar", id="sidebar")


# ---------------------------------------------------------------------------
# Active-item highlight — toggles ``.active`` on the link whose data-path
# matches the current URL. Runs in the browser via clientside_callback.
# ---------------------------------------------------------------------------
clientside_callback(
    """
    function(pathname) {
        if (!pathname) return window.dash_clientside.no_update;
        const nav = document.getElementById('kb-sidebar-nav');
        if (!nav) return window.dash_clientside.no_update;
        const links = nav.querySelectorAll('a.kb-sidebar-item');
        links.forEach(a => {
            // dcc.Link renders <a href="/path">. Match on pathname exactly.
            const hrefAttr = a.getAttribute('href') || '';
            // href may include the leading slash; strip query/hash just in case.
            const cleaned = hrefAttr.split('?')[0].split('#')[0];
            if (cleaned === pathname) a.classList.add('active');
            else a.classList.remove('active');
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output("kb-sidebar-nav", "data-active"),  # dummy output
    Input("url", "pathname"),
)


def tab_bar(tabs: list[dict], active: str) -> dcc.Tabs:
    """Top-level page tabs."""
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
