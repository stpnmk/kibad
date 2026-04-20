"""Top navigation bar — sticky breadcrumb + search + help/bell/avatar.

The breadcrumb is driven by a clientside callback that reads the current URL
and looks up the page name in a JSON map built at render time from
``dash.page_registry``.
"""
from __future__ import annotations

import json

from dash import Input, Output, clientside_callback, html

from app.components.icons import icon


def topbar(pages, user_initials: str = "АК") -> html.Div:
    """Render the topbar once, at the top of the main content area.

    Parameters
    ----------
    pages : iterable
        ``dash.page_registry.values()`` — used to build pathname→title map.
    user_initials : str
        Two-letter initials shown in the avatar (default Russian «АК»).
    """
    path_to_name = {}
    for p in pages:
        nm = p.get("name", "")
        if nm and len(nm) > 2 and nm[0].isdigit() and ". " in nm[:5]:
            nm = nm.split(". ", 1)[1]
        path_to_name[p["path"]] = nm

    # Page-path → section-label (matches sidebar grouping so the breadcrumb
    # can show "Студия / Анализ / EDA" instead of a flat "EDA").
    section_for = {}
    for p in pages:
        order = p.get("order", 99)
        if order == 0:
            section_for[p["path"]] = None
        elif 1 <= order <= 4:
            section_for[p["path"]] = "Данные"
        elif 5 <= order <= 19:
            section_for[p["path"]] = "Анализ"
        else:
            section_for[p["path"]] = "Инструменты"

    route_map = {
        path: {"name": path_to_name[path], "section": section_for[path]}
        for path in path_to_name
    }

    return html.Div(
        [
            # Breadcrumb (populated by clientside callback)
            html.Div(
                id="kb-breadcrumb",
                className="kb-topbar-breadcrumb",
                children=[html.Span("Студия", className="current")],
                **{"data-routes": json.dumps(route_map, ensure_ascii=False)},
            ),
            html.Div(className="kb-topbar-spacer"),

            # Search (visual only for now — ⌘K palette arrives later)
            html.Div(
                [
                    html.Span(icon("search", 14), className="icon"),
                    html.Span("Поиск по студии…"),
                    html.Span("⌘K", className="kbd-hint"),
                ],
                className="kb-topbar-search",
                role="search",
                tabIndex=0,
            ),

            # Help
            html.Button(icon("help", 16), className="kb-topbar-icon-btn",
                        id="kb-topbar-help", **{"aria-label": "Справка"}),

            # Bell (with dot)
            html.Button(
                [icon("bell", 16), html.Span(className="dot")],
                className="kb-topbar-icon-btn",
                id="kb-topbar-bell",
                **{"aria-label": "Уведомления"},
            ),

            # Avatar
            html.Div(user_initials, className="kb-topbar-avatar",
                     title="Профиль"),
        ],
        className="kb-topbar",
        id="kb-topbar",
    )


# Clientside callback: update breadcrumb on URL change.
clientside_callback(
    """
    function(pathname) {
        const el = document.getElementById('kb-breadcrumb');
        if (!el || !pathname) return window.dash_clientside.no_update;
        let routes = {};
        try { routes = JSON.parse(el.getAttribute('data-routes') || '{}'); }
        catch(e) { return window.dash_clientside.no_update; }
        const meta = routes[pathname] || { name: 'Студия', section: null };

        const parts = ['Студия'];
        if (meta.section) parts.push(meta.section);
        if (meta.name && meta.name !== 'Старт') parts.push(meta.name);

        el.innerHTML = parts.map((p, i) => {
            const last = i === parts.length - 1;
            const cls  = last ? 'current' : '';
            const sep  = last ? '' : '<span class="sep">/</span>';
            return `<span class="${cls}">${p}</span>${sep}`;
        }).join('');
        return window.dash_clientside.no_update;
    }
    """,
    Output("kb-breadcrumb", "data-x"),  # dummy output
    Input("url", "pathname"),
)
