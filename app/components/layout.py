"""Page layout components – headers, sections, wrappers."""
from __future__ import annotations

from dash import html, dcc
import dash_bootstrap_components as dbc


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


def empty_state(
    icon: str = "",
    title: str = "",
    description: str = "",
    action=None,
) -> html.Div:
    """Render a centered empty-state placeholder with optional CTA.

    Parameters
    ----------
    icon : str
        Emoji or character for the icon area.
    title : str
        Main heading of the empty state.
    description : str
        Explanatory subtext.
    action : dash component, optional
        CTA button or link shown below the description.
    """
    children = [
        html.Div(icon, className="kb-empty-state-icon"),
        html.Div(title, className="kb-empty-state-title"),
        html.Div(description, className="kb-empty-state-desc"),
    ]
    if action is not None:
        children.append(html.Div(action, style={"marginTop": "20px"}))
    return html.Div(children, className="kb-empty-state")


def data_guard(store_data, page_name: str = "") -> html.Div | None:
    """Return a CTA banner if no data is loaded, else None.

    Parameters
    ----------
    store_data : dict | None
        Contents of the STORE_DATASET.
    page_name : str
        Name of the current page for the message.
    """
    if store_data:
        return None
    msg = f"Для работы страницы «{page_name}» нужны загруженные данные." if page_name else "Данные не загружены."
    return empty_state(
        icon="🗄",
        title="Данные не загружены",
        description=msg,
        action=dcc.Link(
            dbc.Button("Загрузить данные", color="primary", size="sm"),
            href="/data",
        ),
    )


def data_status_bar(store_data: dict | None) -> html.Div:
    """Show a slim status bar with dataset names.

    Parameters
    ----------
    store_data : dict | None
        Dataset store contents; each key is a dataset name, values are
        file path strings pointing to parquet files.
    """
    if not store_data:
        return html.Div()
    names = list(store_data.keys())
    if not names:
        return html.Div()
    name = names[0]
    extra = f" · всего датасетов: {len(names)}" if len(names) > 1 else ""
    return html.Div([
        html.Span("📊 ", style={"color": "var(--accent)"}),
        html.Span(name, className="kb-ds-filename"),
        html.Span(extra, className="kb-ds-meta") if extra else None,
    ], className="kb-data-status-bar")


def controls_panel(title: str, children) -> html.Div:
    """Wrap form controls in a styled panel."""
    return html.Div([
        html.Div(title, className="kb-controls-title"),
        *children,
    ], className="kb-controls")


def related_pages(links: list[tuple[str, str, str]]) -> html.Div:
    """Render a row of related-page links.

    Parameters
    ----------
    links : list of (href, icon_char, label)
    """
    return html.Div([
        html.Span("Связанные разделы:", className="kb-related-label"),
        *[
            dcc.Link(
                html.Span([icon, f" {label}"]),
                href=href,
                className="kb-related-link",
            )
            for href, icon, label in links
        ],
    ], className="kb-related-pages")
