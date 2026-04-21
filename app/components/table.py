"""Styled DataTable component — dark eucalyptus palette, JetBrains Mono cells."""
from __future__ import annotations

import pandas as pd
from dash import dash_table, html


# Colour constants (mirror the tokens from theme.css — dash_table can't read CSS vars)
_SURFACE_0 = "#0F1613"
_SURFACE_1 = "#141C18"
_SURFACE_2 = "#1A2420"
_BORDER_SUBTLE = "rgba(255,255,255,0.06)"
_BORDER_DEFAULT = "rgba(255,255,255,0.09)"
_BORDER_STRONG = "rgba(255,255,255,0.14)"
_TEXT_PRIMARY = "#E8EFEA"
_TEXT_SECONDARY = "#A3B0A8"
_TEXT_TERTIARY = "#6B7A72"
_ACCENT = "#21A066"

_FONT_UI = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
_FONT_MONO = "'JetBrains Mono', 'SF Mono', ui-monospace, 'Consolas', monospace"


def data_table(
    df: pd.DataFrame,
    id: str,
    page_size: int = 20,
    max_rows: int | None = None,
    filter_action: str = "native",
    sort_action: str = "native",
) -> html.Div:
    """Styled ``dash_table.DataTable`` matching the new KIBAD dark theme.

    Cells use JetBrains Mono for tabular alignment; numeric columns are
    right-aligned automatically. Sort/filter/pagination all on by default.
    """
    if df is None or df.empty:
        return html.Div("Нет данных.", className="kb-table-empty")

    display_df = df.head(max_rows) if max_rows else df

    # Build column definitions with numeric alignment
    columns = []
    for c in display_df.columns:
        col = {"name": str(c), "id": str(c)}
        if pd.api.types.is_numeric_dtype(display_df[c]):
            col["type"] = "numeric"
        columns.append(col)

    return html.Div(
        dash_table.DataTable(
            id=id,
            columns=columns,
            data=display_df.to_dict("records"),
            page_size=page_size,
            style_table={
                "overflowX": "auto",
                "border": "none",
            },
            style_header={
                "backgroundColor": _SURFACE_2,
                "color": _TEXT_TERTIARY,
                "fontFamily": _FONT_UI,
                "fontWeight": "600",
                "fontSize": "10px",
                "letterSpacing": "0.1em",
                "textTransform": "uppercase",
                "border": "none",
                "borderBottom": f"1px solid {_BORDER_DEFAULT}",
                "padding": "10px 12px",
                "height": "36px",
            },
            style_cell={
                "backgroundColor": _SURFACE_1,
                "color": _TEXT_PRIMARY,
                "fontFamily": _FONT_MONO,
                "fontSize": "12px",
                "fontVariantNumeric": "tabular-nums",
                "border": "none",
                "borderBottom": f"1px solid {_BORDER_SUBTLE}",
                "padding": "8px 12px",
                "textAlign": "left",
                "maxWidth": "280px",
                "minWidth": "60px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "whiteSpace": "nowrap",
            },
            style_cell_conditional=[
                {
                    "if": {"column_type": "numeric"},
                    "textAlign": "right",
                },
            ],
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": _SURFACE_0,
                },
                {
                    "if": {"state": "active"},
                    "backgroundColor": _SURFACE_2,
                    "border": f"1px solid {_ACCENT}",
                },
                {
                    "if": {"state": "selected"},
                    "backgroundColor": _SURFACE_2,
                },
            ],
            style_filter={
                "backgroundColor": _SURFACE_2,
                "color": _TEXT_PRIMARY,
                "border": "none",
                "borderBottom": f"1px solid {_BORDER_DEFAULT}",
                "height": "32px",
                "padding": "0 12px",
            },
            style_filter_conditional=[
                {
                    "if": {"column_type": "numeric"},
                    "textAlign": "right",
                },
            ],
            filter_action=filter_action,
            sort_action=sort_action,
            sort_mode="multi",
            page_action="native",
            css=[
                # Filter input — make it look like a normal input with placeholder
                {"selector": ".dash-filter input", "rule": (
                    f"background: {_SURFACE_1} !important;"
                    f"color: {_TEXT_PRIMARY} !important;"
                    f"border: 1px solid {_BORDER_DEFAULT} !important;"
                    "border-radius: 4px !important;"
                    "padding: 3px 8px !important;"
                    "height: 24px !important;"
                    f"font-family: {_FONT_MONO} !important;"
                    "font-size: 11px !important;"
                )},
                {"selector": ".dash-filter input::placeholder", "rule": (
                    f"color: {_TEXT_TERTIARY} !important;"
                    "font-style: normal !important;"
                )},
                {"selector": ".dash-filter", "rule": "padding: 4px 8px !important;"},
                # Pagination area
                {"selector": ".previous-next-container", "rule": (
                    f"color: {_TEXT_SECONDARY} !important;"
                    f"font-family: {_FONT_UI} !important;"
                    "font-size: 12px !important;"
                    "padding: 10px 12px !important;"
                    f"background: {_SURFACE_1} !important;"
                    f"border-top: 1px solid {_BORDER_SUBTLE} !important;"
                )},
                {"selector": ".page-number", "rule": (
                    f"color: {_TEXT_SECONDARY} !important;"
                    f"font-family: {_FONT_MONO} !important;"
                )},
                {"selector": ".previous-page, .next-page, .first-page, .last-page", "rule": (
                    f"color: {_TEXT_SECONDARY} !important;"
                    "cursor: pointer !important;"
                )},
                {"selector": ".previous-page:hover, .next-page:hover", "rule": (
                    f"color: {_ACCENT} !important;"
                )},
                # Hover row
                {"selector": ".dash-cell:hover", "rule": "background-color: transparent !important;"},
                {"selector": "tr:hover td", "rule": (
                    f"background-color: {_SURFACE_2} !important;"
                )},
                # Outer container — remove double scrollbars
                {"selector": ".dash-spreadsheet-container", "rule": (
                    "border: none !important;"
                    f"background: {_SURFACE_1} !important;"
                    "border-radius: 8px !important;"
                    "overflow: hidden !important;"
                )},
            ],
        ),
        className="kb-table",
    )
