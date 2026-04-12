"""Styled DataTable component."""
from __future__ import annotations

import pandas as pd
from dash import dash_table, html


def data_table(
    df: pd.DataFrame,
    id: str,
    page_size: int = 20,
    max_rows: int | None = None,
) -> html.Div:
    """Styled ``dash_table.DataTable`` with dark theme.

    Parameters
    ----------
    df : pd.DataFrame
        Data to display.
    id : str
        Component ID for callbacks.
    page_size : int
        Rows per page.
    max_rows : int, optional
        If set, truncate the DataFrame to this many rows.
    """
    if df is None or df.empty:
        return html.Div("No data.", className="kb-text-muted kb-text-center kb-p-3")

    display_df = df.head(max_rows) if max_rows else df

    columns = [{"name": str(c), "id": str(c)} for c in display_df.columns]

    return html.Div(
        dash_table.DataTable(
            id=id,
            columns=columns,
            data=display_df.to_dict("records"),
            page_size=page_size,
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#191c24",
                "color": "#6b7394",
                "fontWeight": "700",
                "fontSize": "0.72rem",
                "textTransform": "uppercase",
                "letterSpacing": "0.06em",
                "border": "none",
                "borderBottom": "1px solid #252a3a",
            },
            style_cell={
                "backgroundColor": "#111318",
                "color": "#e4e7ee",
                "border": "none",
                "borderBottom": "1px solid #1e2232",
                "fontFamily": "'IBM Plex Mono', 'SF Mono', Consolas, monospace",
                "fontSize": "0.8rem",
                "padding": "8px 12px",
                "textAlign": "left",
                "maxWidth": "300px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            },
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": "#13161c",
                },
                {
                    "if": {"state": "active"},
                    "backgroundColor": "#191c24",
                    "border": "1px solid #10b981",
                },
            ],
            style_filter={
                "backgroundColor": "#191c24",
                "color": "#e4e7ee",
            },
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            page_action="native",
        ),
        className="kb-table",
    )
