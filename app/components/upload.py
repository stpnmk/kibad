"""File upload zone component."""
from __future__ import annotations

from dash import dcc, html


def upload_zone(
    id: str,
    label: str = "Перетащите файл или нажмите для загрузки",
    hint: str = "CSV, Excel, Parquet",
    multiple: bool = True,
) -> html.Div:
    """Drag-and-drop ``dcc.Upload`` styled as dashed card.

    Parameters
    ----------
    id : str
        Component ID for the ``dcc.Upload``.
    label : str
        Main instruction text.
    hint : str
        Supported file types hint.
    multiple : bool
        Allow multiple file upload.
    """
    return dcc.Upload(
        id=id,
        children=html.Div([
            html.Div("+", className="kb-upload-zone-icon"),
            html.Div(label, className="kb-upload-zone-text"),
            html.Div(hint, className="kb-upload-zone-hint"),
        ]),
        className="kb-upload-zone",
        multiple=multiple,
    )
