"""File upload zone component (dashed drop-target with icon tile + hint)."""
from __future__ import annotations

from dash import dcc, html

from app.components.icons import icon


def upload_zone(
    id: str,
    label: str = "Перетащите файл или нажмите для выбора",
    hint: str = "CSV · Excel (.xlsx) · Parquet · до 2 ГБ",
    multiple: bool = True,
) -> dcc.Upload:
    """Drag-and-drop ``dcc.Upload`` styled as a dashed panel with a 44-px icon tile.

    Parameters
    ----------
    id : str
        Component ID for the ``dcc.Upload``.
    label : str
        Main instruction text.
    hint : str
        Supported-file-types hint shown below the label.
    multiple : bool
        Allow multiple-file upload.
    """
    return dcc.Upload(
        id=id,
        children=html.Div(
            [
                html.Span(
                    "Auto-detect",
                    className="kb-chip kb-chip--neutral kb-upload-zone-chip",
                ),
                html.Div(
                    icon("upload", 20),
                    className="kb-upload-zone-icon-tile",
                ),
                html.Div(label, className="kb-upload-zone-text"),
                html.Div(hint, className="kb-upload-zone-hint"),
            ],
            className="kb-upload-zone-inner",
        ),
        className="kb-upload-zone",
        multiple=multiple,
    )
