"""Alert banner components."""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

_ICONS = {
    "info":    "ℹ",
    "success": "✓",
    "warning": "⚠",
    "danger":  "✕",
}


def alert_banner(
    msg: str,
    level: str = "info",
    dismissible: bool = False,
    duration: int | None = None,
) -> html.Div | dbc.Alert:
    """Error/warning/info/success banner strip.

    Parameters
    ----------
    msg : str
        Message text.
    level : str
        One of ``"info"``, ``"success"``, ``"warning"``, ``"danger"``.
    dismissible : bool
        Whether the alert can be closed by the user.
    duration : int, optional
        Auto-hide after N milliseconds (None = stays).
    """
    if dismissible or duration:
        return dbc.Alert(
            [html.Span(_ICONS.get(level, ""), style={"marginRight": "8px"}), msg],
            color=level if level != "danger" else "danger",
            dismissable=dismissible,
            duration=duration,
            className=f"kb-alert kb-alert--{level}",
            style={"borderLeft": f"3px solid var(--{'accent' if level == 'success' else 'danger' if level == 'danger' else 'warning' if level == 'warning' else 'info'})"},  # noqa: E501
        )
    return html.Div(
        [html.Span(_ICONS.get(level, ""), style={"marginRight": "8px", "opacity": "0.8"}), msg],
        className=f"kb-alert kb-alert--{level}",
    )
