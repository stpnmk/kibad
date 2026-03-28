"""Alert banner components."""
from __future__ import annotations

from dash import html


def alert_banner(msg: str, level: str = "info") -> html.Div:
    """Error/warning/info/success banner strip.

    Parameters
    ----------
    msg : str
        Message text.
    level : str
        One of ``"info"``, ``"success"``, ``"warning"``, ``"danger"``.
    """
    return html.Div(msg, className=f"kb-alert kb-alert--{level}")
