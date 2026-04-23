"""Inline SVG icon helper — Lucide-style, stroke 1.5, inherits ``currentColor``.

Usage::

    from app.components.icons import icon
    icon("search")              # 16px default
    icon("chevron-down", 14)    # explicit size
    icon("bell", className="kb-topbar-icon")

Dash's ``html`` module does not expose SVG primitives, so we inline real SVG
markup via ``dcc.Markdown(..., dangerously_allow_html=True)``. The ``<p>``
wrapper that Markdown adds is neutralised by the ``.kb-icon`` CSS block
(``display: inline-flex``, ``margin: 0``, ``line-height: 0``).
"""
from __future__ import annotations

from urllib.parse import quote

from dash import html

# All paths live in the 24×24 viewbox, designed for 1.5 stroke.
_PATHS: dict[str, str] = {
    "search":        '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.35-4.35"/>',
    "bell":          '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>',
    "chevron-down":  '<path d="m6 9 6 6 6-6"/>',
    "chevron-right": '<path d="m9 18 6-6-6-6"/>',
    "chevron-up":    '<path d="m18 15-6-6-6 6"/>',
    "chevron-left":  '<path d="m15 18-6-6 6-6"/>',
    "upload":        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
    "plus":          '<path d="M5 12h14"/><path d="M12 5v14"/>',
    "info":          '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "alert":         '<path d="m21.73 18-8-14a2 2 0 0 0-3.46 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "check":         '<path d="M20 6 9 17l-5-5"/>',
    "x":             '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "close":         '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "sort":          '<path d="m3 8 4-4 4 4"/><path d="M7 4v16"/>',
    "arrow-up":      '<path d="m5 12 7-7 7 7"/><path d="M12 19V5"/>',
    "arrow-down":    '<path d="M12 5v14"/><path d="m19 12-7 7-7-7"/>',
    "arrow-right":   '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    "filter":        '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "database":      '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "table":         '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/>',
    "columns":       '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M12 3v18"/>',
    "link":          '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
    "chart":         '<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>',
    "bar-chart":     '<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>',
    "grid":          '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>',
    "settings":      '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
    "trend":         '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "dot":           '<circle cx="12" cy="12" r="2"/>',
    "play":          '<polygon points="5 3 19 12 5 21 5 3"/>',
    "help":          '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    "home":          '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "download":      '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "refresh":       '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10"/><path d="M20.49 15a9 9 0 0 1-14.85 3.36L1 14"/>',
    "wrench":        '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    "layers":        '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "users":         '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "funnel":        '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "target":        '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "arrows-lr":     '<polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/>',
    "tag":           '<path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/>',
    "file-text":     '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>',
    "dice":          '<rect x="2" y="2" width="20" height="20" rx="2" ry="2"/><path d="M16 8h.01"/><path d="M8 8h.01"/><path d="M8 16h.01"/><path d="M16 16h.01"/><path d="M12 12h.01"/>',
    "robot":         '<rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/>',
    "percent":       '<line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
    "pie-chart":     '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/>',
    "command":       '<path d="M18 3a3 3 0 0 0-3 3v12a3 3 0 0 0 3 3 3 3 0 0 0 3-3 3 3 0 0 0-3-3H6a3 3 0 0 0-3 3 3 3 0 0 0 3 3 3 3 0 0 0 3-3V6a3 3 0 0 0-3-3 3 3 0 0 0-3 3 3 3 0 0 0 3 3h12a3 3 0 0 0 3-3 3 3 0 0 0-3-3z"/>',
    # Verdict-card & tests-page glyphs
    "check-circle":  '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "x-circle":      '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
    "clipboard":     '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>',
    "zap":           '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "sigma":         '<path d="M18 7V4H6l7 8-7 8h12v-3"/>',
    "history":       '<path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/><path d="M12 7v5l4 2"/>',
    "rewind":        '<polygon points="11 19 2 12 11 5 11 19"/><polygon points="22 19 13 12 22 5 22 19"/>',
    "calendar":      '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    "git-compare":   '<circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M13 6h3a2 2 0 0 1 2 2v7"/><path d="M11 18H8a2 2 0 0 1-2-2V9"/>',
}


def icon(name: str, size: int = 16, stroke: float = 1.5, className: str = "") -> html.Span:
    """Render a Lucide icon as a CSS-masked box that inherits ``currentColor``.

    Implementation note: Dash's html module doesn't expose SVG tags and
    ``dcc.Markdown`` strips the ``<svg>`` wrapper as non-whitelisted HTML. The
    reliable cross-Dash-version technique is a ``<span>`` whose ``mask-image``
    is an inline SVG data URI — ``background-color: currentColor`` then tints
    the icon to the surrounding text color exactly like a real inline SVG.

    Parameters
    ----------
    name : str
        Icon key (see ``_PATHS``). Unknown keys render an empty spacer.
    size : int
        Pixel width/height of the rendered box.
    stroke : float
        SVG stroke width (default 1.5 to match Lucide default).
    className : str
        Optional extra class on the wrapper ``<span>``.
    """
    # Use single quotes inside the SVG so we can wrap the data URI in double
    # quotes without escaping. All attribute values in ``_PATHS`` are also
    # single-quoted compatible because they contain only numbers and spaces.
    path = _PATHS.get(name, "").replace('"', "'")
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
        f"stroke='black' stroke-width='{stroke}' stroke-linecap='round' "
        f"stroke-linejoin='round'>{path}</svg>"
    )
    data_uri = f'url("data:image/svg+xml;utf8,{quote(svg, safe=":/=")}")'
    return html.Span(
        className=f"kb-icon {className}".strip(),
        style={
            "display": "inline-block",
            "width": f"{size}px",
            "height": f"{size}px",
            "backgroundColor": "currentColor",
            "maskImage": data_uri,
            "WebkitMaskImage": data_uri,
            "maskRepeat": "no-repeat",
            "WebkitMaskRepeat": "no-repeat",
            "maskPosition": "center",
            "WebkitMaskPosition": "center",
            "maskSize": "contain",
            "WebkitMaskSize": "contain",
            "flexShrink": 0,
        },
    )
