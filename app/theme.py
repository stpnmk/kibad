"""
app/theme.py – Design tokens and theme constants for KIBAD.
Single source of truth for all colors, spacing, and typography.
"""

# ---------------------------------------------------------------------------
# Brand Colors
# ---------------------------------------------------------------------------
PRIMARY = "#1F3864"          # Dark navy – primary brand color
PRIMARY_LIGHT = "#2a4d8a"   # Lighter navy for hover states
ACCENT = "#2196F3"           # Blue accent for interactive elements

# ---------------------------------------------------------------------------
# RAG (Traffic Light) System – single canonical palette
# ---------------------------------------------------------------------------
RAG_GREEN = "#198754"
RAG_YELLOW = "#ffc107"
RAG_RED = "#dc3545"
RAG_GRAY = "#6c757d"

RAG_GREEN_BG = "#d1e7dd"
RAG_YELLOW_BG = "#fff3cd"
RAG_RED_BG = "#f8d7da"
RAG_GRAY_BG = "#e9ecef"

RAG_GREEN_TEXT = "#0f5132"
RAG_YELLOW_TEXT = "#664d03"
RAG_RED_TEXT = "#842029"
RAG_GRAY_TEXT = "#41464b"

# ---------------------------------------------------------------------------
# Neutral Palette
# ---------------------------------------------------------------------------
BORDER_LIGHT = "#dee2e6"
BORDER_MEDIUM = "#ced4da"
BG_PAGE = "#f8f9fa"
BG_CARD = "#ffffff"
BG_HOVER = "#f0f4ff"

TEXT_PRIMARY = "#212529"
TEXT_SECONDARY = "#495057"
TEXT_MUTED = "#6c757d"
TEXT_DISABLED = "#adb5bd"

# ---------------------------------------------------------------------------
# Status Colors
# ---------------------------------------------------------------------------
INFO_BG = "#cff4fc"
INFO_TEXT = "#055160"
INFO_BORDER = "#9eeaf9"

SUCCESS_BG = "#d1e7dd"
SUCCESS_TEXT = "#0f5132"
SUCCESS_BORDER = "#a3cfbb"

WARNING_BG = "#fff3cd"
WARNING_TEXT = "#664d03"
WARNING_BORDER = "#ffe69c"

DANGER_BG = "#f8d7da"
DANGER_TEXT = "#842029"
DANGER_BORDER = "#f1aeb5"

# ---------------------------------------------------------------------------
# Typography Scale (rem values)
# ---------------------------------------------------------------------------
FONT_XS = "0.75rem"     # 12px – captions, labels
FONT_SM = "0.875rem"    # 14px – secondary text
FONT_BASE = "1rem"      # 16px – body text
FONT_LG = "1.125rem"    # 18px – lead text
FONT_XL = "1.375rem"    # 22px – subheadings
FONT_2XL = "1.75rem"    # 28px – section headings
FONT_3XL = "2.25rem"    # 36px – page titles
FONT_HERO = "2.75rem"   # 44px – hero/landing

LINE_HEIGHT_TIGHT = "1.2"
LINE_HEIGHT_NORMAL = "1.5"
LINE_HEIGHT_RELAXED = "1.75"

# ---------------------------------------------------------------------------
# Spacing Scale (px values as strings)
# ---------------------------------------------------------------------------
SPACE_XS = "4px"
SPACE_SM = "8px"
SPACE_MD = "16px"
SPACE_LG = "24px"
SPACE_XL = "32px"
SPACE_2XL = "48px"

# ---------------------------------------------------------------------------
# Border Radius
# ---------------------------------------------------------------------------
RADIUS_SM = "4px"
RADIUS_MD = "8px"
RADIUS_LG = "12px"
RADIUS_XL = "16px"
RADIUS_PILL = "50px"

# ---------------------------------------------------------------------------
# Shadows
# ---------------------------------------------------------------------------
SHADOW_SM = "0 1px 3px rgba(0,0,0,.08)"
SHADOW_MD = "0 2px 8px rgba(0,0,0,.12)"
SHADOW_LG = "0 4px 16px rgba(0,0,0,.15)"

# ---------------------------------------------------------------------------
# Gradient definitions
# ---------------------------------------------------------------------------
GRADIENT_PRIMARY = "linear-gradient(135deg, #1F3864 0%, #2563eb 100%)"
GRADIENT_SUCCESS = "linear-gradient(135deg, #059669 0%, #10b981 100%)"
GRADIENT_WARNING = "linear-gradient(135deg, #d97706 0%, #fbbf24 100%)"
GRADIENT_DANGER = "linear-gradient(135deg, #dc2626 0%, #f87171 100%)"
GRADIENT_PURPLE = "linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%)"

# ---------------------------------------------------------------------------
# Extended palette
# ---------------------------------------------------------------------------
BLUE_50 = "#eff6ff"
BLUE_100 = "#dbeafe"
BLUE_500 = "#3b82f6"
BLUE_600 = "#2563eb"
BLUE_700 = "#1d4ed8"
SLATE_50 = "#f8fafc"
SLATE_100 = "#f1f5f9"
SLATE_200 = "#e2e8f0"
SLATE_400 = "#94a3b8"
SLATE_600 = "#475569"
SLATE_800 = "#1e293b"

# ---------------------------------------------------------------------------
# Component-level helpers
# ---------------------------------------------------------------------------
def rag_color(value: float, thresholds: dict) -> str:
    """Return RAG border color based on value vs thresholds.
    thresholds = {"green": (lo, hi), "yellow": (lo, hi)}  – green wins if met.
    Returns one of RAG_GREEN, RAG_YELLOW, RAG_RED.
    """
    g = thresholds.get("green")
    y = thresholds.get("yellow")
    if g and g[0] <= value <= g[1]:
        return RAG_GREEN
    if y and y[0] <= value <= y[1]:
        return RAG_YELLOW
    return RAG_RED


def rag_badge(label: str, color: str) -> str:
    """Return inline HTML badge with RAG color."""
    bg_map = {RAG_GREEN: RAG_GREEN_BG, RAG_YELLOW: RAG_YELLOW_BG, RAG_RED: RAG_RED_BG, RAG_GRAY: RAG_GRAY_BG}
    text_map = {RAG_GREEN: RAG_GREEN_TEXT, RAG_YELLOW: RAG_YELLOW_TEXT, RAG_RED: RAG_RED_TEXT, RAG_GRAY: RAG_GRAY_TEXT}
    bg = bg_map.get(color, RAG_GRAY_BG)
    text = text_map.get(color, RAG_GRAY_TEXT)
    return (
        f"<span style='background:{bg};color:{text};padding:2px 10px;"
        f"border-radius:50px;font-size:0.8rem;font-weight:600;"
        f"border:1px solid {color}'>{label}</span>"
    )
