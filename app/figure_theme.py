"""
app/figure_theme.py – KIBAD dark theme for Plotly figures.

Designed to feel cohesive with the CSS design system:
- paper_bgcolor matches --bg-card (#111318)
- plot_bgcolor is one shade darker (#0d1016) for subtle inner contrast
- Colorway: blue-first ordering, balanced saturation
- Margins: symmetric and generous for breathing room
- Grid lines: very subtle rgba for a refined look
- Hover: dark tooltip with soft accent border

Also exposes a secondary "EDA" preset (handoff-8 eucalyptus design system)
via ``apply_eda_theme`` / ``PLOTLY_THEME_EDA`` / ``KIBAD_EDA_COLORWAY`` /
``EDA_DIVERGING_COLORSCALE``. The EDA preset is used on the Exploratory
Analysis page (p05_explore) and uses a darker eucalyptus-tinted surface
palette with a green-first colorway. The default ``apply_kibad_theme``
remains unchanged for all other pages.
"""
from __future__ import annotations

import plotly.graph_objects as go

# Refined colorway — blue-first, balanced vibrancy
KIBAD_COLORWAY = [
    '#4f8ef7',  # blue          — primary data series
    '#10b981',  # emerald       — positive / accent
    '#f59e0b',  # amber         — secondary / warning
    '#a78bfa',  # violet        — tertiary
    '#fb7185',  # rose          — negative / alert
    '#22d3ee',  # cyan          — info
    '#f97316',  # orange        — highlight
    '#34d399',  # light emerald — additional positive
]

PLOTLY_THEME = dict(
    # Two-layer depth: card surface vs inner plot area
    paper_bgcolor='#111318',
    plot_bgcolor='#0d1016',

    font=dict(
        family='IBM Plex Sans, -apple-system, sans-serif',
        color='#c8cdd9',
        size=12,
    ),
    colorway=KIBAD_COLORWAY,

    title=dict(
        font=dict(size=14, color='#e4e7ee', family='IBM Plex Sans, sans-serif'),
        x=0.0,
        xanchor='left',
        pad=dict(l=0, t=4),
    ),

    xaxis=dict(
        gridcolor='rgba(255,255,255,0.04)',
        gridwidth=1,
        linecolor='rgba(255,255,255,0.07)',
        zerolinecolor='rgba(255,255,255,0.10)',
        zerolinewidth=1,
        tickfont=dict(size=11, color='#707a94'),
        title_font=dict(size=12, color='#8891a5'),
        showgrid=True,
        ticks='outside',
        ticklen=4,
        tickcolor='rgba(255,255,255,0.0)',
    ),
    yaxis=dict(
        gridcolor='rgba(255,255,255,0.04)',
        gridwidth=1,
        linecolor='rgba(255,255,255,0.07)',
        zerolinecolor='rgba(255,255,255,0.10)',
        zerolinewidth=1,
        tickfont=dict(size=11, color='#707a94'),
        title_font=dict(size=12, color='#8891a5'),
        showgrid=True,
        ticks='outside',
        ticklen=4,
        tickcolor='rgba(255,255,255,0.0)',
    ),

    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(255,255,255,0.0)',
        font=dict(color='#8891a5', size=11),
        itemsizing='constant',
        itemclick='toggleothers',
        orientation='h',
        y=-0.18,
        x=0,
    ),

    hoverlabel=dict(
        bgcolor='#161a24',
        font_color='#e4e7ee',
        font_size=12,
        bordercolor='rgba(79,142,247,0.30)',
        namelength=-1,
    ),

    # Balanced margins: left/bottom slightly larger for axis labels
    margin=dict(l=56, r=36, t=48, b=56),

    # Subtle shape defaults
    newshape=dict(line_color='#4f8ef7'),
)


# ---------------------------------------------------------------------------
# Per-chart-type margin presets
# ---------------------------------------------------------------------------
MARGIN_COMPACT = dict(l=36, r=20, t=36, b=36)     # inside accordion / side-by-side
MARGIN_MINI    = dict(l=8,  r=8,  t=28, b=8)      # 3-per-row distribution grid
MARGIN_FULL    = dict(l=64, r=40, t=52, b=64)     # full-width hero charts
MARGIN_GAUGE   = dict(l=16, r=16, t=32, b=16)     # indicator / gauge

# Standard chart heights
HEIGHT_MINI    = 240    # small grid items (was 200 — too cramped)
HEIGHT_COMPACT = 340    # side panels, secondary charts
HEIGHT_DEFAULT = 420    # standard chart
HEIGHT_TALL    = 520    # main hero chart on a page
HEIGHT_HEATMAP = 480    # cohort / correlation heatmaps


def apply_kibad_theme(fig: go.Figure, preset: str = "default") -> go.Figure:
    """Apply the KIBAD dark theme to a Plotly figure.

    Args:
        fig: Plotly figure to theme.
        preset: One of 'default', 'compact', 'mini', 'full', 'gauge'.
                Controls margin and height defaults.

    Mutates the figure in-place and returns it for chaining.
    """
    fig.update_layout(**PLOTLY_THEME)

    # Apply margin preset on top of base theme (can still be overridden per-chart)
    margin_map = {
        'compact': MARGIN_COMPACT,
        'mini':    MARGIN_MINI,
        'full':    MARGIN_FULL,
        'gauge':   MARGIN_GAUGE,
        'default': None,
    }
    margin = margin_map.get(preset)
    if margin:
        fig.update_layout(margin=margin)

    return fig


# ---------------------------------------------------------------------------
# EDA preset — handoff-8 eucalyptus design system
# ---------------------------------------------------------------------------
# Used by the Exploratory Analysis page (p05_explore). Mirrors the CSS
# design tokens of the eucalyptus dark theme:
#   --surface-0      #0F1613   (plot background)
#   --surface-1      #141C18   (paper background)
#   --surface-2      #1A2420   (hover tooltip)
#   --text-primary   #E8EFEA
#   --text-secondary #A3B0A8
#   --text-tertiary  #6B7A72
#   --viz-1          #21A066   (eucalyptus green, primary)

KIBAD_EDA_COLORWAY = [
    '#21A066',  # viz-1 - eucalyptus green (primary)
    '#4A7FB0',  # viz-2 - blue
    '#C98A2E',  # viz-3 - amber
    '#A066C8',  # viz-4 - violet
    '#C8503B',  # viz-5 - red
    '#6B8E8A',  # viz-6 - sage
]

PLOTLY_THEME_EDA = dict(
    # Eucalyptus surfaces: paper = surface-1, plot = surface-0
    paper_bgcolor='#141C18',
    plot_bgcolor='#0F1613',

    font=dict(
        family='Inter, -apple-system, sans-serif',
        color='#E8EFEA',
        size=12,
    ),
    colorway=KIBAD_EDA_COLORWAY,

    title=dict(
        font=dict(size=14, color='#E8EFEA', family='Inter, sans-serif'),
        x=0.0,
        xanchor='left',
        pad=dict(l=0, t=4),
    ),

    xaxis=dict(
        gridcolor='rgba(255,255,255,0.06)',
        gridwidth=1,
        linecolor='rgba(255,255,255,0.09)',
        zerolinecolor='rgba(255,255,255,0.09)',
        zerolinewidth=1,
        tickfont=dict(size=11, color='#6B7A72'),
        title_font=dict(size=12, color='#A3B0A8'),
        showgrid=True,
        ticks='outside',
        ticklen=4,
        tickcolor='rgba(255,255,255,0.0)',
    ),
    yaxis=dict(
        gridcolor='rgba(255,255,255,0.06)',
        gridwidth=1,
        linecolor='rgba(255,255,255,0.09)',
        zerolinecolor='rgba(255,255,255,0.09)',
        zerolinewidth=1,
        tickfont=dict(size=11, color='#6B7A72'),
        title_font=dict(size=12, color='#A3B0A8'),
        showgrid=True,
        ticks='outside',
        ticklen=4,
        tickcolor='rgba(255,255,255,0.0)',
    ),

    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(255,255,255,0.0)',
        font=dict(color='#A3B0A8', size=11),
        itemsizing='constant',
        itemclick='toggleothers',
        orientation='h',
        y=-0.18,
        x=0,
    ),

    hoverlabel=dict(
        bgcolor='#1A2420',
        font_color='#E8EFEA',
        font_size=12,
        bordercolor='rgba(33,160,102,0.30)',
        namelength=-1,
    ),

    margin=dict(l=56, r=36, t=48, b=56),

    newshape=dict(line_color='#21A066'),
)


# Diverging colorscale for heatmaps (e.g. correlation matrices) on the EDA
# page: viz-5 red for negatives, surface-0 for zero, viz-1 green for positives.
EDA_DIVERGING_COLORSCALE = [
    [0.0, '#C8503B'],   # viz-5 (negative)
    [0.5, '#0F1613'],   # surface-0 (zero)
    [1.0, '#21A066'],   # viz-1 (positive)
]


def apply_eda_theme(fig: go.Figure, preset: str = "default") -> go.Figure:
    """Apply the KIBAD EDA (handoff-8) eucalyptus theme to a Plotly figure.

    Use on the Exploratory Analysis page (p05_explore) to match the dark
    eucalyptus design system. Mutates and returns the figure for chaining.
    """
    fig.update_layout(**PLOTLY_THEME_EDA)
    margin_map = {
        'compact': MARGIN_COMPACT,
        'mini':    MARGIN_MINI,
        'full':    MARGIN_FULL,
        'gauge':   MARGIN_GAUGE,
        'default': None,
    }
    margin = margin_map.get(preset)
    if margin:
        fig.update_layout(margin=margin)
    return fig
