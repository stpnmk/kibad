"""
app/figure_theme.py – KIBAD dark theme for Plotly figures.

Designed to feel cohesive with the CSS design system:
- Background matches --bg-card (#111318)
- Grid lines are near-invisible (#1e2232)
- Colorway uses balanced, distinguishable hues
- Generous margins for clean chart appearance
"""
from __future__ import annotations

import plotly.graph_objects as go

PLOTLY_THEME = dict(
    paper_bgcolor='#111318',
    plot_bgcolor='#111318',
    font=dict(family='IBM Plex Sans, -apple-system, sans-serif', color='#e4e7ee', size=12),
    colorway=[
        '#10b981',  # emerald (accent)
        '#3b82f6',  # blue
        '#f59e0b',  # amber
        '#ef4444',  # red
        '#8b5cf6',  # violet
        '#06b6d4',  # cyan
        '#f97316',  # orange
        '#ec4899',  # pink
    ],
    xaxis=dict(
        gridcolor='#1e2232',
        gridwidth=1,
        linecolor='#252a3a',
        zerolinecolor='#252a3a',
        zerolinewidth=1,
        tickfont=dict(size=11, color='#8891a5'),
        title_font=dict(size=12, color='#8891a5'),
    ),
    yaxis=dict(
        gridcolor='#1e2232',
        gridwidth=1,
        linecolor='#252a3a',
        zerolinecolor='#252a3a',
        zerolinewidth=1,
        tickfont=dict(size=11, color='#8891a5'),
        title_font=dict(size=12, color='#8891a5'),
    ),
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8891a5', size=11),
        itemsizing='constant',
    ),
    hoverlabel=dict(
        bgcolor='#191c24',
        font_color='#e4e7ee',
        font_size=12,
        bordercolor='#252a3a',
    ),
    margin=dict(l=48, r=24, t=40, b=40),
)


def apply_kibad_theme(fig: go.Figure) -> go.Figure:
    """Apply the KIBAD dark theme to a Plotly figure.

    Mutates the figure in-place and returns it for chaining.
    """
    fig.update_layout(**PLOTLY_THEME)
    return fig
