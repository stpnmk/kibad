"""
app/figure_theme.py – KIBAD dark theme for Plotly figures.
"""
from __future__ import annotations

import plotly.graph_objects as go

PLOTLY_THEME = dict(
    paper_bgcolor='#141720',
    plot_bgcolor='#141720',
    font=dict(family='IBM Plex Sans', color='#e8eaf0', size=12),
    colorway=[
        '#00c896', '#4b9eff', '#e8a020', '#e05252',
        '#9b59b6', '#1abc9c', '#f39c12', '#3498db',
    ],
    xaxis=dict(gridcolor='#2a2f42', linecolor='#2a2f42', zerolinecolor='#2a2f42'),
    yaxis=dict(gridcolor='#2a2f42', linecolor='#2a2f42', zerolinecolor='#2a2f42'),
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8b92a8'),
    ),
    hoverlabel=dict(
        bgcolor='#1c2030',
        font_color='#e8eaf0',
        bordercolor='#2a2f42',
    ),
)


def apply_kibad_theme(fig: go.Figure) -> go.Figure:
    """Apply the KIBAD dark theme to a Plotly figure.

    Mutates the figure in-place and returns it for chaining.
    """
    fig.update_layout(**PLOTLY_THEME)
    return fig
