"""Form control components."""
from __future__ import annotations

from dash import html, dcc


def select_input(
    label: str,
    id: str,
    options: list,
    value=None,
    multi: bool = False,
    placeholder: str = "",
    clearable: bool = True,
) -> html.Div:
    """Styled ``dcc.Dropdown`` with label.

    Parameters
    ----------
    label : str
        Label text above the dropdown.
    id : str
        Component ID.
    options : list
        Options for the dropdown (strings or dicts with label/value).
    value : optional
        Default selected value.
    multi : bool
        Allow multiple selections.
    placeholder : str
        Placeholder text.
    clearable : bool
        Whether the value can be cleared.
    """
    if options and isinstance(options[0], str):
        opts = [{"label": o, "value": o} for o in options]
    else:
        opts = options

    return html.Div([
        html.Label(label, className="kb-stat-label", style={"marginBottom": "6px"}),
        dcc.Dropdown(
            id=id,
            options=opts,
            value=value,
            multi=multi,
            placeholder=placeholder or f"Select {label.lower()}...",
            clearable=clearable,
            className="kb-select",
        ),
    ], style={"marginBottom": "12px"})


def number_input(label: str, id: str, value=None, min_val=None, max_val=None, step=None) -> html.Div:
    """Styled number input with label."""
    return html.Div([
        html.Label(label, className="kb-stat-label", style={"marginBottom": "6px"}),
        dcc.Input(
            id=id,
            type="number",
            value=value,
            min=min_val,
            max=max_val,
            step=step,
            style={"width": "100%"},
        ),
    ], style={"marginBottom": "12px"})


def text_input(label: str, id: str, value: str = "", placeholder: str = "",
               input_type: str = "text") -> html.Div:
    """Styled text input with label."""
    return html.Div([
        html.Label(label, className="kb-stat-label", style={"marginBottom": "6px"}),
        dcc.Input(
            id=id,
            type=input_type,
            value=value,
            placeholder=placeholder,
            style={"width": "100%"},
        ),
    ], style={"marginBottom": "12px"})


def slider_input(label: str, id: str, min_val=0, max_val=100, value=50, step=1,
                 marks: dict | None = None) -> html.Div:
    """Styled slider with label."""
    return html.Div([
        html.Label(label, className="kb-stat-label", style={"marginBottom": "6px"}),
        dcc.Slider(
            id=id,
            min=min_val,
            max=max_val,
            value=value,
            step=step,
            marks=marks,
        ),
    ], style={"marginBottom": "16px"})


def checklist_input(label: str, id: str, options: list, value: list | None = None) -> html.Div:
    """Styled checklist with label."""
    if options and isinstance(options[0], str):
        opts = [{"label": o, "value": o} for o in options]
    else:
        opts = options

    return html.Div([
        html.Label(label, className="kb-stat-label", style={"marginBottom": "6px"}),
        dcc.Checklist(
            id=id,
            options=opts,
            value=value or [],
            labelStyle={"display": "block", "marginBottom": "4px",
                         "color": "#8b92a8", "fontSize": "0.85rem"},
        ),
    ], style={"marginBottom": "12px"})
