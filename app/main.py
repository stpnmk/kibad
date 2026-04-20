"""
app/main.py – KIBAD Analytics Studio (Dash).

Run with:
    python app/main.py
"""
import sys
from pathlib import Path

# Make project root importable when running from any directory
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

from app.components.nav import sidebar_nav
from app.components.topbar import topbar
from app.state import (
    STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS, STORE_LANG,
    STORE_AUDIT, STORE_FORECAST, STORE_TEST_RESULTS, STORE_ATTRIBUTION,
    STORE_AGG_RESULTS, STORE_REPORT, STORE_SIDEBAR, STORE_DEFAULTS,
)

app = Dash(
    __name__,
    use_pages=True,
    pages_folder=str(ROOT / "app" / "pages"),
    assets_folder=str(ROOT / "app" / "assets"),
    external_stylesheets=[],
    suppress_callback_exceptions=True,
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "KIBAD — Аналитическая студия"
server = app.server  # expose WSGI for gunicorn

app.layout = dbc.Container([
    # Session stores
    dcc.Store(id=STORE_DATASET, storage_type="session", data=STORE_DEFAULTS[STORE_DATASET]),
    dcc.Store(id=STORE_PREPARED, storage_type="session", data=STORE_DEFAULTS[STORE_PREPARED]),
    dcc.Store(id=STORE_ACTIVE_DS, storage_type="session", data=STORE_DEFAULTS[STORE_ACTIVE_DS]),
    dcc.Store(id=STORE_LANG, storage_type="session", data=STORE_DEFAULTS[STORE_LANG]),
    dcc.Store(id=STORE_AUDIT, storage_type="session", data=STORE_DEFAULTS[STORE_AUDIT]),
    dcc.Store(id=STORE_FORECAST, storage_type="session", data=STORE_DEFAULTS[STORE_FORECAST]),
    dcc.Store(id=STORE_TEST_RESULTS, storage_type="session", data=STORE_DEFAULTS[STORE_TEST_RESULTS]),
    dcc.Store(id=STORE_ATTRIBUTION, storage_type="session", data=STORE_DEFAULTS[STORE_ATTRIBUTION]),
    dcc.Store(id=STORE_AGG_RESULTS, storage_type="session", data=STORE_DEFAULTS[STORE_AGG_RESULTS]),
    dcc.Store(id=STORE_REPORT, storage_type="session", data=STORE_DEFAULTS[STORE_REPORT]),
    dcc.Store(id=STORE_SIDEBAR, storage_type="session", data=STORE_DEFAULTS[STORE_SIDEBAR]),

    # Location for URL routing
    dcc.Location(id="url", refresh=False),

    # App shell: sidebar + (topbar + page content)
    html.Div([
        sidebar_nav(dash.page_registry.values()),
        html.Div(
            [
                topbar(dash.page_registry.values()),
                dash.page_container,
            ],
            className="kb-main",
        ),
    ], className="app-shell"),

], fluid=True, className="app-shell", style={"padding": "0", "maxWidth": "100%"})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8501)
