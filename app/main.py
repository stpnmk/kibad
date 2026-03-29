"""
app/main.py – KIBAD Analytics Studio (Dash).

Run with:
    python app/main.py
"""
import logging
import sys
from pathlib import Path

# Make project root importable when running from any directory
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kibad")

import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

from app.components.nav import sidebar_nav
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

    # App shell: sidebar + page content
    html.Div([
        sidebar_nav(dash.page_registry.values()),
        html.Div(
            dash.page_container,
            className="kb-main",
        ),
    ], className="app-shell"),

    # Context menu container (shown/hidden via JS callbacks)
    html.Div(id="context-menu", className="kb-context-menu", children=[]),

], fluid=True, className="app-shell", style={"padding": "0", "maxWidth": "100%"})


def _cleanup_old_session_files(max_age_hours: int = 24) -> None:
    """Remove session Parquet files older than *max_age_hours*."""
    import glob
    import os
    import time
    cutoff = time.time() - max_age_hours * 3600
    pattern = str(ROOT / "data" / "session" / "*.parquet")
    for path in glob.glob(pattern):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                logger.info("Removed stale session file: %s", path)
        except OSError as exc:
            logger.warning("Could not remove session file %s: %s", path, exc)


_cleanup_old_session_files()


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8501))
    app.run(debug=False, host="0.0.0.0", port=port)
