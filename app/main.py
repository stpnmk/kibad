"""
app/main.py – KIBAD Analytics Studio (Dash).

Run with:
    python app/main.py
"""
import glob
import logging
import sys
import time
from pathlib import Path

# Make project root importable when running from any directory
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Logging — configure once here; all modules use getLogger(__name__)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kibad")


# ---------------------------------------------------------------------------
# Session file cleanup — remove Parquet files older than 24 h on startup
# ---------------------------------------------------------------------------
def _cleanup_session_files(max_age_hours: int = 24) -> None:
    session_dir = ROOT / "data" / "session"
    if not session_dir.exists():
        return
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for path in glob.glob(str(session_dir / "*.parquet")):
        try:
            if Path(path).stat().st_mtime < cutoff:
                Path(path).unlink()
                removed += 1
        except OSError:
            pass
    if removed:
        logger.info("Cleaned up %d stale session file(s) from data/session/", removed)


_cleanup_session_files()

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
            id="main-content",
        ),
    ], className="app-shell"),

], fluid=True, style={"padding": "0", "maxWidth": "100%", "minHeight": "100vh", "background": "var(--bg-base)"})


from dash import Input, Output, State, clientside_callback

# Toggle sidebar collapsed state on button click
clientside_callback(
    "function(n, current) { return n > 0 ? !current : (current || false); }",
    Output(STORE_SIDEBAR, "data"),
    Input("sidebar-toggle", "n_clicks"),
    State(STORE_SIDEBAR, "data"),
    prevent_initial_call=True,
)

# Apply/remove collapsed class on sidebar and adjust main margin
clientside_callback(
    """
    function(collapsed) {
        var sidebar = document.getElementById('sidebar');
        var main = document.getElementById('main-content');
        var toggle = document.getElementById('sidebar-toggle');
        if (!sidebar) return window.dash_clientside.no_update;
        if (collapsed) {
            sidebar.classList.add('collapsed');
            if (main) { main.style.marginLeft = '56px'; main.style.width = 'calc(100% - 56px)'; }
            if (toggle) toggle.textContent = '›';
        } else {
            sidebar.classList.remove('collapsed');
            if (main) { main.style.marginLeft = '216px'; main.style.width = 'calc(100% - 216px)'; }
            if (toggle) toggle.textContent = '‹';
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("sidebar-toggle", "title"),
    Input(STORE_SIDEBAR, "data"),
)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8501)
