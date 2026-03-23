"""
app/styles.py – Global CSS injection for KIBAD.
Import inject_global_css() at the top of every page.
"""
import streamlit as st

# ---------------------------------------------------------------------------
# Global baseline CSS – Premium design system
# ---------------------------------------------------------------------------
_GLOBAL_CSS = """
<style>
/* ── Google Fonts – Inter ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset & Base ── */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
    background-color: #f0f4f8 !important;
}

/* ── Page title styling ── */
h1 {
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    color: #1F3864 !important;
    margin-bottom: 0.25rem !important;
    letter-spacing: -0.02em !important;
}
h2 {
    font-weight: 700 !important;
    color: #1e293b !important;
    letter-spacing: -0.01em !important;
}
h3 {
    font-weight: 600 !important;
    color: #1e293b !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 18px 20px 18px 24px;
    box-shadow: 0 2px 12px rgba(31,56,100,0.08);
    border: none !important;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: linear-gradient(135deg, #1F3864 0%, #2563eb 100%);
    border-radius: 4px 0 0 4px;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 8px 24px rgba(31,56,100,0.15);
    transform: translateY(-1px);
}
[data-testid="stMetricLabel"] {
    font-size: 0.625rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #94a3b8 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    color: #1e293b !important;
    line-height: 1.15 !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    display: inline-block;
    padding: 2px 8px;
    border-radius: 50px;
    background: rgba(5,150,105,0.1);
}

/* ── Tabs ── */
[data-testid="stTabs"] {
    border-bottom: 1px solid #e2e8f0;
}
[data-testid="stTabs"] [role="tablist"] {
    gap: 0 !important;
    border-bottom: none !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-weight: 500;
    font-size: 0.9rem;
    padding: 10px 20px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    color: #475569 !important;
    background: transparent !important;
    transition: color 0.15s, border-color 0.15s;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: #1F3864 !important;
    border-bottom-color: #94a3b8 !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    font-weight: 700 !important;
    color: #1F3864 !important;
    border-bottom: 2px solid #1F3864 !important;
    background: transparent !important;
}

/* ── Buttons ── */
[data-testid="stButton"] button[kind="primary"],
[data-testid="stButton"] button {
    font-weight: 500 !important;
    border-radius: 10px !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.01em !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #1F3864 0%, #2563eb 100%) !important;
    border: none !important;
    color: #ffffff !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    filter: brightness(1.1) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
}
[data-testid="stButton"] button[kind="primary"]:active {
    transform: scale(0.98) !important;
}
[data-testid="stButton"] button:hover {
    box-shadow: 0 2px 8px rgba(31,56,100,0.15) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] button:active {
    transform: scale(0.98) !important;
}

/* ── Alerts/info boxes ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
    border-left: none !important;
    padding: 14px 18px !important;
    position: relative;
    overflow: hidden;
    font-size: 0.92rem !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stAlert"]::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    border-radius: 4px 0 0 4px;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="info"]::before {
    background: linear-gradient(135deg, #0369a1 0%, #38bdf8 100%);
}
[data-testid="stAlert"][data-baseweb="notification"][kind="success"]::before {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
}
[data-testid="stAlert"][data-baseweb="notification"][kind="warning"]::before {
    background: linear-gradient(135deg, #d97706 0%, #fbbf24 100%);
}
[data-testid="stAlert"][data-baseweb="notification"][kind="error"]::before {
    background: linear-gradient(135deg, #dc2626 0%, #f87171 100%);
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(31,56,100,0.07) !important;
    margin-bottom: 10px;
    overflow: hidden;
    background: #ffffff !important;
}
[data-testid="stExpander"] > div:first-child {
    background: #f8faff !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 4px 0 !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: #1F3864 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 8px rgba(31,56,100,0.07) !important;
    border: none !important;
}
[data-testid="stDataFrame"] table thead tr {
    background: #f0f4f8 !important;
}
[data-testid="stDataFrame"] table thead th {
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #475569 !important;
}
[data-testid="stDataFrame"] table tbody tr:nth-child(even) {
    background: #f8fafc !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}

/* ── Selectbox / Multiselect / Input ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    border-radius: 10px !important;
    border-color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stMultiSelect"] > div > div:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
}
input[type="text"], textarea {
    border-radius: 10px !important;
    border-color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
}

/* ── Caption styling ── */
[data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Divider ── */
hr {
    border-color: #94a3b8 !important;
    opacity: 0.3 !important;
    margin: 1.5rem 0 !important;
}

/* ── Progress bars ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(135deg, #1F3864 0%, #2563eb 100%) !important;
    border-radius: 50px !important;
}
[data-testid="stProgress"] > div {
    background: #e2e8f0 !important;
    border-radius: 50px !important;
}

/* ── Page link buttons ── */
[data-testid="stPageLink"] a {
    font-weight: 500;
    border-radius: 8px;
    padding: 6px 10px;
    transition: background 0.15s;
    text-decoration: none;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stPageLink"] a:hover {
    background: #eff6ff;
    text-decoration: none;
}

/* ── Block container ── */
.block-container {
    padding-top: 2rem !important;
}

/* ─────────────────────────────────────────────
   Reusable utility classes
───────────────────────────────────────────── */

/* Base card */
.kibad-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 22px 24px;
    box-shadow: 0 2px 12px rgba(31,56,100,0.08);
    margin-bottom: 14px;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.kibad-card:hover {
    box-shadow: 0 8px 24px rgba(31,56,100,0.15);
    transform: translateY(-1px);
}
.kibad-card-primary {
    border-left: 4px solid #1F3864;
}
.kibad-card-success {
    border-left: 4px solid #059669;
    background: #f0fdf4;
}
.kibad-card-warning {
    border-left: 4px solid #d97706;
    background: #fffbeb;
}
.kibad-card-danger {
    border-left: 4px solid #dc2626;
    background: #fef2f2;
}

/* Badge */
.kibad-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    font-family: 'Inter', sans-serif;
}

/* Section header */
.kibad-section-header {
    font-size: 0.72rem;
    font-weight: 700;
    color: #1F3864;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 28px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
    font-family: 'Inter', sans-serif;
}

/* Step number circle */
.kibad-step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    background: linear-gradient(135deg, #1F3864 0%, #2563eb 100%);
    color: white;
    border-radius: 50%;
    font-size: 0.8rem;
    font-weight: 700;
    flex-shrink: 0;
    font-family: 'Inter', sans-serif;
}
.kibad-step-done {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
}

/* Muted text */
.kibad-muted {
    color: #94a3b8;
    font-size: 0.875rem;
    font-family: 'Inter', sans-serif;
}

/* Label */
.kibad-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #94a3b8;
    margin-bottom: 4px;
    font-family: 'Inter', sans-serif;
}

/* ── NEW UTILITY CLASSES ── */

/* Hero section wrapper */
.kibad-hero {
    background: linear-gradient(135deg, #1F3864 0%, #2563eb 100%);
    border-radius: 16px;
    padding: 40px 48px;
    color: #ffffff;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.kibad-hero::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 400px;
    height: 400px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
    pointer-events: none;
}
.kibad-hero::after {
    content: '';
    position: absolute;
    bottom: -60%;
    right: 15%;
    width: 300px;
    height: 300px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
    pointer-events: none;
}

/* Stat card with top colored accent */
.kibad-stat-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px 22px 16px 22px;
    box-shadow: 0 2px 12px rgba(31,56,100,0.08);
    border-top: 3px solid #2563eb;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.kibad-stat-card:hover {
    box-shadow: 0 8px 24px rgba(31,56,100,0.15);
    transform: translateY(-2px);
}

/* Feature card with icon, title, description */
.kibad-feature-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 24px 22px;
    box-shadow: 0 2px 12px rgba(31,56,100,0.08);
    height: 100%;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    display: flex;
    flex-direction: column;
}
.kibad-feature-card:hover {
    box-shadow: 0 8px 24px rgba(31,56,100,0.15);
    transform: translateY(-2px);
}
.kibad-feature-card-icon {
    width: 44px;
    height: 44px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    margin-bottom: 14px;
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
}
.kibad-feature-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 6px;
    font-family: 'Inter', sans-serif;
}
.kibad-feature-card-desc {
    font-size: 0.83rem;
    color: #64748b;
    line-height: 1.55;
    flex: 1;
    font-family: 'Inter', sans-serif;
}

/* Small pill tag */
.kibad-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 50px;
    font-size: 0.72rem;
    font-weight: 600;
    background: #eff6ff;
    color: #2563eb;
    border: 1px solid #bfdbfe;
    letter-spacing: 0.02em;
    font-family: 'Inter', sans-serif;
}
.kibad-tag-green {
    background: #f0fdf4;
    color: #059669;
    border-color: #bbf7d0;
}
.kibad-tag-amber {
    background: #fffbeb;
    color: #d97706;
    border-color: #fde68a;
}

/* Inline code */
.kibad-code {
    font-family: 'SF Mono', 'Fira Code', 'Fira Mono', 'Roboto Mono', monospace;
    font-size: 0.82rem;
    background: #f1f5f9;
    color: #1F3864;
    padding: 2px 7px;
    border-radius: 5px;
    border: 1px solid #e2e8f0;
}

/* Empty state */
.kibad-empty-state {
    text-align: center;
    padding: 60px 24px;
    color: #94a3b8;
}
.kibad-empty-state-icon {
    font-size: 3.5rem;
    line-height: 1;
    margin-bottom: 16px;
    opacity: 0.6;
}
.kibad-empty-state-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #475569;
    margin-bottom: 8px;
    font-family: 'Inter', sans-serif;
}
.kibad-empty-state-desc {
    font-size: 0.88rem;
    color: #94a3b8;
    font-family: 'Inter', sans-serif;
}

/* Step connector vertical line */
.kibad-step-connector {
    width: 2px;
    height: 32px;
    background: linear-gradient(to bottom, #2563eb, #93c5fd);
    margin: 4px auto;
    border-radius: 2px;
    opacity: 0.5;
}
</style>
"""

_SIDEBAR_CSS = """
<style>
/* ── Sidebar container ── */
[data-testid="stSidebar"] {
    background: #f8faff !important;
    border-right: none !important;
    box-shadow: 4px 0 20px rgba(31,56,100,0.07) !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: #f8faff !important;
}

/* ── Sidebar nav items ── */
[data-testid="stSidebar"] [data-testid="stPageLink"] a {
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-size: 0.88rem !important;
    transition: all 0.15s !important;
    color: #475569 !important;
    display: block;
    position: relative;
}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
    background: #eff6ff !important;
    color: #1F3864 !important;
}
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {
    background: #eff6ff !important;
    color: #1F3864 !important;
    font-weight: 600 !important;
    padding-left: 18px !important;
}
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"]::before {
    content: '';
    position: absolute;
    left: 0;
    top: 6px;
    bottom: 6px;
    width: 6px;
    background: linear-gradient(135deg, #1F3864 0%, #2563eb 100%);
    border-radius: 0 4px 4px 0;
}

/* ── Sidebar headings ── */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.4rem !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Sidebar brand ── */
.kibad-sidebar-brand {
    padding: 16px 0 20px 0;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 16px;
}
.kibad-sidebar-brand-name {
    font-size: 1.15rem;
    font-weight: 800;
    background: linear-gradient(135deg, #1F3864 0%, #2563eb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.01em;
    font-family: 'Inter', sans-serif;
}
.kibad-sidebar-brand-version {
    font-size: 0.68rem;
    color: #94a3b8;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 2px;
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar section labels ── */
.kibad-sidebar-section {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    padding: 14px 0 6px 0;
    border-top: 1px solid #e2e8f0;
    margin-top: 10px;
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar step states ── */
.kibad-sidebar-step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    border-radius: 8px;
    font-size: 0.86rem;
    margin-bottom: 2px;
    transition: background 0.15s;
    font-family: 'Inter', sans-serif;
}
.kibad-sidebar-step-active {
    background: #eff6ff;
    color: #1F3864;
    font-weight: 600;
    padding-left: 16px;
    border-left: 3px solid #2563eb;
}
.kibad-sidebar-step-done {
    color: #059669;
}
.kibad-sidebar-step-pending {
    color: #94a3b8;
}
</style>
"""


def inject_global_css() -> None:
    """Inject global KIBAD design system CSS. Call once per page, after set_page_config."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def inject_sidebar_css() -> None:
    """Inject sidebar-specific CSS."""
    st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)


def inject_all_css() -> None:
    """Inject all CSS (global + sidebar). Use on most pages."""
    st.markdown(_GLOBAL_CSS + _SIDEBAR_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """Render a premium gradient-bordered page header with optional subtitle and breadcrumb style."""
    icon_html = f"<span style='margin-right:10px;font-size:1.5rem;vertical-align:middle'>{icon}</span>" if icon else ""
    sub_html = (
        f"<p style='margin:6px 0 0 0;font-size:0.92rem;color:#64748b;font-weight:400;"
        f"font-family:Inter,sans-serif;line-height:1.5'>{subtitle}</p>"
        if subtitle else ""
    )
    st.markdown(
        f"""<div style='margin-bottom:24px;padding-bottom:16px;
        border-bottom:2px solid transparent;
        background-image:linear-gradient(#f0f4f8,#f0f4f8),linear-gradient(135deg,#1F3864 0%,#2563eb 100%);
        background-origin:border-box;background-clip:padding-box,border-box;
        border-radius:0 0 0 0;'>
        <h1 style='margin:0;font-size:1.65rem;font-weight:800;color:#1F3864;line-height:1.2;
        font-family:Inter,sans-serif;letter-spacing:-0.02em'>
        {icon_html}{title}</h1>{sub_html}</div>""",
        unsafe_allow_html=True,
    )


def section_header(title: str, icon: str = "") -> None:
    """Render a modern styled section heading with optional emoji icon."""
    icon_html = f"<span style='margin-right:8px'>{icon}</span>" if icon else ""
    st.markdown(
        f"""<div style='display:flex;align-items:center;margin:28px 0 14px 0;
        padding-bottom:8px;border-bottom:2px solid #e2e8f0;'>
        <span style='font-size:0.72rem;font-weight:700;text-transform:uppercase;
        letter-spacing:0.1em;color:#94a3b8;font-family:Inter,sans-serif'>
        {icon_html}{title}</span></div>""",
        unsafe_allow_html=True,
    )


def info_card(text: str, kind: str = "info") -> None:
    """Render a premium styled info card. kind: info | success | warning | danger"""
    kind_map = {
        "info": ("ℹ️", "#0369a1", "#eff6ff", "linear-gradient(135deg,#0369a1 0%,#38bdf8 100%)"),
        "success": ("✅", "#047857", "#f0fdf4", "linear-gradient(135deg,#059669 0%,#10b981 100%)"),
        "warning": ("⚠️", "#b45309", "#fffbeb", "linear-gradient(135deg,#d97706 0%,#fbbf24 100%)"),
        "danger": ("❌", "#b91c1c", "#fef2f2", "linear-gradient(135deg,#dc2626 0%,#f87171 100%)"),
    }
    icon, color, bg, gradient = kind_map.get(kind, kind_map["info"])
    st.markdown(
        f"""<div style='background:{bg};border-radius:12px;padding:14px 18px;margin:10px 0;
        font-size:0.9rem;color:{color};position:relative;overflow:hidden;
        box-shadow:0 2px 8px rgba(0,0,0,0.06);font-family:Inter,sans-serif'>
        <div style='position:absolute;left:0;top:0;bottom:0;width:4px;
        background:{gradient};border-radius:4px 0 0 4px'></div>
        <div style='margin-left:8px'>{icon} {text}</div></div>""",
        unsafe_allow_html=True,
    )
