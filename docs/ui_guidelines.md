# KIBAD UI/UX Guidelines

## Design Philosophy

KIBAD follows a bank-friendly minimalistic design: clean layouts, neutral colors,
professional typography, and no visual clutter. The interface should feel like an
internal analytical tool used by business analysts, not a consumer web app.

## Color Palette

| Role           | Hex Code   | Usage                                      |
|----------------|------------|--------------------------------------------|
| Primary        | `#3498db`  | Buttons, links, active tab indicators      |
| Background     | `#fafafa`  | Page background, card backgrounds          |
| Text           | `#2c3e50`  | Body text, headings                        |
| Secondary text | `#7f8c8d`  | Labels, captions, placeholder text         |
| Success        | `#27ae60`  | Positive results, passed validations       |
| Warning        | `#f39c12`  | Warnings, borderline results               |
| Danger         | `#e74c3c`  | Errors, failed validations, p < 0.05       |
| Surface        | `#ffffff`  | Cards, panels, modal backgrounds           |
| Border         | `#ecf0f1`  | Dividers, table borders, card outlines     |

## Typography

- **Headings**: System sans-serif stack (system-ui, -apple-system, Segoe UI).
  Page titles use `st.title()`, section headers use `st.header()` and
  `st.subheader()`.
- **Body text**: 14-16px, `#2c3e50` color, 1.5 line height.
- **Monospace**: Used for numeric values in tables and code blocks.
- **No decorative fonts**: consistency and readability are prioritized.

## Page Layout

### Navigation

The left sidebar contains navigation to all 10 pages in logical workflow order:

```
1. Data             -- Load data
2. Prepare          -- Clean & transform
3. GroupAggregate    -- Aggregate
4. Explore          -- Visual EDA
5. Tests            -- Hypothesis testing
6. TimeSeries       -- Time series analysis
7. Attribution      -- Factor attribution
8. Simulation       -- What-if scenarios
9. Report           -- Export results
10. Help            -- Documentation
```

This order reflects the natural analysis workflow: load, clean, transform,
analyze, export.

### Main Workspace

Each page follows a consistent layout pattern:

```
+--sidebar--+------------------main area-------------------+
|            |                                              |
| Settings   |  [Tab 1] [Tab 2] [Tab 3]                    |
| & config   |  +------------------------------------------+|
|            |  |                                          ||
| Parameters |  |  Content area                            ||
|            |  |  (charts, tables, results)               ||
| Filters    |  |                                          ||
|            |  +------------------------------------------+|
| Actions    |  |  Status bar / messages                   ||
+------------+----------------------------------------------+
```

### Sidebar Usage

The sidebar is reserved for:

- **Settings**: parameters that affect the entire page (column selections,
  date ranges, method choices).
- **Filters**: data filters that apply to the current view.
- **Actions**: buttons that trigger operations (run analysis, export).
- **Status**: dataset info (row count, column count, data source).

Content and results always appear in the main area, never in the sidebar.

### Tab Pattern

Pages with multiple views use `st.tabs()`:

```python
tab_chart, tab_table, tab_stats = st.tabs(["Chart", "Table", "Statistics"])

with tab_chart:
    st.plotly_chart(fig, use_container_width=True)

with tab_table:
    st.dataframe(df, use_container_width=True)

with tab_stats:
    st.write(summary_stats)
```

## Component Patterns

### Buttons

- **Primary action**: `st.button("Run Analysis", type="primary")` -- one per
  page, placed prominently in the sidebar or at the top of the main area.
- **Secondary actions**: `st.button("Reset")` -- default styling.
- **Destructive actions**: red text or icon, always with confirmation.

### Form Layout

Group related inputs using `st.form()` to prevent premature reruns:

```python
with st.form("analysis_form"):
    metric = st.selectbox("Metric", options=numeric_cols)
    group = st.selectbox("Group by", options=categorical_cols)
    alpha = st.slider("Significance level", 0.01, 0.10, 0.05)
    submitted = st.form_submit_button("Run", type="primary")
```

### Data Display

- **DataFrames**: use `st.dataframe()` with `use_container_width=True`.
  Highlight key columns with `column_config`.
- **Metrics**: use `st.metric()` for single KPIs with delta indicators.
- **Charts**: use `st.plotly_chart()` with `use_container_width=True`. Set
  consistent Plotly template across all pages.

### Status Messages

| Type      | Component             | Usage                            |
|-----------|-----------------------|----------------------------------|
| Success   | `st.success()`        | Operation completed              |
| Warning   | `st.warning()`        | Non-critical issue               |
| Error     | `st.error()`          | Operation failed                 |
| Info      | `st.info()`           | Neutral information              |

### Expanders

Use `st.expander()` for optional details that should not clutter the main view:

- Methodology explanations.
- Full statistical output.
- Debug information.

## Localization (i18n)

KIBAD supports Russian as the primary language with English fallback. All
user-facing strings are managed through `core/i18n.py`.

### Rules

- All UI labels, button texts, messages, and tooltips use translated strings.
- Error messages include both a user-friendly description and a technical detail
  in an expander.
- Column names from user data are displayed as-is (not translated).
- Chart axis labels use the original column names.

### Implementation

```python
from core.i18n import t

st.header(t("explore.title"))  # Returns Russian or English string
st.button(t("common.run"))
```

## Chart Styling

All Plotly charts use a consistent theme:

```python
CHART_TEMPLATE = {
    "layout": {
        "font": {"family": "system-ui, sans-serif", "color": "#2c3e50"},
        "paper_bgcolor": "#fafafa",
        "plot_bgcolor": "#ffffff",
        "colorway": ["#3498db", "#e74c3c", "#27ae60", "#f39c12",
                      "#9b59b6", "#1abc9c", "#e67e22", "#34495e"],
        "margin": {"l": 60, "r": 20, "t": 40, "b": 60},
    }
}
```

### Chart rules

- Always set `use_container_width=True` for responsive sizing.
- Include axis labels with units where applicable.
- Use hover tooltips for detailed information.
- Limit the number of series to 8-10 for readability.
- Export capability via kaleido (PNG/SVG) for reports.

## Responsive Behavior

Streamlit handles responsiveness natively. Follow these guidelines:

- Avoid fixed pixel widths -- use `use_container_width=True`.
- Use `st.columns()` for side-by-side layouts (2-3 columns maximum).
- On narrow screens, columns stack vertically automatically.
- Test the interface at 1280px and 1920px widths.

## Accessibility

- Use descriptive labels on all form elements.
- Provide alt text context in chart titles and descriptions.
- Maintain sufficient color contrast (WCAG AA: 4.5:1 for text).
- Do not rely on color alone to convey information -- use shapes, labels,
  or patterns as secondary indicators.

## Anti-Patterns

Avoid these common mistakes:

- Placing results in the sidebar (sidebar is for inputs only).
- Using more than 3 levels of nesting (tabs within expanders within columns).
- Showing raw tracebacks to users (wrap in try/except, show friendly message).
- Auto-running expensive computations on every widget change (use forms or
  explicit run buttons).
- Using custom CSS/HTML injection unless absolutely necessary.
