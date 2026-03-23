"""
app/components/ux.py – Reusable Streamlit UI components for KIBAD.
"""
from __future__ import annotations
import streamlit as st


def interpretation_box(title: str, text: str, icon: str = "💡", collapsed: bool = False) -> None:
    """Render a styled interpretation/insight box."""
    with st.expander(f"{icon} {title}", expanded=not collapsed):
        st.markdown(text)


def recommendation_card(
    action_label: str,
    reason: str,
    priority: str,
    on_apply=None,
    key: str = "",
) -> bool:
    """Render a recommendation card. Returns True if user clicked Apply."""
    priority_cfg = {
        "high":   ("🔴", "#ffd7d7", "Высокий"),
        "medium": ("🟡", "#fff3cd", "Средний"),
        "low":    ("🟢", "#d4edda", "Низкий"),
    }
    icon, bg, label = priority_cfg.get(priority, ("⚪", "#f8f9fa", priority))

    st.markdown(
        f"""<div style="background:{bg};border-radius:8px;padding:10px 14px;
        margin-bottom:8px;border-left:4px solid {'#dc3545' if priority=='high' else '#ffc107' if priority=='medium' else '#28a745'}">
        <b>{icon} {label} приоритет</b> — {action_label}<br>
        <small style="color:#555">{reason}</small></div>""",
        unsafe_allow_html=True,
    )
    if on_apply is not None:
        return st.button("✅ Применить", key=f"rec_{key}", help=reason)
    return False


def data_quality_banner(qc: dict) -> None:
    """Show a data quality summary banner after upload."""
    severity = qc.get("severity", "ok")
    n_rows = qc.get("n_rows", 0)
    n_cols = qc.get("n_cols", 0)
    dups = qc.get("duplicate_rows", 0)
    miss_pct = qc.get("overall_missing_pct", 0)
    n_outlier = len(qc.get("outlier_cols", {}))
    n_conflict = len(qc.get("type_conflicts", []))

    if severity == "ok":
        st.success(
            f"✅ **Качество данных: хорошее** — {n_rows:,} строк × {n_cols} колонок. "
            "Критических проблем не обнаружено."
        )
        return

    with st.expander(
        f"{'⚠️' if severity == 'warning' else '🔴'} **Отчёт о качестве данных** — обнаружены проблемы",
        expanded=True,
    ):
        cols = st.columns(4)
        cols[0].metric("Дубликаты", dups, delta=f"-{dups}" if dups else None,
                       delta_color="inverse" if dups else "off")
        cols[1].metric("Пропуски (средн.)", f"{miss_pct:.1f}%",
                       delta=f"{'высокий' if miss_pct > 30 else 'ок'}", delta_color="inverse" if miss_pct > 10 else "off")
        cols[2].metric("Колонки с выбросами", n_outlier)
        cols[3].metric("Конфликты типов", n_conflict)

        if qc.get("null_columns"):
            st.warning(f"🗑️ Полностью пустые колонки: **{', '.join(qc['null_columns'])}** — рекомендуется удалить.")
        if qc.get("high_missing_cols"):
            st.warning(f"📊 Много пропусков (>30%): **{', '.join(qc['high_missing_cols'])}**.")
        if qc.get("outlier_cols"):
            pairs = ", ".join(f"{c} ({v}%)" for c, v in list(qc["outlier_cols"].items())[:5])
            st.info(f"📌 Выбросы обнаружены в: {pairs}.")
        if qc.get("type_conflicts"):
            cols_str = ", ".join(f'«{t["column"]}»' for t in qc["type_conflicts"])
            st.info(f"🔢 Текстовые колонки с числами: {cols_str}. Рекомендуется преобразование типа.")
        if qc.get("constant_cols"):
            st.info(f"📋 Константные колонки (одно значение): **{', '.join(qc['constant_cols'])}**.")


def method_card(
    title: str,
    description: str,
    when_to_use: str,
    requirements: str,
    icon: str = "📊",
) -> None:
    """Render a method selection card with when-to-use guidance."""
    st.markdown(
        f"""<div style="border:1px solid #dee2e6;border-radius:10px;padding:14px;margin-bottom:10px">
        <h4 style="margin:0 0 6px">{icon} {title}</h4>
        <p style="margin:0 0 4px;color:#333">{description}</p>
        <p style="margin:0 0 2px"><b>✅ Когда использовать:</b> {when_to_use}</p>
        <p style="margin:0"><b>📋 Требования:</b> {requirements}</p>
        </div>""",
        unsafe_allow_html=True,
    )


def step_progress(steps: list[str], current: int) -> None:
    """Render a horizontal step progress indicator."""
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        if i < current:
            col.markdown(f"<div style='text-align:center;color:green'>✅<br><small>{step}</small></div>",
                         unsafe_allow_html=True)
        elif i == current:
            col.markdown(f"<div style='text-align:center;color:#0066cc'><b>▶</b><br><small><b>{step}</b></small></div>",
                         unsafe_allow_html=True)
        else:
            col.markdown(f"<div style='text-align:center;color:#aaa'>○<br><small>{step}</small></div>",
                         unsafe_allow_html=True)


def data_filter_widget(
    df: "pd.DataFrame",
    key_prefix: str = "filter",
    max_cat_unique: int = 50,
) -> "pd.DataFrame":
    """Render interactive column filters (like Excel AutoFilter) and return filtered DataFrame.

    Parameters
    ----------
    df:
        Source DataFrame to filter.
    key_prefix:
        Unique prefix for session state keys (use different values on different pages).
    max_cat_unique:
        Max unique values to show as multiselect (above this → text search).

    Returns
    -------
    pd.DataFrame
        Filtered copy of df.
    """
    import pandas as pd

    if df is None or df.empty:
        return df

    with st.expander("🔍 Фильтры данных (Excel AutoFilter)", expanded=False):
        st.caption("Выберите значения для фильтрации строк — как стрелки в заголовках Excel")

        filtered = df.copy()
        n_cols = min(len(df.columns), 6)  # show up to 6 filter controls per row
        filter_cols = st.columns(min(n_cols, 3))

        active_filters: list[str] = []

        for i, col_name in enumerate(df.columns):
            widget_col = filter_cols[i % 3]
            col_data = df[col_name]
            dtype = col_data.dtype

            with widget_col:
                if pd.api.types.is_datetime64_any_dtype(col_data):
                    min_date = col_data.min()
                    max_date = col_data.max()
                    if pd.notna(min_date) and pd.notna(max_date) and min_date != max_date:
                        date_range = st.date_input(
                            f"📅 {col_name}",
                            value=(min_date.date(), max_date.date()),
                            min_value=min_date.date(),
                            max_value=max_date.date(),
                            key=f"{key_prefix}_date_{col_name}",
                        )
                        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                            d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
                            if d0 != pd.Timestamp(min_date.date()) or d1 != pd.Timestamp(max_date.date()):
                                filtered = filtered[(filtered[col_name] >= d0) & (filtered[col_name] <= d1)]
                                active_filters.append(col_name)

                elif pd.api.types.is_numeric_dtype(col_data):
                    col_min = float(col_data.min())
                    col_max = float(col_data.max())
                    if col_min < col_max:
                        rng = st.slider(
                            f"🔢 {col_name}",
                            min_value=col_min, max_value=col_max,
                            value=(col_min, col_max),
                            key=f"{key_prefix}_num_{col_name}",
                        )
                        if rng != (col_min, col_max):
                            filtered = filtered[(filtered[col_name] >= rng[0]) & (filtered[col_name] <= rng[1])]
                            active_filters.append(col_name)

                else:
                    unique_vals = col_data.dropna().unique().tolist()
                    if len(unique_vals) <= max_cat_unique:
                        selected = st.multiselect(
                            f"🏷️ {col_name}",
                            options=sorted(str(v) for v in unique_vals),
                            default=[],
                            key=f"{key_prefix}_cat_{col_name}",
                            placeholder="Все значения",
                        )
                        if selected:
                            filtered = filtered[col_data.astype(str).isin(selected)]
                            active_filters.append(col_name)
                    else:
                        search = st.text_input(
                            f"🔎 {col_name} (поиск)",
                            value="",
                            key=f"{key_prefix}_txt_{col_name}",
                            placeholder="Введите для поиска...",
                        )
                        if search:
                            filtered = filtered[col_data.astype(str).str.contains(search, case=False, na=False)]
                            active_filters.append(col_name)

        n_filtered = len(filtered)
        n_total = len(df)
        if active_filters:
            st.info(
                f"🔍 Активных фильтров: **{len(active_filters)}** ({', '.join(active_filters)}) — "
                f"показано **{n_filtered:,}** из {n_total:,} строк"
            )
        else:
            st.caption(f"Без фильтров — показаны все {n_total:,} строк")

    return filtered


def dataset_stats_sidebar(df: "pd.DataFrame", ds_name: str = "") -> None:
    """Render a compact dataset statistics panel in the sidebar."""
    import pandas as pd

    if df is None or df.empty:
        return

    with st.sidebar:
        st.markdown("---")
        with st.expander(f"📊 Статистика: {ds_name or 'датасет'}", expanded=False):
            num_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            null_pct = df.isnull().mean().mean() * 100
            dup_rows = df.duplicated().sum()

            st.markdown(f"**Строк:** {len(df):,}  **Колонок:** {df.shape[1]}")
            st.markdown(f"**Числовых:** {len(num_cols)}  **Текстовых:** {len(cat_cols)}  **Дат:** {len(date_cols)}")
            st.markdown(f"**Пропусков:** {null_pct:.1f}%  **Дублей:** {dup_rows:,}")

            if num_cols:
                st.markdown("**Числовые столбцы:**")
                desc = df[num_cols].describe().loc[["mean", "min", "max"]].T
                desc.columns = ["Среднее", "Мин", "Макс"]
                st.dataframe(desc.round(2), use_container_width=True)


def active_dataset_warnings(max_warnings: int = 3) -> None:
    """Show a compact sidebar warning panel for the active dataset's QC issues.

    Reads from st.session_state["data_quality_reports"] and surfaces the top
    alerts so analysts don't miss data quality problems regardless of which page
    they are on. Call from any page's sidebar.
    """
    import streamlit as st

    active = st.session_state.get("active_ds")
    if not active:
        return
    qc = st.session_state.get("data_quality_reports", {}).get(active)
    if not qc:
        return

    severity = qc.get("severity", "ok")
    if severity == "ok":
        return  # No warnings to show

    issues = []
    if qc.get("duplicate_rows", 0) > 0:
        issues.append(("🔴", f"Дублей: {qc['duplicate_rows']:,} строк"))
    for col in qc.get("null_columns", [])[:2]:
        issues.append(("🔴", f"Пустая колонка: {col}"))
    miss_pct = qc.get("overall_missing_pct", 0)
    if miss_pct > 10:
        issues.append(("🟡", f"Пропуски: {miss_pct:.1f}% значений"))
    for col in qc.get("high_missing_cols", [])[:2]:
        issues.append(("🟡", f"Много пропусков: {col}"))
    for col, pct in list(qc.get("outlier_cols", {}).items())[:2]:
        issues.append(("🟡", f"Выбросы в {col}: {pct}%"))
    for tc in qc.get("type_conflicts", [])[:2]:
        issues.append(("ℹ️", f"Числа как текст: {tc['column']}"))

    if not issues:
        return

    with st.sidebar:
        with st.expander(
            f"{'⚠️' if severity == 'warning' else '🔴'} Качество данных: {active}",
            expanded=False,
        ):
            for icon, msg in issues[:max_warnings]:
                st.caption(f"{icon} {msg}")
            if len(issues) > max_warnings:
                st.caption(f"…ещё {len(issues) - max_warnings} проблем — откройте «Подготовка»")


def kpi_cards_row(metrics: dict[str, tuple], n_cols: int = 4) -> None:
    """Render a row of st.metric KPI cards.

    Parameters
    ----------
    metrics:
        Dict of {label: (value, delta_str_or_None)}.
    n_cols:
        Number of columns per row.
    """
    keys = list(metrics.keys())
    for start in range(0, len(keys), n_cols):
        batch = keys[start : start + n_cols]
        cols = st.columns(len(batch))
        for col, label in zip(cols, batch):
            val, delta = metrics[label]
            with col:
                if delta is not None:
                    st.metric(label=label, value=val, delta=delta)
                else:
                    st.metric(label=label, value=val)


def apply_recommendation_notification(action: str, df_before, df_after, ds_name: str) -> None:
    """Show a styled before/after notification after auto-applying a recommendation."""
    import streamlit as st
    rows_delta = len(df_after) - len(df_before)
    cols_delta = len(df_after.columns) - len(df_before.columns)
    nulls_before = int(df_before.isnull().sum().sum())
    nulls_after = int(df_after.isnull().sum().sum())
    nulls_delta = nulls_after - nulls_before

    def _delta_str(val, positive_good=True):
        if val == 0:
            return "<span style='color:#6c757d'>без изменений</span>"
        color = "#198754" if (val > 0) == positive_good else "#dc3545"
        sign = "+" if val > 0 else ""
        return f"<span style='color:{color};font-weight:700'>{sign}{val}</span>"

    st.markdown(f"""
    <div style='background:#f0fff4;border:1px solid #a3cfbb;border-left:4px solid #198754;
    border-radius:10px;padding:14px 18px;margin:10px 0'>
    <div style='font-weight:700;color:#0f5132;margin-bottom:8px'>✅ Рекомендация применена: {action}</div>
    <div style='display:flex;gap:24px;font-size:0.9rem'>
    <span>📋 Строки: {_delta_str(rows_delta, positive_good=False)}</span>
    <span>🔢 Пропуски: {_delta_str(nulls_delta, positive_good=False)}</span>
    <span>📊 Колонки: {_delta_str(cols_delta, positive_good=True)}</span>
    </div>
    </div>""", unsafe_allow_html=True)

    # log to audit trail
    if "transform_logs" not in st.session_state:
        st.session_state["transform_logs"] = []
    st.session_state["transform_logs"].append({
        "step": action, "dataset": ds_name,
        "rows_before": len(df_before), "rows_after": len(df_after),
    })
