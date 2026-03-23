"""
pages/2_Prepare.py – Data preparation: column mapping, cleaning, resampling, feature engineering.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

from app.state import init_state, dataset_selectbox, get_active_df, store_prepared
from core.data import apply_type_overrides
from core.autoqc import recommend_preprocessing
from app.components.ux import recommendation_card, show_pending_notification
from core.prepare import (
    parse_dates, resample_timeseries, impute_missing, remove_outliers,
    deduplicate, add_lags, add_rolling, add_ema, add_buckets, normalize,
    add_interaction, RESAMPLE_ALIASES,
)
from app.styles import inject_all_css, page_header, section_header

st.set_page_config(page_title="KIBAD – Prepare", layout="wide")
init_state()
inject_all_css()


def _before_after_card(df_before, df_after, operation: str) -> None:
    """Show a before/after comparison card after a transformation."""
    import pandas as pd
    r_before, c_before = df_before.shape
    r_after, c_after = df_after.shape
    null_before = int(df_before.isnull().sum().sum())
    null_after = int(df_after.isnull().sum().sum())
    dup_before = int(df_before.duplicated().sum())
    dup_after = int(df_after.duplicated().sum())

    r_delta = r_after - r_before
    c_delta = c_after - c_before
    null_delta = null_after - null_before

    def _fmt_delta(d, positive_is_good=False):
        if d == 0:
            return "<span style='color:#6c757d'>без изменений</span>"
        color = "#198754" if (d < 0) == (not positive_is_good) else "#dc3545"
        sign = "+" if d > 0 else ""
        return f"<span style='color:{color};font-weight:600'>{sign}{d:,}</span>"

    st.markdown(
        f"""<div style='background:#f8f9fa;border:1px solid #dee2e6;border-left:4px solid #1F3864;
        border-radius:8px;padding:14px 18px;margin:10px 0'>
        <div style='font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;
        color:#1F3864;margin-bottom:10px'>📊 Результат операции: {operation}</div>
        <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px'>
        <div><div style='font-size:0.72rem;color:#6c757d;text-transform:uppercase;letter-spacing:0.05em'>Строки</div>
        <div style='font-size:1.1rem;font-weight:700'>{r_after:,}</div>
        <div style='font-size:0.82rem'>было: {r_before:,} → {_fmt_delta(r_delta)}</div></div>
        <div><div style='font-size:0.72rem;color:#6c757d;text-transform:uppercase;letter-spacing:0.05em'>Колонки</div>
        <div style='font-size:1.1rem;font-weight:700'>{c_after}</div>
        <div style='font-size:0.82rem'>было: {c_before} → {_fmt_delta(c_delta, positive_is_good=True)}</div></div>
        <div><div style='font-size:0.72rem;color:#6c757d;text-transform:uppercase;letter-spacing:0.05em'>Пропуски</div>
        <div style='font-size:1.1rem;font-weight:700'>{null_after:,}</div>
        <div style='font-size:0.82rem'>было: {null_before:,} → {_fmt_delta(null_delta)}</div></div>
        <div><div style='font-size:0.72rem;color:#6c757d;text-transform:uppercase;letter-spacing:0.05em'>Дубликаты</div>
        <div style='font-size:1.1rem;font-weight:700'>{dup_after:,}</div>
        <div style='font-size:0.82rem'>было: {dup_before:,} → {_fmt_delta(dup_after - dup_before)}</div></div>
        </div></div>""",
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.divider()
    st.markdown("### 📍 Порядок обработки")
    steps_info = [
        ("1", "Типы данных"),
        ("2", "Маппинг колонок"),
        ("3", "Парсинг дат"),
        ("4", "Заполнение пропусков"),
        ("5", "Удаление выбросов"),
        ("6", "Нормализация"),
        ("7", "Фичи и лаги"),
        ("8", "Ресэмплинг"),
        ("9", "Вычисляемые колонки"),
    ]
    for num, label in steps_info:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;padding:4px 6px;"
            f"border-radius:6px;margin-bottom:3px'>"
            f"<span style='display:inline-flex;align-items:center;justify-content:center;"
            f"width:22px;height:22px;background:#1F3864;color:white;border-radius:50%;"
            f"font-size:0.7rem;font-weight:700;flex-shrink:0'>{num}</span>"
            f"<span style='font-size:0.85rem;color:#495057'>{label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

page_header("2. Подготовка данных", "Очистка, трансформация и обогащение", "🔧")
show_pending_notification()

chosen = dataset_selectbox("Датасет для обработки", key="prep_ds_sel")
if not chosen:
    st.stop()

raw_df = st.session_state["datasets"].get(chosen)
if raw_df is None:
    st.error("Датасет не найден.")
    st.stop()

# Working copy – start from prepared if exists
work_df = st.session_state.get("prepared_dfs", {}).get(chosen, raw_df.copy())

_sz_c1, _sz_c2, _sz_c3 = st.columns([1, 1, 4])
_sz_c1.metric("Строк", f"{work_df.shape[0]:,}")
_sz_c2.metric("Столбцов", f"{work_df.shape[1]}")

# --- Pending prepare actions from auto-scan ---
pending = st.session_state.get("pending_prepare_actions", {}).get(chosen, [])
if pending:
    with st.expander(f"⚡ Авторекомендации из проверки качества ({len(pending)} действий)", expanded=True):
        st.info("Следующие действия были рекомендованы автосканированием при загрузке данных.")
        for action in pending:
            st.markdown(f"- **{action.get('action', '?')}** → «{action.get('column', 'вся таблица')}»: {action.get('reason', '')}")
        if st.button("✅ Применить все и продолжить", key="btn_apply_pending"):
            st.session_state["pending_prepare_actions"].pop(chosen, None)
            st.success("Рекомендации применены. Проверьте раздел трансформаций ниже.")
            st.rerun()
        if st.button("❌ Отклонить", key="btn_reject_pending"):
            st.session_state["pending_prepare_actions"].pop(chosen, None)
            st.rerun()

# --- Transform log display ---
logs = st.session_state.get("transform_logs", {}).get(chosen, [])
if logs:
    with st.expander(f"📋 История преобразований ({len(logs)} шагов)", expanded=False):
        for entry in reversed(logs[-10:]):  # last 10
            op = entry.get("operation", entry.get("step", "операция"))
            rows_b = entry.get("rows_before", "?")
            rows_a = entry.get("rows_after", "?")
            ts = entry.get("ts", entry.get("timestamp", ""))
            st.markdown(f"- **{op}**: {rows_b} → {rows_a} строк  <small>{ts}</small>", unsafe_allow_html=True)

# --- Recommendation Panel ---
def _apply_single_rec(df_in: pd.DataFrame, rec: dict) -> pd.DataFrame:
    """Apply a single recommendation and return modified DataFrame."""
    action = rec["action"]
    params = rec.get("auto_params", {})
    col = rec.get("column")

    if action == "drop_nullcol":
        cols_to_drop = params.get("columns", [col] if col else [])
        return df_in.drop(columns=[c for c in cols_to_drop if c in df_in.columns])

    elif action == "drop_duplicates":
        return df_in.drop_duplicates().reset_index(drop=True)

    elif action == "impute":
        target_col = params.get("column", col)
        method = params.get("method", "median")
        if target_col and target_col in df_in.columns:
            if method == "median" and pd.api.types.is_numeric_dtype(df_in[target_col]):
                df_in[target_col] = df_in[target_col].fillna(df_in[target_col].median())
            elif method == "mean" and pd.api.types.is_numeric_dtype(df_in[target_col]):
                df_in[target_col] = df_in[target_col].fillna(df_in[target_col].mean())
            elif method == "mode":
                mv = df_in[target_col].mode()
                if len(mv) > 0:
                    df_in[target_col] = df_in[target_col].fillna(mv[0])
            elif method in ("ffill", "bfill"):
                df_in[target_col] = df_in[target_col].fillna(method=method)
            elif method == "zero":
                df_in[target_col] = df_in[target_col].fillna(0)
            elif method == "drop":
                df_in = df_in.dropna(subset=[target_col]).reset_index(drop=True)
        return df_in

    elif action == "outlier_cap":
        target_col = params.get("column", col)
        threshold = params.get("threshold", 1.5)
        if target_col and target_col in df_in.columns and pd.api.types.is_numeric_dtype(df_in[target_col]):
            s = df_in[target_col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            df_in[target_col] = df_in[target_col].clip(lower=lower, upper=upper)
        return df_in

    elif action == "type_cast":
        target_col = params.get("column", col)
        if target_col and target_col in df_in.columns:
            df_in[target_col] = pd.to_numeric(df_in[target_col], errors="coerce")
        return df_in

    elif action == "drop_constant":
        cols_to_drop = params.get("columns", [col] if col else [])
        return df_in.drop(columns=[c for c in cols_to_drop if c in df_in.columns])

    elif action == "parse_dates":
        target_col = params.get("column", col)
        if target_col and target_col in df_in.columns:
            df_in[target_col] = pd.to_datetime(df_in[target_col], errors="coerce")
        return df_in

    return df_in


qc = st.session_state.get("data_quality_reports", {}).get(chosen, None)
if qc and qc.get("severity") != "ok":
    recs = recommend_preprocessing(raw_df, qc)
    if recs:
        with st.expander("🎯 Рекомендации по обработке данных", expanded=True):
            st.caption(f"Обнаружено {len(recs)} рекомендаций. Вы можете применить их по одной или все сразу.")

            apply_all = st.button("✨ Применить все рекомендации (авто)", type="primary", key="btn_apply_all_recs")

            if apply_all:
                df_before = work_df.copy()
                df_work = work_df.copy()
                applied = []
                for rec in recs:
                    try:
                        df_work = _apply_single_rec(df_work, rec)
                        applied.append(rec["action"])
                    except Exception:
                        pass
                store_prepared(chosen, df_work)
                # Invalidate caches
                st.session_state.get("data_quality_reports", {}).pop(chosen, None)
                st.session_state.get("quality_scores", {}).pop(chosen, None)
                st.session_state.get("auto_insights", {}).pop(chosen, None)
                # Log
                if "transform_logs" not in st.session_state:
                    st.session_state["transform_logs"] = {}
                st.session_state["transform_logs"].setdefault(chosen, []).append({
                    "operation": f"Авто-применение {len(applied)} рекомендаций",
                    "rows_before": len(df_before),
                    "rows_after": len(df_work),
                    "ts": datetime.now().strftime("%H:%M:%S"),
                })
                st.session_state["_rec_notify"] = {
                    "action": f"Применено {len(applied)} рекомендаций ({', '.join(set(applied))})",
                    "rows_before": len(df_before),
                    "rows_after": len(df_work),
                    "rows_delta": len(df_work) - len(df_before),
                    "nulls_before": int(df_before.isnull().sum().sum()),
                    "nulls_after": int(df_work.isnull().sum().sum()),
                    "nulls_delta": int(df_work.isnull().sum().sum()) - int(df_before.isnull().sum().sum()),
                    "ds_name": chosen,
                }
                st.rerun()

            st.divider()

            for i, rec in enumerate(recs):
                col_label = f" → «{rec['column']}»" if rec.get("column") else ""
                action_label = {
                    "drop_nullcol": "Удалить пустую колонку",
                    "drop_duplicates": "Удалить дубликаты",
                    "impute": "Заполнить пропуски",
                    "outlier_cap": "Ограничить выбросы",
                    "type_cast": "Исправить тип данных",
                    "drop_constant": "Удалить константную колонку",
                    "parse_dates": "Распознать дату",
                }.get(rec["action"], rec["action"])
                clicked = recommendation_card(
                    action_label=f"{action_label}{col_label}",
                    reason=rec["reason"],
                    priority=rec["priority"],
                    on_apply=True,
                    key=f"rec_{i}",
                )
                if clicked:
                    df_before = work_df.copy()
                    df_work = _apply_single_rec(work_df.copy(), rec)
                    store_prepared(chosen, df_work)
                    st.session_state.get("data_quality_reports", {}).pop(chosen, None)
                    st.session_state.get("quality_scores", {}).pop(chosen, None)
                    st.session_state.get("auto_insights", {}).pop(chosen, None)
                    st.session_state["transform_logs"].setdefault(chosen, []).append({
                        "operation": f"{action_label}{col_label}",
                        "rows_before": len(df_before),
                        "rows_after": len(df_work),
                        "ts": datetime.now().strftime("%H:%M:%S"),
                    })
                    st.session_state["_rec_notify"] = {
                        "action": f"{action_label}{col_label}",
                        "rows_before": len(df_before),
                        "rows_after": len(df_work),
                        "rows_delta": len(df_work) - len(df_before),
                        "nulls_before": int(df_before.isnull().sum().sum()),
                        "nulls_after": int(df_work.isnull().sum().sum()),
                        "nulls_delta": int(df_work.isnull().sum().sum()) - int(df_before.isnull().sum().sum()),
                        "ds_name": chosen,
                    }
                    st.rerun()

# ---------------------------------------------------------------------------
# 1. Column Mapping
# ---------------------------------------------------------------------------
with st.expander("1. 🗂️ Маппинг колонок (реальные → логические имена)", expanded=True):
    st.markdown(
        "Сопоставьте имена столбцов с логическими ролями, используемыми на последующих страницах."
    )
    mappings = st.session_state.get("col_mappings", {}).get(chosen, {})
    cols = list(work_df.columns)
    na_opt = ["(none)"] + cols

    date_col = st.selectbox("Столбец даты", na_opt,
                             index=na_opt.index(mappings.get("date", "(none)")),
                             key="map_date")
    target_col = st.selectbox("Целевой / метрический столбец", na_opt,
                               index=na_opt.index(mappings.get("target", "(none)")),
                               key="map_target")
    segment_col = st.selectbox("Столбец сегмента / группы (опционально)", na_opt,
                                index=na_opt.index(mappings.get("segment", "(none)")),
                                key="map_segment")
    if st.button("Сохранить маппинг", key="save_mappings"):
        if "col_mappings" not in st.session_state:
            st.session_state["col_mappings"] = {}
        st.session_state["col_mappings"][chosen] = {
            "date": date_col if date_col != "(none)" else "",
            "target": target_col if target_col != "(none)" else "",
            "segment": segment_col if segment_col != "(none)" else "",
        }
        st.success("Маппинг сохранён.")

# ---------------------------------------------------------------------------
# 2. Apply Type Overrides
# ---------------------------------------------------------------------------
with st.expander("2. 🔠 Применение переопределений типов", expanded=False):
    overrides = st.session_state.get("type_overrides", {}).get(chosen, {})
    if overrides:
        if st.button("Применить переопределения типов", key="apply_types"):
            _work_df_snapshot = work_df.copy()
            work_df = apply_type_overrides(work_df, overrides)
            store_prepared(chosen, work_df)
            st.success(f"Применено {len(overrides)} переопределений типов.")
            _before_after_card(_work_df_snapshot, work_df, "Переопределение типов данных")
            st.dataframe(work_df.dtypes.rename("dtype").reset_index(), use_container_width=True)
    else:
        st.info("Переопределения типов не заданы. Перейдите в раздел **Данные → Переопределения типов**.")

# ---------------------------------------------------------------------------
# 3. Date Parsing
# ---------------------------------------------------------------------------
with st.expander("3. 📅 Парсинг дат", expanded=False):
    date_cols_avail = list(work_df.columns)
    dcol = st.selectbox("Столбец для парсинга как дата", date_cols_avail, key="date_parse_col")
    tz_strip = st.checkbox("Удалить информацию о часовом поясе", value=True, key="tz_strip")
    if st.button("Распознать даты", key="btn_parse_dates"):
        try:
            _work_df_snapshot = work_df.copy()
            work_df = parse_dates(work_df, dcol, tz_strip=tz_strip)
            store_prepared(chosen, work_df)
            st.success(f"Столбец '{dcol}' распознан. dtype={work_df[dcol].dtype}")
            _before_after_card(_work_df_snapshot, work_df, f"Парсинг дат: {dcol}")
        except Exception as e:
            st.error(f"Ошибка парсинга дат: {e}")

# ---------------------------------------------------------------------------
# 4. Missing Value Imputation
# ---------------------------------------------------------------------------
with st.expander("4. 🩹 Заполнение пропущенных значений", expanded=False):
    num_cols = work_df.select_dtypes(include="number").columns.tolist()
    impute_cols = st.multiselect("Столбцы для заполнения", work_df.columns.tolist(),
                                  default=num_cols, key="impute_cols")
    method_map = {
        "Медиана": "median", "Среднее": "mean", "Мода": "mode",
        "Заполнение вперёд": "ffill", "Заполнение назад": "bfill",
        "Ноль": "zero", "Удалить строки": "drop",
    }
    impute_method = st.selectbox("Метод", list(method_map.keys()), key="impute_method")

    if impute_cols:
        n_missing = work_df[impute_cols].isna().sum().sum()
        st.info(f"Пропущенных значений в выборке: {n_missing:,}")
    if st.button("Заполнить пропуски", key="btn_impute"):
        try:
            _work_df_snapshot = work_df.copy()
            work_df = impute_missing(work_df, columns=impute_cols, method=method_map[impute_method])
            store_prepared(chosen, work_df)
            st.success(f"Заполнение выполнено ({impute_method}). Форма: {work_df.shape}")
            _before_after_card(_work_df_snapshot, work_df, f"Заполнение пропусков ({impute_method})")
        except Exception as e:
            st.error(f"Ошибка заполнения пропусков: {e}")

# ---------------------------------------------------------------------------
# 5. Outlier Removal
# ---------------------------------------------------------------------------
with st.expander("5. 🎯 Удаление выбросов", expanded=False):
    num_cols_out = work_df.select_dtypes(include="number").columns.tolist()
    out_cols = st.multiselect("Столбцы для проверки выбросов", num_cols_out, key="out_cols")
    out_method = st.selectbox("Метод обнаружения", ["IQR", "Z-score"], key="out_method")
    out_mult = st.slider("Множитель IQR / порог Z-score",
                          min_value=1.0, max_value=5.0, value=1.5, step=0.1, key="out_mult")
    if out_cols and st.button("Удалить выбросы", key="btn_outliers"):
        method_key = "iqr" if out_method == "IQR" else "zscore"
        try:
            _work_df_snapshot = work_df.copy()
            work_df, n_rem = remove_outliers(work_df, out_cols, method=method_key,
                                             iqr_multiplier=out_mult, zscore_threshold=out_mult)
            store_prepared(chosen, work_df)
            st.success(f"Удалено {n_rem} строк с выбросами. Осталось: {work_df.shape[0]:,}")
            _before_after_card(_work_df_snapshot, work_df, f"Удаление выбросов ({out_method})")
        except Exception as e:
            st.error(f"Ошибка удаления выбросов: {e}")

# ---------------------------------------------------------------------------
# 6. Deduplication
# ---------------------------------------------------------------------------
with st.expander("6. 🔂 Дедупликация", expanded=False):
    dup_subset = st.multiselect("Столбцы для проверки (пусто = все)", work_df.columns.tolist(),
                                 key="dup_subset")
    keep_opt = st.radio("Оставить", ["Первое", "Последнее"], horizontal=True, key="dup_keep")
    if st.button("Дедуплицировать", key="btn_dedup"):
        subset = dup_subset if dup_subset else None
        _work_df_snapshot = work_df.copy()
        work_df, n_dup = deduplicate(work_df, subset=subset,
                                     keep="first" if keep_opt == "Первое" else "last")
        store_prepared(chosen, work_df)
        st.success(f"Удалено {n_dup} дублирующихся строк. Осталось: {work_df.shape[0]:,}")
        _before_after_card(_work_df_snapshot, work_df, "Дедупликация")

# ---------------------------------------------------------------------------
# 7. Resampling
# ---------------------------------------------------------------------------
with st.expander("7. ⏱️ Ресэмплинг временного ряда", expanded=False):
    dt_cols = [c for c in work_df.columns
               if pd.api.types.is_datetime64_any_dtype(work_df[c])]
    if not dt_cols:
        st.warning("Столбцы с датами не найдены. Сначала распознайте дату (шаг 3).")
    else:
        rs_date = st.selectbox("Столбец даты", dt_cols, key="rs_date")
        num_cols_rs = work_df.select_dtypes(include="number").columns.tolist()
        rs_vals = st.multiselect("Столбцы значений для агрегации", num_cols_rs, default=num_cols_rs[:1], key="rs_vals")
        rs_freq_label = st.selectbox("Частота ресэмплинга", list(RESAMPLE_ALIASES.keys()), key="rs_freq")
        rs_agg = st.selectbox("Агрегация", ["sum", "mean", "median", "min", "max", "last"], key="rs_agg")
        cat_cols = work_df.select_dtypes(include="object").columns.tolist()
        rs_group = st.selectbox("Группировка (опционально)", ["(none)"] + cat_cols, key="rs_group")
        grp_col = None if rs_group == "(none)" else rs_group

        if rs_vals and st.button("Ресэмплировать", key="btn_resample"):
            try:
                _work_df_snapshot = work_df.copy()
                resampled = resample_timeseries(
                    work_df, rs_date, rs_vals,
                    freq=RESAMPLE_ALIASES[rs_freq_label],
                    agg_func=rs_agg,
                    group_cols=[grp_col] if grp_col else None,
                )
                store_prepared(chosen, resampled)
                work_df = resampled
                st.success(f"Ресэмплинг до {rs_freq_label}. Форма: {resampled.shape}")
                _before_after_card(_work_df_snapshot, work_df, f"Ресэмплинг ({rs_freq_label})")
                st.dataframe(resampled.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка ресэмплинга: {e}")

# ---------------------------------------------------------------------------
# 8. Feature Engineering
# ---------------------------------------------------------------------------
with st.expander("8. ⚙️ Инженерия признаков", expanded=False):
    num_cols_fe = work_df.select_dtypes(include="number").columns.tolist()
    cat_cols_fe = work_df.select_dtypes(include="object").columns.tolist()

    fe_tab_lag, fe_tab_roll, fe_tab_ema, fe_tab_bucket, fe_tab_norm, fe_tab_interact = st.tabs(
        ["Лаги", "Скользящие", "EMA", "Бакеты", "Нормализация", "Взаимодействие"]
    )

    with fe_tab_lag:
        lag_col = st.selectbox("Столбец", num_cols_fe, key="fe_lag_col")
        lag_vals = st.text_input("Периоды лага (через запятую)", "1,2,3,12", key="fe_lag_vals")
        lag_grp = st.selectbox("Группировка (опционально)", ["(none)"] + cat_cols_fe, key="fe_lag_grp")
        if st.button("Добавить лаговые признаки", key="btn_lags"):
            try:
                lags = [int(x.strip()) for x in lag_vals.split(",") if x.strip().isdigit()]
                grp = None if lag_grp == "(none)" else lag_grp
                _work_df_snapshot = work_df.copy()
                work_df = add_lags(work_df, lag_col, lags, group_col=grp)
                store_prepared(chosen, work_df)
                st.success(f"Добавлены лаги: {lags}. Форма: {work_df.shape}")
                _before_after_card(_work_df_snapshot, work_df, f"Добавление лаг-признаков ({lag_col})")
            except Exception as e:
                st.error(f"Ошибка лагов: {e}")

    with fe_tab_roll:
        roll_col = st.selectbox("Столбец", num_cols_fe, key="fe_roll_col")
        roll_windows = st.text_input("Окна (через запятую)", "3,6,12", key="fe_roll_win")
        roll_func = st.selectbox("Функция", ["mean", "std", "sum", "min", "max"], key="fe_roll_fn")
        roll_grp = st.selectbox("Группировка (опционально)", ["(none)"] + cat_cols_fe, key="fe_roll_grp")
        if st.button("Добавить скользящие признаки", key="btn_rolling"):
            try:
                wins = [int(x.strip()) for x in roll_windows.split(",") if x.strip().isdigit()]
                grp = None if roll_grp == "(none)" else roll_grp
                _work_df_snapshot = work_df.copy()
                work_df = add_rolling(work_df, roll_col, wins, func=roll_func, group_col=grp)
                store_prepared(chosen, work_df)
                st.success(f"Добавлены скользящие окна: {wins}. Форма: {work_df.shape}")
                _before_after_card(_work_df_snapshot, work_df, f"Скользящие признаки ({roll_col}, {roll_func})")
            except Exception as e:
                st.error(f"Ошибка скользящих признаков: {e}")

    with fe_tab_ema:
        ema_col = st.selectbox("Столбец", num_cols_fe, key="fe_ema_col")
        ema_spans = st.text_input("Периоды EMA (через запятую)", "3,6,12", key="fe_ema_spans")
        if st.button("Добавить EMA-признаки", key="btn_ema"):
            try:
                spans = [int(x.strip()) for x in ema_spans.split(",") if x.strip().isdigit()]
                _work_df_snapshot = work_df.copy()
                work_df = add_ema(work_df, ema_col, spans)
                store_prepared(chosen, work_df)
                st.success(f"Добавлены EMA-периоды: {spans}. Форма: {work_df.shape}")
                _before_after_card(_work_df_snapshot, work_df, f"Добавление EMA-признаков ({ema_col})")
            except Exception as e:
                st.error(f"Ошибка EMA: {e}")

    with fe_tab_bucket:
        bkt_col = st.selectbox("Столбец", num_cols_fe, key="fe_bkt_col")
        bkt_n = st.slider("Количество квантилей", 2, 10, 4, key="fe_bkt_n")
        bkt_newcol = st.text_input("Имя нового столбца", f"{bkt_col}_bucket", key="fe_bkt_name")
        if st.button("Добавить бакет-признак", key="btn_bucket"):
            try:
                _work_df_snapshot = work_df.copy()
                work_df = add_buckets(work_df, bkt_col, n_quantiles=bkt_n, new_col=bkt_newcol)
                store_prepared(chosen, work_df)
                st.success(f"Добавлен бакет-столбец '{bkt_newcol}'. Форма: {work_df.shape}")
                _before_after_card(_work_df_snapshot, work_df, f"Бакетизация: {bkt_newcol}")
            except Exception as e:
                st.error(f"Ошибка бакетизации: {e}")

    with fe_tab_norm:
        norm_cols = st.multiselect("Столбцы для нормализации", num_cols_fe, key="fe_norm_cols")
        norm_method = st.radio("Метод", ["zscore", "minmax"], horizontal=True, key="fe_norm_method")
        if norm_cols and st.button("Нормализовать", key="btn_norm"):
            _work_df_snapshot = work_df.copy()
            work_df = normalize(work_df, norm_cols, method=norm_method)
            store_prepared(chosen, work_df)
            st.success(f"Нормализовано {len(norm_cols)} столбцов.")
            _before_after_card(_work_df_snapshot, work_df, f"Нормализация ({norm_method})")

    with fe_tab_interact:
        if len(num_cols_fe) >= 2:
            ia_a = st.selectbox("Столбец A", num_cols_fe, key="fe_ia_a")
            ia_b = st.selectbox("Столбец B", num_cols_fe, index=1, key="fe_ia_b")
            ia_op = st.selectbox("Операция", ["multiply", "divide", "add", "subtract"], key="fe_ia_op")
            ia_name = st.text_input("Имя нового столбца", f"{ia_a}_{ia_op}_{ia_b}", key="fe_ia_name")
            if st.button("Добавить взаимодействие", key="btn_interact"):
                try:
                    _work_df_snapshot = work_df.copy()
                    work_df = add_interaction(work_df, ia_a, ia_b, op=ia_op, new_col=ia_name)
                    store_prepared(chosen, work_df)
                    st.success(f"Добавлен столбец взаимодействия '{ia_name}'. Форма: {work_df.shape}")
                    _before_after_card(_work_df_snapshot, work_df, f"Признак взаимодействия: {ia_name}")
                except Exception as e:
                    st.error(f"Ошибка взаимодействия: {e}")
        else:
            st.info("Нужно минимум 2 числовых столбца.")

# ---------------------------------------------------------------------------
# 9. Calculated Column Builder (➕ Формулы)
# ---------------------------------------------------------------------------
with st.expander("9. ➕ Формулы — Построитель вычисляемых колонок"):
    st.markdown(
        "Создавайте новые колонки без написания кода — выбирайте режим и заполняйте поля."
    )

    _num_cols_f = work_df.select_dtypes(include="number").columns.tolist()
    _str_cols_f = work_df.select_dtypes(include="object").columns.tolist()
    _dt_cols_f = [c for c in work_df.columns if pd.api.types.is_datetime64_any_dtype(work_df[c])]
    _all_cols_f = list(work_df.columns)

    _formula_mode = st.selectbox(
        "Режим вычисления",
        [
            "1. Арифметика",
            "2. Условие (IF/CASE)",
            "3. Дата → Признаки",
            "4. Текст → Признаки",
            "5. Банковские формулы",
        ],
        key="formula_mode",
    )

    # -----------------------------------------------------------------------
    # Helper: log transform
    # -----------------------------------------------------------------------
    def _log_formula(ds_name: str, col_name: str, rows_before: int, rows_after: int) -> None:
        if "transform_logs" not in st.session_state:
            st.session_state["transform_logs"] = {}
        if ds_name not in st.session_state["transform_logs"]:
            st.session_state["transform_logs"][ds_name] = []
        st.session_state["transform_logs"][ds_name].append({
            "operation": f"Формула: {col_name}",
            "rows_before": rows_before,
            "rows_after": rows_after,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    # -----------------------------------------------------------------------
    # Helper: validate new column name
    # -----------------------------------------------------------------------
    def _validate_col_name(name: str, df: pd.DataFrame) -> tuple[bool, str]:
        if not name or not name.strip():
            return False, "Имя колонки не может быть пустым."
        if name in df.columns:
            return None, f"Колонка «{name}» уже существует — будет перезаписана."
        return True, ""

    # ===================================================================
    # MODE 1 – Arithmetic
    # ===================================================================
    if _formula_mode == "1. Арифметика":
        st.markdown("#### Арифметика")
        arith_new_col = st.text_input("Имя новой колонки", "новая_колонка", key="arith_new_col")

        _const_left_opt = ["Константа"] + _num_cols_f
        _const_right_opt = ["Константа"] + _num_cols_f

        ca1, ca2, ca3 = st.columns([2, 1, 2])
        with ca1:
            arith_left = st.selectbox("Левый операнд", _const_left_opt, key="arith_left")
            if arith_left == "Константа":
                arith_left_const = st.number_input("Значение (левое)", value=0.0, key="arith_left_val")
            else:
                arith_left_const = None
        with ca2:
            arith_op = st.selectbox(
                "Оператор",
                ["+", "−", "×", "÷", "//", "%", "^"],
                key="arith_op",
            )
        with ca3:
            arith_right = st.selectbox("Правый операнд", _const_right_opt, key="arith_right")
            if arith_right == "Константа":
                arith_right_const = st.number_input("Значение (правое)", value=1.0, key="arith_right_val")
            else:
                arith_right_const = None

        arith_round = st.number_input("Округлить до N знаков (−1 = без округления)", value=2, min_value=-1, max_value=10, step=1, key="arith_round")

        if st.button("▶ Предпросмотр (первые 5 строк)", key="arith_preview"):
            try:
                _lv = work_df[arith_left].astype(float) if arith_left != "Константа" else pd.Series([arith_left_const] * len(work_df), index=work_df.index)
                _rv = work_df[arith_right].astype(float) if arith_right != "Константа" else pd.Series([arith_right_const] * len(work_df), index=work_df.index)
                _op_map = {"+": lambda a, b: a + b, "−": lambda a, b: a - b, "×": lambda a, b: a * b,
                           "÷": lambda a, b: a / b.replace(0, np.nan), "//": lambda a, b: a // b.replace(0, np.nan),
                           "%": lambda a, b: a % b.replace(0, np.nan), "^": lambda a, b: a ** b}
                _res = _op_map[arith_op](_lv, _rv)
                _res = _res.replace([np.inf, -np.inf], np.nan)
                if arith_round >= 0:
                    _res = _res.round(arith_round)
                st.dataframe(pd.DataFrame({arith_new_col: _res}).head(5), use_container_width=True)
            except Exception as _e:
                st.error(f"Ошибка предпросмотра: {_e}")

        if st.button("✅ Применить", key="arith_apply"):
            _ok, _msg = _validate_col_name(arith_new_col, work_df)
            if _ok is False:
                st.error(_msg)
            else:
                if _ok is None:
                    st.warning(_msg)
                try:
                    _rows_b = len(work_df)
                    _work_df_snapshot = work_df.copy()
                    _lv = work_df[arith_left].astype(float) if arith_left != "Константа" else pd.Series([arith_left_const] * len(work_df), index=work_df.index)
                    _rv = work_df[arith_right].astype(float) if arith_right != "Константа" else pd.Series([arith_right_const] * len(work_df), index=work_df.index)
                    _op_map = {"+": lambda a, b: a + b, "−": lambda a, b: a - b, "×": lambda a, b: a * b,
                               "÷": lambda a, b: a / b.replace(0, np.nan), "//": lambda a, b: a // b.replace(0, np.nan),
                               "%": lambda a, b: a % b.replace(0, np.nan), "^": lambda a, b: a ** b}
                    _res = _op_map[arith_op](_lv, _rv)
                    _res = _res.replace([np.inf, -np.inf], np.nan)
                    if arith_round >= 0:
                        _res = _res.round(arith_round)
                    work_df = work_df.copy()
                    work_df[arith_new_col] = _res
                    store_prepared(chosen, work_df)
                    _log_formula(chosen, arith_new_col, _rows_b, len(work_df))
                    st.success(f"Столбец '{arith_new_col}' добавлен. Форма: {work_df.shape}")
                    _before_after_card(_work_df_snapshot, work_df, f"Вычисляемая колонка: {arith_new_col}")
                    _prev_cols = list(work_df.columns[:-4]) if len(work_df.columns) > 4 else list(work_df.columns[:-1])
                    _show_cols = list(work_df.columns[-3:]) + [arith_new_col] if arith_new_col not in list(work_df.columns[-3:]) else list(work_df.columns[-3:])
                    st.dataframe(work_df[_show_cols].head(5), use_container_width=True)
                    st.rerun()
                except Exception as _e:
                    st.error(f"Ошибка вычисления: {_e}")

    # ===================================================================
    # MODE 2 – Condition (IF/CASE)
    # ===================================================================
    elif _formula_mode == "2. Условие (IF/CASE)":
        st.markdown("#### Условие (IF/CASE)")
        cond_new_col = st.text_input("Имя новой колонки", "флаг", key="cond_new_col")

        _cond_ops = [">", "<", ">=", "<=", "==", "!=", "contains", "starts_with", "is_null", "is_not_null"]
        _connectors = ["И", "ИЛИ"]

        n_conditions = st.number_input("Количество условий (1–3)", min_value=1, max_value=3, value=1, step=1, key="cond_n")

        conditions = []
        for _ci in range(int(n_conditions)):
            st.markdown(f"**Условие {_ci + 1}**")
            _cc1, _cc2, _cc3 = st.columns([2, 1, 2])
            with _cc1:
                _cond_col = st.selectbox("Колонка", _all_cols_f, key=f"cond_col_{_ci}")
            with _cc2:
                _cond_op = st.selectbox("Оператор", _cond_ops, key=f"cond_op_{_ci}")
            with _cc3:
                if _cond_op in ("is_null", "is_not_null"):
                    _cond_val = None
                    st.text_input("Значение (не требуется)", value="—", disabled=True, key=f"cond_val_{_ci}")
                else:
                    _cond_val = st.text_input("Значение", key=f"cond_val_{_ci}")

            _true_val = st.text_input(f"Результат если ИСТИНА (условие {_ci + 1})", "1", key=f"cond_true_{_ci}")

            _connector = None
            if _ci < int(n_conditions) - 1:
                _connector = st.radio(f"Связь с условием {_ci + 2}", _connectors, horizontal=True, key=f"cond_conn_{_ci}")

            conditions.append({"col": _cond_col, "op": _cond_op, "val": _cond_val, "true_val": _true_val, "connector": _connector})

        cond_else_val = st.text_input("Значение если ЛОЖЬ (else)", "0", key="cond_else_val")
        cond_result_type = st.selectbox("Тип результата", ["Число", "Текст", "0/1 флаг"], key="cond_result_type")

        def _build_mask(df: pd.DataFrame, cond: dict) -> pd.Series:
            col, op, val = cond["col"], cond["op"], cond["val"]
            s = df[col]
            if op == "is_null":
                return s.isna()
            if op == "is_not_null":
                return s.notna()
            if op == "contains":
                return s.astype(str).str.contains(str(val), na=False)
            if op == "starts_with":
                return s.astype(str).str.startswith(str(val), na=False)
            try:
                _num_val = float(val)
                s_num = pd.to_numeric(s, errors="coerce")
                if op == ">":    return s_num > _num_val
                if op == "<":    return s_num < _num_val
                if op == ">=":   return s_num >= _num_val
                if op == "<=":   return s_num <= _num_val
                if op == "==":   return s_num == _num_val
                if op == "!=":   return s_num != _num_val
            except (ValueError, TypeError):
                pass
            if op == "==":  return s.astype(str) == str(val)
            if op == "!=":  return s.astype(str) != str(val)
            return pd.Series(False, index=df.index)

        if st.button("✅ Применить", key="cond_apply"):
            _ok, _msg = _validate_col_name(cond_new_col, work_df)
            if _ok is False:
                st.error(_msg)
            else:
                if _ok is None:
                    st.warning(_msg)
                try:
                    _rows_b = len(work_df)
                    _work_df_snapshot = work_df.copy()
                    _combined_mask = _build_mask(work_df, conditions[0])
                    for _ci in range(1, len(conditions)):
                        _m = _build_mask(work_df, conditions[_ci])
                        _prev_conn = conditions[_ci - 1]["connector"]
                        if _prev_conn == "ИЛИ":
                            _combined_mask = _combined_mask | _m
                        else:
                            _combined_mask = _combined_mask & _m

                    # Use first condition's true_val (simplification: IF all conditions → true_val[0])
                    _true_v = conditions[0]["true_val"]
                    _else_v = cond_else_val

                    def _cast(v: str, typ: str):
                        if typ == "Число":
                            try:
                                return float(v)
                            except (ValueError, TypeError):
                                return np.nan
                        elif typ == "0/1 флаг":
                            try:
                                return int(float(v))
                            except (ValueError, TypeError):
                                return int(bool(v))
                        return v

                    work_df = work_df.copy()
                    work_df[cond_new_col] = np.where(
                        _combined_mask,
                        _cast(_true_v, cond_result_type),
                        _cast(_else_v, cond_result_type),
                    )
                    store_prepared(chosen, work_df)
                    _log_formula(chosen, cond_new_col, _rows_b, len(work_df))
                    st.success(f"Столбец '{cond_new_col}' добавлен. Форма: {work_df.shape}")
                    _before_after_card(_work_df_snapshot, work_df, f"Условная колонка: {cond_new_col}")
                    _show_cols = list(work_df.columns[-3:])
                    if cond_new_col not in _show_cols:
                        _show_cols = _show_cols[-2:] + [cond_new_col]
                    st.dataframe(work_df[_show_cols].head(5), use_container_width=True)
                    st.rerun()
                except Exception as _e:
                    st.error(f"Ошибка вычисления: {_e}")

    # ===================================================================
    # MODE 3 – Date → Features
    # ===================================================================
    elif _formula_mode == "3. Дата → Признаки":
        st.markdown("#### Дата → Признаки")
        if not _dt_cols_f:
            st.warning("Нет колонок с типом datetime. Выполните парсинг дат (шаг 3) сначала.")
        else:
            dt_src_col = st.selectbox("Источник (datetime колонка)", _dt_cols_f, key="dt_feat_src")
            st.markdown("Выберите части для извлечения:")
            _dt_parts = {
                "Год": ("year", f"{dt_src_col}_год"),
                "Квартал": ("quarter", f"{dt_src_col}_квартал"),
                "Месяц": ("month", f"{dt_src_col}_месяц"),
                "Неделя": ("isocalendar().week", f"{dt_src_col}_неделя"),
                "День": ("day", f"{dt_src_col}_день"),
                "День_недели": ("dayofweek", f"{dt_src_col}_день_недели"),
                "Час": ("hour", f"{dt_src_col}_час"),
            }
            _dt_selected = []
            _dt_col_parts = st.columns(4)
            _dt_part_list = list(_dt_parts.keys())
            for _di, _dname in enumerate(_dt_part_list):
                _checked = _dt_col_parts[_di % 4].checkbox(_dname, key=f"dt_part_{_dname}")
                if _checked:
                    _dt_selected.append(_dname)

            if st.button("✅ Применить", key="dt_feat_apply"):
                if not _dt_selected:
                    st.warning("Выберите хотя бы одну часть даты.")
                else:
                    try:
                        _rows_b = len(work_df)
                        _work_df_snapshot = work_df.copy()
                        work_df = work_df.copy()
                        _added = []
                        _s = work_df[dt_src_col]
                        for _dname in _dt_selected:
                            _attr, _new_col_name = _dt_parts[_dname]
                            if _dname == "Неделя":
                                work_df[_new_col_name] = _s.dt.isocalendar().week.astype(int)
                            else:
                                work_df[_new_col_name] = getattr(_s.dt, _attr)
                            _added.append(_new_col_name)
                        store_prepared(chosen, work_df)
                        for _nc in _added:
                            _log_formula(chosen, _nc, _rows_b, len(work_df))
                        st.success(f"Добавлено {len(_added)} колонок: {', '.join(_added)}. Форма: {work_df.shape}")
                        _before_after_card(_work_df_snapshot, work_df, f"Дата → Признаки ({dt_src_col})")
                        st.dataframe(work_df[_added].head(5), use_container_width=True)
                        st.rerun()
                    except Exception as _e:
                        st.error(f"Ошибка вычисления: {_e}")

    # ===================================================================
    # MODE 4 – Text → Features
    # ===================================================================
    elif _formula_mode == "4. Текст → Признаки":
        st.markdown("#### Текст → Признаки")
        if not _str_cols_f:
            st.warning("Нет строковых (object) колонок.")
        else:
            txt_src_col = st.selectbox("Источник (строковая колонка)", _str_cols_f, key="txt_feat_src")
            txt_op = st.selectbox(
                "Операция",
                [
                    "Верхний регистр",
                    "Нижний регистр",
                    "Длина строки",
                    "Содержит (флаг 0/1)",
                    "Начинается с",
                    "Заменить",
                    "Первые N символов",
                ],
                key="txt_feat_op",
            )
            txt_new_col = st.text_input(
                "Имя новой колонки",
                f"{txt_src_col}_{txt_op.replace(' ', '_').lower()}",
                key="txt_feat_new_col",
            )

            txt_pattern = txt_from = txt_to = txt_n = None
            if txt_op in ("Содержит (флаг 0/1)", "Начинается с"):
                txt_pattern = st.text_input("Паттерн / подстрока", key="txt_feat_pattern")
            elif txt_op == "Заменить":
                _tr1, _tr2 = st.columns(2)
                txt_from = _tr1.text_input("Заменить", key="txt_feat_from")
                txt_to = _tr2.text_input("На", key="txt_feat_to")
            elif txt_op == "Первые N символов":
                txt_n = st.number_input("N символов", min_value=1, max_value=500, value=3, step=1, key="txt_feat_n")

            if st.button("✅ Применить", key="txt_feat_apply"):
                _ok, _msg = _validate_col_name(txt_new_col, work_df)
                if _ok is False:
                    st.error(_msg)
                else:
                    if _ok is None:
                        st.warning(_msg)
                    try:
                        _rows_b = len(work_df)
                        _work_df_snapshot = work_df.copy()
                        work_df = work_df.copy()
                        _s = work_df[txt_src_col].astype(str)
                        if txt_op == "Верхний регистр":
                            work_df[txt_new_col] = _s.str.upper()
                        elif txt_op == "Нижний регистр":
                            work_df[txt_new_col] = _s.str.lower()
                        elif txt_op == "Длина строки":
                            work_df[txt_new_col] = _s.str.len()
                        elif txt_op == "Содержит (флаг 0/1)":
                            work_df[txt_new_col] = _s.str.contains(txt_pattern or "", na=False).astype(int)
                        elif txt_op == "Начинается с":
                            work_df[txt_new_col] = _s.str.startswith(txt_pattern or "", na=False).astype(int)
                        elif txt_op == "Заменить":
                            work_df[txt_new_col] = _s.str.replace(txt_from or "", txt_to or "", regex=False)
                        elif txt_op == "Первые N символов":
                            work_df[txt_new_col] = _s.str[:int(txt_n)]
                        store_prepared(chosen, work_df)
                        _log_formula(chosen, txt_new_col, _rows_b, len(work_df))
                        st.success(f"Столбец '{txt_new_col}' добавлен. Форма: {work_df.shape}")
                        _before_after_card(_work_df_snapshot, work_df, f"Текст → Признак: {txt_new_col}")
                        _show_cols = ([txt_src_col] if txt_src_col in work_df.columns else []) + [txt_new_col]
                        st.dataframe(work_df[_show_cols].head(5), use_container_width=True)
                        st.rerun()
                    except Exception as _e:
                        st.error(f"Ошибка вычисления: {_e}")

    # ===================================================================
    # MODE 5 – Banking Formulas
    # ===================================================================
    elif _formula_mode == "5. Банковские формулы":
        st.markdown("#### Банковские формулы")
        st.info("Выберите готовую банковскую формулу — поля заполнятся автоматически.")

        _bank_formula = st.selectbox(
            "Банковская формула",
            [
                "— выберите —",
                "PAR (Portfolio at Risk)",
                "Флаг NPL",
                "DPD Бакеты",
                "Процентный доход (мес.)",
                "LTV (Loan-to-Value)",
                "Recovery Rate",
                "Cost-to-Income Ratio",
                "Ранг по колонке",
                "Скользящее среднее",
                "Кумулятивная сумма",
                "IFRS 9 Стейджинг (Stage 1/2/3)",
                "Флаг SICR (просрочка 30+ дней или рост DPD на 30%)",
            ],
            key="bank_formula_sel",
        )

        _num_opts = _num_cols_f if _num_cols_f else ["(нет числовых колонок)"]

        if _bank_formula == "PAR (Portfolio at Risk)":
            st.markdown("**PAR = overdue_balance / total_balance × 100**")
            _bk1, _bk2 = st.columns(2)
            par_overdue = _bk1.selectbox("Колонка просроченного баланса", _num_opts, key="par_overdue")
            par_total = _bk2.selectbox("Колонка общего баланса", _num_opts, key="par_total")
            par_new_col = st.text_input("Имя новой колонки", "PAR_%", key="par_new_col")
            if st.button("✅ Применить", key="par_apply"):
                _ok, _msg = _validate_col_name(par_new_col, work_df)
                if _ok is False: st.error(_msg)
                else:
                    if _ok is None: st.warning(_msg)
                    try:
                        _rows_b = len(work_df)
                        _work_df_snapshot = work_df.copy()
                        work_df = work_df.copy()
                        _denom = pd.to_numeric(work_df[par_total], errors="coerce").replace(0, np.nan)
                        work_df[par_new_col] = (pd.to_numeric(work_df[par_overdue], errors="coerce") / _denom * 100).replace([np.inf, -np.inf], np.nan).round(2)
                        store_prepared(chosen, work_df)
                        _log_formula(chosen, par_new_col, _rows_b, len(work_df))
                        st.success(f"Столбец '{par_new_col}' добавлен. Форма: {work_df.shape}")
                        _before_after_card(_work_df_snapshot, work_df, f"Банковская формула: {par_new_col} (PAR)")
                        st.dataframe(work_df[[par_overdue, par_total, par_new_col]].head(5), use_container_width=True)
                        st.rerun()
                    except Exception as _e: st.error(f"Ошибка вычисления: {_e}")

        elif _bank_formula == "Флаг NPL":
            st.markdown("**NPL_flag = IF DPD > 90 THEN 1 ELSE 0**")
            npl_dpd_col = st.selectbox("Колонка DPD (дни просрочки)", _num_opts, key="npl_dpd_col")
            npl_threshold = st.number_input("Порог DPD", value=90, min_value=0, step=1, key="npl_threshold")
            npl_new_col = st.text_input("Имя новой колонки", "NPL_flag", key="npl_new_col")
            if st.button("✅ Применить", key="npl_apply"):
                _ok, _msg = _validate_col_name(npl_new_col, work_df)
                if _ok is False: st.error(_msg)
                else:
                    if _ok is None: st.warning(_msg)
                    try:
                        _rows_b = len(work_df)
                        _work_df_snapshot = work_df.copy()
                        work_df = work_df.copy()
                        work_df[npl_new_col] = (pd.to_numeric(work_df[npl_dpd_col], errors="coerce") > npl_threshold).astype(int)
                        store_prepared(chosen, work_df)
                        _log_formula(chosen, npl_new_col, _rows_b, len(work_df))
                        st.success(f"Столбец '{npl_new_col}' добавлен. Форма: {work_df.shape}")
                        _before_after_card(_work_df_snapshot, work_df, f"Банковская формула: {npl_new_col} (Флаг NPL)")
                        st.dataframe(work_df[[npl_dpd_col, npl_new_col]].head(5), use_container_width=True)
                        st.rerun()
                    except Exception as _e: st.error(f"Ошибка вычисления: {_e}")

        elif _bank_formula == "DPD Бакеты":
            st.markdown("**DPD Бакеты**: 0 / 1–30 / 31–60 / 61–90 / 91–120 / 120+")
            dpd_col = st.selectbox("Колонка DPD", _num_opts, key="dpd_bkt_col")
            dpd_new_col = st.text_input("Имя новой колонки", "DPD_bucket", key="dpd_bkt_new_col")
            if st.button("✅ Применить", key="dpd_bkt_apply"):
                _ok, _msg = _validate_col_name(dpd_new_col, work_df)
                if _ok is False: st.error(_msg)
                else:
                    if _ok is None: st.warning(_msg)
                    try:
                        _rows_b = len(work_df)
                        _work_df_snapshot = work_df.copy()
                        work_df = work_df.copy()
                        _dpd_s = pd.to_numeric(work_df[dpd_col], errors="coerce")
                        _bins = [-np.inf, 0, 30, 60, 90, 120, np.inf]
                        _lbls = ["0 (Текущий)", "1–30", "31–60", "61–90", "91–120", "120+"]
                        work_df[dpd_new_col] = pd.cut(_dpd_s, bins=_bins, labels=_lbls, include_lowest=True).astype(str)
                        store_prepared(chosen, work_df)
                        _log_formula(chosen, dpd_new_col, _rows_b, len(work_df))
                        st.success(f"Столбец '{dpd_new_col}' добавлен. Форма: {work_df.shape}")
                        _before_after_card(_work_df_snapshot, work_df, f"Банковская формула: {dpd_new_col} (DPD Бакеты)")
                        st.dataframe(work_df[[dpd_col, dpd_new_col]].head(10), use_container_width=True)
                        st.rerun()
                    except Exception as _e: st.error(f"Ошибка вычисления: {_e}")

        elif _bank_formula == "Процентный доход (мес.)":
            st.markdown("**monthly_interest = balance × annual_rate / 12 / 100**")
            _bk1, _bk2 = st.columns(2)
            mi_bal_col = _bk1.selectbox("Колонка баланса", _num_opts, key="mi_bal_col")
            mi_rate_col = _bk2.selectbox("Колонка годовой ставки (%)", _num_opts, key="mi_rate_col")
            mi_new_col = st.text_input("Имя новой колонки", "monthly_interest", key="mi_new_col")
            if st.button("✅ Применить", key="mi_apply"):
                _ok, _msg = _validate_col_name(mi_new_col, work_df)
                if _ok is False: st.error(_msg)
                else:
                    if _ok is None: st.warning(_msg)
                    try:
                        _rows_b = len(work_df)
                        _work_df_snapshot = work_df.copy()
                        work_df = work_df.copy()
                        work_df[mi_new_col] = (
                            pd.to_numeric(work_df[mi_bal_col], errors="coerce") *
                            pd.to_numeric(work_df[mi_rate_col], errors="coerce") / 12 / 100
                        ).replace([np.inf, -np.inf], np.nan).round(2)
                        store_prepared(chosen, work_df)
                        _log_formula(chosen, mi_new_col, _rows_b, len(work_df))
                        st.success(f"Столбец '{mi_new_col}' добавлен. Форма: {work_df.shape}")
                        _before_after_card(_work_df_snapshot, work_df, f"Банковская формула: {mi_new_col} (Процентный доход)")
                        st.dataframe(work_df[[mi_bal_col, mi_rate_col, mi_new_col]].head(5), use_container_width=True)
                        st.rerun()
                    except Exception as _e: st.error(f"Ошибка вычисления: {_e}")

        elif _bank_formula == "LTV (Loan-to-Value)":
            st.markdown("**LTV = loan_amount / collateral_value × 100**")
            _bk1, _bk2 = st.columns(2)
            ltv_loan_col = _bk1.selectbox("Колонка суммы кредита", _num_opts, key="ltv_loan_col")
            ltv_coll_col = _bk2.selectbox("Колонка стоимости залога", _num_opts, key="ltv_coll_col")
            ltv_new_col = st.text_input("Имя новой колонки", "LTV_%", key="ltv_new_col")
            if st.button("✅ Применить", key="ltv_apply"):
                _ok, _msg = _validate_col_name(ltv_new_col, work_df)
                if _ok is False: st.error(_msg)
                else:
                    if _ok is None: st.warning(_msg)
                    try:
                        _rows_b = len(work_df)
                        _work_df_snapshot = work_df.copy()
                        work_df = work_df.copy()
                        _denom = pd.to_numeric(work_df[ltv_coll_col], errors="coerce").replace(0, np.nan)
                        work_df[ltv_new_col] = (pd.to_numeric(work_df[ltv_loan_col], errors="coerce") / _denom * 100).replace([np.inf, -np.inf], np.nan).round(2)
                        store_prepared(chosen, work_df)
                        _log_formula(chosen, ltv_new_col, _rows_b, len(work_df))
                        st.success(f"Столбец '{ltv_new_col}' добавлен. Форма: {work_df.shape}")
                        _before_after_card(_work_df_snapshot, work_df, f"Банковская формула: {ltv_new_col} (LTV)")
                        st.dataframe(work_df[[ltv_loan_col, ltv_coll_col, ltv_new_col]].head(5), use_container_width=True)
                        st.rerun()
                    except Exception as _e: st.error(f"Ошибка вычисления: {_e}")

        elif _bank_formula == "Recovery Rate":
            st.markdown("**Recovery Rate = recovered_amount / defaulted_amount × 100**")
            _bk1, _bk2 = st.columns(2)
            rr_rec_col = _bk1.selectbox("Колонка возврата", _num_opts, key="rr_rec_col")
            rr_def_col = _bk2.selectbox("Колонка дефолта", _num_opts, key="rr_def_col")
            rr_new_col = st.text_input("Имя новой колонки", "recovery_rate_%", key="rr_new_col")
            if st.button("✅ Применить", key="rr_apply"):
                _ok, _msg = _validate_col_name(rr_new_col, work_df)
                if _ok is False: st.error(_msg)
                else:
                    if _ok is None: st.warning(_msg)
                    try:
                        _rows_b = len(work_df)
                        _work_df_snapshot = work_df.copy()
                        work_df = work_df.copy()
                        _denom = pd.to_numeric(work_df[rr_def_col], errors="coerce").replace(0, np.nan)
                        work_df[rr_new_col] = (pd.to_numeric(work_df[rr_rec_col], errors="coerce") / _denom * 100).replace([np.inf, -np.inf], np.nan).round(2)
                        store_prepared(chosen, work_df)
                        _log_formula(chosen, rr_new_col, _rows_b, len(work_df))
                        st.success(f"Столбец '{rr_new_col}' добавлен. Форма: {work_df.shape}")
                        _before_after_card(_work_df_snapshot, work_df, f"Банковская формула: {rr_new_col} (Recovery Rate)")
                        st.dataframe(work_df[[rr_rec_col, rr_def_col, rr_new_col]].head(5), use_container_width=True)
                        st.rerun()
                    except Exception as _e: st.error(f"Ошибка вычисления: {_e}")

        elif _bank_formula == "Cost-to-Income Ratio":
            st.markdown("**CTI = costs / income × 100**")
            _bk1, _bk2 = st.columns(2)
            cti_cost_col = _bk1.selectbox("Колонка затрат", _num_opts, key="cti_cost_col")
            cti_inc_col = _bk2.selectbox("Колонка доходов", _num_opts, key="cti_inc_col")
            cti_new_col = st.text_input("Имя новой колонки", "CTI_%", key="cti_new_col")
            if st.button("✅ Применить", key="cti_apply"):
                _ok, _msg = _validate_col_name(cti_new_col, work_df)
                if _ok is False: st.error(_msg)
                else:
                    if _ok is None: st.warning(_msg)
                    try:
                        _rows_b = len(work_df)
                        _work_df_snapshot = work_df.copy()
                        work_df = work_df.copy()
                        _denom = pd.to_numeric(work_df[cti_inc_col], errors="coerce").replace(0, np.nan)
                        work_df[cti_new_col] = (pd.to_numeric(work_df[cti_cost_col], errors="coerce") / _denom * 100).replace([np.inf, -np.inf], np.nan).round(2)
                        store_prepared(chosen, work_df)
                        _log_formula(chosen, cti_new_col, _rows_b, len(work_df))
                        st.success(f"Столбец '{cti_new_col}' добавлен. Форма: {work_df.shape}")
                        _before_after_card(_work_df_snapshot, work_df, f"Банковская формула: {cti_new_col} (CTI)")
                        st.dataframe(work_df[[cti_cost_col, cti_inc_col, cti_new_col]].head(5), use_container_width=True)
                        st.rerun()
                    except Exception as _e: st.error(f"Ошибка вычисления: {_e}")

        elif _bank_formula == "Ранг по колонке":
            st.markdown("**Ранг = rank(method='dense')**")
            rank_col = st.selectbox("Колонка для ранжирования", _num_opts, key="rank_col")
            rank_asc = st.radio("Порядок сортировки", ["По убыванию", "По возрастанию"], horizontal=True, key="rank_asc")
            rank_new_col = st.text_input("Имя новой колонки", f"{rank_col}_rank", key="rank_new_col")
            if st.button("✅ Применить", key="rank_apply"):
                _ok, _msg = _validate_col_name(rank_new_col, work_df)
                if _ok is False: st.error(_msg)
                else:
                    if _ok is None: st.warning(_msg)
                    try:
                        _rows_b = len(work_df)
                        _work_df_snapshot = work_df.copy()
                        work_df = work_df.copy()
                        _ascending = (rank_asc == "По возрастанию")
                        work_df[rank_new_col] = pd.to_numeric(work_df[rank_col], errors="coerce").rank(method="dense", ascending=_ascending).astype("Int64")
                        store_prepared(chosen, work_df)
                        _log_formula(chosen, rank_new_col, _rows_b, len(work_df))
                        st.success(f"Столбец '{rank_new_col}' добавлен. Форма: {work_df.shape}")
                        _before_after_card(_work_df_snapshot, work_df, f"Банковская формула: {rank_new_col} (Ранг)")
                        st.dataframe(work_df[[rank_col, rank_new_col]].head(5), use_container_width=True)
                        st.rerun()
                    except Exception as _e: st.error(f"Ошибка вычисления: {_e}")

        elif _bank_formula == "Скользящее среднее":
            st.markdown("**MA = rolling(window).mean()**")
            ma_col = st.selectbox("Колонка", _num_opts, key="ma_col")
            ma_window = st.number_input("Размер окна", min_value=2, max_value=500, value=3, step=1, key="ma_window")
            ma_new_col = st.text_input("Имя новой колонки", f"{ma_col}_MA{int(ma_window)}", key="ma_new_col")
            if st.button("✅ Применить", key="ma_apply"):
                _ok, _msg = _validate_col_name(ma_new_col, work_df)
                if _ok is False: st.error(_msg)
                else:
                    if _ok is None: st.warning(_msg)
                    try:
                        _rows_b = len(work_df)
                        _work_df_snapshot = work_df.copy()
                        work_df = work_df.copy()
                        work_df[ma_new_col] = pd.to_numeric(work_df[ma_col], errors="coerce").rolling(int(ma_window), min_periods=1).mean().round(4)
                        store_prepared(chosen, work_df)
                        _log_formula(chosen, ma_new_col, _rows_b, len(work_df))
                        st.success(f"Столбец '{ma_new_col}' добавлен. Форма: {work_df.shape}")
                        _before_after_card(_work_df_snapshot, work_df, f"Банковская формула: {ma_new_col} (Скользящее среднее)")
                        st.dataframe(work_df[[ma_col, ma_new_col]].head(5), use_container_width=True)
                        st.rerun()
                    except Exception as _e: st.error(f"Ошибка вычисления: {_e}")

        elif _bank_formula == "Кумулятивная сумма":
            st.markdown("**cumsum = cumulative sum**")
            cs_col = st.selectbox("Колонка", _num_opts, key="cs_col")
            cs_new_col = st.text_input("Имя новой колонки", f"{cs_col}_cumsum", key="cs_new_col")
            if st.button("✅ Применить", key="cs_apply"):
                _ok, _msg = _validate_col_name(cs_new_col, work_df)
                if _ok is False: st.error(_msg)
                else:
                    if _ok is None: st.warning(_msg)
                    try:
                        _rows_b = len(work_df)
                        _work_df_snapshot = work_df.copy()
                        work_df = work_df.copy()
                        work_df[cs_new_col] = pd.to_numeric(work_df[cs_col], errors="coerce").cumsum()
                        store_prepared(chosen, work_df)
                        _log_formula(chosen, cs_new_col, _rows_b, len(work_df))
                        st.success(f"Столбец '{cs_new_col}' добавлен. Форма: {work_df.shape}")
                        _before_after_card(_work_df_snapshot, work_df, f"Банковская формула: {cs_new_col} (Кумулятивная сумма)")
                        st.dataframe(work_df[[cs_col, cs_new_col]].head(5), use_container_width=True)
                        st.rerun()
                    except Exception as _e: st.error(f"Ошибка вычисления: {_e}")

        elif _bank_formula == "IFRS 9 Стейджинг (Stage 1/2/3)":
            st.markdown("**IFRS 9 Stage**: Stage 1 (норма) / Stage 2 (SICR) / Stage 3 (дефолт)")
            _all_opts_f = _all_cols_f if _all_cols_f else ["(нет колонок)"]
            ifrs_dpd_col = st.selectbox("Колонка DPD (дни просрочки)", _num_opts, key="ifrs_dpd_col")
            _ifrs_c1, _ifrs_c2 = st.columns(2)
            ifrs_sicr_col = _ifrs_c1.selectbox(
                "SICR флаг (доп. признак Stage 2) — опционально",
                ["(нет)"] + _all_opts_f,
                key="ifrs_sicr_col",
            )
            ifrs_forbear_col = _ifrs_c2.selectbox(
                "Флаг реструктуризации — опционально",
                ["(нет)"] + _all_opts_f,
                key="ifrs_forbear_col",
            )
            _ifrs_n1, _ifrs_n2 = st.columns(2)
            ifrs_stage2_thresh = _ifrs_n1.number_input("Порог DPD для Stage 2", value=30, min_value=0, step=1, key="ifrs_stage2_thresh")
            ifrs_stage3_thresh = _ifrs_n2.number_input("Порог DPD для Stage 3", value=90, min_value=0, step=1, key="ifrs_stage3_thresh")
            ifrs_out_col = st.text_input("Имя выходной колонки", "stage_ifrs9", key="ifrs_out_col")
            if st.button("✅ Применить", key="ifrs_apply"):
                _ok, _msg = _validate_col_name(ifrs_out_col, work_df)
                if _ok is False: st.error(_msg)
                else:
                    if _ok is None: st.warning(_msg)
                    try:
                        _rows_b = len(work_df)
                        _work_df_snapshot = work_df.copy()
                        work_df = work_df.copy()
                        _dpd_s = pd.to_numeric(work_df[ifrs_dpd_col], errors="coerce").fillna(0)
                        result = pd.Series(1, index=work_df.index)
                        stage2_mask = _dpd_s > ifrs_stage2_thresh
                        _sicr = None if ifrs_sicr_col == "(нет)" else ifrs_sicr_col
                        _forbear = None if ifrs_forbear_col == "(нет)" else ifrs_forbear_col
                        if _sicr and _sicr in work_df.columns:
                            stage2_mask = stage2_mask | (work_df[_sicr].fillna(0) > 0)
                        if _forbear and _forbear in work_df.columns:
                            stage2_mask = stage2_mask | (work_df[_forbear].fillna(0) > 0)
                        stage3_mask = _dpd_s > ifrs_stage3_thresh
                        result[stage2_mask] = 2
                        result[stage3_mask] = 3
                        work_df[ifrs_out_col] = result
                        store_prepared(chosen, work_df)
                        _log_formula(chosen, ifrs_out_col, _rows_b, len(work_df))
                        _vc = work_df[ifrs_out_col].value_counts().sort_index()
                        n1 = int(_vc.get(1, 0))
                        n2 = int(_vc.get(2, 0))
                        n3 = int(_vc.get(3, 0))
                        st.success(f"Стейджинг выполнен: Stage 1: {n1}, Stage 2: {n2}, Stage 3: {n3}")
                        _before_after_card(_work_df_snapshot, work_df, f"IFRS 9 Стейджинг ({ifrs_out_col})")
                        st.bar_chart(work_df[ifrs_out_col].value_counts().sort_index())
                        st.rerun()
                    except Exception as _e: st.error(f"Ошибка вычисления: {_e}")

        elif _bank_formula == "Флаг SICR (просрочка 30+ дней или рост DPD на 30%)":
            st.markdown("**SICR = (DPD > 30) ИЛИ (прирост DPD ≥ 30% от предыдущего периода)**")
            _sicr_c1, _sicr_c2 = st.columns(2)
            sicr_dpd_col = _sicr_c1.selectbox("Колонка DPD (текущий период)", _num_opts, key="sicr_dpd_col")
            sicr_dpd_prev_col = _sicr_c2.selectbox("Колонка DPD (предыдущий период)", _num_opts, key="sicr_dpd_prev_col")
            sicr_out_col = st.text_input("Имя выходной колонки", "sicr_flag", key="sicr_out_col")
            if st.button("✅ Применить", key="sicr_apply"):
                _ok, _msg = _validate_col_name(sicr_out_col, work_df)
                if _ok is False: st.error(_msg)
                else:
                    if _ok is None: st.warning(_msg)
                    try:
                        _rows_b = len(work_df)
                        _work_df_snapshot = work_df.copy()
                        work_df = work_df.copy()
                        _dpd = pd.to_numeric(work_df[sicr_dpd_col], errors="coerce").fillna(0)
                        _dpd_prev = pd.to_numeric(work_df[sicr_dpd_prev_col], errors="coerce").fillna(0)
                        sicr = (_dpd > 30) | (_dpd - _dpd_prev >= 0.3 * _dpd_prev.clip(lower=1))
                        work_df[sicr_out_col] = sicr.astype(int)
                        store_prepared(chosen, work_df)
                        _log_formula(chosen, sicr_out_col, _rows_b, len(work_df))
                        _n_sicr = int(work_df[sicr_out_col].sum())
                        st.success(f"Столбец '{sicr_out_col}' добавлен. SICR = 1: {_n_sicr} строк ({_n_sicr/len(work_df)*100:.1f}%)")
                        _before_after_card(_work_df_snapshot, work_df, f"Банковская формула: {sicr_out_col} (Флаг SICR)")
                        st.dataframe(work_df[[sicr_dpd_col, sicr_dpd_prev_col, sicr_out_col]].head(5), use_container_width=True)
                        st.rerun()
                    except Exception as _e: st.error(f"Ошибка вычисления: {_e}")

        else:
            st.info("Выберите банковскую формулу из списка выше.")

# ---------------------------------------------------------------------------
# Preview final
# ---------------------------------------------------------------------------
st.divider()
section_header("Текущий датасет (предпросмотр)", "👁️")
st.caption(f"Форма: {work_df.shape[0]:,} строк × {work_df.shape[1]} столбцов")
st.dataframe(work_df.head(20), use_container_width=True)
csv_bytes = work_df.to_csv(index=False).encode()
st.download_button("⬇ Скачать подготовленный CSV", csv_bytes,
                   file_name=f"{chosen}_prepared.csv", mime="text/csv")

if st.button("🔄 Сбросить к исходным данным", key="btn_reset"):
    st.session_state["confirm_reset"] = True
if st.session_state.get("confirm_reset"):
    st.warning("⚠️ Все изменения будут потеряны. Вы уверены?")
    c1, c2 = st.columns(2)
    if c1.button("✅ Подтвердить сброс", key="btn_confirm_reset"):
        store_prepared(chosen, raw_df.copy())
        st.info("Сброшено к исходным данным.")
        st.session_state["confirm_reset"] = False
        st.rerun()
    if c2.button("❌ Отмена", key="btn_cancel_reset"):
        st.session_state["confirm_reset"] = False
        st.rerun()
