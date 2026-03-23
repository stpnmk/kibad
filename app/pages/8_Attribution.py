"""
pages/7_Attribution.py – Factor Attribution / Decomposition.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.state import init_state, dataset_selectbox, get_active_df
from app.styles import inject_all_css, page_header, section_header
from core.i18n import t
from app.components.ux import interpretation_box
from core.audit import log_event
from core.attribution import (
    additive_attribution,
    multiplicative_attribution,
    regression_attribution,
    shapley_attribution,
    waterfall_data,
    AttributionResult,
)
from core.explore import plot_waterfall

st.set_page_config(page_title="KIBAD – Атрибуция", layout="wide")
init_state()
inject_all_css()

page_header("8. Факторный анализ", "Декомпозиция вклада факторов в изменение показателя", "🎯")

with st.expander("📋 Ожидаемый формат данных", expanded=False):
    st.markdown("""
    Каждая строка — один сегмент/период. Для каждого драйвера нужны две колонки: текущий и предыдущий период.

    | сегмент | выручка_тек | выручка_пред | стоимость_тек | стоимость_пред |
    |---|---|---|---|---|
    | Север | 12 500 000 | 11 200 000 | 7 800 000 | 6 900 000 |
    | Юг | 8 300 000 | 9 100 000 | 5 100 000 | 5 600 000 |

    **Соглашение по именам:** Используйте суффиксы `_тек`/`_пред` или `_current`/`_last` — они будут распознаны автоматически.
    """)

st.markdown("""
**Что делает:** Объясняет изменение (дельту) целевого показателя,
разбивая его на вклады отдельных факторов-драйверов.

**Когда использовать:** Когда есть парные значения «текущий» и «предыдущий» период
для целевого показателя и его драйверов, и нужно понять *что стало причиной изменения*.

**Когда НЕ использовать / ограничения:**
- Драйверы должны быть содержательно связаны с целевым показателем.
- Высокая мультиколлинеарность драйверов завышает оценки регрессионного метода.
- Мультипликативный метод требует строго положительных значений.
""")

st.divider()

ds_name = dataset_selectbox(label=t("select_dataset"), key="attr_ds",
                            help="Выберите датасет с парными значениями текущего и предыдущего периода")
if not ds_name:
    st.stop()

df = get_active_df()
if df is None or df.empty:
    st.info("📥 Данные не загружены. Перейдите на страницу **[1. Данные](pages/1_Data.py)** и загрузите файл.")
    st.stop()

all_cols = df.columns.tolist()
num_cols = df.select_dtypes(include="number").columns.tolist()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
section_header(t("settings"))

col1, col2 = st.columns(2)
with col1:
    target_col = st.selectbox(t("attr_target_col"), num_cols, key="attr_target",
                              help="Целевой показатель текущего периода (например, выручка_тек)")
with col2:
    target_prev_col = st.selectbox(t("attr_target_prev"), num_cols, key="attr_target_prev",
                                   help="Целевой показатель предыдущего периода (например, выручка_пред)")

# Driver columns
driver_cols = st.multiselect(t("attr_drivers"), [c for c in num_cols if c != target_col and c != target_prev_col],
                             key="attr_drivers",
                             help="Факторы-драйверы текущего периода. Подберите соответствующие предыдущие ниже.")

# Detect _last columns automatically
auto_prev = []
for dc in driver_cols:
    candidate = dc + "_last"
    if candidate in num_cols:
        auto_prev.append(candidate)
    else:
        auto_prev.append(None)

st.markdown(f"*{t('attr_drivers_prev')}*")
driver_prev_cols = []
for i, dc in enumerate(driver_cols):
    default_prev = auto_prev[i] if auto_prev[i] else ""
    remaining = [c for c in num_cols if c not in driver_cols and c != target_col and c != target_prev_col]
    if auto_prev[i] and auto_prev[i] in remaining:
        idx = remaining.index(auto_prev[i])
    else:
        idx = 0
    sel = st.selectbox(f"Предыдущий период: {dc}", remaining, index=idx, key=f"attr_prev_{dc}")
    driver_prev_cols.append(sel)

# Method
method = st.selectbox(t("attr_method"), [
    t("attr_method_additive"),
    t("attr_method_ratio"),
    t("attr_method_regression"),
    t("attr_method_shapley"),
], key="attr_method_sel",
    help="Аддитивный — простая разность; Мультипликативный — дробное разложение; Регрессионный — OLS; Шепли — справедливое распределение вклада.")

# Segment
cat_cols = [c for c in all_cols if c not in num_cols]
segment_col = st.selectbox(
    t("attr_segment_drilldown") + " (необязательно)",
    ["(none)"] + cat_cols,
    key="attr_segment",
    help="Категориальная колонка для детализации атрибуции по сегментам.",
)
segment_col = None if segment_col == "(none)" else segment_col

# ---------------------------------------------------------------------------
# Run attribution
# ---------------------------------------------------------------------------
if st.button(t("run"), key="btn_attr", type="primary"):
    if not driver_cols or not driver_prev_cols:
        st.error("Выберите хотя бы один драйвер и его колонку предыдущего периода.")
        st.stop()

    method_map = {
        t("attr_method_additive"): "additive",
        t("attr_method_ratio"): "multiplicative",
        t("attr_method_regression"): "regression",
        t("attr_method_shapley"): "shapley",
    }
    method_key = method_map.get(method, "additive")

    with st.spinner("Вычисляем атрибуцию..."):
        try:
            if method_key == "additive":
                result = additive_attribution(df, target_col, target_prev_col, driver_cols, driver_prev_cols, segment_col)
            elif method_key == "multiplicative":
                result = multiplicative_attribution(df, target_col, target_prev_col, driver_cols, driver_prev_cols)
            elif method_key == "regression":
                result = regression_attribution(df, target_col, driver_cols, target_prev_col, driver_prev_cols)
            elif method_key == "shapley":
                result = shapley_attribution(df, target_col, target_prev_col, driver_cols, driver_prev_cols)
            else:
                result = additive_attribution(df, target_col, target_prev_col, driver_cols, driver_prev_cols, segment_col)

            st.session_state["attribution_results"].append(result)
            log_event("analysis_run", {
                "type": "factor_attribution",
                "method": method_key,
                "dataset": ds_name,
                "target": target_col,
                "drivers": driver_cols,
            })

        except Exception as e:
            st.error(f"Ошибка атрибуции: {e}")
            st.stop()

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if st.session_state.get("attribution_results"):
    result = st.session_state["attribution_results"][-1]

    st.divider()
    section_header(t("attr_contributions"))
    st.metric("Изменение целевого показателя", f"{result.target_delta:,.4f}")
    st.metric(t("attr_residual"), f"{result.residual:,.4f}")

    st.dataframe(result.contributions, use_container_width=True)

    # Waterfall chart
    st.divider()
    section_header(t("attr_waterfall"))
    cats, vals = waterfall_data(result)
    fig = plot_waterfall(
        cats, vals,
        title=t("attr_waterfall"),
        base_label=target_prev_col or "Start",
        total_label=target_col or "End",
    )
    st.plotly_chart(fig, use_container_width=True)

    interpretation_box(
        "Как читать водопадную диаграмму",
        "Каждый столбец показывает вклад одного фактора в общее изменение целевого показателя. "
        "Зелёные столбцы — положительный вклад (рост), красные — отрицательный (снижение). "
        "Последний столбец — итоговое изменение.",
        icon="📊",
    )

    # Segment drill-down
    if result.segment_detail is not None:
        st.divider()
        section_header(t("attr_segment_drilldown"))
        st.dataframe(result.segment_detail, use_container_width=True)

    # Metadata
    if result.metadata:
        with st.expander("Метаданные модели"):
            st.json(result.metadata)

    st.divider()
    # Export
    csv_data = result.contributions.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 " + t("export_csv"),
        data=csv_data,
        file_name="attribution_result.csv",
        mime="text/csv",
    )
