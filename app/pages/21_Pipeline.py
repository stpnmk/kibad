"""
pages/21_Pipeline.py – Visual automation pipeline builder.

Replaces Python scripts and Excel macros. User defines a sequence of
data transformations graphically, saves it, and can replay it on any
new dataset with one click.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import io
from datetime import datetime

import pandas as pd
import streamlit as st

from app.state import init_state, dataset_selectbox, get_active_df, add_dataset, store_prepared
from core.audit import log_event
from app.styles import inject_all_css, page_header, section_header

st.set_page_config(page_title="KIBAD – Пайплайн", layout="wide")
init_state()
inject_all_css()
page_header("21. Пайплайн", "Автоматизация последовательностей операций", "⚙️")

# ---------------------------------------------------------------------------
# Session state for pipelines
# ---------------------------------------------------------------------------
if "pipelines" not in st.session_state:
    st.session_state["pipelines"] = {}  # name → list[step_dict]
if "current_pipeline" not in st.session_state:
    st.session_state["current_pipeline"] = []
if "pipeline_step_previews" not in st.session_state:
    st.session_state["pipeline_step_previews"] = []
if "pipeline_run_count" not in st.session_state:
    st.session_state["pipeline_run_count"] = 0
# Track active tab index via session state
if "_pipe_active_tab" not in st.session_state:
    st.session_state["_pipe_active_tab"] = 0

# ---------------------------------------------------------------------------
# Pipeline templates
# ---------------------------------------------------------------------------
PIPELINE_TEMPLATES = {
    "🧹 Стандартная очистка": {
        "description": "Удаляет дубликаты, заполняет пропуски медианой/модой, убирает константные колонки",
        "steps": [
            {"type": "🧹 Удалить дубликаты", "params": {"subset": None}, "label": "Удалить дубликаты"},
            {"type": "🩹 Заполнить пропуски", "params": {"col": "Все числовые", "method": "Медиана", "value": None}, "label": "Заполнить пропуски"},
        ],
        "icon": "🧹", "color": "#198754"
    },
    "📊 Подготовка к анализу": {
        "description": "Очистка + удаление дубликатов + заполнение пропусков медианой",
        "steps": [
            {"type": "🧹 Удалить дубликаты", "params": {"subset": None}, "label": "Удалить дубликаты"},
            {"type": "🩹 Заполнить пропуски", "params": {"col": "Все числовые", "method": "Медиана", "value": None}, "label": "Заполнить пропуски"},
        ],
        "icon": "📊", "color": "#0d6efd"
    },
    "🔝 Топ-анализ": {
        "description": "Сортировка по первому числовому столбцу, выбор топ-100 записей",
        "steps": [
            {"type": "🔢 Верхние N строк", "params": {"n": 100}, "label": "Верхние N строк"},
        ],
        "icon": "🔝", "color": "#6f42c1"
    },
    "📅 Извлечь даты": {
        "description": "Из первой datetime колонки извлекает год, месяц, квартал, день недели",
        "steps": [
            {"type": "📅 Извлечь дату-признак", "params": {"col": "—", "parts": ["Год", "Месяц", "Квартал", "День_недели"]}, "label": "Извлечь дату-признак"},
        ],
        "icon": "📅", "color": "#fd7e14"
    },
    "🔗 Агрегация по группам": {
        "description": "Группирует по первой категориальной колонке, считает сумму числовых",
        "steps": [
            {"type": "📁 Группировка и агрегация", "params": {"group_cols": [], "agg_cols": [], "agg_func": "sum"}, "label": "Группировка и агрегация"},
        ],
        "icon": "🔗", "color": "#20c997"
    },
    "🎯 Фильтр + топ": {
        "description": "Оставляет только непустые строки, берёт топ 50",
        "steps": [
            {"type": "🩹 Заполнить пропуски", "params": {"col": "Все числовые", "method": "Медиана", "value": None}, "label": "Заполнить пропуски"},
            {"type": "🔢 Верхние N строк", "params": {"n": 50}, "label": "Верхние N строк"},
        ],
        "icon": "🎯", "color": "#dc3545"
    },
}

# ---------------------------------------------------------------------------
# Step descriptions
# ---------------------------------------------------------------------------
STEP_DESCRIPTIONS = {
    "🔍 Фильтр строк": "Оставляет только строки, удовлетворяющие условию фильтра.",
    "✂️ Выбрать столбцы": "Выбирает только указанные колонки (остальные удаляются).",
    "🗑️ Удалить столбцы": "Удаляет указанные колонки из датасета.",
    "➕ Вычисляемый столбец": "Создаёт новую колонку на основе формулы или условия.",
    "🔤 Переименовать столбец": "Переименовывает колонку.",
    "📊 Сортировка": "Сортирует строки по значению колонки (по возрастанию или убыванию).",
    "🔢 Верхние N строк": "Оставляет только первые N строк после сортировки.",
    "🧹 Удалить дубликаты": "Удаляет полностью дублирующиеся строки.",
    "🩹 Заполнить пропуски": "Заполняет пропущенные значения (медиана, среднее, константа).",
    "📅 Извлечь дату-признак": "Извлекает из колонки с датой отдельные признаки: год, месяц, день недели и т.д.",
    "📁 Группировка и агрегация": "Группирует по колонке и считает агрегаты (сумма, среднее, кол-во).",
}

STEP_EXAMPLES = {
    "🔍 Фильтр строк": "Пример: фильтр age > 25 оставит только строки где возраст больше 25.",
    "✂️ Выбрать столбцы": "Пример: выбрать [name, age, salary] — удалит все остальные колонки.",
    "🗑️ Удалить столбцы": "Пример: удалить [id, temp] — уберёт технические столбцы.",
    "➕ Вычисляемый столбец": "Пример: revenue = price * quantity вычислит выручку.",
    "🔤 Переименовать столбец": "Пример: переименовать 'dt' → 'date' для читаемости.",
    "📊 Сортировка": "Пример: сортировка по 'revenue' по убыванию покажет топ-клиентов первыми.",
    "🔢 Верхние N строк": "Пример: топ-1000 строк для работы с выборкой данных.",
    "🧹 Удалить дубликаты": "Пример: удалить дубли по [order_id] оставит уникальные заказы.",
    "🩹 Заполнить пропуски": "Пример: заполнить пропуски медианой сохранит распределение данных.",
    "📅 Извлечь дату-признак": "Пример: из 'created_at' извлечь Год, Месяц для временного анализа.",
    "📁 Группировка и агрегация": "Пример: группировка по 'region', сумма 'sales' — продажи по регионам.",
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.caption(
    "Составьте последовательность операций над данными, сохраните как шаблон "
    "и применяйте к любому новому датасету в один клик. Аналог макросов Excel или Python-скрипта."
)

# ---------------------------------------------------------------------------
# Sidebar statistics card
# ---------------------------------------------------------------------------
with st.sidebar:
    n_steps = len(st.session_state["current_pipeline"])
    n_saved = len(st.session_state.get("pipelines", {}))
    n_runs = st.session_state.get("pipeline_run_count", 0)
    st.markdown(
        f"""
        <div style="background:#f0f4ff;border:1px solid #c7d7ff;border-radius:8px;padding:12px 16px;margin-bottom:16px">
            <b>📊 Пайплайны</b><br>
            <span style="color:#555">Шагов в текущем:</span> <b>{n_steps}</b><br>
            <span style="color:#555">Сохранённых:</span> <b>{n_saved}</b><br>
            <span style="color:#555">Запусков:</span> <b>{n_runs}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_templates, tab_build, tab_run, tab_manage = st.tabs([
    "📚 Шаблоны",
    "🔧 Построение",
    "▶ Запуск",
    "💾 Управление",
])

# ===========================================================================
# TAB 1 — Templates
# ===========================================================================
with tab_templates:
    section_header("Готовые шаблоны пайплайнов")
    st.caption("Выберите шаблон, чтобы быстро загрузить набор шагов в текущий пайплайн.")

    template_names = list(PIPELINE_TEMPLATES.keys())
    # Render 2 per row
    for row_start in range(0, len(template_names), 2):
        cols = st.columns(2, gap="medium")
        for col_idx, col in enumerate(cols):
            tpl_idx = row_start + col_idx
            if tpl_idx >= len(template_names):
                break
            tname = template_names[tpl_idx]
            tdata = PIPELINE_TEMPLATES[tname]
            with col:
                n_tpl_steps = len(tdata["steps"])
                st.markdown(
                    f"""
                    <div style="border:1.5px solid {tdata['color']};border-radius:10px;
                                padding:16px 18px;margin-bottom:4px;background:#fafbfc">
                        <div style="font-size:1.5rem;margin-bottom:4px">{tdata['icon']}</div>
                        <b style="font-size:1rem">{tname}</b>
                        <span style="background:{tdata['color']};color:white;border-radius:12px;
                                     padding:1px 8px;font-size:0.75rem;margin-left:8px">
                            {n_tpl_steps} шаг{'а' if 1 < n_tpl_steps < 5 else ('ов' if n_tpl_steps >= 5 else '')}
                        </span>
                        <p style="color:#555;font-size:0.88rem;margin-top:8px;margin-bottom:12px">
                            {tdata['description']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"⚡ Применить", key=f"tpl_apply_{tpl_idx}"):
                    st.session_state["current_pipeline"] = [s.copy() for s in tdata["steps"]]
                    st.success(f"Шаблон «{tname}» загружен ({n_tpl_steps} шагов). Перейдите на вкладку 🔧 Построение.")

# ===========================================================================
# TAB 2 — Build
# ===========================================================================
with tab_build:
    left, right = st.columns([1, 1], gap="large")

    # --- LEFT: dataset + step builder ---
    with left:
        section_header("1. Выберите датасет")
        ds_name = dataset_selectbox("Датасет для построения пайплайна", key="pipe_ds")

        df = get_active_df() if ds_name else None

        if df is not None:
            st.success(f"**{ds_name}**: {df.shape[0]:,} × {df.shape[1]}")

        st.divider()
        section_header("2. Добавьте шаг")

        step_type = st.selectbox(
            "Тип операции:",
            [
                "🔍 Фильтр строк",
                "✂️ Выбрать столбцы",
                "🗑️ Удалить столбцы",
                "➕ Вычисляемый столбец",
                "🔤 Переименовать столбец",
                "📊 Сортировка",
                "🔢 Верхние N строк",
                "🧹 Удалить дубликаты",
                "🩹 Заполнить пропуски",
                "📅 Извлечь дату-признак",
                "📁 Группировка и агрегация",
            ],
            key="pipe_step_type",
        )

        # Show step description and example
        if step_type in STEP_DESCRIPTIONS:
            st.info(f"ℹ️ {STEP_DESCRIPTIONS[step_type]}")
            st.caption(f"💡 {STEP_EXAMPLES[step_type]}")

        step_params = {}
        all_cols = df.columns.tolist() if df is not None else []
        num_cols = df.select_dtypes(include="number").columns.tolist() if df is not None else []
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist() if df is not None else []
        date_cols = [c for c in all_cols if df is not None and pd.api.types.is_datetime64_any_dtype(df[c])]

        if step_type == "🔍 Фильтр строк":
            col1, col2, col3 = st.columns(3)
            with col1:
                f_col = st.selectbox("Столбец:", all_cols or ["—"], key="pipe_f_col",
                                     help="Столбец, по которому применяется условие фильтра")
            with col2:
                f_op = st.selectbox("Оператор:", [">", "<", ">=", "<=", "==", "!=", "содержит", "не содержит",
                                                   "IS NULL", "IS NOT NULL"], key="pipe_f_op",
                                    help="Условие сравнения для фильтрации строк")
            with col3:
                if f_op not in ("IS NULL", "IS NOT NULL"):
                    f_val = st.text_input("Значение:", key="pipe_f_val",
                                          help="Значение для сравнения (число или строка)")
                else:
                    f_val = ""
            step_params = {"col": f_col, "op": f_op, "val": f_val}

        elif step_type == "✂️ Выбрать столбцы":
            sel_cols = st.multiselect("Оставить столбцы:", all_cols, default=all_cols[:5] if all_cols else [],
                                      key="pipe_sel_cols",
                                      help="Только выбранные столбцы останутся в датасете")
            step_params = {"columns": sel_cols}

        elif step_type == "🗑️ Удалить столбцы":
            drop_cols = st.multiselect("Удалить столбцы:", all_cols, key="pipe_drop_cols",
                                       help="Выбранные столбцы будут удалены из датасета")
            step_params = {"columns": drop_cols}

        elif step_type == "➕ Вычисляемый столбец":
            p1, p2 = st.columns(2)
            with p1:
                new_col_name = st.text_input("Название новой колонки:", key="pipe_calc_name",
                                             help="Имя создаваемого столбца")
            with p2:
                expr_type = st.selectbox("Тип:", ["Арифметика", "IF условие"], key="pipe_calc_type",
                                         help="Арифметика: формула; IF: условная логика")

            if expr_type == "Арифметика":
                c1, c2, c3 = st.columns(3)
                with c1:
                    left_op = st.selectbox("Левый операнд:", ["Константа"] + num_cols, key="pipe_left_op",
                                           help="Колонка или константа слева от оператора")
                    if left_op == "Константа":
                        left_val = st.number_input("Значение:", key="pipe_left_val")
                    else:
                        left_val = left_op
                with c2:
                    operator = st.selectbox("Оператор:", ["+", "-", "*", "/", "**", "%"], key="pipe_arith_op",
                                            help="Арифметическая операция")
                with c3:
                    right_op = st.selectbox("Правый операнд:", ["Константа"] + num_cols, key="pipe_right_op",
                                            help="Колонка или константа справа от оператора")
                    if right_op == "Константа":
                        right_val = st.number_input("Значение:", value=1.0, key="pipe_right_val")
                    else:
                        right_val = right_op
                step_params = {
                    "new_col": new_col_name, "type": "arithmetic",
                    "left": left_val, "op": operator, "right": right_val,
                }
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    if_col = st.selectbox("Если столбец:", all_cols or ["—"], key="pipe_if_col",
                                          help="Столбец для проверки условия")
                with c2:
                    if_op = st.selectbox("Оператор:", [">", "<", ">=", "<=", "==", "!="], key="pipe_if_op",
                                         help="Оператор сравнения")
                with c3:
                    if_val = st.text_input("Значение:", key="pipe_if_val",
                                           help="Значение для сравнения")
                then_val = st.text_input("ТОГДА (значение):", key="pipe_then_val",
                                         help="Значение новой колонки когда условие истинно")
                else_val = st.text_input("ИНАЧЕ (значение):", key="pipe_else_val",
                                         help="Значение новой колонки когда условие ложно")
                step_params = {
                    "new_col": new_col_name, "type": "if",
                    "col": if_col, "op": if_op, "val": if_val,
                    "then": then_val, "else": else_val,
                }

        elif step_type == "🔤 Переименовать столбец":
            p1, p2 = st.columns(2)
            with p1:
                old_name = st.selectbox("Текущее название:", all_cols or ["—"], key="pipe_ren_old",
                                        help="Столбец, который нужно переименовать")
            with p2:
                new_name = st.text_input("Новое название:", key="pipe_ren_new",
                                         help="Новое имя столбца")
            step_params = {"old": old_name, "new": new_name}

        elif step_type == "📊 Сортировка":
            p1, p2 = st.columns(2)
            with p1:
                sort_col = st.selectbox("Сортировать по:", all_cols or ["—"], key="pipe_sort_col",
                                        help="Столбец, по которому выполняется сортировка")
            with p2:
                sort_asc = st.radio("Порядок:", ["↑ По возрастанию", "↓ По убыванию"],
                                    key="pipe_sort_dir", horizontal=True,
                                    help="Направление сортировки") == "↑ По возрастанию"
            step_params = {"col": sort_col, "ascending": sort_asc}

        elif step_type == "🔢 Верхние N строк":
            n_rows = st.number_input("Количество строк:", min_value=1, value=1000, step=100, key="pipe_topn",
                                     help="Максимальное число строк для отбора")
            step_params = {"n": int(n_rows)}

        elif step_type == "🧹 Удалить дубликаты":
            dup_subset = st.multiselect("По столбцам (пусто = все):", all_cols, key="pipe_dup_cols",
                                        help="Дубликаты определяются по выбранным столбцам; если пусто — по всем")
            step_params = {"subset": dup_subset or None}

        elif step_type == "🩹 Заполнить пропуски":
            p1, p2 = st.columns(2)
            with p1:
                fill_col = st.selectbox("Столбец:", ["Все числовые"] + num_cols, key="pipe_fill_col",
                                        help="Столбец с пропусками или все числовые")
            with p2:
                fill_method = st.selectbox("Метод:", ["Среднее", "Медиана", "0", "Вперёд (ffill)", "Назад (bfill)", "Значение"],
                                           key="pipe_fill_method",
                                           help="Способ заполнения пропущенных значений")
            fill_value = None
            if fill_method == "Значение":
                fill_value = st.text_input("Значение:", key="pipe_fill_val",
                                           help="Константа для заполнения пропусков")
            step_params = {"col": fill_col, "method": fill_method, "value": fill_value}

        elif step_type == "📅 Извлечь дату-признак":
            p1, p2 = st.columns(2)
            with p1:
                dt_col = st.selectbox("Столбец с датой:", date_cols or ["—"], key="pipe_dt_col",
                                      help="Колонка типа datetime для извлечения признаков")
            with p2:
                dt_parts = st.multiselect("Признаки:", ["Год", "Квартал", "Месяц", "Неделя", "День", "День_недели"],
                                           default=["Год", "Месяц"], key="pipe_dt_parts",
                                           help="Временные признаки, которые будут добавлены как новые столбцы")
            step_params = {"col": dt_col, "parts": dt_parts}

        elif step_type == "📁 Группировка и агрегация":
            p1, p2 = st.columns(2)
            with p1:
                grp_cols = st.multiselect("Группировать по:", all_cols, key="pipe_grp_cols",
                                          help="Столбцы для группировки (категориальные)")
            with p2:
                agg_cols = st.multiselect("Агрегировать:", num_cols, key="pipe_agg_cols",
                                          help="Числовые столбцы для расчёта агрегатов")
            agg_func = st.selectbox("Функция:", ["sum", "mean", "count", "median", "max", "min"],
                                    key="pipe_agg_func",
                                    help="Агрегирующая функция")
            step_params = {"group_cols": grp_cols, "agg_cols": agg_cols, "agg_func": agg_func}

        # Add step button
        step_label = step_type.split(" ", 1)[1] if " " in step_type else step_type
        if st.button(f"➕ Добавить шаг «{step_label}»", key="btn_add_step"):
            step = {"type": step_type, "params": step_params, "label": step_label}
            st.session_state["current_pipeline"].append(step)
            st.success(f"Шаг добавлен. Всего шагов: {len(st.session_state['current_pipeline'])}")
            st.rerun()

    # --- RIGHT: pipeline view ---
    with right:
        section_header("3. Текущий пайплайн")

        pipeline = st.session_state["current_pipeline"]

        if not pipeline:
            st.info("Пайплайн пуст. Добавьте шаги слева или выберите шаблон на вкладке 📚 Шаблоны.")
        else:
            for i, step in enumerate(pipeline):
                step_col1, step_col2 = st.columns([4, 1])
                with step_col1:
                    params_str = ", ".join(f"{k}={v}" for k, v in step["params"].items()
                                           if v is not None and v != [] and v != "")
                    st.markdown(
                        f"<div style='background:#f8f9fa;border-left:3px solid #3182ce;"
                        f"padding:8px 12px;border-radius:4px;margin-bottom:6px'>"
                        f"<b>{i + 1}. {step['type']}</b><br>"
                        f"<small style='color:#666'>{params_str[:100]}</small></div>",
                        unsafe_allow_html=True,
                    )
                with step_col2:
                    if st.button("✕", key=f"del_step_{i}", help="Удалить шаг"):
                        st.session_state["current_pipeline"].pop(i)
                        st.rerun()

            if st.button("🗑️ Очистить пайплайн", key="btn_clear_pipe"):
                st.session_state["current_pipeline"] = []
                st.rerun()

        # Share / export pipeline JSON inline
        st.divider()
        if pipeline:
            pipe_name_share = st.session_state.get("pipe_save_name", "Мой пайплайн")
            pipe_json_str = json.dumps(
                {"name": pipe_name_share, "steps": pipeline, "created": datetime.now().isoformat()},
                ensure_ascii=False, indent=2
            )
            with st.expander("📤 Поделиться пайплайном (JSON)"):
                st.caption("Скопируйте JSON ниже, чтобы поделиться пайплайном или сохранить его вручную.")
                st.code(pipe_json_str, language="json")

# ===========================================================================
# TAB 3 — Run
# ===========================================================================
with tab_run:
    section_header("4. Запустить пайплайн")

    pipeline = st.session_state["current_pipeline"]

    run_ds = dataset_selectbox("Применить к датасету:", key="pipe_run_ds")
    run_df = get_active_df()

    if not pipeline:
        st.info("Пайплайн пуст. Добавьте шаги на вкладке 🔧 Построение.")
    elif run_df is None:
        st.warning("Выберите датасет для применения пайплайна.")
    else:
        st.markdown(f"**Шагов в пайплайне:** {len(pipeline)}")
        if st.button("▶ Запустить пайплайн", type="primary", key="btn_run_pipe"):
            result_df = run_df.copy()
            errors = []
            step_log = []
            step_previews = []

            for i, step in enumerate(pipeline):
                rows_before = len(result_df)
                preview_df = None
                try:
                    stype = step["type"]
                    p = step["params"]

                    if stype == "🔍 Фильтр строк":
                        col, op, val = p["col"], p["op"], p.get("val", "")
                        if op == "IS NULL":
                            result_df = result_df[result_df[col].isna()]
                        elif op == "IS NOT NULL":
                            result_df = result_df[result_df[col].notna()]
                        elif op == "содержит":
                            result_df = result_df[result_df[col].astype(str).str.contains(str(val), na=False)]
                        elif op == "не содержит":
                            result_df = result_df[~result_df[col].astype(str).str.contains(str(val), na=False)]
                        else:
                            try:
                                num_val = float(val)
                            except (ValueError, TypeError):
                                num_val = val
                            if op == ">":    result_df = result_df[result_df[col] > num_val]
                            elif op == "<":  result_df = result_df[result_df[col] < num_val]
                            elif op == ">=": result_df = result_df[result_df[col] >= num_val]
                            elif op == "<=": result_df = result_df[result_df[col] <= num_val]
                            elif op == "==": result_df = result_df[result_df[col] == num_val]
                            elif op == "!=": result_df = result_df[result_df[col] != num_val]

                    elif stype == "✂️ Выбрать столбцы":
                        cols = [c for c in p["columns"] if c in result_df.columns]
                        if cols:
                            result_df = result_df[cols]

                    elif stype == "🗑️ Удалить столбцы":
                        cols = [c for c in p["columns"] if c in result_df.columns]
                        result_df = result_df.drop(columns=cols)

                    elif stype == "➕ Вычисляемый столбец":
                        new_col = p["new_col"] or f"col_{i}"
                        if p["type"] == "arithmetic":
                            left_v = result_df[p["left"]] if p["left"] in result_df.columns else float(p["left"])
                            right_v = result_df[p["right"]] if p["right"] in result_df.columns else float(p["right"])
                            op = p["op"]
                            if op == "+":   result_df[new_col] = left_v + right_v
                            elif op == "-": result_df[new_col] = left_v - right_v
                            elif op == "*": result_df[new_col] = left_v * right_v
                            elif op == "/": result_df[new_col] = left_v / right_v.replace(0, float("nan")) if hasattr(right_v, "replace") else left_v / (right_v if right_v != 0 else float("nan"))
                            elif op == "**": result_df[new_col] = left_v ** right_v
                            elif op == "%":  result_df[new_col] = left_v % right_v
                        else:  # IF
                            col, op2, val2 = p["col"], p["op"], p.get("val", "")
                            try:
                                num_v = float(val2)
                            except (ValueError, TypeError):
                                num_v = val2
                            if op2 == ">":   mask = result_df[col] > num_v
                            elif op2 == "<": mask = result_df[col] < num_v
                            elif op2 == ">=": mask = result_df[col] >= num_v
                            elif op2 == "<=": mask = result_df[col] <= num_v
                            elif op2 == "==": mask = result_df[col] == num_v
                            else:             mask = result_df[col] != num_v
                            then_v = p.get("then", "1")
                            else_v = p.get("else", "0")
                            try:
                                then_v, else_v = float(then_v), float(else_v)
                            except (ValueError, TypeError):
                                pass
                            result_df[new_col] = then_v
                            result_df.loc[~mask, new_col] = else_v

                    elif stype == "🔤 Переименовать столбец":
                        if p["old"] in result_df.columns and p["new"]:
                            result_df = result_df.rename(columns={p["old"]: p["new"]})

                    elif stype == "📊 Сортировка":
                        if p["col"] in result_df.columns:
                            result_df = result_df.sort_values(p["col"], ascending=p["ascending"])

                    elif stype == "🔢 Верхние N строк":
                        result_df = result_df.head(p["n"])

                    elif stype == "🧹 Удалить дубликаты":
                        subset = p.get("subset") or None
                        result_df = result_df.drop_duplicates(subset=subset)

                    elif stype == "🩹 Заполнить пропуски":
                        target_cols = result_df.select_dtypes(include="number").columns if p["col"] == "Все числовые" else [p["col"]]
                        for tc in target_cols:
                            if tc not in result_df.columns:
                                continue
                            m = p["method"]
                            if m == "Среднее":          result_df[tc] = result_df[tc].fillna(result_df[tc].mean())
                            elif m == "Медиана":        result_df[tc] = result_df[tc].fillna(result_df[tc].median())
                            elif m == "0":              result_df[tc] = result_df[tc].fillna(0)
                            elif m == "Вперёд (ffill)": result_df[tc] = result_df[tc].ffill()
                            elif m == "Назад (bfill)":  result_df[tc] = result_df[tc].bfill()
                            elif m == "Значение" and p.get("value") is not None:
                                try:
                                    result_df[tc] = result_df[tc].fillna(float(p["value"]))
                                except (ValueError, TypeError):
                                    result_df[tc] = result_df[tc].fillna(p["value"])

                    elif stype == "📅 Извлечь дату-признак":
                        col = p["col"]
                        if col not in result_df.columns:
                            # try to find first datetime column (handles placeholder "—")
                            dt_cols = result_df.select_dtypes(include=["datetime64"]).columns.tolist()
                            if dt_cols:
                                col = dt_cols[0]
                            else:
                                raise ValueError(f"Не найдена колонка с датами: '{p['col']}'")
                        if col in result_df.columns:
                            dt = result_df[col]
                            part_map = {
                                "Год": ("_год", dt.dt.year),
                                "Квартал": ("_квартал", dt.dt.quarter),
                                "Месяц": ("_месяц", dt.dt.month),
                                "Неделя": ("_неделя", dt.dt.isocalendar().week.astype(int)),
                                "День": ("_день", dt.dt.day),
                                "День_недели": ("_день_нед", dt.dt.dayofweek),
                            }
                            for part in p.get("parts", []):
                                if part in part_map:
                                    suffix, values = part_map[part]
                                    result_df[col + suffix] = values

                    elif stype == "📁 Группировка и агрегация":
                        grp = [c for c in p["group_cols"] if c in result_df.columns]
                        agg = [c for c in p["agg_cols"] if c in result_df.columns]
                        if grp and agg:
                            result_df = result_df.groupby(grp)[agg].agg(p["agg_func"]).reset_index()

                    rows_after = len(result_df)
                    preview_df = result_df.head(3).copy()
                    step_log.append({"step": i + 1, "op": step["label"],
                                     "rows_before": rows_before, "rows_after": rows_after,
                                     "status": "✅ OK"})
                    step_previews.append({
                        "label": step["label"],
                        "status": "✅ OK",
                        "rows_before": rows_before,
                        "rows_after": rows_after,
                        "preview": preview_df,
                    })
                except Exception as e:
                    rows_after = len(result_df)
                    errors.append(f"Шаг {i + 1} ({step['label']}): {e}")
                    step_log.append({"step": i + 1, "op": step["label"],
                                     "rows_before": rows_before, "rows_after": rows_after,
                                     "status": f"❌ {e}"})
                    step_previews.append({
                        "label": step["label"],
                        "status": f"❌ {e}",
                        "rows_before": rows_before,
                        "rows_after": rows_after,
                        "preview": None,
                    })

            # Store step previews
            st.session_state["pipeline_step_previews"] = step_previews

            # Increment run counter
            st.session_state["pipeline_run_count"] = st.session_state.get("pipeline_run_count", 0) + 1

            # Show results
            st.success(f"Пайплайн выполнен: {len(result_df):,} строк × {result_df.shape[1]} столбцов")

            if errors:
                for err in errors:
                    st.warning(f"⚠️ {err}")

            # Step-by-step expanders with preview
            st.markdown("**Детали выполнения по шагам:**")
            for sp in step_previews:
                delta = sp["rows_after"] - sp["rows_before"]
                delta_color = "green" if delta >= 0 else "red"
                delta_str = f"<span style='color:{delta_color}'>{'+' if delta >= 0 else ''}{delta:,}</span>"
                with st.expander(f"{sp['status']} Шаг: {sp['label']}  —  {sp['rows_before']:,} → {sp['rows_after']:,} строк ({delta_str})", expanded=False):
                    st.markdown(
                        f"Строк до: **{sp['rows_before']:,}** &nbsp;→&nbsp; Строк после: **{sp['rows_after']:,}** &nbsp;({delta_str})",
                        unsafe_allow_html=True,
                    )
                    if sp["preview"] is not None and not sp["preview"].empty:
                        st.caption("Первые 3 строки результата:")
                        st.dataframe(sp["preview"], use_container_width=True)

            # Full step log table
            with st.expander("📋 Лог выполнения (таблица)", expanded=False):
                st.dataframe(pd.DataFrame(step_log), use_container_width=True)

            # Preview full result
            st.dataframe(result_df.head(20), use_container_width=True)

            # Save result
            result_name = f"{run_ds}_pipeline"
            save_col1, save_col2 = st.columns(2)
            with save_col1:
                if st.button("💾 Сохранить как новый датасет", key="btn_save_pipe_result"):
                    add_dataset(result_name, result_df.reset_index(drop=True))
                    st.success(f"Сохранено: «{result_name}»")
            with save_col2:
                csv_bytes = result_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button("📥 Скачать (CSV)", data=csv_bytes,
                                   file_name=f"{result_name}.csv", mime="text/csv")

            log_event("analysis_run", {"type": "pipeline", "dataset": run_ds,
                                       "steps": len(pipeline), "errors": len(errors)})

    # Show cached step previews if run already happened
    cached_previews = st.session_state.get("pipeline_step_previews", [])
    if cached_previews and run_df is not None and pipeline:
        # Only show if we haven't just run (avoid double display)
        pass

# ===========================================================================
# TAB 4 — Manage
# ===========================================================================
with tab_manage:
    section_header("5. Управление пайплайнами")

    pipeline = st.session_state["current_pipeline"]

    pipe_mgmt_col1, pipe_mgmt_col2 = st.columns(2)

    with pipe_mgmt_col1:
        pipe_name = st.text_input("Название пайплайна:", value="Мой пайплайн", key="pipe_save_name",
                                  help="Имя для сохранения пайплайна в библиотеке")
        if st.button("💾 Сохранить пайплайн", key="btn_save_pipe"):
            if pipeline:
                st.session_state["pipelines"][pipe_name] = {
                    "steps": pipeline.copy(),
                    "created": datetime.now().isoformat(),
                    "n_steps": len(pipeline),
                }
                st.success(f"Пайплайн «{pipe_name}» сохранён.")
            else:
                st.warning("Пайплайн пуст.")

        # Export as JSON download
        if pipeline:
            pipe_json = json.dumps(
                {"name": pipe_name, "steps": pipeline, "created": datetime.now().isoformat()},
                ensure_ascii=False, indent=2
            ).encode("utf-8")
            st.download_button(
                "📤 Экспорт пайплайна (JSON)",
                data=pipe_json,
                file_name=f"{pipe_name}.json",
                mime="application/json",
                key="btn_export_pipe",
            )

    with pipe_mgmt_col2:
        # Load saved pipeline
        saved = st.session_state.get("pipelines", {})
        if saved:
            st.markdown("**Сохранённые пайплайны:**")
            for pname, pdata in saved.items():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"📋 **{pname}** — {pdata['n_steps']} шагов")
                with col_b:
                    if st.button("Загрузить", key=f"load_pipe_{pname}"):
                        st.session_state["current_pipeline"] = pdata["steps"].copy()
                        st.success(f"Пайплайн «{pname}» загружен.")
                        st.rerun()
        else:
            st.info("Нет сохранённых пайплайнов.")

        # Import from JSON
        uploaded_pipe = st.file_uploader("📥 Импорт пайплайна (JSON):", type=["json"], key="pipe_import")
        if uploaded_pipe:
            try:
                data = json.load(uploaded_pipe)
                if "steps" in data:
                    name = data.get("name", "Импортированный")
                    st.session_state["pipelines"][name] = {
                        "steps": data["steps"],
                        "created": data.get("created", ""),
                        "n_steps": len(data["steps"]),
                    }
                    st.success(f"Пайплайн «{name}» импортирован с {len(data['steps'])} шагами.")
                    st.rerun()
                else:
                    st.error("Неверный формат файла.")
            except Exception as e:
                st.error(f"Ошибка импорта: {e}")
