"""
pages/1_Data.py – Dataset ingestion, catalog, schema preview, profiling.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.state import init_state, store_dataset, list_dataset_names, get_active_df
from core.data import load_file, profile_dataframe, describe_numeric, infer_column_types
from core.i18n import t
from core.audit import log_event
from core.autoqc import check_upload
from app.components.ux import data_quality_banner
from app.styles import inject_all_css, page_header, section_header

st.set_page_config(page_title="KIBAD – Данные", layout="wide")
init_state()
inject_all_css()

page_header("1. Данные", "Загрузка и подключение источников данных", "📥",
            next_page="pages/2_Prepare.py", next_label="Подготовка данных")

# ---------------------------------------------------------------------------
# Upload tab
# ---------------------------------------------------------------------------
tab_upload, tab_postgres, tab_catalog = st.tabs([
    t("upload_files"), t("page_data_db_section"), t("page_data_schema"),
])

with tab_upload:
    section_header(t("page_data_upload_label"), "📂")
    uploaded_files = st.file_uploader(
        t("upload_files"), type=["csv", "xlsx", "xls"],
        accept_multiple_files=True, key="file_uploader",
        help="Поддерживаются форматы CSV, Excel (xlsx, xls). Можно загрузить несколько файлов одновременно.",
    )
    if uploaded_files:
        sep = st.selectbox(
            t("page_data_separator"), [",", ";", "\t", "|"], index=0, key="csv_sep",
            help="Разделитель столбцов в CSV-файле. Для Excel не используется.",
        )

        if st.button(t("apply"), key="btn_load_files"):
            for uploaded in uploaded_files:
                ds_name = uploaded.name.rsplit(".", 1)[0]
                with st.spinner(f"Загрузка {uploaded.name}..."):
                    try:
                        raw_bytes = uploaded.getvalue()
                        df = load_file(raw_bytes, filename=uploaded.name, sep=sep)
                        store_dataset(ds_name, df, source="upload")
                        qc = check_upload(df)
                        if "data_quality_reports" not in st.session_state:
                            st.session_state["data_quality_reports"] = {}
                        st.session_state["data_quality_reports"][ds_name] = qc
                        log_event("file_loaded", {
                            "filename": uploaded.name,
                            "dataset": ds_name,
                            "rows": df.shape[0],
                            "columns": df.shape[1],
                        })
                        st.success(f"✅ Датасет «{ds_name}» загружен: {len(df):,} строк × {len(df.columns)} колонок")
                        with st.expander("👁️ Предварительный просмотр данных", expanded=True):
                            preview_c1, preview_c2, preview_c3, preview_c4 = st.columns(4)
                            preview_c1.metric("Строк", f"{len(df):,}")
                            preview_c2.metric("Столбцов", len(df.columns))
                            preview_c3.metric("Числовых", len(df.select_dtypes(include='number').columns))
                            preview_c4.metric("Пропусков", f"{df.isnull().sum().sum():,}")
                            st.dataframe(df.head(10), use_container_width=True)
                            # Column schema
                            schema_df = pd.DataFrame({
                                "Колонка": df.columns,
                                "Тип": df.dtypes.astype(str).values,
                                "Заполнено %": (df.notna().mean() * 100).round(1).values,
                                "Уникальных": df.nunique().values,
                                "Пример": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—" for c in df.columns],
                            })
                            st.markdown("**Схема данных:**")
                            st.dataframe(schema_df, use_container_width=True, hide_index=True)
                        data_quality_banner(qc)
                    except Exception as e:
                        st.error(f"{uploaded.name}: {e}")

        # Show QC for already loaded datasets
        qc_reports = st.session_state.get("data_quality_reports", {})
        if qc_reports:
            with st.expander("🔍 Статус качества данных", expanded=False):
                for ds, qc in qc_reports.items():
                    sev_icon = "✅" if qc["severity"] == "ok" else ("⚠️" if qc["severity"] == "warning" else "🔴")
                    st.markdown(f"{sev_icon} **{ds}**: {qc['n_rows']:,} строк, {qc['overall_missing_pct']:.1f}% пропусков, {qc['duplicate_rows']} дубликатов")

# ---------------------------------------------------------------------------
# PostgreSQL / NoCode query builder tab
# ---------------------------------------------------------------------------
with tab_postgres:
    section_header("Подключение к базе данных", "🗄️")

    # ── helpers stored in session_state ──────────────────────────────────────
    def _ss(key, default=None):
        return st.session_state.setdefault(key, default)

    _ss("db_connected", False)
    _ss("db_conn_params", {})
    _ss("db_schemas", [])
    _ss("db_tables", [])
    _ss("db_columns", [])          # list[dict]: name, data_type, is_nullable
    _ss("db_filters", [])          # list[dict]: col, op, val, connector
    _ss("db_order_by", [])         # list[dict]: col, direction
    _ss("db_group_by", [])         # list[str]
    _ss("db_computed_cols", [])    # list[dict]: expr, alias
    _ss("db_joins", [])            # list[dict]: type, schema, table, on_left, on_right
    _ss("db_aggregations", [])     # list[dict]: func, col, alias
    _ss("db_having", "")           # str
    _ss("db_saved_queries", {})    # dict: name -> {sql, table, saved_at}

    # ── STEP 1 – Connection parameters ───────────────────────────────────────
    with st.expander("🔌 Параметры подключения", expanded=not st.session_state["db_connected"]):
        conn_col1, conn_col2, conn_col3 = st.columns([3, 1, 1])
        conn_host = conn_col1.text_input("Хост", value=st.session_state["db_conn_params"].get("host", "localhost"), key="db_host", help="Адрес PostgreSQL-сервера, например: localhost или 192.168.1.10")
        conn_port = conn_col2.text_input("Порт", value=st.session_state["db_conn_params"].get("port", "5432"), key="db_port", help="Порт PostgreSQL (по умолчанию 5432)")
        conn_db = conn_col3.text_input("База данных", value=st.session_state["db_conn_params"].get("database", ""), key="db_database", help="Имя базы данных для подключения")

        auth_col1, auth_col2 = st.columns(2)
        conn_user = auth_col1.text_input("Пользователь", value=st.session_state["db_conn_params"].get("user", ""), key="db_user", help="Имя пользователя PostgreSQL")
        conn_pass = auth_col2.text_input("Пароль", type="password", key="db_password", help="Пароль пользователя (хранится только в текущей сессии)")

        connect_btn, disconnect_btn, _ = st.columns([2, 2, 6])
        if connect_btn.button("🔗 Подключиться", type="primary", key="btn_db_connect"):
            if not all([conn_host, conn_port, conn_db, conn_user]):
                st.warning("Заполните хост, порт, базу данных и пользователя.")
            else:
                with st.spinner("Подключение..."):
                    try:
                        from services.db import test_connection, list_schemas
                        ok, msg = test_connection(conn_host, conn_port, conn_db, conn_user, conn_pass)
                        if ok:
                            params = {"host": conn_host, "port": conn_port, "database": conn_db,
                                      "user": conn_user, "password": conn_pass}
                            st.session_state["db_conn_params"] = params
                            st.session_state["db_connected"] = True
                            schemas = list_schemas(**params)
                            st.session_state["db_schemas"] = schemas
                            # Reset downstream state
                            st.session_state["db_tables"] = []
                            st.session_state["db_columns"] = []
                            st.session_state["db_filters"] = []
                            st.session_state["db_order_by"] = []
                            st.session_state["db_group_by"] = []
                            st.success(f"✅ Подключено к **{conn_db}** на {conn_host}:{conn_port}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
                    except Exception as exc:
                        st.error(f"Ошибка: {exc}")

        if disconnect_btn.button("⛔ Отключиться", key="btn_db_disconnect",
                                 disabled=not st.session_state["db_connected"]):
            st.session_state["db_connected"] = False
            st.session_state["db_conn_params"] = {}
            st.session_state["db_schemas"] = []
            st.session_state["db_tables"] = []
            st.session_state["db_columns"] = []
            st.rerun()

    if not st.session_state["db_connected"]:
        st.info("💡 Заполните параметры подключения и нажмите **Подключиться**.")
        st.stop()

    params = st.session_state["db_conn_params"]
    st.success(f"✅ Подключено: **{params.get('database')}** @ {params.get('host')}:{params.get('port')} — пользователь: **{params.get('user')}**")

    # ── STEP 2 – Schema & Table selection ────────────────────────────────────
    st.divider()
    section_header("Шаг 1 — Выбор таблицы", "📁")

    schema_col, table_col = st.columns([1, 2])

    schemas = st.session_state["db_schemas"] or ["public"]
    selected_schema = schema_col.selectbox("Схема", schemas, key="db_selected_schema")

    # Load tables when schema changes
    if st.session_state.get("_last_schema") != selected_schema:
        from services.db import list_tables
        with st.spinner("Загрузка таблиц..."):
            tables = list_tables(**params, schema=selected_schema)
        st.session_state["db_tables"] = tables
        st.session_state["_last_schema"] = selected_schema
        st.session_state["db_columns"] = []

    db_tables = st.session_state["db_tables"]
    if not db_tables:
        table_col.warning(f"В схеме **{selected_schema}** нет таблиц.")
        st.stop()

    selected_table = table_col.selectbox("Таблица", db_tables, key="db_selected_table")

    # Load columns when table changes
    table_key = f"{selected_schema}.{selected_table}"
    if st.session_state.get("_last_table") != table_key:
        from services.db import get_table_columns, get_row_count
        with st.spinner("Загрузка структуры таблицы..."):
            columns_info = get_table_columns(**params, table=selected_table, schema=selected_schema)
            row_count = get_row_count(**params, table=selected_table, schema=selected_schema)
        st.session_state["db_columns"] = columns_info
        st.session_state["db_row_count"] = row_count
        st.session_state["_last_table"] = table_key
        st.session_state["db_filters"] = []
        st.session_state["db_order_by"] = []
        st.session_state["db_group_by"] = []

    columns_info = st.session_state["db_columns"]
    if not columns_info:
        st.warning("Не удалось загрузить структуру таблицы.")
        st.stop()

    row_count = st.session_state.get("db_row_count")
    if row_count is not None:
        st.caption(f"Таблица **{selected_schema}.{selected_table}** — приблизительно **{row_count:,}** строк, **{len(columns_info)}** колонок")

    # ── STEP 3 – Column selection ─────────────────────────────────────────────
    st.divider()
    section_header("Шаг 2 — Выбор колонок", "📋")

    col_names = [c["column_name"] for c in columns_info]
    col_types = {c["column_name"]: c["data_type"] for c in columns_info}

    # Type badge helper
    def _type_badge(dtype: str) -> str:
        dtype_lower = dtype.lower()
        if any(t in dtype_lower for t in ("int", "serial", "numeric", "float", "real", "double", "decimal", "money")):
            return "🔢"
        if any(t in dtype_lower for t in ("char", "text", "varchar", "uuid")):
            return "🔤"
        if any(t in dtype_lower for t in ("date", "time", "timestamp", "interval")):
            return "📅"
        if "bool" in dtype_lower:
            return "☑️"
        return "📦"

    use_distinct = st.checkbox(
        "DISTINCT — убрать дубликаты строк",
        value=False,
        key="db_distinct",
        help="SELECT DISTINCT возвращает только уникальные строки. Используйте, когда одна строка может встречаться несколько раз из-за структуры таблицы.",
    )

    sel_all_col, _ = st.columns([3, 7])
    if sel_all_col.checkbox(
        "Выбрать все колонки",
        value=True,
        key="db_select_all_cols",
        help="Если включено — выбираются все колонки таблицы (SELECT *). Отключите, чтобы выбрать только нужные.",
    ):
        default_cols = col_names
    else:
        default_cols = col_names[:min(5, len(col_names))]

    # Build multiselect options with type icons
    col_labels = {c: f"{_type_badge(col_types[c])} {c} ({col_types[c]})" for c in col_names}

    selected_col_labels = st.multiselect(
        "Колонки для выборки",
        options=col_names,
        default=default_cols,
        format_func=lambda c: col_labels[c],
        key="db_selected_columns",
        help="Выберите колонки, которые войдут в результат запроса. Порядок соответствует порядку в таблице.",
    )

    if not selected_col_labels:
        st.warning("Выберите хотя бы одну колонку.")
        st.stop()

    # ── STEP 2б – Computed columns ────────────────────────────────────────────
    st.divider()
    section_header("Шаг 2б — Вычисляемые выражения (необязательно)", "🧮")
    st.caption("Создавайте новые колонки через SQL-выражения прямо в запросе — без Python.")

    computed_cols = st.session_state["db_computed_cols"]

    add_comp_col, clear_comp_col, _ = st.columns([2, 2, 6])
    if add_comp_col.button(
        "➕ Добавить выражение",
        key="btn_add_computed",
        help="Добавить новое вычисляемое выражение в SELECT.",
    ):
        computed_cols.append({"expr": "", "alias": ""})
        st.session_state["db_computed_cols"] = computed_cols
        st.rerun()

    if clear_comp_col.button(
        "🗑️ Очистить",
        key="btn_clear_computed",
        disabled=len(computed_cols) == 0,
        help="Удалить все вычисляемые выражения.",
    ):
        st.session_state["db_computed_cols"] = []
        st.rerun()

    if computed_cols:
        with st.expander("💡 Примеры выражений"):
            st.markdown("""
| Выражение | Результат |
|-----------|-----------|
| `amount * 0.2` | 20% от суммы |
| `COALESCE(col, 0)` | Замена NULL на 0 |
| `UPPER(name)` | Текст в верхний регистр |
| `date_trunc('month', created_at)` | Округление даты до месяца |
| `CASE WHEN amount > 1000 THEN 'высокий' ELSE 'низкий' END` | Условная метка |
| `CONCAT(first_name, ' ', last_name)` | Объединение строк |
| `EXTRACT(YEAR FROM created_at)` | Год из даты |
| `amount / SUM(amount) OVER ()` | Доля от общей суммы |
""")

    comp_to_remove = []
    for idx, comp in enumerate(computed_cols):
        cc1, cc2, cc3 = st.columns([4, 2, 1])
        comp["expr"] = cc1.text_input(
            "SQL-выражение",
            value=comp.get("expr", ""),
            key=f"comp_expr_{idx}",
            placeholder="Например: amount * 0.18 или UPPER(city)",
            help="Любое SQL-выражение: арифметика, функции (COALESCE, CASE WHEN, CONCAT), агрегаты (SUM, COUNT), оконные функции (ROW_NUMBER() OVER ()).",
            label_visibility="collapsed" if idx > 0 else "visible",
        )
        comp["alias"] = cc2.text_input(
            "Псевдоним (AS)",
            value=comp.get("alias", ""),
            key=f"comp_alias_{idx}",
            placeholder="Имя колонки в результате",
            help="Имя, под которым результат выражения появится в таблице. Обязательно, если выражение содержит пробелы или функции.",
            label_visibility="collapsed" if idx > 0 else "visible",
        )
        if cc3.button("✕", key=f"comp_del_{idx}", help="Удалить выражение"):
            comp_to_remove.append(idx)

    if comp_to_remove:
        st.session_state["db_computed_cols"] = [c for i, c in enumerate(computed_cols) if i not in comp_to_remove]
        st.rerun()

    # ── STEP 2в – JOIN builder ────────────────────────────────────────────────
    st.divider()
    section_header("Шаг 2в — Объединение таблиц JOIN (необязательно)", "🔗")
    st.caption("Присоедините данные из другой таблицы по общему ключу.")

    joins = st.session_state["db_joins"]

    add_join_col, clear_join_col, _ = st.columns([2, 2, 6])
    if add_join_col.button(
        "➕ Добавить JOIN",
        key="btn_add_join",
        help="Добавить соединение с другой таблицей.",
    ):
        joins.append({"type": "INNER JOIN", "table": "", "schema": selected_schema, "on_left": col_names[0] if col_names else "", "on_right": ""})
        st.session_state["db_joins"] = joins
        st.rerun()

    if clear_join_col.button(
        "🗑️ Очистить JOIN",
        key="btn_clear_joins",
        disabled=len(joins) == 0,
        help="Удалить все JOIN-соединения.",
    ):
        st.session_state["db_joins"] = []
        st.rerun()

    JOIN_TYPES = {
        "INNER JOIN": "Только совпадающие строки в обеих таблицах",
        "LEFT JOIN": "Все строки из левой + совпадающие из правой (NULL если нет)",
        "RIGHT JOIN": "Все строки из правой + совпадающие из левой (NULL если нет)",
        "FULL OUTER JOIN": "Все строки из обеих таблиц",
    }

    joins_to_remove = []
    for idx, j in enumerate(joins):
        jc1, jc2, jc3, jc4, jc5 = st.columns([2, 2, 2, 2, 1])
        j["type"] = jc1.selectbox(
            "Тип JOIN",
            list(JOIN_TYPES.keys()),
            index=list(JOIN_TYPES.keys()).index(j.get("type", "INNER JOIN")),
            key=f"j_type_{idx}",
            format_func=lambda t: t,
            help="\n".join(f"**{k}** — {v}" for k, v in JOIN_TYPES.items()),
            label_visibility="collapsed" if idx > 0 else "visible",
        )
        j["schema"] = jc2.text_input(
            "Схема",
            value=j.get("schema", selected_schema),
            key=f"j_schema_{idx}",
            help="Схема присоединяемой таблицы (обычно public).",
            label_visibility="collapsed" if idx > 0 else "visible",
        )
        j["table"] = jc2.text_input(
            "Таблица для JOIN",
            value=j.get("table", ""),
            key=f"j_table_{idx}",
            placeholder="Имя таблицы",
            help="Имя таблицы, которую хотите присоединить.",
        )
        j["on_left"] = jc3.text_input(
            "Ключ из основной таблицы",
            value=j.get("on_left", col_names[0] if col_names else ""),
            key=f"j_on_l_{idx}",
            placeholder="колонка-ключ",
            help="Колонка из основной таблицы для условия соединения (ON).",
            label_visibility="collapsed" if idx > 0 else "visible",
        )
        j["on_right"] = jc4.text_input(
            "Ключ из присоединяемой таблицы",
            value=j.get("on_right", ""),
            key=f"j_on_r_{idx}",
            placeholder="колонка-ключ",
            help="Колонка из присоединяемой таблицы для условия соединения (ON).",
            label_visibility="collapsed" if idx > 0 else "visible",
        )
        if jc5.button("✕", key=f"j_del_{idx}", help="Удалить JOIN"):
            joins_to_remove.append(idx)

    if joins_to_remove:
        st.session_state["db_joins"] = [j for i, j in enumerate(joins) if i not in joins_to_remove]
        st.rerun()

    # ── STEP 4 – Filter builder ───────────────────────────────────────────────
    st.divider()
    section_header("Шаг 3 — Фильтры (WHERE)", "🔍")

    # Operators grouped by type
    _text_ops = ["=", "!=", "LIKE", "NOT LIKE", "ILIKE", "IS NULL", "IS NOT NULL", "IN", "NOT IN"]
    _num_ops  = ["=", "!=", "<", "<=", ">", ">=", "IS NULL", "IS NOT NULL", "BETWEEN"]
    _date_ops = ["=", "<", "<=", ">", ">=", "IS NULL", "IS NOT NULL"]

    def _ops_for(colname: str) -> list[str]:
        dtype = col_types.get(colname, "text").lower()
        if any(t in dtype for t in ("int", "numeric", "float", "real", "double", "decimal", "money", "serial")):
            return _num_ops
        if any(t in dtype for t in ("date", "time", "timestamp")):
            return _date_ops
        return _text_ops

    filters = st.session_state["db_filters"]

    add_filter_col, clear_filter_col, _ = st.columns([2, 2, 6])
    if add_filter_col.button("➕ Добавить фильтр", key="btn_add_filter"):
        filters.append({"col": col_names[0], "op": "=", "val": "", "connector": "AND"})
        st.session_state["db_filters"] = filters
        st.rerun()

    if clear_filter_col.button("🗑️ Очистить фильтры", key="btn_clear_filters",
                               disabled=len(filters) == 0):
        st.session_state["db_filters"] = []
        st.rerun()

    filters_to_remove = []
    for idx, filt in enumerate(filters):
        f_cols = st.columns([1, 2, 2, 3, 1]) if idx == 0 else st.columns([1, 1, 2, 3, 1])

        if idx > 0:
            connector = f_cols[0].selectbox(
                "Связка", ["AND", "OR"], index=0 if filt.get("connector") == "AND" else 1,
                key=f"f_conn_{idx}", label_visibility="collapsed",
            )
            filt["connector"] = connector

        col_offset = 1 if idx > 0 else 0
        filt["col"] = f_cols[col_offset].selectbox(
            "Колонка", col_names, index=col_names.index(filt["col"]) if filt["col"] in col_names else 0,
            key=f"f_col_{idx}", label_visibility="collapsed" if idx > 0 else "visible",
        )
        ops = _ops_for(filt["col"])
        current_op = filt["op"] if filt["op"] in ops else ops[0]
        filt["op"] = f_cols[col_offset + 1].selectbox(
            "Оператор", ops, index=ops.index(current_op),
            key=f"f_op_{idx}", label_visibility="collapsed" if idx > 0 else "visible",
        )
        if filt["op"] not in ("IS NULL", "IS NOT NULL"):
            filt["val"] = f_cols[col_offset + 2].text_input(
                "Значение", value=str(filt.get("val", "")),
                key=f"f_val_{idx}", label_visibility="collapsed" if idx > 0 else "visible",
                placeholder="IN: значения через запятую" if filt["op"] in ("IN", "NOT IN") else "Значение",
            )
        else:
            f_cols[col_offset + 2].empty()

        if f_cols[-1].button("✕", key=f"f_del_{idx}", help="Удалить фильтр"):
            filters_to_remove.append(idx)

    if filters_to_remove:
        st.session_state["db_filters"] = [f for i, f in enumerate(filters) if i not in filters_to_remove]
        st.rerun()

    # ── STEP 5 – Sort & Group ─────────────────────────────────────────────────
    st.divider()
    sort_col, group_col = st.columns(2)

    with sort_col:
        section_header("Шаг 4 — Сортировка (ORDER BY)", "↕️")
        order_by = st.session_state["db_order_by"]

        if st.button("➕ Добавить сортировку", key="btn_add_order"):
            order_by.append({"col": col_names[0], "direction": "ASC"})
            st.session_state["db_order_by"] = order_by
            st.rerun()

        ob_to_remove = []
        for idx, ob in enumerate(order_by):
            ob_c1, ob_c2, ob_c3 = st.columns([3, 2, 1])
            ob["col"] = ob_c1.selectbox(
                "Колонка", col_names, index=col_names.index(ob["col"]) if ob["col"] in col_names else 0,
                key=f"ob_col_{idx}", label_visibility="collapsed",
            )
            ob["direction"] = ob_c2.selectbox(
                "Направление", ["ASC", "DESC"], index=0 if ob["direction"] == "ASC" else 1,
                key=f"ob_dir_{idx}", label_visibility="collapsed",
            )
            if ob_c3.button("✕", key=f"ob_del_{idx}"):
                ob_to_remove.append(idx)

        if ob_to_remove:
            st.session_state["db_order_by"] = [o for i, o in enumerate(order_by) if i not in ob_to_remove]
            st.rerun()

    with group_col:
        section_header("Шаг 5 — Группировка (GROUP BY)", "📊")
        st.caption("Используйте совместно с агрегационными функциями (COUNT, SUM и т.д.) в именах колонок.")
        gb_selected = st.multiselect(
            "Группировать по",
            options=col_names,
            default=st.session_state["db_group_by"],
            key="db_group_by_sel",
            help="Выберите колонки для группировки. Все остальные колонки в SELECT должны использоваться с агрегационными функциями.",
        )
        st.session_state["db_group_by"] = gb_selected

        if gb_selected:
            st.markdown("**📐 Агрегационные функции**")
            st.caption(
                "Выберите функции для числовых колонок. Они автоматически добавятся в SELECT.",
                help="Агрегации вычисляют одно значение для каждой группы: SUM суммирует, COUNT считает, AVG усредняет.",
            )

            agg_funcs_list = st.session_state["db_aggregations"]

            num_col_names = [c["column_name"] for c in columns_info
                             if any(t in c["data_type"].lower()
                                    for t in ("int", "numeric", "float", "real", "double", "decimal", "money", "serial"))]

            if st.button(
                "➕ Добавить агрегат",
                key="btn_add_agg",
                help="Добавить агрегационную функцию (SUM, COUNT, AVG и др.) к выбранной колонке.",
            ):
                agg_funcs_list.append({
                    "func": "SUM",
                    "col": num_col_names[0] if num_col_names else col_names[0],
                    "alias": "",
                })
                st.session_state["db_aggregations"] = agg_funcs_list
                st.rerun()

            AGG_FUNCS = {
                "SUM": "Сумма значений",
                "COUNT": "Количество строк",
                "COUNT DISTINCT": "Уникальных значений",
                "AVG": "Среднее значение",
                "MIN": "Минимум",
                "MAX": "Максимум",
                "STDDEV": "Стандартное отклонение",
                "VARIANCE": "Дисперсия",
            }

            agg_to_remove = []
            for idx, agg in enumerate(agg_funcs_list):
                ag1, ag2, ag3, ag4 = st.columns([2, 2, 2, 1])
                agg["func"] = ag1.selectbox(
                    "Функция",
                    list(AGG_FUNCS.keys()),
                    index=list(AGG_FUNCS.keys()).index(agg.get("func", "SUM")),
                    key=f"agg_fn_{idx}",
                    format_func=lambda f: f"{f} — {AGG_FUNCS[f]}",
                    help="SUM — сумма; COUNT — количество; AVG — среднее; MIN/MAX — минимум/максимум; STDDEV — стандартное отклонение.",
                    label_visibility="collapsed" if idx > 0 else "visible",
                )
                agg["col"] = ag2.selectbox(
                    "Колонка",
                    col_names,
                    index=col_names.index(agg["col"]) if agg["col"] in col_names else 0,
                    key=f"agg_col_{idx}",
                    label_visibility="collapsed" if idx > 0 else "visible",
                    help="Колонка, к которой применяется функция. Для COUNT(*) выберите любую.",
                )
                agg["alias"] = ag3.text_input(
                    "Псевдоним",
                    value=agg.get("alias", f"{agg['func'].lower().replace(' ', '_')}_{agg['col']}"),
                    key=f"agg_alias_{idx}",
                    label_visibility="collapsed" if idx > 0 else "visible",
                    help="Имя агрегированной колонки в результате (например: total_amount).",
                )
                if ag4.button("✕", key=f"agg_del_{idx}", help="Удалить агрегат"):
                    agg_to_remove.append(idx)

            if agg_to_remove:
                st.session_state["db_aggregations"] = [a for i, a in enumerate(agg_funcs_list) if i not in agg_to_remove]
                st.rerun()

            # HAVING
            st.markdown("**🔎 HAVING — фильтр по агрегатам**")
            st.caption(
                "Фильтрует группы ПОСЛЕ группировки (в отличие от WHERE, который фильтрует строки ДО группировки).",
                help="Пример: HAVING SUM(amount) > 10000 — покажет только группы, где суммарная сумма > 10000.",
            )
            having_expr = st.text_input(
                "HAVING условие",
                value=st.session_state["db_having"],
                key="db_having_input",
                placeholder="Например: SUM(amount) > 1000 или COUNT(*) >= 5",
                help="Введите условие без слова HAVING. Можно использовать агрегаты: SUM(col) > 100, COUNT(*) >= 10, AVG(score) < 0.5.",
            )
            st.session_state["db_having"] = having_expr

    # ── STEP 6 – Limit ────────────────────────────────────────────────────────
    st.divider()
    lim_col, name_col = st.columns(2)
    row_limit = lim_col.number_input(
        "Максимум строк (LIMIT)", min_value=1, max_value=1_000_000,
        value=10_000, step=1_000, key="db_row_limit",
    )
    ds_name_pg = name_col.text_input(
        "Имя датасета в KIBAD",
        value=f"{selected_schema}_{selected_table}",
        key="db_dataset_name",
    )

    # ── STEP 7 – SQL preview ──────────────────────────────────────────────────
    from services.db import build_nocode_query
    _preview_cols = selected_col_labels if not st.session_state.get("db_select_all_cols", True) else None
    _preview_sql = build_nocode_query(
        schema=selected_schema,
        table=selected_table,
        columns=_preview_cols,
        computed_cols=st.session_state.get("db_computed_cols") or None,
        distinct=st.session_state.get("db_distinct", False),
        joins=st.session_state.get("db_joins") or None,
        filters=[f for f in st.session_state["db_filters"] if f.get("col")],
        group_by=st.session_state["db_group_by"] or None,
        aggregations=st.session_state.get("db_aggregations") or None,
        having=st.session_state.get("db_having") or None,
        order_by=st.session_state["db_order_by"] or None,
        limit=int(row_limit),
    )

    with st.expander("🔎 Предварительный просмотр SQL-запроса", expanded=False):
        st.code(_preview_sql, language="sql")
        st.caption("SQL генерируется автоматически на основе выборок выше. Вы можете скопировать и изменить его вручную.")

    # ── STEP 7б – Saved queries ───────────────────────────────────────────────
    st.divider()
    section_header("Сохранённые запросы", "💾")

    sv_c1, sv_c2 = st.columns([3, 1])
    save_name = sv_c1.text_input(
        "Название запроса",
        value=f"{selected_schema}_{selected_table}_query",
        key="db_save_query_name",
        help="Дайте запросу понятное имя. Сохранённые запросы доступны в течение сессии.",
        label_visibility="visible",
    )
    if sv_c2.button(
        "💾 Сохранить запрос",
        key="btn_save_query",
        use_container_width=True,
        help="Сохранить текущий SQL-запрос под указанным именем для повторного использования.",
    ):
        st.session_state["db_saved_queries"][save_name] = {
            "sql": _preview_sql,
            "table": f"{selected_schema}.{selected_table}",
            "saved_at": pd.Timestamp.now().strftime("%H:%M:%S"),
        }
        st.success(f"Запрос «{save_name}» сохранён.")

    saved = st.session_state["db_saved_queries"]
    if saved:
        chosen_saved = st.selectbox(
            "Загрузить сохранённый запрос",
            ["— выберите —"] + list(saved.keys()),
            key="db_load_saved_sel",
            help="Выберите ранее сохранённый запрос для просмотра SQL.",
        )
        if chosen_saved != "— выберите —":
            q = saved[chosen_saved]
            st.info(f"📌 Таблица: **{q['table']}** | Сохранён в {q['saved_at']}")
            st.code(q["sql"], language="sql")
            del_col, _ = st.columns([2, 8])
            if del_col.button(
                "🗑️ Удалить запрос",
                key="btn_del_saved",
                help="Удалить выбранный запрос из списка сохранённых.",
            ):
                del st.session_state["db_saved_queries"][chosen_saved]
                st.rerun()

    # ── STEP 8 – Execute ──────────────────────────────────────────────────────
    st.divider()
    exec_col, prev_col, _ = st.columns([2, 2, 6])

    if exec_col.button("▶ Загрузить данные", type="primary", key="btn_pg_execute"):
        with st.spinner("Выполнение запроса..."):
            try:
                from services.db import query_to_dataframe
                df_pg = query_to_dataframe(**params, query=_preview_sql)
                if df_pg.empty:
                    st.warning("Запрос вернул 0 строк. Проверьте фильтры.")
                else:
                    store_dataset(ds_name_pg, df_pg, source="postgres")
                    log_event("file_loaded", {
                        "source": "postgres",
                        "dataset": ds_name_pg,
                        "table": f"{selected_schema}.{selected_table}",
                        "rows": df_pg.shape[0],
                        "columns": df_pg.shape[1],
                    })
                    st.session_state["db_last_result"] = df_pg
                    st.success(f"✅ Датасет «{ds_name_pg}» загружен: {len(df_pg):,} строк × {len(df_pg.columns)} колонок")
                    with st.expander("👁️ Предварительный просмотр данных", expanded=True):
                        preview_c1, preview_c2, preview_c3, preview_c4 = st.columns(4)
                        preview_c1.metric("Строк", f"{len(df_pg):,}")
                        preview_c2.metric("Столбцов", len(df_pg.columns))
                        preview_c3.metric("Числовых", len(df_pg.select_dtypes(include='number').columns))
                        preview_c4.metric("Пропусков", f"{df_pg.isnull().sum().sum():,}")
                        st.dataframe(df_pg.head(10), use_container_width=True)
                        # Column schema
                        schema_df = pd.DataFrame({
                            "Колонка": df_pg.columns,
                            "Тип": df_pg.dtypes.astype(str).values,
                            "Заполнено %": (df_pg.notna().mean() * 100).round(1).values,
                            "Уникальных": df_pg.nunique().values,
                            "Пример": [str(df_pg[c].dropna().iloc[0]) if df_pg[c].notna().any() else "—" for c in df_pg.columns],
                        })
                        st.markdown("**Схема данных:**")
                        st.dataframe(schema_df, use_container_width=True, hide_index=True)
                    st.rerun()
            except Exception as exc:
                st.error(f"Ошибка выполнения запроса: {exc}")

    # Preview last result
    last_result = st.session_state.get("db_last_result")
    if last_result is not None:
        section_header("Предварительный просмотр результата", "👁️")
        st.dataframe(last_result.head(20), use_container_width=True)
        csv_bytes = last_result.to_csv(index=False).encode()
        st.download_button("⬇️ Скачать CSV", csv_bytes,
                           file_name=f"{ds_name_pg}.csv", mime="text/csv",
                           key="btn_pg_csv")

# ---------------------------------------------------------------------------
# Dataset catalog
# ---------------------------------------------------------------------------
with tab_catalog:
    section_header(t("page_data_schema"), "🗂️")
    names = list_dataset_names()
    if not names:
        st.info(t("no_datasets"))
    else:
        chosen = st.selectbox(t("select_dataset"), names, key="catalog_select")
        if chosen:
            st.session_state["active_ds"] = chosen
            raw_df = st.session_state["datasets"].get(chosen)
            prep_df = st.session_state.get("prepared_dfs", {}).get(chosen, raw_df)
            df_view = prep_df if prep_df is not None else raw_df

            st.markdown(f"**{t('rows')}:** {df_view.shape[0]:,}  **{t('columns')}:** {df_view.shape[1]}")

            subtab_prev, subtab_profile, subtab_stats, subtab_types = st.tabs(
                [t("preview"), t("page_data_profile"), "Числовая статистика", "Переопределение типов"]
            )

            with subtab_prev:
                n_rows = st.slider(t("rows"), 5, 100, 10, key="prev_rows")
                st.dataframe(df_view.head(n_rows), use_container_width=True)
                csv_bytes = df_view.to_csv(index=False).encode()
                st.download_button("⬇ Скачать CSV", csv_bytes, file_name=f"{chosen}.csv", mime="text/csv")

            with subtab_profile:
                with st.spinner("Профилирование..."):
                    profile = profile_dataframe(df_view)
                st.dataframe(profile, use_container_width=True)

            with subtab_stats:
                desc = describe_numeric(df_view)
                if desc.empty:
                    st.info("Числовых столбцов не найдено.")
                else:
                    st.dataframe(desc, use_container_width=True)

            with subtab_types:
                st.markdown("Переопределите автоматически выведенные типы столбцов. Изменения применяются в разделе **Подготовка**.")
                inferred = infer_column_types(df_view)
                overrides = st.session_state.get("type_overrides", {}).get(chosen, {})
                updated: dict = {}
                type_options = ["(auto)", "datetime", "numeric", "categorical", "boolean"]
                cols_for_override = list(df_view.columns)
                for col in cols_for_override:
                    current = overrides.get(col, "(auto)")
                    sel = st.selectbox(
                        f"`{col}` (определён как: {inferred.get(col, '?')})",
                        type_options,
                        index=type_options.index(current) if current in type_options else 0,
                        key=f"type_{chosen}_{col}",
                    )
                    if sel != "(auto)":
                        updated[col] = sel
                if st.button(t("apply"), key="save_types"):
                    if "type_overrides" not in st.session_state:
                        st.session_state["type_overrides"] = {}
                    st.session_state["type_overrides"][chosen] = updated
                    st.success("Переопределения типов сохранены.")

            if st.button(f"Удалить '{chosen}'", key="btn_remove_ds"):
                del st.session_state["datasets"][chosen]
                st.session_state.get("prepared_dfs", {}).pop(chosen, None)
                if st.session_state.get("active_ds") == chosen:
                    remaining = list_dataset_names()
                    st.session_state["active_ds"] = remaining[0] if remaining else None
                st.rerun()
