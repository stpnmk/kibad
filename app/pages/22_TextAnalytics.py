"""
pages/22_TextAnalytics.py – Text column analysis for bank employees.

Analyzes text columns (client comments, transaction descriptions, etc.):
- Word frequency & cloud
- N-gram analysis
- Text length statistics
- Keyword search & flagging
- Simple categorization by keywords
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import re
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.state import init_state, dataset_selectbox, get_active_df, store_prepared
from app.styles import inject_all_css, page_header, section_header

st.set_page_config(page_title="KIBAD – Текст", layout="wide")
init_state()
inject_all_css()

page_header("22. Анализ текста", "Частота слов, категоризация и поиск по тексту", "📝")

ds_name = dataset_selectbox("Датасет", key="txt_ds")
if not ds_name:
    st.stop()

df = get_active_df()
if df is None or df.empty:
    st.info("Загрузите данные на странице «Данные».")
    st.stop()

# Only show text columns
text_cols = df.select_dtypes(include=["object"]).columns.tolist()
if not text_cols:
    st.warning("В датасете нет текстовых столбцов (тип object).")
    st.stop()

col_sel = st.selectbox("Текстовый столбец для анализа:", text_cols, key="txt_col")
series = df[col_sel].dropna().astype(str)

st.markdown(f"**Строк с данными:** {len(series):,}  |  **Уникальных значений:** {series.nunique():,}")

# ---------------------------------------------------------------------------
# Stopwords (Russian + banking common words to optionally exclude)
# ---------------------------------------------------------------------------
DEFAULT_STOPWORDS = {
    "и", "в", "на", "с", "по", "к", "за", "из", "от", "у", "о", "об", "для",
    "не", "что", "как", "это", "а", "но", "или", "то", "же", "бы", "если",
    "все", "так", "при", "до", "со", "во", "его", "ее", "их", "ее", "они",
    "он", "она", "оно", "мы", "вы", "я", "мне", "вам", "нам", "был", "была",
    "было", "были", "есть", "нет", "да", "по", "чтобы", "что", "когда",
    "очень", "уже", "еще", "также", "же", "ли",
}

BANKING_STOPWORDS = {
    "банк", "счет", "счёт", "клиент", "платеж", "платёж", "перевод",
    "сумма", "руб", "рублей", "дата", "номер",
}

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "📊 Статистика", "🔤 Частота слов", "🔍 Поиск и фильтр",
    "🏷️ Категоризация", "📏 Длины строк"
])

# ===========================================================================
# TAB 1 — Statistics
# ===========================================================================
with tabs[0]:
    st.markdown("### Обзор текстового столбца")

    c1, c2, c3, c4 = st.columns(4)
    avg_len = series.str.len().mean()
    max_len = series.str.len().max()
    min_len = series.str.len().min()
    empty_count = (series.str.strip() == "").sum()

    c1.metric("Ср. длина (символов)", f"{avg_len:.0f}")
    c2.metric("Макс. длина", max_len)
    c3.metric("Мин. длина", min_len)
    c4.metric("Пустых строк", empty_count)

    # Top values
    top_vals = series.value_counts().head(20)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Топ-20 значений:**")
        top_df = pd.DataFrame({"Значение": top_vals.index, "Количество": top_vals.values})
        top_df["Доля %"] = (top_df["Количество"] / len(series) * 100).round(1)
        st.dataframe(top_df, use_container_width=True, height=350)

    with col_b:
        fig = px.bar(
            top_df.head(15), x="Количество", y="Значение", orientation="h",
            title="Топ-15 значений", template="plotly_white", color="Количество",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=400, yaxis={"autorange": "reversed"}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 2 — Word frequency
# ===========================================================================
with tabs[1]:
    st.markdown("### Частота слов")

    wf1, wf2 = st.columns(2)
    with wf1:
        min_word_len = st.slider("Минимальная длина слова:", 2, 8, 3, key="txt_min_wlen")
        top_n_words = st.slider("Топ N слов:", 10, 100, 30, key="txt_topn")
    with wf2:
        use_stopwords = st.checkbox("Исключить стоп-слова (русские)", value=True, key="txt_sw")
        use_bank_sw = st.checkbox("Исключить банковские термины", value=False, key="txt_bank_sw")
        custom_sw_input = st.text_input("Дополнительные стоп-слова (через запятую):", key="txt_custom_sw")

    stopwords = set()
    if use_stopwords:
        stopwords |= DEFAULT_STOPWORDS
    if use_bank_sw:
        stopwords |= BANKING_STOPWORDS
    if custom_sw_input:
        stopwords |= {w.strip().lower() for w in custom_sw_input.split(",")}

    # Tokenize
    @st.cache_data(show_spinner=False)
    def get_word_freq(texts: tuple, min_len: int, sw: frozenset) -> list:
        all_words = []
        for text in texts:
            words = re.findall(r"[а-яёa-z]+", text.lower())
            for w in words:
                if len(w) >= min_len and w not in sw:
                    all_words.append(w)
        return Counter(all_words).most_common()

    with st.spinner("Подсчёт слов..."):
        word_freq = get_word_freq(tuple(series.tolist()), min_word_len, frozenset(stopwords))

    if word_freq:
        word_df = pd.DataFrame(word_freq[:top_n_words], columns=["Слово", "Частота"])

        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            fig_wf = px.bar(
                word_df, x="Частота", y="Слово", orientation="h",
                title=f"Топ-{top_n_words} слов", template="plotly_white",
                color="Частота", color_continuous_scale="Viridis",
            )
            fig_wf.update_layout(height=max(400, top_n_words * 18), yaxis={"autorange": "reversed"},
                                  showlegend=False)
            st.plotly_chart(fig_wf, use_container_width=True)

        with col_table:
            st.dataframe(word_df, use_container_width=True, height=500)
            csv_words = word_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("📥 Скачать частоты (CSV)", data=csv_words,
                               file_name="word_freq.csv", mime="text/csv")

        # Bigrams
        st.markdown("### Биграммы (пары слов)")

        @st.cache_data(show_spinner=False)
        def get_bigrams(texts: tuple, min_len: int, sw: frozenset) -> list:
            bigrams = []
            for text in texts:
                words = [w for w in re.findall(r"[а-яёa-z]+", text.lower())
                         if len(w) >= min_len and w not in sw]
                for i in range(len(words) - 1):
                    bigrams.append(f"{words[i]} {words[i+1]}")
            return Counter(bigrams).most_common(20)

        bigrams = get_bigrams(tuple(series.tolist()), min_word_len, frozenset(stopwords))
        if bigrams:
            bg_df = pd.DataFrame(bigrams, columns=["Биграмм", "Частота"])
            fig_bg = px.bar(bg_df, x="Частота", y="Биграмм", orientation="h",
                            title="Топ-20 биграмм", template="plotly_white")
            fig_bg.update_layout(height=450, yaxis={"autorange": "reversed"})
            st.plotly_chart(fig_bg, use_container_width=True)
    else:
        st.info("Слова не найдены. Попробуйте уменьшить минимальную длину слова.")

# ===========================================================================
# TAB 3 — Search & Filter
# ===========================================================================
with tabs[2]:
    st.markdown("### Поиск по ключевым словам")
    st.caption("Найдите строки, содержащие определённые слова или фразы")

    s1, s2 = st.columns(2)
    with s1:
        search_query = st.text_input("Поисковый запрос (часть слова или фраза):", key="txt_search",
                                      placeholder="Например: кредит, просроч, жалоб")
        case_sensitive = st.checkbox("Учитывать регистр", value=False, key="txt_case")
    with s2:
        match_mode = st.radio("Режим совпадения:", ["Содержит", "Точное совпадение", "Начинается с"],
                              key="txt_match_mode", horizontal=True)

    if search_query:
        if match_mode == "Содержит":
            mask = series.str.contains(search_query, case=case_sensitive, na=False, regex=False)
        elif match_mode == "Точное совпадение":
            if case_sensitive:
                mask = series == search_query
            else:
                mask = series.str.lower() == search_query.lower()
        else:  # Начинается с
            mask = series.str.startswith(search_query, na=False) if case_sensitive \
                   else series.str.lower().str.startswith(search_query.lower(), na=False)

        n_found = mask.sum()
        pct_found = n_found / len(series) * 100

        if n_found > 0:
            st.success(f"Найдено: **{n_found:,}** строк ({pct_found:.1f}% от всех)")
            found_df = df[df[col_sel].notna() & df[col_sel].astype(str).str.contains(
                search_query, case=case_sensitive, na=False, regex=False)]
            st.dataframe(found_df.head(200), use_container_width=True, height=350)

            # Save as new dataset
            if st.button("💾 Сохранить результат поиска как датасет", key="btn_save_search"):
                from app.state import add_dataset
                add_dataset(f"{ds_name}_search_{search_query[:15]}", found_df.reset_index(drop=True))
                st.success("Сохранено!")

            # Download
            csv_found = found_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("📥 Скачать найденные строки (CSV)", data=csv_found,
                               file_name="search_result.csv", mime="text/csv")
        else:
            st.warning(f"По запросу «{search_query}» ничего не найдено.")

    # Multi-keyword filter
    st.markdown("---")
    st.markdown("### Мультиключевой фильтр")
    multi_kw = st.text_area("Ключевые слова (каждое с новой строки):", height=100,
                             key="txt_multi_kw",
                             placeholder="кредит\nпросрочка\nжалоба")
    if multi_kw:
        keywords = [kw.strip() for kw in multi_kw.strip().splitlines() if kw.strip()]
        if keywords:
            keyword_counts = {}
            for kw in keywords:
                count = series.str.contains(kw, case=False, na=False, regex=False).sum()
                keyword_counts[kw] = count

            kw_df = pd.DataFrame(list(keyword_counts.items()), columns=["Ключевое слово", "Количество совпадений"])
            kw_df["Доля %"] = (kw_df["Количество совпадений"] / len(series) * 100).round(1)
            kw_df = kw_df.sort_values("Количество совпадений", ascending=False)

            st.dataframe(kw_df, use_container_width=True)
            fig_kw = px.bar(kw_df, x="Ключевое слово", y="Количество совпадений",
                            title="Встречаемость ключевых слов", template="plotly_white",
                            color="Количество совпадений", color_continuous_scale="Oranges")
            fig_kw.update_layout(height=350)
            st.plotly_chart(fig_kw, use_container_width=True)

# ===========================================================================
# TAB 4 — Keyword Categorization (KILLER FEATURE for banks)
# ===========================================================================
with tabs[3]:
    st.markdown("### Категоризация по ключевым словам")
    st.caption(
        "Разметьте строки по категориям автоматически — "
        "укажите ключевые слова для каждой категории и получите новый столбец с метками."
    )

    st.markdown("**Определите категории:**")
    st.info("Укажите категории и ключевые слова (через запятую). Строки будут помечены первой подходящей категорией.")

    # Dynamic category builder
    if "cat_rules" not in st.session_state:
        st.session_state["cat_rules"] = [
            {"name": "Жалоба", "keywords": "жалоба,плохо,ужасно,недоволен"},
            {"name": "Вопрос", "keywords": "вопрос,как,подскажите,помогите"},
            {"name": "Благодарность", "keywords": "спасибо,благодарю,отлично,хорошо"},
        ]

    rules = st.session_state["cat_rules"]

    for i, rule in enumerate(rules):
        col_r1, col_r2, col_r3 = st.columns([2, 5, 1])
        with col_r1:
            rules[i]["name"] = st.text_input("Категория:", value=rule["name"], key=f"cat_name_{i}")
        with col_r2:
            rules[i]["keywords"] = st.text_input("Ключевые слова (через запятую):",
                                                  value=rule["keywords"], key=f"cat_kw_{i}")
        with col_r3:
            st.write("")
            if st.button("✕", key=f"del_cat_{i}"):
                rules.pop(i)
                st.rerun()

    if st.button("➕ Добавить категорию", key="btn_add_cat"):
        rules.append({"name": f"Категория {len(rules) + 1}", "keywords": ""})
        st.rerun()

    default_cat = st.text_input("Категория по умолчанию (если не подходит ни одна):", value="Другое", key="cat_default")
    new_col_name = st.text_input("Название новой колонки с категориями:", value=f"{col_sel}_категория", key="cat_col_name")

    if st.button("▶ Категоризировать", type="primary", key="btn_categorize"):
        result_series = pd.Series([default_cat] * len(df), index=df.index)
        total_matched = 0

        for rule in reversed(rules):  # reversed so first rule wins
            if not rule["name"] or not rule["keywords"]:
                continue
            kws = [kw.strip().lower() for kw in rule["keywords"].split(",") if kw.strip()]
            if not kws:
                continue
            pattern = "|".join(re.escape(kw) for kw in kws)
            mask = df[col_sel].astype(str).str.lower().str.contains(pattern, na=False, regex=True)
            result_series[mask] = rule["name"]
            total_matched += mask.sum()

        work_df = df.copy()
        work_df[new_col_name] = result_series

        # Show distribution
        dist = result_series.value_counts()
        st.success(f"Категоризировано: **{total_matched:,}** строк помечены ({total_matched/len(df)*100:.1f}%)")

        dist_col, chart_col = st.columns(2)
        with dist_col:
            dist_df = pd.DataFrame({"Категория": dist.index, "Количество": dist.values})
            dist_df["Доля %"] = (dist_df["Количество"] / len(df) * 100).round(1)
            st.dataframe(dist_df, use_container_width=True)
        with chart_col:
            fig_cat = px.pie(dist_df, names="Категория", values="Количество",
                             title="Распределение по категориям", template="plotly_white")
            fig_cat.update_layout(height=350)
            st.plotly_chart(fig_cat, use_container_width=True)

        store_prepared(ds_name, work_df)
        st.success(f"Столбец «{new_col_name}» добавлен в датасет «{ds_name}».")

# ===========================================================================
# TAB 5 — String lengths
# ===========================================================================
with tabs[4]:
    st.markdown("### Распределение длин строк")

    lengths = series.str.len()

    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Среднее", f"{lengths.mean():.1f}")
    l2.metric("Медиана", f"{lengths.median():.0f}")
    l3.metric("Макс.", int(lengths.max()))
    l4.metric("Мин.", int(lengths.min()))

    fig_len = px.histogram(
        lengths, nbins=50, title="Распределение длин строк (символов)",
        labels={"value": "Длина (символов)", "count": "Количество"},
        template="plotly_white", color_discrete_sequence=["#3182ce"],
    )
    fig_len.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_len, use_container_width=True)

    # Outliers — very long strings
    p95 = lengths.quantile(0.95)
    long_strings = df[series.str.len() > p95].head(20)
    if not long_strings.empty:
        with st.expander(f"📌 Аномально длинные строки (> 95-й перцентиль, >{p95:.0f} символов)"):
            st.dataframe(long_strings[[col_sel]], use_container_width=True)
