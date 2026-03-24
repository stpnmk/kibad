"""
pages/6_Tests.py – Statistical tests with rich visualizations and Russian UI.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

try:
    from statsmodels.stats.power import TTestIndPower, TTestPower as TTestOneSampPower, NormalIndPower, FTestAnovaPower
    from statsmodels.stats.proportion import proportion_effectsize
    _STATSMODELS_OK = True
except Exception:
    _STATSMODELS_OK = False

from app.state import init_state, dataset_selectbox, get_active_df
from core.interpret import interpret_pvalue, interpret_effect_size
from app.components.ux import interpretation_box
from app.styles import inject_all_css, page_header, section_header
from core.tests import (
    ttest_independent, mann_whitney, chi_square_independence,
    correlation_test, bootstrap_test, ab_test, bh_correction,
    permutation_test, cliffs_delta,
    normality_test, levene_test, diagnose_groups, NormalityResult,
)

st.set_page_config(page_title="KIBAD – Статистические тесты", layout="wide")
init_state()
inject_all_css()
page_header("6. Статистические тесты", "Проверка гипотез: t-тест, Манна-Уитни, хи-квадрат, корреляция, бутстрап, A/B", "🔬")

chosen = dataset_selectbox("Датасет", key="tests_ds_sel")
if not chosen:
    st.stop()

df = get_active_df()
if df is None:
    st.error("Данные не найдены. Загрузите датасет.")
    st.stop()

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

# ---------------------------------------------------------------------------
# Inline config panel (replaces sidebar)
# ---------------------------------------------------------------------------
cfg_col1, cfg_col2, cfg_col3, cfg_col4, cfg_col5 = st.columns([2, 1, 1, 1, 1])

with cfg_col1:
    alpha = st.slider(
        "Уровень значимости (α)",
        0.01, 0.10, 0.05, 0.01, key="alpha",
        help=(
            "Порог, при котором результат считается статистически значимым. "
            "0.05 — стандарт для науки и бизнеса (5% риск ложного позитива). "
            "0.01 — строже, для критически важных решений."
        ),
    )

cfg_col2.metric("Строк", f"{len(df):,}", help="Количество строк в активном датасете")
cfg_col3.metric("Числовых", len(num_cols), help="Колонки с числовыми значениями — доступны для большинства тестов")
cfg_col4.metric("Категориальных", len(cat_cols), help="Текстовые/категориальные колонки — нужны для группировки и χ²-теста")
cfg_col5.metric(
    "α порог",
    f"{alpha:.2f}",
    delta="строгий" if alpha <= 0.01 else ("стандарт" if alpha == 0.05 else "мягкий"),
    help="Текущий выбранный уровень значимости",
)

st.divider()

with st.expander("🧭 Быстрый справочник: какой тест выбрать?", expanded=False):
    st.markdown("""
    | Ситуация | Рекомендуемый тест | Вкладка |
    |---|---|---|
    | Две группы, нормальное распределение, одинаковые дисперсии | **t-тест Стьюдента** | 📊 t-Тест |
    | Две группы, нормальное распределение, разные дисперсии | **t-тест Уэлча** | 📊 t-Тест |
    | Две группы, ненормальное распределение | **Манна-Уитни U** | 🏆 Манна-Уитни |
    | Неизвестно/нет предположений о распределении | **Бутстрап** | 🎲 Бутстрап |
    | A/B тест с несколькими метриками | **A/B комплексный** | 🧪 A/B тест |
    | Две категориальные переменные | **χ² тест** | χ² Хи-квадрат |
    | Линейная связь между двумя числовыми | **Корреляция Пирсона** | 🔗 Корреляция |
    | Ранговая связь или выбросы | **Корреляция Спирмена** | 🔗 Корреляция |
    | 3+ группы сравниваются | **Поправка BH/FDR** | 🔢 Мн. сравнения |

    > 💡 **Не уверены?** Перейдите на вкладку **🔍 Шаг 1: Диагностика данных** — KIBAD автоматически проверит нормальность и порекомендует тест.
    """)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_card(result):
    is_sig = result.significant
    color = "#d4edda" if is_sig else "#f8d7da"
    icon = "✅" if is_sig else "❌"
    verdict = "ЗНАЧИМО" if is_sig else "НЕ ЗНАЧИМО"
    st.markdown(
        f'<div style="background:{color};border-radius:8px;padding:12px 16px;margin:8px 0">'
        f'<b>{icon} {result.name}</b> — <b>{verdict}</b> при α={alpha}<br/>'
        f'<span style="font-size:13px">p-value = <b>{result.p_value}</b> | '
        f'Статистика = {result.statistic} | '
        f'Размер эффекта = {result.effect_size if result.effect_size is not None else "N/A"}'
        f'{"  |  " + result.effect_label if result.effect_label else ""}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if result.ci:
        st.info(f"📊 95% ДИ: [{result.ci[0]}, {result.ci[1]}]")
    st.markdown(f"**Интерпретация:** {result.interpretation}")
    st.session_state["test_results"].append(result)


def _overlay_hist(df, grp_col, val_col, ga, gb):
    a = df[df[grp_col] == ga][val_col].dropna()
    b = df[df[grp_col] == gb][val_col].dropna()
    fig = go.Figure()
    for series, name, color in [(a, str(ga), "#3498db"), (b, str(gb), "#e74c3c")]:
        fig.add_trace(go.Histogram(
            x=series, name=name, opacity=0.55,
            marker_color=color, nbinsx=30, histnorm="probability density",
        ))
        if len(series) > 5 and scipy_stats is not None:
            try:
                xi = np.linspace(float(series.min()), float(series.max()), 200)
                kde = scipy_stats.gaussian_kde(series)
                fig.add_trace(go.Scatter(x=xi, y=kde(xi), mode="lines",
                                         name=f"KDE {name}",
                                         line=dict(color=color, width=2.5), showlegend=False))
            except Exception:
                pass
        fig.add_vline(x=float(series.mean()), line_dash="dash", line_color=color,
                      annotation_text=f"μ={series.mean():.2f}")
    fig.update_layout(barmode="overlay",
                      title=f"Распределение «{val_col}» по группам",
                      xaxis_title=val_col, yaxis_title="Плотность",
                      template="plotly_white",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def _ci_bars(df, grp_col, val_col, ga, gb):
    a = df[df[grp_col] == ga][val_col].dropna()
    b = df[df[grp_col] == gb][val_col].dropna()
    fig = go.Figure()
    for series, name, color in [(a, str(ga), "#3498db"), (b, str(gb), "#e74c3c")]:
        m = series.mean()
        se = (series.std() / np.sqrt(len(series))) if len(series) > 1 else 0
        fig.add_trace(go.Bar(
            x=[name], y=[m], marker_color=color,
            error_y=dict(type="data", symmetric=False,
                         array=[1.96*se], arrayminus=[1.96*se]),
            text=[f"{m:.3f}"], textposition="outside",
        ))
    fig.update_layout(title=f"Среднее ± 95% ДИ: «{val_col}»",
                      yaxis_title=val_col, template="plotly_white",
                      showlegend=False, barmode="group")
    return fig


def _gauge(value, thresholds, title):
    try:
        val = abs(float(value))
    except Exception:
        return None
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        title={"text": title, "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, thresholds[-1] * 1.3]},
            "bar": {"color": "#3498db"},
            "steps": [
                {"range": [0, thresholds[0]], "color": "#ecf0f1"},
                {"range": [thresholds[0], thresholds[1]], "color": "#f9e79f"},
                {"range": [thresholds[1], thresholds[2]], "color": "#f0b27a"},
                {"range": [thresholds[2], thresholds[-1]*1.3], "color": "#ec7063"},
            ],
        },
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_diag, tab_ttest, tab_mw, tab_chi2, tab_corr, tab_boot, tab_ab, tab_multi, tab_power, tab_history = st.tabs([
    "🔍 Шаг 1: Диагностика данных",
    "📊 t-Тест", "🏆 Манна-Уитни", "χ² Хи-квадрат",
    "🔗 Корреляция", "🎲 Бутстрап", "🧪 A/B тест",
    "🔢 Мн. сравнения", "⚡ Анализ мощности", "📋 История",
])


# ===========================================================================
# Diagnostics (Step 1)
# ===========================================================================
with tab_diag:
    section_header("Диагностика данных перед выбором теста")
    st.info(
        "**Зачем это нужно?** Выбор корректного статистического теста зависит от структуры данных. "
        "Здесь вы оцениваете нормальность каждой группы и однородность дисперсий — "
        "и получаете автоматическую рекомендацию, какой тест использовать."
    )

    if not num_cols or not cat_cols:
        st.warning("Нужна хотя бы одна числовая и одна категориальная колонка.")
    else:
        d_c1, d_c2 = st.columns(2)
        diag_val = d_c1.selectbox("Числовая колонка", num_cols, key="diag_val")
        diag_grp = d_c2.selectbox("Группирующая колонка", cat_cols, key="diag_grp")
        diag_groups = df[diag_grp].dropna().unique().tolist()

        if len(diag_groups) >= 2:
            d_c3, d_c4 = st.columns(2)
            diag_ga = d_c3.selectbox("Группа A", diag_groups, key="diag_ga")
            diag_gb = d_c4.selectbox("Группа B",
                                     [g for g in diag_groups if g != diag_ga],
                                     key="diag_gb")

            if st.button("🔍 Запустить диагностику", type="primary", key="btn_diag"):
                a_s = df[df[diag_grp] == diag_ga][diag_val].dropna()
                b_s = df[df[diag_grp] == diag_gb][diag_val].dropna()

                try:
                    diag = diagnose_groups(a_s, b_s, alpha=alpha,
                                           label_a=str(diag_ga), label_b=str(diag_gb))
                    st.session_state["diag_result"] = diag
                    st.session_state["diag_a"] = a_s
                    st.session_state["diag_b"] = b_s
                    st.session_state["diag_labels"] = (str(diag_ga), str(diag_gb), diag_val)
                except Exception as e:
                    st.error(f"Ошибка диагностики: {e}")

        else:
            st.warning("Группирующая колонка должна содержать минимум 2 значения.")

    # ── Display stored result ──────────────────────────────────────────────
    if "diag_result" in st.session_state:
        diag = st.session_state["diag_result"]
        a_s = st.session_state["diag_a"]
        b_s = st.session_state["diag_b"]
        label_a, label_b, val_col = st.session_state["diag_labels"]

        st.divider()

        # ── Recommendation card ────────────────────────────────────────────
        rec_color = {
            "ttest_welch": "#1F3864", "ttest_student": "#1F3864",
            "mann_whitney": "#155724", "bootstrap": "#856404",
        }.get(diag["rec_test"], "#1F3864")
        rec_bg = {
            "ttest_welch": "#eef2ff", "ttest_student": "#eef2ff",
            "mann_whitney": "#d1e7dd", "bootstrap": "#fff3cd",
        }.get(diag["rec_test"], "#eef2ff")

        st.markdown(
            f"""<div style='background:{rec_bg};border:2px solid {rec_color};
            border-radius:10px;padding:18px 22px;margin-bottom:16px'>
            <div style='font-size:0.75rem;font-weight:700;text-transform:uppercase;
            letter-spacing:0.08em;color:{rec_color};margin-bottom:6px'>
            ✅ Рекомендуемый тест</div>
            <div style='font-size:1.4rem;font-weight:800;color:{rec_color}'>
            {diag["rec_name"]}</div>
            <div style='font-size:0.9rem;color:#495057;margin-top:6px'>
            {diag["rec_reason"]}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Auto-apply recommendation button ─────────────────────────────
        if st.button("▶ Применить рекомендацию", key="btn_auto_apply"):
            rec = diag["rec_test"]
            _val_col = st.session_state.get("diag_val")
            _grp_col = st.session_state.get("diag_grp")
            _ga = st.session_state.get("diag_ga")
            _gb = st.session_state.get("diag_gb")
            _tab_name = ""
            if rec in ("ttest_welch", "ttest_student"):
                st.session_state["tt_val"] = _val_col
                st.session_state["tt_grp"] = _grp_col
                st.session_state["tt_ga"] = _ga
                st.session_state["tt_gb"] = _gb
                st.session_state["tt_eqvar"] = (rec == "ttest_student")
                _tab_name = "📊 t-Тест"
            elif rec == "mann_whitney":
                st.session_state["mw_val"] = _val_col
                st.session_state["mw_grp"] = _grp_col
                st.session_state["mw_ga"] = _ga
                st.session_state["mw_gb"] = _gb
                _tab_name = "🏆 Манна-Уитни"
            elif rec == "bootstrap":
                st.session_state["bt_val"] = _val_col
                st.session_state["bt_grp"] = _grp_col
                st.session_state["bt_ga"] = _ga
                st.session_state["bt_gb"] = _gb
                _tab_name = "🎲 Бутстрап"
            st.session_state["auto_applied_test"] = rec
            if _tab_name:
                st.success(
                    f"✅ Параметры заполнены! Перейдите на вкладку «{_tab_name}» "
                    "и нажмите кнопку запуска."
                )
                st.markdown(f"""
<div style='background:#e8f4ff;border:1px solid #9eeaf9;border-left:4px solid #0d6efd;
border-radius:10px;padding:12px 16px;margin:8px 0'>
<b>⚡ Параметры теста заполнены автоматически</b><br>
<span style='font-size:0.9rem;color:#495057'>Перейдите на вкладку <b>«{_tab_name}»</b> —
все поля уже настроены по диагностике данных.</span>
</div>""", unsafe_allow_html=True)

        # Warnings
        for w in diag["warnings"]:
            st.warning(w)

        # ── Per-group normality ────────────────────────────────────────────
        section_header("Тест нормальности по группам", "📊")

        def _norm_card(nr: NormalityResult, series: "pd.Series") -> None:
            icon = "✅" if nr.is_normal else "❌"
            color = "#d1e7dd" if nr.is_normal else "#f8d7da"
            border = "#198754" if nr.is_normal else "#dc3545"
            verdict = "нормальное" if nr.is_normal else "ненормальное"
            st.markdown(
                f"""<div style='background:{color};border-left:5px solid {border};
                border-radius:8px;padding:14px 18px;margin-bottom:12px'>
                <b style='font-size:1rem'>{icon} Распределение: {verdict}</b><br>
                <span style='font-size:0.85rem;color:#333'>
                Тест: <b>{nr.test_name}</b> &nbsp;|&nbsp;
                W = <b>{nr.statistic}</b> &nbsp;|&nbsp;
                p-value = <b>{nr.p_value}</b><br>
                n = {nr.n} &nbsp;|&nbsp;
                Асимметрия = {nr.skewness} ({nr.skew_label}) &nbsp;|&nbsp;
                Эксцесс = {nr.kurtosis} ({nr.kurt_label})
                </span></div>""",
                unsafe_allow_html=True,
            )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Группа A: {label_a}** (n={diag['n_a']})")
            _norm_card(diag["norm_a"], a_s)
        with col_b:
            st.markdown(f"**Группа B: {label_b}** (n={diag['n_b']})")
            _norm_card(diag["norm_b"], b_s)

        # ── Levene test ────────────────────────────────────────────────────
        section_header("Однородность дисперсий (тест Левена)", "⚖️")
        lev = diag["levene"]
        lev_icon = "❌" if lev.significant else "✅"
        lev_color = "#f8d7da" if lev.significant else "#d1e7dd"
        lev_border = "#dc3545" if lev.significant else "#198754"
        lev_verdict = "дисперсии различаются (гетероскедастичность)" if lev.significant \
            else "дисперсии однородны (гомоскедастичность)"
        st.markdown(
            f"""<div style='background:{lev_color};border-left:5px solid {lev_border};
            border-radius:8px;padding:14px 18px;margin-bottom:16px'>
            <b>{lev_icon} {lev_verdict}</b><br>
            <span style='font-size:0.85rem;color:#333'>
            W = {lev.statistic} &nbsp;|&nbsp; p-value = {lev.p_value}
            </span></div>""",
            unsafe_allow_html=True,
        )

        # ── Normality interpretation card ─────────────────────────────────
        both_normal = diag["both_normal"]
        equal_var = diag["equal_var"]
        if both_normal and equal_var:
            _interp_msg = (
                "→ Обе группы нормальны, дисперсии однородны → "
                "рекомендуется **t-тест Стьюдента**"
            )
            _interp_color = "#d1e7dd"
            _interp_border = "#198754"
        elif both_normal and not equal_var:
            _interp_msg = (
                "→ Обе группы нормальны, дисперсии различаются → "
                "рекомендуется **t-тест Уэлча**"
            )
            _interp_color = "#cfe2ff"
            _interp_border = "#084298"
        else:
            _interp_msg = (
                "→ Хотя бы одна группа ненормальна → "
                "рекомендуется **Манна-Уитни** или **Бутстрап**"
            )
            _interp_color = "#fff3cd"
            _interp_border = "#856404"
        st.markdown(
            f"""<div style='background:{_interp_color};border-left:5px solid {_interp_border};
            border-radius:8px;padding:14px 18px;margin-bottom:16px'>
            <b>📌 Автоматическая интерпретация</b><br>
            <span style='font-size:0.95rem'>{_interp_msg}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Visual diagnostics ────────────────────────────────────────────
        section_header("Визуальная диагностика", "📉")

        viz_tabs = st.tabs(["QQ-графики", "Гистограммы + норм. кривая", "Box Plot"])

        with viz_tabs[0]:
            # QQ-plots side by side
            qq_c1, qq_c2 = st.columns(2)
            for col_qq, series, lbl in [(qq_c1, a_s, label_a), (qq_c2, b_s, label_b)]:
                arr = series.values
                with col_qq:
                    if scipy_stats is not None:
                        (osm, osr), (slope, intercept, _) = scipy_stats.probplot(arr)
                        fig_qq = go.Figure()
                        fig_qq.add_trace(go.Scatter(
                            x=list(osm), y=list(osr), mode="markers",
                            marker=dict(color="#3498db", size=5, opacity=0.7),
                            name="Данные",
                        ))
                        x_line = [min(osm), max(osm)]
                        fig_qq.add_trace(go.Scatter(
                            x=x_line,
                            y=[slope * x + intercept for x in x_line],
                            mode="lines",
                            line=dict(color="#e74c3c", dash="dash", width=2),
                            name="Нормальная линия",
                        ))
                        fig_qq.update_layout(
                            title=f"QQ-график: {lbl}",
                            xaxis_title="Теоретические квантили",
                            yaxis_title="Выборочные квантили",
                            template="plotly_white", height=350,
                        )
                        st.plotly_chart(fig_qq, use_container_width=True)
                    else:
                        st.info("scipy не установлен — QQ-график недоступен.")

        with viz_tabs[1]:
            # Histogram + normal curve
            fig_hist = go.Figure()
            for series, lbl, color in [(a_s, label_a, "#3498db"), (b_s, label_b, "#e74c3c")]:
                arr = series.values
                fig_hist.add_trace(go.Histogram(
                    x=arr, name=lbl, opacity=0.5,
                    marker_color=color, nbinsx=30,
                    histnorm="probability density",
                ))
                if len(arr) > 5 and scipy_stats is not None:
                    mu, sigma = float(arr.mean()), float(arr.std())
                    if sigma > 0:
                        xi = np.linspace(float(arr.min()), float(arr.max()), 200)
                        normal_curve = scipy_stats.norm.pdf(xi, mu, sigma)
                        fig_hist.add_trace(go.Scatter(
                            x=xi, y=normal_curve, mode="lines",
                            line=dict(color=color, width=2.5, dash="dash"),
                            name=f"Норм. кривая {lbl}", showlegend=True,
                        ))
            fig_hist.update_layout(
                barmode="overlay",
                title=f"Распределение «{val_col}»",
                xaxis_title=val_col, yaxis_title="Плотность",
                template="plotly_white",
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with viz_tabs[2]:
            sub_box = df[df[diag_grp].isin([diag_ga, diag_gb])][[diag_grp, diag_val]].dropna() \
                if "diag_grp" in dir() else \
                pd.concat([a_s.rename(label_a), b_s.rename(label_b)]).reset_index()
            fig_box = go.Figure()
            for series, lbl, color in [(a_s, label_a, "#3498db"), (b_s, label_b, "#e74c3c")]:
                fig_box.add_trace(go.Box(
                    y=series.values, name=lbl,
                    marker_color=color, boxmean=True,
                    jitter=0.3, pointpos=-1.8,
                ))
            fig_box.update_layout(
                title=f"Box Plot: «{val_col}»",
                yaxis_title=val_col, template="plotly_white",
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # ── Summary table ─────────────────────────────────────────────────
        section_header("Сводная таблица диагностики", "📋")
        summary_rows = []
        for nr, lbl in [(diag["norm_a"], label_a), (diag["norm_b"], label_b)]:
            summary_rows.append({
                "Группа": lbl,
                "n": nr.n,
                "Среднее": round(float(a_s.mean() if lbl == label_a else b_s.mean()), 4),
                "Медиана": round(float(a_s.median() if lbl == label_a else b_s.median()), 4),
                "Std": round(float(a_s.std() if lbl == label_a else b_s.std()), 4),
                "Асимметрия": nr.skewness,
                "Эксцесс": nr.kurtosis,
                "Тест норм-ти": nr.test_name,
                "p-value": nr.p_value,
                "Нормальное?": "✅ Да" if nr.is_normal else "❌ Нет",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ===========================================================================
# t-Test
# ===========================================================================
with tab_ttest:
    section_header("Независимые выборки — t-тест Уэлча")
    st.info(
        "**H₀:** Средние двух групп равны. "
        "Тест Уэлча не требует равенства дисперсий. "
        "Предполагает приблизительную нормальность в каждой группе."
    )
    if not num_cols or not cat_cols:
        st.warning("Нужна числовая и категориальная колонки.")
    else:
        c1, c2 = st.columns(2)
        tt_val = c1.selectbox("Числовая колонка", num_cols, key="tt_val")
        tt_grp = c2.selectbox("Группирующая колонка", cat_cols, key="tt_grp")
        groups = df[tt_grp].dropna().unique().tolist()
        if len(groups) >= 2:
            c3, c4 = st.columns(2)
            tt_ga = c3.selectbox("Группа A", groups, key="tt_ga")
            tt_gb = c4.selectbox("Группа B", [g for g in groups if g != tt_ga], key="tt_gb")
            equal_var = st.checkbox("Предполагать равные дисперсии (Стьюдент)", key="tt_eqvar")
            if st.button("▶ Запустить t-тест", type="primary", key="btn_ttest"):
                try:
                    a = df[df[tt_grp] == tt_ga][tt_val].dropna()
                    b = df[df[tt_grp] == tt_gb][tt_val].dropna()
                    result = ttest_independent(a, b, alpha=alpha, equal_var=equal_var,
                                              label_a=str(tt_ga), label_b=str(tt_gb))
                    _result_card(result)
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.plotly_chart(_overlay_hist(df, tt_grp, tt_val, tt_ga, tt_gb),
                                        use_container_width=True)
                    with col2:
                        st.plotly_chart(_ci_bars(df, tt_grp, tt_val, tt_ga, tt_gb),
                                        use_container_width=True)
                    # Violin
                    sub = df[df[tt_grp].isin([tt_ga, tt_gb])][[tt_grp, tt_val]].dropna()
                    fig_v = px.violin(sub, x=tt_grp, y=tt_val, box=True, color=tt_grp,
                                     points="outliers",
                                     title=f"Violin + Box: «{tt_val}»",
                                     labels={tt_grp: "Группа", tt_val: tt_val})
                    fig_v.update_layout(template="plotly_white", showlegend=False)
                    st.plotly_chart(fig_v, use_container_width=True)
                    # Effect gauge
                    try:
                        g = _gauge(result.effect_size, [0.2, 0.5, 0.8], "Cohen's d")
                        if g:
                            st.plotly_chart(g, use_container_width=True)
                            st.caption("Cohen's d: 0.2 = малый | 0.5 = средний | 0.8 = большой")
                    except Exception:
                        pass
                    # Interpretation box
                    try:
                        p_val = float(result.p_value)
                        interp_parts = interpret_pvalue(p_val, alpha=alpha)
                        if result.effect_size is not None:
                            interp_parts += "\n\n" + interpret_effect_size(float(result.effect_size), "cohen_d")
                        interpretation_box("Интерпретация результата", interp_parts, icon="📊", collapsed=False)
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Ошибка t-теста: {e}")
        else:
            st.warning("Группирующая колонка должна иметь минимум 2 значения.")


# ===========================================================================
# Mann-Whitney
# ===========================================================================
with tab_mw:
    section_header("U-тест Манна-Уитни (непараметрический)")
    st.info(
        "**H₀:** Случайная вероятность превосходства группы A над B равна 0.5. "
        "Не предполагает нормальности. Cliff's δ показывает размер эффекта."
    )
    if not num_cols or not cat_cols:
        st.warning("Нужна числовая и категориальная колонки.")
    else:
        c1, c2 = st.columns(2)
        mw_val = c1.selectbox("Числовая колонка", num_cols, key="mw_val")
        mw_grp = c2.selectbox("Группирующая колонка", cat_cols, key="mw_grp")
        groups_mw = df[mw_grp].dropna().unique().tolist()
        if len(groups_mw) >= 2:
            c3, c4 = st.columns(2)
            mw_ga = c3.selectbox("Группа A", groups_mw, key="mw_ga")
            mw_gb = c4.selectbox("Группа B", [g for g in groups_mw if g != mw_ga], key="mw_gb")
            if st.button("▶ Запустить Манна-Уитни", type="primary", key="btn_mw"):
                try:
                    a = df[df[mw_grp] == mw_ga][mw_val].dropna()
                    b = df[df[mw_grp] == mw_gb][mw_val].dropna()
                    result = mann_whitney(a, b, alpha=alpha,
                                         label_a=str(mw_ga), label_b=str(mw_gb))
                    _result_card(result)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(_overlay_hist(df, mw_grp, mw_val, mw_ga, mw_gb),
                                        use_container_width=True)
                    with col2:
                        # ECDF
                        fig_ecdf = go.Figure()
                        for series, name, color in [(a, str(mw_ga), "#3498db"),
                                                     (b, str(mw_gb), "#e74c3c")]:
                            ss = np.sort(series)
                            ec = np.arange(1, len(ss)+1) / len(ss)
                            fig_ecdf.add_trace(go.Scatter(x=ss, y=ec, mode="lines",
                                                          name=name,
                                                          line=dict(color=color, width=2.5)))
                        fig_ecdf.update_layout(title="ECDF",
                                               xaxis_title=mw_val, yaxis_title="F(x)",
                                               template="plotly_white",
                                               legend=dict(orientation="h", y=-0.2))
                        st.plotly_chart(fig_ecdf, use_container_width=True)
                    # Cliff's delta
                    try:
                        d_result = cliffs_delta(a, b, label_a=str(mw_ga), label_b=str(mw_gb))
                        g = _gauge(d_result.effect_size, [0.147, 0.33, 0.474], "Cliff's δ")
                        if g:
                            st.plotly_chart(g, use_container_width=True)
                            st.caption("Cliff's δ: 0.147 = мал. | 0.33 = средн. | 0.474 = большой")
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Ошибка: {e}")
        else:
            st.warning("Нужно минимум 2 группы.")


# ===========================================================================
# Chi-Square
# ===========================================================================
with tab_chi2:
    section_header("Хи-квадрат тест независимости")
    st.info("**H₀:** Две категориальные переменные независимы (нет ассоциации).")
    if len(cat_cols) < 2:
        st.warning("Нужно минимум 2 категориальные колонки.")
    else:
        c1, c2 = st.columns(2)
        chi_a = c1.selectbox("Переменная A", cat_cols, key="chi_a")
        chi_b = c2.selectbox("Переменная B", [c for c in cat_cols if c != chi_a], key="chi_b")
        if st.button("▶ Запустить χ²-тест", type="primary", key="btn_chi2"):
            try:
                result = chi_square_independence(df, chi_a, chi_b, alpha=alpha)
                _result_card(result)
                ct = pd.crosstab(df[chi_a].astype(str), df[chi_b].astype(str))
                col1, col2 = st.columns(2)
                with col1:
                    fig_ct = go.Figure(go.Heatmap(
                        z=ct.values, x=ct.columns.tolist(), y=ct.index.tolist(),
                        colorscale="Blues", texttemplate="%{z}", textfont=dict(size=12),
                    ))
                    fig_ct.update_layout(
                        title=f"Наблюдаемые частоты: {chi_a} × {chi_b}",
                        xaxis_title=chi_b, yaxis_title=chi_a, template="plotly_white")
                    st.plotly_chart(fig_ct, use_container_width=True)
                with col2:
                    ct_norm = ct.div(ct.sum(axis=1), axis=0).round(3)
                    fig_norm = go.Figure(go.Heatmap(
                        z=ct_norm.values, x=ct_norm.columns.tolist(), y=ct_norm.index.tolist(),
                        colorscale="RdYlGn", zmin=0, zmax=1,
                        texttemplate="%{z:.1%}", textfont=dict(size=12),
                    ))
                    fig_norm.update_layout(
                        title="Доля по строкам (%)",
                        xaxis_title=chi_b, yaxis_title=chi_a, template="plotly_white")
                    st.plotly_chart(fig_norm, use_container_width=True)
                # Grouped bar
                fig_bar = px.bar(
                    ct.reset_index().melt(id_vars=chi_a),
                    x=chi_a, y="value", color="variable", barmode="group",
                    title=f"Частоты: {chi_a} vs {chi_b}",
                    labels={"value": "Количество", "variable": chi_b},
                )
                fig_bar.update_layout(template="plotly_white")
                st.plotly_chart(fig_bar, use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка χ²-теста: {e}")


# ===========================================================================
# Correlation
# ===========================================================================
with tab_corr:
    section_header("Тест значимости корреляции")
    st.info(
        "**H₀:** Корреляция между переменными равна нулю. "
        "Пирсон — линейная связь. Спирмен — ранговая, устойчива к выбросам."
    )
    if len(num_cols) < 2:
        st.warning("Нужно минимум 2 числовых колонки.")
    else:
        c1, c2, c3 = st.columns(3)
        corr_x = c1.selectbox("X", num_cols, key="corr_x")
        corr_y = c2.selectbox("Y", num_cols, index=min(1, len(num_cols)-1), key="corr_y")
        corr_method = c3.selectbox("Метод", ["pearson", "spearman"], key="corr_test_method")
        if st.button("▶ Запустить тест", type="primary", key="btn_corr"):
            try:
                result = correlation_test(df[corr_x], df[corr_y], method=corr_method,
                                          alpha=alpha, label_x=corr_x, label_y=corr_y)
                _result_card(result)
                sub = df[[corr_x, corr_y]].dropna()
                col1, col2 = st.columns(2)
                with col1:
                    fig_sc = px.scatter(sub, x=corr_x, y=corr_y, trendline="ols",
                                        opacity=0.6,
                                        title=f"Scatter + OLS: {corr_x} vs {corr_y}")
                    fig_sc.update_layout(template="plotly_white")
                    st.plotly_chart(fig_sc, use_container_width=True)
                with col2:
                    x_v = sub[corr_x].values
                    y_v = sub[corr_y].values
                    coeffs = np.polyfit(x_v, y_v, 1)
                    residuals = y_v - np.polyval(coeffs, x_v)
                    y_pred = np.polyval(coeffs, x_v)
                    fig_res = go.Figure()
                    fig_res.add_trace(go.Scatter(
                        x=y_pred, y=residuals, mode="markers",
                        marker=dict(color="#3498db", opacity=0.6, size=5)))
                    fig_res.add_hline(y=0, line_dash="dash", line_color="#e74c3c")
                    fig_res.update_layout(
                        title="Остатки OLS",
                        xaxis_title="Предсказанные", yaxis_title="Остатки",
                        template="plotly_white")
                    st.plotly_chart(fig_res, use_container_width=True)
                try:
                    g = _gauge(result.effect_size, [0.1, 0.3, 0.5], f"r ({corr_method})")
                    if g:
                        st.plotly_chart(g, use_container_width=True)
                        st.caption("r: 0.1 = слабая | 0.3 = умеренная | 0.5 = сильная")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Ошибка: {e}")


# ===========================================================================
# Bootstrap
# ===========================================================================
with tab_boot:
    section_header("Бутстрап / Перестановочный тест")
    st.info(
        "Непараметрические тесты без допущений о распределении. "
        "**Перестановка**: перемешивает метки групп N раз и смотрит, "
        "как часто случайная разность не меньше наблюдаемой."
    )
    if not num_cols or not cat_cols:
        st.warning("Нужна числовая и категориальная колонки.")
    else:
        c1, c2 = st.columns(2)
        bt_val = c1.selectbox("Числовая колонка", num_cols, key="bt_val")
        bt_grp = c2.selectbox("Группирующая колонка", cat_cols, key="bt_grp")
        c3, c4, c5 = st.columns(3)
        bt_stat = c3.radio("Статистика", ["mean", "median"], horizontal=True, key="bt_stat")
        bt_n = c4.slider("Итераций", 500, 5000, 2000, 500, key="bt_n")
        bt_mode = c5.radio("Режим", ["bootstrap", "permutation"], horizontal=True, key="bt_mode",
                            format_func=lambda x: "Бутстрап" if x == "bootstrap" else "Перестановка")
        groups_bt = df[bt_grp].dropna().unique().tolist()
        if len(groups_bt) >= 2:
            c6, c7 = st.columns(2)
            bt_ga = c6.selectbox("Группа A", groups_bt, key="bt_ga")
            bt_gb = c7.selectbox("Группа B", [g for g in groups_bt if g != bt_ga], key="bt_gb")
            if st.button("▶ Запустить", type="primary", key="btn_boot"):
                with st.spinner(f"Запускаем {bt_n} итераций..."):
                    try:
                        a = df[df[bt_grp] == bt_ga][bt_val].dropna()
                        b = df[df[bt_grp] == bt_gb][bt_val].dropna()
                        if bt_mode == "permutation":
                            result = permutation_test(a, b, statistic=bt_stat,
                                                       n_permutations=bt_n, alpha=alpha,
                                                       label_a=str(bt_ga), label_b=str(bt_gb))
                        else:
                            result = bootstrap_test(a, b, statistic=bt_stat,
                                                    n_bootstrap=bt_n, alpha=alpha,
                                                    label_a=str(bt_ga), label_b=str(bt_gb))
                        _result_card(result)
                        # Null distribution
                        if hasattr(result, "details") and result.details:
                            null_dist = (result.details.get("null_distribution") or
                                         result.details.get("permutation_distribution"))
                            observed = result.details.get("observed_statistic")
                            if null_dist is not None and observed is not None:
                                fig_null = go.Figure()
                                fig_null.add_trace(go.Histogram(
                                    x=null_dist, nbinsx=60,
                                    marker_color="#3498db", opacity=0.7,
                                    histnorm="probability density",
                                    name="Нулевое распределение",
                                ))
                                fig_null.add_vline(x=float(observed), line_dash="dash",
                                                   line_color="#e74c3c",
                                                   annotation_text=f"Наблюдаемая: {observed:.4f}")
                                fig_null.add_vline(x=-abs(float(observed)), line_dash="dash",
                                                   line_color="#e74c3c")
                                fig_null.update_layout(
                                    title="Нулевое распределение разностей",
                                    xaxis_title="Разность статистик",
                                    yaxis_title="Плотность",
                                    template="plotly_white",
                                )
                                st.plotly_chart(fig_null, use_container_width=True)
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
        else:
            st.warning("Нужно минимум 2 группы.")


# ===========================================================================
# A/B Test
# ===========================================================================
with tab_ab:
    section_header("A/B тестирование — комплексный анализ")
    st.info(
        "Запускает t-тест, Манна-Уитни и бутстрап одновременно. "
        "Согласованность 3/3 тестов — признак надёжного результата."
    )
    if not num_cols or not cat_cols:
        st.warning("Нужна числовая и категориальная колонки.")
    else:
        c1, c2 = st.columns(2)
        ab_val = c1.selectbox("Метрика", num_cols, key="ab_val")
        ab_grp = c2.selectbox("Группа", cat_cols, key="ab_grp")
        groups_ab = df[ab_grp].dropna().unique().tolist()
        if len(groups_ab) >= 2:
            c3, c4 = st.columns(2)
            ab_ctrl = c3.selectbox("Контроль (A)", groups_ab, key="ab_ctrl")
            ab_trt = c4.selectbox("Тест (B)", [g for g in groups_ab if g != ab_ctrl], key="ab_trt")
            if st.button("▶ Запустить A/B тест", type="primary", key="btn_ab"):
                try:
                    ctrl = df[df[ab_grp] == ab_ctrl][ab_val].dropna()
                    trt = df[df[ab_grp] == ab_trt][ab_val].dropna()
                    res = ab_test(ctrl, trt, alpha=alpha,
                                  label_ctrl=str(ab_ctrl), label_trt=str(ab_trt))
                    lift_pct = res["lift_pct"]
                    color = "#d4edda" if lift_pct > 0 else "#f8d7da"
                    st.markdown(
                        f'<div style="background:{color};border-radius:8px;'
                        f'padding:16px;font-size:16px;font-weight:bold;">'
                        f'{"🔺" if lift_pct > 0 else "🔻"} {res["summary"]}</div>',
                        unsafe_allow_html=True,
                    )
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Контроль (A)", f"{ctrl.mean():.4f}",
                              help=f"n={len(ctrl):,}, std={ctrl.std():.4f}")
                    m2.metric("Тест (B)", f"{trt.mean():.4f}",
                              help=f"n={len(trt):,}, std={trt.std():.4f}")
                    m3.metric("Абсолютный лифт", f"{trt.mean()-ctrl.mean():+.4f}")
                    m4.metric("Относительный лифт", f"{lift_pct:+.2f}%")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(_overlay_hist(df, ab_grp, ab_val, ab_ctrl, ab_trt),
                                        use_container_width=True)
                    with col2:
                        st.plotly_chart(_ci_bars(df, ab_grp, ab_val, ab_ctrl, ab_trt),
                                        use_container_width=True)

                    section_header("Детали трёх тестов")
                    for sub_res in [res["ttest"], res["mann_whitney"], res["bootstrap"]]:
                        with st.expander(f"{'✅' if sub_res.significant else '❌'} {sub_res.name}"):
                            _result_card(sub_res)
                    votes = sum([res["ttest"].significant,
                                 res["mann_whitney"].significant,
                                 res["bootstrap"].significant])
                    if votes == 3:
                        st.success("✅ Все 3 теста подтверждают значимость — результат надёжен.")
                    elif votes == 2:
                        st.warning("⚠️ 2/3 тестов значимы — нужно больше данных для уверенности.")
                    else:
                        st.error("❌ Менее 2 тестов значимы — эффект статистически не подтверждён.")
                except Exception as e:
                    st.error(f"Ошибка A/B теста: {e}")
        else:
            st.warning("Нужно минимум 2 группы.")


# ===========================================================================
# Multiple Comparisons
# ===========================================================================
with tab_multi:
    section_header("Поправка на множественные сравнения (BH/FDR)")
    st.info(
        "При N тестах вероятность хотя бы одного ложного позитива растёт. "
        "**Бенджамини-Хохберг (BH)** контролирует ожидаемую долю ложных открытий (FDR). "
        "Применяйте при сравнении 3+ групп или метрик одновременно."
    )
    if num_cols and cat_cols:
        c1, c2 = st.columns(2)
        mc_val = c1.selectbox("Числовая колонка", num_cols, key="mc_val")
        mc_grp = c2.selectbox("Группирующая колонка", cat_cols, key="mc_grp")
        mc_fdr = st.slider("FDR порог (q)", 0.01, 0.20, 0.05, 0.01, key="mc_fdr")
        if st.button("▶ Все попарные тесты + BH", type="primary", key="btn_multi"):
            groups_all = df[mc_grp].dropna().unique().tolist()
            if len(groups_all) < 2:
                st.warning("Нужно минимум 2 группы.")
            else:
                from itertools import combinations
                pairs = list(combinations(groups_all, 2))
                raw_p, labels = [], []
                for ga, gb in pairs:
                    a = df[df[mc_grp] == ga][mc_val].dropna()
                    b = df[df[mc_grp] == gb][mc_val].dropna()
                    try:
                        r = ttest_independent(a, b, alpha=mc_fdr,
                                              label_a=str(ga), label_b=str(gb))
                        raw_p.append(float(r.p_value))
                    except Exception:
                        raw_p.append(1.0)
                    labels.append(f"{ga} vs {gb}")
                bh = bh_correction(raw_p, alpha=mc_fdr)
                bh_df = pd.DataFrame({
                    "Пара": labels,
                    "p (сырое)": raw_p,
                    "p_adj (BH)": [r["adjusted_p"] for r in bh],
                    "Значимо": [r["significant"] for r in bh],
                    "Ранг": [r["rank"] for r in bh],
                }).sort_values("p (сырое)")
                st.dataframe(
                    bh_df.style.applymap(
                        lambda v: "background-color:#d4edda" if v is True else
                                  ("background-color:#f8d7da" if v is False else ""),
                        subset=["Значимо"],
                    ), use_container_width=True)
                # Forest plot
                fig_f = go.Figure()
                colors_f = ["#2ecc71" if s else "#e74c3c" for s in bh_df["Значимо"]]
                fig_f.add_trace(go.Bar(
                    x=-np.log10(np.array(bh_df["p (сырое)"]).clip(1e-10)),
                    y=bh_df["Пара"], orientation="h",
                    marker_color=colors_f, name="-log10(p)",
                ))
                fig_f.add_vline(x=-np.log10(mc_fdr), line_dash="dash",
                                line_color="#e74c3c",
                                annotation_text=f"q={mc_fdr}")
                fig_f.update_layout(
                    title="−log₁₀(p) по попарным сравнениям (зелёный = значимо после BH)",
                    xaxis_title="−log₁₀(p-value)", template="plotly_white")
                st.plotly_chart(fig_f, use_container_width=True)
                n_sig = bh_df["Значимо"].sum()
                st.success(f"После BH: {n_sig} из {len(pairs)} пар значимы (FDR ≤ {mc_fdr})")
                st.download_button("⬇ Скачать CSV",
                                   bh_df.to_csv(index=False).encode(),
                                   file_name="bh_results.csv")
    else:
        st.warning("Нужна числовая и категориальная колонки.")


# ===========================================================================
# Power Analysis
# ===========================================================================
with tab_power:
    section_header("⚡ Анализ мощности (Power Analysis)")
    st.markdown(
        "Рассчитайте **необходимый размер выборки**, **мощность теста** или "
        "**минимально обнаруживаемый эффект (MDE)** для корректного планирования экспериментов."
    )

    _power_ok = _STATSMODELS_OK
    if not _power_ok:
        st.error("Модуль statsmodels недоступен. Запустите: `pip install statsmodels`")

    if _power_ok:
        # ---- concept explanation ----
        with st.expander("📖 Как читать результаты (нажмите, чтобы раскрыть)", expanded=False):
            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.info("**α (alpha)**\nВероятность ложного позитива (ошибка I рода). Стандарт: 0.05")
            ec2.info("**β (beta)**\nВероятность ложного негатива (ошибка II рода). Стандарт: 0.20")
            ec3.info("**Мощность (1−β)**\nВероятность обнаружить реальный эффект. Цель: ≥ 0.80")
            ec4.info("**Размер эффекта d**\nКоэн: малый=0.2, средний=0.5, большой=0.8")

        pw_mode = st.radio(
            "Что хотите рассчитать?",
            ["📏 Размер выборки", "💪 Мощность теста", "🎯 Мин. обнаруживаемый эффект (MDE)"],
            horizontal=True,
            key="pw_mode",
            help="Выберите параметр для расчёта — два остальных зафиксируйте вручную."
        )

        pw_test = st.selectbox(
            "Тип теста",
            [
                "Двухвыборочный t-тест (независимые группы)",
                "Однвыборочный t-тест",
                "Тест для двух пропорций (конверсии, A/B)",
                "ANOVA (несколько групп)",
            ],
            key="pw_test",
            help="Выберите тип статистического теста, мощность которого нужно рассчитать."
        )

        pw_col1, pw_col2, pw_col3 = st.columns(3)

        with pw_col1:
            pw_alpha = st.slider(
                "Уровень значимости α",
                0.01, 0.10, float(st.session_state.get("alpha", 0.05)), 0.01,
                key="pw_alpha",
                help="Порог для отклонения нулевой гипотезы. Стандарт: 0.05"
            )

        with pw_col2:
            if pw_mode != "💪 Мощность теста":
                pw_power = st.slider(
                    "Целевая мощность (1−β)",
                    0.50, 0.99, 0.80, 0.01,
                    key="pw_power",
                    help="Желаемая вероятность обнаружить эффект. Рекомендуется ≥ 0.80"
                )
            else:
                pw_power = None

        with pw_col3:
            if pw_mode != "🎯 Мин. обнаруживаемый эффект (MDE)":
                if pw_test.startswith("Тест для двух пропорций"):
                    pw_p1 = st.number_input(
                        "Конверсия группы A (%)", 0.1, 99.9, 10.0, 0.5,
                        key="pw_p1",
                        help="Базовая конверсия контрольной группы, в процентах."
                    ) / 100
                    pw_p2 = st.number_input(
                        "Конверсия группы B (%)", 0.1, 99.9, 12.0, 0.5,
                        key="pw_p2",
                        help="Ожидаемая конверсия экспериментальной группы, в процентах."
                    ) / 100
                    pw_effect = proportion_effectsize(pw_p1, pw_p2)
                    st.caption(f"Cohen's h = **{pw_effect:.3f}** ({('малый' if abs(pw_effect)<0.3 else 'средний' if abs(pw_effect)<0.5 else 'большой')})")
                    pw_effect_label = "Cohen's h"
                else:
                    pw_effect = st.slider(
                        "Размер эффекта (Cohen's d)",
                        0.10, 2.00, 0.50, 0.05,
                        key="pw_effect",
                        help="0.2=малый, 0.5=средний, 0.8=большой. Оцените по прошлым данным или экспертно."
                    )
                    pw_effect_label = "Cohen's d"
            else:
                pw_effect = None
                pw_effect_label = "Cohen's d"

        if pw_test == "ANOVA (несколько групп)":
            pw_k = st.number_input(
                "Количество групп (k)", 2, 20, 3, 1,
                key="pw_k",
                help="Число сравниваемых групп. Для двух групп используйте t-тест."
            )
        else:
            pw_k = 2

        if pw_mode != "📏 Размер выборки":
            pw_n = st.number_input(
                "Размер выборки (n на группу)", 10, 100000, 100, 10,
                key="pw_n",
                help="Число наблюдений в каждой группе. При дисбалансе используйте меньшую группу."
            )

        if st.button("▶ Рассчитать", key="btn_pw_calc", type="primary"):
            try:
                # select analysis object
                if pw_test.startswith("Двух"):
                    analysis = TTestIndPower()
                elif pw_test.startswith("Одн"):
                    analysis = TTestOneSampPower()
                elif pw_test.startswith("Тест для двух пропорций"):
                    analysis = NormalIndPower()
                else:  # ANOVA
                    analysis = FTestAnovaPower()

                # --- CALCULATE ---
                if pw_mode == "📏 Размер выборки":
                    if pw_test == "ANOVA (несколько групп)":
                        n_result = analysis.solve_power(
                            effect_size=pw_effect, alpha=pw_alpha, power=pw_power, k_groups=pw_k
                        )
                    else:
                        n_result = analysis.solve_power(
                            effect_size=pw_effect, alpha=pw_alpha, power=pw_power
                        )
                    n_result = int(np.ceil(n_result))
                    total = n_result * pw_k

                    # big result
                    r1, r2, r3 = st.columns(3)
                    r1.metric("На группу", f"{n_result:,}", help="Минимальное число наблюдений на каждую группу")
                    r2.metric("Всего наблюдений", f"{total:,}", help=f"{n_result:,} × {pw_k} групп")
                    r3.metric("Реальная мощность", f"{pw_power:.0%}")

                    # interpretation
                    if n_result <= 30:
                        st.success(f"✅ Небольшая выборка ({n_result} на группу) — тест будет быстрым и экономным.")
                    elif n_result <= 300:
                        st.info(f"ℹ️ Умеренная выборка ({n_result} на группу) — стандартный эксперимент.")
                    else:
                        st.warning(f"⚠️ Большая выборка ({n_result} на группу) — возможно, эффект слишком мал или α/β слишком строги. Пересмотрите параметры.")

                elif pw_mode == "💪 Мощность теста":
                    if pw_test == "ANOVA (несколько групп)":
                        power_result = analysis.solve_power(
                            effect_size=pw_effect, alpha=pw_alpha, nobs=pw_n, k_groups=pw_k
                        )
                    else:
                        power_result = analysis.solve_power(
                            effect_size=pw_effect, alpha=pw_alpha, nobs=pw_n
                        )

                    r1, r2, r3 = st.columns(3)
                    r1.metric("Мощность (1−β)", f"{power_result:.1%}")
                    r2.metric("Риск пропустить эффект (β)", f"{1-power_result:.1%}")
                    r3.metric("n на группу", f"{pw_n:,}")

                    if power_result >= 0.80:
                        st.success(f"✅ Мощность {power_result:.1%} — тест надёжен, эффект будет обнаружен с высокой вероятностью.")
                    elif power_result >= 0.60:
                        st.warning(f"⚠️ Мощность {power_result:.1%} — ниже рекомендуемого порога 80%. Увеличьте выборку.")
                    else:
                        st.error(f"❌ Мощность {power_result:.1%} — тест слишком слаб. Результаты будут ненадёжными.")

                else:  # MDE
                    if pw_test == "ANOVA (несколько групп)":
                        mde_result = analysis.solve_power(
                            alpha=pw_alpha, power=pw_power, nobs=pw_n, k_groups=pw_k
                        )
                    else:
                        mde_result = analysis.solve_power(
                            alpha=pw_alpha, power=pw_power, nobs=pw_n
                        )

                    r1, r2, r3 = st.columns(3)
                    r1.metric("MDE (Cohen's d)", f"{mde_result:.3f}")
                    r2.metric("Размер эффекта", "малый" if mde_result < 0.3 else ("средний" if mde_result < 0.7 else "большой"))
                    r3.metric("n на группу", f"{pw_n:,}")

                    st.info(
                        f"При n={pw_n} на группу, α={pw_alpha} и мощности {pw_power:.0%} "
                        f"тест обнаружит эффект размером **≥ {mde_result:.3f}** (Cohen's d). "
                        f"Меньший эффект может быть пропущен."
                    )

                # ================================================================
                # VISUALIZATIONS
                # ================================================================
                st.divider()
                viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                    "📈 Кривая мощности", "🔢 Тепловая карта n×эффект", "📊 Сравнение α"
                ])

                with viz_tab1:
                    # power curve: power vs n
                    ns = np.arange(10, 501, 5)
                    d_vals = [0.2, 0.35, 0.5, 0.65, 0.8] if pw_effect_label != "Cohen's h" else [0.1, 0.2, 0.3, 0.5, 0.8]
                    fig_pc = go.Figure()
                    colors_d = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
                    for d_v, col_d in zip(d_vals, colors_d):
                        pows = []
                        for n_v in ns:
                            try:
                                if pw_test == "ANOVA (несколько групп)":
                                    p = analysis.solve_power(effect_size=d_v, alpha=pw_alpha, nobs=n_v, k_groups=pw_k)
                                else:
                                    p = analysis.solve_power(effect_size=d_v, alpha=pw_alpha, nobs=n_v)
                                pows.append(min(p, 1.0))
                            except Exception:
                                pows.append(np.nan)
                        lbl = f"малый ({d_v})" if d_v == 0.2 else (f"средний ({d_v})" if d_v == 0.5 else (f"большой ({d_v})" if d_v == 0.8 else str(d_v)))
                        fig_pc.add_trace(go.Scatter(x=ns, y=pows, name=lbl, line=dict(color=col_d, width=2)))

                    fig_pc.add_hline(y=0.80, line_dash="dash", line_color="#7f8c8d",
                                     annotation_text="Порог 80%", annotation_position="right")
                    fig_pc.add_hline(y=0.90, line_dash="dot", line_color="#95a5a6",
                                     annotation_text="Порог 90%", annotation_position="right")
                    fig_pc.update_layout(
                        title=f"Кривая мощности при α={pw_alpha}",
                        xaxis_title="Размер выборки (n на группу)",
                        yaxis_title="Мощность (1−β)",
                        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                        template="plotly_white",
                        legend_title="Размер эффекта",
                        height=420,
                    )
                    st.plotly_chart(fig_pc, use_container_width=True)
                    st.caption("Каждая линия — отдельный размер эффекта. Пересечение с пунктиром 80% = минимальная рекомендуемая выборка.")

                with viz_tab2:
                    # heatmap: n x effect_size -> power
                    heat_ns = np.arange(20, 401, 20)
                    heat_ds = np.round(np.arange(0.10, 1.05, 0.10), 2)
                    heat_z = []
                    for d_v in heat_ds:
                        row = []
                        for n_v in heat_ns:
                            try:
                                if pw_test == "ANOVA (несколько групп)":
                                    p = analysis.solve_power(effect_size=d_v, alpha=pw_alpha, nobs=n_v, k_groups=pw_k)
                                else:
                                    p = analysis.solve_power(effect_size=d_v, alpha=pw_alpha, nobs=n_v)
                                row.append(round(min(p, 1.0), 3))
                            except Exception:
                                row.append(np.nan)
                        heat_z.append(row)

                    fig_hm = go.Figure(go.Heatmap(
                        z=heat_z,
                        x=[str(n) for n in heat_ns],
                        y=[str(d) for d in heat_ds],
                        colorscale=[
                            [0.0, "#e74c3c"], [0.5, "#f39c12"],
                            [0.7, "#f1c40f"], [0.8, "#2ecc71"], [1.0, "#1a5276"]
                        ],
                        zmin=0, zmax=1,
                        text=[[f"{v:.0%}" for v in row] for row in heat_z],
                        texttemplate="%{text}",
                        textfont=dict(size=9),
                        colorbar=dict(title="Мощность", tickformat=".0%"),
                    ))
                    fig_hm.update_layout(
                        title=f"Мощность по n и размеру эффекта (α={pw_alpha})",
                        xaxis_title="n на группу",
                        yaxis_title="Размер эффекта (Cohen's d)",
                        template="plotly_white",
                        height=420,
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)
                    st.caption("🟢 Зелёный — мощность ≥80% (надёжно). 🟡 Жёлтый — 60–80% (риск). 🔴 Красный — <50% (ненадёжно).")

                with viz_tab3:
                    # compare alpha levels
                    alphas_cmp = [0.01, 0.05, 0.10]
                    ds_cmp = np.round(np.arange(0.1, 1.05, 0.05), 2)
                    fig_alpha = go.Figure()
                    cmp_colors = {"0.01": "#3498db", "0.05": "#2ecc71", "0.10": "#e74c3c"}
                    n_fixed = pw_n if pw_mode != "📏 Размер выборки" else n_result

                    for a_v in alphas_cmp:
                        pows_a = []
                        for d_v in ds_cmp:
                            try:
                                if pw_test == "ANOVA (несколько групп)":
                                    p = analysis.solve_power(effect_size=d_v, alpha=a_v, nobs=n_fixed, k_groups=pw_k)
                                else:
                                    p = analysis.solve_power(effect_size=d_v, alpha=a_v, nobs=n_fixed)
                                pows_a.append(min(p, 1.0))
                            except Exception:
                                pows_a.append(np.nan)
                        fig_alpha.add_trace(go.Scatter(
                            x=ds_cmp, y=pows_a,
                            name=f"α = {a_v}",
                            line=dict(color=cmp_colors[str(a_v)], width=2.5)
                        ))

                    fig_alpha.add_hline(y=0.80, line_dash="dash", line_color="#7f8c8d",
                                        annotation_text="80%")
                    fig_alpha.update_layout(
                        title=f"Влияние α на мощность (n={n_fixed} на группу)",
                        xaxis_title="Размер эффекта (Cohen's d)",
                        yaxis_title="Мощность (1−β)",
                        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                        template="plotly_white",
                        legend_title="Уровень α",
                        height=420,
                    )
                    st.plotly_chart(fig_alpha, use_container_width=True)
                    st.caption("Более мягкий α (0.10) даёт бо́льшую мощность, но увеличивает риск ложного позитива. Выбирайте по контексту задачи.")

                # --- data quality warning for low sample ----
                if len(df) < 30:
                    st.warning("⚠️ В датасете менее 30 строк — оценки размера эффекта могут быть нестабильными.")

            except Exception as e:
                st.error(f"Ошибка расчёта: {e}. Проверьте параметры — некоторые комбинации (например, эффект=0) не допустимы.")

        # --- auto-suggest from data ---
        st.divider()
        section_header("Автооценка эффекта по данным", "🔎")
        st.caption("Выберите две группы из датасета — KIBAD рассчитает реальный размер эффекта и оценит мощность.")

        ae_col1, ae_col2, ae_col3 = st.columns(3)
        ae_val = ae_col1.selectbox("Числовая переменная", num_cols, key="pw_ae_val",
                                    help="Метрика, которую сравниваем между группами")
        ae_grp = ae_col2.selectbox("Группировка", cat_cols, key="pw_ae_grp",
                                    help="Категориальный столбец с группами A и B") if cat_cols else None
        ae_n_override = ae_col3.number_input(
            "Планируемый n на группу (для оценки)", 10, 10000, 100, 10,
            key="pw_ae_n",
            help="Сколько наблюдений планируете собрать — для расчёта ожидаемой мощности"
        )

        if ae_grp and ae_val and st.button("🔎 Оценить по данным", key="btn_pw_auto"):
            grp_vals = df[ae_grp].dropna().unique()
            if len(grp_vals) >= 2:
                g1 = df[df[ae_grp] == grp_vals[0]][ae_val].dropna()
                g2 = df[df[ae_grp] == grp_vals[1]][ae_val].dropna()
                pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
                if pooled_std > 0:
                    d_obs = abs(g1.mean() - g2.mean()) / pooled_std
                    analysis_ae = TTestIndPower()
                    try:
                        n_needed = int(np.ceil(analysis_ae.solve_power(
                            effect_size=d_obs, alpha=pw_alpha, power=0.80
                        )))
                        pwr_obs = analysis_ae.solve_power(
                            effect_size=d_obs, alpha=pw_alpha, nobs=ae_n_override
                        )
                        ae_c1, ae_c2, ae_c3, ae_c4 = st.columns(4)
                        ae_c1.metric("Наблюд. Cohen's d", f"{d_obs:.3f}",
                                     help="Размер эффекта, вычисленный по текущим данным")
                        ae_c2.metric("Размер эффекта", "малый" if d_obs < 0.3 else ("средний" if d_obs < 0.7 else "большой"))
                        ae_c3.metric("Нужно n (80% мощн.)", f"{n_needed:,}",
                                     help="Минимальная выборка для надёжного обнаружения этого эффекта")
                        ae_c4.metric(f"Мощность при n={ae_n_override}", f"{pwr_obs:.1%}",
                                     delta="✅ OK" if pwr_obs >= 0.8 else "⚠️ Мало",
                                     help="Насколько надёжен тест с планируемой выборкой")
                        if pwr_obs < 0.80:
                            st.warning(
                                f"⚠️ При n={ae_n_override} мощность составит {pwr_obs:.1%}. "
                                f"Для мощности 80% нужно не менее **{n_needed}** наблюдений на группу."
                            )
                        else:
                            st.success(f"✅ Планируемая выборка достаточна — ожидаемая мощность {pwr_obs:.1%}.")
                    except Exception:
                        st.info(f"Наблюдаемый Cohen's d = **{d_obs:.3f}**. Не удалось рассчитать n — проверьте размер эффекта.")
                else:
                    st.warning("Стандартное отклонение равно нулю — невозможно рассчитать размер эффекта.")
            else:
                st.warning(f"В столбце «{ae_grp}» должно быть не менее двух групп.")


# ===========================================================================
# History
# ===========================================================================
with tab_history:
    section_header("История тестов (сессия)")
    results = st.session_state.get("test_results", [])
    if results:
        hist_df = pd.DataFrame([{
            "Тест": r.name, "Статистика": r.statistic,
            "p-value": r.p_value,
            "Значимо": "✅" if r.significant else "❌",
            "Размер эффекта": r.effect_size if r.effect_size is not None else "—",
            "Метка": r.effect_label or "—",
        } for r in results])
        st.dataframe(
            hist_df.style.applymap(
                lambda v: "background-color:#d4edda" if v == "✅" else
                          ("background-color:#f8d7da" if v == "❌" else ""),
                subset=["Значимо"],
            ), use_container_width=True)
        # p-value comparison
        if len(results) > 1:
            pv_df = pd.DataFrame({
                "Тест": [r.name for r in results],
                "p-value": [float(r.p_value) for r in results],
                "Значимо": [r.significant for r in results],
            })
            fig_pv = px.bar(pv_df, x="Тест", y="p-value",
                            color="Значимо",
                            color_discrete_map={True: "#2ecc71", False: "#e74c3c"},
                            title="p-value по всем тестам",
                            text="p-value")
            fig_pv.add_hline(y=alpha, line_dash="dash", line_color="#e74c3c",
                             annotation_text=f"α={alpha}")
            fig_pv.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_pv.update_layout(template="plotly_white")
            st.plotly_chart(fig_pv, use_container_width=True)
        st.download_button("⬇ CSV", hist_df.to_csv(index=False).encode(),
                           file_name="test_history.csv")
        if st.button("🗑 Очистить историю", key="btn_clear_tests"):
            st.session_state["test_results"] = []
            st.rerun()
    else:
        st.info("Тесты ещё не проводились.")
