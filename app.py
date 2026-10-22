"""Prototipo interactivo · ¿Entra o no entra? SVM lineal vs RBF sobre tiros de la NBA."""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ml
from ml import FEATURES, LABELS, LEAKAGE, RANGES

st.set_page_config(page_title="SVM · Tiros NBA", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Tokens de diseño
# ---------------------------------------------------------------------------
NAVY, INK2, MUTED, GRID = "#0b2545", "#4a5b73", "#7a8aa0", "#e6ebf2"
MODEL_COLOR = {"dummy": "#9aa8bc", "linear": "#3a6fd0", "rbf": "#eb6834"}
MODEL_NAME = {"dummy": "Baseline (siempre falla)", "linear": "SVM lineal", "rbf": "SVM RBF"}
MADE, MISSED = "#2a78d6", "#e34948"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, button, input, textarea, select {{ font-family: 'Inter', system-ui, sans-serif; }}
.stApp {{ background: #f4f7fa; color: {NAVY}; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1480px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
div[class*="st-key-card"] {{ background: #fff; border: 1px solid #e3e9f1; border-radius: 18px;
  box-shadow: 0 1px 2px rgba(11,37,69,.04), 0 10px 28px rgba(11,37,69,.06); padding: 1.2rem 1.4rem 1.1rem; }}
div[class*="st-key-card"] h3 {{ margin: 0 0 .1rem; padding: 0; font-size: 1.12rem; font-weight: 700; color: {NAVY}; }}
.topbar {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }}
.topbar .title {{ font-size: 1.75rem; font-weight: 700; letter-spacing: -.02em; line-height: 1.15; }}
.topbar .sub {{ font-size: .88rem; color: {INK2}; margin-top: .15rem; }}
.pills {{ display: flex; gap: .5rem; flex-wrap: wrap; }}
.pill {{ font-size: .78rem; font-weight: 500; background: #fff; border: 1px solid #dfe6ef; border-radius: 999px; padding: .32rem .8rem; }}
.pill b {{ color: #0a6b74; font-weight: 600; }}
.mgrid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin-bottom: 1rem; }}
.mcard {{ position: relative; overflow: hidden; background: #fff; border: 1px solid #e3e9f1; border-radius: 16px;
  padding: .9rem 1rem .9rem 1.2rem; box-shadow: 0 1px 2px rgba(11,37,69,.04), 0 8px 22px rgba(11,37,69,.05); }}
.mcard::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: var(--accent); }}
.mlabel {{ font-size: .72rem; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: {INK2}; }}
.mvalue {{ font-size: 2rem; font-weight: 700; line-height: 1.05; margin: .4rem 0 .1rem; letter-spacing: -.02em; white-space: nowrap; }}
.mvalue small {{ font-size: .85rem; font-weight: 500; color: {INK2}; margin-left: .25rem; }}
.msub {{ font-size: .78rem; color: {INK2}; }}
.note {{ font-size: .8rem; color: {INK2}; margin: .5rem .1rem 0; line-height: 1.45; }}
.callout {{ background: #eef5ff; border: 1px solid #d3e3fb; border-radius: 14px; padding: .8rem 1rem; font-size: .88rem; line-height: 1.5; color: {NAVY}; margin: .4rem 0 .8rem; }}
.callout b {{ color: #13407a; }}
table.cmp {{ width: 100%; border-collapse: collapse; font-size: .86rem; margin-top: .4rem; }}
table.cmp th {{ text-align: right; font-weight: 600; color: {INK2}; font-size: .7rem; letter-spacing: .05em; text-transform: uppercase; padding: .5rem .6rem; border-bottom: 1px solid #e3e9f1; }}
table.cmp td {{ text-align: right; padding: .5rem .6rem; border-bottom: 1px solid #eef2f7; font-variant-numeric: tabular-nums; }}
table.cmp th:first-child, table.cmp td:first-child, table.cmp th:nth-child(2), table.cmp td:nth-child(2) {{ text-align: left; }}
table.cmp td .sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: .5rem; }}
table.cmp td.best {{ font-weight: 700; color: #0a5f68; }}
.tablewrap {{ overflow-x: auto; }}
.result {{ border-radius: 16px; padding: 1rem 1.2rem; color: #fff; }}
.result .k {{ font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; opacity: .85; }}
.result .v {{ font-size: 2.2rem; font-weight: 700; line-height: 1.1; margin: .2rem 0; }}
.result .meta {{ font-size: .8rem; opacity: .9; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .25rem; flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; padding: .5rem .8rem; }}
.stButton > button {{ border-radius: 12px; font-weight: 600; }}
</style>
"""


# ---------------------------------------------------------------------------
# Carga (cacheada: se hace una sola vez por sesión del servidor)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando y limpiando los 128 069 tiros…")
def get_clean() -> ml.Clean:
    return ml.load_clean()


@st.cache_resource(show_spinner=False)
def get_artifacts() -> tuple[dict, dict] | None:
    path = ml.ARTIFACTS / "results.json"
    if not path.exists():
        return None
    results = json.loads(path.read_text(encoding="utf-8"))
    models = {k: joblib.load(ml.ARTIFACTS / f"{k}.joblib") for k in ("linear", "rbf")}
    return results, models


@st.cache_data(show_spinner=False)
def get_heatmap() -> pd.DataFrame:
    return ml.fg_heatmap(get_clean().data)


@st.cache_data(show_spinner=False, max_entries=64)
def get_lab(kernel: str, C: float, gamma: float) -> dict:
    t = time.perf_counter()
    lab = ml.lab_2d(get_clean().data, kernel, C, gamma)
    return {"lab": lab, "seconds": time.perf_counter() - t}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def style(fig: go.Figure, height: int = 420, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=6, r=6, t=34 if legend else 14, b=6),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=NAVY, size=13),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(size=13, color=NAVY)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=12, color=INK2)),
    )
    for ax in (fig.update_xaxes, fig.update_yaxes):
        ax(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED),
           title_font=dict(color=INK2, size=12))
    return fig


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def cards(items: list[tuple[str, str, str, str]]) -> None:
    """items = (etiqueta, valor, subtítulo, color de acento)."""
    md('<div class="mgrid">' + "".join(
        f'<div class="mcard" style="--accent:{c}"><div class="mlabel">{l}</div>'
        f'<div class="mvalue">{v}</div><div class="msub">{s}</div></div>'
        for l, v, s, c in items) + "</div>")


def n(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}<small>%</small>"


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
def header(results: dict | None) -> None:
    pills = '<span class="pill">NBA 2014-15 · 128 069 tiros</span>'
    if results:
        ds = results["dataset"]
        pills += (f'<span class="pill"><b>d = {ds["dimensions"]}</b> variables</span>'
                  f'<span class="pill">{n(ds["n_train"])} entrenamiento · {n(ds["n_test"])} prueba</span>')
    md(f"""<div class="topbar"><div>
        <div class="title">¿Entra o no entra? · SVM lineal vs kernel RBF</div>
        <div class="sub">Clasificación binaria de enceste (y = +1) vs fallo (y = −1) con Máquinas de Soporte Vectorial</div>
        </div><div class="pills">{pills}</div></div>""")


# ---------------------------------------------------------------------------
# 1 · Datos
# ---------------------------------------------------------------------------
def tab_data(clean: ml.Clean, results: dict | None) -> None:
    d = clean.data
    raw = clean.funnel[0][1]
    sample = results["dataset"] if results else None
    cards([
        ("Tiros crudos", n(raw), "shot_logs.csv · Kaggle", "#9aa8bc"),
        ("Tiros limpios", n(len(d)), f"{100 * len(d) / raw:.1f} % del total", "#0e9aa7"),
        ("Muestra SVM", n(sample["sample"]) if sample else "—",
         "estratificada · semilla 42" if sample else "corre train.py", "#3a6fd0"),
        ("Dimensiones", f"d = {len(FEATURES)}", "x ∈ ℝ⁶ (y ℝ² en el laboratorio)", "#eb6834"),
        ("Encestes", pct(d[ml.TARGET].mean()), f"baseline «siempre falla» = {100 * (1 - d[ml.TARGET].mean()):.1f} %", MADE),
    ])

    c1, c2 = st.columns([1, 1], gap="medium")
    with c1, st.container(key="card_funnel"):
        st.markdown("### Limpieza paso a paso")
        steps = [s for s, _ in clean.funnel][::-1]
        vals = [v for _, v in clean.funnel][::-1]
        fig = go.Figure(go.Bar(x=vals, y=steps, orientation="h", marker_color="#0e9aa7",
                               text=[n(v) for v in vals], textposition="inside",
                               insidetextanchor="end", textfont=dict(color="#fff")))
        fig.update_xaxes(range=[min(vals) * 0.9, max(vals) * 1.01], title="filas")
        show(style(fig, 260), "funnel")
        md('<div class="note"><b>SHOT_CLOCK nulo</b>: el reloj de posesión se apaga en los últimos 24 s del periodo. '
           '<b>TOUCH_TIME &lt; 0</b>: error del tracking. <b>Distancia incoherente</b>: triples a &lt; 22 ft o dobles '
           'a &gt; 23.75 ft → la distancia (nuestra variable principal) está mal medida.</div>')
    with c2, st.container(key="card_features"):
        st.markdown("### Variables del vector x")
        desc = d[FEATURES].describe().T
        rows = "".join(
            f"<tr><td>{f}</td><td>{LABELS[f]}</td><td>{desc.loc[f, 'min']:.1f}</td>"
            f"<td>{desc.loc[f, '50%']:.1f}</td><td>{desc.loc[f, 'max']:.1f}</td></tr>" for f in FEATURES)
        md(f'<div class="tablewrap"><table class="cmp"><tr><th>Columna</th><th>Significado</th><th>Mín</th>'
           f'<th>Mediana</th><th>Máx</th></tr>{rows}</table></div>')
        leak = "".join(f"<li><b>{k}</b> — {v}</li>" for k, v in LEAKAGE.items())
        md(f'<div class="note"><b>Excluidas por fuga de información</b> (se conocen después del tiro):'
           f'<ul style="margin:.3rem 0 0 1rem;padding:0">{leak}</ul>'
           'También se excluyen identificadores (jugador, partido, defensor). Todas las variables se '
           'estandarizan con <b>z = (x − μ)/σ</b> (μ, σ del entrenamiento) dentro del Pipeline.</div>')

    with st.container(key="card_heat"):
        st.markdown("### Por qué hace falta una frontera no lineal")
        md('<div class="callout">Mirada <b>sola</b>, la distancia del defensor casi no cambia el % de acierto. '
           'Pero <b>dentro de cada rango de distancia</b> el efecto es enorme: cerca del aro, un tiro sin marca '
           'entra muchísimo más que uno con el defensor encima. Los defensores se pegan más en los tiros cercanos '
           '(que entran más), y eso cancela el efecto en el agregado — <b>paradoja de Simpson</b>. '
           'Es una <b>interacción</b> entre variables: una recta no la modela, el kernel RBF sí.</div>')
        h1, h2 = st.columns([1.6, 1], gap="medium")
        hm = get_heatmap()
        with h1:
            piv = hm.pivot(index="dist", columns="defd", values="mean")
            cnt = hm.pivot(index="dist", columns="defd", values="size")
            fig = go.Figure(go.Heatmap(
                z=piv.values * 100, x=[f"{c.left:g}–{c.right:g} ft" for c in piv.columns],
                y=[f"{r.left:g}–{r.right:g} ft" for r in piv.index],
                colorscale=[[0, "#f6f0e6"], [0.5, "#f3a35c"], [1, "#b3361b"]],
                text=[[f"{v * 100:.0f} %<br><span style='font-size:10px'>n={int(c)}</span>"
                       for v, c in zip(rv, rc)] for rv, rc in zip(piv.values, cnt.values)],
                texttemplate="%{text}", colorbar=dict(title="FG %", ticksuffix=" %"),
                hovertemplate="Distancia %{y}<br>Defensor %{x}<br>Acierto %{z:.1f} %<extra></extra>"))
            fig.update_xaxes(title="Distancia del defensor más cercano")
            fig.update_yaxes(title="Distancia al aro", autorange="reversed")
            show(style(fig, 430), "heat")
        with h2:
            agg = d.groupby(pd.cut(d["CLOSE_DEF_DIST"], [0, 2, 4, 6, 10, 60], include_lowest=True),
                            observed=True)[ml.TARGET].mean()
            fig = go.Figure(go.Bar(x=[f"{c.left:g}–{c.right:g}" for c in agg.index], y=agg.values * 100,
                                   marker_color="#9aa8bc", text=[f"{v * 100:.0f} %" for v in agg.values],
                                   textposition="outside"))
            fig.update_yaxes(range=[0, 70], title="FG % (todos los tiros)", ticksuffix=" %")
            fig.update_xaxes(title="Distancia del defensor (ft)")
            show(style(fig, 430), "agg")

    with st.container(key="card_hist"):
        st.markdown("### Las clases se solapan")
        fig = go.Figure()
        for val, name, col in ((1, "Encestado", MADE), (0, "Fallado", MISSED)):
            fig.add_trace(go.Histogram(x=d.loc[d[ml.TARGET] == val, "SHOT_DIST"], name=name,
                                       marker_color=col, opacity=.55, xbins=dict(size=1)))
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title="Distancia al aro (ft)")
        fig.update_yaxes(title="tiros")
        show(style(fig, 320, legend=True), "hist")
        md('<div class="note">Dos picos (bajo el aro y la línea de triple) y encestes/fallos en casi todas las '
           'distancias: no existe separación perfecta → hace falta <b>margen blando</b> (C, ξ).</div>')


# ---------------------------------------------------------------------------
# 2 · Fundamento
# ---------------------------------------------------------------------------
def tab_theory(results: dict | None) -> None:
    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(key="card_t1"):
        st.markdown("### 1 · Hiperplano y margen")
        st.latex(r"f(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + b,\qquad \hat y = \operatorname{sign} f(\mathbf{x})")
        st.latex(r"\gamma_i = \frac{y_i(\mathbf{w}^\top\mathbf{x}_i + b)}{\lVert\mathbf{w}\rVert}"
                 r"\;\Rightarrow\; \text{ancho} = \frac{2}{\lVert\mathbf{w}\rVert}")
        md('<div class="note">Con la normalización canónica (punto más cercano con f = ±1), maximizar el margen '
           'equivale a minimizar ½‖w‖². <b>w</b> fija la orientación; <b>b</b> traslada la frontera.</div>')
    with c2, st.container(key="card_t2"):
        st.markdown("### 2 · Margen blando (C, ξ)")
        st.latex(r"\min_{\mathbf{w},b,\xi}\ \tfrac12\lVert\mathbf{w}\rVert^2 + C\sum_i \xi_i")
        st.latex(r"\text{s.a.}\quad y_i(\mathbf{w}^\top\mathbf{x}_i+b)\ge 1-\xi_i,\quad \xi_i\ge 0")
        st.latex(r"\equiv\ \min\ \tfrac12\lVert\mathbf{w}\rVert^2 + C\sum_i \max\bigl(0,\,1-y_i f(\mathbf{x}_i)\bigr)")
        md('<div class="note">Pérdida bisagra. <b>C pequeño</b>: errores baratos, calle ancha, subajuste. '
           '<b>C grande</b>: errores carísimos, frontera retorcida, sobreajuste.</div>')
    c3, c4 = st.columns(2, gap="medium")
    with c3, st.container(key="card_t3"):
        st.markdown("### 3 · Dual y vectores de soporte")
        st.latex(r"\max_\alpha \sum_i\alpha_i - \tfrac12\sum_{i,j}\alpha_i\alpha_j y_i y_j\,K(\mathbf{x}_i,\mathbf{x}_j)")
        st.latex(r"\text{s.a.}\quad 0\le\alpha_i\le C,\quad \sum_i\alpha_i y_i = 0")
        st.latex(r"f(\mathbf{x}) = \sum_{i\in SV}\alpha_i y_i\,K(\mathbf{x}_i,\mathbf{x}) + b")
        md('<div class="note">Solo los puntos con α<sub>i</sub> &gt; 0 (vectores de soporte) aportan a f. '
           'Los datos aparecen solo en productos punto → se pueden sustituir por un kernel.</div>')
    with c4, st.container(key="card_t4"):
        st.markdown("### 4 · Kernel RBF (truco del kernel)")
        st.latex(r"K(\mathbf{x},\mathbf{z}) = \exp\!\bigl(-\gamma\lVert\mathbf{x}-\mathbf{z}\rVert^2\bigr),"
                 r"\qquad \gamma = \frac{1}{2\sigma^2}")
        scale = f"{results['config']['gamma_scale_value']:.3f}" if results else "1/6"
        md(f'<div class="note">Similitud por distancia: 1 si los tiros son iguales, → 0 si son muy distintos. '
           f'Equivale a un espacio de <b>dimensión infinita</b> sin construirlo. <b>γ pequeño</b>: influencia amplia, '
           f'frontera suave. <b>γ grande</b>: «islas», sobreajuste. <code>gamma="scale"</code> = 1/(d·Var X) = {scale} '
           'tras estandarizar.</div>')

    with st.container(key="card_map"):
        st.markdown("### Conexión teoría ↔ aplicación")
        rows = [
            ("Vector xᵢ ∈ ℝ⁶", "Distancia al aro, distancia del defensor, reloj de posesión, botes, tiempo con el balón, periodo"),
            ("Etiqueta yᵢ", "+1 encestado · −1 fallado"),
            ("Estandarización z", "Todas las variables en la misma escala para que las distancias del kernel tengan sentido"),
            ("Hiperplano wᵀx + b = 0", "Frontera lineal (modelo lineal) entre encestes y fallos"),
            ("Holgura ξᵢ y C", "Tiros «parecidos» con resultado distinto: se permiten pero se cobran"),
            ("Kernel RBF y γ", "Frontera curva: captura la interacción distancia × defensor"),
            ("Vectores de soporte", "Los tiros ambiguos cerca de la frontera; definen el modelo"),
            ("Validación cruzada 5-fold", "Elige C y γ sin tocar el conjunto de prueba"),
        ]
        md('<div class="tablewrap"><table class="cmp"><tr><th>Concepto</th><th>En los tiros de la NBA</th></tr>'
           + "".join(f"<tr><td><b>{a}</b></td><td>{b}</td></tr>" for a, b in rows) + "</table></div>")
        st.code('pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf"))])\n'
                'grid = GridSearchCV(pipe, {"svm__C": [0.1, 1, 10, 100],\n'
                '                           "svm__gamma": [0.001, 0.01, 0.1, 1, "scale"]},\n'
                '                    cv=StratifiedKFold(5, shuffle=True, random_state=42),\n'
                '                    scoring="roc_auc")\n'
                'grid.fit(X_train, y_train)', language="python")


# ---------------------------------------------------------------------------
# 3 · Lineal vs RBF
# ---------------------------------------------------------------------------
def confusion_fig(cm: list[list[int]], color: str) -> go.Figure:
    z = np.array(cm)
    lab = [["VN", "FP"], ["FN", "VP"]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Pred. fallo", "Pred. enceste"], y=["Real fallo", "Real enceste"],
        colorscale=[[0, "#ffffff"], [1, color]], showscale=False,
        text=[[f"{lab[i][j]}<br><b>{n(z[i, j])}</b>" for j in range(2)] for i in range(2)],
        texttemplate="%{text}", textfont=dict(size=15),
        hovertemplate="%{y} · %{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return style(fig, 260)


def tab_results(results: dict) -> None:
    m = results["models"]
    lin, rbf, dum = m["linear"], m["rbf"], m["dummy"]
    cards([
        ("Accuracy RBF", pct(rbf["accuracy"]), f"lineal {100 * lin['accuracy']:.1f} % · baseline {100 * dum['accuracy']:.1f} %", MODEL_COLOR["rbf"]),
        ("ROC-AUC RBF", f"{rbf['roc_auc']:.3f}", f"lineal {lin['roc_auc']:.3f} · azar 0.500", MODEL_COLOR["rbf"]),
        ("F1 RBF", f"{rbf['f1']:.3f}", f"lineal {lin['f1']:.3f} · baseline 0.000", MODEL_COLOR["rbf"]),
        ("Entrenamiento", f"{rbf['fit_seconds']:.1f}<small>s RBF</small>", f"lineal {lin['fit_seconds']:.1f} s · {n(results['dataset']['n_train'])} tiros", MODEL_COLOR["linear"]),
        ("Vectores de soporte", pct(rbf["support_fraction"], 0), f"lineal {100 * lin['support_fraction']:.0f} % del train", MODEL_COLOR["linear"]),
    ])

    with st.container(key="card_table"):
        st.markdown("### Comparación en el conjunto de prueba")
        cols = [("accuracy", "Accuracy", True), ("precision", "Precisión", True), ("recall", "Recall", True),
                ("f1", "F1", True), ("roc_auc", "ROC-AUC", True), ("train_accuracy", "Acc. train", None),
                ("fit_seconds", "Fit (s)", False), ("predict_seconds", "Predict (ms)", False)]
        best = {k: max((m[x][k] for x in m), key=float) if hb else min(m[x][k] for x in ("linear", "rbf"))
                for k, _, hb in cols if hb is not None}
        body = ""
        for key in ("dummy", "linear", "rbf"):
            r = m[key]
            params = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            tds = ""
            for k, _, hb in cols:
                v = r[k]
                txt = f"{v * 1000:.0f}" if k == "predict_seconds" else (f"{v:.2f}" if k == "fit_seconds" else f"{v:.3f}")
                cls = ' class="best"' if hb is not None and key != "dummy" and v == best[k] else ""
                tds += f"<td{cls}>{txt}</td>"
            sv = f"{n(r['n_support'])}" if "n_support" in r else "—"
            body += (f'<tr><td><span class="sw" style="background:{MODEL_COLOR[key]}"></span>{MODEL_NAME[key]}</td>'
                     f"<td>{params}</td>{tds}<td>{sv}</td></tr>")
        head = "".join(f"<th>{name}</th>" for _, name, _ in cols)
        md(f'<div class="tablewrap"><table class="cmp"><tr><th>Modelo</th><th>Hiperparámetros</th>{head}'
           f"<th>n.º SV</th></tr>{body}</table></div>")
        md(f'<div class="note">Positivo = enceste. Test = {n(results["dataset"]["n_test"])} tiros nunca vistos '
           'durante la búsqueda. «Acc. train» cercano a la de test ⇒ no hay sobreajuste fuerte. '
           'Fit = un solo entrenamiento con los mejores hiperparámetros (1 núcleo).</div>')
        md(f'<div class="callout"><b>Lectura.</b> Ambas SVM superan al baseline. El AUC casi no cambia '
           f'({lin["roc_auc"]:.3f} lineal vs {rbf["roc_auc"]:.3f} RBF): las dos <b>ordenan</b> los tiros igual de bien. '
           f'Lo que cambia es el punto de operación del umbral f(x) = 0: el RBF es más <b>conservador</b> '
           f'(precisión {rbf["precision"]:.2f}, recall {rbf["recall"]:.2f}) y gana en accuracy; el lineal es más '
           f'equilibrado y gana en F1. La SVM lineal tarda más ({lin["fit_seconds"]:.0f} s vs {rbf["fit_seconds"]:.0f} s) porque '
           '<code>SVC(kernel="linear")</code> usa el mismo solver dual y converge lento con clases tan solapadas '
           '(~85 % de los puntos son vectores de soporte).</div>')

    c1, c2, c3 = st.columns(3, gap="medium")
    for col, key in zip((c1, c2, c3), ("dummy", "linear", "rbf")):
        with col, st.container(key=f"card_cm_{key}"):
            st.markdown(f"### {MODEL_NAME[key]}")
            show(confusion_fig(m[key]["confusion"], MODEL_COLOR[key]), f"cm_{key}")

    c4, c5 = st.columns(2, gap="medium")
    with c4, st.container(key="card_roc"):
        st.markdown("### Curvas ROC")
        fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar (AUC 0.5)",
                                   line=dict(color="#9aa8bc", dash="dash")))
        for key in ("linear", "rbf"):
            r = m[key]
            fig.add_trace(go.Scatter(x=r["roc"]["fpr"], y=r["roc"]["tpr"], mode="lines",
                                     name=f"{MODEL_NAME[key]} (AUC {r['roc_auc']:.3f})",
                                     line=dict(color=MODEL_COLOR[key], width=3)))
        fig.update_xaxes(title="Tasa de falsos positivos (1 − especificidad)", range=[0, 1])
        fig.update_yaxes(title="Tasa de verdaderos positivos (recall)", range=[0, 1])
        show(style(fig, 400, legend=True), "roc")
    with c5, st.container(key="card_scaling"):
        st.markdown("### Costo computacional: tiempo de entrenamiento vs n")
        sc = results["scaling"]
        fig = go.Figure()
        for key in ("linear", "rbf"):
            fig.add_trace(go.Scatter(x=sc["n"], y=sc[key], mode="lines+markers", name=MODEL_NAME[key],
                                     line=dict(color=MODEL_COLOR[key], width=3), marker=dict(size=8)))
        fig.update_xaxes(title="tiros de entrenamiento")
        fig.update_yaxes(title="segundos")
        show(style(fig, 400, legend=True), "scaling")
        md('<div class="note">Duplicar n multiplica el tiempo ~4–5×: el solver de SVC trabaja con la matriz de '
           'kernels n×n (≈ O(n²)–O(n³)). Por eso se usa una muestra estratificada y no los 120 000 tiros.</div>')


# ---------------------------------------------------------------------------
# 4 · GridSearch
# ---------------------------------------------------------------------------
def tab_grid(results: dict) -> None:
    sr = results["search"]
    rbf = pd.DataFrame(sr["rbf"]["table"])
    lin = pd.DataFrame(sr["linear"]["table"])
    bp_r = {k.replace("svm__", ""): v for k, v in sr["rbf"]["best_params"].items()}
    bp_l = {k.replace("svm__", ""): v for k, v in sr["linear"]["best_params"].items()}
    cards([
        ("Mejor RBF", f"C={bp_r['C']:g} · γ={bp_r['gamma']}", f"AUC-CV {sr['rbf']['best_cv_roc_auc']:.4f}", MODEL_COLOR["rbf"]),
        ("Mejor lineal", f"C={bp_l['C']:g}", f"AUC-CV {sr['linear']['best_cv_roc_auc']:.4f}", MODEL_COLOR["linear"]),
        ("Combinaciones", f"{len(rbf) + len(lin)}", f"× {results['config']['cv_folds']} folds = {(len(rbf) + len(lin)) * results['config']['cv_folds']} entrenamientos", "#0e9aa7"),
        ("Cómputo de la búsqueda", f"{(sr['rbf']['total_seconds'] + sr['linear']['total_seconds']) / 60:.1f}<small>min CPU</small>", "suma de todos los fits (en paralelo)", "#9aa8bc"),
    ])
    md('<div class="callout"><b>Criterios de la búsqueda.</b> (1) Grilla <b>logarítmica</b>: C y γ actúan por órdenes '
       'de magnitud. (2) C y γ se buscan <b>juntos</b> porque están acoplados. (3) Selección por <b>ROC-AUC</b> con '
       'validación cruzada <b>5-fold estratificada</b>: clases casi balanceadas (45/55) y el AUC mide cómo f(x) ordena '
       'encestes sobre fallos sin depender del umbral; también se registran accuracy y F1. (4) Se compara '
       '<b>train vs CV</b> para detectar sobreajuste. (5) El test nunca participa en la búsqueda.</div>')

    rbf["gamma_lbl"] = rbf["gamma"].astype(str)
    gammas = [g for g in ["0.001", "0.01", "0.1", "1", "scale"] if g in set(rbf["gamma_lbl"])]
    Cs = sorted(rbf["C"].unique())
    c1, c2 = st.columns(2, gap="medium")
    for col, metric, title, scale in (
            (c1, "cv_roc_auc", "RBF · ROC-AUC en validación cruzada", [[0, "#f6f0e6"], [1, "#b3361b"]]),
            (c2, "gap", "RBF · sobreajuste (AUC train − AUC CV)", [[0, "#ffffff"], [1, "#5f4bb6"]])):
        with col, st.container(key=f"card_{metric}"):
            st.markdown(f"### {title}")
            rbf["gap"] = rbf["train_roc_auc"] - rbf["cv_roc_auc"]
            z = [[rbf[(rbf.C == c) & (rbf.gamma_lbl == g)][metric].iloc[0] for g in gammas] for c in Cs]
            fig = go.Figure(go.Heatmap(z=z, x=[f"γ={g}" for g in gammas], y=[f"C={c:g}" for c in Cs],
                                       colorscale=scale, text=[[f"{v:.3f}" for v in row] for row in z],
                                       texttemplate="%{text}", showscale=False,
                                       hovertemplate="%{y} · %{x}: %{z:.4f}<extra></extra>"))
            if metric == "cv_roc_auc":
                fig.add_trace(go.Scatter(x=[f"γ={bp_r['gamma']}"], y=[f"C={bp_r['C']:g}"], mode="markers",
                                         marker=dict(symbol="square-open", size=52, line=dict(width=3, color=NAVY)),
                                         hoverinfo="skip"))
            show(style(fig, 360), f"hm_{metric}")
    c3, c4 = st.columns(2, gap="medium")
    with c3, st.container(key="card_lin"):
        st.markdown("### Lineal · ROC-AUC por C")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"C={c:g}" for c in lin.C], y=lin.cv_roc_auc, name="CV",
                             marker_color=MODEL_COLOR["linear"], error_y=dict(array=lin.cv_roc_auc_std),
                             text=[f"{v:.4f}" for v in lin.cv_roc_auc], textposition="outside"))
        fig.add_trace(go.Scatter(x=[f"C={c:g}" for c in lin.C], y=lin.train_roc_auc, name="Train",
                                 mode="markers", marker=dict(color=NAVY, size=10, symbol="diamond")))
        lo = min(lin.cv_roc_auc.min(), lin.train_roc_auc.min())
        fig.update_yaxes(range=[lo - 0.01, max(lin.train_roc_auc.max(), lin.cv_roc_auc.max()) + 0.01], title="ROC-AUC")
        show(style(fig, 340, legend=True), "lin_bar")
    with c4, st.container(key="card_time"):
        st.markdown("### Tiempo medio por entrenamiento (s)")
        z = [[rbf[(rbf.C == c) & (rbf.gamma_lbl == g)]["mean_fit_seconds"].iloc[0] for g in gammas] for c in Cs]
        fig = go.Figure(go.Heatmap(z=z, x=[f"γ={g}" for g in gammas], y=[f"C={c:g}" for c in Cs],
                                   colorscale=[[0, "#ffffff"], [1, "#0e9aa7"]], showscale=False,
                                   text=[[f"{v:.1f}" for v in row] for row in z], texttemplate="%{text}"))
        show(style(fig, 340), "hm_time")
        md('<div class="note">C grande ⇒ el solver tarda más (más restricciones activas). Lineal: '
           + ", ".join(f"C={c:g} → {t:.1f} s" for c, t in zip(lin.C, lin.mean_fit_seconds)) + ".</div>")
    with st.expander("Tabla completa de la búsqueda"):
        cols = ["C", "gamma", "cv_roc_auc", "cv_roc_auc_std", "cv_accuracy", "cv_f1", "train_roc_auc", "mean_fit_seconds", "rank"]
        st.dataframe(pd.concat([lin.assign(modelo="lineal"), rbf.assign(modelo="RBF")])[["modelo"] + cols]
                     .sort_values("cv_roc_auc", ascending=False).round(4), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# 5 · Laboratorio 2D
# ---------------------------------------------------------------------------
C_VALUES = [0.01, 0.1, 1, 10, 100, 1000]
G_VALUES = [0.01, 0.1, 0.5, 1, 5, 20, 100]


def tab_lab() -> None:
    md('<div class="callout">SVM entrenada <b>en vivo</b> con solo 2 variables (distancia al aro y distancia del '
       'defensor) para poder dibujar la frontera. Mueve C y γ y observa: <b>línea negra</b> = frontera f(x) = 0, '
       '<b>líneas punteadas</b> = bordes del margen f(x) = ±1, <b>puntos con borde</b> = vectores de soporte. '
       'Fondo azul ⇒ predice enceste; rojo ⇒ predice fallo.</div>')
    ctrl, plot = st.columns([1, 2.6], gap="medium")
    with ctrl, st.container(key="card_labctrl"):
        st.markdown("### Controles")
        kernel = st.radio("Kernel", ["rbf", "linear"], format_func={"rbf": "RBF (no lineal)", "linear": "Lineal"}.get,
                          horizontal=True, key="lab_kernel")
        C = st.select_slider("C (costo de violar el margen)", C_VALUES, value=1, key="lab_C")
        gamma = st.select_slider("γ (alcance de cada vector de soporte)", G_VALUES, value=0.5, key="lab_gamma",
                                 disabled=kernel == "linear")
        show_sv = st.toggle("Resaltar vectores de soporte", value=True, key="lab_sv")
        md('<div class="note">Prueba: γ = 100 y C = 1000 ⇒ «islas» alrededor de puntos (sobreajuste: train sube, '
           'test baja). γ = 0.01 ⇒ casi una recta. Lineal ⇒ la frontera no puede curvarse.</div>')
    with st.spinner("Entrenando…"):
        out = get_lab(kernel, float(C), float(gamma))
    lab: ml.Lab2D = out["lab"]
    with ctrl:
        nsv = len(lab.support)
        cards([("Acc. train", pct(lab.train_acc), "", "#3a6fd0"),
               ("Acc. test", pct(lab.test_acc), f"{n(nsv)} SV · {100 * nsv / len(lab.y):.0f} %", "#eb6834")])
        md(f'<div class="note">Entrenado con {n(len(lab.y))} tiros en {out["seconds"]:.2f} s.</div>')
    with plot, st.container(key="card_labplot"):
        f = np.clip(lab.f, -2.5, 2.5)
        fig = go.Figure(go.Heatmap(x=lab.grid_x, y=lab.grid_y, z=f, zmid=0, showscale=False, opacity=.55,
                                   colorscale=[[0, MISSED], [0.5, "#ffffff"], [1, MADE]], hoverinfo="skip"))
        fig.add_trace(go.Contour(x=lab.grid_x, y=lab.grid_y, z=lab.f, showscale=False, hoverinfo="skip",
                                 contours=dict(start=0, end=0, size=1, coloring="none"),
                                 line=dict(color=NAVY, width=3), name="f(x)=0"))
        fig.add_trace(go.Contour(x=lab.grid_x, y=lab.grid_y, z=lab.f, showscale=False, hoverinfo="skip",
                                 contours=dict(start=-1, end=1, size=2, coloring="none"),
                                 line=dict(color=NAVY, width=1.5, dash="dash"), name="f(x)=±1"))
        rng = np.random.default_rng(0)
        jitter = rng.normal(0, 0.12, size=lab.X.shape)  # las distancias vienen redondeadas a 0.1 ft
        is_sv = np.zeros(len(lab.y), bool)
        is_sv[lab.support] = True
        for val, name, col in ((1, "Encestado", MADE), (0, "Fallado", MISSED)):
            mask = lab.y == val
            for sv in (False, True):
                mm = mask & (is_sv == sv)
                if not mm.any():
                    continue
                outline = show_sv and sv
                fig.add_trace(go.Scatter(
                    x=lab.X[mm, 0] + jitter[mm, 0], y=lab.X[mm, 1] + jitter[mm, 1], mode="markers",
                    name=f"{name}{' · SV' if outline else ''}", legendgroup=name,
                    showlegend=not sv or show_sv,
                    marker=dict(color=col, size=7 if outline else 5, opacity=.9 if outline else .6,
                                line=dict(color=NAVY if outline else "rgba(0,0,0,0)", width=1.2)),
                    hovertemplate="Dist %{x:.1f} ft · Defensor %{y:.1f} ft<extra>" + name + "</extra>"))
        fig.update_xaxes(title="Distancia al aro (ft)", range=[0, 30])
        fig.update_yaxes(title="Distancia del defensor (ft)", range=[0, 15])
        show(style(fig, 560, legend=True), "lab")


# ---------------------------------------------------------------------------
# 6 · Simulador de tiro
# ---------------------------------------------------------------------------
PRESETS = {
    "Bandeja libre": dict(SHOT_DIST=1.5, CLOSE_DEF_DIST=6.0, SHOT_CLOCK=14.0, DRIBBLES=2, TOUCH_TIME=2.5, PERIOD=1),
    "Bandeja marcada": dict(SHOT_DIST=2.0, CLOSE_DEF_DIST=0.8, SHOT_CLOCK=10.0, DRIBBLES=4, TOUCH_TIME=4.0, PERIOD=2),
    "Triple abierto": dict(SHOT_DIST=24.0, CLOSE_DEF_DIST=7.0, SHOT_CLOCK=12.0, DRIBBLES=0, TOUCH_TIME=0.8, PERIOD=3),
    "Triple forzado": dict(SHOT_DIST=26.5, CLOSE_DEF_DIST=2.0, SHOT_CLOCK=1.0, DRIBBLES=9, TOUCH_TIME=8.0, PERIOD=4),
    "Media marcada": dict(SHOT_DIST=16.0, CLOSE_DEF_DIST=2.5, SHOT_CLOCK=8.0, DRIBBLES=3, TOUCH_TIME=3.5, PERIOD=2),
}


def apply_preset(name: str) -> None:
    for k, v in PRESETS[name].items():
        st.session_state[f"sim_{k}"] = v


def result_box(model_key: str, fx: float) -> str:
    made = fx >= 0
    bg = (f"linear-gradient(135deg, {MADE} 0%, #13407a 100%)" if made
          else f"linear-gradient(135deg, {MISSED} 0%, #7b2a3f 100%)")
    return (f'<div class="result" style="background:{bg}"><div class="k">{MODEL_NAME[model_key]}</div>'
            f'<div class="v">{"🏀 Enceste" if made else "✖ Fallo"}</div>'
            f'<div class="meta">f(x) = {fx:+.3f} · {"lejos de" if abs(fx) > 1 else "dentro de"} la calle '
            f'(|f| {"&gt;" if abs(fx) > 1 else "≤"} 1)</div></div>')


def tab_sim(models: dict) -> None:
    for k, v in PRESETS["Bandeja libre"].items():
        st.session_state.setdefault(f"sim_{k}", v)
    st.markdown("**Casos de prueba**")
    pcols = st.columns(len(PRESETS))
    for col, name in zip(pcols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), width="stretch", key=f"pre_{name}")
    left, right = st.columns([1, 1.6], gap="medium")
    with left, st.container(key="card_simin"):
        st.markdown("### Describe el tiro")
        x = {}
        for f in FEATURES:
            lo, hi, step = RANGES[f]
            if isinstance(step, int):
                x[f] = st.slider(LABELS[f], int(lo), int(hi), step=step, key=f"sim_{f}")
            else:
                x[f] = st.slider(LABELS[f], float(lo), float(hi), step=float(step), key=f"sim_{f}")
    X = pd.DataFrame([x])[FEATURES]
    fx = {k: float(models[k].decision_function(X)[0]) for k in ("linear", "rbf")}
    with right, st.container(key="card_simout"):
        st.markdown("### Predicción")
        a, b = st.columns(2)
        a.markdown(result_box("linear", fx["linear"]), unsafe_allow_html=True)
        b.markdown(result_box("rbf", fx["rbf"]), unsafe_allow_html=True)
        md('<div class="note">f(x) es la salida de la SVM: el signo decide la clase y |f(x)| indica cuán lejos '
           'de la frontera está el tiro (|f| ≤ 1 ⇒ dentro del margen, zona ambigua).</div>')
        which = st.radio("Variar una variable dejando las demás fijas", ["SHOT_DIST", "CLOSE_DEF_DIST", "SHOT_CLOCK"],
                         format_func=LABELS.get, horizontal=True, key="sim_vary")
        lo, hi, _ = RANGES[which]
        grid = np.linspace(lo, hi, 121)
        Xg = pd.DataFrame([x] * len(grid))[FEATURES]
        Xg[which] = grid
        fig = go.Figure()
        fig.add_hrect(y0=-1, y1=1, fillcolor="#eef2f7", line_width=0, layer="below")
        fig.add_hline(y=0, line=dict(color=NAVY, width=1.5))
        for k in ("linear", "rbf"):
            fig.add_trace(go.Scatter(x=grid, y=models[k].decision_function(Xg), mode="lines",
                                     name=MODEL_NAME[k], line=dict(color=MODEL_COLOR[k], width=3)))
        fig.add_vline(x=x[which], line=dict(color=MUTED, dash="dot"))
        fig.update_xaxes(title=LABELS[which])
        fig.update_yaxes(title="f(x)  (> 0 ⇒ enceste)")
        show(style(fig, 330, legend=True), "sim_curve")
        md('<div class="note">La SVM lineal solo puede responder con una <b>recta</b> a cada variable; '
           'la RBF puede <b>curvarse</b> y cambiar su respuesta según el resto del tiro (interacción). '
           'Franja gris = margen (−1 ≤ f ≤ 1).</div>')


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
def main() -> None:
    md(CSS)
    clean = get_clean()
    art = get_artifacts()
    results, models = art if art else (None, None)
    header(results)
    tabs = st.tabs(["1 · Datos", "2 · Fundamento", "3 · Lineal vs RBF", "4 · GridSearch",
                    "5 · Laboratorio 2D", "6 · Simulador de tiro"])
    with tabs[0]:
        tab_data(clean, results)
    with tabs[1]:
        tab_theory(results)
    missing = "Faltan los artefactos del experimento. Ejecuta `python train.py` en esta carpeta (tarda unos minutos)."
    # Sin artefactos solo funcionan Datos, Fundamento y el laboratorio (que entrena en vivo).
    for tab, render in ((tabs[2], tab_results), (tabs[3], tab_grid)):
        with tab:
            if results:
                render(results)
            else:
                st.warning(missing)
    with tabs[4]:
        tab_lab()
    with tabs[5]:
        if models:
            tab_sim(models)
        else:
            st.warning(missing)


main()
