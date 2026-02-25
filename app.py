
from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Evaluación Diagnóstica 2026 Calendario A", layout="wide")

REQUIRED_COLUMNS = [
    "Sede", "ID Estudiante", "Edad Estudiante", "Genero", "Grado", "Curso",
    "Antiguedad", "Prueba", "Acierto", "QuestionId", "Pregunta",
    "AnswerId", "Respuesta", "Competencia"
]

COLUMN_ALIASES = {
    "género": "Genero",
    "genero": "Genero",
    "edad estudiante": "Edad Estudiante",
    "id estudiante": "ID Estudiante",
    "questionid": "QuestionId",
    "answerid": "AnswerId",
}

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DEFAULT_DATA_CANDIDATES = [
    BASE_DIR / "data" / "EvaluarParaAvanzar_CalA.xlsx",
    BASE_DIR / "EvaluarParaAvanzar_CalA.xlsx",
    Path("data") / "EvaluarParaAvanzar_CalA.xlsx",
    Path("EvaluarParaAvanzar_CalA.xlsx"),
]

GRADE_ORDER = {
    "tercero": 3,
    "cuarto": 4,
    "quinto": 5,
    "sexto": 6,
    "septimo": 7,
    "séptimo": 7,
    "octavo": 8,
    "noveno": 9,
    "decimo": 10,
    "décimo": 10,
    "undecimo": 11,
    "undécimo": 11,
    "primero": 101,
    "segundo": 102,
    "transicion": 103,
    "transición": 103,
    "jardin": 104,
    "jardín": 104,
    "prejardin": 105,
    "pre jardín": 105,
    "prejardín": 105,
}

COMPETENCY_ORDER = [
    "Conocimientos",
    "Pensamiento Social",
    "Argumentación en contextos ciudadanos",
    "Multiperspectivismo",
    "pensamiento sistémico",
    "Pensamiento Reflexivo y Sistémico",
    "Interpretación y Análisis de Perspectivas",
    "Explicación de fenómenos",
    "Indagación",
    "Conocimiento científico",
    "Interpretación y representación",
    "Formulación y ejecución",
    "Razonamiento",
    "Comunicación",
    "Resolución de problemas",
    "Argumentación",
]

P_BIS_TARGET = 0.20
D27_TARGET = 0.20


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in text if not unicodedata.combining(ch))


COMPETENCY_ORDER_MAP = {strip_accents(x).lower(): i for i, x in enumerate(COMPETENCY_ORDER, start=1)}


def competency_sort_key(value: str) -> tuple[int, str]:
    raw = str(value).strip()
    key = strip_accents(raw).lower()
    return (COMPETENCY_ORDER_MAP.get(key, 999), raw)


def clean_sede_label(value: str) -> str:
    raw = str(value).strip()
    cleaned = re.sub(r"^\s*innova\s+", "", raw, flags=re.I)
    return cleaned.title() if cleaned else raw


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for col in df.columns:
        c = str(col).strip()
        cols.append(COLUMN_ALIASES.get(c.lower(), c))
    df = df.copy()
    df.columns = cols
    return df


def grade_sort_key(value: str) -> tuple[int, str]:
    raw = str(value).strip()
    if not raw:
        return (999, raw)
    normalized = strip_accents(raw).lower()
    if normalized in GRADE_ORDER:
        return (GRADE_ORDER[normalized], raw)
    num_match = re.search(r"(\d+)", normalized)
    if num_match:
        return (int(num_match.group(1)), raw)
    return (999, raw)


def course_sort_key(value: str) -> tuple[int, str]:
    raw = str(value).strip()
    if not raw:
        return (999, raw)
    m = re.search(r"(\d+)", raw)
    num = int(m.group(1)) if m else 999
    return (num, raw)


def grade_display_label(value: str) -> str:
    raw = str(value).strip()
    normalized = strip_accents(raw).lower()
    if normalized in GRADE_ORDER:
        grade_num = GRADE_ORDER[normalized]
        if grade_num < 100:
            return f"{grade_num}°"
    num_match = re.search(r"(\d+)", normalized)
    if num_match:
        return f"{int(num_match.group(1))}°"
    return raw


def shorten_text(value: str, max_len: int = 90) -> str:
    raw = str(value).strip().replace("\n", " ")
    raw = re.sub(r"\s+", " ", raw)
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 1].rstrip() + "…"


def compute_axis_bounds(series: pd.Series, min_floor: float = 20, padding: float = 3) -> list[float]:
    clean = pd.Series(series).dropna().astype(float)
    if clean.empty:
        return [min_floor, 100]
    lower = max(min_floor, float(np.floor(clean.min() - padding)))
    upper = min(100.0, float(np.ceil(clean.max() + padding)))
    if upper <= lower:
        upper = min(100.0, lower + 5)
    return [lower, upper]


def clean_prueba_label(value: str) -> str:
    raw = str(value).strip()
    cleaned = re.sub(r"\s+\d+\s*°?$", "", raw).strip()
    key = strip_accents(cleaned).lower()
    if "sociales" in key or "competencias ciudadanas" in key or "pensamiento ciudadano" in key:
        return "Ciencias sociales"
    return cleaned or raw


def load_default_file_if_exists() -> Path | None:
    for path in DEFAULT_DATA_CANDIDATES:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def read_dataframe_from_path(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            try:
                return pd.read_csv(path, encoding="latin1")
            except Exception:
                return pd.read_csv(path, sep=None, engine="python")
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError("Formato no soportado. Usa CSV o Excel.")


@st.cache_data(show_spinner=False)
def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        uploaded_file.seek(0)
        try:
            return pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except Exception:
            uploaded_file.seek(0)
            try:
                return pd.read_csv(uploaded_file, encoding="latin1")
            except Exception:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, sep=None, engine="python")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)
    raise ValueError("Formato no soportado. Usa CSV o Excel.")


def validate_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


@st.cache_data(show_spinner=False)
def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw)

    for col in ["Acierto", "Edad Estudiante", "Antiguedad", "QuestionId", "AnswerId"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Acierto"] = df["Acierto"].fillna(0).clip(lower=0, upper=1)

    text_cols = ["Sede", "Genero", "Grado", "Curso", "Prueba", "Pregunta", "Respuesta", "Competencia"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["", "nan", "None"]), col] = np.nan

    df["Respuesta Limpia"] = df["Respuesta"].fillna("Sin respuesta")
    df["Prueba Base"] = df["Prueba"].map(clean_prueba_label)
    df["Sede Corta"] = df["Sede"].map(clean_sede_label)
    df["Grado Orden"] = df["Grado"].map(lambda x: grade_sort_key(x)[0])
    df["Curso Orden"] = df["Curso"].map(lambda x: course_sort_key(x)[0])

    return df


def add_benchmark(df: pd.DataFrame, focus_df: pd.DataFrame, dim: str) -> pd.DataFrame:
    net = (
        df.groupby(dim, dropna=False)
        .agg(acierto_red=("Acierto", "mean"), estudiantes_red=("ID Estudiante", "nunique"), respuestas_red=("Acierto", "size"))
        .reset_index()
    )
    focus = (
        focus_df.groupby(dim, dropna=False)
        .agg(acierto_sede=("Acierto", "mean"), estudiantes_sede=("ID Estudiante", "nunique"), respuestas_sede=("Acierto", "size"))
        .reset_index()
    )
    out = net.merge(focus, on=dim, how="left")
    out["acierto_red_pct"] = (out["acierto_red"] * 100).round(2)
    out["acierto_sede_pct"] = (out["acierto_sede"] * 100).round(2)
    out["brecha_pp"] = (out["acierto_sede_pct"] - out["acierto_red_pct"]).round(2)
    return out


def sort_benchmark(out: pd.DataFrame, dim: str) -> pd.DataFrame:
    if out.empty:
        return out
    out = out.copy()
    if dim == "Competencia":
        out["__orden__"] = out[dim].map(lambda x: competency_sort_key(x)[0])
        out = out.sort_values(["__orden__", dim]).drop(columns="__orden__")
    elif dim == "Grado":
        out["__orden__"] = out[dim].map(lambda x: grade_sort_key(x)[0])
        out = out.sort_values(["__orden__", dim]).drop(columns="__orden__")
    elif dim == "Sede":
        out = out.sort_values("acierto_red_pct", ascending=False)
    else:
        out = out.sort_values(dim)
    return out


def friendly_comp_table(comp: pd.DataFrame) -> pd.DataFrame:
    if comp.empty:
        return comp
    out = sort_benchmark(comp, "Competencia").copy()
    out = out.rename(columns={
        "acierto_sede_pct": "% de acierto en la sede",
        "acierto_red_pct": "% de acierto en Colombia",
        "brecha_pp": "Brecha frente a Colombia (pp)",
    })
    return out[["Competencia", "% de acierto en la sede", "% de acierto en Colombia", "Brecha frente a Colombia (pp)"]]


def safe_pct(series: pd.Series) -> float:
    return float(series.mean() * 100) if len(series) else 0.0


def safe_nunique(series: pd.Series) -> int:
    return int(series.nunique()) if len(series) else 0


@st.cache_data(show_spinner=False)
def item_metrics(df_scope: pd.DataFrame) -> pd.DataFrame:
    if df_scope.empty:
        return pd.DataFrame()

    base = df_scope.copy()
    student_scores = (
        base.groupby("ID Estudiante", dropna=False)
        .agg(total=("Acierto", "sum"), n_items=("Acierto", "size"))
        .reset_index()
    )
    student_scores["prop"] = student_scores["total"] / student_scores["n_items"].replace(0, np.nan)

    merged = base.merge(student_scores[["ID Estudiante", "total", "n_items", "prop"]], on="ID Estudiante", how="left")
    merged["adjusted_total"] = merged["total"] - merged["Acierto"]

    q_low = student_scores["prop"].quantile(0.27)
    q_high = student_scores["prop"].quantile(0.73)
    upper = set(student_scores.loc[student_scores["prop"] >= q_high, "ID Estudiante"].tolist())
    lower = set(student_scores.loc[student_scores["prop"] <= q_low, "ID Estudiante"].tolist())
    merged["grupo_27"] = np.where(
        merged["ID Estudiante"].isin(upper), "superior",
        np.where(merged["ID Estudiante"].isin(lower), "inferior", "medio")
    )

    base_stats = (
        merged.groupby(["QuestionId", "Prueba Base", "Prueba", "Competencia", "Pregunta"], dropna=False)
        .agg(
            intentos=("Acierto", "size"),
            aciertos=("Acierto", "sum"),
            dificultad=("Acierto", "mean"),
            Grado=("Grado", "first"),
        )
        .reset_index()
    )

    d27 = (
        merged[merged["grupo_27"].isin(["superior", "inferior"])]
        .groupby(["QuestionId", "grupo_27"], dropna=False)["Acierto"]
        .mean()
        .unstack(fill_value=np.nan)
        .reset_index()
    )
    if "superior" not in d27.columns:
        d27["superior"] = np.nan
    if "inferior" not in d27.columns:
        d27["inferior"] = np.nan
    d27["d27"] = d27["superior"] - d27["inferior"]

    def pbis(group: pd.DataFrame) -> float:
        x = group["Acierto"].astype(float)
        y = group["adjusted_total"].astype(float)
        if x.nunique() < 2 or y.nunique() < 2:
            return np.nan
        return x.corr(y)

    pb = (
        merged.groupby("QuestionId", dropna=False)
        .apply(pbis, include_groups=False)
        .reset_index(name="p_bis")
    )

    options = (
        merged.groupby(["QuestionId", "Respuesta Limpia", "Acierto"], dropna=False)
        .size()
        .reset_index(name="selecciones")
    )
    attempts = merged.groupby("QuestionId", dropna=False).size().rename("intentos_total").reset_index()
    options = options.merge(attempts, on="QuestionId", how="left")
    options["pct_item"] = options["selecciones"] / options["intentos_total"].replace(0, np.nan)

    correct = (
        options[options["Acierto"] == 1]
        .sort_values(["QuestionId", "selecciones"], ascending=[True, False])
        .drop_duplicates("QuestionId")
        [["QuestionId", "Respuesta Limpia", "selecciones", "pct_item"]]
        .rename(columns={"Respuesta Limpia": "opcion_correcta", "selecciones": "sel_correcta", "pct_item": "pct_correcta"})
    )

    distractors = options[options["Acierto"] == 0].copy()
    if distractors.empty:
        distractor_top = pd.DataFrame(columns=["QuestionId", "distractor_top", "sel_distractor_top", "pct_distractor_top"])
        distractor_health = pd.DataFrame(columns=["QuestionId", "n_distractores", "distractores_no_funcionales"])
    else:
        distractor_top = (
            distractors.sort_values(["QuestionId", "selecciones"], ascending=[True, False])
            .drop_duplicates("QuestionId")
            [["QuestionId", "Respuesta Limpia", "selecciones", "pct_item"]]
            .rename(columns={"Respuesta Limpia": "distractor_top", "selecciones": "sel_distractor_top", "pct_item": "pct_distractor_top"})
        )
        distractor_health = (
            distractors.groupby("QuestionId", dropna=False)
            .agg(
                n_distractores=("Respuesta Limpia", "nunique"),
                distractores_no_funcionales=("pct_item", lambda s: int((s < 0.05).sum()))
            )
            .reset_index()
        )

    out = (
        base_stats
        .merge(d27[["QuestionId", "d27"]], on="QuestionId", how="left")
        .merge(pb, on="QuestionId", how="left")
        .merge(correct, on="QuestionId", how="left")
        .merge(distractor_top, on="QuestionId", how="left")
        .merge(distractor_health, on="QuestionId", how="left")
    )

    out["Grado Etiqueta"] = out["Grado"].map(grade_display_label)
    out["Grado Orden"] = out["Grado"].map(lambda x: grade_sort_key(x)[0])
    out["dificultad_pct"] = (out["dificultad"] * 100).round(2)
    out["d27"] = out["d27"].round(3)
    out["p_bis"] = out["p_bis"].round(3)
    out["pct_distractor_top"] = (out["pct_distractor_top"] * 100).round(2)
    out["pct_correcta"] = (out["pct_correcta"] * 100).round(2)

    def difficulty_band(p: float) -> str:
        if pd.isna(p):
            return "Sin dato"
        if p < 0.30:
            return "Muy difícil"
        if p < 0.50:
            return "Difícil"
        if p < 0.80:
            return "Adecuada"
        return "Muy fácil"

    def discrim_band(v: float) -> str:
        if pd.isna(v):
            return "Sin dato"
        if v < 0:
            return "Negativa"
        if v < 0.10:
            return "Muy baja"
        if v < 0.20:
            return "Baja"
        if v < 0.30:
            return "Media"
        return "Alta"

    def item_flag(row: pd.Series) -> str:
        p = row["dificultad"]
        pb = row["p_bis"]
        d27 = row["d27"]
        if pd.isna(p):
            return "Revisar"
        if ((pd.notna(pb) and pb < 0) or (pd.notna(d27) and d27 < 0)):
            return "Alerta psicométrica"
        if p < 0.20 and ((pd.isna(pb) or pb < 0.15) and (pd.isna(d27) or d27 < 0.15)):
            return "Posible ítem muy complejo o ambiguo"
        if p > 0.90 and ((pd.isna(pb) or pb < 0.10) and (pd.isna(d27) or d27 < 0.10)):
            return "Posible ítem demasiado fácil"
        if pd.notna(pb) and pb >= 0.30 and 0.30 <= p <= 0.80:
            return "Buen ítem"
        return "Revisar"

    out["dificultad_nivel"] = out["dificultad"].map(difficulty_band)
    out["discriminacion_nivel"] = out["p_bis"].map(discrim_band)
    out["estado_item"] = out.apply(item_flag, axis=1)

    return out.sort_values(["Prueba Base", "Grado Orden", "dificultad", "p_bis"], ascending=[True, True, True, False])


def psychometric_summary(item_df: pd.DataFrame) -> dict:
    if item_df.empty:
        return {
            "n_items": 0,
            "pct_target_difficulty": 0.0,
            "pct_good_discrimination": 0.0,
            "pct_alert": 0.0,
        }

    target_difficulty = item_df["dificultad"].between(0.30, 0.80, inclusive="both")
    good_disc = (item_df["p_bis"] >= P_BIS_TARGET) | (item_df["d27"] >= D27_TARGET)
    alert = item_df["estado_item"].isin(["Alerta psicométrica", "Posible ítem muy complejo o ambiguo"])

    return {
        "n_items": int(len(item_df)),
        "pct_target_difficulty": float(target_difficulty.mean() * 100),
        "pct_good_discrimination": float(good_disc.mean() * 100),
        "pct_alert": float(alert.mean() * 100),
    }


def make_indicator(title: str, value: float, delta: float | None = None, suffix: str = "%", reference: float | None = None) -> go.Figure:
    fig = go.Figure()
    indicator_kwargs = dict(
        mode="number+gauge" if delta is None else "number+delta+gauge",
        value=float(value),
        title={"text": title, "font": {"size": 18}},
        number={"suffix": suffix},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.35},
            "steps": [
                {"range": [0, 40], "color": "#fde2e4"},
                {"range": [40, 60], "color": "#fff1c9"},
                {"range": [60, 80], "color": "#dff7e3"},
                {"range": [80, 100], "color": "#d9f0ff"},
            ],
        },
    )
    if delta is not None:
        indicator_kwargs["delta"] = {"reference": float(reference if reference is not None else value - delta), "valueformat": ".2f"}
    fig.add_trace(go.Indicator(**indicator_kwargs))
    fig.update_layout(height=220, margin=dict(l=18, r=18, t=55, b=10))
    return fig


def benchmark_cards(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str):
    colombia_pct = safe_pct(df["Acierto"])
    focus_pct = safe_pct(focus_df["Acierto"])
    brecha = focus_pct - colombia_pct
    sedes = (
        df.groupby(["Sede", "Sede Corta"], dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )
    best_sede = sedes.iloc[0]["Sede Corta"] if not sedes.empty else "Sin dato"
    best_sede_pct = float(sedes.iloc[0]["Acierto"]) if not sedes.empty else 0.0

    by_prueba = (
        df.groupby("Prueba Base", dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .sort_values()
        .reset_index()
    )
    prueba_critica = by_prueba.iloc[0]["Prueba Base"] if not by_prueba.empty else "Sin dato"
    prueba_critica_pct = float(by_prueba.iloc[0]["Acierto"]) if not by_prueba.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Promedio Colombia", f"{colombia_pct:.1f}%")
    c2.metric(f"Promedio {focus_label}", f"{focus_pct:.1f}%", delta=f"{brecha:+.2f} pp")
    c3.metric("Mejor sede", f"{best_sede}", delta=f"{best_sede_pct:.1f}%")
    c4.metric("Prueba con menor acierto", f"{prueba_critica}", delta=f"{prueba_critica_pct:.1f}%")


def render_radar_cards(df: pd.DataFrame):
    sedes = (
        df[["Sede", "Sede Corta"]]
        .dropna()
        .drop_duplicates()
        .sort_values("Sede Corta")
    )
    pruebas = sorted(df["Prueba Base"].dropna().unique().tolist())
    if sedes.empty or not pruebas:
        st.info("No hay suficientes datos para construir perfiles por sede.")
        return

    colombia = (
        df.groupby("Prueba Base", dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .reindex(pruebas)
        .fillna(0)
    )
    radar_cols = st.columns(2)
    for idx, (_, row) in enumerate(sedes.iterrows()):
        sede_value = row["Sede"]
        sede_label = row["Sede Corta"]
        sede_series = (
            df.loc[df["Sede"] == sede_value]
            .groupby("Prueba Base", dropna=False)["Acierto"]
            .mean()
            .mul(100)
            .reindex(pruebas)
            .fillna(0)
        )
        theta = pruebas + [pruebas[0]]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=colombia.tolist() + [float(colombia.iloc[0])],
            theta=theta,
            fill="toself",
            name="Colombia",
            line=dict(color="#334155"),
            fillcolor="rgba(51, 65, 85, 0.15)",
        ))
        fig.add_trace(go.Scatterpolar(
            r=sede_series.tolist() + [float(sede_series.iloc[0])],
            theta=theta,
            fill="toself",
            name=sede_label,
            line=dict(color="#0284c7"),
            fillcolor="rgba(2, 132, 199, 0.25)",
        ))
        fig.update_layout(
            title=f"Perfil de {sede_label} frente a Colombia",
            polar=dict(radialaxis=dict(range=[0, 100], tickfont=dict(size=10))),
            showlegend=True,
            height=380,
            margin=dict(l=30, r=30, t=60, b=20),
        )
        with radar_cols[idx % 2]:
            st.plotly_chart(fig, use_container_width=True)
        if idx % 2 == 1 and idx + 1 < len(sedes):
            radar_cols = st.columns(2)


def show_overview_tab(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str):
    st.subheader("Resultados globales")

    with st.expander("Cómo leer esta pestaña", expanded=False):
        st.markdown(
            """
            - **Promedio Colombia** es la referencia nacional del tablero con los filtros activos.
            - **Promedio de la sede focal** muestra dónde está la sede seleccionada frente a esa referencia.
            - Los **radares** comparan a cada sede contra Colombia usando las mismas pruebas, para detectar fortalezas y rezagos de un vistazo.
            - Las barras y tablas ayudan a priorizar dónde conviene intervenir primero.
            """
        )

    benchmark_cards(df, focus_df, focus_label)

    by_sede = (
        df.groupby(["Sede", "Sede Corta"], dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .reset_index(name="acierto_colombia_pct")
        .sort_values("acierto_colombia_pct", ascending=False)
    )
    by_sede["destacado"] = np.where(by_sede["Sede Corta"] == focus_label, focus_label, "Otras sedes")

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(
            by_sede,
            x="Sede Corta",
            y="acierto_colombia_pct",
            color="destacado",
            title="% de acierto por sede",
            text="acierto_colombia_pct",
            color_discrete_map={focus_label: "#0284c7", "Otras sedes": "#94a3b8"},
        )
        promedio_colombia = safe_pct(df["Acierto"])
        fig.add_hline(
            y=promedio_colombia,
            line_dash="dash",
            annotation_text="Promedio Colombia",
            annotation_position="top left"
        )
        y_range = compute_axis_bounds(by_sede["acierto_colombia_pct"])
        if promedio_colombia > y_range[1]:
            y_range[1] = min(100.0, float(np.ceil(promedio_colombia + 3)))
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(yaxis_title="% de acierto", xaxis_title="", yaxis_range=y_range, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        by_prueba = add_benchmark(df, focus_df, "Prueba Base").sort_values("acierto_red_pct", ascending=False)
        melted = by_prueba.melt(
            id_vars="Prueba Base",
            value_vars=["acierto_red_pct", "acierto_sede_pct"],
            var_name="Serie",
            value_name="Porcentaje"
        )
        melted["Serie"] = melted["Serie"].map({
            "acierto_red_pct": "Colombia",
            "acierto_sede_pct": focus_label,
        })
        fig = px.line(
            melted,
            x="Prueba Base",
            y="Porcentaje",
            color="Serie",
            markers=True,
            title=f"% de acierto por prueba: {focus_label} vs Colombia"
        )
        fig.update_layout(yaxis_title="% de acierto", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Perfil por sede frente a Colombia**")
    render_radar_cards(df)

    st.markdown("**Mapa de calor por sede y prueba**")
    heat = (
        df.groupby(["Sede Corta", "Prueba Base"], dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
        .pivot(index="Sede Corta", columns="Prueba Base", values="Acierto")
    )
    fig = px.imshow(heat, text_auto=True, aspect="auto", title="% de acierto por sede y prueba")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Trayectoria por grado: sede focal vs Colombia**")
    by_grade_net = (
        df.groupby("Grado", dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .reset_index()
    )
    by_grade_focus = (
        focus_df.groupby("Grado", dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"Acierto": "Acierto Focus"})
    )
    merged = by_grade_net.merge(by_grade_focus, on="Grado", how="left")
    merged["orden"] = merged["Grado"].map(lambda x: grade_sort_key(x)[0])
    merged = merged.sort_values(["orden", "Grado"])
    melted = merged.melt(id_vars="Grado", value_vars=["Acierto", "Acierto Focus"], var_name="Serie", value_name="Porcentaje")
    melted["Serie"] = melted["Serie"].map({"Acierto": "Colombia", "Acierto Focus": focus_label})
    fig = px.line(melted, x="Grado", y="Porcentaje", color="Serie", markers=True, title=f"Trayectoria por grado: {focus_label} vs Colombia")
    fig.update_layout(yaxis_title="% de acierto", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Lectura rápida para docentes y directivos**")
    comp = add_benchmark(df, focus_df, "Competencia")
    comp = sort_benchmark(comp, "Competencia")

    weakest = comp.sort_values("brecha_pp").head(5).rename(columns={
        "acierto_sede_pct": "% de acierto en la sede",
        "acierto_red_pct": "% de acierto en Colombia",
        "brecha_pp": "Brecha frente a Colombia (pp)",
    })[["Competencia", "% de acierto en la sede", "% de acierto en Colombia", "Brecha frente a Colombia (pp)"]]

    strongest = comp.sort_values("brecha_pp", ascending=False).head(5).rename(columns={
        "acierto_sede_pct": "% de acierto en la sede",
        "acierto_red_pct": "% de acierto en Colombia",
        "brecha_pp": "Brecha frente a Colombia (pp)",
    })[["Competencia", "% de acierto en la sede", "% de acierto en Colombia", "Brecha frente a Colombia (pp)"]]

    st.markdown(f"**Competencias donde {focus_label} necesita más apoyo**")
    st.dataframe(weakest, use_container_width=True, hide_index=True)
    st.markdown(f"**Competencias donde {focus_label} muestra mejor desempeño**")
    st.dataframe(strongest, use_container_width=True, hide_index=True)


def render_prueba_panel(prueba: str, df_prueba: pd.DataFrame, focus_prueba: pd.DataFrame, focus_label: str):
    if df_prueba.empty:
        st.info("No hay datos para esta prueba.")
        return

    colombia_pct = safe_pct(df_prueba["Acierto"])
    focus_pct = safe_pct(focus_prueba["Acierto"]) if not focus_prueba.empty else np.nan
    brecha = focus_pct - colombia_pct if pd.notna(focus_pct) else np.nan

    focus_bench = focus_prueba if not focus_prueba.empty else df_prueba.iloc[0:0]
    by_comp = add_benchmark(df_prueba, focus_bench, "Competencia")
    by_comp = sort_benchmark(by_comp, "Competencia")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Promedio Colombia", f"{colombia_pct:.2f}%")
    c2.metric(f"Promedio {focus_label}", f"{focus_pct:.2f}%" if pd.notna(focus_pct) else "Sin dato")
    c3.metric("Brecha frente a Colombia", f"{brecha:+.2f} pp" if pd.notna(brecha) else "Sin dato")
    c4.metric("Competencias evaluadas", f"{safe_nunique(df_prueba['Competencia']):,}")

    st.markdown("**Comparación por competencia**")
    melted = by_comp.melt(
        id_vars="Competencia",
        value_vars=["acierto_red_pct", "acierto_sede_pct"],
        var_name="Serie",
        value_name="Porcentaje"
    )
    melted["Serie"] = melted["Serie"].map({
        "acierto_red_pct": "Colombia",
        "acierto_sede_pct": focus_label,
    })
    melted["orden"] = melted["Competencia"].map(lambda x: competency_sort_key(x)[0])
    melted = melted.sort_values(["orden", "Competencia"])
    fig = px.bar(
        melted,
        x="Competencia",
        y="Porcentaje",
        color="Serie",
        barmode="group",
        title=f"{prueba}: competencias, {focus_label} vs Colombia"
    )
    fig.update_layout(yaxis_title="% de acierto", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns([1, 1])
    with c5:
        heat = (
            df_prueba.groupby(["Sede Corta", "Competencia"], dropna=False)["Acierto"]
            .mean()
            .mul(100)
            .round(1)
            .reset_index()
        )
        heat["orden"] = heat["Competencia"].map(lambda x: competency_sort_key(x)[0])
        heat = heat.sort_values(["Sede Corta", "orden", "Competencia"]).drop(columns="orden")
        heat = heat.pivot(index="Sede Corta", columns="Competencia", values="Acierto")
        fig = px.imshow(heat, text_auto=True, aspect="auto", title=f"{prueba}: desempeño por sede y competencia")
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        grade_net = (
            df_prueba.groupby("Grado", dropna=False)["Acierto"]
            .mean()
            .mul(100)
            .reset_index()
        )
        grade_focus = (
            focus_prueba.groupby("Grado", dropna=False)["Acierto"]
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={"Acierto": "Acierto Focus"})
        )
        grade = grade_net.merge(grade_focus, on="Grado", how="left")
        grade["orden"] = grade["Grado"].map(lambda x: grade_sort_key(x)[0])
        grade = grade.sort_values(["orden", "Grado"])
        melted = grade.melt(id_vars="Grado", value_vars=["Acierto", "Acierto Focus"], var_name="Serie", value_name="Porcentaje")
        melted["Serie"] = melted["Serie"].map({"Acierto": "Colombia", "Acierto Focus": focus_label})
        fig = px.line(melted, x="Grado", y="Porcentaje", color="Serie", markers=True, title=f"{prueba}: trayectoria por grado")
        fig.update_layout(yaxis_title="% de acierto", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Tabla de comparación por competencia**")
    st.dataframe(
        friendly_comp_table(by_comp),
        use_container_width=True,
        hide_index=True
    )


def show_pruebas_tab(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str):
    st.subheader("Detalle por prueba")
    with st.expander("Cómo leer esta pestaña", expanded=False):
        st.markdown(
            """
            - Cada pestaña resume una prueba completa.
            - La comparación principal siempre es **sede focal vs Colombia**.
            - Si una competencia queda por debajo de Colombia, suele ser un buen punto de partida para planear refuerzos.
            - La trayectoria por grado ayuda a detectar en qué momento cae o mejora el desempeño.
            """
        )
    pruebas = sorted(df["Prueba Base"].dropna().unique().tolist())
    prueba_tabs = st.tabs(pruebas)

    for tab, prueba in zip(prueba_tabs, pruebas):
        with tab:
            df_prueba = df[df["Prueba Base"] == prueba].copy()
            focus_prueba = focus_df[focus_df["Prueba Base"] == prueba].copy()
            render_prueba_panel(prueba, df_prueba, focus_prueba, focus_label)


def distractor_table(df_scope: pd.DataFrame, question_id: int | float) -> pd.DataFrame:
    data = df_scope[df_scope["QuestionId"] == question_id].copy()
    if data.empty:
        return pd.DataFrame()
    out = (
        data.groupby(["Respuesta Limpia", "Acierto"], dropna=False)
        .size()
        .reset_index(name="selecciones")
        .sort_values("selecciones", ascending=False)
    )
    total = out["selecciones"].sum()
    out["porcentaje"] = (out["selecciones"] / total * 100).round(2)
    out["tipo"] = np.where(out["Acierto"] == 1, "Respuesta correcta", "Distractor")
    return out[["tipo", "Respuesta Limpia", "selecciones", "porcentaje", "Acierto"]]


def build_question_bank(items_prueba: pd.DataFrame) -> pd.DataFrame:
    if items_prueba.empty:
        return items_prueba
    ordered = items_prueba.copy()
    ordered = ordered.sort_values(["Grado Orden", "Grado", "dificultad", "p_bis"], ascending=[True, True, True, False]).reset_index(drop=True)
    ordered["orden_en_grado"] = ordered.groupby("Grado").cumcount() + 1
    ordered["selector_label"] = ordered.apply(
        lambda row: f"Pregunta {int(row['orden_en_grado'])} · Grado {row['Grado Etiqueta']}",
        axis=1,
    )
    return ordered


def build_option_comparison(question_id: int | float, colombia_df: pd.DataFrame, focus_df: pd.DataFrame) -> pd.DataFrame:
    def _profile(frame: pd.DataFrame, pct_col: str) -> pd.DataFrame:
        data = frame[frame["QuestionId"] == question_id].copy()
        if data.empty:
            return pd.DataFrame(columns=["Respuesta Limpia", "Acierto", pct_col])
        out = (
            data.groupby(["Respuesta Limpia", "Acierto"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        total = out["n"].sum()
        out[pct_col] = np.where(total > 0, out["n"] / total * 100, 0)
        return out[["Respuesta Limpia", "Acierto", pct_col]]

    colombia = _profile(colombia_df, "pct_colombia")
    focus = _profile(focus_df, "pct_sede")
    merged = colombia.merge(focus, on=["Respuesta Limpia", "Acierto"], how="outer").fillna(0)
    if merged.empty:
        return merged
    merged["tipo"] = np.where(merged["Acierto"] == 1, "Respuesta correcta", "Distractor")
    merged["color"] = np.where(merged["Acierto"] == 1, "#16a34a", "#94a3b8")
    merged["Opción de respuesta"] = merged["Respuesta Limpia"].map(lambda x: shorten_text(x, 100))
    merged["pico"] = merged[["pct_colombia", "pct_sede"]].max(axis=1)
    merged = merged.sort_values(["Acierto", "pico"], ascending=[False, False]).reset_index(drop=True)
    merged["pct_colombia"] = merged["pct_colombia"].round(2)
    merged["pct_sede"] = merged["pct_sede"].round(2)
    return merged


def option_dumbbell_chart(option_comp: pd.DataFrame, focus_label: str) -> go.Figure:
    fig = go.Figure()
    if option_comp.empty:
        return fig

    for _, row in option_comp.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["pct_colombia"], row["pct_sede"]],
            y=[row["Opción de respuesta"], row["Opción de respuesta"]],
            mode="lines",
            line=dict(color=row["color"], width=4),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=option_comp["pct_colombia"],
        y=option_comp["Opción de respuesta"],
        mode="markers+text",
        name="Colombia",
        text=[f"{v:.1f}%" for v in option_comp["pct_colombia"]],
        textposition="middle left",
        marker=dict(
            symbol="circle",
            size=12,
            color=option_comp["color"].tolist(),
            line=dict(color="#0f172a", width=1),
        ),
    ))
    fig.add_trace(go.Scatter(
        x=option_comp["pct_sede"],
        y=option_comp["Opción de respuesta"],
        mode="markers+text",
        name=focus_label,
        text=[f"{v:.1f}%" for v in option_comp["pct_sede"]],
        textposition="middle right",
        marker=dict(
            symbol="diamond",
            size=12,
            color=option_comp["color"].tolist(),
            line=dict(color="#0f172a", width=1),
        ),
    ))
    fig.update_layout(
        title=f"Cómo se repartieron las respuestas: {focus_label} vs Colombia",
        xaxis_title="% de estudiantes que marcó la opción",
        yaxis_title="",
        height=max(380, 90 + 70 * len(option_comp)),
        margin=dict(l=40, r=20, t=60, b=20),
    )
    return fig


def show_psychometrics_tab(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str):
    st.subheader("Análisis de las respuestas")
    with st.expander("Cómo leer esta pestaña", expanded=False):
        st.markdown(
            """
            - Aquí el foco no es técnico, sino **pedagógico**: ver qué preguntas costaron más y qué errores fueron más frecuentes.
            - Las preguntas están ordenadas por **grado** y por **dificultad** dentro de cada grado.
            - En la comparación de opciones, **verde** indica la respuesta correcta y **gris** los distractores.
            - **Círculo** representa Colombia y **rombo** la sede focal.
            """
        )

    scope_options = ["Colombia"]
    if focus_label != "Colombia":
        scope_options.append(focus_label)
    scope = st.radio("Ordenar preguntas según", scope_options, horizontal=True)
    selected = df if scope == "Colombia" else focus_df

    if selected.empty:
        st.warning("No hay datos en el alcance seleccionado.")
        return

    items = item_metrics(selected)
    if items.empty:
        st.warning("No fue posible calcular métricas por pregunta.")
        return

    pruebas = sorted(items["Prueba Base"].dropna().unique().tolist())
    selected_prueba = st.selectbox("Prueba para analizar", pruebas)
    items_prueba = build_question_bank(items[items["Prueba Base"] == selected_prueba].copy())
    scope_prueba_df = selected[selected["Prueba Base"] == selected_prueba].copy()
    colombia_prueba_df = df[df["Prueba Base"] == selected_prueba].copy()
    focus_prueba_df = focus_df[focus_df["Prueba Base"] == selected_prueba].copy()

    if items_prueba.empty:
        st.info("No hay preguntas para esta prueba con el filtro actual.")
        return

    hardest = items_prueba.nsmallest(1, "dificultad")
    easiest = items_prueba.nlargest(1, "dificultad")
    best_disc = items_prueba.sort_values(["p_bis", "d27"], ascending=[False, False]).head(1)
    worst_disc = items_prueba.sort_values(["p_bis", "d27"], ascending=[True, True]).head(1)

    def _card_label(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "Sin dato"
        row = frame.iloc[0]
        return f"{row['selector_label']}"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Preguntas analizadas", f"{len(items_prueba):,}")
    c2.metric("Pregunta más difícil", _card_label(hardest))
    c3.metric("Pregunta más fácil", _card_label(easiest))
    c4.metric("Mayor discriminación", _card_label(best_disc))
    c5.metric("Menor discriminación", _card_label(worst_disc))

    difficult_table = items_prueba[["selector_label", "Competencia", "dificultad_pct", "p_bis", "d27"]].rename(columns={
        "selector_label": "Pregunta",
        "dificultad_pct": "% de acierto",
        "p_bis": "Discriminación (p bis)",
        "d27": "Discriminación (D27)",
    })
    st.markdown("**Preguntas ordenadas por grado y dificultad**")
    st.dataframe(difficult_table, use_container_width=True, hide_index=True, height=420)

    option_ids = items_prueba["QuestionId"].tolist()
    label_map = dict(zip(items_prueba["QuestionId"], items_prueba["selector_label"]))
    default_q = hardest.iloc[0]["QuestionId"] if not hardest.empty else option_ids[0]
    selected_q = st.selectbox(
        "Explorar una pregunta",
        option_ids,
        index=option_ids.index(default_q) if default_q in option_ids else 0,
        format_func=lambda qid: label_map.get(qid, str(qid)),
    )
    item_row = items_prueba[items_prueba["QuestionId"] == selected_q].iloc[0]

    st.markdown("### Lectura pedagógica de la pregunta")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("% de acierto", f"{item_row['dificultad_pct']:.2f}%")
    m2.metric("Discriminación (p bis)", f"{item_row['p_bis']:.3f}" if pd.notna(item_row["p_bis"]) else "Sin dato")
    m3.metric("Discriminación (D27)", f"{item_row['d27']:.3f}" if pd.notna(item_row["d27"]) else "Sin dato")
    m4.metric("Grado", str(item_row["Grado Etiqueta"]))

    st.markdown(f"**Competencia:** {item_row['Competencia']}")
    st.markdown(f"**Pregunta:** {item_row['Pregunta']}")

    option_comp = build_option_comparison(selected_q, colombia_prueba_df, focus_prueba_df if not focus_prueba_df.empty else colombia_prueba_df)
    if option_comp.empty:
        st.info("No hay opciones de respuesta para esta pregunta con el filtro actual.")
        return

    st.caption("Verde = respuesta correcta | Gris = distractores | Círculo = Colombia | Rombo = sede focal")
    st.plotly_chart(option_dumbbell_chart(option_comp, focus_label), use_container_width=True)

    display_table = option_comp[["Opción de respuesta", "pct_colombia", "pct_sede", "tipo"]].rename(columns={
        "pct_colombia": "% que la marcó en Colombia",
        "pct_sede": f"% que la marcó en {focus_label}",
        "tipo": "Tipo de opción",
    })
    st.dataframe(display_table, use_container_width=True, hide_index=True)

    st.markdown("**Sugerencia de lectura docente**")
    notes = []
    if item_row["dificultad"] < 0.30:
        notes.append("Es una pregunta exigente. Conviene revisar si el contenido ya fue trabajado con suficiente profundidad o si el enunciado requiere ajuste.")
    elif item_row["dificultad"] > 0.80:
        notes.append("Es una pregunta accesible. Sirve para verificar aprendizajes básicos, aunque suele diferenciar menos entre estudiantes.")
    else:
        notes.append("Tiene una dificultad equilibrada y ayuda a observar diferencias reales de comprensión.")
    if pd.notna(item_row["p_bis"]) and item_row["p_bis"] < 0:
        notes.append("La discriminación negativa es una alerta. Vale la pena revisar la clave, el enunciado o la alineación con lo enseñado.")
    elif pd.notna(item_row["p_bis"]) and item_row["p_bis"] >= 0.20:
        notes.append("La pregunta diferencia razonablemente entre niveles de desempeño.")
    strongest_distractor = option_comp[option_comp["tipo"] == "Distractor"]
    if not strongest_distractor.empty and strongest_distractor.iloc[0]["pct_sede"] >= 25:
        notes.append("El distractor principal está capturando muchas respuestas en la sede focal. Puede revelar una confusión frecuente que merece retroalimentación explícita.")
    for note in notes:
        st.write(f"- {note}")

    scatter = items_prueba.copy()
    scatter["item_label"] = scatter["selector_label"]
    fig = px.scatter(
        scatter,
        x="dificultad",
        y="p_bis",
        color="Competencia",
        hover_name="item_label",
        hover_data={"d27": True, "dificultad_pct": True},
        title=f"{selected_prueba}: mapa de preguntas (dificultad y discriminación)"
    )
    fig.add_vline(x=0.50, line_dash="dash")
    fig.add_hline(y=0.20, line_dash="dash")
    fig.update_layout(
        xaxis_title="Dificultad (proporción de acierto)",
        yaxis_title="Discriminación (point-biserial)"
    )
    st.plotly_chart(fig, use_container_width=True)


def show_antiguedad_tab(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str):
    st.subheader("Análisis de antigüedad del estudiante")
    with st.expander("Cómo leer esta pestaña", expanded=False):
        st.markdown(
            """
            - Esta vista muestra si el desempeño cambia según los años de permanencia del estudiante.
            - La línea compara la sede focal con Colombia.
            - La brecha ayuda a ver si la ventaja o rezago crece, se mantiene o se corrige con el tiempo.
            """
        )

    net = add_benchmark(df.dropna(subset=["Antiguedad"]), focus_df.dropna(subset=["Antiguedad"]), "Antiguedad")
    if net.empty:
        st.info("No hay datos de antigüedad para el filtro actual.")
        return

    net = net.sort_values("Antiguedad")
    c1, c2 = st.columns(2)
    with c1:
        melted = net.melt(
            id_vars="Antiguedad",
            value_vars=["acierto_red_pct", "acierto_sede_pct"],
            var_name="Serie",
            value_name="Porcentaje"
        )
        melted["Serie"] = melted["Serie"].map({
            "acierto_red_pct": "Colombia",
            "acierto_sede_pct": focus_label,
        })
        fig = px.line(melted, x="Antiguedad", y="Porcentaje", color="Serie", markers=True, title="Desempeño por antigüedad")
        fig.update_layout(yaxis_title="% de acierto", xaxis_title="Años de antigüedad")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(net, x="Antiguedad", y="brecha_pp", title=f"Brecha por antigüedad: {focus_label} vs Colombia", text="brecha_pp")
        fig.add_hline(y=0, line_dash="dash")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(yaxis_title="Puntos porcentuales", xaxis_title="Años de antigüedad")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Cruce de antigüedad por prueba en la sede focal**")
    heat = (
        focus_df.dropna(subset=["Antiguedad"])
        .groupby(["Antiguedad", "Prueba Base"], dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
        .pivot(index="Antiguedad", columns="Prueba Base", values="Acierto")
    )
    if heat.empty:
        st.info("La sede focal no tiene suficiente información en este filtro para el cruce.")
    else:
        fig = px.imshow(heat, text_auto=True, aspect="auto", title=f"{focus_label}: % de acierto por antigüedad y prueba")
        st.plotly_chart(fig, use_container_width=True)


def show_exports_tab(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str):
    st.subheader("Descargas agregadas")
    summary_sede = add_benchmark(df, focus_df, "Sede")
    summary_prueba = add_benchmark(df, focus_df, "Prueba Base")
    summary_comp = add_benchmark(df, focus_df, "Competencia")
    summary_ant = add_benchmark(df.dropna(subset=["Antiguedad"]), focus_df.dropna(subset=["Antiguedad"]), "Antiguedad")

    item_net = item_metrics(df)
    item_focus = item_metrics(focus_df) if not focus_df.empty else pd.DataFrame()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Resumen por sede**")
        st.dataframe(summary_sede, use_container_width=True, hide_index=True, height=260)
        st.download_button(
            "Descargar resumen por sede",
            data=summary_sede.to_csv(index=False).encode("utf-8-sig"),
            file_name="resumen_por_sede.csv",
            mime="text/csv"
        )
    with c2:
        st.markdown("**Resumen por prueba**")
        st.dataframe(summary_prueba, use_container_width=True, hide_index=True, height=260)
        st.download_button(
            "Descargar resumen por prueba",
            data=summary_prueba.to_csv(index=False).encode("utf-8-sig"),
            file_name="resumen_por_prueba.csv",
            mime="text/csv"
        )

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Resumen por competencia**")
        st.dataframe(summary_comp, use_container_width=True, hide_index=True, height=260)
        st.download_button(
            "Descargar resumen por competencia",
            data=summary_comp.to_csv(index=False).encode("utf-8-sig"),
            file_name="resumen_por_competencia.csv",
            mime="text/csv"
        )
    with c4:
        st.markdown("**Resumen por antigüedad**")
        st.dataframe(summary_ant, use_container_width=True, hide_index=True, height=260)
        st.download_button(
            "Descargar resumen por antigüedad",
            data=summary_ant.to_csv(index=False).encode("utf-8-sig"),
            file_name="resumen_por_antiguedad.csv",
            mime="text/csv"
        )

    st.markdown("**Banco psicométrico agregado**")
    scope_choice = st.radio("Fuente del banco", ["Colombia", focus_label], horizontal=True)
    table = item_net if scope_choice == "Colombia" else item_focus
    if table.empty:
        st.info("No hay datos para el banco seleccionado.")
    else:
        st.dataframe(
            table[[
                "QuestionId", "Prueba Base", "Competencia", "dificultad_pct",
                "p_bis", "d27", "distractor_top", "pct_distractor_top", "estado_item"
            ]],
            use_container_width=True,
            hide_index=True,
            height=420
        )
        st.download_button(
            "Descargar banco psicométrico",
            data=table.to_csv(index=False).encode("utf-8-sig"),
            file_name="banco_psicometrico.csv",
            mime="text/csv"
        )


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    st.sidebar.header("Enfoque del tablero")

    sedes_base = (
        df[["Sede", "Sede Corta"]]
        .dropna()
        .drop_duplicates()
        .sort_values("Sede Corta")
    )
    sede_labels = ["Todas las sedes"]
    label_to_sede = {"Todas las sedes": None}
    for _, row in sedes_base.iterrows():
        sede_labels.append(row["Sede Corta"])
        label_to_sede[row["Sede Corta"]] = row["Sede"]

    default_index = 1 if len(sede_labels) > 1 else 0
    sede_focal_label = st.sidebar.selectbox("Sede focal", sede_labels, index=default_index)

    grades = sorted(df["Grado"].dropna().unique().tolist(), key=grade_sort_key)
    selected_grades = st.sidebar.multiselect("Grado", grades, default=grades)

    df_grado = df[df["Grado"].isin(selected_grades)] if selected_grades else df.iloc[0:0]

    courses = sorted(df_grado["Curso"].dropna().unique().tolist(), key=course_sort_key)
    selected_courses = st.sidebar.multiselect("Curso", courses, default=courses)

    pruebas = sorted(df["Prueba Base"].dropna().unique().tolist())
    selected_pruebas = st.sidebar.multiselect("Prueba", pruebas, default=pruebas)

    filtered = df.copy()
    if selected_grades:
        filtered = filtered[filtered["Grado"].isin(selected_grades)]
    if selected_courses:
        filtered = filtered[filtered["Curso"].isin(selected_courses)]
    if selected_pruebas:
        filtered = filtered[filtered["Prueba Base"].isin(selected_pruebas)]

    sede_value = label_to_sede.get(sede_focal_label)
    if sede_value is None:
        focus_df = filtered.copy()
        focus_label = "Colombia"
    else:
        focus_df = filtered[filtered["Sede"] == sede_value].copy()
        focus_label = sede_focal_label

    return filtered, focus_df, focus_label


def main():
    st.title("Evaluación Diagnóstica 2026 Calendario A")
    st.caption("Tablero pedagógico para comparar el desempeño de una sede frente a Colombia y leer mejor lo que cuentan las preguntas.")

    local_file = load_default_file_if_exists()
    if local_file is None:
        st.error("No encontré el archivo base. Verifica que exista en data/EvaluarParaAvanzar_CalA.xlsx")
        st.stop()

    raw = read_dataframe_from_path(str(local_file))
    raw = normalize_columns(raw)
    missing = validate_columns(raw)
    if missing:
        st.error(f"Faltan columnas obligatorias: {', '.join(missing)}")
        st.stop()

    df = prepare_data(raw)
    filtered, focus_df, focus_label = apply_filters(df)

    if filtered.empty:
        st.warning("No hay datos para la combinación de filtros elegida.")
        st.stop()

    tabs = st.tabs([
        "Tablero directivo",
        "Detalle por prueba",
        "Análisis de las respuestas",
        "Análisis de antigüedad del estudiante"
    ])

    with tabs[0]:
        show_overview_tab(filtered, focus_df, focus_label)
    with tabs[1]:
        show_pruebas_tab(filtered, focus_df, focus_label)
    with tabs[2]:
        show_psychometrics_tab(filtered, focus_df, focus_label)
    with tabs[3]:
        show_antiguedad_tab(filtered, focus_df, focus_label)


if __name__ == "__main__":
    main()
