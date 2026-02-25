
from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Visualizador pedagógico y psicométrico", layout="wide")

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
    "transicion": 0,
    "transición": 0,
    "prejardin": -2,
    "pre jardín": -2,
    "prejardín": -2,
    "jardin": -1,
    "jardín": -1,
    "primero": 1,
    "segundo": 2,
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
}

P_BIS_TARGET = 0.20
D27_TARGET = 0.20


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in text if not unicodedata.combining(ch))


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


def clean_prueba_label(value: str) -> str:
    raw = str(value).strip()
    cleaned = re.sub(r"\s+\d+\s*°?$", "", raw).strip()
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

    return out.sort_values(["Prueba Base", "dificultad", "p_bis"], ascending=[True, True, False])


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
        title={"text": title},
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
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def benchmark_cards(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str):
    red_pct = safe_pct(df["Acierto"])
    focus_pct = safe_pct(focus_df["Acierto"])
    brecha = focus_pct - red_pct
    sedes = (
        df.groupby("Sede", dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )
    best_sede = sedes.iloc[0]["Sede"] if not sedes.empty else "Sin dato"
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
    with c1:
        st.plotly_chart(make_indicator("Promedio red", red_pct), use_container_width=True)
    with c2:
        st.plotly_chart(make_indicator(f"Promedio {focus_label}", focus_pct, delta=brecha, reference=red_pct), use_container_width=True)
    with c3:
        st.plotly_chart(make_indicator(f"Mejor sede: {best_sede}", best_sede_pct), use_container_width=True)
    with c4:
        st.plotly_chart(make_indicator(f"Prueba crítica: {prueba_critica}", prueba_critica_pct), use_container_width=True)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Estudiantes red", f"{safe_nunique(df['ID Estudiante']):,}")
    c6.metric(f"Estudiantes {focus_label}", f"{safe_nunique(focus_df['ID Estudiante']):,}")
    c7.metric("Respuestas analizadas", f"{len(df):,}")
    c8.metric("Brecha sede vs red", f"{brecha:+.2f} pp")


def show_overview_tab(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str):
    st.subheader("Panorama global para conversación pedagógica")

    benchmark_cards(df, focus_df, focus_label)

    by_sede = add_benchmark(df, focus_df, "Sede").sort_values("acierto_red_pct", ascending=False)
    by_sede["promedio_red_global"] = safe_pct(df["Acierto"])

    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig = px.bar(
            by_sede,
            x="Sede",
            y="acierto_red_pct",
            title="% de acierto por sede",
            text="acierto_red_pct"
        )
        fig.add_hline(y=safe_pct(df["Acierto"]), line_dash="dash", annotation_text="Promedio red")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(yaxis_title="% de acierto", xaxis_title="")
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
            "acierto_red_pct": "Red",
            "acierto_sede_pct": focus_label,
        })
        fig = px.bar(
            melted,
            x="Prueba Base",
            y="Porcentaje",
            color="Serie",
            barmode="group",
            title=f"% de acierto por prueba: {focus_label} vs red"
        )
        fig.update_layout(yaxis_title="% de acierto", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Mapa de calor por sede y prueba**")
    heat = (
        df.groupby(["Sede", "Prueba Base"], dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
        .pivot(index="Sede", columns="Prueba Base", values="Acierto")
    )
    fig = px.imshow(heat, text_auto=True, aspect="auto", title="% de acierto por sede y prueba")
    st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
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
        melted["Serie"] = melted["Serie"].map({"Acierto": "Red", "Acierto Focus": focus_label})
        fig = px.line(melted, x="Grado", y="Porcentaje", color="Serie", markers=True, title=f"Trayectoria por grado: {focus_label} vs red")
        fig.update_layout(yaxis_title="% de acierto", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        comp = add_benchmark(df, focus_df, "Competencia").sort_values("brecha_pp")
        fig = px.bar(
            comp,
            x="brecha_pp",
            y="Competencia",
            orientation="h",
            title=f"Brecha por competencia: {focus_label} vs red",
            text="brecha_pp"
        )
        fig.add_vline(x=0, line_dash="dash")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(xaxis_title="Puntos porcentuales", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Lectura rápida para profesores**")
    comp = add_benchmark(df, focus_df, "Competencia").sort_values("brecha_pp")
    weakest = comp.head(3)[["Competencia", "brecha_pp", "acierto_sede_pct", "acierto_red_pct"]].copy()
    strongest = comp.tail(3).sort_values("brecha_pp", ascending=False)[["Competencia", "brecha_pp", "acierto_sede_pct", "acierto_red_pct"]].copy()

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"**Donde {focus_label} necesita más apoyo**")
        st.dataframe(weakest, use_container_width=True, hide_index=True)
    with t2:
        st.markdown(f"**Donde {focus_label} está más fuerte**")
        st.dataframe(strongest, use_container_width=True, hide_index=True)


def render_prueba_panel(prueba: str, df_prueba: pd.DataFrame, focus_prueba: pd.DataFrame, focus_label: str):
    if df_prueba.empty:
        st.info("No hay datos para esta prueba.")
        return

    red_pct = safe_pct(df_prueba["Acierto"])
    focus_pct = safe_pct(focus_prueba["Acierto"]) if not focus_prueba.empty else np.nan
    brecha = focus_pct - red_pct if pd.notna(focus_pct) else np.nan

    by_comp = add_benchmark(df_prueba, focus_prueba if not focus_prueba.empty else df_prueba.iloc[0:0], "Competencia").sort_values("acierto_red_pct")
    comp_critica = by_comp.iloc[0]["Competencia"] if not by_comp.empty else "Sin dato"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Promedio red", f"{red_pct:.2f}%")
    c2.metric(f"Promedio {focus_label}", f"{focus_pct:.2f}%" if pd.notna(focus_pct) else "Sin dato")
    c3.metric("Brecha", f"{brecha:+.2f} pp" if pd.notna(brecha) else "Sin dato")
    c4.metric("Competencia crítica", comp_critica)

    c5, c6 = st.columns([1.1, 1])
    with c5:
        melted = by_comp.melt(
            id_vars="Competencia",
            value_vars=["acierto_red_pct", "acierto_sede_pct"],
            var_name="Serie",
            value_name="Porcentaje"
        )
        melted["Serie"] = melted["Serie"].map({
            "acierto_red_pct": "Red",
            "acierto_sede_pct": focus_label,
        })
        fig = px.bar(
            melted,
            x="Competencia",
            y="Porcentaje",
            color="Serie",
            barmode="group",
            title=f"{prueba}: competencias, {focus_label} vs red"
        )
        fig.update_layout(yaxis_title="% de acierto", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        heat = (
            df_prueba.groupby(["Sede", "Competencia"], dropna=False)["Acierto"]
            .mean()
            .mul(100)
            .round(1)
            .reset_index()
            .pivot(index="Sede", columns="Competencia", values="Acierto")
        )
        fig = px.imshow(heat, text_auto=True, aspect="auto", title=f"{prueba}: desempeño por sede y competencia")
        st.plotly_chart(fig, use_container_width=True)

    c7, c8 = st.columns([1, 1])
    with c7:
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
        melted["Serie"] = melted["Serie"].map({"Acierto": "Red", "Acierto Focus": focus_label})
        fig = px.line(melted, x="Grado", y="Porcentaje", color="Serie", markers=True, title=f"{prueba}: trayectoria por grado")
        st.plotly_chart(fig, use_container_width=True)

    with c8:
        comp = by_comp.sort_values("brecha_pp")
        fig = px.bar(
            comp,
            x="brecha_pp",
            y="Competencia",
            orientation="h",
            title=f"{prueba}: brecha {focus_label} vs red",
            text="brecha_pp"
        )
        fig.add_vline(x=0, line_dash="dash")
        fig.update_traces(texttemplate="%{text:.2f}")
        fig.update_layout(xaxis_title="Puntos porcentuales", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        by_comp[["Competencia", "acierto_sede_pct", "acierto_red_pct", "brecha_pp", "estudiantes_red"]],
        use_container_width=True,
        hide_index=True
    )


def show_pruebas_tab(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str):
    st.subheader("Hojas por prueba")
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
    out["tipo"] = np.where(out["Acierto"] == 1, "Clave", "Distractor")
    return out[["tipo", "Respuesta Limpia", "selecciones", "porcentaje"]]


def show_psychometrics_tab(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str):
    st.subheader("Laboratorio psicométrico")
    scope_options = ["Red"]
    if focus_label != "Sede focal":
        scope_options.append(focus_label)
    scope = st.radio("Analizar ítems en", scope_options, horizontal=True)
    selected = df if scope == "Red" else focus_df

    if selected.empty:
        st.warning("No hay datos en el alcance seleccionado.")
        return

    items = item_metrics(selected)
    if items.empty:
        st.warning("No fue posible calcular métricas de ítem.")
        return

    pruebas = sorted(items["Prueba Base"].dropna().unique().tolist())
    selected_prueba = st.selectbox("Prueba para análisis psicométrico", pruebas)
    items_prueba = items[items["Prueba Base"] == selected_prueba].copy()
    scope_prueba_df = selected[selected["Prueba Base"] == selected_prueba].copy()

    summary = psychometric_summary(items_prueba)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ítems analizados", f"{summary['n_items']:,}")
    c2.metric("Dificultad en rango útil", f"{summary['pct_target_difficulty']:.1f}%")
    c3.metric("Buena discriminación", f"{summary['pct_good_discrimination']:.1f}%")
    c4.metric("Ítems en alerta", f"{summary['pct_alert']:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    hardest = items_prueba.nsmallest(1, "dificultad")
    easiest = items_prueba.nlargest(1, "dificultad")
    best_disc = items_prueba.sort_values(["p_bis", "d27"], ascending=[False, False]).head(1)
    worst_disc = items_prueba.sort_values(["p_bis", "d27"], ascending=[True, True]).head(1)

    c5.metric("Pregunta más difícil", str(int(hardest.iloc[0]["QuestionId"])) if not hardest.empty else "Sin dato")
    c6.metric("Pregunta más fácil", str(int(easiest.iloc[0]["QuestionId"])) if not easiest.empty else "Sin dato")
    c7.metric("Mayor discriminación", str(int(best_disc.iloc[0]["QuestionId"])) if not best_disc.empty else "Sin dato")
    c8.metric("Menor discriminación", str(int(worst_disc.iloc[0]["QuestionId"])) if not worst_disc.empty else "Sin dato")

    c9, c10 = st.columns([1.15, 0.85])
    with c9:
        scatter = items_prueba.copy()
        scatter["item_label"] = scatter["QuestionId"].astype(str)
        fig = px.scatter(
            scatter,
            x="dificultad",
            y="p_bis",
            color="estado_item",
            hover_name="item_label",
            hover_data={"d27": True, "dificultad_pct": True, "Competencia": True},
            title=f"{selected_prueba}: mapa de dificultad y discriminación"
        )
        fig.add_vline(x=0.50, line_dash="dash")
        fig.add_hline(y=0.20, line_dash="dash")
        fig.update_layout(xaxis_title="Dificultad (proporción de acierto)", yaxis_title="Correlación biserial puntual")
        st.plotly_chart(fig, use_container_width=True)

    with c10:
        status = (
            items_prueba["estado_item"]
            .value_counts(dropna=False)
            .reset_index()
        )
        status.columns = ["estado_item", "n_items"]
        fig = px.pie(status, names="estado_item", values="n_items", title="Semáforo de calidad de ítems")
        st.plotly_chart(fig, use_container_width=True)

    c11, c12 = st.columns(2)
    with c11:
        hardest15 = items_prueba.nsmallest(15, "dificultad")[["QuestionId", "Competencia", "dificultad_pct", "p_bis", "d27", "estado_item"]]
        st.markdown("**Ítems más difíciles (TCT)**")
        st.dataframe(hardest15, use_container_width=True, hide_index=True, height=360)
    with c12:
        disc15 = items_prueba.sort_values(["p_bis", "d27"], ascending=False).head(15)[["QuestionId", "Competencia", "dificultad_pct", "p_bis", "d27", "estado_item"]]
        st.markdown("**Ítems con mejor discriminación**")
        st.dataframe(disc15, use_container_width=True, hide_index=True, height=360)

    item_options = items_prueba["QuestionId"].tolist()
    default_q = int(hardest.iloc[0]["QuestionId"]) if not hardest.empty else int(item_options[0])
    selected_q = st.selectbox("Explorar un ítem", item_options, index=item_options.index(default_q) if default_q in item_options else 0)
    item_row = items_prueba[items_prueba["QuestionId"] == selected_q].iloc[0]

    st.markdown("### Radiografía de la pregunta")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Dificultad", f"{item_row['dificultad_pct']:.2f}%")
    m2.metric("P bis", f"{item_row['p_bis']:.3f}" if pd.notna(item_row["p_bis"]) else "Sin dato")
    m3.metric("D27", f"{item_row['d27']:.3f}" if pd.notna(item_row["d27"]) else "Sin dato")
    m4.metric("Distractor top", f"{item_row['pct_distractor_top']:.2f}%" if pd.notna(item_row["pct_distractor_top"]) else "0.00%")
    m5.metric("Estado", item_row["estado_item"])

    st.markdown(f"**Competencia:** {item_row['Competencia']}")
    st.markdown(f"**Pregunta:** {item_row['Pregunta']}")

    c13, c14 = st.columns([1.1, 0.9])
    with c13:
        distract = distractor_table(scope_prueba_df, selected_q)
        fig = px.bar(
            distract,
            x="Respuesta Limpia",
            y="porcentaje",
            color="tipo",
            title="Opciones elegidas por los estudiantes",
            text="porcentaje"
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(xaxis_title="", yaxis_title="% de selecciones")
        st.plotly_chart(fig, use_container_width=True)

    with c14:
        wrong = distract[distract["tipo"] == "Distractor"].copy()
        st.markdown("**Distractores que más atrajeron respuestas**")
        if wrong.empty:
            st.success("No hay distractores registrados para este ítem.")
        else:
            st.dataframe(wrong, use_container_width=True, hide_index=True, height=300)
            top = wrong.iloc[0]
            st.info(
                f"El distractor más fuerte capturó {top['porcentaje']:.2f}% de las respuestas del ítem. "
                f"Si concentra mucho volumen, conviene revisar si el error revela una confusión conceptual específica."
            )

    st.markdown("**Interpretación pedagógica sugerida**")
    notes = []
    if item_row["dificultad"] < 0.30:
        notes.append("Es un ítem de alta exigencia. Útil para detectar vacíos conceptuales, pero conviene revisar si el enunciado o la clave generan ruido.")
    elif item_row["dificultad"] > 0.80:
        notes.append("Es un ítem muy accesible. Sirve para verificar aprendizajes básicos, aunque discrimina menos si casi todos aciertan.")
    else:
        notes.append("Tiene una dificultad pedagógicamente útil: permite ver diferencias reales de dominio sin bloquear a la mayoría.")
    if pd.notna(item_row["p_bis"]) and item_row["p_bis"] < 0:
        notes.append("La discriminación negativa sugiere una alerta: estudiantes de mejor desempeño fallan más que los demás. Vale la pena revisar clave, redacción o alineación curricular.")
    elif pd.notna(item_row["p_bis"]) and item_row["p_bis"] >= 0.20:
        notes.append("La discriminación es sana: el ítem separa razonablemente a quienes dominan mejor el contenido.")
    if pd.notna(item_row["pct_distractor_top"]) and item_row["pct_distractor_top"] >= 25:
        notes.append("El distractor principal es muy potente. Ese error probablemente representa una idea alternativa extendida en el aula.")
    for note in notes:
        st.write(f"- {note}")


def show_antiguedad_tab(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str):
    st.subheader("Antigüedad y trayectoria")

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
            "acierto_red_pct": "Red",
            "acierto_sede_pct": focus_label,
        })
        fig = px.line(melted, x="Antiguedad", y="Porcentaje", color="Serie", markers=True, title="Desempeño por antigüedad")
        fig.update_layout(yaxis_title="% de acierto", xaxis_title="Años de antigüedad")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(net, x="Antiguedad", y="brecha_pp", title=f"Brecha por antigüedad: {focus_label} vs red", text="brecha_pp")
        fig.add_hline(y=0, line_dash="dash")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(yaxis_title="Puntos porcentuales", xaxis_title="Años de antigüedad")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Cruce antigüedad x prueba**")
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
    scope_choice = st.radio("Fuente del banco", ["Red", focus_label], horizontal=True)
    table = item_net if scope_choice == "Red" else item_focus
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

    grades = sorted(df["Grado"].dropna().unique().tolist(), key=grade_sort_key)
    selected_grades = st.sidebar.multiselect("Grado", grades, default=grades)

    df_grado = df[df["Grado"].isin(selected_grades)] if selected_grades else df.iloc[0:0]

    courses = sorted(df_grado["Curso"].dropna().unique().tolist(), key=course_sort_key)
    selected_courses = st.sidebar.multiselect("Curso", courses, default=courses)

    pruebas = sorted(df["Prueba Base"].dropna().unique().tolist())
    selected_pruebas = st.sidebar.multiselect("Prueba", pruebas, default=pruebas)

    with st.sidebar.expander("Cambiar archivo", expanded=False):
        uploaded_file = st.file_uploader("Cargar otro CSV o Excel", type=["csv", "xlsx", "xls"], key="upload_new_file")
        st.caption("Si no cargas nada, se usa el archivo del repositorio.")
    st.session_state["uploaded_file_obj"] = uploaded_file

    filtered = df.copy()
    if selected_grades:
        filtered = filtered[filtered["Grado"].isin(selected_grades)]
    if selected_courses:
        filtered = filtered[filtered["Curso"].isin(selected_courses)]
    if selected_pruebas:
        filtered = filtered[filtered["Prueba Base"].isin(selected_pruebas)]

    sedes = sorted(filtered["Sede"].dropna().unique().tolist())
    sede_focal = st.sidebar.selectbox("Sede focal", ["Toda la red"] + sedes)

    if sede_focal == "Toda la red":
        focus_df = filtered.copy()
        focus_label = "Sede focal"
    else:
        focus_df = filtered[filtered["Sede"] == sede_focal].copy()
        focus_label = sede_focal

    return filtered, focus_df, focus_label


def main():
    st.title("Visualizador pedagógico y psicométrico")
    st.caption("Diseñado para lectura docente: más señales útiles, menos ruido. Aquí los datos hablan con gráficos y los ítems dejan huellas.")

    local_file = load_default_file_if_exists()

    uploaded = st.session_state.get("uploaded_file_obj")
    if uploaded is not None:
        source_label = f"Archivo cargado: {uploaded.name}"
        raw = read_uploaded_file(uploaded)
    elif local_file is not None:
        source_label = f"Archivo del repositorio: {local_file.relative_to(BASE_DIR) if local_file.is_relative_to(BASE_DIR) else local_file}"
        raw = read_dataframe_from_path(str(local_file))
    else:
        st.error("No encontré el archivo base. Verifica que exista en data/EvaluarParaAvanzar_CalA.xlsx")
        st.stop()

    raw = normalize_columns(raw)
    missing = validate_columns(raw)
    if missing:
        st.error(f"Faltan columnas obligatorias: {', '.join(missing)}")
        st.stop()

    df = prepare_data(raw)

    st.success(source_label)

    with st.expander("Qué aporta este tablero", expanded=False):
        st.markdown(
            """
            - Resume resultados globales de la red y de una sede focal.
            - Compara sedes, pruebas, grados, competencias y antigüedad.
            - Despliega un laboratorio psicométrico con dificultad, discriminación y distractores.
            - Prioriza resultados agregados para decisiones de aula, acompañamiento y planeación.
            """
        )

    filtered, focus_df, focus_label = apply_filters(df)

    if filtered.empty:
        st.warning("No hay datos para la combinación de filtros elegida.")
        st.stop()

    tabs = st.tabs([
        "Tablero directivo",
        "Hojas por prueba",
        "Laboratorio psicométrico",
        "Antigüedad",
        "Descargas"
    ])

    with tabs[0]:
        show_overview_tab(filtered, focus_df, focus_label)
    with tabs[1]:
        show_pruebas_tab(filtered, focus_df, focus_label)
    with tabs[2]:
        show_psychometrics_tab(filtered, focus_df, focus_label)
    with tabs[3]:
        show_antiguedad_tab(filtered, focus_df, focus_label)
    with tabs[4]:
        show_exports_tab(filtered, focus_df, focus_label)


if __name__ == "__main__":
    main()
