import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Visualizador académico", layout="wide")

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

DEFAULT_DATA_CANDIDATES = [
    Path("data/EvaluarParaAvanzar_CalA.xlsx"),
    Path("EvaluarParaAvanzar_CalA.xlsx"),
]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = []
    for c in df.columns.tolist():
        c2 = str(c).strip()
        key = c2.lower()
        cleaned.append(COLUMN_ALIASES.get(key, c2))
    df.columns = cleaned
    return df

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

def clean_data(df: pd.DataFrame, deduplicate: bool = True) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    df = normalize_columns(df)

    if "Acierto" in df.columns:
        df["Acierto"] = pd.to_numeric(df["Acierto"], errors="coerce").fillna(0)
        df["Acierto"] = df["Acierto"].clip(lower=0, upper=1)

    for col in ["Edad Estudiante", "Antiguedad", "QuestionId", "AnswerId"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = ["Sede", "Genero", "Grado", "Curso", "Prueba", "Pregunta", "Respuesta", "Competencia"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["nan", "None", ""]), col] = np.nan

    duplicate_count = 0
    key_cols = [c for c in ["ID Estudiante", "Prueba", "QuestionId"] if c in df.columns]
    if deduplicate and len(key_cols) == 3:
        duplicate_count = int(df.duplicated(subset=key_cols).sum())
        df = df.drop_duplicates(subset=key_cols, keep="last")

    return df, duplicate_count

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    st.sidebar.header("Filtros")
    for col in ["Sede", "Genero", "Grado", "Curso", "Prueba", "Competencia"]:
        if col in filtered.columns:
            options = sorted([x for x in filtered[col].dropna().unique().tolist()])
            if options:
                selected = st.sidebar.multiselect(col, options, default=options)
                filtered = filtered[filtered[col].isin(selected)]

    if "Edad Estudiante" in filtered.columns and filtered["Edad Estudiante"].notna().any():
        min_age = int(filtered["Edad Estudiante"].min())
        max_age = int(filtered["Edad Estudiante"].max())
        age_range = st.sidebar.slider("Edad", min_age, max_age, (min_age, max_age))
        filtered = filtered[filtered["Edad Estudiante"].fillna(min_age).between(age_range[0], age_range[1])]

    if "Antiguedad" in filtered.columns and filtered["Antiguedad"].notna().any():
        min_ant = int(filtered["Antiguedad"].min())
        max_ant = int(filtered["Antiguedad"].max())
        ant_range = st.sidebar.slider("Antigüedad", min_ant, max_ant, (min_ant, max_ant))
        filtered = filtered[filtered["Antiguedad"].fillna(min_ant).between(ant_range[0], ant_range[1])]

    return filtered

def build_student_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    g = (
        df.groupby("ID Estudiante", dropna=False)
        .agg(
            respuestas=("Acierto", "size"),
            aciertos=("Acierto", "sum"),
            promedio=("Acierto", "mean"),
            sede=("Sede", "first"),
            grado=("Grado", "first"),
            curso=("Curso", "first"),
            genero=("Genero", "first"),
            edad=("Edad Estudiante", "first"),
            antiguedad=("Antiguedad", "first"),
        )
        .reset_index()
    )
    g["porcentaje"] = (g["promedio"] * 100).round(2)
    g["riesgo"] = pd.cut(
        g["porcentaje"],
        bins=[-0.1, 39.99, 59.99, 79.99, 100.0],
        labels=["Alto", "Medio", "Bajo", "Sobresaliente"]
    )
    return g.sort_values("porcentaje", ascending=False)

def build_question_summary(df: pd.DataFrame, student_summary: pd.DataFrame) -> pd.DataFrame:
    if df.empty or student_summary.empty:
        return pd.DataFrame()

    student_perf = student_summary[["ID Estudiante", "promedio"]].copy()
    q1 = student_perf["promedio"].quantile(0.25)
    q3 = student_perf["promedio"].quantile(0.75)

    merged = df.merge(student_perf, on="ID Estudiante", how="left")
    merged["segmento"] = np.where(
        merged["promedio"] >= q3,
        "Alto",
        np.where(merged["promedio"] <= q1, "Bajo", "Medio")
    )

    base = (
        merged.groupby(["QuestionId", "Pregunta", "Prueba", "Competencia"], dropna=False)
        .agg(
            intentos=("Acierto", "size"),
            dificultad=("Acierto", "mean"),
        )
        .reset_index()
    )

    seg = (
        merged.groupby(["QuestionId", "segmento"], dropna=False)["Acierto"]
        .mean()
        .unstack(fill_value=np.nan)
        .reset_index()
    )

    out = base.merge(seg, on="QuestionId", how="left")
    out["discriminacion_simple"] = out.get("Alto", np.nan) - out.get("Bajo", np.nan)
    out["dificultad_pct"] = (out["dificultad"] * 100).round(2)
    out["discriminacion_simple"] = out["discriminacion_simple"].round(3)
    return out.sort_values(["dificultad", "intentos"], ascending=[True, False])

def kpi_block(df: pd.DataFrame):
    n_rows = len(df)
    n_students = df["ID Estudiante"].nunique() if "ID Estudiante" in df.columns else 0
    accuracy = df["Acierto"].mean() if not df.empty else 0
    n_questions = df["QuestionId"].nunique() if "QuestionId" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas analizadas", f"{n_rows:,}")
    c2.metric("Estudiantes", f"{n_students:,}")
    c3.metric("Preguntas", f"{n_questions:,}")
    c4.metric("% acierto", f"{accuracy * 100:,.2f}%")

def show_general_tab(df: pd.DataFrame, student_summary: pd.DataFrame):
    st.subheader("Panorama general")

    if df.empty:
        st.warning("No hay datos con los filtros seleccionados.")
        return

    kpi_block(df)

    c1, c2 = st.columns(2)
    with c1:
        by_prueba = (
            df.groupby("Prueba", dropna=False)["Acierto"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index()
            .sort_values("Acierto", ascending=False)
        )
        fig = px.bar(by_prueba, x="Prueba", y="Acierto", title="% de acierto por prueba")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        by_comp = (
            df.groupby("Competencia", dropna=False)["Acierto"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index()
            .sort_values("Acierto", ascending=False)
        )
        fig = px.bar(by_comp, x="Competencia", y="Acierto", title="% de acierto por competencia")
        st.plotly_chart(fig, use_container_width=True)

    by_sede = (
        df.groupby("Sede", dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index()
        .sort_values("Acierto", ascending=False)
    )
    fig = px.bar(by_sede, x="Sede", y="Acierto", title="% de acierto por sede")
    st.plotly_chart(fig, use_container_width=True)

    heat = (
        df.groupby(["Grado", "Competencia"], dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
        .pivot(index="Grado", columns="Competencia", values="Acierto")
    )
    st.markdown("**Matriz de desempeño: grado x competencia**")
    st.dataframe(heat, use_container_width=True)

    st.markdown("**Distribución del desempeño estudiantil**")
    fig = px.histogram(student_summary, x="porcentaje", nbins=20, title="Distribución del % de acierto por estudiante")
    st.plotly_chart(fig, use_container_width=True)

def show_students_tab(df: pd.DataFrame, student_summary: pd.DataFrame):
    st.subheader("Exploración por estudiante")

    if student_summary.empty:
        st.warning("No hay estudiantes para mostrar.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        selected = st.selectbox("Selecciona un estudiante", student_summary["ID Estudiante"].dropna().tolist())
    with c2:
        sort_mode = st.selectbox("Orden tabla", ["Menor desempeño", "Mayor desempeño"])

    ranking = student_summary.sort_values("porcentaje", ascending=(sort_mode == "Menor desempeño"))
    st.markdown("**Ranking de estudiantes (filtrado)**")
    st.dataframe(
        ranking[["ID Estudiante", "sede", "grado", "curso", "porcentaje", "riesgo", "respuestas"]].head(100),
        use_container_width=True,
        height=320
    )

    one = student_summary[student_summary["ID Estudiante"] == selected].iloc[0]
    df_student = df[df["ID Estudiante"] == selected].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acierto", f"{one['porcentaje']:.2f}%")
    c2.metric("Respuestas", int(one["respuestas"]))
    c3.metric("Grado / Curso", f"{one['grado']} / {one['curso']}")
    c4.metric("Riesgo", str(one["riesgo"]))

    c1, c2 = st.columns(2)
    with c1:
        by_prueba = (
            df_student.groupby("Prueba", dropna=False)["Acierto"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index()
            .sort_values("Acierto", ascending=False)
        )
        fig = px.bar(by_prueba, x="Prueba", y="Acierto", title="% de acierto del estudiante por prueba")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        by_comp = (
            df_student.groupby("Competencia", dropna=False)["Acierto"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index()
            .sort_values("Acierto", ascending=False)
        )
        fig = px.bar(by_comp, x="Competencia", y="Acierto", title="% de acierto del estudiante por competencia")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Preguntas incorrectas del estudiante**")
    wrong = df_student[df_student["Acierto"] == 0][["Prueba", "QuestionId", "Competencia", "Pregunta", "Respuesta"]]
    st.dataframe(wrong, use_container_width=True, height=350)

def show_questions_tab(df: pd.DataFrame, question_summary: pd.DataFrame):
    st.subheader("Análisis por pregunta")

    if question_summary.empty:
        st.warning("No hay preguntas para mostrar.")
        return

    c1, c2 = st.columns(2)
    with c1:
        hardest = question_summary.nsmallest(20, "dificultad")[["QuestionId", "Prueba", "Competencia", "dificultad_pct", "intentos"]]
        st.markdown("**Preguntas más difíciles**")
        st.dataframe(hardest, use_container_width=True, height=350)

    with c2:
        best_disc = question_summary.sort_values("discriminacion_simple", ascending=False).head(20)
        best_disc = best_disc[["QuestionId", "Prueba", "Competencia", "discriminacion_simple", "intentos"]]
        st.markdown("**Preguntas con mejor discriminación**")
        st.dataframe(best_disc, use_container_width=True, height=350)

    selected_q = st.selectbox("Selecciona una pregunta", question_summary["QuestionId"].dropna().tolist())
    q_meta = question_summary[question_summary["QuestionId"] == selected_q].iloc[0]

    st.markdown(f"**Prueba:** {q_meta['Prueba']}")
    st.markdown(f"**Competencia:** {q_meta['Competencia']}")
    st.markdown(f"**Dificultad:** {q_meta['dificultad_pct']:.2f}%")
    if pd.notna(q_meta.get("discriminacion_simple", np.nan)):
        st.markdown(f"**Discriminación simple:** {q_meta['discriminacion_simple']:.3f}")
    st.markdown("**Enunciado:**")
    st.write(q_meta["Pregunta"])

    df_q = df[df["QuestionId"] == selected_q].copy()
    by_group = (
        df_q.groupby(["Grado", "Curso"], dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index()
        .sort_values("Acierto", ascending=False)
    )
    fig = px.bar(by_group, x="Curso", y="Acierto", color="Grado", title="% de acierto por curso")
    st.plotly_chart(fig, use_container_width=True)

    answers = (
        df_q.groupby(["Respuesta", "Acierto"], dropna=False)
        .size()
        .reset_index(name="conteo")
        .sort_values("conteo", ascending=False)
    )
    st.markdown("**Respuestas elegidas**")
    st.dataframe(answers, use_container_width=True, height=300)

def show_comparisons_tab(df: pd.DataFrame):
    st.subheader("Comparativos")

    if df.empty:
        st.warning("No hay datos para comparar.")
        return

    dim = st.selectbox("Comparar por", ["Sede", "Grado", "Curso", "Genero", "Prueba", "Competencia"])

    comp = (
        df.groupby(dim, dropna=False)
        .agg(
            estudiantes=("ID Estudiante", "nunique"),
            respuestas=("Acierto", "size"),
            acierto=("Acierto", "mean"),
        )
        .reset_index()
    )
    comp["acierto_pct"] = (comp["acierto"] * 100).round(2)
    comp = comp.sort_values("acierto_pct", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(comp, x=dim, y="acierto_pct", title=f"% de acierto por {dim}")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(comp, x="estudiantes", y="acierto_pct", size="respuestas", hover_name=dim, title="Cobertura vs desempeño")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(comp[[dim, "estudiantes", "respuestas", "acierto_pct"]], use_container_width=True)

def show_data_tab(df: pd.DataFrame):
    st.subheader("Datos filtrados")

    search = st.text_input("Buscar texto en pregunta o respuesta")
    data = df.copy()
    if search:
        mask_q = data["Pregunta"].fillna("").str.contains(search, case=False, na=False)
        mask_r = data["Respuesta"].fillna("").str.contains(search, case=False, na=False)
        data = data[mask_q | mask_r]

    st.dataframe(data, use_container_width=True, height=500)

    csv = data.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Descargar datos filtrados",
        data=csv,
        file_name="datos_filtrados.csv",
        mime="text/csv"
    )

def load_default_file_if_exists():
    for path in DEFAULT_DATA_CANDIDATES:
        if path.exists():
            return path
    return None

def main():
    st.title("Visualizador académico de resultados")
    st.caption("Explora desempeño por sede, curso, prueba, competencia, estudiante y pregunta.")

    with st.expander("Qué hace este visualizador", expanded=False):
        st.markdown(
            """
            - Resume el desempeño general.
            - Permite filtrar por variables académicas y demográficas.
            - Analiza estudiantes individualmente.
            - Analiza dificultad y discriminación simple por pregunta.
            - Permite exportar los datos filtrados.
            """
        )

    local_file = load_default_file_if_exists()
    uploaded_file = st.file_uploader("Si quieres, carga otro archivo (CSV o Excel)", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        source_label = f"Archivo cargado: {uploaded_file.name}"
        raw = read_uploaded_file(uploaded_file)
    elif local_file is not None:
        source_label = f"Archivo del repositorio: {local_file}"
        raw = read_dataframe_from_path(str(local_file))
    else:
        st.info("No encontré un archivo en el repositorio. Sube uno para comenzar.")
        st.stop()

    st.success(source_label)

    dedupe = st.checkbox(
        "Eliminar duplicados por ID Estudiante + Prueba + QuestionId",
        value=True,
        help="Recomendado cuando existe más de un registro para la misma respuesta del mismo estudiante."
    )

    raw = normalize_columns(raw)
    missing = validate_columns(raw)
    if missing:
        st.error(f"Faltan columnas obligatorias: {', '.join(missing)}")
        st.stop()

    df, duplicate_count = clean_data(raw, deduplicate=dedupe)

    if duplicate_count > 0 and dedupe:
        st.warning(f"Se detectaron y removieron {duplicate_count:,} duplicados potenciales.")

    if df.empty:
        st.warning("El archivo quedó vacío después de la limpieza.")
        st.stop()

    df_filtered = apply_filters(df)
    student_summary = build_student_summary(df_filtered)
    question_summary = build_question_summary(df_filtered, student_summary)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Resumen", "Estudiantes", "Preguntas", "Comparativos", "Datos"])

    with tab1:
        show_general_tab(df_filtered, student_summary)
    with tab2:
        show_students_tab(df_filtered, student_summary)
    with tab3:
        show_questions_tab(df_filtered, question_summary)
    with tab4:
        show_comparisons_tab(df_filtered)
    with tab5:
        show_data_tab(df_filtered)

if __name__ == "__main__":
    main()
