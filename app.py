from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Visualizador agregado académico", layout="wide")

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

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_CANDIDATES = [
    BASE_DIR / "data" / "EvaluarParaAvanzar_CalA.xlsx",
    BASE_DIR / "EvaluarParaAvanzar_CalA.xlsx",
    Path("data/EvaluarParaAvanzar_CalA.xlsx"),
    Path("EvaluarParaAvanzar_CalA.xlsx"),
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = []
    for c in df.columns:
        c2 = str(c).strip()
        key = c2.lower()
        renamed.append(COLUMN_ALIASES.get(key, c2))
    out = df.copy()
    out.columns = renamed
    return out


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


@st.cache_data(show_spinner=False)
def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(missing)}")

    df = df.copy()
    df["Acierto"] = pd.to_numeric(df["Acierto"], errors="coerce")
    df["Acierto"] = df["Acierto"].clip(lower=0, upper=1)

    numeric_cols = ["Edad Estudiante", "Antiguedad", "QuestionId", "AnswerId"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = ["Sede", "Genero", "Grado", "Curso", "Prueba", "Pregunta", "Respuesta", "Competencia"]
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()
        df.loc[df[col].isin(["", "nan", "None"]), col] = pd.NA

    return df


@st.cache_data(show_spinner=False)
def compute_aggregates(df: pd.DataFrame) -> dict:
    out = {}

    out["global_kpis"] = {
        "filas": int(len(df)),
        "estudiantes": int(df["ID Estudiante"].nunique(dropna=True)),
        "preguntas": int(df["QuestionId"].nunique(dropna=True)),
        "pruebas": int(df["Prueba"].nunique(dropna=True)),
        "acierto_red": float(df["Acierto"].mean()) if len(df) else np.nan,
    }

    out["by_sede"] = (
        df.groupby("Sede", dropna=False)
        .agg(
            registros=("Acierto", "size"),
            estudiantes=("ID Estudiante", pd.Series.nunique),
            pruebas=("Prueba", pd.Series.nunique),
            acierto=("Acierto", "mean"),
        )
        .reset_index()
        .sort_values("acierto", ascending=False)
    )
    out["by_sede"]["acierto_pct"] = (out["by_sede"]["acierto"] * 100).round(2)

    out["by_prueba"] = (
        df.groupby("Prueba", dropna=False)
        .agg(
            registros=("Acierto", "size"),
            estudiantes=("ID Estudiante", pd.Series.nunique),
            sedes=("Sede", pd.Series.nunique),
            acierto=("Acierto", "mean"),
        )
        .reset_index()
        .sort_values("acierto", ascending=False)
    )
    out["by_prueba"]["acierto_pct"] = (out["by_prueba"]["acierto"] * 100).round(2)

    out["sede_x_prueba"] = (
        df.groupby(["Sede", "Prueba"], dropna=False)
        .agg(
            registros=("Acierto", "size"),
            estudiantes=("ID Estudiante", pd.Series.nunique),
            acierto=("Acierto", "mean"),
        )
        .reset_index()
    )
    out["sede_x_prueba"]["acierto_pct"] = (out["sede_x_prueba"]["acierto"] * 100).round(2)

    out["heat_sede_prueba"] = (
        out["sede_x_prueba"]
        .pivot(index="Sede", columns="Prueba", values="acierto_pct")
    )

    out["red_x_competencia_prueba"] = (
        df.groupby(["Prueba", "Competencia"], dropna=False)
        .agg(
            registros=("Acierto", "size"),
            estudiantes=("ID Estudiante", pd.Series.nunique),
            acierto=("Acierto", "mean"),
        )
        .reset_index()
    )
    out["red_x_competencia_prueba"]["acierto_pct"] = (
        out["red_x_competencia_prueba"]["acierto"] * 100
    ).round(2)

    out["sede_x_competencia_prueba"] = (
        df.groupby(["Prueba", "Competencia", "Sede"], dropna=False)
        .agg(
            registros=("Acierto", "size"),
            estudiantes=("ID Estudiante", pd.Series.nunique),
            acierto=("Acierto", "mean"),
        )
        .reset_index()
    )
    out["sede_x_competencia_prueba"]["acierto_pct"] = (
        out["sede_x_competencia_prueba"]["acierto"] * 100
    ).round(2)

    return out


def load_default_file_if_exists() -> Path | None:
    for path in DEFAULT_DATA_CANDIDATES:
        if path.exists():
            return path
    return None


def sidebar_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    st.sidebar.header("Configuración")

    sedes = sorted(df["Sede"].dropna().unique().tolist())
    sede_focal = st.sidebar.selectbox(
        "Sede focal para comparar con la red",
        options=["Todas"] + sedes,
        index=0,
        help="La red siempre conserva todas las sedes dentro de los filtros generales. La sede focal solo se usa para el comparativo."
    )

    filtered = df.copy()

    for col in ["Grado", "Curso", "Genero"]:
        options = sorted([x for x in filtered[col].dropna().unique().tolist()])
        if options:
            selected = st.sidebar.multiselect(f"Filtrar {col}", options, default=options)
            filtered = filtered[filtered[col].isin(selected)]

    pruebas = sorted([x for x in filtered["Prueba"].dropna().unique().tolist()])
    if pruebas:
        selected_pruebas = st.sidebar.multiselect("Filtrar Prueba", pruebas, default=pruebas)
        filtered = filtered[filtered["Prueba"].isin(selected_pruebas)]

    if "Edad Estudiante" in filtered.columns and filtered["Edad Estudiante"].notna().any():
        min_age = int(filtered["Edad Estudiante"].min())
        max_age = int(filtered["Edad Estudiante"].max())
        age_range = st.sidebar.slider("Rango de edad", min_age, max_age, (min_age, max_age))
        filtered = filtered[filtered["Edad Estudiante"].fillna(min_age).between(age_range[0], age_range[1])]

    if "Antiguedad" in filtered.columns and filtered["Antiguedad"].notna().any():
        min_ant = int(filtered["Antiguedad"].min())
        max_ant = int(filtered["Antiguedad"].max())
        ant_range = st.sidebar.slider("Rango de antigüedad", min_ant, max_ant, (min_ant, max_ant))
        filtered = filtered[filtered["Antiguedad"].fillna(min_ant).between(ant_range[0], ant_range[1])]

    return filtered, (None if sede_focal == "Todas" else sede_focal)



def kpi_row(agg: dict, sede_focal: str | None):
    global_kpis = agg["global_kpis"]
    by_sede = agg["by_sede"]

    red_pct = global_kpis["acierto_red"] * 100 if pd.notna(global_kpis["acierto_red"]) else np.nan

    if sede_focal is None:
        sede_pct = red_pct
        brecha = 0.0
        sede_label = "Red completa"
    else:
        sede_row = by_sede[by_sede["Sede"] == sede_focal]
        if sede_row.empty:
            sede_pct = np.nan
            brecha = np.nan
        else:
            sede_pct = float(sede_row.iloc[0]["acierto_pct"])
            brecha = sede_pct - red_pct
        sede_label = sede_focal

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Promedio red", f"{red_pct:,.2f}%")
    c2.metric(f"Promedio {sede_label}", f"{sede_pct:,.2f}%")
    c3.metric("Brecha vs red", f"{brecha:,.2f} p.p.")
    c4.metric("Registros analizados", f"{global_kpis['filas']:,}")



def show_resumen_ejecutivo(df: pd.DataFrame, agg: dict, sede_focal: str | None):
    st.subheader("Resumen global")

    if df.empty:
        st.warning("No hay datos con los filtros seleccionados.")
        return

    kpi_row(agg, sede_focal)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Promedio por sede**")
        chart = agg["by_sede"].copy()
        fig = px.bar(chart, x="Sede", y="acierto_pct", text="acierto_pct", title="% de acierto por sede")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(chart[["Sede", "estudiantes", "registros", "acierto_pct"]], use_container_width=True)

    with c2:
        st.markdown("**Promedio por prueba en la red**")
        chart = agg["by_prueba"].copy()
        fig = px.bar(chart, x="Prueba", y="acierto_pct", title="% de acierto por prueba")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(chart[["Prueba", "estudiantes", "registros", "acierto_pct"]], use_container_width=True, height=320)

    st.markdown("**Matriz sede x prueba**")
    st.dataframe(agg["heat_sede_prueba"], use_container_width=True, height=260)



def show_detalle_sedes_y_pruebas(df: pd.DataFrame, agg: dict, sede_focal: str | None):
    st.subheader("Comportamiento por sede y por prueba")

    sxp = agg["sede_x_prueba"].copy()
    red_prueba = agg["by_prueba"][["Prueba", "acierto_pct"]].rename(columns={"acierto_pct": "red_pct"})
    sxp = sxp.merge(red_prueba, on="Prueba", how="left")
    sxp["brecha_vs_red"] = (sxp["acierto_pct"] - sxp["red_pct"]).round(2)

    if sede_focal is not None:
        foco = sxp[sxp["Sede"] == sede_focal].sort_values("brecha_vs_red", ascending=False)
        st.markdown(f"**{sede_focal} frente a la red, prueba por prueba**")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                foco,
                x="Prueba",
                y="acierto_pct",
                title=f"% de acierto de {sede_focal} por prueba"
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            comp_plot = foco[["Prueba", "acierto_pct", "red_pct"]].melt(
                id_vars="Prueba", value_vars=["acierto_pct", "red_pct"],
                var_name="Grupo", value_name="Porcentaje"
            )
            comp_plot["Grupo"] = comp_plot["Grupo"].replace({"acierto_pct": sede_focal, "red_pct": "Red"})
            fig = px.bar(comp_plot, x="Prueba", y="Porcentaje", color="Grupo", barmode="group", title="Comparación prueba a prueba")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            foco[["Prueba", "estudiantes", "registros", "acierto_pct", "red_pct", "brecha_vs_red"]],
            use_container_width=True,
            height=420
        )
    else:
        st.markdown("**Comparativo de todas las sedes por prueba**")
        fig = px.bar(sxp, x="Prueba", y="acierto_pct", color="Sede", barmode="group", title="% de acierto por prueba y sede")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            sxp[["Sede", "Prueba", "estudiantes", "registros", "acierto_pct", "red_pct", "brecha_vs_red"]].sort_values(["Prueba", "acierto_pct"], ascending=[True, False]),
            use_container_width=True,
            height=500
        )



def show_prueba_tab(df: pd.DataFrame, agg: dict, prueba: str, sede_focal: str | None):
    df_prueba = df[df["Prueba"] == prueba].copy()
    if df_prueba.empty:
        st.warning("No hay datos para esta prueba con los filtros seleccionados.")
        return

    red_prueba_row = agg["by_prueba"][agg["by_prueba"]["Prueba"] == prueba]
    red_pct = float(red_prueba_row.iloc[0]["acierto_pct"]) if not red_prueba_row.empty else np.nan

    if sede_focal is not None:
        sede_prueba_row = agg["sede_x_prueba"][
            (agg["sede_x_prueba"]["Prueba"] == prueba) &
            (agg["sede_x_prueba"]["Sede"] == sede_focal)
        ]
        sede_pct = float(sede_prueba_row.iloc[0]["acierto_pct"]) if not sede_prueba_row.empty else np.nan
        brecha = sede_pct - red_pct if pd.notna(sede_pct) and pd.notna(red_pct) else np.nan
        c1, c2, c3 = st.columns(3)
        c1.metric("Promedio red en la prueba", f"{red_pct:,.2f}%")
        c2.metric(f"Promedio {sede_focal}", f"{sede_pct:,.2f}%")
        c3.metric("Brecha vs red", f"{brecha:,.2f} p.p.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Promedio red en la prueba", f"{red_pct:,.2f}%")
        c2.metric("Competencias", f"{df_prueba['Competencia'].nunique(dropna=True):,}")
        c3.metric("Registros", f"{len(df_prueba):,}")

    red_comp = agg["red_x_competencia_prueba"]
    red_comp = red_comp[red_comp["Prueba"] == prueba][["Competencia", "acierto_pct", "registros"]]
    red_comp = red_comp.rename(columns={"acierto_pct": "Red", "registros": "registros_red"})

    sede_comp = agg["sede_x_competencia_prueba"]
    sede_comp = sede_comp[sede_comp["Prueba"] == prueba]

    if sede_focal is not None:
        sede_comp_f = sede_comp[sede_comp["Sede"] == sede_focal][["Competencia", "acierto_pct", "registros"]]
        sede_comp_f = sede_comp_f.rename(columns={"acierto_pct": sede_focal, "registros": "registros_sede"})
        comp = red_comp.merge(sede_comp_f, on="Competencia", how="outer")
        comp["brecha_vs_red"] = (comp[sede_focal] - comp["Red"]).round(2)

        plot_df = comp[["Competencia", "Red", sede_focal]].melt(
            id_vars="Competencia", value_vars=["Red", sede_focal],
            var_name="Grupo", value_name="Porcentaje"
        )
        fig = px.bar(plot_df, x="Competencia", y="Porcentaje", color="Grupo", barmode="group", title=f"{prueba}: comparación por competencia")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Comparativo por competencia: {sede_focal} vs red**")
        st.dataframe(comp.sort_values("brecha_vs_red", ascending=False), use_container_width=True, height=340)
    else:
        plot_df = sede_comp[["Competencia", "Sede", "acierto_pct"]].copy()
        fig = px.bar(plot_df, x="Competencia", y="acierto_pct", color="Sede", barmode="group", title=f"{prueba}: desempeño por competencia y sede")
        st.plotly_chart(fig, use_container_width=True)

        pivot_comp = sede_comp.pivot(index="Sede", columns="Competencia", values="acierto_pct")
        st.markdown("**Matriz de competencia por sede**")
        st.dataframe(pivot_comp, use_container_width=True, height=320)

    st.markdown("**Detalle general de la prueba por sede**")
    sede_prueba = agg["sede_x_prueba"]
    sede_prueba = sede_prueba[sede_prueba["Prueba"] == prueba][["Sede", "estudiantes", "registros", "acierto_pct"]].sort_values("acierto_pct", ascending=False)
    st.dataframe(sede_prueba, use_container_width=True, height=240)



def main():
    st.title("Visualizador agregado de resultados académicos")
    st.caption("Carga automática del archivo del repositorio, análisis agregado y comparación de cada sede frente a la red.")

    local_file = load_default_file_if_exists()

    with st.sidebar.expander("Reemplazar archivo fuente", expanded=False):
        uploaded_file = st.file_uploader(
            "Opcional: sube otro archivo (CSV o Excel)",
            type=["csv", "xlsx", "xls"],
            help="Si no subes nada, la app usará el archivo que encuentre en la carpeta data/ del repositorio."
        )

    if local_file is not None:
        raw = read_dataframe_from_path(str(local_file))
        source_label = f"Archivo cargado automáticamente: {local_file.relative_to(BASE_DIR) if local_file.is_absolute() and BASE_DIR in local_file.parents else local_file}"
    elif uploaded_file is not None:
        raw = read_uploaded_file(uploaded_file)
        source_label = f"Archivo cargado manualmente: {uploaded_file.name}"
    else:
        st.error("No encontré el archivo en data/ y tampoco se subió uno manualmente.")
        st.stop()

    if uploaded_file is not None:
        raw = read_uploaded_file(uploaded_file)
        source_label = f"Archivo cargado manualmente: {uploaded_file.name}"

    try:
        df = prepare_data(raw)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.success(source_label)
    st.info("Esta versión no elimina registros ni muestra resultados individuales. Todo el análisis es agregado.")

    df_filtered, sede_focal = sidebar_filters(df)

    if df_filtered.empty:
        st.warning("No hay datos con los filtros seleccionados.")
        st.stop()

    agg = compute_aggregates(df_filtered)
    pruebas = agg["by_prueba"]["Prueba"].dropna().tolist()

    main_tabs = st.tabs(["Resumen ejecutivo", "Sedes y pruebas", "Hojas por prueba"])

    with main_tabs[0]:
        show_resumen_ejecutivo(df_filtered, agg, sede_focal)

    with main_tabs[1]:
        show_detalle_sedes_y_pruebas(df_filtered, agg, sede_focal)

    with main_tabs[2]:
        st.markdown("Cada pestaña muestra la comparación de una prueba entre la red y la sede focal.")
        if not pruebas:
            st.warning("No hay pruebas para mostrar.")
        else:
            prueba_tabs = st.tabs(pruebas)
            for i, prueba in enumerate(pruebas):
                with prueba_tabs[i]:
                    show_prueba_tab(df_filtered, agg, prueba, sede_focal)


if __name__ == "__main__":
    main()
