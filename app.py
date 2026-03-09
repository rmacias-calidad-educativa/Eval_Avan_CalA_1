
from __future__ import annotations

from pathlib import Path
import re
import textwrap
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Evaluación Diagnóstica 2026 Calendario A", layout="wide")

# =========================
# SOCIOEMOCIONAL
# =========================

socio_SOCIO_REQUIRED_COLUMNS = [
    "Sede", "OrgDefinedId", "EdadEst", "Genero", "Grado", "Seccion",
    "Antiguedad", "SurveyName", "TexQuestion", "UsAnswerSurv"
]
socio_COLUMN_ALIASES = {
    "género": "Genero",
    "genero": "Genero",
    "edad estudiante": "Edad Estudiante",
    "id estudiante": "ID Estudiante",
    "questionid": "QuestionId",
    "answerid": "AnswerId",
    "orgdefinedid": "OrgDefinedId",
    "edadest": "EdadEst",
    "texquestion": "TexQuestion",
    "usanswersurv": "UsAnswerSurv",
    "surveyname": "SurveyName",
    "seccion": "Seccion",
    "respuesta correcta": "Respuesta Correcta",
}
socio_BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
socio_DEFAULT_DATA_CANDIDATES = [
    socio_BASE_DIR / "data" / "Auxiliares.xlsx",
    socio_BASE_DIR / "Auxiliares.xlsx",
    Path("data") / "Auxiliares.xlsx",
    Path("Auxiliares.xlsx"),
]
socio_GRADE_ORDER = {
    "tercero": 3, "cuarto": 4, "quinto": 5, "sexto": 6, "septimo": 7, "séptimo": 7,
    "octavo": 8, "noveno": 9, "decimo": 10, "décimo": 10, "undecimo": 11, "undécimo": 11,
    "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11,
}
socio_SOCIO_INDICATOR_ORDER = [
    "Autoconciencia emocional",
    "Empatía",
    "Regulación emocional",
    "Aprendizaje colaborativo",
    "Efectos de cambio",
    "Recursos físicos",
    "Apoyo familiar",
    "Prácticas docentes",
    "Mentalidad de crecimiento",
]
socio_SOCIO_DIMENSION_ORDER = [
    "Habilidades socioemocionales",
    "Factores asociados",
    "Situaciones de cambio",
    "Mentalidad de crecimiento",
]

SOCIO_DISPLAY_MAPS = {
    "emotion": {
        "alegria": "Alegría",
        "rabia": "Rabia",
        "tristeza": "Tristeza",
        "sorpresa": "Sorpresa",
    },
    "yes_no": {"si": "Sí", "no": "No"},
    "three_mucho": {"ningun dia": "Ningún día", "algunos dias": "Algunos días", "muchos dias": "Muchos días"},
    "three_mucho_literal": {"nada": "Nada", "poco": "Poco", "mucho": "Mucho"},
    "si_algunas_no": {"no": "No", "algunas veces": "Algunas veces", "si": "Sí"},
    "agree4": {
        "muy en desacuerdo": "Muy en desacuerdo",
        "en desacuerdo": "En desacuerdo",
        "de acuerdo": "De acuerdo",
        "muy de acuerdo": "Muy de acuerdo",
    },
    "freq4": {
        "nunca": "Nunca",
        "pocas veces": "Pocas veces",
        "muchas veces": "Muchas veces",
        "siempre": "Siempre",
    },
    "satisfaction3": {"insatisfecho": "Insatisfecho", "me da igual": "Me da igual", "satisfecho": "Satisfecho"},
    "learn_compare": {
        "aprendo menos que cuando estaba en casa": "Aprendo menos que cuando estaba en casa",
        "aprendo igual que cuando estaba en casa": "Aprendo igual que cuando estaba en casa",
        "aprendo mas que cuando estaba en casa": "Aprendo más que cuando estaba en casa",
    },
}


def socio_strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def socio_normalize_text_key(value) -> str:
    if pd.isna(value):
        return ""
    raw = socio_strip_accents(str(value)).replace("\xa0", " ")
    raw = re.sub(r"\s+", " ", raw).strip().lower()
    if raw in {"nan", "none"}:
        return ""
    raw = raw.replace("deacuerdo", "de acuerdo")
    return raw


socio_INDICATOR_ORDER_MAP = {x.lower(): i for i, x in enumerate(socio_SOCIO_INDICATOR_ORDER, start=1)}
socio_DIMENSION_ORDER_MAP = {x.lower(): i for i, x in enumerate(socio_SOCIO_DIMENSION_ORDER, start=1)}


def socio_indicator_sort_key(value: str) -> tuple[int, str]:
    raw = str(value).strip()
    return (socio_INDICATOR_ORDER_MAP.get(raw.lower(), 999), raw)


def socio_dimension_sort_key(value: str) -> tuple[int, str]:
    raw = str(value).strip()
    return (socio_DIMENSION_ORDER_MAP.get(raw.lower(), 999), raw)


def socio_clean_sede_label(value: str) -> str:
    raw = str(value).strip()
    cleaned = re.sub(r"^\s*innova\s+", "", raw, flags=re.I)
    return cleaned.title() if cleaned else raw


def socio_normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for col in df.columns:
        c = str(col).strip()
        cols.append(socio_COLUMN_ALIASES.get(c.lower(), c))
    out = df.copy()
    out.columns = cols
    return out


def socio_grade_sort_key(value: str) -> tuple[int, str]:
    raw = str(value).strip()
    if not raw:
        return (999, raw)
    normalized = socio_strip_accents(raw).lower()
    if normalized in socio_GRADE_ORDER:
        return (socio_GRADE_ORDER[normalized], raw)
    m = re.search(r"(\d+)", normalized)
    if m:
        return (int(m.group(1)), raw)
    return (999, raw)


def socio_grade_display_label(value: str) -> str:
    raw = str(value).strip()
    normalized = socio_strip_accents(raw).lower()
    if normalized in socio_GRADE_ORDER:
        return f"{socio_GRADE_ORDER[normalized]}°"
    m = re.search(r"(\d+)", normalized)
    if m:
        return f"{int(m.group(1))}°"
    return raw


def socio_course_sort_key(value: str) -> tuple[int, str]:
    raw = str(value).strip()
    if not raw:
        return (999, raw)
    m = re.search(r"(\d+)", raw)
    num = int(m.group(1)) if m else 999
    return (num, raw)


def socio_wrap_plot_label(value: str, width: int = 34, max_lines: int = 4) -> str:
    raw = str(value).strip().replace("\n", " ")
    raw = re.sub(r"\s+", " ", raw)
    if not raw:
        return ""
    lines = textwrap.wrap(raw, width=width, break_long_words=False, break_on_hyphens=False)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .,;:") + "…"
    return "<br>".join(lines)


def socio_load_default_file_if_exists() -> Path | None:
    for path in socio_DEFAULT_DATA_CANDIDATES:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def socio_read_dataframe_from_path(path_str: str, file_version: int) -> pd.DataFrame:
    path = Path(path_str)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(path, sep=None, engine="python")
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError("Formato no soportado. Usa CSV o Excel.")


def socio_validate_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in socio_SOCIO_REQUIRED_COLUMNS if c not in df.columns]


def socio_build_socio_question_specs() -> list[dict]:
    specs: list[dict] = []

    def add(indicator: str, dimension: str, scale: str, questions: list[str], reverse: list[str] | None = None, survey_scope: list[str] | None = None) -> None:
        reverse_keys = {socio_normalize_text_key(x) for x in reverse or []}
        scope = {str(x) for x in survey_scope or []}
        for q in questions:
            q_key = socio_normalize_text_key(q)
            specs.append({
                "question_key": q_key,
                "indicator": indicator,
                "dimension": dimension,
                "scale": scale,
                "reverse": q_key in reverse_keys,
                "survey_scope": scope,
            })

    add("Autoconciencia emocional", "Habilidades socioemocionales", "emotion", ["Alegria", "Rabia", "Tristeza", "Sorpresa"], survey_scope=["Auxiliar Educación básica primaria"])
    add("Empatía", "Habilidades socioemocionales", "yes_no", [
        "Me afecta cuando veo que alguien molesta a un amigo o una amiga.",
        "Me enojo cuando veo que tratan mal a otra persona.",
        "Me preocupa cuando alguien se siente mal.",
        "Me siento triste cuando veo que una persona se siente triste.",
    ])
    add("Regulación emocional", "Habilidades socioemocionales", "yes_no", [
        "Les digo a mis padres lo que siento en ese momento.",
        "Me encierro en mi cuarto.",
        "Prefiero NO hablar con mis padres.",
        "Respiro profundamente para calmarme.",
    ], reverse=["Me encierro en mi cuarto.", "Prefiero NO hablar con mis padres."])
    # Autoeficacia se elimina del tablero por solicitud del usuario.
    add("Aprendizaje colaborativo", "Factores asociados", "three_mucho_literal", [
        "Respeto las ideas de todos mis compañeros cuando trabajo en grupo.",
        "Aprendo más cuando trabajo con otros compañeros.",
        "Mis compañeros me pueden ayudar si es necesario.",
    ])
    add("Efectos de cambio", "Situaciones de cambio", "yes_no", [
        "Me siento solo.",
        "Disfruto aprendiendo con mis profesores y compañeros.",
        "Volví a jugar y hacer deportes.",
        "Tengo otras formas de hablar con mis amigos.",
    ], reverse=["Me siento solo."], survey_scope=["Auxiliar Educación básica primaria"])
    add("Efectos de cambio", "Situaciones de cambio", "agree4", [
        "Me siento solo.",
        "Disfruto aprendiendo con mis profesores y compañeros.",
        "Disfruto aprendiendo en mi colegio.",
        "Tengo otras formas de interactuar con mis amigos.",
    ], reverse=["Me siento solo."], survey_scope=["Auxiliar educación básica secundaria", "Auxiliar educación media"])
    add("Efectos de cambio", "Situaciones de cambio", "learn_compare", ["Desde que regresé a mi colegio:"])
    add("Efectos de cambio", "Situaciones de cambio", "satisfaction3", [
        "Las instrucciones de las actividades proporcionadas por mis docentes",
        "Desde que regresé a mi colegio. estoy satisfecho con:Las instrucciones de las actividades proporcionadas por mis docentes",
        "El apoyo que recibo de mis docentes.",
        "Los recursos para el aprendizaje proporcionados por mi colegio. tales como libros. cartillas. guías. entre otros.",
        "La cantidad de tareas que me envían para la casa.",
    ])
    add("Recursos físicos", "Situaciones de cambio", "si_algunas_no", [
        "Puedo usar internet.",
        "Mi internet funciona para completar mis tareas escolares",
        "Puedo usar un dispositivo como celular. tableta o computador cuando lo necesito",
        "Tengo un espacio tranquilo para estudiar.",
        "Puedo usar material impreso como libros. guías o cartillas para completar mis tareas escolares.",
        "Tengo los útiles escolares necesarios como cuadernos o lápices",
        "Donde vivo: Tengo que compartir un dispositivo como celular. tableta o computador para hacer mis tareas",
    ], reverse=["Donde vivo: Tengo que compartir un dispositivo como celular. tableta o computador para hacer mis tareas"], survey_scope=["Auxiliar Educación básica primaria"])
    add("Recursos físicos", "Situaciones de cambio", "freq4", [
        "Tengo acceso a internet.",
        "Mi internet es adecuado para completar mis tareas escolares.",
        "Tengo acceso cuando lo necesito a un dispositivo como celular. tableta o computador.",
        "Tengo un espacio tranquilo para estudiar.",
        "Tengo que compartir un dispositivo como celular. tableta o computador para hacer mis tareas.",
        "Tengo acceso a material impreso como libros. guías o cartillas para completar mis tareas escolares.",
        "Tengo los útiles escolares necesarios como cuadremos o lápices.",
        "Tengo que estudiar y también cuidar a una persona que vive en mi casa",
    ], reverse=[
        "Tengo que compartir un dispositivo como celular. tableta o computador para hacer mis tareas.",
        "Tengo que estudiar y también cuidar a una persona que vive en mi casa",
    ], survey_scope=["Auxiliar educación básica secundaria", "Auxiliar educación media"])
    add("Apoyo familiar", "Situaciones de cambio", "three_mucho", [
        "Me ayuda con las tareas escolares.",
        "Me pregunta los temas que estoy estudiando.",
        "Me revisa que haga las tareas escolares.",
        "Me explica temas difíciles.",
        "Me enseña de diferentes maneras.",
    ], survey_scope=["Auxiliar Educación básica primaria"])
    add("Apoyo familiar", "Situaciones de cambio", "freq4", [
        "Me ayuda con tareas escolares.",
        "Me pregunta qué estoy aprendiendo.",
        "Se asegura que avance en mis tareas escolares",
        "Me explica temas difíciles.",
        "Me ayuda a encontrar herramientas adicionales para el aprendizaje",
    ], survey_scope=["Auxiliar educación básica secundaria"])
    add("Prácticas docentes", "Situaciones de cambio", "three_mucho", [
        "Desde que regresé a mi colegio. cuántos días mis docentes:Me envían tareas.",
        "Desde que regresé a mi colegio. cuántos días mis docentes:Me ayudan con mis tareas cuando lo necesito.",
        "Desde que regresé a mi colegio. cuántos días mis docentes:Me envían actividades y evaluaciones por internet",
    ], survey_scope=["Auxiliar Educación básica primaria"])
    add("Prácticas docentes", "Situaciones de cambio", "freq4", [
        "Mis docentes me envían tareas.",
        "Mis docentes están disponibles cuando necesito ayuda. por ejemplo. a través de horas de atención. teléfono. correo electrónico. chat. entre otros.",
        "Mis docentes me envían actividades y evaluaciones por internet.",
        "Mis docentes me envían actividades y evaluaciones para internet.",
        "Mis docentes me brindan consejos útiles sobre cómo aprender más efectivamente por mi cuenta.",
    ], survey_scope=["Auxiliar educación básica secundaria", "Auxiliar educación media"])
    add("Mentalidad de crecimiento", "Mentalidad de crecimiento", "agree4", [
        "Puedo aprender por mi cuenta gracias a lo que me han enseñado mis docentes.",
        "Creo en mi talento e inteligencia aunque me equivoque o se me dificulte entender.",
        "Creo en mi talento e inteligencia. aunque me equivoque o se me dificulte entender.",
        "Puedo mejorar mi rendimiento académico con mi esfuerzo y la ayuda de mis docentes.",
        "Mis docentes clasifican y separan a los que más y menos entienden de un tema.",
        "Pienso que algunos estudiantes son más inteligentes que otros por las notas que obtienen.",
        "Siento que mis docentes nos dan a todos las mismas oportunidades para aprender y tener buenas notas.",
        "Creo que es más difícil aprender si me equivoco en las actividades de clase.",
        "Mis docentes me han enseñado que cuando me equivoco tengo la oportunidad de aprender.",
        "Confío en que si me equivoco puedo probar diferentes opciones para aprender.",
    ], reverse=[
        "Mis docentes clasifican y separan a los que más y menos entienden de un tema.",
        "Pienso que algunos estudiantes son más inteligentes que otros por las notas que obtienen.",
        "Creo que es más difícil aprender si me equivoco en las actividades de clase.",
    ])
    return specs


socio_SOCIO_QUESTION_SPECS = socio_build_socio_question_specs()


def socio_get_socio_question_spec(question: str, survey_name: str) -> dict | None:
    q_key = socio_normalize_text_key(question)
    survey_name = str(survey_name)
    for spec in socio_SOCIO_QUESTION_SPECS:
        if spec["question_key"] != q_key:
            continue
        if not spec["survey_scope"] or survey_name in spec["survey_scope"]:
            return spec
    return None


def socio_ordinal_score(scale: str, answer_key: str, reverse: bool, expected_key: str = "") -> float:
    if not answer_key:
        return np.nan
    if scale == "emotion":
        return 100.0 if answer_key == expected_key else 0.0
    if scale == "yes_no":
        mapping = {"no": 0.0, "si": 100.0}
    elif scale == "three_mucho":
        mapping = {"ningun dia": 0.0, "algunos dias": 50.0, "muchos dias": 100.0}
    elif scale == "three_mucho_literal":
        mapping = {"nada": 0.0, "poco": 50.0, "mucho": 100.0}
    elif scale == "si_algunas_no":
        mapping = {"no": 0.0, "algunas veces": 50.0, "si": 100.0}
    elif scale == "agree4":
        mapping = {"muy en desacuerdo": 0.0, "en desacuerdo": 33.0, "de acuerdo": 67.0, "muy de acuerdo": 100.0}
    elif scale == "freq4":
        mapping = {"nunca": 0.0, "pocas veces": 33.0, "muchas veces": 67.0, "siempre": 100.0}
    elif scale == "satisfaction3":
        mapping = {"insatisfecho": 0.0, "me da igual": 50.0, "satisfecho": 100.0}
    elif scale == "learn_compare":
        mapping = {
            "aprendo menos que cuando estaba en casa": 0.0,
            "aprendo igual que cuando estaba en casa": 50.0,
            "aprendo mas que cuando estaba en casa": 100.0,
        }
    else:
        return np.nan

    value = mapping.get(answer_key, np.nan)
    if pd.isna(value):
        return np.nan
    return 100.0 - float(value) if reverse else float(value)


def socio_favorable_topbox(scale: str, answer_key: str, reverse: bool, expected_key: str = "") -> float:
    if not answer_key:
        return np.nan
    if scale == "emotion":
        return 1.0 if answer_key == expected_key else 0.0
    if scale == "agree4":
        positive = {"de acuerdo", "muy de acuerdo"}
        if reverse:
            positive = {"en desacuerdo", "muy en desacuerdo"}
    elif scale == "freq4":
        positive = {"muchas veces", "siempre"}
        if reverse:
            positive = {"nunca", "pocas veces"}
    elif scale == "yes_no":
        positive = {"si"}
        if reverse:
            positive = {"no"}
    elif scale == "three_mucho":
        positive = {"muchos dias"}
        if reverse:
            positive = {"ningun dia"}
    elif scale == "three_mucho_literal":
        positive = {"mucho"}
        if reverse:
            positive = {"nada"}
    elif scale == "si_algunas_no":
        positive = {"si"}
        if reverse:
            positive = {"no"}
    elif scale == "satisfaction3":
        positive = {"satisfecho"}
        if reverse:
            positive = {"insatisfecho"}
    elif scale == "learn_compare":
        positive = {"aprendo mas que cuando estaba en casa"}
        if reverse:
            positive = {"aprendo menos que cuando estaba en casa"}
    else:
        return np.nan
    return 1.0 if answer_key in positive else 0.0


def socio_report_answer_label(scale: str, answer_key: str, raw_value) -> str | float:
    if not answer_key or answer_key in {"0", "0.0"}:
        return np.nan
    mapping = SOCIO_DISPLAY_MAPS.get(scale, {})
    if mapping:
        return mapping.get(answer_key, np.nan)
    raw = str(raw_value).strip()
    if raw in {"", "0", "0.0", "nan", "None"}:
        return np.nan
    return raw


@st.cache_data(show_spinner=False)
def socio_prepare_socio_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = socio_normalize_columns(raw)

    for col in ["EdadEst", "Antiguedad"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = ["Sede", "OrgDefinedId", "Genero", "Grado", "Seccion", "SurveyName", "TexQuestion", "UsAnswerSurv"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["", "nan", "None"]), col] = np.nan

    df["Sede Corta"] = df["Sede"].map(socio_clean_sede_label)
    df["Grado Orden"] = df["Grado"].map(lambda x: socio_grade_sort_key(x)[0])
    df["Grado Etiqueta"] = df["Grado"].map(socio_grade_display_label)
    df["Curso Orden"] = df["Seccion"].map(lambda x: socio_course_sort_key(x)[0])
    df["Question Key"] = df["TexQuestion"].map(socio_normalize_text_key)
    df["Answer Key"] = df["UsAnswerSurv"].map(socio_normalize_text_key)

    specs = df.apply(lambda r: socio_get_socio_question_spec(r["TexQuestion"], r["SurveyName"]), axis=1)
    df["Indicador"] = specs.map(lambda x: x["indicator"] if x else np.nan)
    df["Dimension"] = specs.map(lambda x: x["dimension"] if x else np.nan)
    df["Escala"] = specs.map(lambda x: x["scale"] if x else np.nan)
    df["Inversa"] = specs.map(lambda x: bool(x["reverse"]) if x else False)
    df["Respuesta Válida"] = df.apply(lambda r: bool(socio_get_socio_question_spec(r["TexQuestion"], r["SurveyName"])) and bool(r["Answer Key"]), axis=1)

    def _score_row(row: pd.Series) -> float:
        spec = socio_get_socio_question_spec(row["TexQuestion"], row["SurveyName"])
        if not spec:
            return np.nan
        return socio_ordinal_score(
            scale=spec["scale"],
            answer_key=row["Answer Key"],
            reverse=bool(spec["reverse"]),
            expected_key=row["Question Key"],
        )

    def _fav_row(row: pd.Series) -> float:
        spec = socio_get_socio_question_spec(row["TexQuestion"], row["SurveyName"])
        if not spec:
            return np.nan
        return socio_favorable_topbox(
            scale=spec["scale"],
            answer_key=row["Answer Key"],
            reverse=bool(spec["reverse"]),
            expected_key=row["Question Key"],
        )

    def _display_row(row: pd.Series):
        spec = socio_get_socio_question_spec(row["TexQuestion"], row["SurveyName"])
        if not spec:
            return np.nan
        return socio_report_answer_label(spec["scale"], row["Answer Key"], row["UsAnswerSurv"])

    df["Puntaje Socio"] = df.apply(_score_row, axis=1)
    df["Respuesta Favorable"] = df.apply(_fav_row, axis=1)
    df["Respuesta Reporte"] = df.apply(_display_row, axis=1)
    return df


def socio_safe_pct(series: pd.Series) -> float:
    clean = pd.Series(series).dropna().astype(float)
    return float(clean.mean()) if len(clean) else 0.0


def socio_safe_nunique(series: pd.Series) -> int:
    clean = pd.Series(series).dropna()
    return int(clean.nunique()) if len(clean) else 0


def socio_socio_benchmark(df: pd.DataFrame, focus_df: pd.DataFrame, dim: str) -> pd.DataFrame:
    net = (
        df.groupby(dim, dropna=False)
        .agg(
            puntaje_red=("Puntaje Socio", "mean"),
            favorable_red=("Respuesta Favorable", "mean"),
            cobertura_red=("Respuesta Válida", "mean"),
            estudiantes_red=("OrgDefinedId", pd.Series.nunique),
        )
        .reset_index()
    )
    focus = (
        focus_df.groupby(dim, dropna=False)
        .agg(
            puntaje_sede=("Puntaje Socio", "mean"),
            favorable_sede=("Respuesta Favorable", "mean"),
            cobertura_sede=("Respuesta Válida", "mean"),
            estudiantes_sede=("OrgDefinedId", pd.Series.nunique),
        )
        .reset_index()
    )
    out = net.merge(focus, on=dim, how="left")
    for col in ["favorable_red", "favorable_sede", "cobertura_red", "cobertura_sede"]:
        out[col] = (out[col] * 100).round(2)
    out["puntaje_red"] = out["puntaje_red"].round(2)
    out["puntaje_sede"] = out["puntaje_sede"].round(2)
    out["brecha_puntaje"] = (out["puntaje_sede"] - out["puntaje_red"]).round(2)
    return out


def socio_sort_socio_benchmark(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if dim == "Indicador":
        out["__orden__"] = out[dim].map(lambda x: socio_indicator_sort_key(x)[0])
        out = out.sort_values(["__orden__", dim]).drop(columns="__orden__")
    elif dim == "Dimension":
        out["__orden__"] = out[dim].map(lambda x: socio_dimension_sort_key(x)[0])
        out = out.sort_values(["__orden__", dim]).drop(columns="__orden__")
    elif dim == "Grado":
        out["__orden__"] = out[dim].map(lambda x: socio_grade_sort_key(x)[0])
        out = out.sort_values(["__orden__", dim]).drop(columns="__orden__")
    else:
        out = out.sort_values(dim)
    return out


def socio_question_summary(df: pd.DataFrame, focus_df: pd.DataFrame) -> pd.DataFrame:
    red = (
        df.groupby(["SurveyName", "Indicador", "TexQuestion"], dropna=False)
        .agg(
            puntaje_red=("Puntaje Socio", "mean"),
            favorable_red=("Respuesta Favorable", "mean"),
            cobertura_red=("Respuesta Válida", "mean"),
            estudiantes_red=("OrgDefinedId", pd.Series.nunique),
        )
        .reset_index()
    )
    focus = (
        focus_df.groupby(["SurveyName", "Indicador", "TexQuestion"], dropna=False)
        .agg(
            puntaje_sede=("Puntaje Socio", "mean"),
            favorable_sede=("Respuesta Favorable", "mean"),
            cobertura_sede=("Respuesta Válida", "mean"),
            estudiantes_sede=("OrgDefinedId", pd.Series.nunique),
        )
        .reset_index()
    )
    out = red.merge(focus, on=["SurveyName", "Indicador", "TexQuestion"], how="left")
    for col in ["favorable_red", "favorable_sede", "cobertura_red", "cobertura_sede"]:
        out[col] = (out[col] * 100).round(2)
    for col in ["puntaje_red", "puntaje_sede"]:
        out[col] = out[col].round(2)
    out["brecha_puntaje"] = (out["puntaje_sede"] - out["puntaje_red"]).round(2)
    return out


def socio_response_profile(frame: pd.DataFrame, question: str) -> pd.DataFrame:
    sub = frame[(frame["TexQuestion"] == question) & (frame["Respuesta Reporte"].notna())].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Respuesta Reporte", "pct"])
    out = sub.groupby("Respuesta Reporte", dropna=False).size().reset_index(name="n")
    total = out["n"].sum()
    out["pct"] = np.where(total > 0, out["n"] / total * 100, 0)
    return out.sort_values("pct", ascending=False)


def socio_yes_no_summary(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str, indicator: str) -> pd.DataFrame:
    base = df[(df["Indicador"] == indicator) & (df["Escala"] == "yes_no") & (df["Respuesta Reporte"].isin(["Sí", "No"]))].copy()
    focus = focus_df[(focus_df["Indicador"] == indicator) & (focus_df["Escala"] == "yes_no") & (focus_df["Respuesta Reporte"].isin(["Sí", "No"]))].copy()
    other = base[base["Sede Corta"] != focus_label].copy()

    focus_counts = (
        focus.groupby(["TexQuestion", "Respuesta Reporte"], dropna=False)
        .size()
        .reset_index(name="Conteo sede")
    )
    other_counts = (
        other.groupby(["TexQuestion", "Respuesta Reporte"], dropna=False)
        .size()
        .reset_index(name="Conteo demás sedes")
    )

    out = focus_counts.merge(other_counts, on=["TexQuestion", "Respuesta Reporte"], how="outer").fillna(0)
    if out.empty:
        return out
    out["Conteo sede"] = out["Conteo sede"].astype(int)
    out["Conteo demás sedes"] = out["Conteo demás sedes"].astype(int)
    out["Pregunta"] = out["TexQuestion"].map(lambda x: shorten_text(x, 120))
    return out[["Pregunta", "Respuesta Reporte", "Conteo sede", "Conteo demás sedes"]].sort_values(["Pregunta", "Respuesta Reporte"])


def socio_emotion_summary(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str) -> pd.DataFrame:
    emotion_map = SOCIO_DISPLAY_MAPS["emotion"]
    base = df[(df["Indicador"] == "Autoconciencia emocional") & (df["Escala"] == "emotion")].copy()
    focus = focus_df[(focus_df["Indicador"] == "Autoconciencia emocional") & (focus_df["Escala"] == "emotion")].copy()
    other = base[base["Sede Corta"] != focus_label].copy()

    def _aggregate(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["Emoción", f"Aciertos {prefix}", f"Confusiones {prefix}", f"% acierto {prefix}"])
        tmp = frame.copy()
        tmp["Emoción"] = tmp["Question Key"].map(lambda x: emotion_map.get(x, str(x).title()))
        tmp["Es acierto"] = tmp["Answer Key"] == tmp["Question Key"]
        tmp = tmp[tmp["Answer Key"].isin(set(emotion_map.keys()))].copy()
        out = (
            tmp.groupby("Emoción", dropna=False)
            .agg(
                total=("Es acierto", "size"),
                aciertos=("Es acierto", "sum"),
            )
            .reset_index()
        )
        out[f"Aciertos {prefix}"] = out["aciertos"].astype(int)
        out[f"Confusiones {prefix}"] = (out["total"] - out["aciertos"]).astype(int)
        out[f"% acierto {prefix}"] = np.where(out["total"] > 0, out["aciertos"] / out["total"] * 100, 0).round(2)
        return out[["Emoción", f"Aciertos {prefix}", f"Confusiones {prefix}", f"% acierto {prefix}"]]

    focus_out = _aggregate(focus, "sede")
    other_out = _aggregate(other, "demás sedes")
    out = focus_out.merge(other_out, on="Emoción", how="outer").fillna(0)
    if out.empty:
        return out
    numeric_cols = [c for c in out.columns if c != "Emoción"]
    for col in numeric_cols:
        if "% acierto" not in col:
            out[col] = out[col].astype(int)
    order = ["Alegría", "Rabia", "Tristeza", "Sorpresa"]
    out["__orden__"] = out["Emoción"].map(lambda x: order.index(x) if x in order else 999)
    return out.sort_values(["__orden__", "Emoción"]).drop(columns="__orden__")


def socio_emotion_confusion_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    emotion_map = SOCIO_DISPLAY_MAPS["emotion"]
    sub = frame[(frame["Indicador"] == "Autoconciencia emocional") & (frame["Escala"] == "emotion")].copy()
    sub = sub[sub["Question Key"].isin(set(emotion_map.keys())) & sub["Answer Key"].isin(set(emotion_map.keys()))].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["Emoción objetivo"] = sub["Question Key"].map(emotion_map)
    sub["Respuesta elegida"] = sub["Answer Key"].map(emotion_map)
    out = (
        sub.groupby(["Emoción objetivo", "Respuesta elegida"], dropna=False)
        .size()
        .reset_index(name="Conteo")
        .pivot(index="Emoción objetivo", columns="Respuesta elegida", values="Conteo")
        .fillna(0)
    )
    return out.astype(int)


# =========================
# ACADÉMICO
# =========================

REQUIRED_COLUMNS = [
    "Sede", "ID Estudiante", "Edad Estudiante", "Genero", "Grado", "Curso",
    "Antiguedad", "Prueba", "Acierto", "QuestionId", "Pregunta",
    "AnswerId", "Respuesta", "Competencia",
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
    "tercero": 3, "cuarto": 4, "quinto": 5, "sexto": 6, "septimo": 7, "séptimo": 7,
    "octavo": 8, "noveno": 9, "decimo": 10, "décimo": 10, "undecimo": 11, "undécimo": 11,
    "primero": 101, "segundo": 102, "transicion": 103, "transición": 103,
    "jardin": 104, "jardín": 104, "prejardin": 105, "pre jardín": 105, "prejardín": 105,
}
COMPETENCY_ORDER = [
    "Conocimientos", "Pensamiento Social", "Argumentación en contextos ciudadanos",
    "Multiperspectivismo", "pensamiento sistémico", "Pensamiento Reflexivo y Sistémico",
    "Interpretación y Análisis de Perspectivas", "Explicación de fenómenos", "Indagación",
    "Conocimiento científico", "Interpretación y representación", "Formulación y ejecución",
    "Razonamiento", "Comunicación", "Resolución de problemas", "Argumentación",
]
P_BIS_TARGET = 0.20
D27_TARGET = 0.20
ENGLISH_LEVEL_ORDER = ["Pre A1", "A2", "A1", "B1"]
ENGLISH_GROWTH_FACTORS = {"Pre A1": 1.0, "A2": 2.0, "A1": 3.0, "B1": 4.0}
ENGLISH_LEVEL_FROM_NUM = {1: "Pre A1", 2: "A2", 3: "A1", 4: "B1"}



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
    out = df.copy()
    out.columns = cols
    return out


def grade_sort_key(value: str) -> tuple[int, str]:
    raw = str(value).strip()
    if not raw:
        return (999, raw)
    normalized = strip_accents(raw).lower()
    if normalized in GRADE_ORDER:
        return (GRADE_ORDER[normalized], raw)
    m = re.search(r"(\d+)", normalized)
    if m:
        return (int(m.group(1)), raw)
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
    m = re.search(r"(\d+)", normalized)
    if m:
        return f"{int(m.group(1))}°"
    return raw


def shorten_text(value: str, max_len: int = 90) -> str:
    raw = str(value).strip().replace("\n", " ")
    raw = re.sub(r"\s+", " ", raw)
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 1].rstrip() + "…"


def wrap_plot_label(value: str, width: int = 42, max_lines: int = 4) -> str:
    raw = str(value).strip().replace("\n", " ")
    raw = re.sub(r"\s+", " ", raw)
    if not raw:
        return ""
    lines = textwrap.wrap(raw, width=width, break_long_words=False, break_on_hyphens=False)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .,;:") + "…"
    return "<br>".join(lines)


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


def is_english_prueba(value: str) -> bool:
    key = strip_accents(str(value)).lower()
    return ("ingles" in key) or ("english" in key)


def extract_english_level(*values) -> str | float:
    """Extrae el nivel CEFR (Pre A1, A2, A1, B1) desde cualquier texto disponible.

    Nota: en muchos exportes el nivel viene escrito en la columna *Competencia*,
    pero aquí solo lo usamos para **identificar el nivel del ítem**, no para mostrar
    competencias en el tablero.
    """
    text = " ".join(strip_accents(str(v)).lower() for v in values if pd.notna(v))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return np.nan

    # Variantes frecuentes: "Pre A1", "Pre-A1", "PreA1", "Pre_A1"
    if re.search(r"\bpre\s*[-_]?\s*a\s*1\b", text) or re.search(r"\bprea1\b", text):
        return "Pre A1"

    # B1 / A2 / A1 con separadores opcionales (p.ej. "A 2", "B-1")
    if re.search(r"\bb\s*[-_]?\s*1\b", text):
        return "B1"
    if re.search(r"\ba\s*[-_]?\s*2\b", text):
        return "A2"
    if re.search(r"\ba\s*[-_]?\s*1\b", text):
        return "A1"

    return np.nan

def load_default_file_if_exists() -> Path | None:
    for path in DEFAULT_DATA_CANDIDATES:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def read_dataframe_from_path(path_str: str, file_version: int) -> pd.DataFrame:
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
    df["Nivel Inglés"] = df.apply(
        lambda r: extract_english_level(r.get("Competencia"), r.get("Pregunta"), r.get("Prueba"), r.get("Prueba Base")) if is_english_prueba(r.get("Prueba Base")) else np.nan,
        axis=1,
    )
    return df


def add_benchmark(df: pd.DataFrame, focus_df: pd.DataFrame, dim: str) -> pd.DataFrame:
    net = (
        df.groupby(dim, dropna=False)
        .agg(
            acierto_red=("Acierto", "mean"),
            estudiantes_red=("ID Estudiante", "nunique"),
            respuestas_red=("Acierto", "size"),
        )
        .reset_index()
    )
    focus = (
        focus_df.groupby(dim, dropna=False)
        .agg(
            acierto_sede=("Acierto", "mean"),
            estudiantes_sede=("ID Estudiante", "nunique"),
            respuestas_sede=("Acierto", "size"),
        )
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
        "acierto_sede_pct": "% de rendimiento en la sede",
        "acierto_red_pct": "% de rendimiento en Colombia",
        "brecha_pp": "Brecha frente a Colombia (pp)",
    })
    return out[["Competencia", "% de rendimiento en la sede", "% de rendimiento en Colombia", "Brecha frente a Colombia (pp)"]]


def safe_pct(series: pd.Series) -> float:
    return float(series.mean() * 100) if len(series) else 0.0


def safe_nunique(series: pd.Series) -> int:
    return int(series.nunique()) if len(series) else 0


def get_theme_tokens() -> dict:
    base = (st.get_option("theme.base") or "light").lower()
    primary = st.get_option("theme.primaryColor") or ("#60a5fa" if base == "dark" else "#2563eb")
    text = st.get_option("theme.textColor") or ("#f8fafc" if base == "dark" else "#0f172a")
    background = st.get_option("theme.backgroundColor") or ("#0e1117" if base == "dark" else "#ffffff")
    secondary_bg = st.get_option("theme.secondaryBackgroundColor") or ("#1f2937" if base == "dark" else "#f8fafc")
    return {
        "base": base,
        "primary": primary,
        "secondary": "#f59e0b" if base == "dark" else "#d97706",
        "text": text,
        "background": background,
        "secondary_bg": secondary_bg,
        "muted": "#94a3b8" if base == "dark" else "#64748b",
        "grid": "rgba(148, 163, 184, 0.28)" if base == "dark" else "rgba(51, 65, 85, 0.18)",
        "success": "#22c55e" if base == "dark" else "#15803d",
        "warning": "#f59e0b" if base == "dark" else "#b45309",
        "danger": "#f87171" if base == "dark" else "#dc2626",
        "plotly_template": "plotly_dark" if base == "dark" else "plotly_white",
        "card_bg": "rgba(30, 41, 59, 0.32)" if base == "dark" else "rgba(248, 250, 252, 0.96)",
        "card_border": "rgba(148, 163, 184, 0.22)" if base == "dark" else "rgba(100, 116, 139, 0.22)",
        "heat_scale": [
            [0.0, "#7f1d1d" if base == "dark" else "#fee2e2"],
            [0.5, "#d97706" if base == "dark" else "#fde68a"],
            [1.0, "#16a34a" if base == "dark" else "#15803d"],
        ],
        "gauge_steps": (
            [
                {"range": [0, 40], "color": "#3f1d1d"},
                {"range": [40, 60], "color": "#4a3a11"},
                {"range": [60, 80], "color": "#183225"},
                {"range": [80, 100], "color": "#17324d"},
            ]
            if base == "dark"
            else [
                {"range": [0, 40], "color": "#fee2e2"},
                {"range": [40, 60], "color": "#fef3c7"},
                {"range": [60, 80], "color": "#dcfce7"},
                {"range": [80, 100], "color": "#dbeafe"},
            ]
        ),
    }


def apply_accessible_figure_style(fig: go.Figure, theme: dict) -> go.Figure:
    fig.update_layout(
        template=theme["plotly_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme["text"]),
        legend=dict(font=dict(color=theme["text"])),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color=theme["text"]), title_font=dict(color=theme["text"]), linecolor=theme["grid"])
    fig.update_yaxes(gridcolor=theme["grid"], zerolinecolor=theme["grid"], tickfont=dict(color=theme["text"]), title_font=dict(color=theme["text"]))
    return fig


def render_theme_metric_card(title: str, value: str, subtitle: str, theme: dict) -> None:
    st.markdown(
        f"""
        <div style="
            background:{theme['card_bg']};
            border:1px solid {theme['card_border']};
            border-radius:14px;
            padding:1rem 1.05rem;
            min-height:118px;
        ">
            <div style="font-size:0.95rem; color:{theme['muted']}; margin-bottom:0.35rem;">{title}</div>
            <div style="font-size:2rem; font-weight:700; color:{theme['text']}; line-height:1.1;">{value}</div>
            <div style="font-size:0.9rem; color:{theme['muted']}; margin-top:0.35rem;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    try:
        pb = merged.groupby("QuestionId", dropna=False).apply(pbis, include_groups=False).reset_index(name="p_bis")
    except TypeError:
        pb = merged.groupby("QuestionId", dropna=False).apply(pbis).reset_index(name="p_bis")

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

    def item_flag(row: pd.Series) -> str:
        p = row["dificultad"]
        pbv = row["p_bis"]
        d27v = row["d27"]
        if pd.isna(p):
            return "Revisar"
        if ((pd.notna(pbv) and pbv < 0) or (pd.notna(d27v) and d27v < 0)):
            return "Alerta psicométrica"
        if p < 0.20 and ((pd.isna(pbv) or pbv < 0.15) and (pd.isna(d27v) or d27v < 0.15)):
            return "Posible ítem muy complejo o ambiguo"
        if p > 0.90 and ((pd.isna(pbv) or pbv < 0.10) and (pd.isna(d27v) or d27v < 0.10)):
            return "Posible ítem demasiado fácil"
        if pd.notna(pbv) and pbv >= 0.30 and 0.30 <= p <= 0.80:
            return "Buen ítem"
        return "Revisar"

    out["estado_item"] = out.apply(item_flag, axis=1)
    return out.sort_values(["Prueba Base", "Grado Orden", "dificultad", "p_bis"], ascending=[True, True, True, False])


def make_compact_gauge(value: float) -> go.Figure:
    theme = get_theme_tokens()
    gauge_value = float(np.clip(value, 0, 100))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            number={"suffix": "%", "font": {"size": 20, "color": theme["text"]}},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100], "tickfont": {"size": 10, "color": theme["muted"]}},
                "bar": {"color": theme["success"], "thickness": 0.30},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": theme["gauge_steps"],
            },
        )
    )
    fig.update_layout(height=165, margin=dict(l=6, r=6, t=6, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(color=theme["text"]))
    return fig


def render_kpi_block(title: str, main_text: str, gauge_value: float, chip_text: str | None = None, chip_positive: bool = True) -> None:
    theme = get_theme_tokens()
    chip_html = ""
    if chip_text:
        chip_bg = "rgba(22, 163, 74, 0.18)" if chip_positive else "rgba(220, 38, 38, 0.18)"
        chip_color = theme["success"] if chip_positive else theme["danger"]
        chip_html = f"""
        <div style="display:inline-block; margin-top:0.35rem; padding:0.22rem 0.55rem; border-radius:999px; background:{chip_bg}; color:{chip_color}; font-size:0.95rem; font-weight:600;">
            {chip_text}
        </div>
        """
    st.markdown(
        f"""
        <div style="
            background:{theme['card_bg']};
            border:1px solid {theme['card_border']};
            border-radius:14px;
            padding:0.9rem 1rem;
            min-height:118px;
        ">
            <div style="font-size:1.05rem; font-weight:600; color:{theme['muted']}; line-height:1.2; margin-bottom:0.35rem;">{title}</div>
            <div style="font-size:2.1rem; font-weight:700; color:{theme['text']}; line-height:1.05; word-break:break-word;">{main_text}</div>
            {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(make_compact_gauge(gauge_value), use_container_width=True, config={"displayModeBar": False})


def benchmark_cards(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str) -> None:
    colombia_pct = safe_pct(df["Acierto"])
    focus_pct = safe_pct(focus_df["Acierto"])
    brecha = focus_pct - colombia_pct

    by_prueba_desc = (
        df.groupby("Prueba Base", dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )
    prueba_destacada = by_prueba_desc.iloc[0]["Prueba Base"] if not by_prueba_desc.empty else "Sin dato"
    prueba_destacada_pct = float(by_prueba_desc.iloc[0]["Acierto"]) if not by_prueba_desc.empty else 0.0

    by_prueba_critica = by_prueba_desc.sort_values("Acierto", ascending=True).reset_index(drop=True)
    prueba_critica = by_prueba_critica.iloc[0]["Prueba Base"] if not by_prueba_critica.empty else "Sin dato"
    prueba_critica_pct = float(by_prueba_critica.iloc[0]["Acierto"]) if not by_prueba_critica.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_block("Promedio Colombia", f"{colombia_pct:.1f}%", colombia_pct)
    with c2:
        render_kpi_block(f"Promedio {focus_label}", f"{focus_pct:.1f}%", focus_pct, chip_text=f"{brecha:+.2f} pp", chip_positive=brecha >= 0)
    with c3:
        render_kpi_block("Prueba con mayor rendimiento", str(prueba_destacada), prueba_destacada_pct, chip_text=f"{prueba_destacada_pct:.1f}%", chip_positive=True)
    with c4:
        render_kpi_block("Prueba con menor rendimiento", str(prueba_critica), prueba_critica_pct, chip_text=f"{prueba_critica_pct:.1f}%", chip_positive=prueba_critica_pct >= colombia_pct)


def render_radar_cards(df: pd.DataFrame) -> None:
    theme = get_theme_tokens()
    sedes = df[["Sede", "Sede Corta"]].dropna().drop_duplicates().sort_values("Sede Corta")
    pruebas = sorted(df["Prueba Base"].dropna().unique().tolist())
    if sedes.empty or not pruebas:
        st.info("No hay suficientes datos para construir perfiles por sede.")
        return

    colombia = df.groupby("Prueba Base", dropna=False)["Acierto"].mean().mul(100).reindex(pruebas).fillna(0)
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
            line=dict(color=theme["muted"]),
            fillcolor="rgba(148, 163, 184, 0.12)",
        ))
        fig.add_trace(go.Scatterpolar(
            r=sede_series.tolist() + [float(sede_series.iloc[0])],
            theta=theta,
            fill="toself",
            name=sede_label,
            line=dict(color=theme["primary"]),
            fillcolor="rgba(37, 99, 235, 0.18)" if theme["base"] == "light" else "rgba(96, 165, 250, 0.18)",
        ))
        fig.update_layout(
            template=theme["plotly_template"],
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme["text"]),
            title=f"Perfil de {sede_label} frente a Colombia",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(range=[30, 100], tickvals=[30, 40, 50, 60, 70, 80, 90, 100], tickfont=dict(size=10, color=theme["text"]), gridcolor=theme["grid"], linecolor=theme["grid"]),
                angularaxis=dict(tickfont=dict(color=theme["text"])),
            ),
            showlegend=True,
            height=380,
            margin=dict(l=30, r=30, t=60, b=20),
        )
        with radar_cols[idx % 2]:
            st.plotly_chart(fig, use_container_width=True)
        if idx % 2 == 1 and idx + 1 < len(sedes):
            radar_cols = st.columns(2)


def show_network_map_tab(df: pd.DataFrame) -> None:
    st.subheader("Mapa general de la red")
    st.caption("Vista inicial para entender el rendimiento global, por prueba y por grado, antes de entrar al detalle por sede.")
    if df.empty:
        st.info("No hay datos para construir el mapa general.")
        return

    theme = get_theme_tokens()
    global_pct = safe_pct(df["Acierto"])
    total_students = safe_nunique(df["ID Estudiante"])
    total_sedes = safe_nunique(df["Sede"])

    by_prueba = (
        df.groupby("Prueba Base", dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .reset_index(name="Rendimiento")
        .sort_values("Rendimiento", ascending=False)
    )

    by_grade = (
        df.groupby("Grado", dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .reset_index(name="Rendimiento")
    )
    by_grade["orden"] = by_grade["Grado"].map(lambda x: grade_sort_key(x)[0])
    by_grade["Grado Etiqueta"] = by_grade["Grado"].map(grade_display_label)
    by_grade = by_grade.sort_values(["orden", "Grado"]).reset_index(drop=True)

    heat_df = (
        df.groupby(["Grado", "Prueba Base"], dropna=False)["Acierto"]
        .mean()
        .mul(100)
        .reset_index(name="Rendimiento")
    )
    heat_df["orden"] = heat_df["Grado"].map(lambda x: grade_sort_key(x)[0])
    heat_df["Grado Etiqueta"] = heat_df["Grado"].map(grade_display_label)
    heat_df = heat_df.sort_values(["orden", "Grado Etiqueta", "Prueba Base"])
    heat = heat_df.pivot(index="Grado Etiqueta", columns="Prueba Base", values="Rendimiento")

    c1, c2, c3 = st.columns(3)
    with c1:
        render_theme_metric_card("Rendimiento global", f"{global_pct:.1f}%", "Promedio de aciertos de toda la red", theme)
    with c2:
        render_theme_metric_card("Estudiantes únicos", f"{total_students:,}", "Incluye los filtros activos", theme)
    with c3:
        render_theme_metric_card("Sedes incluidas", f"{total_sedes:,}", "Cobertura de la red en esta vista", theme)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(by_prueba, x="Prueba Base", y="Rendimiento", text="Rendimiento", title="Rendimiento por prueba", color_discrete_sequence=[theme["primary"]])
        fig.add_hline(y=global_pct, line_dash="dash", line_color=theme["muted"], annotation_text="Promedio global", annotation_font_color=theme["text"], annotation_position="top left")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(yaxis_title="% de rendimiento", xaxis_title="", yaxis_range=compute_axis_bounds(by_prueba["Rendimiento"]), showlegend=False)
        apply_accessible_figure_style(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(by_grade, x="Grado Etiqueta", y="Rendimiento", markers=True, title="Rendimiento por grado")
        fig.update_traces(line=dict(color=theme["success"], width=3), marker=dict(color=theme["success"], size=9))
        fig.add_hline(y=global_pct, line_dash="dash", line_color=theme["muted"], annotation_text="Promedio global", annotation_font_color=theme["text"], annotation_position="top left")
        fig.update_layout(yaxis_title="% de rendimiento", xaxis_title="", yaxis_range=compute_axis_bounds(by_grade["Rendimiento"]), showlegend=False)
        apply_accessible_figure_style(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Mapa cruzado de grado por prueba**")
    fig = px.imshow(heat, text_auto=".1f", aspect="auto", color_continuous_scale=theme["heat_scale"], title="Mapa general de rendimiento de la red")
    fig.update_layout(xaxis_title="Prueba", yaxis_title="Grado", coloraxis_colorbar_title="%")
    apply_accessible_figure_style(fig, theme)
    st.plotly_chart(fig, use_container_width=True)


def show_overview_tab(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str) -> None:
    theme = get_theme_tokens()
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
            title="% de rendimiento por sede",
            text="acierto_colombia_pct",
            color_discrete_map={focus_label: theme["primary"], "Otras sedes": theme["muted"]},
        )
        promedio_colombia = safe_pct(df["Acierto"])
        fig.add_hline(y=promedio_colombia, line_dash="dash", line_color=theme["secondary"], annotation_text="Promedio Colombia", annotation_font_color=theme["text"], annotation_position="top left")
        y_range = compute_axis_bounds(by_sede["acierto_colombia_pct"])
        if promedio_colombia > y_range[1]:
            y_range[1] = min(100.0, float(np.ceil(promedio_colombia + 3)))
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(yaxis_title="% de rendimiento", xaxis_title="", yaxis_range=y_range, showlegend=False)
        apply_accessible_figure_style(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        by_prueba = add_benchmark(df, focus_df, "Prueba Base").sort_values("acierto_red_pct", ascending=False)
        melted = by_prueba.melt(id_vars="Prueba Base", value_vars=["acierto_red_pct", "acierto_sede_pct"], var_name="Serie", value_name="Porcentaje")
        melted["Serie"] = melted["Serie"].map({"acierto_red_pct": "Colombia", "acierto_sede_pct": focus_label})
        fig = px.line(melted, x="Prueba Base", y="Porcentaje", color="Serie", markers=True, title=f"% de rendimiento por prueba: {focus_label} vs Colombia", color_discrete_map={"Colombia": theme["muted"], focus_label: theme["primary"]})
        fig.update_layout(yaxis_title="% de rendimiento", xaxis_title="")
        apply_accessible_figure_style(fig, theme)
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
    fig = px.imshow(heat, text_auto=True, aspect="auto", title="% de rendimiento por sede y prueba", color_continuous_scale=theme["heat_scale"])
    apply_accessible_figure_style(fig, theme)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Trayectoria por grado: sede focal vs Colombia**")
    by_grade_net = df.groupby("Grado", dropna=False)["Acierto"].mean().mul(100).reset_index()
    by_grade_focus = focus_df.groupby("Grado", dropna=False)["Acierto"].mean().mul(100).reset_index().rename(columns={"Acierto": "Acierto Focus"})
    merged = by_grade_net.merge(by_grade_focus, on="Grado", how="left")
    merged["orden"] = merged["Grado"].map(lambda x: grade_sort_key(x)[0])
    merged = merged.sort_values(["orden", "Grado"])
    merged["Grado Etiqueta"] = merged["Grado"].map(grade_display_label)

    melted = merged.melt(id_vars=["Grado", "Grado Etiqueta"], value_vars=["Acierto", "Acierto Focus"], var_name="Serie", value_name="Porcentaje")
    melted["Serie"] = melted["Serie"].map({"Acierto": "Colombia", "Acierto Focus": focus_label})
    fig = px.line(melted, x="Grado Etiqueta", y="Porcentaje", color="Serie", markers=True, title=f"Trayectoria por grado: {focus_label} vs Colombia", color_discrete_map={"Colombia": theme["muted"], focus_label: theme["primary"]})
    fig.update_layout(yaxis_title="% de rendimiento", xaxis_title="")
    apply_accessible_figure_style(fig, theme)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Lectura rápida para docentes y directivos**")
    comp_df = df[~df["Prueba Base"].map(is_english_prueba)].copy()
    comp_focus_df = focus_df[~focus_df["Prueba Base"].map(is_english_prueba)].copy()
    if comp_df.empty:
        comp_df = df.copy()
        comp_focus_df = focus_df.copy()

    comp = sort_benchmark(add_benchmark(comp_df, comp_focus_df, "Competencia"), "Competencia")
    weakest = comp.sort_values("brecha_pp").head(5).rename(columns={
        "acierto_sede_pct": "% de rendimiento en la sede",
        "acierto_red_pct": "% de rendimiento en Colombia",
        "brecha_pp": "Brecha frente a Colombia (pp)",
    })[["Competencia", "% de rendimiento en la sede", "% de rendimiento en Colombia", "Brecha frente a Colombia (pp)"]]
    strongest = comp.sort_values("brecha_pp", ascending=False).head(5).rename(columns={
        "acierto_sede_pct": "% de rendimiento en la sede",
        "acierto_red_pct": "% de rendimiento en Colombia",
        "brecha_pp": "Brecha frente a Colombia (pp)",
    })[["Competencia", "% de rendimiento en la sede", "% de rendimiento en Colombia", "Brecha frente a Colombia (pp)"]]

    st.markdown(f"**Competencias donde {focus_label} necesita más apoyo**")
    st.dataframe(weakest, use_container_width=True, hide_index=True)
    st.markdown(f"**Competencias donde {focus_label} muestra mejor desempeño**")
    st.dataframe(strongest, use_container_width=True, hide_index=True)



def english_level_accuracy_by_grade(df_english: pd.DataFrame) -> pd.DataFrame:
    """% de acierto (0-100) por nivel CEFR dentro de cada grado."""
    if df_english.empty:
        return pd.DataFrame(columns=["Grado", "Grado Etiqueta", "Nivel", "respuestas", "correctas", "pct_acierto"])

    work = df_english[df_english["Nivel Inglés"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["Grado", "Grado Etiqueta", "Nivel", "respuestas", "correctas", "pct_acierto"])

    out = (
        work.groupby(["Grado", "Nivel Inglés"], dropna=False)
        .agg(respuestas=("Acierto", "size"), correctas=("Acierto", "sum"))
        .reset_index()
        .rename(columns={"Nivel Inglés": "Nivel"})
    )
    out["pct_acierto"] = np.where(out["respuestas"] > 0, out["correctas"] / out["respuestas"] * 100, np.nan)
    out["pct_acierto"] = out["pct_acierto"].round(2)
    out["Grado Etiqueta"] = out["Grado"].map(grade_display_label)
    out["Grado Orden"] = out["Grado"].map(lambda x: grade_sort_key(x)[0])

    # Orden estable de niveles (si falta alguno, se mantiene con NaN en pivotes)
    out["Nivel"] = out["Nivel"].astype(str)
    out = out.sort_values(["Grado Orden", "Grado", "Nivel"]).drop(columns="Grado Orden")
    return out


def english_accuracy_pivot(level_accuracy: pd.DataFrame) -> pd.DataFrame:
    """Tabla ancho: filas=grado, columnas=nivel, valores=% de acierto."""
    if level_accuracy.empty:
        return pd.DataFrame(columns=["Grado", *ENGLISH_LEVEL_ORDER])

    pivot = (
        level_accuracy.pivot(index="Grado Etiqueta", columns="Nivel", values="pct_acierto")
        .reindex(columns=ENGLISH_LEVEL_ORDER)
        .reset_index()
        .rename(columns={"Grado Etiqueta": "Grado"})
    )
    return pivot


def english_skill_by_student(df_english: pd.DataFrame) -> pd.DataFrame:
    """Nivel de habilidad por estudiante usando factores de crecimiento por nivel.

    - Cada respuesta correcta aporta (factor).
    - Se normaliza por el máximo posible del estudiante (suma de factores).
    - El índice ponderado queda en 0-100 y el nivel continuo en 0-4.
    - El nivel final se asigna redondeando a 1..4 y mapeando:
      1=Pre A1, 2=A2, 3=A1, 4=B1 (según la tabla solicitada).
    """
    cols = ["ID Estudiante", "Sede", "Sede Corta", "Grado", "Grado Etiqueta", "indice_ponderado", "nivel_continuo", "Nivel habilidad"]
    if df_english.empty:
        return pd.DataFrame(columns=cols)

    work = df_english[df_english["Nivel Inglés"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=cols)

    work["factor"] = work["Nivel Inglés"].map(ENGLISH_GROWTH_FACTORS)
    work = work[work["factor"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=cols)

    work["w_total"] = work["factor"].astype(float)
    work["w_correct"] = work["Acierto"].astype(float) * work["factor"].astype(float)

    agg = (
        work.groupby(["ID Estudiante", "Sede", "Sede Corta", "Grado"], dropna=False)
        .agg(w_total=("w_total", "sum"), w_correct=("w_correct", "sum"))
        .reset_index()
    )
    agg["indice_ponderado"] = np.where(agg["w_total"] > 0, agg["w_correct"] / agg["w_total"] * 100, np.nan)
    agg["nivel_continuo"] = np.where(agg["w_total"] > 0, agg["w_correct"] / agg["w_total"] * 4, np.nan)

    # Asignación a nivel discreto 1..4
    agg["nivel_num"] = np.round(agg["nivel_continuo"]).clip(lower=1, upper=4)
    agg["nivel_num"] = agg["nivel_num"].fillna(1).astype(int)
    agg["Nivel habilidad"] = agg["nivel_num"].map(ENGLISH_LEVEL_FROM_NUM)

    agg["Grado Etiqueta"] = agg["Grado"].map(grade_display_label)
    agg["indice_ponderado"] = agg["indice_ponderado"].round(2)
    agg["nivel_continuo"] = agg["nivel_continuo"].round(3)

    return agg[cols]


def english_modal_skill(skill_df: pd.DataFrame) -> str:
    if skill_df.empty or skill_df["Nivel habilidad"].dropna().empty:
        return "Sin dato"
    freq = skill_df["Nivel habilidad"].value_counts()
    for lvl in ["Pre A1", "A2", "A1", "B1"]:
        if lvl in freq.index and freq[lvl] == freq.max():
            return lvl
    return str(freq.index[0])


def render_english_level_lines(level_accuracy: pd.DataFrame, title: str, theme: dict) -> None:
    if level_accuracy.empty:
        st.info("No hay respuestas clasificables por nivel con los filtros actuales.")
        return

    # Orden de grados para el eje X
    grade_order = (
        level_accuracy[["Grado", "Grado Etiqueta"]]
        .drop_duplicates()
        .assign(_o=lambda d: d["Grado"].map(lambda x: grade_sort_key(x)[0]))
        .sort_values(["_o", "Grado"])
    )["Grado Etiqueta"].tolist()

    fig = px.line(
        level_accuracy,
        x="Grado Etiqueta",
        y="pct_acierto",
        color="Nivel",
        markers=True,
        title=title,
        category_orders={"Nivel": ENGLISH_LEVEL_ORDER, "Grado Etiqueta": grade_order},
        custom_data=["respuestas"],
    )
    fig.update_traces(
        hovertemplate=(
            "Grado %{x}<br>"
            "Nivel %{legendgroup}<br>"
            "% acierto %{y:.1f}%<br>"
            "Respuestas %{customdata[0]:.0f}<extra></extra>"
        )
    )
    fig.update_layout(xaxis_title="Grado", yaxis_title="% de acierto (0-100)", yaxis_range=[0, 100])
    apply_accessible_figure_style(fig, theme)
    st.plotly_chart(fig, use_container_width=True)


def render_english_prueba_panel(prueba: str, df_prueba: pd.DataFrame, focus_prueba: pd.DataFrame, focus_label: str) -> None:

    theme = get_theme_tokens()

    # Nivel de habilidad por estudiante (ponderado por nivel del ítem)
    red_skill = english_skill_by_student(df_prueba)
    sede_skill = english_skill_by_student(focus_prueba)

    if red_skill.empty:
        st.warning(
            "No pude asignar niveles CEFR (Pre A1, A2, A1, B1) dentro de la prueba de Inglés. "
            "Revisa que en los textos del instrumento aparezcan esas etiquetas para poder calcular el nivel por estudiante."
        )
        return

    red_idx = float(red_skill["indice_ponderado"].mean()) if not red_skill.empty else np.nan
    sede_idx = float(sede_skill["indice_ponderado"].mean()) if not sede_skill.empty else np.nan
    brecha = sede_idx - red_idx if pd.notna(red_idx) and pd.notna(sede_idx) else np.nan

    modal_sede = english_modal_skill(sede_skill)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Índice de habilidad (red, 0-100)", f"{red_idx:.1f}" if pd.notna(red_idx) else "Sin dato")
    c2.metric(f"Índice de habilidad ({focus_label}, 0-100)", f"{sede_idx:.1f}" if pd.notna(sede_idx) else "Sin dato")
    c3.metric("Brecha (sede - red)", f"{brecha:+.1f}" if pd.notna(brecha) else "Sin dato")
    c4.metric(f"Nivel modal ({focus_label})", modal_sede)

    st.markdown("### Cómo leer Inglés (sin competencias)")
    st.markdown(
        """
        - **Índice de habilidad (0-100):** cada respuesta correcta se pondera por el nivel del ítem (Pre A1=1, A2=2, A1=3, B1=4) y se normaliza por el máximo posible del estudiante.
        - **Distribución por nivel:** muestra qué **% de estudiantes** del grado queda clasificado en cada nivel (cada grado suma 100%).
        - **Brecha (pp):** diferencia en puntos porcentuales **(sede - red)** por nivel y grado. Valores positivos = la sede tiene más estudiantes en ese nivel; negativos = menos.
        """
    )

    def _grade_order(skill_df: pd.DataFrame) -> list[str]:
        if skill_df.empty:
            return []
        tmp = (
            skill_df[["Grado", "Grado Etiqueta"]]
            .drop_duplicates()
            .assign(_o=lambda d: d["Grado"].map(lambda x: grade_sort_key(x)[0]))
            .sort_values(["_o", "Grado"])
        )
        return tmp["Grado Etiqueta"].tolist()

    def _dist(skill_df: pd.DataFrame, scope: str) -> pd.DataFrame:
        if skill_df.empty:
            return pd.DataFrame(columns=["Alcance", "Grado Etiqueta", "Nivel habilidad", "estudiantes", "pct_estudiantes"])
        dist = (
            skill_df.groupby(["Grado Etiqueta", "Nivel habilidad"], dropna=False)["ID Estudiante"]
            .nunique()
            .reset_index(name="estudiantes")
        )
        dist["pct_estudiantes"] = dist["estudiantes"] / dist.groupby("Grado Etiqueta")["estudiantes"].transform("sum") * 100
        dist["pct_estudiantes"] = dist["pct_estudiantes"].round(2)
        dist["Alcance"] = scope
        return dist

    grade_order = _grade_order(red_skill)
    level_order = ["Pre A1", "A2", "A1", "B1"]

    # 1) Primero: índice promedio por grado (para ver brecha mejor)
    st.markdown("**Índice de habilidad promedio por grado (sede vs red)**")

    def _mean_by_grade(skill_df: pd.DataFrame, scope: str) -> pd.DataFrame:
        if skill_df.empty:
            return pd.DataFrame(columns=["Alcance", "Grado Etiqueta", "Indice"])
        out = (
            skill_df.groupby(["Grado", "Grado Etiqueta"], dropna=False)["indice_ponderado"]
            .mean()
            .reset_index(name="Indice")
        )
        out["orden"] = out["Grado"].map(lambda x: grade_sort_key(x)[0])
        out["Alcance"] = scope
        return out.sort_values(["orden", "Grado"]).drop(columns="orden")

    mean_sede = _mean_by_grade(sede_skill, focus_label)
    mean_red = _mean_by_grade(red_skill, "Red")
    mean_all = pd.concat([mean_red, mean_sede], ignore_index=True)

    if not mean_all.empty:
        fig = px.line(
            mean_all,
            x="Grado Etiqueta",
            y="Indice",
            color="Alcance",
            markers=True,
            title="Índice de habilidad promedio por grado",
            category_orders={"Grado Etiqueta": grade_order},
            color_discrete_map={"Red": theme["muted"], focus_label: theme["primary"]},
        )
        fig.update_layout(
            xaxis_title="Grado",
            yaxis_title="Índice (0-100)",
            yaxis_range=[50, 100],  # <- para ver mejor la brecha
        )
        apply_accessible_figure_style(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

    # 2) Distribución de niveles (sede vs red)
    st.markdown("**Distribución de niveles por grado (sede vs red)**")

    dist_sede = _dist(sede_skill, focus_label)
    dist_red = _dist(red_skill, "Red")

    def _plot_dist(dist: pd.DataFrame, title: str) -> None:
        if dist.empty:
            st.info("No hay estudiantes clasificables con los filtros actuales.")
            return
        fig = px.bar(
            dist,
            x="Grado Etiqueta",
            y="pct_estudiantes",
            color="Nivel habilidad",
            barmode="group",
            title=title,
            text="pct_estudiantes",
            category_orders={"Nivel habilidad": level_order, "Grado Etiqueta": grade_order},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(xaxis_title="Grado", yaxis_title="% de estudiantes", yaxis_range=[0, 100])
        apply_accessible_figure_style(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        _plot_dist(dist_sede, f"{focus_label}: % de estudiantes por nivel")
    with col2:
        _plot_dist(dist_red, "Red filtrada: % de estudiantes por nivel")

    # 3) Solo diferencias: gráfico + (opcional) tabla de diferencias
    st.markdown("**Brecha de distribución por nivel (sede - red, pp)**")

    def _pivot(dist: pd.DataFrame) -> pd.DataFrame:
        if dist.empty:
            return pd.DataFrame(columns=["Grado", *level_order])
        piv = (
            dist.pivot(index="Grado Etiqueta", columns="Nivel habilidad", values="pct_estudiantes")
            .reindex(columns=level_order)
            .fillna(0.0)
            .round(2)
        )
        return piv

    p_sede = _pivot(dist_sede)
    p_red = _pivot(dist_red)

    if not p_sede.empty and not p_red.empty:
        all_grades = sorted(set(p_sede.index.tolist()) | set(p_red.index.tolist()), key=lambda x: grade_sort_key(x)[0])
        delta = p_sede.reindex(index=all_grades, columns=level_order).fillna(0.0) - p_red.reindex(index=all_grades, columns=level_order).fillna(0.0)
        delta = delta.round(2)

        zmax = float(np.nanmax(np.abs(delta.to_numpy()))) if delta.size else 0.0
        zmax = max(1.0, zmax)

        fig = px.imshow(
            delta,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-zmax,
            zmax=zmax,
            title="Mapa de brechas (pp): positivo = sede por encima",
        )
        fig.update_layout(xaxis_title="Nivel", yaxis_title="Grado", coloraxis_colorbar_title="pp")
        apply_accessible_figure_style(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Tabla de diferencias (pp)", expanded=False):
            delta_table = delta.reset_index().rename(columns={"index": "Grado"})
            # renombrar columnas para dejar claro que son pp
            delta_table = delta_table.rename(columns={lvl: f"Δ {lvl} (pp)" for lvl in level_order})
            st.dataframe(delta_table, use_container_width=True, hide_index=True)



def render_prueba_panel(prueba: str, df_prueba: pd.DataFrame, focus_prueba: pd.DataFrame, focus_label: str) -> None:
    theme = get_theme_tokens()
    if df_prueba.empty:
        st.info("No hay datos para esta prueba.")
        return

    if is_english_prueba(prueba):
        render_english_prueba_panel(prueba, df_prueba, focus_prueba, focus_label)
        return

    colombia_pct = safe_pct(df_prueba["Acierto"])
    focus_pct = safe_pct(focus_prueba["Acierto"]) if not focus_prueba.empty else np.nan
    brecha = focus_pct - colombia_pct if pd.notna(focus_pct) else np.nan

    focus_bench = focus_prueba if not focus_prueba.empty else df_prueba.iloc[0:0]
    by_comp = sort_benchmark(add_benchmark(df_prueba, focus_bench, "Competencia"), "Competencia")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Promedio Colombia", f"{colombia_pct:.2f}%")
    c2.metric(f"Promedio {focus_label}", f"{focus_pct:.2f}%" if pd.notna(focus_pct) else "Sin dato")
    c3.metric("Brecha frente a Colombia", f"{brecha:+.2f} pp" if pd.notna(brecha) else "Sin dato")
    c4.metric("Competencias evaluadas", f"{safe_nunique(df_prueba['Competencia']):,}")

    st.markdown("**Comparación por competencia**")
    melted = by_comp.melt(id_vars="Competencia", value_vars=["acierto_red_pct", "acierto_sede_pct"], var_name="Serie", value_name="Porcentaje")
    melted["Serie"] = melted["Serie"].map({"acierto_red_pct": "Colombia", "acierto_sede_pct": focus_label})
    melted["orden"] = melted["Competencia"].map(lambda x: competency_sort_key(x)[0])
    melted = melted.sort_values(["orden", "Competencia"])

    fig = px.bar(
        melted,
        x="Competencia",
        y="Porcentaje",
        color="Serie",
        barmode="group",
        title=f"{prueba}: competencias, {focus_label} vs Colombia",
        color_discrete_map={"Colombia": theme["muted"], focus_label: theme["primary"]},
    )
    fig.update_layout(yaxis_title="% de rendimiento", xaxis_title="")
    apply_accessible_figure_style(fig, theme)
    st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
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
        fig = px.imshow(heat, text_auto=True, aspect="auto", title=f"{prueba}: desempeño por sede y competencia", color_continuous_scale=theme["heat_scale"])
        apply_accessible_figure_style(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        grade_net = df_prueba.groupby("Grado", dropna=False)["Acierto"].mean().mul(100).reset_index()
        grade_focus = focus_prueba.groupby("Grado", dropna=False)["Acierto"].mean().mul(100).reset_index().rename(columns={"Acierto": "Acierto Focus"})
        grade = grade_net.merge(grade_focus, on="Grado", how="left")
        grade["orden"] = grade["Grado"].map(lambda x: grade_sort_key(x)[0])
        grade = grade.sort_values(["orden", "Grado"])
        grade["Grado Etiqueta"] = grade["Grado"].map(grade_display_label)

        melted = grade.melt(id_vars=["Grado", "Grado Etiqueta"], value_vars=["Acierto", "Acierto Focus"], var_name="Serie", value_name="Porcentaje")
        melted["Serie"] = melted["Serie"].map({"Acierto": "Colombia", "Acierto Focus": focus_label})
        fig = px.line(melted, x="Grado Etiqueta", y="Porcentaje", color="Serie", markers=True, title=f"{prueba}: trayectoria por grado", color_discrete_map={"Colombia": theme["muted"], focus_label: theme["primary"]})
        fig.update_layout(yaxis_title="% de rendimiento", xaxis_title="")
        apply_accessible_figure_style(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Tabla de comparación por competencia**")
    st.dataframe(friendly_comp_table(by_comp), use_container_width=True, hide_index=True)


def show_pruebas_tab(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str) -> None:
    st.subheader("Detalle por prueba")
    with st.expander("Cómo leer esta pestaña", expanded=False):
        st.markdown(
            """
            - Cada pestaña resume una prueba completa.
            - La comparación principal siempre es **sede focal vs Colombia**.
            - En **Inglés** la lectura cambia: se mira el **% de respuestas correctas** en cada nivel CEFR (Pre A1, A1, A2, B1).
            - Si una competencia queda por debajo de Colombia, suele ser un buen punto de partida para planear refuerzos.
            """
        )

    pruebas = sorted(df["Prueba Base"].dropna().unique().tolist())
    if not pruebas:
        st.info("No hay pruebas disponibles con el filtro actual.")
        return

    prueba_tabs = st.tabs(pruebas)
    for tab, prueba in zip(prueba_tabs, pruebas):
        with tab:
            df_prueba = df[df["Prueba Base"] == prueba].copy()
            focus_prueba = focus_df[focus_df["Prueba Base"] == prueba].copy()
            render_prueba_panel(prueba, df_prueba, focus_prueba, focus_label)


def build_question_bank(items_prueba: pd.DataFrame) -> pd.DataFrame:
    if items_prueba.empty:
        return items_prueba
    ordered = items_prueba.copy()
    ordered = ordered.sort_values(["Grado Orden", "Grado", "dificultad", "p_bis"], ascending=[True, True, True, False]).reset_index(drop=True)
    ordered["orden_en_grado"] = ordered.groupby("Grado").cumcount() + 1
    ordered["selector_label"] = ordered.apply(lambda row: f"Pregunta {int(row['orden_en_grado'])} · Grado {row['Grado Etiqueta']}", axis=1)
    return ordered


def build_option_comparison(question_id: int | float, colombia_df: pd.DataFrame, focus_df: pd.DataFrame) -> pd.DataFrame:
    def _profile(frame: pd.DataFrame, pct_col: str) -> pd.DataFrame:
        data = frame[frame["QuestionId"] == question_id].copy()
        if data.empty:
            return pd.DataFrame(columns=["Respuesta Limpia", "Acierto", pct_col])
        out = data.groupby(["Respuesta Limpia", "Acierto"], dropna=False).size().reset_index(name="n")
        total = out["n"].sum()
        out[pct_col] = np.where(total > 0, out["n"] / total * 100, 0)
        return out[["Respuesta Limpia", "Acierto", pct_col]]

    colombia = _profile(colombia_df, "pct_colombia")
    focus = _profile(focus_df, "pct_sede")
    merged = colombia.merge(focus, on=["Respuesta Limpia", "Acierto"], how="outer").fillna(0)
    if merged.empty:
        return merged

    merged["tipo"] = np.where(merged["Acierto"] == 1, "Respuesta correcta", "Distractor")
    merged["Opción de respuesta"] = merged["Respuesta Limpia"].map(lambda x: shorten_text(x, 110))
    merged["Opción de respuesta gráfico"] = merged["Opción de respuesta"].map(lambda x: wrap_plot_label(x, width=42, max_lines=4))
    merged["pico"] = merged[["pct_colombia", "pct_sede"]].max(axis=1)
    merged = merged.sort_values(["Acierto", "pico"], ascending=[False, False]).reset_index(drop=True)
    merged["pct_colombia"] = merged["pct_colombia"].round(2)
    merged["pct_sede"] = merged["pct_sede"].round(2)
    return merged


def option_dumbbell_chart(option_comp: pd.DataFrame, focus_label: str) -> go.Figure:
    theme = get_theme_tokens()
    fig = go.Figure()
    if option_comp.empty:
        return fig

    max_pct = float(option_comp[["pct_colombia", "pct_sede"]].to_numpy().max()) if not option_comp.empty else 0.0
    x_upper = min(100.0, max(12.0, float(np.ceil(max_pct + 6))))

    for _, row in option_comp.iterrows():
        line_color = theme["success"] if row["tipo"] == "Respuesta correcta" else theme["muted"]
        fig.add_trace(go.Scatter(
            x=[row["pct_colombia"], row["pct_sede"]],
            y=[row["Opción de respuesta gráfico"], row["Opción de respuesta gráfico"]],
            mode="lines",
            line=dict(color=line_color, width=4),
            hoverinfo="skip",
            showlegend=False,
        ))

    marker_colors = [theme["success"] if t == "Respuesta correcta" else theme["muted"] for t in option_comp["tipo"]]
    fig.add_trace(go.Scatter(
        x=option_comp["pct_colombia"],
        y=option_comp["Opción de respuesta gráfico"],
        mode="markers+text",
        name="Colombia",
        text=[f"{v:.1f}%" for v in option_comp["pct_colombia"]],
        textposition="top left",
        textfont=dict(color=theme["text"]),
        cliponaxis=False,
        marker=dict(symbol="circle", size=12, color=marker_colors, line=dict(color=theme["background"], width=1)),
    ))
    fig.add_trace(go.Scatter(
        x=option_comp["pct_sede"],
        y=option_comp["Opción de respuesta gráfico"],
        mode="markers+text",
        name=focus_label,
        text=[f"{v:.1f}%" for v in option_comp["pct_sede"]],
        textposition="top right",
        textfont=dict(color=theme["text"]),
        cliponaxis=False,
        marker=dict(symbol="diamond", size=12, color=marker_colors, line=dict(color=theme["background"], width=1)),
    ))
    fig.update_layout(
        template=theme["plotly_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme["text"]),
        title=f"Cómo se repartieron las respuestas: {focus_label} vs Colombia",
        xaxis_title="% de estudiantes que marcó la opción",
        yaxis_title="",
        height=max(420, 120 + 95 * len(option_comp)),
        margin=dict(l=10, r=30, t=70, b=20),
        yaxis=dict(automargin=True, tickfont=dict(size=12, color=theme["text"])),
        xaxis=dict(tickfont=dict(size=12, color=theme["text"]), range=[0, x_upper], gridcolor=theme["grid"], zerolinecolor=theme["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def render_compact_summary_card(title: str, value: str, tone: str | None = None) -> None:
    theme = get_theme_tokens()
    body = str(value).replace(" · ", "<br>")
    tone = tone or theme["text"]
    st.markdown(
        f"""
        <div style="
            background:{theme['card_bg']};
            border:1px solid {theme['card_border']};
            border-radius:14px;
            padding:0.9rem 0.95rem;
            min-height:112px;
        ">
            <div style="font-size:0.95rem; color:{theme['muted']}; margin-bottom:0.45rem; line-height:1.25;">{title}</div>
            <div style="font-size:1.45rem; font-weight:700; color:{tone}; line-height:1.22; word-break:break-word;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_psychometrics_tab(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str) -> None:
    theme = get_theme_tokens()
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
    if not pruebas:
        st.info("No hay pruebas disponibles con el filtro actual.")
        return

    selected_prueba = st.selectbox("Prueba para analizar", pruebas)
    items_prueba = build_question_bank(items[items["Prueba Base"] == selected_prueba].copy())
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
    with c1:
        render_compact_summary_card("Preguntas analizadas", f"{len(items_prueba):,}", tone=theme["text"])
    with c2:
        render_compact_summary_card("Pregunta más difícil", _card_label(hardest), tone=theme["warning"])
    with c3:
        render_compact_summary_card("Pregunta más fácil", _card_label(easiest), tone=theme["success"])
    with c4:
        render_compact_summary_card("Mayor discriminación", _card_label(best_disc), tone=theme["primary"])
    with c5:
        render_compact_summary_card("Menor discriminación", _card_label(worst_disc), tone=theme["danger"])

    difficult_table = items_prueba[["selector_label", "Competencia", "dificultad_pct", "p_bis", "d27"]].rename(columns={
        "selector_label": "Pregunta",
        "dificultad_pct": "% de rendimiento",
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
    m1.metric("% de rendimiento", f"{item_row['dificultad_pct']:.2f}%")
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
        title=f"{selected_prueba}: mapa de preguntas (dificultad y discriminación)",
    )
    fig.add_vline(x=0.50, line_dash="dash", line_color=theme["muted"])
    fig.add_hline(y=0.20, line_dash="dash", line_color=theme["muted"])
    fig.update_layout(xaxis_title="Dificultad (proporción de respuestas correctas)", yaxis_title="Discriminación (point-biserial)")
    apply_accessible_figure_style(fig, theme)
    st.plotly_chart(fig, use_container_width=True)


def show_antiguedad_tab(df: pd.DataFrame, focus_df: pd.DataFrame, focus_label: str) -> None:
    theme = get_theme_tokens()
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
        melted = net.melt(id_vars="Antiguedad", value_vars=["acierto_red_pct", "acierto_sede_pct"], var_name="Serie", value_name="Porcentaje")
        melted["Serie"] = melted["Serie"].map({"acierto_red_pct": "Colombia", "acierto_sede_pct": focus_label})
        fig = px.line(melted, x="Antiguedad", y="Porcentaje", color="Serie", markers=True, title="Desempeño por antigüedad", color_discrete_map={"Colombia": theme["muted"], focus_label: theme["primary"]})
        fig.update_layout(yaxis_title="% de rendimiento", xaxis_title="Años de antigüedad")
        apply_accessible_figure_style(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        bar_colors = [theme["success"] if x >= 0 else theme["danger"] for x in net["brecha_pp"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=net["Antiguedad"], y=net["brecha_pp"], text=net["brecha_pp"], marker_color=bar_colors))
        fig.add_hline(y=0, line_dash="dash", line_color=theme["muted"])
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(title=f"Brecha por antigüedad: {focus_label} vs Colombia", yaxis_title="Puntos porcentuales", xaxis_title="Años de antigüedad")
        apply_accessible_figure_style(fig, theme)
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
        fig = px.imshow(heat, text_auto=True, aspect="auto", title=f"{focus_label}: % de rendimiento por antigüedad y prueba", color_continuous_scale=theme["heat_scale"])
        apply_accessible_figure_style(fig, theme)
        st.plotly_chart(fig, use_container_width=True)


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    st.sidebar.header("Enfoque del tablero")

    sedes_base = df[["Sede", "Sede Corta"]].dropna().drop_duplicates().sort_values("Sede Corta")
    if sedes_base.empty:
        return df.iloc[0:0], df.iloc[0:0], "Sin sede"

    sede_labels = sedes_base["Sede Corta"].tolist()
    label_to_sede = {row["Sede Corta"]: row["Sede"] for _, row in sedes_base.iterrows()}
    sede_focal_label = st.sidebar.selectbox("Sede focal", sede_labels, index=0)

    grades = sorted(df["Grado"].dropna().unique().tolist(), key=grade_sort_key)
    selected_grades = st.sidebar.multiselect("Grado", grades, default=grades)

    df_grado = df[df["Grado"].isin(selected_grades)] if selected_grades else df.iloc[0:0]
    courses = sorted(df_grado["Curso"].dropna().unique().tolist(), key=course_sort_key)
    selected_courses = st.sidebar.multiselect("Curso", courses, default=courses)

    filtered = df[df["Grado"].isin(selected_grades)].copy() if selected_grades else df.iloc[0:0].copy()
    filtered = filtered[filtered["Curso"].isin(selected_courses)].copy() if selected_courses else filtered.iloc[0:0].copy()

    sede_value = label_to_sede[sede_focal_label]
    focus_df = filtered[filtered["Sede"] == sede_value].copy()
    return filtered, focus_df, sede_focal_label




@st.cache_data(show_spinner=False)
def load_prepared_socio_default() -> pd.DataFrame | None:
    local_file = socio_load_default_file_if_exists()
    if local_file is None:
        return None
    raw = socio_read_dataframe_from_path(str(local_file), int(local_file.stat().st_mtime_ns))
    raw = socio_normalize_columns(raw)
    missing = socio_validate_columns(raw)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en Auxiliares.xlsx: {', '.join(missing)}")
    return socio_prepare_socio_data(raw)


def align_socio_to_academic_scope(socio_df: pd.DataFrame, academic_filtered: pd.DataFrame, focus_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if socio_df is None or socio_df.empty:
        empty = pd.DataFrame()
        return empty, empty

    out = socio_df.copy()
    out = out[out["Indicador"].notna()].copy()
    out = out[out["Indicador"] != "Autoeficacia"].copy()

    grade_values = sorted(academic_filtered["Grado"].dropna().astype(str).unique().tolist(), key=socio_grade_sort_key)
    if grade_values:
        overlap_grades = [g for g in grade_values if g in set(out["Grado"].dropna().astype(str).tolist())]
        if overlap_grades:
            out = out[out["Grado"].astype(str).isin(overlap_grades)].copy()

    gender_values = sorted(academic_filtered["Genero"].dropna().astype(str).unique().tolist())
    if gender_values:
        overlap_genders = [g for g in gender_values if g in set(out["Genero"].dropna().astype(str).tolist())]
        if overlap_genders:
            out = out[out["Genero"].astype(str).isin(overlap_genders)].copy()

    academic_courses = {str(x).strip() for x in academic_filtered["Curso"].dropna().astype(str).tolist() if str(x).strip()}
    socio_sections = {str(x).strip() for x in out["Seccion"].dropna().astype(str).tolist() if str(x).strip()}
    overlap_sections = sorted(academic_courses & socio_sections, key=socio_course_sort_key)
    if overlap_sections:
        out = out[out["Seccion"].astype(str).isin(overlap_sections)].copy()

    focus_df = out[out["Sede Corta"] == focus_label].copy()
    return out, focus_df


def socio_status_label(score: float, gap: float, coverage: float) -> str:
    if pd.isna(coverage) or coverage < 60:
        return "Dato insuficiente"
    if pd.notna(score) and pd.notna(gap) and score >= 75 and gap >= -5:
        return "Fortaleza"
    if pd.notna(score) and pd.notna(gap) and (score < 60 or gap <= -8):
        return "Atención prioritaria"
    return "Seguimiento"


def socio_teacher_note(score: float, gap: float, coverage: float) -> str:
    if pd.isna(coverage) or coverage < 60:
        return "Revisar captura o cobertura antes de intervenir."
    if pd.notna(score) and pd.notna(gap) and score >= 75 and gap >= 0:
        return "Mantener esta práctica y usarla como referencia para otros cursos."
    if pd.notna(score) and pd.notna(gap) and gap <= -8:
        return "Priorizar acompañamiento y seguimiento cercano frente a la red."
    if pd.notna(score) and score < 60:
        return "Conviene abrir conversación con docentes y mentores sobre este punto."
    return "Monitorear con el director de grupo y revisar evolución en el siguiente corte."


def render_socio_insight_card(title: str, indicator: str, detail: str, tone: str) -> None:
    theme = get_theme_tokens()
    st.markdown(
        f"""
        <div style="
            background:{theme['card_bg']};
            border:1px solid {theme['card_border']};
            border-radius:14px;
            padding:0.95rem 1rem;
            min-height:128px;
        ">
            <div style="font-size:0.9rem; color:{theme['muted']}; margin-bottom:0.35rem;">{title}</div>
            <div style="font-size:1.15rem; font-weight:700; color:{tone}; line-height:1.25;">{indicator}</div>
            <div style="font-size:0.9rem; color:{theme['text']}; margin-top:0.35rem; line-height:1.35;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_embedded_socioemocional_tab(academic_filtered: pd.DataFrame, focus_label: str) -> None:
    theme = get_theme_tokens()
    st.subheader("Lectura socioemocional de la sede")
    st.caption("Una sola hoja para contrastar la sede focal frente a la red en clima, apoyo, recursos y habilidades socioemocionales.")

    try:
        socio_df = load_prepared_socio_default()
    except Exception as exc:
        st.warning(f"No pude cargar Auxiliares.xlsx: {exc}")
        return

    if socio_df is None:
        st.info("Para activar esta hoja, agrega `data/Auxiliares.xlsx` al repositorio.")
        return

    scoped_all, scoped_focus_all = align_socio_to_academic_scope(socio_df, academic_filtered, focus_label)
    if scoped_all.empty:
        st.info("No encontré registros socioemocionales que coincidan con la sede o los filtros activos del tablero.")
        return

    survey_options = ["Todos los ciclos"] + sorted(scoped_all["SurveyName"].dropna().unique().tolist())
    selected_survey = st.radio("Ciclo a revisar", survey_options, horizontal=True, key="embedded_socio_survey_filter")
    if selected_survey == "Todos los ciclos":
        scoped = scoped_all.copy()
        scoped_focus = scoped_focus_all.copy()
    else:
        scoped = scoped_all[scoped_all["SurveyName"] == selected_survey].copy()
        scoped_focus = scoped_focus_all[scoped_focus_all["SurveyName"] == selected_survey].copy()

    if scoped.empty:
        st.info("No hay información socioemocional para ese ciclo con los filtros actuales.")
        return
    if scoped_focus.empty:
        st.warning(f"La sede {focus_label} no tiene registros socioemocionales en el alcance seleccionado.")
        return

    # Excluir Autoeficacia si apareciera en el archivo (no se reporta en esta versión)
    if "Indicador" in scoped.columns:
        scoped = scoped[~scoped["Indicador"].astype(str).str.strip().str.lower().eq("autoeficacia")].copy()
    if "Indicador" in scoped_focus.columns:
        scoped_focus = scoped_focus[~scoped_focus["Indicador"].astype(str).str.strip().str.lower().eq("autoeficacia")].copy()

    indicator_bench = socio_sort_socio_benchmark(socio_socio_benchmark(scoped, scoped_focus, "Indicador"), "Indicador")
    indicator_bench = indicator_bench[indicator_bench["Indicador"].notna()].copy()
    if indicator_bench.empty:
        st.info("No hay indicadores socioemocionales interpretables para esta vista.")
        return

    # Vista principal: solo por indicador (encuesta agrupada)

    chart_df = indicator_bench.melt(id_vars="Indicador", value_vars=["puntaje_red", "puntaje_sede"], var_name="Serie", value_name="Puntaje")
    chart_df["Serie"] = chart_df["Serie"].map({"puntaje_red": "Red", "puntaje_sede": focus_label})
    chart_df["Indicador corto"] = chart_df["Indicador"].map(lambda x: socio_wrap_plot_label(x, width=26, max_lines=3))

    col1, col2 = st.columns([1.15, 0.85])
    with col1:
        fig = px.bar(chart_df, y="Indicador corto", x="Puntaje", color="Serie", barmode="group", orientation="h", title=f"Indicadores socioemocionales: {focus_label} vs red", color_discrete_map={"Red": theme["muted"], focus_label: theme["primary"]})
        fig.update_layout(yaxis_title="", xaxis_title="Puntaje 0-100", legend_title_text="")
        apply_accessible_figure_style(fig, theme)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        gap_df = indicator_bench.copy()
        gap_df["Indicador corto"] = gap_df["Indicador"].map(lambda x: socio_wrap_plot_label(x, width=22, max_lines=3))
        gap_df["Estado"] = np.where(gap_df["brecha_puntaje"] >= 0, "Por encima", "Por debajo")
        fig = px.bar(gap_df, y="Indicador corto", x="brecha_puntaje", color="Estado", orientation="h", text="brecha_puntaje", title="Brecha de la sede por indicador", color_discrete_map={"Por encima": theme["success"], "Por debajo": theme["danger"]})
        fig.add_vline(x=0, line_dash="dash", line_color=theme["muted"])
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(yaxis_title="", xaxis_title="Puntos frente a la red", showlegend=False)
        apply_accessible_figure_style(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

    priority_df = indicator_bench.copy()
    priority_df["Semáforo"] = priority_df.apply(lambda r: socio_status_label(r["puntaje_sede"], r["brecha_puntaje"], r["cobertura_sede"]), axis=1)
    priority_df["Lectura sugerida"] = priority_df.apply(lambda r: socio_teacher_note(r["puntaje_sede"], r["brecha_puntaje"], r["cobertura_sede"]), axis=1)

    # (Se omiten tarjetas de lectura rápida para enfocarnos en el contraste por indicador.)

    teacher_table = priority_df[["Indicador", "puntaje_sede", "puntaje_red", "brecha_puntaje", "favorable_sede", "cobertura_sede", "Semáforo", "Lectura sugerida"]].rename(columns={
        "puntaje_sede": f"Puntaje {focus_label}",
        "puntaje_red": "Puntaje red",
        "brecha_puntaje": "Brecha vs red",
        "favorable_sede": f"% favorable {focus_label}",
        "cobertura_sede": f"% cobertura {focus_label}",
    })
    st.markdown("**Lectura por indicador para docentes y directivos**")
    st.dataframe(teacher_table, use_container_width=True, hide_index=True)

    st.markdown("**Bajar al detalle: una lectura concreta**")
    indicator_options = priority_df["Indicador"].dropna().tolist()
    selected_indicator = st.selectbox("Indicador para profundizar", indicator_options, key="embedded_socio_indicator_selector")

    qsum = socio_question_summary(scoped[scoped["Indicador"] == selected_indicator], scoped_focus[scoped_focus["Indicador"] == selected_indicator])
    # Añadir conteo de opciones de respuesta (distintas) por pregunta en red y sede
    scoped_ind = scoped[(scoped["Indicador"] == selected_indicator) & (scoped["Respuesta Reporte"].notna())].copy()
    focus_ind = scoped_focus[(scoped_focus["Indicador"] == selected_indicator) & (scoped_focus["Respuesta Reporte"].notna())].copy()

    opt_red = scoped_ind.groupby("TexQuestion", dropna=False)["Respuesta Reporte"].nunique()
    opt_sede = focus_ind.groupby("TexQuestion", dropna=False)["Respuesta Reporte"].nunique()
    n_red = scoped_ind.groupby("TexQuestion", dropna=False)["Respuesta Reporte"].size()
    n_sede = focus_ind.groupby("TexQuestion", dropna=False)["Respuesta Reporte"].size()

    qsum = qsum[qsum["TexQuestion"].notna()].copy()
    qsum["Opciones red"] = qsum["TexQuestion"].map(opt_red).fillna(0).astype(int)
    qsum["Opciones sede"] = qsum["TexQuestion"].map(opt_sede).fillna(0).astype(int)
    qsum["Respuestas red"] = qsum["TexQuestion"].map(n_red).fillna(0).astype(int)
    qsum["Respuestas sede"] = qsum["TexQuestion"].map(n_sede).fillna(0).astype(int)

    if qsum.empty:
        st.info("No hay preguntas disponibles para ese indicador.")
        return

    qsum = qsum.sort_values(["brecha_puntaje", "cobertura_sede", "puntaje_sede"], ascending=[True, True, True]).reset_index(drop=True)
    question_options = qsum["TexQuestion"].tolist()
    selected_question = st.selectbox(
        "Pregunta guía",
        question_options,
        index=0,
        format_func=lambda x: socio_wrap_plot_label(x, width=90, max_lines=2).replace("<br>", " "),
        key="embedded_socio_question_selector",
    )
    qrow = qsum[qsum["TexQuestion"] == selected_question].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Puntaje red", f"{qrow['puntaje_red']:.1f}")
    m2.metric(f"Puntaje {focus_label}", f"{qrow['puntaje_sede']:.1f}" if pd.notna(qrow["puntaje_sede"]) else "Sin dato")
    m3.metric("Brecha", f"{qrow['brecha_puntaje']:+.1f} pp" if pd.notna(qrow["brecha_puntaje"]) else "Sin dato")
    m4.metric("Cobertura sede", f"{qrow['cobertura_sede']:.1f}%" if pd.notna(qrow["cobertura_sede"]) else "Sin dato")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Opciones red", f"{int(qrow.get('Opciones red', 0)):,}")
    o2.metric(f"Opciones {focus_label}", f"{int(qrow.get('Opciones sede', 0)):,}")
    o3.metric("Respuestas red", f"{int(qrow.get('Respuestas red', 0)):,}")
    o4.metric(f"Respuestas {focus_label}", f"{int(qrow.get('Respuestas sede', 0)):,}")
    st.markdown(f"**Pregunta:** {selected_question}")

    if selected_indicator == "Autoconciencia emocional":
        st.markdown("**Autoconciencia emocional: aciertos y confusiones por emoción**")
        emo_summary = socio_emotion_summary(scoped, scoped_focus, focus_label)
        st.dataframe(emo_summary, use_container_width=True, hide_index=True)

        cmat1, cmat2 = st.columns(2)
        with cmat1:
            st.markdown(f"**{focus_label}: cómo se confundieron las emociones**")
            focus_matrix = socio_emotion_confusion_matrix(scoped_focus)
            if focus_matrix.empty:
                st.info("No hay suficiente detalle para esta sede.")
            else:
                fig = px.imshow(focus_matrix, text_auto=True, aspect="auto", title=f"{focus_label}: emoción objetivo vs respuesta elegida", color_continuous_scale=theme["heat_scale"])
                apply_accessible_figure_style(fig, theme)
                st.plotly_chart(fig, use_container_width=True)
        with cmat2:
            st.markdown("**Demás sedes: referencia de confusiones**")
            other_matrix = socio_emotion_confusion_matrix(scoped[scoped["Sede Corta"] != focus_label].copy())
            if other_matrix.empty:
                st.info("No hay suficiente detalle en las demás sedes.")
            else:
                fig = px.imshow(other_matrix, text_auto=True, aspect="auto", title="Demás sedes: emoción objetivo vs respuesta elegida", color_continuous_scale=theme["heat_scale"])
                apply_accessible_figure_style(fig, theme)
                st.plotly_chart(fig, use_container_width=True)

    else:
        # Distribución de opciones de respuesta (proporción dentro de la sede y dentro de la red)
        st.markdown("**Distribución de opciones de respuesta (porcentaje dentro de cada grupo)**")

        red_profile = socio_response_profile(scoped, selected_question).rename(columns={"pct": "Red"})
        focus_profile = socio_response_profile(scoped_focus, selected_question).rename(columns={"pct": focus_label})

        merged = red_profile.merge(focus_profile, on="Respuesta Reporte", how="outer").fillna(0)
        if merged.empty:
            st.info("No hay respuestas suficientes para construir la distribución en este filtro.")
        else:
            merged["Δ (pp)"] = merged[focus_label] - merged["Red"]

            # Orden sugerido según tipo de escala (evita lecturas raras y mejora legibilidad)
            scale_series = scoped.loc[scoped["TexQuestion"] == selected_question, "Escala"].dropna()
            scale = str(scale_series.iloc[0]) if not scale_series.empty else None
            order_map = {
                "emotion": ["Alegría", "Rabia", "Tristeza", "Sorpresa"],
                "yes_no": ["Sí", "No"],
                "three_mucho": ["Ningún día", "Algunos días", "Muchos días"],
                "three_mucho_literal": ["Nada", "Poco", "Mucho"],
                "si_algunas_no": ["No", "Algunas veces", "Sí"],
                "agree4": ["Muy en desacuerdo", "En desacuerdo", "De acuerdo", "Muy de acuerdo"],
                "freq4": ["Nunca", "Pocas veces", "Muchas veces", "Siempre"],
                "satisfaction3": ["Insatisfecho", "Me da igual", "Satisfecho"],
                "learn_compare": [
                    "Aprendo menos que cuando estaba en casa",
                    "Aprendo igual que cuando estaba en casa",
                    "Aprendo más que cuando estaba en casa",
                ],
            }
            if scale in order_map:
                merged = merged.set_index("Respuesta Reporte").reindex(order_map[scale]).fillna(0).reset_index()

            merged = merged.rename(columns={
                "Respuesta Reporte": "Opción",
                "Red": "Red (%)",
                focus_label: f"{focus_label} (%)",
            })
            merged["Red (%)"] = merged["Red (%)"].round(2)
            merged[f"{focus_label} (%)"] = merged[f"{focus_label} (%)"].round(2)
            merged["Δ (pp)"] = merged["Δ (pp)"].round(2)

            st.dataframe(
                merged[["Opción", "Red (%)", f"{focus_label} (%)", "Δ (pp)"]],
                use_container_width=True,
                hide_index=True,
            )

def main() -> None:
    st.title("Evaluación Diagnóstica 2026 Calendario A")
    st.caption("Tablero pedagógico para contrastar la sede frente a la red y sumar una lectura socioemocional útil para docentes.")

    local_file = load_default_file_if_exists()
    if local_file is None:
        st.error("No encontré el archivo base. Verifica que exista en data/EvaluarParaAvanzar_CalA.xlsx")
        st.stop()

    raw = read_dataframe_from_path(str(local_file), int(local_file.stat().st_mtime_ns))
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
        "Mapa general de la red",
        "Tablero directivo",
        "Detalle por prueba",
        "Análisis de las respuestas",
        "Análisis de antigüedad del estudiante",
        "Lectura socioemocional",
    ])

    with tabs[0]:
        show_network_map_tab(filtered)
    with tabs[1]:
        show_overview_tab(filtered, focus_df, focus_label)
    with tabs[2]:
        show_pruebas_tab(filtered, focus_df, focus_label)
    with tabs[3]:
        show_psychometrics_tab(filtered, focus_df, focus_label)
    with tabs[4]:
        show_antiguedad_tab(filtered, focus_df, focus_label)
    with tabs[5]:
        show_embedded_socioemocional_tab(filtered, focus_label)


if __name__ == "__main__":
    main()
