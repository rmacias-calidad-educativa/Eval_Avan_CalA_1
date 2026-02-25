# Visualizador académico

Aplicación en Streamlit para explorar resultados académicos por:

- sede
- grado
- curso
- prueba
- competencia
- estudiante
- pregunta

## Archivos del proyecto

- `app.py`: aplicación principal
- `requirements.txt`: dependencias
- `data/EvaluarParaAvanzar_CalA.xlsx`: base de datos cargada por defecto
- `.streamlit/config.toml`: configuración visual básica

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar en Streamlit Community Cloud

1. Sube esta carpeta a un repositorio en GitHub.
2. Entra a Streamlit Community Cloud.
3. Crea una nueva app conectando tu repositorio.
4. Selecciona:
   - Branch: `main`
   - Main file path: `app.py`
5. Deploy.

## Notas

- Si dejas el archivo Excel dentro de `data/`, la app lo abrirá automáticamente.
- También puedes subir otro archivo desde la interfaz.
- La app puede remover duplicados por `ID Estudiante + Prueba + QuestionId`.
