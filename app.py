import json
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

INTEGRANTES = [
    "Daniel Varela Obregon"
]
RUTA = Path(__file__).parent / "artefactos"
FACTOR_MAX = 3.0

CARACTERISTICAS = {
    "radius": "Radio",
    "texture": "Textura",
    "perimeter": "Perímetro",
    "area": "Área",
    "smoothness": "Suavidad",
    "compactness": "Compacidad",
    "concavity": "Concavidad",
    "concave_points": "Puntos cóncavos",
    "symmetry": "Simetría",
    "fractal_dimension": "Dimensión fractal",
}
GRUPOS = {"mean": "Promedio", "error": "Error estándar", "worst": "Peor valor"}


def etiqueta(col: str) -> str:
    if col.startswith("mean_"):
        base, grupo = col[5:], "mean"
    elif col.startswith("worst_"):
        base, grupo = col[6:], "worst"
    else:
        base, grupo = col[: -len("_error")], "error"
    return f"{CARACTERISTICAS[base]} ({GRUPOS[grupo].lower()})"


@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA / "modelo.joblib")


@st.cache_data
def cargar_json(nombre: str):
    return json.loads((RUTA / nombre).read_text(encoding="utf-8"))


@st.cache_data
def cargar_datos() -> pd.DataFrame:
    return pd.read_csv(RUTA / "datos_cancer_mama_limpios.csv")


def validar_entrada(valores: dict, rangos: dict):
    errores, advertencias = [], []
    for col, v in valores.items():
        nombre = etiqueta(col)
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float, np.number)):
            errores.append(f"{nombre}: debe ser un número.")
            continue
        if not math.isfinite(v):
            errores.append(f"{nombre}: el valor no es finito.")
            continue
        r = rangos[col]
        if v < 0:
            errores.append(f"{nombre}: no puede ser negativo.")
        elif v > r["max"] * FACTOR_MAX:
            errores.append(
                f"{nombre}: {v:g} es demasiado alto (máximo aceptado {r['max'] * FACTOR_MAX:.4g}).")
        elif v < r["min"] or v > r["max"]:
            advertencias.append(
                f"{nombre}: {v:g} está fuera del rango observado "
                f"({r['min']:.4g} a {r['max']:.4g}); la predicción es menos confiable.")
    if not errores:
        for base in ["radius", "perimeter", "area"]:
            if valores[f"worst_{base}"] < valores[f"mean_{base}"]:
                advertencias.append(
                    f"{CARACTERISTICAS[base]}: el peor valor es menor que el promedio, "
                    "lo cual es poco habitual.")
    return errores, advertencias


def predecir(modelo, valores: dict, columnas: list):
    X = pd.DataFrame([valores])[columnas]
    clase = int(modelo.predict(X)[0])
    prob = float(modelo.predict_proba(X)[0, 1])
    return clase, prob


def fijar_valores(valores: dict):
    for col, v in valores.items():
        st.session_state[f"in_{col}"] = float(v)


st.set_page_config(page_title="Clasificación de tumores de mama", page_icon="🩺", layout="wide")

try:
    modelo = cargar_modelo()
    rangos = cargar_json("rangos.json")
    metricas = cargar_json("metricas.json")
    registro = cargar_json("registro_conocido.json")
    datos = cargar_datos()
except FileNotFoundError:
    st.error("Faltan los archivos de la carpeta `artefactos/`. "
             "Ejecuta `python entrenar_modelo.py` y vuelve a cargar la app.")
    st.stop()

COLUMNAS = list(rangos.keys())

st.title("🩺 Clasificación de tumores de mama")
st.warning("**Prototipo académico.** Esta aplicación no es un dispositivo médico y no "
           "sustituye el diagnóstico ni la decisión de un profesional de la salud.")

tab_proy, tab_eda, tab_pred, tab_met = st.tabs(
    ["📋 Proyecto", "📊 Exploración (EDA)", "🔮 Predicción", "📈 Métricas y limitaciones"])

with tab_proy:
    st.header("Presentación del proyecto")
    st.markdown(
        "**Objetivo:** predecir si un tumor de mama es **maligno** o **benigno** a partir de "
        "30 medidas numéricas de los núcleos celulares, y comparar regresión logística con un "
        "árbol de decisión. El modelo desplegado es la **regresión logística**.\n\n"
        "**Fuente de los datos:** Breast Cancer Wisconsin (Diagnostic), incluido en "
        "[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) "
        "(569 registros, 30 variables).\n\n"
        "**Curso:** Minería de Datos, Universidad de la Costa. Tercer corte.")
    st.subheader("Integrantes")
    for nombre in INTEGRANTES:
        st.markdown(f"- {nombre}")
    st.subheader("Variables del modelo")
    st.markdown(
        "Se miden diez características de cada núcleo celular y cada una aparece en tres "
        "versiones: **promedio**, **error estándar** y **peor valor** (el más grande). "
        "Los valores provienen de imágenes digitalizadas y se usan en las unidades del dataset.")

with tab_eda:
    st.header("Exploración de datos")
    etiquetas_dx = datos["target"].map({0: "Maligno", 1: "Benigno"})

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribución de diagnósticos")
        conteo = etiquetas_dx.value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(conteo.index, conteo.values, color=["#4C78A8", "#E45756"][: len(conteo)])
        for i, v in enumerate(conteo.values):
            ax.text(i, v + 4, str(v), ha="center")
        ax.set_ylabel("Registros")
        st.pyplot(fig)
        st.caption("Hay más casos benignos que malignos; por eso se evalúa el recall "
                   "y no solo el accuracy.")
    with c2:
        st.subheader("Radio promedio según diagnóstico")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.boxplot([datos.loc[etiquetas_dx == d, "mean_radius"] for d in ["Benigno", "Maligno"]],
                   tick_labels=["Benigno", "Maligno"])
        ax.set_ylabel("Radio promedio")
        st.pyplot(fig)
        st.caption("Los tumores malignos tienen radios mayores, pero las cajas se solapan.")

    st.subheader("Radio promedio frente a textura promedio")
    fig, ax = plt.subplots(figsize=(7, 4))
    for d, color in [("Benigno", "#4C78A8"), ("Maligno", "#E45756")]:
        m = etiquetas_dx == d
        ax.scatter(datos.loc[m, "mean_radius"], datos.loc[m, "mean_texture"],
                   s=14, alpha=0.6, label=d, color=color)
    ax.set_xlabel("Radio promedio")
    ax.set_ylabel("Textura promedio")
    ax.legend()
    st.pyplot(fig)
    st.caption("Hay zonas donde las clases se separan y otras donde se mezclan, "
               "por eso el modelo usa varias variables a la vez.")

with tab_pred:
    st.header("Predicción")
    st.write("Ingresa las 30 medidas del tumor o usa uno de los ejemplos. "
             "Los valores se validan antes de predecir.")

    if "in_" + COLUMNAS[0] not in st.session_state:
        fijar_valores({c: rangos[c]["mediana"] for c in COLUMNAS})

    b1, b2, _ = st.columns([1, 1, 2])
    b1.button("Cargar caso normal (medianas)", width="stretch",
              on_click=fijar_valores, args=({c: rangos[c]["mediana"] for c in COLUMNAS},))
    b2.button("Cargar registro conocido de prueba", width="stretch",
              on_click=fijar_valores, args=(registro["valores"],))

    with st.form("formulario"):
        pestañas = st.tabs([f"{GRUPOS[g]}" for g in GRUPOS])
        entradas = {}
        for pestaña, g in zip(pestañas, GRUPOS):
            with pestaña:
                cols_g = [c for c in COLUMNAS
                          if (c.startswith("mean_") and g == "mean")
                          or (c.startswith("worst_") and g == "worst")
                          or (c.endswith("_error") and g == "error")]
                fila = st.columns(2)
                for i, c in enumerate(cols_g):
                    r = rangos[c]
                    entradas[c] = fila[i % 2].number_input(
                        etiqueta(c), key=f"in_{c}", format="%.5f", step=None,
                        help=f"Rango observado: {r['min']:.4g} a {r['max']:.4g}")
        enviar = st.form_submit_button("Predecir", type="primary")

    if enviar:
        valores = {c: entradas[c] for c in COLUMNAS}
        errores, advertencias = validar_entrada(valores, rangos)
        if errores:
            st.error("No se puede predecir. Corrige estos valores:")
            for e in errores:
                st.markdown(f"- {e}")
        else:
            for a in advertencias:
                st.warning(a)
            clase, prob = predecir(modelo, valores, COLUMNAS)
            nombre_clase = "Maligno" if clase == 1 else "Benigno"
            (st.error if clase == 1 else st.success)(f"Predicción: **{nombre_clase}**")
            m1, m2 = st.columns(2)
            m1.metric("Probabilidad de maligno", f"{prob:.1%}")
            m2.metric("Probabilidad de benigno", f"{1 - prob:.1%}")
            st.progress(prob)
            st.caption("La probabilidad la calcula `predict_proba()` de la regresión logística; "
                       "no equivale a una probabilidad clínica real.")

    with st.expander("Comprobación contra el notebook (registro conocido)"):
        st.write(f"Registro número **{registro['indice_en_dataset_limpio']}** del dataset limpio, "
                 "del conjunto de prueba (`random_state=42`).")
        st.write(f"Clase real: **{registro['clase_real']}**. "
                 f"Predicción guardada al entrenar: **{registro['prediccion']}** "
                 f"(prob. maligno {registro['probabilidad_maligno']:.4f}).")
        clase_k, prob_k = predecir(modelo, registro["valores"], COLUMNAS)
        ok = (("Maligno" if clase_k else "Benigno") == registro["prediccion"]
              and abs(prob_k - registro["probabilidad_maligno"]) < 1e-9)
        st.write(f"Predicción de la app para el mismo registro: "
                 f"**{'Maligno' if clase_k else 'Benigno'}** (prob. maligno {prob_k:.4f}) → "
                 + ("✅ coincide" if ok else "❌ NO coincide"))

with tab_met:
    st.header("Métricas finales")
    st.write(f"Partición estratificada 80/20 (`random_state=42`): "
             f"{metricas['n_train']} registros de entrenamiento y {metricas['n_test']} de prueba. "
             "La clase positiva es **maligno**.")

    filas = []
    for nombre, m in metricas["modelos"].items():
        p = m["prueba"]
        filas.append({"Modelo": nombre, "Accuracy": p["accuracy"], "Precision": p["precision"],
                      "Recall": p["recall"], "F1": p["f1"], "AUC": p["auc"],
                      "Falsos negativos": p["falsos_negativos"],
                      "Falsos positivos": p["falsos_positivos"]})
    st.subheader("Conjunto de prueba")
    st.dataframe(pd.DataFrame(filas).set_index("Modelo").style.format(
        {"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
         "F1": "{:.3f}", "AUC": "{:.3f}"}), width="stretch")

    st.subheader("Validación cruzada (5 folds estratificados)")
    filas = []
    for nombre, m in metricas["modelos"].items():
        fila = {"Modelo": nombre}
        for k, v in m["validacion_cruzada_5_folds"].items():
            fila[k.capitalize()] = f"{v['media']:.3f} ± {v['desviacion']:.3f}"
        filas.append(fila)
    st.dataframe(pd.DataFrame(filas).set_index("Modelo"), width="stretch")

    st.subheader("Matrices de confusión")
    cols = st.columns(len(metricas["modelos"]))
    for col, (nombre, m) in zip(cols, metricas["modelos"].items()):
        with col:
            cm = np.array(m["matriz_confusion"])
            fig, ax = plt.subplots(figsize=(4, 3.4))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1], ["Benigno", "Maligno"])
            ax.set_yticks([0, 1], ["Benigno", "Maligno"])
            ax.set_xlabel("Predicho")
            ax.set_ylabel("Real")
            ax.set_title(nombre, fontsize=10)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            st.pyplot(fig)

    st.subheader("Por qué se eligió la regresión logística")
    st.markdown(
        "- Tiene mejor accuracy, precision, recall, F1 y AUC que el árbol.\n"
        "- Deja **menos falsos negativos** (tumores malignos clasificados como benignos), "
        "que es el error más grave en este problema.\n"
        "- Su resultado es más estable en la validación cruzada.\n"
        "- El escalado está dentro del `Pipeline`, así que se guarda y se usa completo.")

    st.subheader("Limitaciones")
    st.markdown(
        "- Los datos son pocos (569) y vienen de un contexto específico; no hay evidencia de que "
        "el modelo funcione igual en otros hospitales ni con población colombiana.\n"
        "- Aun con el mejor modelo hay falsos negativos; el recall de validación cruzada varía "
        "entre folds.\n"
        "- Las variables de tamaño (radio, perímetro, área) están muy correlacionadas, lo que "
        "dificulta interpretar los coeficientes.\n"
        "- La evaluación usa una sola partición y validación cruzada interna; no hay validación "
        "externa.\n"
        "- Las probabilidades no están calibradas clínicamente.\n"
        "- Es un prototipo académico: no sustituye una decisión profesional.")
