# Proyecto: clasificación de tumores de mama

**Asignatura:** Minería de Datos  
**Universidad:** Universidad de la Costa  
**Entrega:** tercer corte

## ¿De qué trata el proyecto?

En este trabajo estamos desarrollando una aplicación que recibe varias medidas de las células de un tumor y usa un modelo de aprendizaje automático para clasificar el caso como benigno o maligno.

Para hacerlo usamos el conjunto de datos *Breast Cancer Wisconsin (Diagnostic)*, que viene incluido en la librería scikit-learn. Tiene 569 registros y 30 características numéricas. Comparamos dos modelos: regresión logística y árbol de decisión. Para la aplicación dejamos la regresión logística.

Es importante aclarar que este proyecto es un ejercicio académico. No sirve para dar diagnósticos médicos ni reemplaza la opinión de un profesional de la salud.

## Archivos del proyecto

- `app.py`: contiene la aplicación que se abre con Streamlit.
- `entrenar_modelo.py`: prepara los datos, entrena los modelos y guarda sus resultados.
- `pruebas.py`: ejecuta unas comprobaciones para revisar que la aplicación funcione.
- `avance_2_mineria.ipynb`: notebook donde se trabajó la exploración de los datos y la comparación de modelos.
- `requirements.txt`: lista las librerías que se necesitan.
- `artefactos/`: guarda el modelo entrenado, las métricas, los rangos de las variables y los datos limpios.

## Aplicación pública: https://proyecto-streamli-dvarela8.streamlit.app

## Cómo ejecutar el proyecto

Primero abre una terminal en la carpeta del proyecto y ejecuta:

```bash
pip install -r requirements.txt
python entrenar_modelo.py
streamlit run app.py
```

Si quieres ejecutar las pruebas automáticas, usa:

```bash
python pruebas.py
```

El archivo `requirements.txt` fija la versión de scikit-learn usada para guardar el modelo. Es mejor instalar las dependencias desde ese archivo para evitar problemas al cargarlo.

## ¿Qué revisa la aplicación?

Antes de hacer una predicción, la aplicación comprueba que los datos sean números válidos y que no sean negativos ni demasiado grandes. Si un valor está fuera del rango que aparece en los datos originales, muestra una advertencia. También avisa cuando el peor valor de radio, perímetro o área queda por debajo del promedio, porque no es lo habitual.

La aplicación tiene secciones para presentar el proyecto, explorar los datos, ingresar medidas y consultar las métricas de los modelos.

## Resultados y limitaciones

Se compararon la regresión logística y el árbol de decisión usando datos de entrenamiento y de prueba. También se hizo validación cruzada para revisar si los resultados se mantienen al cambiar la división de los datos.

Aunque las métricas ayudan a comparar los modelos, no significan que el sistema esté listo para utilizarse en un hospital. Los datos vienen de un conjunto específico, la cantidad de registros es limitada y pueden existir diferencias con otros grupos de pacientes. Además, todavía pueden ocurrir errores, incluidos casos malignos clasificados como benignos.

## Uso de inteligencia artificial

**Herramienta utilizada:** ChatGPT (OpenAI).

**¿Para qué la usamos?** La usamos como apoyo para entender cómo organizar el proyecto, revisar el código y mejorar la explicación de algunas partes. No tomamos sus respuestas como si fueran correctas automáticamente; las revisamos y las comparamos con lo que realmente hacía el programa.

**Prompt utilizado:**

Estoy trabajando en un proyecto de Minería de Datos para la universidad. El tema es clasificar tumores de mama como benignos o malignos usando el conjunto de datos Breast Cancer Wisconsin que trae scikit-learn. Ayúdame a revisar los archivos del proyecto y a organizar el código para que se pueda ejecutar en Streamlit. Necesito que expliques las partes de forma sencilla, como para un estudiante que está aprendiendo el tema. También quiero que revises la preparación de los datos, la comparación entre regresión logística y árbol de decisión, las métricas y las pruebas de la aplicación. No inventes resultados: si algo no se puede comprobar con los archivos, déjalo señalado para revisarlo.

**¿Cómo verificamos la respuesta?**

Después de recibir la ayuda, revisé el proyecto ejecutando los archivos. Primero ejecuté `entrenar_modelo.py` para volver a preparar los datos y entrenar los modelos. Después comparé las métricas que se guardaron con las que aparecen en el notebook. También ejecuté `pruebas.py`, que revisa tres situaciones: un caso normal, un dato inválido y un registro que ya conozco. Por último, comprobé que la aplicación mostrara las clases como “Benigno” y “Maligno”, y no solamente como números.

Durante la revisión encontré algunos detalles que se debian tener en cuenta. Por ejemplo, en el conjunto original la clase 0 corresponde a maligno y la clase 1 a benigno, pero para evaluar el modelo se cambió la codificación para que maligno quedara como la clase positiva. También tuvé que revisar cómo se interpreta la probabilidad que entrega el modelo.

**Conclusión sobre el uso de IA:** la herramienta me sirvió como apoyo, pero la responsabilidad de revisar el código y los resultados sigue siendo mia. No se debe entregar una explicación o un resultado solo porque lo haya dicho la IA.
