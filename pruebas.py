import json
from pathlib import Path

import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest

BASE = Path(__file__).parent
APP = str(BASE / "app.py")
registro = json.loads((BASE / "artefactos" / "registro_conocido.json").read_text(encoding="utf-8"))


def nueva_app():
    return AppTest.from_file(APP, default_timeout=60).run()


def enviar(at):
    at.button[-1].click().run()


at = nueva_app()
at.button[0].click().run()
enviar(at)
assert not at.exception, at.exception
assert any("Predicción:" in x.value for x in list(at.success) + list(at.error)), "no hubo predicción"
print("OK 1. Caso normal:", [m.value for m in at.metric])

at = nueva_app()
at.number_input[0].set_value(-5.0)
enviar(at)
assert any("No se puede predecir" in e.value for e in at.error)
assert not at.metric, "no debería mostrar probabilidades con entrada inválida"
print("OK 2. Entrada inválida bloqueada")

at = nueva_app()
at.button[1].click().run()
enviar(at)
modelo = joblib.load(BASE / "artefactos" / "modelo.joblib")
X = pd.DataFrame([registro["valores"]])
prob_nb = float(modelo.predict_proba(X)[0, 1])
clase_nb = "Maligno" if modelo.predict(X)[0] == 1 else "Benigno"
texto = [x.value for x in list(at.success) + list(at.error)]
assert any(clase_nb in t for t in texto), (clase_nb, texto)
assert f"{prob_nb:.1%}" in [m.value for m in at.metric]
print(f"OK 3. Registro conocido: {clase_nb}, prob. maligno {prob_nb:.4f} (coincide con el notebook)")
print("Todas las pruebas pasaron.")
