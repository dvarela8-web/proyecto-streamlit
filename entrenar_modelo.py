import json
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SALIDA = Path("artefactos")
SALIDA.mkdir(exist_ok=True)

df = load_breast_cancer(as_frame=True).frame.copy()
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))
df = df.drop_duplicates().reset_index(drop=True)
df.to_csv(SALIDA / "datos_cancer_mama_limpios.csv", index=False)

X = df.drop(columns="target")
y = (df["target"] == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

modelos = {
    "Regresión logística": Pipeline([
        ("escalado", StandardScaler()),
        ("modelo", LogisticRegression(max_iter=2000, random_state=42)),
    ]),
    "Árbol de decisión": DecisionTreeClassifier(max_depth=4, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metricas = {"sklearn": sklearn.__version__, "n_train": len(X_train),
            "n_test": len(X_test), "modelos": {}}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    prob = modelo.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    cvr = cross_validate(modelo, X, y, cv=cv,
                         scoring=["accuracy", "precision", "recall", "f1"])
    metricas["modelos"][nombre] = {
        "prueba": {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "auc": roc_auc_score(y_test, prob),
            "falsos_negativos": int(cm[1, 0]),
            "falsos_positivos": int(cm[0, 1]),
        },
        "validacion_cruzada_5_folds": {
            m: {"media": float(cvr[f"test_{m}"].mean()),
                "desviacion": float(cvr[f"test_{m}"].std())}
            for m in ["accuracy", "precision", "recall", "f1"]
        },
        "matriz_confusion": cm.tolist(),
    }

final = modelos["Regresión logística"]
joblib.dump(final, SALIDA / "modelo.joblib")

rangos = {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()),
              "mediana": float(X_train[c].median())} for c in X.columns}
(SALIDA / "rangos.json").write_text(json.dumps(rangos, indent=2), encoding="utf-8")

idx = int(X_test.index[0])
fila = X_test.iloc[[0]]
registro = {
    "indice_en_dataset_limpio": idx,
    "valores": {c: float(v) for c, v in fila.iloc[0].items()},
    "clase_real": "Maligno" if int(y_test.iloc[0]) == 1 else "Benigno",
    "prediccion": "Maligno" if int(final.predict(fila)[0]) == 1 else "Benigno",
    "probabilidad_maligno": float(final.predict_proba(fila)[0, 1]),
}
(SALIDA / "registro_conocido.json").write_text(
    json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
(SALIDA / "metricas.json").write_text(
    json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

print("Listo. scikit-learn", sklearn.__version__)
for n, m in metricas["modelos"].items():
    p = m["prueba"]
    print(f"{n}: accuracy={p['accuracy']:.4f} recall={p['recall']:.4f} "
          f"FN={p['falsos_negativos']} FP={p['falsos_positivos']}")
print("Registro conocido:", registro["prediccion"],
      f"(prob. maligno = {registro['probabilidad_maligno']:.4f})")
