from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.preparar_dados import separar_x_y


def treinar(df: pd.DataFrame, model_path="reports/modelo_fraude.joblib"):
    X, y = separar_x_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    modelo = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    modelo.fit(X_train, y_train)

    prob = modelo.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.50).astype(int)

    print("\n=== MATRIZ DE CONFUSÃO ===")
    print(confusion_matrix(y_test, pred))

    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
    print(classification_report(y_test, pred, digits=4))

    print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, model_path)

    resultados = X_test.copy()
    resultados["fraude_real"] = y_test.values
    resultados["probabilidade_fraude"] = prob
    resultados["fraude_prevista"] = pred

    importancia = pd.DataFrame({
        "variavel": X.columns,
        "importancia": modelo.feature_importances_,
    }).sort_values("importancia", ascending=False)

    return modelo, resultados, importancia
