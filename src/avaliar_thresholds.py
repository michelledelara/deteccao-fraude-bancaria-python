import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def avaliar_thresholds(
    resultados: pd.DataFrame,
    thresholds=None,
) -> pd.DataFrame:
    """
    Compara diferentes thresholds para classificação de fraude.
    """

    if thresholds is None:
        thresholds = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10]

    y_real = resultados["fraude_real"]
    probabilidades = resultados["probabilidade_fraude"]

    comparacao = []

    for threshold in thresholds:

        previsoes = (
            probabilidades >= threshold
        ).astype(int)

        precision = precision_score(
            y_real,
            previsoes,
            zero_division=0,
        )

        recall = recall_score(
            y_real,
            previsoes,
            zero_division=0,
        )

        f1 = f1_score(
            y_real,
            previsoes,
            zero_division=0,
        )

        tn, fp, fn, tp = confusion_matrix(
            y_real,
            previsoes,
        ).ravel()

        comparacao.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "fraudes_detectadas": tp,
                "fraudes_nao_detectadas": fn,
                "falsos_positivos": fp,
                "verdadeiros_negativos": tn,
            }
        )

    return pd.DataFrame(comparacao)