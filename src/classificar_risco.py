import pandas as pd


def classificar_nivel_risco(probabilidade: float) -> str:
    """
    Classifica o score do modelo em uma faixa
    experimental de risco.
    """

    if probabilidade >= 0.50:
        return "Critico"

    if probabilidade >= 0.30:
        return "Alto"

    if probabilidade >= 0.20:
        return "Moderado"

    return "Baixo"


def adicionar_nivel_risco(
    resultados: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adiciona uma classificação de risco
    às transações analisadas.
    """

    dados = resultados.copy()

    dados["nivel_risco"] = (
        dados["probabilidade_fraude"]
        .apply(classificar_nivel_risco)
    )

    return dados