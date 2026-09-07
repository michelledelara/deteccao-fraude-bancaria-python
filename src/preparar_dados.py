import numpy as np
import pandas as pd


COLUNAS_MODELO = [
    "valor",
    "hora",
    "distancia_casa_km",
    "transacoes_1h",
    "novo_dispositivo",
    "transacao_internacional",
    "cartao_presente",
    "media_valor_cliente",
    "valor_log",
    "madrugada",
    "valor_vs_media_cliente",
]


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis derivadas usando Pandas."""
    dados = df.copy()

    dados["valor_log"] = np.log1p(dados["valor"])
    dados["madrugada"] = ((dados["hora"] <= 4) | (dados["hora"] >= 23)).astype(int)
    dados["valor_vs_media_cliente"] = (
        dados["valor"] / dados["media_valor_cliente"].replace(0, np.nan)
    ).fillna(0)

    return dados


def separar_x_y(df: pd.DataFrame):
    dados = criar_features(df)
    X = dados[COLUNAS_MODELO]
    y = dados["fraude"]
    return X, y
