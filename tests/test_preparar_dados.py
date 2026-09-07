import pandas as pd

from src.preparar_dados import criar_features


def test_criar_features():
    df = pd.DataFrame({
        "valor": [100.0],
        "hora": [2],
        "distancia_casa_km": [10.0],
        "transacoes_1h": [1],
        "novo_dispositivo": [0],
        "transacao_internacional": [0],
        "cartao_presente": [1],
        "media_valor_cliente": [50.0],
        "fraude": [0],
    })

    resultado = criar_features(df)

    assert "valor_log" in resultado.columns
    assert "madrugada" in resultado.columns
    assert "valor_vs_media_cliente" in resultado.columns
    assert resultado.loc[0, "madrugada"] == 1
    assert resultado.loc[0, "valor_vs_media_cliente"] == 2.0
