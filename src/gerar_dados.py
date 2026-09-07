import numpy as np
import pandas as pd


def gerar_transacoes(n=12000, seed=42):
    """Gera um dataset sintético de transações bancárias."""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "id_transacao": [f"TX{i:06d}" for i in range(1, n + 1)],
        "valor": rng.lognormal(mean=3.6, sigma=1.0, size=n).round(2),
        "hora": rng.integers(0, 24, size=n),
        "distancia_casa_km": rng.gamma(shape=2.0, scale=8.0, size=n).round(2),
        "transacoes_1h": rng.poisson(lam=1.5, size=n),
        "novo_dispositivo": rng.binomial(1, 0.10, size=n),
        "transacao_internacional": rng.binomial(1, 0.06, size=n),
        "cartao_presente": rng.binomial(1, 0.72, size=n),
        "media_valor_cliente": rng.lognormal(mean=3.4, sigma=0.65, size=n).round(2),
    })

    risco = (
        1.0 * (df["valor"] > 250).astype(int)
        + 1.0 * ((df["hora"] <= 4) | (df["hora"] >= 23)).astype(int)
        + 1.0 * (df["distancia_casa_km"] > 45).astype(int)
        + 1.0 * (df["transacoes_1h"] >= 5).astype(int)
        + 0.8 * df["novo_dispositivo"]
        + 0.8 * df["transacao_internacional"]
        + 0.5 * (df["cartao_presente"] == 0).astype(int)
        + 1.0 * (df["valor"] > 3.5 * df["media_valor_cliente"]).astype(int)
    )

    prob_fraude = np.clip(0.004 + risco * 0.035, 0, 0.75)
    df["fraude"] = rng.binomial(1, prob_fraude)

    return df
