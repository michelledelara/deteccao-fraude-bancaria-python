from pathlib import Path

from src.gerar_dados import gerar_transacoes
from src.treinar_modelo import treinar


def main():
    Path("data").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    df = gerar_transacoes()

    print("=== PRIMEIRAS LINHAS ===")
    print(df.head())

    print("\n=== DIMENSÕES ===")
    print(df.shape)

    print("\n=== DISTRIBUIÇÃO DA VARIÁVEL FRAUDE ===")
    print(df["fraude"].value_counts())
    print("\nPercentual:")
    print((df["fraude"].value_counts(normalize=True) * 100).round(2))

    df.to_csv("data/transacoes.csv", index=False)

    modelo, resultados, importancia = treinar(df)

    suspeitas = (
        resultados[resultados["probabilidade_fraude"] >= 0.50]
        .sort_values("probabilidade_fraude", ascending=False)
    )

    suspeitas.to_csv("reports/transacoes_suspeitas.csv", index=False)
    importancia.to_csv("reports/importancia_variaveis.csv", index=False)

    print("\n=== TOP 10 VARIÁVEIS MAIS IMPORTANTES ===")
    print(importancia.head(10).to_string(index=False))

    print("\nArquivos gerados:")
    print("- data/transacoes.csv")
    print("- reports/modelo_fraude.joblib")
    print("- reports/transacoes_suspeitas.csv")
    print("- reports/importancia_variaveis.csv")


if __name__ == "__main__":
    main()
