from pathlib import Path

from src.avaliar_thresholds import avaliar_thresholds
from src.classificar_risco import adicionar_nivel_risco
from src.gerar_dados import gerar_transacoes
from src.gerar_visualizacoes import gerar_visualizacoes
from src.treinar_modelo import treinar


def main():
    # -------------------------------------------------
    # Criar diretórios
    # -------------------------------------------------

    Path("data").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # -------------------------------------------------
    # 1. Gerar dataset sintético
    # -------------------------------------------------

    df = gerar_transacoes()

    print("=== PRIMEIRAS LINHAS ===")
    print(df.head())

    print("\n=== DIMENSÕES DO DATASET ===")
    print(df.shape)

    print("\n=== DISTRIBUIÇÃO DA VARIÁVEL FRAUDE ===")
    print(df["fraude"].value_counts())

    print("\n=== DISTRIBUIÇÃO PERCENTUAL ===")
    print(
        (
            df["fraude"]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    # Salvar dataset
    df.to_csv(
        "data/transacoes.csv",
        index=False,
    )

    # -------------------------------------------------
    # 2. Treinar modelo
    # -------------------------------------------------

    modelo, resultados, importancia = treinar(df)

    # -------------------------------------------------
    # 3. Classificação por nível de risco
    # -------------------------------------------------

    resultados = adicionar_nivel_risco(
        resultados
    )

    print(
        "\n=== DISTRIBUIÇÃO POR NÍVEL DE RISCO ==="
    )

    distribuicao_risco = (
        resultados["nivel_risco"]
        .value_counts()
    )

    print(distribuicao_risco)

    print(
        "\n=== DISTRIBUIÇÃO PERCENTUAL DE RISCO ==="
    )

    print(
        (
            resultados["nivel_risco"]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    resultados.to_csv(
        "reports/transacoes_classificadas_risco.csv",
        index=False,
    )

    # -------------------------------------------------
    # 4. Avaliar diferentes thresholds
    # -------------------------------------------------

    comparacao_thresholds = avaliar_thresholds(
        resultados
    )

    print(
        "\n=== COMPARAÇÃO DE THRESHOLDS ==="
    )

    print(
        comparacao_thresholds.to_string(
            index=False
        )
    )

    comparacao_thresholds.to_csv(
        "reports/comparacao_thresholds.csv",
        index=False,
    )

    # -------------------------------------------------
    # 5. Transações suspeitas
    # Threshold baseline = 0.50
    # -------------------------------------------------

    suspeitas = (
        resultados[
            resultados["probabilidade_fraude"]
            >= 0.50
        ]
        .sort_values(
            "probabilidade_fraude",
            ascending=False,
        )
    )

    suspeitas.to_csv(
        "reports/transacoes_suspeitas.csv",
        index=False,
    )

    print(
        "\n=== TRANSAÇÕES SUSPEITAS "
        "(THRESHOLD 0.50) ==="
    )

    print(
        f"Quantidade de alertas: "
        f"{len(suspeitas)}"
    )

    # -------------------------------------------------
    # 6. Importância das variáveis
    # -------------------------------------------------

    importancia.to_csv(
        "reports/importancia_variaveis.csv",
        index=False,
    )

    print(
        "\n=== TOP 10 VARIÁVEIS "
        "MAIS IMPORTANTES ==="
    )

    print(
        importancia
        .head(10)
        .to_string(index=False)
    )

    # -------------------------------------------------
    # 7. Gerar visualizações
    # -------------------------------------------------

    gerar_visualizacoes(
        df=df,
        resultados=resultados,
        importancia=importancia,
    )

    # -------------------------------------------------
    # 8. Resumo operacional de risco
    # -------------------------------------------------

    print(
        "\n=== ESTRATÉGIA EXPERIMENTAL "
        "DE RISCO ==="
    )

    print(
        "Baixo     : score < 0.20"
    )

    print(
        "Moderado  : 0.20 <= score < 0.30"
    )

    print(
        "Alto      : 0.30 <= score < 0.50"
    )

    print(
        "Critico   : score >= 0.50"
    )

    print(
        "\nObservação: as faixas representam "
        "scores experimentais do modelo e não "
        "probabilidades calibradas de fraude."
    )

    # -------------------------------------------------
    # 9. Arquivos gerados
    # -------------------------------------------------

    print("\n=== ARQUIVOS GERADOS ===")

    print(
        "- data/transacoes.csv"
    )

    print(
        "- reports/modelo_fraude.joblib"
    )

    print(
        "- reports/transacoes_suspeitas.csv"
    )

    print(
        "- reports/transacoes_classificadas_risco.csv"
    )

    print(
        "- reports/importancia_variaveis.csv"
    )

    print(
        "- reports/comparacao_thresholds.csv"
    )

    print(
        "- reports/distribuicao_fraudes.png"
    )

    print(
        "- reports/matriz_confusao.png"
    )

    print(
        "- reports/importancia_variaveis.png"
    )

    print(
        "- reports/probabilidade_fraude.png"
    )
    print(
    "- reports/distribuicao_nivel_risco.png"
)

if __name__ == "__main__":
    main()