from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


def gerar_visualizacoes(
    df: pd.DataFrame,
    resultados: pd.DataFrame,
    importancia: pd.DataFrame,
    output_dir: str = "reports",
):
    """
    Gera visualizações para análise e documentação
    do modelo de detecção de fraude.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # 1. Distribuição das classes
    # ----------------------------------------

    contagem = (
        df["fraude"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    contagem.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title("Distribuição das Transações")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade")

    ax.set_xticklabels(
        ["Legítima", "Fraude"],
        rotation=0,
    )

    fig.tight_layout()

    fig.savefig(
        output / "distribuicao_fraudes.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ----------------------------------------
    # 2. Matriz de confusão
    # ----------------------------------------

    matriz = confusion_matrix(
        resultados["fraude_real"],
        resultados["fraude_prevista"],
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    imagem = ax.imshow(matriz)

    ax.set_title("Matriz de Confusão")
    ax.set_xlabel("Classe Prevista")
    ax.set_ylabel("Classe Real")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Legítima", "Fraude"]
    )

    ax.set_yticklabels(
        ["Legítima", "Fraude"]
    )

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(
                j,
                i,
                matriz[i, j],
                ha="center",
                va="center",
            )

    fig.colorbar(
        imagem,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        output / "matriz_confusao.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ----------------------------------------
    # 3. Importância das variáveis
    # ----------------------------------------

    top_features = (
        importancia
        .sort_values(
            "importancia",
            ascending=True,
        )
        .tail(10)
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        top_features["variavel"],
        top_features["importancia"],
    )

    ax.set_title(
        "Importância das Variáveis"
    )

    ax.set_xlabel("Importância")

    fig.tight_layout()

    fig.savefig(
        output / "importancia_variaveis.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ----------------------------------------
    # 4. Probabilidade estimada de fraude
    # ----------------------------------------

    legitimas = resultados[
        resultados["fraude_real"] == 0
    ]["probabilidade_fraude"]

    fraudes = resultados[
        resultados["fraude_real"] == 1
    ]["probabilidade_fraude"]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        legitimas,
        bins=30,
        alpha=0.6,
        label="Legítimas",
    )

    ax.hist(
        fraudes,
        bins=30,
        alpha=0.6,
        label="Fraudes",
    )

    ax.set_title(
        "Distribuição da Probabilidade de Fraude"
    )

    ax.set_xlabel(
        "Probabilidade prevista"
    )

    ax.set_ylabel("Quantidade")

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output / "probabilidade_fraude.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)
    
        # ----------------------------------------
    # 5. Distribuição por nível de risco
    # ----------------------------------------

    ordem_risco = [
        "Baixo",
        "Moderado",
        "Alto",
        "Critico",
    ]

    distribuicao_risco = (
        resultados["nivel_risco"]
        .value_counts()
        .reindex(ordem_risco)
        .fillna(0)
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    distribuicao_risco.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Distribuição das Transações por Nível de Risco"
    )

    ax.set_xlabel(
        "Nível de risco"
    )

    ax.set_ylabel(
        "Quantidade de transações"
    )

    ax.set_xticklabels(
        ordem_risco,
        rotation=0,
    )

    fig.tight_layout()

    fig.savefig(
        output / "distribuicao_nivel_risco.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)