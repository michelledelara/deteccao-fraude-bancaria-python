# Detecção de Fraude Bancária com Python e Pandas

Projeto de portfólio para análise e detecção de transações bancárias potencialmente fraudulentas usando **Python**, **Pandas** e **Machine Learning**.

O foco do projeto é demonstrar, de forma prática:
- tratamento e exploração de dados com `pandas`;
- identificação de dados desbalanceados;
- criação de variáveis para análise de risco;
- treinamento de modelo de classificação;
- avaliação com métricas adequadas para fraude;
- geração de relatório de transações suspeitas.

## Tecnologias

- Python 3.11+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Pytest

## Estrutura do projeto

```text
deteccao-fraude-bancaria-python/
├── .github/
│   └── workflows/
│       └── python-ci.yml
├── data/
│   └── .gitkeep
├── notebooks/
│   └── analise_exploratoria.ipynb
├── reports/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── gerar_dados.py
│   ├── preparar_dados.py
│   └── treinar_modelo.py
├── tests/
│   └── test_preparar_dados.py
├── .gitignore
├── LICENSE
├── main.py
├── requirements.txt
└── README.md
```

## Problema de negócio

Fraudes bancárias costumam representar uma pequena parcela das transações. Isso cria um problema de **classificação desbalanceada**.

Por exemplo: se apenas 1% das transações forem fraudulentas, um modelo que classifique tudo como legítimo pode atingir 99% de acurácia e ainda assim ser inútil.

Por isso, este projeto dá mais atenção a:

- **Precision**: entre as transações classificadas como fraude, quantas realmente eram fraude;
- **Recall**: entre todas as fraudes reais, quantas o modelo conseguiu identificar;
- **F1-score**: equilíbrio entre precision e recall;
- **ROC-AUC**: capacidade geral de separação entre as classes.

## Variáveis analisadas

O dataset sintético contém:

- `valor`
- `hora`
- `distancia_casa_km`
- `transacoes_1h`
- `novo_dispositivo`
- `transacao_internacional`
- `cartao_presente`
- `media_valor_cliente`
- `fraude`

Também são criadas variáveis derivadas:

- `valor_log`
- `madrugada`
- `valor_vs_media_cliente`

## Como executar

### 1. Criar ambiente virtual

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar o projeto

```bash
python main.py
```

O projeto irá:

1. gerar um dataset sintético de transações;
2. salvar os dados em `data/transacoes.csv`;
3. fazer a preparação dos dados com Pandas;
4. treinar um modelo Random Forest;
5. mostrar as métricas no terminal;
6. salvar as transações suspeitas em `reports/transacoes_suspeitas.csv`;
7. salvar a importância das variáveis em `reports/importancia_variaveis.csv`.

## Exemplo de uso com Pandas

```python
import pandas as pd

df = pd.read_csv("data/transacoes.csv")

print(df.head())
print(df.info())
print(df["fraude"].value_counts(normalize=True))
```

## Observação importante

Este projeto utiliza **dados sintéticos**, criados apenas para fins educacionais e de portfólio. Ele não deve ser utilizado diretamente em ambiente bancário real sem validação, governança, controles e monitoramento adequados.

## Melhorias futuras

- SMOTE para balanceamento;
- XGBoost ou LightGBM;
- Isolation Forest para anomalias;
- SHAP para explicabilidade;
- Streamlit para dashboard;
- API com FastAPI;
- Docker;
- monitoramento de drift;
- integração com AWS/Azure.

## Autora

**Michelle Ferraz**

Governança de TI | Dados | Cloud Computing | GRC | Privacidade | Segurança
