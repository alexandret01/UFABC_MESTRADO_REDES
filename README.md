# Detecção Preditiva de Falhas em Redes Ópticas
### Mestrado UFABC — Redes de Comunicação

Pipeline completo de geração de base de dados e análise preditiva multiclasse de falhas em redes de comunicação óptica, utilizando Machine Learning e Deep Learning.

---

## Estrutura do Projeto

```
UFABC_MESTRADO_REDES/
├── Artigos/                          # Artigos científicos de referência
│   ├── 1-s2.0-S1389128625001276-main.pdf
│   ├── Learning_Long-_and_Short-Term_Temporal_Patterns_(...).pdf
│   └── Reliable_and_scalable_Kafka-based_framework_(...).pdf
│
├── Datasets/                         # Datasets de entrada (não versionados)
│   ├── HardFailure_dataset.csv
│   ├── SoftFailure_dataset.csv
│   ├── Lightpath_756_label_4_QoT_dataset_train_900.parquet
│   └── output/
│       ├── dataset_unificado.csv
│       └── dataset_unificado.parquet
│
├── geracao_dataset/                  # Pipeline de geração da base unificada
│   ├── geracao_dataset.ipynb         # Notebook principal
│   └── README.md                     # Documentação do pipeline
│
├── analise_dados/                    # Pipeline de análise e modelagem
│   ├── analise_dados.ipynb           # Notebook principal
│   ├── ANALISE_COMPLETA.md           # Relatório detalhado de resultados
│   └── modelos_salvos/               # Modelos treinados (gerado na execução)
│
├── requirements.txt                  # Dependências do projeto
└── README.md                         # Este arquivo
```

---

## Pré-requisitos

- Python 3.10+
- Anaconda ou Miniconda (recomendado)
- ~4 GB de RAM livre para o pipeline de geração
- ~8 GB de RAM livre para análise com 5M linhas

---

## Instalação

### 1. Criar e ativar o ambiente conda

```bash
conda create -n UFABC_MESTRADO_REDES python=3.11 -y
conda activate UFABC_MESTRADO_REDES
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

> **Nota:** O FLAML 2.x requer `lightgbm` e `xgboost` instalados separadamente para habilitar o AutoML. Ambos estão incluídos no `requirements.txt`.

---

## Ordem de Execução

Execute os notebooks **nesta ordem**:

### Etapa 1 — Geração da Base de Dados

```
geracao_dataset/geracao_dataset.ipynb
```

Gera o arquivo `Datasets/output/dataset_unificado.parquet` com 5.000.000 linhas unificando os três datasets.

> Tempo estimado: **30–90 minutos** (dominado pelo IterativeImputer)

**O que faz:**
1. Carrega Hard Failure, Soft Failure e Lightpath QoT
2. Padroniza esquema de colunas e converte timestamps
3. Interpolação linear intra-série + imputação ML (IterativeImputer)
4. Aumento de dados: SMOTE + ruído gaussiano até 5M linhas
5. Exporta CSV e Parquet

**Configuração** (célula "Configuração Global"):
```python
TARGET_ROWS    = 5_000_000   # tamanho-alvo da base
IMPUTER_SAMPLE = 200_000     # linhas para treinar o imputer
NOISE_SCALE    = 0.01        # magnitude do ruído gaussiano
```

---

### Etapa 2 — Análise e Modelagem

```
analise_dados/analise_dados.ipynb
```

Realiza EDA completa, engenharia de features e treina/avalia 4 modelos.

> Tempo estimado: **60–180 minutos** (dominado pelo AutoML e LSTM)

**O que faz:**
1. EDA: distribuição de classes, boxplots, correlação, série temporal, outliers, validação KS
2. Engenharia de features: rolling window, lags, slope, hora cíclica (~81 features)
3. Modelagem: Random Forest (baseline), RF (features selecionadas), AutoML (FLAML), LSTM
4. Avaliação: matrizes de confusão, ROC, Precision-Recall, aprendizado, SHAP, threshold
5. Salvamento dos modelos em `analise_dados/modelos_salvos/`

**Configuração** (célula "Configuração Global"):
```python
RANDOM_STATE  = 42
SAMPLE_SIZE   = 500_000   # amostra para modelagem
AUTOML_TIME   = 120       # segundos de busca do AutoML
```

---

## Datasets

| Arquivo | Formato | Linhas | Descrição |
|---|---|---|---|
| `HardFailure_dataset.csv` | CSV | ~65.733 | Falhas abruptas de sinal |
| `SoftFailure_dataset.csv` | CSV | ~53.697 | Degradações graduais |
| `Lightpath_756_...parquet` | Parquet | ~2.721.600 | QoT com 4 tipos de falha |
| `dataset_unificado.parquet` | Parquet | 5.000.000 | Base gerada — entrada da análise |

> Os datasets **não são versionados** no git. Certifique-se de que estão na pasta `Datasets/` antes de executar.

---

## Classes de Falha

| Classe | Rótulo | Origem |
|---|---|---|
| `0` | Normal | Todos os datasets |
| `1` | Falha / ECL | Hard Failure, Soft Failure, Lightpath |
| `2` | EDFA | Lightpath |
| `3` | NLI | Lightpath |

---

## Dependências Principais

| Pacote | Versão | Uso |
|---|---|---|
| pandas | 2.3.3 | Manipulação de dados |
| numpy | 2.3.4 | Operações numéricas |
| scikit-learn | 1.8.0 | ML, métricas, pré-processamento |
| imbalanced-learn | 0.14.1 | SMOTE |
| flaml | 2.6.0 | AutoML |
| lightgbm | 4.6.0 | Dependência do FLAML AutoML |
| xgboost | 3.2.0 | Dependência do FLAML AutoML |
| tensorflow | 2.21.0 | LSTM |
| shap | 0.47.2 | Explicabilidade |
| scipy | 1.17.1 | Teste KS |
| pyarrow | 24.0.0 | Leitura/escrita Parquet |
| joblib | 1.5.3 | Salvamento de modelos |

---

## Resultados

Consulte o relatório detalhado em:

```
analise_dados/ANALISE_COMPLETA.md
```

---

## Artigos de Referência

- **Learning Long- and Short-Term Temporal Patterns for ML-Driven Fault Management in Optical Communication Networks** — base metodológica para feature engineering e LSTNet
- **Reliable and Scalable Kafka-based Framework for Optical Network Telemetry** — referência para pipeline de inferência em tempo real
- **1-s2.0-S1389128625001276** — referência complementar (Elsevier/ScienceDirect)
