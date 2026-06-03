# Geração de Base de Dados — Redes Ópticas

Documentação do pipeline de geração, tratamento e aumento da base de dados unificada para análise preditiva de falhas em redes de comunicação óptica.

---

## Visão Geral

O notebook `geracao_dataset.ipynb` implementa um pipeline completo que:

1. **Carrega** três datasets de origens distintas
2. **Unifica** e padroniza os dados em um esquema comum
3. **Trata** valores ausentes com interpolação linear e imputação por ML
4. **Aumenta** a base com dados sintéticos (SMOTE + Ruído Gaussiano)
5. **Exporta** a base final em CSV e Parquet

**Resultado:** base com **5.000.000 linhas** pronta para modelagem.

---

## Datasets de Entrada

| Arquivo | Formato | Linhas | Descrição |
|---|---|---|---|
| `HardFailure_dataset.csv` | CSV | ~65.733 | Telemetria de falhas abruptas (perda total de sinal) |
| `SoftFailure_dataset.csv` | CSV | ~53.697 | Telemetria de falhas graduais (degradação de sinal) |
| `Lightpath_756_label_4_QoT_dataset_train_900.parquet` | Parquet | ~2.721.600 | Qualidade de Transmissão (QoT) com 4 tipos de falha |

### Colunas por dataset

**Hard/Soft Failure:**
| Coluna | Tipo | Descrição |
|---|---|---|
| `Timestamp` | Unix epoch (s) | Momento da medição |
| `Type` | string | `Infrastructure` (amplificadores) ou `Devices` (transponders) |
| `ID` | string | Identificador do equipamento (ex: `Ampli1`, `SPO1/18/11`) |
| `BER` | float | Bit Error Rate — razão linear (ex: `2.28e-08`) |
| `OSNR` | float | Optical Signal-to-Noise Ratio em dB |
| `InputPower` | float | Potência de entrada em dBm |
| `OutputPower` | float | Potência de saída em dBm |
| `Failure` | int | `0` = Normal, `1` = Falha |

**Lightpath QoT:**
| Coluna | Tipo | Descrição |
|---|---|---|
| `Timestamp` | sequencial | Índice temporal (1, 2, 3...) |
| `LP_length_km` | float | Comprimento do lightpath em km (~514 km) |
| `Laser_current_mA` | float | Corrente do laser em mA |
| `LP_power_dBm` | float | Potência do lightpath em dBm |
| `OSNR` | float | Optical Signal-to-Noise Ratio em dB |
| `BER_dB` | float | BER em escala logarítmica dB |
| `Failure_type` | int | `0`=Normal, `1`=ECL, `2`=EDFA, `3`=NLI |

---

## Schema da Base Unificada

Após a junção, a base possui as seguintes colunas:

| Coluna | Tipo Final | Origem |
|---|---|---|
| `Timestamp` | datetime64 | Todos |
| `Type` | string | Hard/Soft (`"N/A"` para Lightpath e sintéticos) |
| `ID` | string | Hard/Soft (`"N/A"` para Lightpath e sintéticos) |
| `BER` | float64 | Todos (Lightpath convertido de dB → razão linear) |
| `OSNR` | float64 | Todos |
| `InputPower` | float64 | Hard/Soft (imputado para Lightpath) |
| `OutputPower` | float64 | Hard/Soft (imputado para Lightpath) |
| `LP_length_km` | float64 | Lightpath (imputado para Hard/Soft) |
| `Laser_current_mA` | float64 | Lightpath (imputado para Hard/Soft) |
| `LP_power_dBm` | float64 | Lightpath (imputado para Hard/Soft) |
| `Failure_type` | int8 | Todos |
| `source` | string | Identificador de origem da linha |

### Valores de `source`

| Valor | Descrição |
|---|---|
| `hard_failure` | Dado original do HardFailure_dataset |
| `soft_failure` | Dado original do SoftFailure_dataset |
| `lightpath` | Dado original do Lightpath QoT |
| `synthetic_smote` | Gerado pelo SMOTE |
| `synthetic_gaussian` | Gerado por ruído gaussiano |

---

## Pipeline Detalhado

### 1. Carregamento

- Hard/Soft lidos via `pd.read_csv`
- Lightpath lido via `pd.read_parquet` (convertido previamente do TXT original de 222 MB)

### 2. Padronização e Junção

- Coluna `Failure` renomeada para `Failure_type` (Hard/Soft)
- BER do Lightpath convertido de dB para razão linear:
  ```
  BER_ratio = 10 ^ (BER_dB / 10)
  ```
- Datasets concatenados com `pd.concat` — colunas ausentes preenchidas com `NaN`
- Timestamp convertido para `datetime64`:
  - Hard/Soft: Unix epoch em segundos
  - Lightpath: índice sequencial → datetime relativo ao início do Hard Failure
- Dados ordenados cronologicamente
- `Failure_type` NaN → `0` (ausência de rótulo = operação normal)

### 3. Tratamento de Valores Ausentes

**Etapa 1 — Interpolação linear intra-série:**
Preenche NaN *dentro de cada fonte* respeitando a continuidade temporal. Agrupa por `source` para não misturar séries distintas.

**Etapa 2 — Imputação por ML (IterativeImputer / MICE):**
Preenche NaN *estruturais* entre fontes (ex: `InputPower` não existe no Lightpath; `LP_length_km` não existe no Hard/Soft).

- **Algoritmo:** `IterativeImputer` com `ExtraTreesRegressor`
- **Como funciona:** para cada coluna com NaN, treina uma árvore de regressão usando as demais colunas como features e imputa os valores faltantes iterativamente
- **Otimização:** `fit` em amostra estratificada de `200.000` linhas → `transform` no dataset completo (evita treinar em 2.8M linhas)
- **Parâmetros:** `n_estimators=10`, `max_iter=5`, `n_jobs=-1`

Colunas categóricas (`Type`, `ID`) preenchidas com `"N/A"`.

### 4. Aumento de Dados

**SMOTE (Synthetic Minority Over-sampling Technique):**
- Gera amostras sintéticas interpolando os k vizinhos mais próximos da classe minoritária
- Estratégia: eleva cada classe de falha até a mesma quantidade da classe Normal
- `k_neighbors=5`
- Aplicado **após** a imputação — usa todas as 7 features numéricas

**Ruído Gaussiano:**
- Completa a base até atingir `TARGET_ROWS = 5.000.000`
- Perturbação de `1%` do desvio padrão de cada coluna
- Timestamps gerados aleatoriamente dentro do intervalo original
- Mantém a proporção de classes dos dados originais

### 5. Exportação

| Arquivo | Formato | Localização |
|---|---|---|
| `dataset_unificado.csv` | CSV | `Datasets/output/` |
| `dataset_unificado.parquet` | Parquet | `Datasets/output/` |

---

## Configuração Global

Todas as constantes de configuração estão na célula **"Configuração Global"** no topo do notebook:

```python
TARGET_ROWS    = 5_000_000  # tamanho-alvo da base final
RANDOM_STATE   = 42         # semente para reproducibilidade
IMPUTER_SAMPLE = 200_000    # linhas para treinar o imputer
NOISE_SCALE    = 0.01       # magnitude do ruído gaussiano
IMPUTE_COLS    = ["BER", "OSNR", "InputPower", "OutputPower",
                  "LP_length_km", "Laser_current_mA", "LP_power_dBm"]
```

---

## Dependências

```
pandas
numpy
scikit-learn
imbalanced-learn
pyarrow ou fastparquet
```

---

## Estrutura de Arquivos

```
geracao_dataset/
├── geracao_dataset.ipynb   # Pipeline principal
└── README.md               # Esta documentação

Datasets/
├── HardFailure_dataset.csv
├── SoftFailure_dataset.csv
├── Lightpath_756_label_4_QoT_dataset_train_900.parquet
└── output/
    ├── dataset_unificado.csv
    └── dataset_unificado.parquet
```
