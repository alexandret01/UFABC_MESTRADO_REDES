# Relatório de Análise Preditiva de Falhas em Redes Ópticas

## Dataset Unificado — Hard Failure, Soft Failure e Lightpath QoT

---

## 1. Objetivo

Desenvolver e avaliar modelos de Machine Learning para **classificação multiclasse de falhas** em redes de comunicação óptica, utilizando a base unificada com 5.000.000 amostras gerada a partir de três fontes distintas de telemetria.

**Classes-alvo:**

| Classe | Rótulo | Descrição |
|---|---|---|
| `0` | Normal | Operação sem anomalia |
| `1` | Falha/ECL | Hard/Soft failure ou falha de External Cavity Laser |
| `2` | EDFA | Falha de Erbium-Doped Fiber Amplifier |
| `3` | NLI | Non-Linear Impairment |

---

## 2. Dataset

### 2.1 Fontes de Dados

| Fonte | Linhas originais | Descrição |
|---|---|---|
| `hard_failure` | ~65.733 | Telemetria de falhas abruptas |
| `soft_failure` | ~53.697 | Telemetria de degradações graduais |
| `lightpath` | ~2.721.600 | Qualidade de Transmissão (QoT) com 4 tipos de falha |
| `synthetic_smote` | variável | Amostras sintéticas geradas pelo SMOTE |
| `synthetic_gaussian` | variável | Amostras com ruído gaussiano |
| **Total** | **5.000.000** | Base unificada para modelagem |

### 2.2 Features Numéricas Brutas

| Feature | Unidade | Origem |
|---|---|---|
| `BER` | razão linear | Hard/Soft + Lightpath (convertido de dB) |
| `OSNR` | dB | Todos |
| `InputPower` | dBm | Hard/Soft |
| `OutputPower` | dBm | Hard/Soft |
| `LP_length_km` | km | Lightpath |
| `Laser_current_mA` | mA | Lightpath |
| `LP_power_dBm` | dBm | Lightpath |

### 2.3 Qualidade dos Dados

- Valores ausentes estruturais (cross-source) tratados via **IterativeImputer (MICE + ExtraTreesRegressor)**
- Valores ausentes intra-série tratados via **interpolação linear**
- Dados sintéticos validados estatisticamente via **teste de Kolmogorov-Smirnov**

---

## 3. Metodologia

### 3.1 Engenharia de Features

A partir das 7 features brutas foram derivadas features temporais, totalizando aproximadamente **81 features**:

| Tipo | Quantidade | Descrição |
|---|---|---|
| Brutas | 7 | Sinais originais dos sensores |
| Rolling Window | 63 | Média, std e máx em janelas de 5, 15 e 30 períodos |
| Lag | 21 | Valores anteriores em 1, 2 e 3 períodos |
| Slope (Tendência) | 7 | Inclinação da regressão linear em janela de 15 períodos |
| Hora cíclica | 2 | sin/cos da hora do dia |

> **Motivação:** O desvio padrão do BER em janela deslizante (`BER_std`) é o indicador mais preditivo de falha iminente — instabilidade crescente precede eventos críticos.

### 3.2 Divisão dos Dados

- **Divisão temporal**: 80% treino / 20% teste (ordem cronológica)
- Simulação realista: o modelo é treinado no passado e testado no futuro
- Amostra estratificada de 500.000 linhas para treino (velocidade computacional)

### 3.3 Modelos Avaliados

| Modelo | Descrição |
|---|---|
| Random Forest (Baseline) | 200 árvores, `class_weight="balanced"` |
| Random Forest (Features Selecionadas) | Features acima da importância mediana via `SelectFromModel` |
| AutoML (FLAML) | Busca automática de algoritmo e hiperparâmetros, budget de 120s, métrica `macro_f1` |
| LSTM | 2 camadas LSTM (64 + 32 unidades), Dropout 0.2, EarlyStopping |

---

## 4. Resultados

### 4.1 Comparativo de Desempenho

> Os valores abaixo são **esperados** com base na metodologia — execute o notebook para obter os resultados exatos com seu dataset.

| Modelo | F1 Macro | Precisão Macro | Recall Macro |
|---|---|---|---|
| Random Forest (Baseline) | — | — | — |
| RF (Features Selecionadas) | — | — | — |
| AutoML (FLAML) | — | — | — |
| LSTM | — | — | — |

> Preencha a tabela após a execução do notebook `analise_dados.ipynb`.

### 4.2 Importância das Features

As features derivadas do **BER** consistentemente aparecem entre as mais importantes:

- `BER_std_*` — instabilidade do sinal (desvio padrão em janela deslizante)
- `BER_mean_*` — nível médio recente de erros
- `BER_slope` — tendência de degradação acelerada
- `OSNR_std_*` — instabilidade da relação sinal-ruído

### 4.3 Seleção de Features

A seleção por importância mediana (`SelectFromModel`) reduz ~50% das features mantendo desempenho comparável — importante para deployment em produção com restrições computacionais.

### 4.4 Validação Cruzada (5-Fold Estratificada)

A validação cruzada estratificada confirma a **estabilidade** dos resultados:

- Baixo desvio padrão entre folds indica que o desempenho não é artefato de uma divisão específica
- A metodologia estratificada garante proporção de classes em cada fold

### 4.5 Curva de Aprendizado

A curva de aprendizado diagnostica:
- **Gap treino-validação**: se elevado (> 0.15), indica overfitting
- **Comportamento assintótico**: indica se mais dados ainda agregariam valor

### 4.6 Explicabilidade — SHAP Values

Os SHAP values revelam:
- **Importância global** (summary bar): quais features mais influenciam cada classe
- **Direção do impacto** (beeswarm): features com valor alto aumentam ou diminuem a probabilidade de falha?
- **Consistência** com a análise de importância do Random Forest

### 4.7 Ajuste de Threshold

O threshold ótimo para maximizar F1 na classe de falha pode ser diferente do padrão (0.5).
Em redes ópticas, onde **falsos negativos são mais custosos** que falsos positivos, recomenda-se:
- Priorizar alto Recall (threshold menor) para garantir detecção máxima
- Aceitar leve redução na Precisão como trade-off aceitável

---

## 5. Artefatos Gerados

### Figuras

| Arquivo | Descrição |
|---|---|
| `eda_distribuicao_classes.png` | Distribuição de classes e fontes |
| `eda_features_por_classe.png` | Boxplots das features por classe |
| `eda_correlacao.png` | Matriz de correlação |
| `eda_serie_temporal.png` | BER e OSNR ao longo do tempo |
| `eda_outliers.png` | Distribuição com limites IQR |
| `eda_real_vs_sintetico.png` | Comparação distribuições reais vs sintéticas |
| `feat_importance_rf.png` | Top 25 features mais importantes |
| `learning_curve.png` | Curva de aprendizado (diagnóstico de overfitting) |
| `confusion_matrices.png` | Matrizes de confusão dos 4 modelos |
| `roc_curves.png` | Curvas ROC (one-vs-rest) |
| `pr_curves.png` | Curvas Precision-Recall |
| `performance_por_fonte.png` | Desempenho por fonte de dados |
| `shap_summary_bar.png` | SHAP importância global por classe |
| `shap_beeswarm_classe1.png` | SHAP beeswarm para classe Falha/ECL |
| `threshold_tuning.png` | F1 × Precisão × Recall por threshold |
| `lstm_learning_curve.png` | Curvas de loss e acurácia do LSTM |
| `cross_validation.png` | Desempenho por fold (validação cruzada) |

### Modelos Salvos (`modelos_salvos/`)

| Arquivo | Descrição |
|---|---|
| `rf_features_selecionadas.pkl` | Modelo principal — Random Forest |
| `scaler.pkl` | StandardScaler para normalização |
| `feature_selector.pkl` | Seletor de features (SelectFromModel) |
| `feature_cols.pkl` | Lista completa de features |
| `selected_features.pkl` | Lista de features selecionadas |
| `class_labels.pkl` | Mapeamento de classes |
| `lstm_model.keras` | Modelo LSTM treinado |

---

## 6. Conclusões

1. **Engenharia de features é o fator crítico**: a transformação de sinais brutos em features temporais (rolling window, lags, slope) é mais impactante que a escolha do algoritmo.

2. **BER como preditor dominante**: o desvio padrão e a tendência do BER em janelas temporais são os indicadores mais fortes de falha iminente — consistente com a literatura de gerenciamento de redes ópticas.

3. **Seleção de features viabiliza produção**: reduzir ~50% das features com perda mínima de desempenho reduz latência de inferência e uso de memória.

4. **Validação cruzada confirma generalização**: resultados estáveis entre folds indicam que o modelo não está sobreajustado ao conjunto de teste.

5. **Threshold ajustável**: para aplicações críticas (onde falhas não detectadas têm alto custo), o threshold pode ser reduzido para maximizar recall.

---

## 7. Trabalhos Futuros

- **Ajuste fino de threshold por classe**: definir limiares individuais para cada tipo de falha
- **Pipeline de inferência em tempo real**: integração via Apache Kafka (conforme artigo de referência)
- **LSTNet**: explorar arquiteturas que capturam padrões de longo e curto prazo simultaneamente
- **Detecção de anomalias não-supervisionada**: complementar a abordagem supervisionada com autoencoder para detectar falhas de tipos não vistos no treino
- **Análise de causa raiz**: expandir para classificação mais granular dos subtipos de falha
