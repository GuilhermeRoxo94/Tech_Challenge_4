# Tech Challenge — Fase 04 · Classificação de Níveis de Obesidade

Projeto de Data Analytics (POSTECH) que entrega uma **pipeline completa de Machine Learning** para classificar pacientes em 7 categorias de peso corporal, além de um **painel executivo interativo** para apoiar a tomada de decisão da equipe médica.

A solução foi construída a partir de dados demográficos, histórico familiar e hábitos de consumo/estilo de vida, e abrange desde a engenharia de atributos até a exportação do modelo serializado para produção.

---

## 🩺 Contexto do problema

Atuando como cientista de dados de um hospital, o desafio é desenvolver um modelo preditivo que auxilie médicos e médicas a identificar o nível de obesidade de uma pessoa. A obesidade é uma condição multifatorial (fatores genéticos, ambientais e comportamentais), e o objetivo é transformar esses dados em apoio à decisão clínica.

---

## 📁 Estrutura do repositório

| Arquivo | Descrição |
|---|---|
| `01_EDA.ipynb` | Análise exploratória: inspeção, limpeza, distribuições e principais insights. Exporta `obesity_clean.csv`. |
| `obesity_analysis.ipynb` | Pipeline de ML end-to-end: feature engineering, treinamento, GridSearch e exportação do modelo. |
| `app.py` | Painel executivo analítico em Streamlit (visão de negócio para a equipe médica). |
| `Obesity.csv` | Base de dados original (2.111 registros, 17 colunas). |
| `requirements.txt` | Dependências do painel Streamlit. |
| `pipeline_obesity_model.pkl` | Artefato do modelo campeão, gerado ao executar `obesity_analysis.ipynb`. |

> **Observação sobre o dicionário de dados:** a coluna de tempo em dispositivos tecnológicos aparece como `TER` no enunciado, mas no dataset e no código ela é nomeada `TUE`.

---

## 📊 Desempenho e comparação de modelos

Os algoritmos foram avaliados no conjunto de teste (20% da base) com amostragem estratificada. Acurácia global obtida:

| Modelo | Acurácia |
|---|---|
| Logistic Regression | 87,00% |
| Random Forest (padrão e otimizado) | 93,62% |
| **Gradient Boosting (vencedor)** | **95,74%** |

### Por que o Gradient Boosting foi escolhido?

O Gradient Boosting apresentou maior especialização nas categorias limítrofes (como separar `Normal_Weight` de `Overweight_Level_I`), onde modelos baseados em árvores isoladas costumam falhar devido à sobreposição dos dados. O modelo final atingiu um **Macro F1-Score de 0,96**, demonstrando desempenho uniforme nas 7 classes alvo.

Todos os modelos superam com folga o requisito mínimo de assertividade (75%).

---

## 🧪 A pipeline de Machine Learning

A esteira é construída com `Pipeline` e `ColumnTransformer` do Scikit-Learn, recebendo os dados **brutos** e aplicando todo o pré-processamento internamente:

- **Contínuas** (`Height`, `Weight`): padronização com `StandardScaler`.
- **Numéricas ordinais com ruído** (`Age`, `FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`): arredondamento + padronização.
- **Categóricas ordinais** (`CAEC`, `CALC`): `OrdinalEncoder` seguindo a escala de frequência (`no → Sometimes → Frequently → Always`) + padronização.
- **Categóricas nominais** (`Gender`, `family_history`, `FAVC`, `SMOKE`, `SCC`, `MTRANS`): `OneHotEncoder`.

O treinamento compara os três algoritmos, executa **GridSearchCV com validação cruzada (K-Fold)** para o Random Forest e, ao final, uma rotina de **salvamento dinâmico** detecta automaticamente o modelo de maior acurácia e exporta apenas o campeão.

---

## 🚀 Como utilizar o modelo em produção

O artefato binário gerado é o **`pipeline_obesity_model.pkl`**. Por conter toda a esteira de pré-processamento, ele recebe os dados brutos do paciente diretamente:

```python
import joblib

# Carrega a esteira de processamento e o modelo treinado de uma só vez
modelo = joblib.load('pipeline_obesity_model.pkl')

# Realiza a predição direta enviando o DataFrame/JSON original do paciente
predicao = modelo.predict(dados_brutos_novo_paciente)
print(f"Categoria de peso diagnosticada: {predicao[0]}")
```

---

## 🖥️ Painel executivo (dashboard analítico)

Dashboard em Streamlit com visual escuro inspirado em Power BI, trazendo KPIs, distribuição dos níveis de obesidade, fatores de risco (idade, histórico familiar, gênero), análise de hábitos e insights executivos para a equipe médica.

Na pasta do projeto, execute:

```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Se o comando `py` não funcionar:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

---

## ⚙️ Dependências

- **Painel (`app.py`):** ver `requirements.txt` — `streamlit`, `pandas`, `plotly`.
- **Notebooks (EDA e modelagem):** além dos acima, são necessários `scikit-learn`, `numpy`, `joblib`, `matplotlib` e `seaborn`.

---

## 🔗 Entregáveis

- **Aplicação preditiva (Streamlit):** _adicionar link_
- **Painel analítico (Streamlit):** _adicionar link_
- **Repositório GitHub:** _adicionar link_
- **Vídeo de apresentação (4–10 min):** _adicionar link_