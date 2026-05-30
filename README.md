# Classificação Computacional de Níveis de Obesidade via Machine Learning

Este repositório contém a arquitetura de engenharia de dados e modelagem preditiva para classificar indivíduos em 7 diferentes categorias de peso corporal, utilizando dados demográficos, históricos familiares e hábitos de consumo/estilo de vida.

A entrega engloba:

* **Construção da Pipeline End-to-End:** Implementação de esteiras automatizadas do Scikit-Learn.
* **Feature Engineering:** Mapeamento avançado e tratamento de ruídos estatísticos.
* **Treinamento & Ajuste de Hiperparâmetros:** Execução de Grid Search com validação cruzada (K-Fold).
* **Avaliação Crítica:** Análise de matrizes de confusão sob a ótica de negócios/saúde.
* **Exportação do Modelo (Deploy):** Criação do artefato serializado para produção.

---

## 📊 Desempenho e Comparação de Modelos
Os algoritmos foram avaliados no conjunto de teste (20% da base) utilizando amostragem estratificada. Os resultados obtidos de acurácia global foram:

* 📌 **Logistic Regression:** 87.00%
* 📌 **Random Forest (Padrão e Otimizado):** 93.62%
* 📌 **Gradient Boosting (Vencedor):** **95.74%**

### Por que o Gradient Boosting foi escolhido?
O Gradient Boosting apresentou maior especialização nas categorias limítrofes (como separar `Normal_Weight` de `Overweight_Level_I`), onde os modelos baseados em árvores isoladas costumam falhar devido à sobreposição de dados. O modelo final atingiu um **Macro F1-Score de 0.96**, provando um desempenho uniforme em todas as 7 classes alvo.

---

## 🚀 Como Executar o Modelo em Produção
O script possui uma rotina de salvamento dinâmico que detecta automaticamente o algoritmo com maior acurácia e exporta apenas o modelo campeão.

O arquivo binário gerado foi o **`pipeline_obesity_model.pkl`**. Para utilizá-lo em uma API ou sistema web para classificar um novo paciente com dados brutos, basta executar:

```python
import joblib
import pandas as pd

# Carrega a esteira de processamento e o modelo treinado de uma só vez
modelo = joblib.load('pipeline_obesity_model.pkl')

# Realiza a predição direta enviando o JSON/DataFrame original do paciente
predicao = modelo.predict(dados_brutos_novo_paciente)
print(f"Categoria de peso diagnosticada: {predicao[0]}")
