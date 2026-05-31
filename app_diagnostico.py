import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. CONFIGURAÇÃO DA PÁGINA (Deve ser o primeiro comando do script)
st.set_page_config(
    page_title="Predição de Obesidade - Hospital",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. DEFINIÇÃO DA FUNÇÃO CUSTOMIZADA (Obrigatório para o correto carregamento do .pkl)
def arredondar_valores(X):
    return np.round(X)

# 3. CARREGAMENTO DO MODELO COM CACHE DE PERFORMANCE
@st.cache_resource
def carregar_modelo():
    return joblib.load('pipeline_obesity_model.pkl')

try:
    modelo = carregar_modelo()
except Exception as e:
    st.error(f"Erro ao carregar o arquivo 'pipeline_obesity_model.pkl': {e}")
    st.stop()

# 4. CABEÇALHO PRINCIPAL DA INTERFACE
st.title("🏥 Sistema de Auxílio ao Diagnóstico de Obesidade")
st.markdown("""
    Este sistema utiliza um modelo preditivo de Machine Learning de alta assertividade para auxiliar a equipe médica 
    no diagnóstico precoce e na previsão do risco de obesidade, considerando fatores multifatoriais.
""")
st.write("---")

# 5. BLOCO VISUAL DE MÉTRICAS CLÍNICAS
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric(label="Assertividade do Modelo", value="95.74%", delta="Meta: >75%")
with col_m2:
    st.metric(label="Tempo de Resposta", value="< 1s", delta="Tempo Real")
with col_m3:
    st.metric(label="Fatores Analisados", value="16 Variáveis", delta="Multifatorial")

st.write("---")

# 6. FORMULÁRIO DE CAPTURA DE DADOS (EM PORTUGUÊS)
with st.form("formulario_clinico"):
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Perfil Demográfico e Biométrico")
        gender = st.selectbox("Gênero Biológico:", ["Masculino", "Feminino"])
        age = st.number_input("Idade (anos):", min_value=14, max_value=100, value=25, step=1)
        height = st.number_input("Altura (metros):", min_value=1.00, max_value=2.50, value=1.70, step=0.01, format="%.2f")
        weight = st.number_input("Peso Corporal (kg):", min_value=30.0, max_value=250.0, value=70.0, step=0.1, format="%.1f")
        
        st.write("---")
        st.subheader("🧬 Histórico e Padrão Alimentar")
        family_history = st.selectbox("Histórico familiar de excesso de peso?", ["Sim", "Não"])
        favc = st.selectbox("Consome alimentos altamente calóricos com frequência?", ["Sim", "Não"])
        fcvc_label = st.selectbox("Frequência de consumo de vegetais nas refeições:", ["Raramente", "Às vezes", "Sempre"])
        ncp_label = st.selectbox("Número de refeições principais por dia:", ["Uma refeição", "Duas refeições", "Três refeições", "Quatro ou mais"])

    with col2:
        st.subheader("🥗 Comportamento Rotineiro")
        caec = st.selectbox("Consome lanches/comes entre as refeições?", ["Não consome", "Às vezes (Sometimes)", "Frequentemente (Frequently)", "Sempre (Always)"])
        scc = st.selectbox("Monitora a ingestão calórica diária?", ["Sim", "Não"])
        ch2o_label = st.selectbox("Consumo diário de água habitual:", ["Menos de 1 Litro por dia", "Entre 1 e 2 Litros por dia", "Mais de 2 Litros por dia"])

        st.write("---")
        st.subheader("🚴 Estilo de Vida e Hábitos")
        smoke = st.selectbox("O paciente possui o hábito de fumar?", ["Não", "Sim"])
        faf_label = st.selectbox("Frequência semanal de atividade física:", ["Nenhuma", "1 a 2 vezes por semana", "3 a 4 vezes por semana", "5 ou mais vezes por semana"])
        tue_label = st.selectbox("Tempo diário de uso de dispositivos eletrônicos:", ["0 a 2 horas por dia", "3 a 5 horas por dia", "Mais de 5 horas por dia"])
        calc = st.selectbox("Com que frequência consome bebida alcoólica?", ["Não bebe", "Às vezes (Sometimes)", "Frequentemente (Frequently)", "Sempre (Always)"])
        mtrans = st.selectbox("Meio de transporte habitual utilizado:", ["Transporte Público", "Automóvel próprio", "A pé / Caminhando", "Motocicleta", "Bicicleta"])

    st.write("")
    submetido = st.form_submit_button("Gerar Análise Diagnóstica")

# 7. TRADUÇÃO E PROCESSAMENTO DA PREDIÇÃO
if submetido:
    
    # Mapeamento de variáveis binárias simples
    gender_ing = "Male" if gender == "Masculino" else "Female"
    family_history_ing = "yes" if family_history == "Sim" else "no"
    favc_ing = "yes" if favc == "Sim" else "no"
    smoke_ing = "yes" if smoke == "Sim" else "no"
    scc_ing = "yes" if scc == "Sim" else "no"
    
    # Mapeamento dos novos selects textuais para os valores numéricos de treino
    fcvc_map = {"Raramente": 1.0, "Às vezes": 2.0, "Sempre": 3.0}
    ncp_map = {"Uma refeição": 1.0, "Duas refeições": 2.0, "Três refeições": 3.0, "Quatro ou mais": 4.0}
    ch2o_map = {"Menos de 1 Litro por dia": 1.0, "Entre 1 e 2 Litros por dia": 2.0, "Mais de 2 Litros por dia": 3.0}
    faf_map = {"Nenhuma": 0.0, "1 a 2 vezes por semana": 1.0, "3 a 4 vezes por semana": 2.0, "5 ou mais vezes por semana": 3.0}
    tue_map = {"0 a 2 horas por dia": 0.0, "3 a 5 horas por dia": 1.0, "Mais de 5 horas por dia": 2.0}
    
    # Mapeamento das strings de frequência e transporte
    caec_map = {"Não consome": "no", "Às vezes (Sometimes)": "Sometimes", "Frequentemente (Frequently)": "Frequently", "Sempre (Always)": "Always"}
    calc_map = {"Não bebe": "no", "Às vezes (Sometimes)": "Sometimes", "Frequentemente (Frequently)": "Frequently", "Sempre (Always)": "Always"}
    mtrans_map = {"Transporte Público": "Public_Transportation", "Automóvel próprio": "Automobile", "A pé / Caminhando": "Walking", "Motocicleta": "Motorbike", "Bicicleta": "Bike"}

    # Criação estruturada do DataFrame que entra na Pipeline do modelo
    dados_paciente = pd.DataFrame([{
        'Gender': gender_ing,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history': family_history_ing,
        'FAVC': favc_ing,
        'FCVC': fcvc_map[fcvc_label],
        'NCP': ncp_map[ncp_label],
        'CAEC': caec_map[caec],
        'SMOKE': smoke_ing,
        'CH2O': ch2o_map[ch2o_label],
        'SCC': scc_ing,
        'FAF': faf_map[faf_label],
        'TUE': tue_map[tue_label],
        'CALC': calc_map[calc],
        'MTRANS': mtrans_map[mtrans]
    }])
    
    # Executa a predição chamando a esteira inteligente
    with st.spinner("Processando dados clínicos..."):
        resultado_bruto = modelo.predict(dados_paciente)[0]
    
    # Dicionário de tradução médica dos alvos do modelo para exibição amigável
    traducao_resultados = {
        "Insufficient_Weight": "Abaixo do Peso Ideal (Peso Insuficiente)",
        "Normal_Weight": "Peso Normal / Saudável",
        "Overweight_Level_I": "Sobrepeso Grau I",
        "Overweight_Level_II": "Sobrepeso Grau II",
        "Obesity_Type_I": "Obesidade Grau I",
        "Obesity_Type_II": "Obesidade Grau II (Severa)",
        "Obesity_Type_III": "Obesidade Grau III (Mórbida)"
    }
    
    diagnostico_final = traducao_resultados.get(resultado_bruto, resultado_bruto)
    
    # Exibição do Diagnóstico Final na tela
    st.write("---")
    st.subheader("📊 Resultado da Análise Preditiva:")
    st.success(f"Classificação Diagnóstica Estimada: **{diagnostico_final}**")