import streamlit as st
import os
import matplotlib.pyplot as plt
import chat

try:
    import ghi_mamdani
    import ghi_sugeno
except ImportError as e:
    st.error(f"Erro ao importar modelos fuzzy: {e}. Verifique se os arquivos estão na pasta raiz.")
    st.stop()

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(
    page_title="Avaliação Solar - Fuzzy vs ML",
    page_icon="☀️",
    layout="wide"
)

# Inicializa histórico do chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Função auxiliar para plotar gráficos fuzzy no Streamlit
def plot_variable(variable, input_val=None, title=""):
    """Gera a figura matplotlib de uma variável fuzzy."""
    fig, ax = plt.subplots(figsize=(8, 3))
    variable.view(sim=None, ax=ax)
    if input_val is not None:
        ax.axvline(x=input_val, color='red', linestyle='--', label=f'Input: {input_val:.2f}')
        ax.legend()
    ax.set_title(title)
    return fig

# --- TÍTULO ---
st.title("☀️ Dashboard de Avaliação: Irradiação Solar RN")
st.markdown("Comparativo de Lógica Fuzzy (Mamdani vs Sugeno) e Assistente de IA.")

# --- ABAS ---
tab_chat, tab_manual = st.tabs([
    "Chatbot",
    "Painel de Controle Fuzzy"
])

# ==============================================================================
# ABA 1: CHATBOT COM LLM
# ==============================================================================
with tab_chat:
    st.header("Sabatina com a IA")
    st.markdown("Utilize este chat para questionar sobre a metodologia, métricas e visualizar os gráficos de avaliação do projeto.")

    # Container para o histórico
    chat_container = st.container()

    # Renderiza histórico
    with chat_container:
        for interaction in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(interaction["question"])
            
            with st.chat_message("model"):
                st.markdown(interaction["answer"])
                if interaction["images"]:
                    for img_path in interaction["images"]:
                        if os.path.exists(img_path):
                            st.image(img_path, caption=f"Visualização: {os.path.basename(img_path)}")
                        else:
                            st.warning(f"⚠️ Imagem não encontrada: {img_path}")

    # Input do usuário
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([6, 1])
        with col_input:
            user_prompt = st.text_input("Pergunta:", placeholder="Ex: Qual a diferença de erro entre Mamdani e Sugeno?")
        with col_btn:
            submit_button = st.form_submit_button("Enviar")

    if submit_button and user_prompt:
        with st.spinner("Consultando contexto e gerando resposta..."):
            try:
                api_text, api_images = chat.run_ai(user_prompt)
                
                st.session_state.chat_history.append({
                    "question": user_prompt,
                    "answer": api_text,
                    "images": api_images
                })
                st.rerun()
            except Exception as e:
                st.error(f"Erro na comunicação com a IA: {e}")

# ==============================================================================
# ABA 2: PAINEL MANUAL (SIMULADOR FUZZY)
# ==============================================================================
with tab_manual:
    st.header("Simulação em Tempo Real: Mamdani vs Sugeno")
    
    # --- 1. Inputs Globais ---
    st.markdown("### 1. Defina as Condições Climáticas")
    col_in1, col_in2, col_in3, col_in4 = st.columns(4)
    
    with col_in1:
        h_cos = st.slider("Hora Cosseno (Elevação)", -1.0, 1.0, -1.0, 0.01, help="-1: Meio dia, 1: Meia noite")
    with col_in2:
        h_sin = st.slider("Hora Seno (Dia/Tarde)", -1.0, 1.0, 0.0, 0.01, help="Negativo: Tarde, Positivo: Manhã")
    with col_in3:
        nuvem = st.slider("Cobertura de Nuvens", 0.0, 10.0, 0.0, 0.1, help="0: Limpo, 10: Fechado")
    with col_in4:
        temp = st.slider("Temperatura (°C)", 10.0, 45.0, 30.0, 0.1)

    st.markdown("---")

    # --- 2. Execução dos Modelos ---
    
    # Mamdani
    try:
        ghi_m = ghi_mamdani.avaliar_ghi_mamdani(h_sin, h_cos, nuvem, temp)
    except Exception as e:
        ghi_m = 0.0
        st.error(f"Erro Mamdani: {e}")

    # Sugeno
    try:
        if hasattr(ghi_sugeno, 'avaliar_ghi_sugeno'):
            ghi_s = ghi_sugeno.avaliar_ghi_sugeno(h_sin, h_cos, nuvem, temp)
        else:
            ghi_s = 0.0 
            st.warning("Função 'avaliar_ghi_sugeno' não encontrada no módulo.")
    except Exception as e:
        ghi_s = 0.0
        st.error(f"Erro Sugeno: {e}")

    # --- 3. Resultados Comparativos ---
    st.markdown("### 2. Resultado da Predição (GHI W/m²)")
    
    col_res1, col_res2, col_diff = st.columns(3)
    
    with col_res1:
        st.metric(label="Mamdani (Centroide)", value=f"{ghi_m:.2f} W/m²")
        st.info("Melhor para interpretabilidade e transições suaves.")
        
    with col_res2:
        st.metric(label="Sugeno (Ponderado)", value=f"{ghi_s:.2f} W/m²")
        st.success("Geralmente mais computacionalmente eficiente e preciso nas pontas.")

    with col_diff:
        diff = ghi_m - ghi_s
        st.metric(label="Divergência (M - S)", value=f"{diff:.2f}", delta_color="off")
        if abs(diff) > 100:
            st.warning("Alta divergência entre modelos!")
        else:
            st.caption("Modelos concordantes.")