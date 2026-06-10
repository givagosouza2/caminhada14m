import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import plotly.express as px

# -----------------------------------------------------------------------------
# Configuração da Página Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Analisador de Dados de Aceleração", layout="wide")
st.title("📈 Analisador de Dados de Aceleração")
st.markdown("""
Faça o upload do arquivo de texto contendo as colunas: **Tempo (ms)**, **Acc X (m/s²)**, **Acc Y (m/s²)** e **Acc Z (m/s²)**.  
O aplicativo irá remover a tendência linear (detrend), interpolar os dados para 100 Hz, calcular a norma da aceleração e permitir a seleção de uma janela de tempo para análise.
""")

# -----------------------------------------------------------------------------
# Upload do Arquivo
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Escolha o arquivo de texto (.txt, .csv, .dat)", type=["txt", "csv", "dat"])

if uploaded_file is not None:
    try:
        # Ler o arquivo usando ponto e vírgula como separador
        df = pd.read_csv(uploaded_file, sep=';')
        
        # Padronizar nomes das colunas
        df.columns = df.columns.str.strip().str.lower()
        
        # Mapear as colunas específicas
        col_map = {
            'tempo (ms)': 'time_ms',
            'acc x (m/s²)': 'acc_x',
            'acc y (m/s²)': 'acc_y',
            'acc z (m/s²)': 'acc_z'
        }
        df = df.rename(columns=col_map)
        
        required_cols = ['time_ms', 'acc_x', 'acc_y', 'acc_z']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Colunas obrigatórias ausentes. Esperado: {list(col_map.keys())}. Encontrado: {list(df.columns)}")
            st.stop()

        # Converter tempo de milissegundos para segundos
        df['time'] = df['time_ms'] / 1000.0
        
        t = df['time'].values
        x = df['acc_x'].values
        y = df['acc_y'].values
        z = df['acc_z'].values

        # -----------------------------------------------------------------------------
        # 1. Remoção de Tendência (Detrend)
        # -----------------------------------------------------------------------------
        x_detrended = signal.detrend(x)
        y_detrended = signal.detrend(y)
        z_detrended = signal.detrend(z)

        # -----------------------------------------------------------------------------
        # 2. Interpolação para 100 Hz
        # -----------------------------------------------------------------------------
        fs_target = 100.0  # Frequência de amostragem alvo em Hz
        dt = 1.0 / fs_target
        
        t_min, t_max = t.min(), t.max()
        t_new = np.arange(t_min, t_max + dt, dt)
        
        interp_x = interp1d(t, x_detrended, kind='linear', fill_value="extrapolate")
        interp_y = interp1d(t, y_detrended, kind='linear', fill_value="extrapolate")
        interp_z = interp1d(t, z_detrended, kind='linear', fill_value="extrapolate")
        
        x_interp = interp_x(t_new)
        y_interp = interp_y(t_new)
        z_interp = interp_z(t_new)

        # -----------------------------------------------------------------------------
        # 3. Cálculo da Norma
        # -----------------------------------------------------------------------------
        norm = np.sqrt(x_interp**2 + y_interp**2 + z_interp**2)

        # -----------------------------------------------------------------------------
        # 3.1 Filtro Passa-Baixa de 0.5 Hz na Norma
        # -----------------------------------------------------------------------------
        cutoff_freq = 1.5           # Frequência de corte em Hz
        nyquist = fs_target / 2.0   # Frequência de Nyquist (50 Hz)
        normal_cutoff = cutoff_freq / nyquist  # Frequência normalizada (0.01)
        filter_order = 4            # Ordem do filtro Butterworth
        
        # Projetar o filtro Butterworth
        b, a = signal.butter(filter_order, normal_cutoff, btype='low', analog=False)
        
        # Aplicar filtfilt (filtra nos dois sentidos -> fase zero, sem atraso)
        norm_filtered = signal.filtfilt(b, a, norm)

        # Criar DataFrame com os dados processados
        df_processed = pd.DataFrame({
            'Tempo (s)': t_new,
            'Acc X': x_interp,
            'Acc Y': y_interp,
            'Acc Z': z_interp,
            'Norma': norm,
            'Norma Filtrada (0.5 Hz)': norm_filtered
        })

        # -----------------------------------------------------------------------------
        # 4. Seleção de Janela de Tempo
        # -----------------------------------------------------------------------------
        st.subheader("Análise por Janela de Tempo")
        st.markdown("Use o controle deslizante abaixo para selecionar uma janela de tempo específica para análise detalhada.")
        
        min_time = float(t_new.min())
        max_time = float(t_new.max())
        default_max = min(min_time + 10.0, max_time)
        
        time_window = st.slider(
            "Selecionar Janela de Tempo (segundos)",
            min_value=min_time,
            max_value=max_time,
            value=(min_time, default_max),
            step=0.1,
            format="%.2f"
        )

        # -----------------------------------------------------------------------------
        # 5. Plotagem dos Dados Completos com a Janela Destacada
        # -----------------------------------------------------------------------------
        st.subheader("Dados Processados Completos")
        fig_full = px.line(
            df_processed, 
            x='Tempo (s)', 
            y='Norma', 
            title='Norma da Aceleração (Destendida e Interpolada para 100 Hz)',
            labels={'Norma': 'Norma da Aceleração (m/s²)'},
            height=400
        )
        
        # Adiciona um retângulo semi-transparente cobrindo a janela selecionada
        fig_full.add_vrect(
            x0=time_window[0], 
            x1=time_window[1], 
            fillcolor="LightSalmon", 
            opacity=0.3, 
            layer="below", 
            line_width=0,
            annotation_text="Janela de Análise",
            annotation_position="top left"
        )
        fig_full.add_vline(x=time_window[0], line_dash="dash", line_color="red")
        fig_full.add_vline(x=time_window[1], line_dash="dash", line_color="red")

        st.plotly_chart(fig_full, use_container_width=True)

        # -----------------------------------------------------------------------------
        # 6. Análise Detalhada da Janela
        # -----------------------------------------------------------------------------
        mask = (df_processed['Tempo (s)'] >= time_window[0]) & (df_processed['Tempo (s)'] <= time_window[1])
        df_window = df_processed[mask]

        # Métricas da janela selecionada (para a norma original e filtrada)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Média da Norma", f"{df_window['Norma'].mean():.4f}")
        with col2:
            st.metric("Máx. da Norma", f"{df_window['Norma'].max():.4f}")
        with col3:
            st.metric("Média (Filtrada)", f"{df_window['Norma Filtrada (0.5 Hz)'].mean():.4f}")
        with col4:
            st.metric("Desvio Padrão", f"{df_window['Norma'].std():.4f}")

        # Plotagem da janela com zoom - agora mostrando a norma original E a filtrada
        fig_window = px.line(
            df_window, 
            x='Tempo (s)', 
            y=['Norma', 'Norma Filtrada (0.5 Hz)'], 
            title=f'Perfil Acelerométrico na Janela ({time_window[0]:.2f}s - {time_window[1]:.2f}s)',
            labels={'value': 'Aceleração (m/s²)', 'variable': 'Série'},
            height=350
        )
        
        # Ajustar nomes da legenda para ficar mais elegante
        fig_window.for_each_trace(lambda t: t.update(
            name='Norma Original' if t.name == 'Norma' else t.name,
            legendgroup='Norma Original' if t.name == 'Norma Original' else t.legendgroup
        ))
        
        # Personalizar cores e estilos para distinguir as curvas
        fig_window.update_traces(
            selector=dict(name='Norma Original'),
            line=dict(color='rgba(31, 119, 180, 0.5)', width=1),  # Azul claro, mais fino
        )
        fig_window.update_traces(
            selector=dict(name='Norma Filtrada (0.5 Hz)'),
            line=dict(color='red', width=2.5),  # Vermelha e mais grossa
        )
        
        fig_window.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        col1,col2,col3 = st.columns([0.2,1,0.2])
        with col2:
            st.plotly_chart(fig_window, use_container_width=True)

        # Opcional: Mostrar a tabela de dados processados da janela selecionada
        with st.expander("Visualizar Tabela de Dados Processados (Janela Selecionada)"):
            st.dataframe(df_window, use_container_width=True)

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")
        st.info("Por favor, verifique se o arquivo possui os cabeçalhos exatos e está separado por ponto e vírgula (;).")

else:
    st.info("👆 Por favor, faça o upload do arquivo de texto para começar.")
    
    st.markdown("### Formato Esperado do Arquivo:")
    st.code("""Tempo (ms);Acc X (m/s²);Acc Y (m/s²);Acc Z (m/s²)
0;9.8018;0.0526;0.2633
18;9.9646;0.0383;0.2489
35;9.9742;0.0430;0.1963
...""", language="text")
