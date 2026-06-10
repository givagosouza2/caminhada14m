import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.stats import linregress
import plotly.express as px
import plotly.graph_objects as go

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
        fs_target = 100.0
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
        cutoff_freq = 1.5
        nyquist = fs_target / 2.0
        normal_cutoff = cutoff_freq / nyquist
        filter_order = 4
        
        b, a = signal.butter(filter_order, normal_cutoff, btype='low', analog=False)
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
        # 4.1 Parâmetros para Detecção de Picos
        # -----------------------------------------------------------------------------
        st.markdown("#### 🔍 Parâmetros de Detecção de Picos")
        
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            min_height = st.number_input(
                "Altura Mínima (m/s²)",
                min_value=0.0,
                max_value=float(norm.max()),
                value=float(norm.mean()),
                step=0.1,
                help="Apenas picos acima deste valor serão detectados"
            )
        
        with col_p2:
            min_distance_sec = st.number_input(
                "Distância Mínima entre Picos (s)",
                min_value=0.01,
                max_value=5.0,
                value=0.5,
                step=0.1,
                help="Distância mínima em segundos entre picos consecutivos"
            )
            min_distance_samples = int(min_distance_sec * fs_target)
        
        with col_p3:
            min_prominence = st.number_input(
                "Proeminência Mínima (m/s²)",
                min_value=0.0,
                max_value=float(norm.max() - norm.min()),
                value=1.0,
                step=0.1,
                help="Quanto o pico deve se destacar em relação aos vales ao redor"
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

        # -----------------------------------------------------------------------------
        # 6.1 Detecção de Picos na Janela Selecionada
        # -----------------------------------------------------------------------------
        norm_window = df_window['Norma'].values
        peaks, properties = signal.find_peaks(
            norm_window,
            height=min_height,
            distance=min_distance_samples,
            prominence=min_prominence
        )
        
        num_peaks = len(peaks)

        # Métricas básicas da janela
        st.markdown("#### 📊 Métricas Básicas da Janela")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Média da Norma", f"{df_window['Norma'].mean():.4f}")
        with col2:
            st.metric("Máx. da Norma", f"{df_window['Norma'].max():.4f}")
        with col3:
            st.metric("Média (Filtrada)", f"{df_window['Norma Filtrada (0.5 Hz)'].mean():.4f}")
        with col4:
            st.metric("Desvio Padrão", f"{df_window['Norma'].std():.4f}")
        with col5:
            st.metric("🎯 Número de Picos", f"{num_peaks}")

        # Plotagem da janela com zoom
        fig_window = px.line(
            df_window, 
            x='Tempo (s)', 
            y=['Norma', 'Norma Filtrada (0.5 Hz)'], 
            title=f'Perfil Acelerométrico na Janela ({time_window[0]:.2f}s - {time_window[1]:.2f}s)',
            labels={'value': 'Aceleração (m/s²)', 'variable': 'Série'},
            height=350
        )
        
        fig_window.for_each_trace(lambda t: t.update(
            name='Norma Original' if t.name == 'Norma' else t.name,
            legendgroup='Norma Original' if t.name == 'Norma Original' else t.legendgroup
        ))
        
        fig_window.update_traces(
            selector=dict(name='Norma Original'),
            line=dict(color='rgba(31, 119, 180, 0.5)', width=1),
        )
        fig_window.update_traces(
            selector=dict(name='Norma Filtrada (0.5 Hz)'),
            line=dict(color='red', width=2.5),
        )
        
        # Adicionar marcadores nos picos
        if num_peaks > 0:
            peak_times = df_window['Tempo (s)'].iloc[peaks].values
            peak_values = norm_window[peaks]
            
            fig_window.add_trace(
                px.scatter(
                    x=peak_times,
                    y=peak_values,
                    labels={'x': 'Tempo (s)', 'y': 'Aceleração (m/s²)'}
                ).data[0]
            )
            
            fig_window.data[-1].update(
                mode='markers',
                marker=dict(color='green', size=12, symbol='star', line=dict(width=2, color='black')),
                name='Picos Detectados',
                showlegend=True
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

        st.plotly_chart(fig_window, use_container_width=True)

        # =========================================================================
        # 7. ANÁLISE APROFUNDADA BASEADA NOS PICOS
        # =========================================================================
        if num_peaks >= 2:
            st.markdown("---")
            st.header("🔬 Análise Aprofundada dos Picos")
            
            # ---------------------------------------------------------------------
            # 7.1 Extração de Características dos Picos
            # ---------------------------------------------------------------------
            peak_times = df_window['Tempo (s)'].iloc[peaks].values
            peak_amplitudes = norm_window[peaks]
            peak_prominences = properties['prominences'] if 'prominences' in properties else None
            
            # Calcular intervalos entre picos consecutivos
            peak_intervals = np.diff(peak_times)
            
            # Calcular largura dos picos (se disponível)
            peak_widths = None
            if 'widths' in properties:
                peak_widths = properties['widths'] / fs_target  # Converter para segundos
            
            # Criar DataFrame com características dos picos
            peaks_df = pd.DataFrame({
                'Pico #': range(1, num_peaks + 1),
                'Tempo (s)': peak_times,
                'Amplitude (m/s²)': peak_amplitudes,
                'Intervalo até próximo (s)': np.append(peak_intervals, np.nan),
            })
            
            if peak_prominences is not None:
                peaks_df['Proeminência (m/s²)'] = peak_prominences
            
            if peak_widths is not None:
                peaks_df['Largura (s)'] = peak_widths
            
            # ---------------------------------------------------------------------
            # 7.2 Estatísticas Descritivas
            # ---------------------------------------------------------------------
            st.subheader("📈 Estatísticas dos Picos")
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                st.markdown("**Intervalos entre Picos**")
                st.metric("Média", f"{np.mean(peak_intervals):.3f} s")
                st.metric("Desvio Padrão", f"{np.std(peak_intervals):.3f} s")
                st.metric("CV", f"{(np.std(peak_intervals) / np.mean(peak_intervals) * 100):.1f}%")
            
            with col_s2:
                st.markdown("**Amplitudes dos Picos**")
                st.metric("Média", f"{np.mean(peak_amplitudes):.3f} m/s²")
                st.metric("Desvio Padrão", f"{np.std(peak_amplitudes):.3f} m/s²")
                st.metric("CV", f"{(np.std(peak_amplitudes) / np.mean(peak_amplitudes) * 100):.1f}%")
            
            with col_s3:
                st.markdown("**Frequência de Picos**")
                freq_mean = 1.0 / np.mean(peak_intervals)
                st.metric("Frequência Média", f"{freq_mean:.2f} Hz")
                st.metric("Período Médio", f"{np.mean(peak_intervals):.3f} s")
                st.metric("Total", f"{num_peaks} picos")
            
            with col_s4:
                st.markdown("**Duração da Janela**")
                window_duration = time_window[1] - time_window[0]
                st.metric("Duração", f"{window_duration:.2f} s")
                st.metric("Taxa de Picos", f"{num_peaks / window_duration:.2f} picos/s")
                if peak_prominences is not None:
                    st.metric("Proeminência Média", f"{np.mean(peak_prominences):.3f} m/s²")
            
            # ---------------------------------------------------------------------
            # 7.3 Análise de Regularidade
            # ---------------------------------------------------------------------
            st.subheader("⏱️ Análise de Regularidade")
            
            cv_intervals = np.std(peak_intervals) / np.mean(peak_intervals) * 100
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                st.metric("Coeficiente de Variação (Intervalos)", f"{cv_intervals:.2f}%")
                if cv_intervals < 5:
                    st.success("✅ Alta regularidade (CV < 5%)")
                elif cv_intervals < 15:
                    st.info("ℹ️ Regularidade moderada (CV 5-15%)")
                else:
                    st.warning("⚠️ Baixa regularidade (CV > 15%)")
            
            with col_r2:
                st.metric("Intervalo Mínimo", f"{np.min(peak_intervals):.3f} s")
                st.metric("Intervalo Máximo", f"{np.max(peak_intervals):.3f} s")
                st.metric("Amplitude", f"{np.max(peak_intervals) - np.min(peak_intervals):.3f} s")
            
            with col_r3:
                st.metric("Mediana", f"{np.median(peak_intervals):.3f} s")
                st.metric("Q1 (25%)", f"{np.percentile(peak_intervals, 25):.3f} s")
                st.metric("Q3 (75%)", f"{np.percentile(peak_intervals, 75):.3f} s")
            
            # ---------------------------------------------------------------------
            # 7.4 Análise de Tendência Temporal
            # ---------------------------------------------------------------------
            st.subheader("📉 Análise de Tendência Temporal")
            st.markdown("Verifica se há mudança sistemática nas características ao longo do tempo (possível fadiga ou adaptação).")
            
            # Regressão linear para intervalos
            x_reg = np.arange(len(peak_intervals))
            slope_intervals, intercept_intervals, r_intervals, p_intervals, std_err_intervals = linregress(x_reg, peak_intervals)
            
            # Regressão linear para amplitudes
            x_reg_amp = np.arange(len(peak_amplitudes))
            slope_amps, intercept_amps, r_amps, p_amps, std_err_amps = linregress(x_reg_amp, peak_amplitudes)
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.markdown("**Tendência dos Intervalos**")
                st.metric("Inclinação (slope)", f"{slope_intervals:.5f} s/pico")
                st.metric("R²", f"{r_intervals**2:.3f}")
                st.metric("Valor-p", f"{p_intervals:.4f}")
                
                if p_intervals < 0.05:
                    if slope_intervals > 0:
                        st.warning("⚠️ Intervalos estão **aumentando** ao longo do tempo (possível fadiga)")
                    else:
                        st.warning("⚠️ Intervalos estão **diminuindo** ao longo do tempo (aceleração)")
                else:
                    st.success("✅ Sem tendência significativa (p ≥ 0.05)")
            
            with col_t2:
                st.markdown("**Tendência das Amplitudes**")
                st.metric("Inclinação (slope)", f"{slope_amps:.5f} m/s²/pico")
                st.metric("R²", f"{r_amps**2:.3f}")
                st.metric("Valor-p", f"{p_amps:.4f}")
                
                if p_amps < 0.05:
                    if slope_amps > 0:
                        st.info("📈 Amplitudes estão **aumentando** ao longo do tempo")
                    else:
                        st.info("📉 Amplitudes estão **diminuindo** ao longo do tempo")
                else:
                    st.success("✅ Sem tendência significativa (p ≥ 0.05)")
            
            # ---------------------------------------------------------------------
            # 7.5 Visualizações
            # ---------------------------------------------------------------------
            st.subheader("📊 Visualizações")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Distribuição de Intervalos", "Distribuição de Amplitudes", "Tendência Temporal", "Comparação"])
            
            with tab1:
                fig_hist_int = px.histogram(
                    x=peak_intervals,
                    nbins=20,
                    title='Distribuição dos Intervalos entre Picos',
                    labels={'x': 'Intervalo (s)', 'y': 'Contagem'},
                    color_discrete_sequence=['#636EFA']
                )
                fig_hist_int.add_vline(x=np.mean(peak_intervals), line_dash="dash", line_color="red",
                                      annotation_text=f"Média: {np.mean(peak_intervals):.3f}s")
                st.plotly_chart(fig_hist_int, use_container_width=True)
            
            with tab2:
                fig_hist_amp = px.histogram(
                    x=peak_amplitudes,
                    nbins=20,
                    title='Distribuição das Amplitudes dos Picos',
                    labels={'x': 'Amplitude (m/s²)', 'y': 'Contagem'},
                    color_discrete_sequence=['#EF553B']
                )
                fig_hist_amp.add_vline(x=np.mean(peak_amplitudes), line_dash="dash", line_color="blue",
                                      annotation_text=f"Média: {np.mean(peak_amplitudes):.3f} m/s²")
                st.plotly_chart(fig_hist_amp, use_container_width=True)
            
            with tab3:
                # Gráfico de tendência temporal
                fig_trend = go.Figure()
                
                # Intervalos
                fig_trend.add_trace(go.Scatter(
                    x=np.arange(len(peak_intervals)),
                    y=peak_intervals,
                    mode='lines+markers',
                    name='Intervalos (s)',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ))
                
                # Linha de tendência
                y_trend_int = slope_intervals * x_reg + intercept_intervals
                fig_trend.add_trace(go.Scatter(
                    x=x_reg,
                    y=y_trend_int,
                    mode='lines',
                    name='Tendência Linear',
                    line=dict(color='red', dash='dash', width=2)
                ))
                
                fig_trend.update_layout(
                    title='Evolução Temporal dos Intervalos entre Picos',
                    xaxis_title='Número do Pico',
                    yaxis_title='Intervalo (s)',
                    height=400
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with tab4:
                # Box plot comparativo
                fig_box = go.Figure()
                
                fig_box.add_trace(go.Box(y=peak_intervals, name='Intervalos (s)', boxmean=True))
                fig_box.add_trace(go.Box(y=peak_amplitudes, name='Amplitudes (m/s²)', boxmean=True))
                
                fig_box.update_layout(
                    title='Distribuição dos Parâmetros dos Picos',
                    height=400
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # ---------------------------------------------------------------------
            # 7.6 Tabela Detalhada
            # ---------------------------------------------------------------------
            with st.expander(f"📋 Ver Tabela Completa dos {num_peaks} Picos"):
                st.dataframe(peaks_df, use_container_width=True, height=400)
                
                # Download da tabela
                csv = peaks_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Baixar Tabela (CSV)",
                    data=csv,
                    file_name="analise_picos.csv",
                    mime="text/csv"
                )
            
            # ---------------------------------------------------------------------
            # 7.7 Resumo Executivo
            # ---------------------------------------------------------------------
            st.subheader("📝 Resumo Executivo")
            
            summary = f"""
            **Análise de {num_peaks} picos detectados em {window_duration:.2f} segundos:**
            
            - **Frequência média**: {freq_mean:.2f} Hz ({np.mean(peak_intervals):.3f}s entre picos)
            - **Regularidade**: {cv_intervals:.1f}% de variação nos intervalos
            - **Amplitude média**: {np.mean(peak_amplitudes):.3f} m/s²
            - **Tendência temporal**: {"Detectada" if p_intervals < 0.05 or p_amps < 0.05 else "Não detectada"}
            
            **Interpretação:**
            """
            
            if cv_intervals < 5:
                summary += "\n- ✅ Movimento altamente regular e rítmico"
            elif cv_intervals < 15:
                summary += "\n- ℹ️ Movimento com regularidade moderada"
            else:
                summary += "\n- ⚠️ Movimento irregular ou variável"
            
            if p_intervals < 0.05 and slope_intervals > 0:
                summary += "\n- ⚠️ Possível fadiga (intervalos aumentam ao longo do tempo)"
            elif p_intervals < 0.05 and slope_intervals < 0:
                summary += "\n- 📈 Aceleração do movimento ao longo do tempo"
            
            if p_amps < 0.05 and slope_amps < 0:
                summary += "\n- 📉 Redução na intensidade do movimento (possível fadiga)"
            elif p_amps < 0.05 and slope_amps > 0:
                summary += "\n- 📈 Aumento na intensidade do movimento"
            
            st.info(summary)
            
        elif num_peaks == 1:
            st.warning("⚠️ Apenas um pico foi detectado. São necessários pelo menos 2 picos para análise de intervalos.")
        else:
            st.warning("⚠️ Nenhum pico foi detectado com os parâmetros atuais. Ajuste os parâmetros de detecção.")

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
