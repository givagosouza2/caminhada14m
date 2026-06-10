import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import plotly.express as px

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Acceleration Data Analyzer", layout="wide")
st.title("📈 Acceleration Data Analyzer")
st.markdown("""
Upload a text file containing 4 columns: **Time**, **Acceleration X**, **Acceleration Y**, and **Acceleration Z**.  
The app will detrend the data, interpolate it to 100 Hz, calculate the acceleration norm, and allow you to select a time window for analysis.
""")

# -----------------------------------------------------------------------------
# File Upload
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Choose a text file", type=["txt", "csv", "dat"])

if uploaded_file is not None:
    try:
        # Read the file (engine='python' allows flexible whitespace/comma separation)
        df = pd.read_csv(uploaded_file, sep=r'\s+|,', engine='python')
        
        # Standardize column names (strip whitespace and make lowercase for robust matching)
        df.columns = df.columns.str.strip().str.lower()
        
        # Map expected columns (adjust these keys if your headers are slightly different)
        col_map = {
            'Tempo (ms)': 'time',
            'Acc X (m/s²)': 'acc_x',
            'Acc Y (m/s²)': 'acc_y',
            'Acc Z (m/s²)': 'acc_z'
        }
        
        # Rename columns to standard names
        df = df.rename(columns=col_map)
        
        # Verify all required columns are present
        required_cols = ['Tempo (ms)', 'Acc X (m/s²)', 'Acc Y (m/s²)', 'Acc Z (m/s²)']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns. Expected: {list(col_map.keys())}. Found: {list(df.columns)}")
            st.stop()

        # Extract data
        t = df['time'].values
        x = df['acc_x'].values
        y = df['acc_y'].values
        z = df['acc_z'].values

        # -----------------------------------------------------------------------------
        # 1. Detrending
        # -----------------------------------------------------------------------------
        x_detrended = signal.detrend(x)
        y_detrended = signal.detrend(y)
        z_detrended = signal.detrend(z)

        # -----------------------------------------------------------------------------
        # 2. Interpolation to 100 Hz
        # -----------------------------------------------------------------------------
        fs_target = 100.0  # Target sampling frequency in Hz
        dt = 1.0 / fs_target
        
        t_min, t_max = t.min(), t.max()
        t_new = np.arange(t_min, t_max + dt, dt)
        
        # Create interpolation functions (linear interpolation, extrapolate if needed)
        interp_x = interp1d(t, x_detrended, kind='linear', fill_value="extrapolate")
        interp_y = interp1d(t, y_detrended, kind='linear', fill_value="extrapolate")
        interp_z = interp1d(t, z_detrended, kind='linear', fill_value="extrapolate")
        
        x_interp = interp_x(t_new)
        y_interp = interp_y(t_new)
        z_interp = interp_z(t_new)

        # -----------------------------------------------------------------------------
        # 3. Calculate Norm
        # -----------------------------------------------------------------------------
        norm = np.sqrt(x_interp**2 + y_interp**2 + z_interp**2)

        # Create a DataFrame for the processed data
        df_processed = pd.DataFrame({
            'Time (s)': t_new,
            'Acc X': x_interp,
            'Acc Y': y_interp,
            'Acc Z': z_interp,
            'Norm': norm
        })

        # -----------------------------------------------------------------------------
        # 4. Plotting Full Data
        # -----------------------------------------------------------------------------
        st.subheader("Full Processed Data")
        fig_full = px.line(
            df_processed, 
            x='Time (s)', 
            y='Norm', 
            title='Acceleration Norm (Detrended & Interpolated to 100 Hz)',
            labels={'Norm': 'Acceleration Norm (m/s² or g)'},
            height=400
        )
        st.plotly_chart(fig_full, use_container_width=True)

        # -----------------------------------------------------------------------------
        # 5. Time Window Selection & Additional Analysis
        # -----------------------------------------------------------------------------
        st.subheader("Time Window Analysis")
        st.markdown("Use the slider below to select a specific time window for detailed analysis.")
        
        min_time = float(t_new.min())
        max_time = float(t_new.max())
        
        # Default to the first 10 seconds or the whole dataset if it's shorter
        default_max = min(min_time + 10.0, max_time)
        
        time_window = st.slider(
            "Select Time Window (seconds)",
            min_value=min_time,
            max_value=max_time,
            value=(min_time, default_max),
            step=0.1,
            format="%.2f"
        )

        # Filter data based on selected window
        mask = (df_processed['Time (s)'] >= time_window[0]) & (df_processed['Time (s)'] <= time_window[1])
        df_window = df_processed[mask]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Norm", f"{df_window['Norm'].mean():.4f}")
        with col2:
            st.metric("Max Norm", f"{df_window['Norm'].max():.4f}")
        with col3:
            st.metric("Min Norm", f"{df_window['Norm'].min():.4f}")
        with col4:
            st.metric("Std Dev", f"{df_window['Norm'].std():.4f}")

        # Plot zoomed window
        fig_window = px.line(
            df_window, 
            x='Time (s)', 
            y='Norm', 
            title=f'Acceleration Norm in Selected Window ({time_window[0]:.2f}s - {time_window[1]:.2f}s)',
            labels={'Norm': 'Acceleration Norm'},
            height=300
        )
        st.plotly_chart(fig_window, use_container_width=True)

        # Optional: Show the raw processed data table for the selected window
        with st.expander("View Processed Data Table (Selected Window)"):
            st.dataframe(df_window, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.info("Please ensure your text file has headers exactly or closely matching: 'Time', 'Acceleration X', 'Acceleration Y', 'Acceleration Z' and uses spaces or commas as separators.")

else:
    st.info("👆 Please upload a text file to begin.")
    
    # Provide a sample format for the user
    st.markdown("### Expected File Format Example:")
    st.code("""Time, Acceleration X, Acceleration Y, Acceleration Z
0.00, 0.12, -0.05, 9.81
0.01, 0.15, -0.04, 9.82
0.02, 0.11, -0.06, 9.80
...""", language="text")
