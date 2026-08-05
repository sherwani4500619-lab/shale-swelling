import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import io

# --- Mathematical Model Functions for the Solver ---
def custom_model(t, A, tau):
    """Equation 2: Custom Asymptotic Swelling Model"""
    x = t / tau
    exp_val = 0.671205
    return A * (np.abs(x)**exp_val) / (1 + (np.abs(x)**exp_val))
 
def higuchi_model(t, K_H):
    """Higuchi Model: S(t) = K_H * sqrt(t)"""
    return K_H * np.sqrt(np.maximum(t, 0))
 
def peppas_model(t, K_P, n):
    """Korsmeyer-Peppas Model: S(t) = K_P * t^n"""
    return K_P * (np.maximum(t, 0) ** n)
 
# --- Page Configuration ---
st.set_page_config(
   page_title="Shale Swelling Model", 
   page_icon="🪨",
   layout="wide", 
   initial_sidebar_state="expanded"
)
 
# --- Header Section ---
st.title("🪨 Shale Swelling Prediction Model & Optimization Dashboard")
st.markdown("Calculate theoretical shale swelling or upload lab data to automatically fit empirical models with explicit axis labeling and statistical evaluation.")
st.divider()
 
# --- Sidebar: Equation Selection ---
st.sidebar.title("⚙️ Model Parameters")
equation_choice = st.sidebar.selectbox(
    "1. Select Swelling Equation", 
    [
        "Empirical CEC Model (Theoretical)", 
        "Custom Equation (Auto-Fit)",
        "Higuchi Model (Auto-Fit)",
        "Korsmeyer-Peppas Model (Auto-Fit)",
        "Compare All Auto-Fit Models"
    ]
)
 
# ==========================================
# EQUATION 1: EMPIRICAL CEC MODEL (NO DATA UPLOAD REQUIRED)
# ==========================================
if equation_choice == "Empirical CEC Model (Theoretical)":
    with st.sidebar.expander("2. Rock Properties", expanded=True):
        cec = st.number_input("CEC (meq/100 g)", value=25.0, step=0.5)
        sm = st.number_input("Smectite (Sm)", value=20.0, step=0.5)
        q = st.number_input("Quartz (Q)", value=35.0, step=0.5)
        dol = st.number_input("Dolomite (Dol)", value=10.0, step=0.5)
        cal = st.number_input("Calcite (Cal)", value=15.0, step=0.5)
        hal = st.number_input("Halite (Hal)", value=5.0, step=0.5)
 
    with st.sidebar.expander("3. Time Settings", expanded=True):
        time_unit = st.selectbox("Input Time Unit", ["Hours", "Minutes", "Seconds", "Days"], index=0)
        max_time_input = st.number_input(f"Simulation Duration ({time_unit})", min_value=0.1, value=24.0, step=1.0)
 
    unit_to_seconds = {"Seconds": 1.0, "Minutes": 60.0, "Hours": 3600.0, "Days": 86400.0}
    max_time_sec = max_time_input * unit_to_seconds[time_unit]
    t_sec = np.arange(0, max_time_sec + 1, 10)
    denominator = q + dol + cal + hal
 
    if denominator == 0:
        st.error("⚠️ Error: The sum of non-swelling minerals cannot be zero.")
    else:
        R = (cec * sm) / denominator
        A = 4.211172 + 0.915226 * (R ** 1.957788)
        tau = 1089.786186 + 4877.982915 * (R ** 1.096567)
        
        x = t_sec / tau
        exp_num = 0.565351
        exp_denom = 1.446685
        swelling = A * (np.abs(x) ** exp_num) / ((1 + (np.abs(x) ** exp_denom)) ** (exp_num / exp_denom))
 
        st.markdown("### 📊 Final Output Summary")
        with st.info(f"Using {equation_choice}:"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Mineral Factor (R)", f"{R:.4f}")
            col2.metric("Max Capacity (A)", f"{A:.4f} %")
            col3.metric("Characteristic Time (τ)", f"{tau:.2f} s")
 
        chart_df = pd.DataFrame({
            "Time (seconds)": t_sec,
            "Theoretical Swelling": swelling
        })
        
        tab1, tab2 = st.tabs(["📈 Interactive Swelling Chart", "💾 Export Data"])
        with tab1:
            line_chart = alt.Chart(chart_df).mark_line(strokeWidth=2).encode(
                x=alt.X('Time (seconds):Q', title='Time (seconds)'),
                y=alt.Y('Theoretical Swelling:Q', title='Swelling (%)'),
                tooltip=['Time (seconds):Q', 'Theoretical Swelling:Q']
            ).properties(width=700, height=450).interactive()
            st.altair_chart(line_chart, use_container_width=True)
            
        with tab2:
            st.download_button(
                "📥 Download Results (CSV)",
                data=chart_df.to_csv(index=False).encode('utf-8'),
                file_name="cec_model_results.csv", 
                mime="text/csv"
            )
 
# ==========================================
# AUTO-FIT MODELS (DATA UPLOAD REQUIRED)
# ==========================================
else:
    st.sidebar.markdown("### 2. Upload Lab Data")
    st.sidebar.caption("Upload a CSV or Excel file. Column 1 must be Time (seconds), Column 2 must be Swelling (%).")
    uploaded_file = st.sidebar.file_uploader("Upload Experimental Data", type=["csv", "xlsx"])
 
    if uploaded_file is None:
        st.info("👋 Please upload your experimental data (CSV or Excel) in the sidebar to run the auto-solver.")
        st.stop()
 
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_name = st.sidebar.selectbox("Select Cell Sheet", excel_file.sheet_names)
        
        lab_data = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=14)
        lab_data.columns = lab_data.columns.astype(str).str.strip()
        
        raw_time = lab_data.iloc[:, 1]  # Elap Time column
        s_exp = pd.to_numeric(lab_data.iloc[:, 2], errors='coerce').values  # Swell (%)
        
        def time_to_sec(t_str):
            try:
                if isinstance(t_str, (int, float)):
                    return float(t_str)
                parts = str(t_str).split(':')
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                elif len(parts) == 2:
                    return int(parts[0]) * 60 + float(parts[1])
                return float(t_str)
            except:
                return np.nan
                 
        t_sec = np.array([time_to_sec(t) for t in raw_time])
        
        valid_idx = ~np.isnan(t_sec) & ~np.isnan(s_exp)
        t_exp = t_sec[valid_idx]
        s_exp = s_exp[valid_idx]
 
        st.markdown(f"### 📊 Auto-Solver Results (Sheet: {sheet_name})")
        
        # Performance downsampling for fast solving
        step = max(1, len(t_exp) // 300)
        t_fit_x = t_exp[::step]
        s_fit_y = s_exp[::step]

        df_chart = pd.DataFrame({"Time (seconds)": t_exp, "Experimental Data": s_exp})
        metrics_list = []
        
        # 1. Custom Model Fit
        if equation_choice in ["Custom Equation (Auto-Fit)", "Compare All Auto-Fit Models"]:
            try:
                p0_custom = [max(s_fit_y), np.median(t_fit_x)] 
                popt_custom, _ = curve_fit(custom_model, t_fit_x, s_fit_y, p0=p0_custom, bounds=(0, np.inf), maxfev=5000)
                fit_A, fit_tau = popt_custom
                
                y_pred_full = custom_model(t_exp, fit_A, fit_tau)
                df_chart["Custom Model Fit"] = y_pred_full
                
                rmse = np.sqrt(mean_squared_error(s_exp, y_pred_full))
                r2 = r2_score(s_exp, y_pred_full)
                metrics_list.append({"Model": "Custom Model", "RMSE (%)": round(rmse, 4), "R² Score": round(r2, 4)})
                
                with st.success("Custom Model Parameters:"):
                    c1, c2 = st.columns(2)
                    c1.metric("Fitted Max Capacity (A)", f"{fit_A:.4f} %")
                    c2.metric("Fitted Time Constant (τ)", f"{fit_tau:.2f} s")
            except Exception as e:
                st.warning(f"Custom Model fit skipped: {e}")
 
        # 2. Higuchi Model Fit
        if equation_choice in ["Higuchi Model (Auto-Fit)", "Compare All Auto-Fit Models"]:
            try:
                popt_higuchi, _ = curve_fit(higuchi_model, t_fit_x, s_fit_y, bounds=(0, np.inf), maxfev=5000)
                fit_KH = popt_higuchi[0]
                
                y_pred_full = higuchi_model(t_exp, fit_KH)
                df_chart["Higuchi Fit"] = y_pred_full
                
                rmse = np.sqrt(mean_squared_error(s_exp, y_pred_full))
                r2 = r2_score(s_exp, y_pred_full)
                metrics_list.append({"Model": "Higuchi Model", "RMSE (%)": round(rmse, 4), "R² Score": round(r2, 4)})
                
                with st.info("Higuchi Model Parameters:"):
                    st.metric("Fitted Constant (K_H)", f"{fit_KH:.6f}")
            except Exception as e:
                st.warning(f"Higuchi Model fit skipped: {e}")
 
        # 3. Korsmeyer-Peppas Model Fit
        if equation_choice in ["Korsmeyer-Peppas Model (Auto-Fit)", "Compare All Auto-Fit Models"]:
            try:
                p0_peppas = [0.1, 0.5]
                popt_peppas, _ = curve_fit(peppas_model, t_fit_x, s_fit_y, p0=p0_peppas, bounds=(0, [np.inf, 2.0]), maxfev=5000)
                fit_KP, fit_n = popt_peppas
                
                y_pred_full = peppas_model(t_exp, fit_KP, fit_n)
                df_chart["Peppas Fit"] = y_pred_full
                
                rmse = np.sqrt(mean_squared_error(s_exp, y_pred_full))
                r2 = r2_score(s_exp, y_pred_full)
                metrics_list.append({"Model": "Korsmeyer-Peppas", "RMSE (%)": round(rmse, 4), "R² Score": round(r2, 4)})
                
                with st.warning("Korsmeyer-Peppas Parameters:"):
                    c1, c2 = st.columns(2)
                    c1.metric("Fitted Constant (K_P)", f"{fit_KP:.6f}")
                    c2.metric("Release Exponent (n)", f"{fit_n:.4f}")
            except Exception as e:
                st.warning(f"Korsmeyer-Peppas fit skipped: {e}")
 
        metrics_df = pd.DataFrame(metrics_list)
        st.divider()
 
        tab1, tab2, tab3 = st.tabs(["📈 Data vs. Model Comparison", "📊 Statistical Performance", "💾 Export Data"])
        
        with tab1:
            st.markdown("*(Explicit Axis Labeling: Time in seconds vs Swelling percentage)*")
            
            melted_df = df_chart.melt(
                id_vars=["Time (seconds)"], 
                var_name="Legend / Model", 
                value_name="Swelling (%)"
            )
            
            line_chart = alt.Chart(melted_df).mark_line(strokeWidth=2).encode(
                x=alt.X('Time (seconds):Q', title='Time (seconds)'),
                y=alt.Y('Swelling (%):Q', title='Swelling (%)'),
                color=alt.Color('Legend / Model:N', title='Models & Data'),
                tooltip=['Time (seconds):Q', 'Swelling (%):Q', 'Legend / Model:N']
            ).properties(width=700, height=450).interactive()
            
            st.altair_chart(line_chart, use_container_width=True)
            
        with tab2:
            st.markdown("#### Model Accuracy Metrics")
            if not metrics_df.empty:
                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.info("No models successfully evaluated.")
            
        with tab3:
            st.markdown("#### Download Fitted Data & Metrics")
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_chart.to_excel(writer, sheet_name='Model Predictions', index=False)
                if not metrics_df.empty:
                    metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
            excel_data = output.getvalue()
            
            st.download_button(
                label="📥 Download Complete Results as Excel (.xlsx)",
                data=excel_data,
                file_name="auto_fitted_models_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
 
    except Exception as e:
        st.error(f"⚠️ Error fitting data: {e}")
