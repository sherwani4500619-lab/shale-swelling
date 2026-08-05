import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# --- Mathematical Model Functions for the Solver ---
def custom_model(t, A, tau):
    """Equation 2: Custom Asymptotic Swelling Model"""
    x = t / tau
    exp_val = 0.671205
    return A * (np.abs(x)**exp_val) / (1 + (np.abs(x)**exp_val))

def higuchi_model(t, K_H):
    """Higuchi Model: S(t) = K_H * sqrt(t)"""
    return K_H * np.sqrt(t)

def peppas_model(t, K_P, n):
    """Korsmeyer-Peppas Model: S(t) = K_P * t^n"""
    return K_P * (t ** n)

# --- Page Configuration ---
st.set_page_config(
    page_title="Shale Swelling Model", 
    page_icon="🪨", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Header Section ---
st.title("🪨 Shale Swelling Prediction Model")
st.markdown("Calculate theoretical shale swelling or upload lab data to automatically fit empirical models.")
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

    # Calculate Model
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

        chart_data = pd.DataFrame({"Time (in hrs)": t_sec / 3600.0, "Predicted Swelling": swelling}).set_index("Time (in hrs)")
        
        tab1, tab2 = st.tabs(["📈 Interactive Swelling Chart", "💾 Export Data"])
        with tab1:
            st.line_chart(chart_data)
        with tab2:
            results_df = pd.DataFrame({"Time (in sec)": t_sec, "Predicted Swelling": swelling})
            st.download_button("📥 Download Results (CSV)", data=results_df.to_csv(index=False).encode('utf-8'), file_name="cec_model_results.csv", mime="text/csv")

# ==========================================
# AUTO-FIT MODELS (DATA UPLOAD REQUIRED)
# ==========================================
else:
    st.sidebar.markdown("### 2. Upload Lab Data")
    st.sidebar.caption("Upload a CSV file. Column 1 must be Time (seconds), Column 2 must be Swelling (%).")
    uploaded_file = st.sidebar.file_uploader("Upload Experimental Data (CSV)", type=["csv"])

    if uploaded_file is None:
        st.info("👋 Please upload your experimental CSV data in the sidebar to run the auto-solver.")
        st.stop() # Halts the app cleanly until data is uploaded

    try:
        # Read the uploaded data
        lab_data = pd.read_csv(uploaded_file)
        t_exp = lab_data.iloc[:, 0].values # First column: Time
        s_exp = lab_data.iloc[:, 1].values # Second column: Swelling

        st.markdown("### 📊 Auto-Solver Results")
        
        # Create a dataframe for the graph. We start with the experimental data.
        df_chart = pd.DataFrame({"Time (sec)": t_exp, "Experimental Data": s_exp})
        
        # 1. Custom Model Fit
        if equation_choice in ["Custom Equation (Auto-Fit)", "Compare All Auto-Fit Models"]:
            # Initial guesses for A and tau
            p0_custom = [max(s_exp), np.median(t_exp)] 
            popt_custom, _ = curve_fit(custom_model, t_exp, s_exp, p0=p0_custom, bounds=(0, np.inf))
            
            fit_A, fit_tau = popt_custom
            df_chart["Custom Model Fit"] = custom_model(t_exp, fit_A, fit_tau)
            
            with st.success("Custom Model Parameters:"):
                c1, c2 = st.columns(2)
                c1.metric("Fitted Max Capacity (A)", f"{fit_A:.4f} %")
                c2.metric("Fitted Time Constant (τ)", f"{fit_tau:.2f} s")

        # 2. Higuchi Model Fit
        if equation_choice in ["Higuchi Model (Auto-Fit)", "Compare All Auto-Fit Models"]:
            popt_higuchi, _ = curve_fit(higuchi_model, t_exp, s_exp, bounds=(0, np.inf))
            fit_KH = popt_higuchi[0]
            df_chart["Higuchi Fit"] = higuchi_model(t_exp, fit_KH)
            
            with st.info("Higuchi Model Parameters:"):
                st.metric("Fitted Constant (K_H)", f"{fit_KH:.6f}")

        # 3. Korsmeyer-Peppas Model Fit
        if equation_choice in ["Korsmeyer-Peppas Model (Auto-Fit)", "Compare All Auto-Fit Models"]:
            p0_peppas = [0.1, 0.5] # Guesses for K_P and n
            popt_peppas, _ = curve_fit(peppas_model, t_exp, s_exp, p0=p0_peppas, bounds=(0, [np.inf, 2.0]))
            
            fit_KP, fit_n = popt_peppas
            df_chart["Peppas Fit"] = peppas_model(t_exp, fit_KP, fit_n)
            
            with st.warning("Korsmeyer-Peppas Parameters:"):
                c1, c2 = st.columns(2)
                c1.metric("Fitted Constant (K_P)", f"{fit_KP:.6f}")
                c2.metric("Release Exponent (n)", f"{fit_n:.4f}")

        st.divider()

        # Render Visuals
        tab1, tab2 = st.tabs(["📈 Data vs. Model Comparison", "💾 Export Fitted Data"])
        
        with tab1:
            st.markdown("*(Note: For clarity, the chart below displays time in seconds matching your uploaded lab data)*")
            # Set index to Time for Streamlit to chart correctly
            st.line_chart(df_chart.set_index("Time (sec)"))
            
        with tab2:
            st.markdown("#### Download Fitted Data")
            csv_data = df_chart.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Model Comparison (CSV)",
                data=csv_data,
                file_name="auto_fitted_models.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"⚠️ An error occurred while fitting the data. Please ensure your CSV contains only numbers and no text headers in the data rows. Error details: {e}")
