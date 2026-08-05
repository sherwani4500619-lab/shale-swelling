import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import io

# --- Mathematical Model Functions for the Solver ---
def single_exponential(t, A, k):
    """Single Exponential (First-order kinetic): S(t) = A * (1 - exp(-k * t))"""
    return A * (1.0 - np.exp(-k * np.maximum(t, 0)))

def double_exponential(t, A1, k1, A2, k2):
    """Double Exponential: S(t) = A1*(1 - exp(-k1*t)) + A2*(1 - exp(-k2*t))"""
    t_safe = np.maximum(t, 0)
    return A1 * (1.0 - np.exp(-k1 * t_safe)) + A2 * (1.0 - np.exp(-k2 * t_safe))

def weibull_model(t, A, beta, alpha):
    """Weibull Model: S(t) = A * (1 - exp(-(t / beta)^alpha))"""
    t_safe = np.maximum(t, 0)
    return A * (1.0 - np.exp(-((t_safe / np.maximum(beta, 1e-6)) ** np.maximum(alpha, 1e-4))))

def logistic_model(t, A, k, t0):
    """Logistic Model: S(t) = A / (1 + exp(-k * (t - t0)))"""
    return A / (1.0 + np.exp(-k * (t - t0)))

def gompertz_model(t, A, b, M):
    """Gompertz Model: S(t) = A * exp(-exp(-b * (t - M)))"""
    return A * np.exp(-np.exp(-b * (t - M)))

def power_law(t, K, n):
    """Power Law Model: S(t) = K * t^n"""
    return K * (np.maximum(t, 0) ** n)

def peleg_model(t, k1, k2):
    """Peleg Model: S(t) = t / (k1 + k2 * t)"""
    t_safe = np.maximum(t, 0)
    return t_safe / (k1 + k2 * t_safe + 1e-6)

# --- Page Configuration ---
st.set_page_config(
   page_title="Shale Swelling Advanced Model & Comparison", 
   page_icon="🪨",
   layout="wide", 
   initial_sidebar_state="expanded"
)
 
# --- Header Section ---
st.title("🪨 Advanced Shale Swelling Prediction & Multi-Model Comparison Dashboard")
st.markdown("Compare experimental lab data against all 7 standard kinetic and sorption models with robust curve fitting, statistical evaluation, and dark experimental data contrast.")
st.divider()
 
# --- Sidebar: Equation Selection ---
st.sidebar.title("⚙️ Model Parameters")
equation_choice = st.sidebar.selectbox(
    "1. Select Evaluation Mode", 
    [
        "Empirical CEC Model (Theoretical)", 
        "Single Exponential (First-order)",
        "Double Exponential",
        "Weibull Model",
        "Logistic Model",
        "Gompertz Model",
        "Power Law",
        "Peleg Model",
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
            line_chart = alt.Chart(chart_df).mark_line(strokeWidth=3, color='#1f77b4').encode(
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
        
        raw_time = lab_data.iloc[:, 1]  # Elapsed Time column
        s_exp = pd.to_numeric(lab_data.iloc[:, 2], errors='coerce').values  # Swelling (%) column
        
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
 
        st.markdown(f"### 📊 Auto-Solver & Comparison Results (Sheet: {sheet_name})")
        
        # Performance downsampling for swift curve fitting
        step = max(1, len(t_exp) // 300)
        t_fit_x = t_exp[::step]
        s_fit_y = s_exp[::step]

        df_chart = pd.DataFrame({"Time (seconds)": t_exp, "Actual Experimental Swelling": s_exp})
        metrics_list = []
        
        # 1. Single Exponential
        if equation_choice in ["Single Exponential (First-order)", "Compare All Auto-Fit Models"]:
            try:
                popt, _ = curve_fit(single_exponential, t_fit_x, s_fit_y, p0=[max(s_fit_y), 1e-4], bounds=(0, np.inf), maxfev=5000)
                y_pred_full = single_exponential(t_exp, *popt)
                df_chart["Single Exponential"] = y_pred_full
                rmse = np.sqrt(mean_squared_error(s_exp, y_pred_full))
                r2 = r2_score(s_exp, y_pred_full)
                metrics_list.append({"Model": "Single Exponential", "RMSE (%)": round(rmse, 4), "R² Score": round(r2, 4)})
            except Exception as e:
                st.warning(f"Single Exponential fit skipped: {e}")

        # 2. Double Exponential
        if equation_choice in ["Double Exponential", "Compare All Auto-Fit Models"]:
            try:
                p0_double = [max(s_fit_y)*0.6, 1e-3, max(s_fit_y)*0.4, 1e-5]
                popt, _ = curve_fit(double_exponential, t_fit_x, s_fit_y, p0=p0_double, bounds=(0, np.inf), maxfev=10000)
                y_pred_full = double_exponential(t_exp, *popt)
                df_chart["Double Exponential"] = y_pred_full
                rmse = np.sqrt(mean_squared_error(s_exp, y_pred_full))
                r2 = r2_score(s_exp, y_pred_full)
                metrics_list.append({"Model": "Double Exponential", "RMSE (%)": round(rmse, 4), "R² Score": round(r2, 4)})
            except Exception as e:
                st.warning(f"Double Exponential fit skipped: {e}")

        # 3. Weibull Model
        if equation_choice in ["Weibull Model", "Compare All Auto-Fit Models"]:
            try:
                p0_weibull = [max(s_fit_y), np.median(t_fit_x), 1.0]
                popt, _ = curve_fit(weibull_model, t_fit_x, s_fit_y, p0=p0_weibull, bounds=([0, 0, 0], [np.inf, np.inf, 5]), maxfev=8000)
                y_pred_full = weibull_model(t_exp, *popt)
                df_chart["Weibull Model"] = y_pred_full
                rmse = np.sqrt(mean_squared_error(s_exp, y_pred_full))
                r2 = r2_score(s_exp, y_pred_full)
                metrics_list.append({"Model": "Weibull Model", "RMSE (%)": round(rmse, 4), "R² Score": round(r2, 4)})
            except Exception as e:
                st.warning(f"Weibull Model fit skipped: {e}")

        # 4. Logistic Model
        if equation_choice in ["Logistic Model", "Compare All Auto-Fit Models"]:
            try:
                p0_logistic = [max(s_fit_y), 1e-4, np.median(t_fit_x)]
                popt, _ = curve_fit(logistic_model, t_fit_x, s_fit_y, p0=p0_logistic, bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]), maxfev=8000)
                y_pred_full = logistic_model(t_exp, *popt)
                df_chart["Logistic Model"] = y_pred_full
                rmse = np.sqrt(mean_squared_error(s_exp, y_pred_full))
                r2 = r2_score(s_exp, y_pred_full)
                metrics_list.append({"Model": "Logistic Model", "RMSE (%)": round(rmse, 4), "R² Score": round(r2, 4)})
            except Exception as e:
                st.warning(f"Logistic Model fit skipped: {e}")

        # 5. Gompertz Model
        if equation_choice in ["Gompertz Model", "Compare All Auto-Fit Models"]:
            try:
                p0_gomp = [max(s_fit_y), 1e-4, np.median(t_fit_x)]
                popt, _ = curve_fit(gompertz_model, t_fit_x, s_fit_y, p0=p0_gomp, bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]), maxfev=8000)
                y_pred_full = gompertz_model(t_exp, *popt)
                df_chart["Gompertz Model"] = y_pred_full
                rmse = np.sqrt(mean_squared_error(s_exp, y_pred_full))
                r2 = r2_score(s_exp, y_pred_full)
                metrics_list.append({"Model": "Gompertz Model", "RMSE (%)": round(rmse, 4), "R² Score": round(r2, 4)})
            except Exception as e:
                st.warning(f"Gompertz Model fit skipped: {e}")

        # 6. Power Law
        if equation_choice in ["Power Law", "Compare All Auto-Fit Models"]:
            try:
                p0_power = [0.1, 0.5]
                popt, _ = curve_fit(power_law, t_fit_x, s_fit_y, p0=p0_power, bounds=(0, [np.inf, 2.0]), maxfev=5000)
                y_pred_full = power_law(t_exp, *popt)
                df_chart["Power Law"] = y_pred_full
                rmse = np.sqrt(mean_squared_error(s_exp, y_pred_full))
                r2 = r2_score(s_exp, y_pred_full)
                metrics_list.append({"Model": "Power Law", "RMSE (%)": round(rmse, 4), "R² Score": round(r2, 4)})
            except Exception as e:
                st.warning(f"Power Law fit skipped: {e}")

        # 7. Peleg Model
        if equation_choice in ["Peleg Model", "Compare All Auto-Fit Models"]:
            try:
                p0_peleg = [10.0, 0.1]
                popt, _ = curve_fit(peleg_model, t_fit_x, s_fit_y, p0=p0_peleg, bounds=(0, np.inf), maxfev=5000)
                y_pred_full = peleg_model(t_exp, *popt)
                df_chart["Peleg Model"] = y_pred_full
                rmse = np.sqrt(mean_squared_error(s_exp, y_pred_full))
                r2 = r2_score(s_exp, y_pred_full)
                metrics_list.append({"Model": "Peleg Model", "RMSE (%)": round(rmse, 4), "R² Score": round(r2, 4)})
            except Exception as e:
                st.warning(f"Peleg Model fit skipped: {e}")
 
        metrics_df = pd.DataFrame(metrics_list)
        st.divider()
 
        tab1, tab2, tab3 = st.tabs(["📈 Data vs. Model Comparison", "📊 Statistical Performance", "💾 Export Data"])
        
        with tab1:
            st.markdown("*(Actual experimental swelling is highlighted with a bold dark line/marker for clear comparison against fitted models)*")
            
            melted_df = df_chart.melt(
                id_vars=["Time (seconds)"], 
                var_name="Legend / Model", 
                value_name="Swelling (%)"
            )
            
            # Custom color scale to keep Actual Experimental Data dark/black
            domain_list = melted_df["Legend / Model"].unique().tolist()
            range_list = []
            for name in domain_list:
                if "Actual" in name:
                    range_list.append("#000000")  # Solid Black for Actual Data
                else:
                    range_list.append(
                        "#1f77b4" if len(range_list) == 0 else 
                        "#ff7f0e" if len(range_list) == 1 else 
                        "#2ca02c" if len(range_list) == 2 else 
                        "#d62728" if len(range_list) == 3 else 
                        "#9467bd" if len(range_list) == 4 else 
                        "#8c564b" if len(range_list) == 5 else "#e377c2"
                    )

            base_chart = alt.Chart(melted_df).mark_line(strokeWidth=2.5).encode(
                x=alt.X('Time (seconds):Q', title='Time (seconds)'),
                y=alt.Y('Swelling (%):Q', title='Swelling (%)'),
                color=alt.Color('Legend / Model:N', scale=alt.Scale(domain=domain_list, range=range_list), title='Models & Data'),
                tooltip=['Time (seconds):Q', 'Swelling (%):Q', 'Legend / Model:N']
            ).properties(width=700, height=480).interactive()
            
            st.altair_chart(base_chart, use_container_width=True)
            
        with tab2:
            st.markdown("#### Model Accuracy Metrics Comparison")
            if not metrics_df.empty:
                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.info("No models successfully evaluated.")
            
        with tab3:
            st.markdown("#### Download Fitted Data & Performance Metrics")
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_chart.to_excel(writer, sheet_name='Model Predictions', index=False)
                if not metrics_df.empty:
                    metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
            excel_data = output.getvalue()
            
            st.download_button(
                label="📥 Download Complete Results as Excel (.xlsx)",
                data=excel_data,
                file_name="shale_swelling_all_models_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
 
    except Exception as e:
        st.error(f"⚠️ Error fitting data: {e}")
