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
    x = t / np.maximum(tau, 1e-6)
    exp_val = 0.5
    return A * (np.abs(x)**exp_val) / (1.0 + (np.abs(x)**exp_val))

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
st.markdown("Compare experimental lab data against your Custom Asymptotic Model and other kinetic models with multi-select comparison, sorted light-shaded performance charts with value labels, and Excel export.")
st.divider()
 
# --- Sidebar: Mode Selection ---
st.sidebar.title("⚙️ Model Parameters")
evaluation_mode = st.sidebar.radio(
    "1. Select Evaluation Mode", 
    [
        "Empirical CEC Model (Theoretical)", 
        "Experimental Data Fitting & Comparison"
    ]
)
 
# ==========================================
# EQUATION 1: EMPIRICAL CEC MODEL (NO DATA UPLOAD REQUIRED)
# ==========================================
if evaluation_mode == "Empirical CEC Model (Theoretical)":
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
        with st.info(f"Using Empirical CEC Model (Theoretical):"):
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
# EXPERIMENTAL DATA FITTING & MULTI-MODEL SELECTION
# ==========================================
else:
    st.sidebar.markdown("### 2. Upload Lab Data")
    st.sidebar.caption("Upload a CSV or Excel file. Column 1 must be Time (seconds), Column 2 must be Swelling (%).")
    uploaded_file = st.sidebar.file_uploader("Upload Experimental Data", type=["csv", "xlsx"])
 
    if uploaded_file is None:
        st.info("👋 Please upload your experimental data (CSV or Excel) in the sidebar to run the auto-solver.")
        st.stop()
        
    # Sidebar Model Multiselect Checklist
    st.sidebar.markdown("### 3. Select Models to Compare")
    available_models = [
        "Custom Asymptotic Model",
        "Single Exponential (First-order)",
        "Double Exponential",
        "Weibull Model",
        "Logistic Model",
        "Gompertz Model",
        "Power Law",
        "Peleg Model"
    ]
    selected_models = st.sidebar.multiselect(
        "Choose models for evaluation:",
        options=available_models,
        default=["Custom Asymptotic Model", "Single Exponential (First-order)"]
    )
 
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
 
        st.markdown(f"### 📊 Auto-Solver & Multi-Model Comparison (Sheet: {sheet_name})")
        
        if not selected_models:
            st.warning("⚠️ Please select at least one model from the sidebar multiselect box.")
            st.stop()
            
        # Performance downsampling for swift curve fitting
        step = max(1, len(t_exp) // 300)
        t_fit_x = t_exp[::step]
        s_fit_y = s_exp[::step]

        df_chart = pd.DataFrame({"Time (seconds)": t_exp, "Actual Experimental Swelling": s_exp})
        metrics_list = []
        
        # 1. Custom Asymptotic Model
        if "Custom Asymptotic Model" in selected_models:
            try:
                p0_custom = [max(s_fit_y), np.median(t_fit_x)] 
                popt, _ = curve_fit(custom_model, t_fit_x, s_fit_y, p0=p0_custom, bounds=(0, np.inf), maxfev=5000)
                y_pred_full = custom_model(t_exp, *popt)
                df_chart["Custom Asymptotic Model"] = y_pred_full
                rmse = np.sqrt(mean_squared_error(s_exp, y_pred_full))
                r2 = r2_score(s_exp, y_pred_full)
                metrics_list.append({"Model": "Custom Asymptotic Model", "RMSE (%)": round(rmse, 4), "R² Score": round(r2, 4)})
            except Exception as e:
                st.warning(f"Custom Asymptotic Model fit skipped: {e}")

        # 2. Single Exponential
        if "Single Exponential (First-order)" in selected_models:
            try:
                popt, _ = curve_fit(single_exponential, t_fit_x, s_fit_y, p0=[max(s_fit_y), 1e-4], bounds=(0, np.inf), maxfev=5000)
                y_pred_full = single_exponential(t_exp, *popt)
                df_chart["Single Exponential"] = y_pred_full
                rmse = np.sqrt(mean_squared_error(s_exp, y_pred_full))
                r2 = r2_score(s_exp, y_pred_full)
                metrics_list.append({"Model": "Single Exponential", "RMSE (%)": round(rmse, 4), "R² Score": round(r2, 4)})
            except Exception as e:
                st.warning(f"Single Exponential fit skipped: {e}")

        # 3. Double Exponential
        if "Double Exponential" in selected_models:
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

        # 4. Weibull Model
        if "Weibull Model" in selected_models:
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

        # 5. Logistic Model
        if "Logistic Model" in selected_models:
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

        # 6. Gompertz Model
        if "Gompertz Model" in selected_models:
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

        # 7. Power Law
        if "Power Law" in selected_models:
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

        # 8. Peleg Model
        if "Peleg Model" in selected_models:
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
            st.markdown("### 📈 Swelling vs. Time Model Comparison")
            st.markdown("*(Actual experimental swelling is highlighted with a bold dark line/marker)*")
            
            melted_df = df_chart.melt(
                id_vars=["Time (seconds)"], 
                var_name="Legend / Model", 
                value_name="Swelling (%)"
            )
            
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
            ).properties(width=700, height=450).interactive()
            
            st.altair_chart(base_chart, use_container_width=True)
            
            # --- SORTED LIGHT-SHADED R² AND RMSE BAR CHARTS WITH VALUE LABELS ---
            if not metrics_df.empty:
                st.markdown("### 📊 Model Performance Comparison Bar Charts")
                
                col_bar1, col_bar2 = st.columns(2)
                
                with col_bar1:
                    st.markdown("#### Root Mean Squared Error (RMSE %)")
                    
                    # Base bar chart (lightly shaded with opacity=0.6, sorted descending)
                    bars_rmse = alt.Chart(metrics_df).mark_bar(color='#d62728', opacity=0.6).encode(
                        x=alt.X('Model:N', sort='-y', title='Model', axis=alt.Axis(labelAngle=-20)),
                        y=alt.Y('RMSE (%):Q', title='RMSE (%)'),
                        tooltip=['Model:N', 'RMSE (%):Q']
                    )
                    # Text labels on top of bars
                    text_rmse = bars_rmse.mark_text(
                        align='center',
                        baseline='bottom',
                        dy=-4,
                        fontSize=11,
                        color='black'
                    ).encode(text=alt.Text('RMSE (%):Q', format='.3f'))
                    
                    rmse_final = (bars_rmse + text_rmse).properties(width=320, height=320).interactive()
                    st.altair_chart(rmse_final, use_container_width=True)
                
                with col_bar2:
                    st.markdown("#### Coefficient of Determination (R² Score)")
                    
                    # Base bar chart (lightly shaded with opacity=0.6, sorted descending)
                    bars_r2 = alt.Chart(metrics_df).mark_bar(color='#2ca02c', opacity=0.6).encode(
                        x=alt.X('Model:N', sort='-y', title='Model', axis=alt.Axis(labelAngle=-20)),
                        y=alt.Y('R² Score:Q', title='R² Score'),
                        tooltip=['Model:N', 'R² Score:Q']
                    )
                    # Text labels on top of bars
                    text_r2 = bars_r2.mark_text(
                        align='center',
                        baseline='bottom',
                        dy=-4,
                        fontSize=11,
                        color='black'
                    ).encode(text=alt.Text('R² Score:Q', format='.3f'))
                    
                    r2_final = (bars_r2 + text_r2).properties(width=320, height=320).interactive()
                    st.altair_chart(r2_final, use_container_width=True)
            
        with tab2:
            st.markdown("#### Model Accuracy Metrics Comparison Table")
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
                file_name="shale_swelling_selected_models_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
 
    except Exception as e:
        st.error(f"⚠️ Error fitting data: {e}")
