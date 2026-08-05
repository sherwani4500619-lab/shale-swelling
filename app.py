import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- Mathematical Model Functions for the Solver ---
def custom_model(t, A, tau):
    """Equation 2: Custom Asymptotic Swelling Model"""
    x = t / np.maximum(tau, 1e-6)
    exp_val = 0.671205
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

# --- Page Configuration & Dark Theme ---
st.set_page_config(
   page_title="Shale Swelling Advanced Analysis Suite", 
   page_icon="🪨",
   layout="wide", 
   initial_sidebar_state="expanded"
)

# Custom Styling for Dark Engineering Theme
st.markdown("""
    <style>
        .stButton>button { width: 100%; border-radius: 6px; font-weight: 600; background-color: #1f6feb; color: white; border: none; }
        .stButton>button:hover { background-color: #388bfd; color: white; }
        .stDownloadButton>button { width: 100%; border-radius: 6px; font-weight: 600; background-color: #238636; color: white; border: none; }
        .stDownloadButton>button:hover { background-color: #2ea043; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.title("🪨 Shale Swelling Kinetic Analysis & Multi-Model Suite")
st.markdown("High-performance engineering dashboard for comparative curve fitting, theoretical XRD/CEC estimation, and statistical evaluation.")
st.divider()

# --- Sidebar: Control Panel ---
st.sidebar.title("⚙️ Control Panel")
evaluation_mode = st.sidebar.radio(
    "Select Workflow Mode", 
    [
        "Empirical CEC Model (Theoretical)", 
        "Experimental Data Fitting & Comparison"
    ]
)

# ==========================================
# WORKFLOW 1: EMPIRICAL CEC MODEL (THEORETICAL)
# ==========================================
if evaluation_mode == "Empirical CEC Model (Theoretical)":
    st.sidebar.markdown("---")
    with st.sidebar.expander("🧪 Rock Mineralogy & CEC", expanded=True):
        cec = st.number_input("CEC (meq/100 g)", value=25.0, step=0.5)
        sm = st.number_input("Smectite (Sm %)", value=20.0, step=0.5)
        q = st.number_input("Quartz (Q %)", value=35.0, step=0.5)
        dol = st.number_input("Dolomite (Dol %)", value=10.0, step=0.5)
        cal = st.number_input("Calcite (Cal %)", value=15.0, step=0.5)
        hal = st.number_input("Halite (Hal %)", value=5.0, step=0.5)
 
    with st.sidebar.expander("⏱️ Simulation Settings", expanded=True):
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
 
        st.markdown("### 📊 Theoretical Model Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mineral Factor (R)", f"{R:.4f}")
        with col2:
            st.metric("Max Capacity (A)", f"{A:.4f} %")
        with col3:
            st.metric("Characteristic Time (τ)", f"{tau:.2f} s")
 
        chart_df = pd.DataFrame({
            "Time (seconds)": t_sec,
            "Theoretical Swelling": swelling
        })
        
        st.markdown("")
        tab1, tab2 = st.tabs(["📈 Interactive Swelling Trend", "💾 Export Results"])
        with tab1:
            with st.container(border=True):
                line_chart = alt.Chart(chart_df).mark_line(strokeWidth=3, color='#58a6ff').encode(
                    x=alt.X('Time (seconds):Q', title='Time (seconds)'),
                    y=alt.Y('Theoretical Swelling:Q', title='Swelling (%)'),
                    tooltip=['Time (seconds):Q', 'Theoretical Swelling:Q']
                ).properties(width=750, height=450).interactive()
                st.altair_chart(line_chart, use_container_width=True)
            
        with tab2:
            with st.container(border=True):
                st.markdown("#### Download Simulation Dataset")
                st.download_button(
                    "📥 Download Results as CSV",
                    data=chart_df.to_csv(index=False).encode('utf-8'),
                    file_name="cec_model_results.csv", 
                    mime="text/csv"
                )
 
# ==========================================
# WORKFLOW 2: EXPERIMENTAL DATA FITTING & MULTI-MODEL SELECTION
# ==========================================
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload Excel / CSV Workbook", type=["csv", "xlsx"])
 
    if uploaded_file is None:
        st.info("👋 **Getting Started:** Please upload your experimental shale swelling dataset workbook via the sidebar control panel.")
        st.stop()
        
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
    
    st.sidebar.markdown("### 🧬 Kinetic Models")
    selected_models = st.sidebar.multiselect(
        "Select models for comparative fitting:",
        options=available_models,
        default=["Custom Asymptotic Model", "Single Exponential (First-order)", "Weibull Model"]
    )
 
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_name = st.sidebar.selectbox("Select Sheet", excel_file.sheet_names)
        
        # Universal Auto-Scanning Parser
        lab_data = None
        raw_time = None
        s_exp = None
        
        for skip_row in range(20):
            try:
                temp_df = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=skip_row)
                temp_df.columns = temp_df.columns.astype(str).str.strip().str.lower()
                
                t_col, s_col = None, None
                for col in temp_df.columns:
                    if any(kw in col for kw in ['time', 'elapsed', 'sec', 'min', 'hr']):
                        if t_col is None:
                            t_col = col
                    if any(kw in col for kw in ['s(t)', 'swelling', 'swell', 'pct', '%', 'actual']):
                        if s_col is None:
                            s_col = col
                            
                if t_col and s_col:
                    lab_data = temp_df
                    raw_time = lab_data[t_col]
                    s_exp = pd.to_numeric(lab_data[s_col], errors='coerce').values
                    break
            except:
                continue
                
        if raw_time is None or s_exp is None or np.all(np.isnan(s_exp)):
            lab_data = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            if lab_data.shape[1] >= 2:
                raw_time = lab_data.iloc[:, 0]
                s_exp = pd.to_numeric(lab_data.iloc[:, 1], errors='coerce').values
            else:
                st.error("⚠️ Could not automatically detect time and swelling columns in this sheet.")
                st.stop()
        
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
 
        st.markdown(f"### 📈 Experimental Analysis Workspace — *{sheet_name}*")
        
        if not selected_models:
            st.warning("⚠️ Please select at least one kinetic model from the sidebar multiselect box.")
            st.stop()
            
        step = max(1, len(t_exp) // 300)
        t_fit_x = t_exp[::step]
        s_fit_y = s_exp[::step]

        df_chart = pd.DataFrame({"Time (seconds)": t_exp, "Actual Experimental Swelling": s_exp})
        metrics_list = []
        
        # Model Fitting Loop
        if "Custom Asymptotic Model" in selected_models:
            try:
                popt, _ = curve_fit(custom_model, t_fit_x, s_fit_y, p0=[max(s_fit_y), np.median(t_fit_x)], bounds=(0, np.inf), maxfev=5000)
                y_pred = custom_model(t_exp, *popt)
                df_chart["Custom Asymptotic Model"] = y_pred
                metrics_list.append({"Model": "Custom Asymptotic Model", "RMSE (%)": round(np.sqrt(mean_squared_error(s_exp, y_pred)), 4), "R² Score": round(r2_score(s_exp, y_pred), 4)})
            except Exception as e:
                st.warning(f"Custom Asymptotic Model skipped: {e}")

        if "Single Exponential (First-order)" in selected_models:
            try:
                popt, _ = curve_fit(single_exponential, t_fit_x, s_fit_y, p0=[max(s_fit_y), 1e-4], bounds=(0, np.inf), maxfev=5000)
                y_pred = single_exponential(t_exp, *popt)
                df_chart["Single Exponential"] = y_pred
                metrics_list.append({"Model": "Single Exponential", "RMSE (%)": round(np.sqrt(mean_squared_error(s_exp, y_pred)), 4), "R² Score": round(r2_score(s_exp, y_pred), 4)})
            except Exception as e:
                st.warning(f"Single Exponential skipped: {e}")

        if "Double Exponential" in selected_models:
            try:
                popt, _ = curve_fit(double_exponential, t_fit_x, s_fit_y, p0=[max(s_fit_y)*0.6, 1e-3, max(s_fit_y)*0.4, 1e-5], bounds=(0, np.inf), maxfev=10000)
                y_pred = double_exponential(t_exp, *popt)
                df_chart["Double Exponential"] = y_pred
                metrics_list.append({"Model": "Double Exponential", "RMSE (%)": round(np.sqrt(mean_squared_error(s_exp, y_pred)), 4), "R² Score": round(r2_score(s_exp, y_pred), 4)})
            except Exception as e:
                st.warning(f"Double Exponential skipped: {e}")

        if "Weibull Model" in selected_models:
            try:
                popt, _ = curve_fit(weibull_model, t_fit_x, s_fit_y, p0=[max(s_fit_y), np.median(t_fit_x), 1.0], bounds=([0, 0, 0], [np.inf, np.inf, 5]), maxfev=8000)
                y_pred = weibull_model(t_exp, *popt)
                df_chart["Weibull Model"] = y_pred
                metrics_list.append({"Model": "Weibull Model", "RMSE (%)": round(np.sqrt(mean_squared_error(s_exp, y_pred)), 4), "R² Score": round(r2_score(s_exp, y_pred), 4)})
            except Exception as e:
                st.warning(f"Weibull Model skipped: {e}")

        if "Logistic Model" in selected_models:
            try:
                popt, _ = curve_fit(logistic_model, t_fit_x, s_fit_y, p0=[max(s_fit_y), 1e-4, np.median(t_fit_x)], bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]), maxfev=8000)
                y_pred = logistic_model(t_exp, *popt)
                df_chart["Logistic Model"] = y_pred
                metrics_list.append({"Model": "Logistic Model", "RMSE (%)": round(np.sqrt(mean_squared_error(s_exp, y_pred)), 4), "R² Score": round(r2_score(s_exp, y_pred), 4)})
            except Exception as e:
                st.warning(f"Logistic Model skipped: {e}")

        if "Gompertz Model" in selected_models:
            try:
                popt, _ = curve_fit(gompertz_model, t_fit_x, s_fit_y, p0=[max(s_fit_y), 1e-4, np.median(t_fit_x)], bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]), maxfev=8000)
                y_pred = gompertz_model(t_exp, *popt)
                df_chart["Gompertz Model"] = y_pred
                metrics_list.append({"Model": "Gompertz Model", "RMSE (%)": round(np.sqrt(mean_squared_error(s_exp, y_pred)), 4), "R² Score": round(r2_score(s_exp, y_pred), 4)})
            except Exception as e:
                st.warning(f"Gompertz Model skipped: {e}")

        if "Power Law" in selected_models:
            try:
                popt, _ = curve_fit(power_law, t_fit_x, s_fit_y, p0=[0.1, 0.5], bounds=(0, [np.inf, 2.0]), maxfev=5000)
                y_pred = power_law(t_exp, *popt)
                df_chart["Power Law"] = y_pred
                metrics_list.append({"Model": "Power Law", "RMSE (%)": round(np.sqrt(mean_squared_error(s_exp, y_pred)), 4), "R² Score": round(r2_score(s_exp, y_pred), 4)})
            except Exception as e:
                st.warning(f"Power Law skipped: {e}")

        if "Peleg Model" in selected_models:
            try:
                popt, _ = curve_fit(peleg_model, t_fit_x, s_fit_y, p0=[10.0, 0.1], bounds=(0, np.inf), maxfev=5000)
                y_pred = peleg_model(t_exp, *popt)
                df_chart["Peleg Model"] = y_pred
                metrics_list.append({"Model": "Peleg Model", "RMSE (%)": round(np.sqrt(mean_squared_error(s_exp, y_pred)), 4), "R² Score": round(r2_score(s_exp, y_pred), 4)})
            except Exception as e:
                st.warning(f"Peleg Model skipped: {e}")
 
        metrics_df = pd.DataFrame(metrics_list)
        st.markdown("")
 
        tab1, tab2, tab3 = st.tabs(["📉 Model Fit & Performance Layout", "📋 Statistical Metrics Table", "💾 Export Results"])
        
        with tab1:
            # 2-Column Split Layout for Wide Desktop Screens
            col_left, col_right = st.columns([1.1, 0.9])
            
            with col_left:
                with st.container(border=True):
                    st.markdown("#### 📈 Swelling vs. Time Overlay")
                    st.caption("Actual experimental data shown with a solid black line/marker.")
                    
                    melted_df = df_chart.melt(id_vars=["Time (seconds)"], var_name="Legend / Model", value_name="Swelling (%)")
                    domain_list = melted_df["Legend / Model"].unique().tolist()
                    range_list = ["#ffffff" if "Actual" in name else f"C{i}" for i, name in enumerate(domain_list)]

                    base_chart = alt.Chart(melted_df).mark_line(strokeWidth=2.5).encode(
                        x=alt.X('Time (seconds):Q', title='Time (seconds)'),
                        y=alt.Y('Swelling (%):Q', title='Swelling (%)'),
                        color=alt.Color('Legend / Model:N', scale=alt.Scale(domain=domain_list, range=range_list), title='Model Legend'),
                        tooltip=['Time (seconds):Q', 'Swelling (%):Q', 'Legend / Model:N']
                    ).properties(width=500, height=420).interactive()
                    
                    st.altair_chart(base_chart, use_container_width=True)
            
            with col_right:
                if not metrics_df.empty:
                    with st.container(border=True):
                        st.markdown("#### 📊 Model Performance Bar Charts")
                        
                        # RMSE Bar Chart
                        fig, ax = plt.subplots(figsize=(5, 3.2))
                        sorted_rmse = metrics_df.sort_values(by="RMSE (%)", ascending=False)
                        bars = ax.bar(sorted_rmse["Model"], sorted_rmse["RMSE (%)"], color='#f85149', alpha=0.7, edgecolor='white', linewidth=0.6)
                        ax.set_title("Root Mean Squared Error (RMSE %)", fontsize=10, fontweight='bold', color='white', pad=8)
                        ax.set_ylabel("RMSE (%)", fontsize=9, color='white')
                        ax.tick_params(colors='white', labelsize=8)
                        plt.xticks(rotation=20, ha='right')
                        for bar in bars:
                            h = bar.get_height()
                            ax.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold', color='white')
                        ax.set_facecolor('#0e1117')
                        fig.patch.set_facecolor('#0e1117')
                        sns.despine(top=True, right=True, left=True, bottom=True)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # R2 Bar Chart
                        fig, ax = plt.subplots(figsize=(5, 3.2))
                        sorted_r2 = metrics_df.sort_values(by="R² Score", ascending=False)
                        bars = ax.bar(sorted_r2["Model"], sorted_r2["R² Score"], color='#2ea043', alpha=0.7, edgecolor='white', linewidth=0.6)
                        ax.set_title("Coefficient of Determination (R² Score)", fontsize=10, fontweight='bold', color='white', pad=8)
                        ax.set_ylabel("R² Score", fontsize=9, color='white')
                        ax.tick_params(colors='white', labelsize=8)
                        plt.xticks(rotation=20, ha='right')
                        for bar in bars:
                            h = bar.get_height()
                            ax.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold', color='white')
                        ax.set_facecolor('#0e1117')
                        fig.patch.set_facecolor('#0e1117')
                        sns.despine(top=True, right=True, left=True, bottom=True)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
            
        with tab2:
            with st.container(border=True):
                st.markdown("#### Detailed Accuracy Metrics")
                if not metrics_df.empty:
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No models successfully evaluated.")
            
        with tab3:
            with st.container(border=True):
                st.markdown("#### Multi-Sheet Excel Export Package")
                st.markdown("Export all fitted model predictions and performance evaluation metrics into a formatted Excel report workbook.")
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_chart.to_excel(writer, sheet_name='Model Predictions', index=False)
                    if not metrics_df.empty:
                        metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📥 Download Complete Excel Report (.xlsx)",
                    data=excel_data,
                    file_name="shale_swelling_model_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
 
    except Exception as e:
        st.error(f"⚠️ Error executing model fitting: {e}")
