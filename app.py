# app.py
import io
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


# -----------------------------
# Page config + helpers
# -----------------------------
st.set_page_config(page_title="Time Series Forecasting (ARIMA/SARIMA)", layout="wide")

def _title():
    st.title("📈 Time Series Forecasting App (AR / MA / ARIMA / SARIMA)")
    st.caption(
        "Upload your time series CSV (e.g., Daily Minimum Temperatures in Melbourne), "
        "run stationarity tests, compare models, and forecast next days."
    )

def safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def eval_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    mp = safe_mape(y_true, y_pred)
    return mae, rmse, mp

def ljung_box_pvalue(residuals, lags=20):
    res = pd.Series(residuals).dropna()
    if len(res) < lags + 3:
        return np.nan
    lb = acorr_ljungbox(res, lags=[lags], return_df=True)
    return float(lb["lb_pvalue"].iloc[0])

def run_adf(series):
    s = pd.Series(series).dropna()
    stat, p, *_ = adfuller(s, autolag="AIC")
    return float(stat), float(p)

def run_kpss(series):
    s = pd.Series(series).dropna()
    # regression="c" for level stationarity
    stat, p, *_ = kpss(s, regression="c", nlags="auto")
    return float(stat), float(p)

def plot_series(ts, title, xlabel="Date", ylabel="Value"):
    fig = plt.figure(figsize=(12, 4))
    plt.plot(ts.index, ts.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_train_test_pred(train, test, pred, title):
    fig = plt.figure(figsize=(12, 4))
    plt.plot(train.index, train.values, label="Train")
    plt.plot(test.index, test.values, label="Test")
    plt.plot(pred.index, pred.values, label="Predicted")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_forecast(ts, forecast, title):
    fig = plt.figure(figsize=(12, 4))
    plt.plot(ts.index, ts.values, label="History")
    plt.plot(forecast.index, forecast.values, label="Forecast")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def parse_uploaded_csv(uploaded_file, encoding="utf-8"):
    # Read into pandas safely
    raw = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(raw), encoding=encoding)

def infer_date_col(df):
    # Try common names first
    candidates = [c for c in df.columns if c.lower() in ("date", "ds", "time", "timestamp")]
    if candidates:
        return candidates[0]
    # Otherwise pick first column that can parse many datetimes
    best_col = None
    best_rate = 0.0
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            rate = parsed.notna().mean()
            if rate > best_rate and rate > 0.6:
                best_rate = rate
                best_col = c
        except Exception:
            continue
    return best_col

def infer_value_col(df, date_col=None):
    # Common names for value
    for c in df.columns:
        if c.lower() in ("temp", "temperature", "value", "y"):
            if c != date_col:
                return c
    # Otherwise choose first numeric (excluding date col)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != date_col]
    return numeric_cols[0] if numeric_cols else None

def make_timeseries(df, date_col, value_col, freq="D", fill_method="interpolate"):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    d = d.sort_values(date_col).set_index(date_col)

    # Force numeric
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    # Set frequency (daily) to ensure a continuous calendar
    d = d.asfreq(freq)

    if fill_method == "interpolate":
        d[value_col] = d[value_col].interpolate()
    elif fill_method == "ffill":
        d[value_col] = d[value_col].ffill()
    elif fill_method == "dropna":
        d = d.dropna(subset=[value_col])

    ts = d[value_col].dropna()
    return ts

def fit_arima(train, order):
    model = ARIMA(train, order=order)
    return model.fit()

def fit_sarima(train, order, seasonal_order):
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)

@st.cache_data(show_spinner=False)
def cached_decompose(ts, period):
    # decomposition can be slow; cache output
    return seasonal_decompose(ts, model="additive", period=period)

# -----------------------------
# App UI
# -----------------------------
_title()

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    encoding = st.selectbox("CSV Encoding", ["utf-8", "latin1", "ISO-8859-1"], index=0)
    freq = st.selectbox("Time frequency", ["D", "W", "M"], index=0)
    fill_method = st.selectbox("Missing values", ["interpolate", "ffill", "dropna"], index=0)

    st.divider()
    st.subheader("Split & Forecast")
    test_size = st.number_input("Test size (last N points)", min_value=30, max_value=2000, value=365, step=5)
    forecast_horizon = st.number_input("Forecast horizon (days)", min_value=7, max_value=365, value=30, step=1)

    st.divider()
    st.subheader("Seasonality")
    decomp_period = st.number_input("Decomposition period", min_value=2, max_value=400, value=365, step=1)

    # IMPORTANT: SARIMA with s=365 can be memory heavy on Streamlit Cloud.
    sarima_seasonal = st.selectbox("SARIMA seasonal period (safe)", [7, 30, 365], index=0)
    include_sarima = st.checkbox("Include SARIMA in model comparison", value=True)

    st.divider()
    st.subheader("Model search")
    st.caption("Smaller grids are safer for Streamlit Cloud.")
    ar_p_list = st.multiselect("AR p candidates", [1,2,3,5,7], default=[1,2,3,5])
    ma_q_list = st.multiselect("MA q candidates", [1,2,3,5,7], default=[1,2,3,5])
    arima_p_list = st.multiselect("ARIMA p candidates", [0,1,2,3], default=[0,1,2])
    arima_q_list = st.multiselect("ARIMA q candidates", [0,1,2,3], default=[0,1,2])
    max_models_note = st.checkbox("Show memory tips", value=False)

if max_models_note:
    st.info(
        "Tip: If your app runs out of memory, reduce model candidates and keep SARIMA seasonal period at 7 or 30. "
        "SARIMA with 365 can be heavy."
    )

if not uploaded:
    st.warning("Upload a CSV to begin. Example columns: Date, Temp")
    st.stop()

# -----------------------------
# Load data
# -----------------------------
df = parse_uploaded_csv(uploaded, encoding=encoding)
st.subheader("1) Data Preview")
st.dataframe(df.head(20), use_container_width=True)

# Column selection
inferred_date = infer_date_col(df)
inferred_value = infer_value_col(df, date_col=inferred_date)

col1, col2 = st.columns(2)
with col1:
    date_col = st.selectbox("Select Date column", df.columns.tolist(), index=df.columns.get_loc(inferred_date) if inferred_date in df.columns else 0)
with col2:
    value_col = st.selectbox("Select Value column", df.columns.tolist(), index=df.columns.get_loc(inferred_value) if inferred_value in df.columns else min(1, len(df.columns)-1))

# Build time series
ts = make_timeseries(df, date_col=date_col, value_col=value_col, freq=freq, fill_method=fill_method)

st.write(f"✅ Time series points: **{len(ts):,}** | Date range: **{ts.index.min().date()} → {ts.index.max().date()}**")
st.write(f"Missing handling: **{fill_method}** | Frequency: **{freq}**")

st.subheader("2) Exploratory Plot")
plot_series(ts, f"Time Series: {value_col}")

# Basic stats
with st.expander("Show descriptive statistics"):
    st.write(ts.describe())

# -----------------------------
# Stationarity tests
# -----------------------------
st.subheader("3) Stationarity Tests (ADF & KPSS)")
adf_stat, adf_p = run_adf(ts)
kpss_stat, kpss_p = run_kpss(ts)

m1, m2 = st.columns(2)
with m1:
    st.metric("ADF p-value", f"{adf_p:.6f}")
    st.caption("ADF H₀: Non-stationary. p<0.05 ⇒ Stationary.")
with m2:
    st.metric("KPSS p-value", f"{kpss_p:.6f}")
    st.caption("KPSS H₀: Stationary. p<0.05 ⇒ Non-stationary.")

# Decide differencing d
d = 0
if (adf_p >= 0.05) or (kpss_p <= 0.05):
    d = 1
st.write(f"📌 Selected differencing order: **d = {d}**")

if d == 1:
    ts_diff = ts.diff().dropna()
    st.subheader("3.1) Differenced Series")
    plot_series(ts_diff, "Differenced Series (d=1)")
    adf2_stat, adf2_p = run_adf(ts_diff)
    kpss2_stat, kpss2_p = run_kpss(ts_diff)
    st.write(f"After differencing: ADF p={adf2_p:.6f}, KPSS p={kpss2_p:.6f}")

# -----------------------------
# Decomposition
# -----------------------------
st.subheader("4) Time Series Decomposition")
st.caption("Decomposition helps visualize trend, seasonality, and residual noise.")
try:
    decomp = cached_decompose(ts, period=int(decomp_period))
    fig = decomp.plot()
    fig.set_size_inches(12, 8)
    st.pyplot(fig)
    plt.close(fig)
except Exception as e:
    st.warning(f"Decomposition failed (often due to short series vs. chosen period). Error: {e}")

# -----------------------------
# Train/Test split
# -----------------------------
st.subheader("5) Train–Test Split")
if test_size >= len(ts) - 5:
    st.error("Test size is too large for the dataset. Reduce test size.")
    st.stop()

train = ts.iloc[:-int(test_size)]
test = ts.iloc[-int(test_size):]
st.write(f"Train points: **{len(train):,}** | Test points: **{len(test):,}**")

plot_series(train, "Train Series")
plot_series(test, "Test Series")

# -----------------------------
# Model comparison
# -----------------------------
st.subheader("6) Model Fitting & Comparison")
st.caption("We compare models using AIC/BIC, Ljung–Box residual test, and forecasting errors (MAE/RMSE/MAPE).")

run_btn = st.button("🚀 Run Model Comparison", type="primary")

if run_btn:
    results = []

    def _append_result(model_name, order, seasonal_order, fit, fc):
        fc = pd.Series(fc, index=test.index)
        mae, rmse, mp = eval_metrics(test.values, fc.values)
        lb_p = ljung_box_pvalue(fit.resid, lags=20)
        results.append([model_name, order, seasonal_order, float(fit.aic), float(fit.bic), lb_p, mae, rmse, mp])

    with st.spinner("Fitting models..."):
        # AR: ARIMA(p, d, 0)
        for p in ar_p_list:
            try:
                fit = fit_arima(train, order=(p, d, 0))
                fc = fit.forecast(steps=len(test))
                _append_result("AR", (p, d, 0), None, fit, fc)
                del fit, fc
                gc.collect()
            except:
                gc.collect()

        # MA: ARIMA(0, d, q)
        for q in ma_q_list:
            try:
                fit = fit_arima(train, order=(0, d, q))
                fc = fit.forecast(steps=len(test))
                _append_result("MA", (0, d, q), None, fit, fc)
                del fit, fc
                gc.collect()
            except:
                gc.collect()

        # ARIMA: small grid
        for p in arima_p_list:
            for q in arima_q_list:
                if p == 0 and q == 0:
                    continue
                try:
                    fit = fit_arima(train, order=(p, d, q))
                    fc = fit.forecast(steps=len(test))
                    _append_result("ARIMA", (p, d, q), None, fit, fc)
                    del fit, fc
                    gc.collect()
                except:
                    gc.collect()

        # SARIMA (optional) — safe seasonal period
        # Note: SARIMA with s=365 can be heavy on Streamlit Cloud.
        if include_sarima:
            s = int(sarima_seasonal)
            for p in [1, 2]:
                for q in [1, 2]:
                    for P in [0, 1]:
                        for Q in [0, 1]:
                            try:
                                fit = fit_sarima(train, order=(p, d, q), seasonal_order=(P, 1, Q, s))
                                fc = fit.forecast(steps=len(test))
                                _append_result("SARIMA", (p, d, q), (P, 1, Q, s), fit, fc)
                                del fit, fc
                                gc.collect()
                            except:
                                gc.collect()

    if not results:
        st.error("No models were successfully fitted. Reduce grid or check data.")
        st.stop()

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Order", "Seasonal_Order", "AIC", "BIC", "LjungBox_p(20)", "MAE", "RMSE", "MAPE"]
    )

    # Sort: prefer good BIC/AIC, lower RMSE
    results_df = results_df.sort_values(["BIC", "AIC", "RMSE"]).reset_index(drop=True)

    st.subheader("6.1) Model Comparison Table")
    st.dataframe(results_df, use_container_width=True)

    # Choose best (prefer LjungBox p>0.05 if available)
    filtered = results_df[results_df["LjungBox_p(20)"].fillna(-1) > 0.05]
    best = filtered.iloc[0] if len(filtered) > 0 else results_df.iloc[0]

    st.success(f"✅ Best model selected: **{best['Model']}** | Order={best['Order']} | Seasonal={best['Seasonal_Order']}")

    # Fit best model on train and plot actual vs predicted
    st.subheader("7) Best Model: Actual vs Predicted")
    model_name = best["Model"]
    order = tuple(best["Order"])
    seasonal = best["Seasonal_Order"]

    if model_name == "SARIMA" and isinstance(seasonal, (list, tuple)):
        seasonal = tuple(seasonal)
        best_fit = fit_sarima(train, order=order, seasonal_order=seasonal)
    elif model_name == "SARIMA" and pd.notna(seasonal):
        # Streamlit can store tuple as object; ensure correct parsing
        best_fit = fit_sarima(train, order=order, seasonal_order=tuple(seasonal))
    else:
        best_fit = fit_arima(train, order=order)

    pred = pd.Series(best_fit.forecast(steps=len(test)), index=test.index)
    plot_train_test_pred(train, test, pred, f"Actual vs Predicted ({model_name})")

    # Residual diagnostics
    st.subheader("8) Residual Diagnostics (Best Model)")
    resid = pd.Series(best_fit.resid).dropna()
    lbp = ljung_box_pvalue(resid, lags=20)
    st.write(f"**Ljung–Box p-value (lag 20):** {lbp:.6f}" if not np.isnan(lbp) else "**Ljung–Box p-value (lag 20):** Not enough residuals")

    fig = plt.figure(figsize=(12, 3))
    plt.plot(resid.index, resid.values)
    plt.title("Residuals")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Forecast next N
    st.subheader("9) Forecast Next Days")
    st.caption("For final forecast, we refit the best model on the full series for maximum learning.")

    if model_name == "SARIMA" and best["Seasonal_Order"] is not None and pd.notna(best["Seasonal_Order"]):
        seasonal_order = tuple(best["Seasonal_Order"])
        final_fit = fit_sarima(ts, order=order, seasonal_order=seasonal_order)
    else:
        final_fit = fit_arima(ts, order=order)

    fc = final_fit.forecast(steps=int(forecast_horizon))

    # Build forecast index continuation
    last_date = ts.index[-1]
    if freq == "D":
        fc_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=int(forecast_horizon), freq="D")
    elif freq == "W":
        fc_index = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=int(forecast_horizon), freq="W")
    else:
        fc_index = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=int(forecast_horizon), freq="MS")

    forecast = pd.Series(fc.values, index=fc_index, name="Forecast")

    plot_forecast(ts, forecast, f"{forecast_horizon}-Step Forecast ({model_name})")

    st.subheader("9.1) Forecast Table")
    st.dataframe(forecast.to_frame(), use_container_width=True)

    # Download forecast
    out = forecast.reset_index()
    out.columns = ["Date", "Forecast"]
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv_bytes,
        file_name="forecast.csv",
        mime="text/csv",
    )

    # Cleanup
    del best_fit, final_fit
    gc.collect()

st.divider()
st.caption("Built for Streamlit Cloud • Classic Time Series • ARIMA/SARIMA • Diagnostics + Forecast")
st.caption("© 2025 by [Dr. Krishnajyothi Nath] 
           