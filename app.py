from __future__ import annotations

from pathlib import Path
import os
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "sim_1year.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "model" / "model.pkl"
REQUIRED_COLUMNS = {"timestamp", "data_consumption_gb"}
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def get_config_value(key: str, default: str = "") -> str:
    try:
        value = st.secrets.get(key, default)
    except Exception:
        value = default

    if not value:
        value = os.getenv(key, default)

    return str(value).strip()


def ensure_model_available(model_path: Path) -> Path:
    if model_path.exists():
        return model_path

    model_url = get_config_value("MODEL_URL")
    if not model_url:
        raise FileNotFoundError(
            f"Model not found at `{model_path}` and `MODEL_URL` is not configured in Streamlit secrets."
        )

    auth_token = get_config_value("MODEL_AUTH_TOKEN")
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else None

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(model_url, stream=True, timeout=120, headers=headers) as response:
        response.raise_for_status()
        with open(model_path, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    file_obj.write(chunk)

    return model_path


@st.cache_data
def read_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def read_csv_from_upload(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    prepared = df[["timestamp", "data_consumption_gb"]].copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
    prepared["data_consumption_gb"] = pd.to_numeric(prepared["data_consumption_gb"], errors="coerce")
    prepared = prepared.dropna().sort_values("timestamp").reset_index(drop=True)

    if prepared.empty:
        raise ValueError("Dataset is empty after cleaning.")

    return prepared


def infer_series_frequency(index: pd.DatetimeIndex) -> str:
    inferred = pd.infer_freq(index)
    if inferred:
        return inferred

    if len(index) < 2:
        return "h"

    deltas = pd.Series(index[1:] - index[:-1])
    if deltas.empty:
        return "h"

    step = deltas.mode().iloc[0]
    return pd.tseries.frequencies.to_offset(step).freqstr


def get_forecast(model, steps: int) -> tuple[pd.Series, pd.Series | None, pd.Series | None]:
    forecast_res = model.get_forecast(steps=steps)

    mean = pd.Series(forecast_res.predicted_mean, dtype="float64")

    lower = None
    upper = None
    try:
        conf = forecast_res.conf_int(alpha=0.05)
        if isinstance(conf, pd.DataFrame) and conf.shape[1] >= 2:
            lower = pd.Series(conf.iloc[:, 0], dtype="float64")
            upper = pd.Series(conf.iloc[:, 1], dtype="float64")
        else:
            conf_arr = np.asarray(conf, dtype="float64")
            if conf_arr.ndim == 2 and conf_arr.shape[1] >= 2:
                lower = pd.Series(conf_arr[:, 0])
                upper = pd.Series(conf_arr[:, 1])
    except Exception:
        pass

    return mean, lower, upper


def reindex_forecast(
    mean: pd.Series,
    lower: pd.Series | None,
    upper: pd.Series | None,
    last_timestamp: pd.Timestamp,
    freq: str,
) -> tuple[pd.Series, pd.Series | None, pd.Series | None]:
    offset = pd.tseries.frequencies.to_offset(freq)
    future_index = pd.date_range(start=last_timestamp + offset, periods=len(mean), freq=freq)

    mean.index = future_index
    if lower is not None:
        lower.index = future_index
    if upper is not None:
        upper.index = future_index

    return mean, lower, upper


def main() -> None:
    st.set_page_config(page_title="Network Traffic Forecasting", layout="wide")
    st.title("Network Traffic Forecasting")
    st.caption("Deploy-ready Streamlit app for telecom data consumption forecasting.")

    st.sidebar.header("Configuration")
    model_path = Path(st.sidebar.text_input("Model path", value=str(DEFAULT_MODEL_PATH)))
    st.sidebar.caption("If this file is missing, the app downloads it from MODEL_URL secret.")
    horizon = st.sidebar.slider("Forecast horizon (steps)", min_value=1, max_value=24 * 30, value=24 * 7)

    source = st.sidebar.radio("Data source", options=["Default dataset", "Upload CSV"], index=0)

    try:
        if source == "Default dataset":
            raw_df = read_csv_from_path(str(DEFAULT_DATA_PATH))
        else:
            upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])
            if upload is None:
                st.info("Upload a CSV file that includes `timestamp` and `data_consumption_gb`.")
                return
            raw_df = read_csv_from_upload(upload.getvalue())

        df = prepare_dataframe(raw_df)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    series = df.set_index("timestamp")["data_consumption_gb"]
    freq = infer_series_frequency(series.index)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Start", str(df["timestamp"].iloc[0]))
    c3.metric("End", str(df["timestamp"].iloc[-1]))

    st.subheader("Historical data")
    st.line_chart(series)

    if not st.button("Generate forecast", type="primary"):
        return

    try:
        resolved_model_path = ensure_model_available(model_path)
        model = load_model(str(resolved_model_path))
    except Exception as exc:
        st.error(f"Failed to load model from `{model_path}`: {exc}")
        return

    try:
        mean, lower, upper = get_forecast(model, steps=horizon)
        mean, lower, upper = reindex_forecast(mean, lower, upper, series.index[-1], freq)
    except Exception as exc:
        st.error(f"Forecast failed: {exc}")
        return

    st.subheader("Forecast")
    combined = pd.concat(
        [series.rename("actual"), mean.rename("forecast")],
        axis=1,
    )
    st.line_chart(combined)

    result_df = pd.DataFrame(
        {
            "timestamp": mean.index,
            "forecast": mean.values,
        }
    )

    if lower is not None and upper is not None:
        result_df["lower_95"] = lower.values
        result_df["upper_95"] = upper.values

    st.dataframe(result_df, use_container_width=True)
    st.download_button(
        label="Download forecast CSV",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="forecast_output.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

