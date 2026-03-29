# Network Traffic Forecasting - Synthetic Data Simulator

This repository now includes a realistic synthetic dataset generator for **univariate telecom data consumption forecasting**.

## What it simulates

The script builds a time series with:

- **Growth over time** (trend)
- **Daily and weekly seasonality**
- **Sudden spikes/events** (e.g., sports, launches, holidays)
- **Network drops/outages** (temporary deep reductions)
- **Random noise**

## Files

- `data/simulate_telecom_consumption.py`: main generator script
- `app.py`: Streamlit forecasting app
- `model/model.pkl`: trained SARIMAX model artifact
- `data/sim_1year.csv`: default sample dataset used by the app
- `requirements.txt`: Python dependencies

## Quick start

```bash
pip install -r requirements.txt
python data/simulate_telecom_consumption.py
```

This writes:

- `simulated_telecom_consumption.csv`
- `simulated_telecom_consumption_meta.json`

## Example custom run

```bash
python data/simulate_telecom_consumption.py \
  --periods 35040 \
  --freq h \
  --start-date 2024-01-01 \
  --trend-per-step 0.01 \
  --spike-probability 0.004 \
  --drop-probability 0.002 \
  --output-csv data/sim_hourly_4years.csv \
  --output-metadata data/sim_hourly_4years_meta.json
```

## Main parameters

- `--periods`: number of time steps
- `--freq`: pandas frequency (`h`, `15min`, `D`, ...)
- `--base-consumption`: baseline data usage
- `--trend-per-step`: incremental growth each step
- `--daily-amplitude`, `--weekly-amplitude`: seasonality strength
- `--spike-probability`: chance of spike at each step
- `--spike-min-magnitude`, `--spike-max-magnitude`: spike size range
- `--drop-probability`: chance of outage at each step
- `--drop-min-depth`, `--drop-max-depth`: fraction removed during drops
- `--seed`: reproducibility

## Notes

- Output column is `data_consumption_gb` and is always non-negative.
- You can tune probabilities and magnitudes to mimic different telecom markets.

## Streamlit app (local)

Run the forecasting app with the included model artifact:

```bat
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`.

The app expects:

- model path: `model/model.pkl`
- data columns: `timestamp`, `data_consumption_gb`

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud and select **New app**.
3. Choose your repository and branch.
4. Set the entry point to `app.py`.
5. Deploy and check logs if dependency errors appear.

If logs show a missing package, add it to `requirements.txt`, push, and redeploy.

