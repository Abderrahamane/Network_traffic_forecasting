from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    periods: int = 24 * 365
    months: int = 6
    years: int = 0
    freq: str = "h"
    start_date: str = "2023-01-01"
    seed: int = 42

    base_consumption: float = 120.0
    trend_per_step: float = 0.006
    monthly_growth_rate: float = 0.012
    noise_std: float = 5.0
    ar1_strength: float = 0.62

    daily_amplitude: float = 25.0
    weekly_amplitude: float = 12.0
    monthly_amplitude: float = 6.0
    weekend_factor: float = 0.86
    evening_peak_boost: float = 18.0

    spike_probability: float = 0.003
    spike_min_magnitude: float = 70.0
    spike_max_magnitude: float = 220.0
    spike_min_duration: int = 1
    spike_max_duration: int = 12
    event_cluster_scale: float = 0.25

    drop_probability: float = 0.0018
    drop_min_depth: float = 0.35
    drop_max_depth: float = 0.85
    drop_min_duration: int = 1
    drop_max_duration: int = 10
    recovery_min_duration: int = 2
    recovery_max_duration: int = 10


def _resolve_timestamps(cfg: SimulationConfig) -> pd.DatetimeIndex:
    start = pd.Timestamp(cfg.start_date)
    if cfg.months > 0 or cfg.years > 0:
        end = start + pd.DateOffset(months=cfg.months, years=cfg.years)
        return pd.date_range(start=start, end=end, freq=cfg.freq, inclusive="left")
    return pd.date_range(start=start, periods=cfg.periods, freq=cfg.freq)


def _add_spikes(
    values: np.ndarray, rng: np.random.Generator, cfg: SimulationConfig
) -> tuple[np.ndarray, int]:
    """Inject positive bursts to mimic events (sports finals, launches, holidays)."""
    n = len(values)
    event_count = 0
    i = 0
    cluster_left = 0
    cluster_boost = 1.0

    # Cluster model creates periods with elevated chance of related spikes.
    while i < n:
        if cluster_left > 0:
            p = min(1.0, cfg.spike_probability * cluster_boost)
            cluster_left -= 1
        else:
            p = cfg.spike_probability

        if rng.random() < p:
            event_count += 1
            if cluster_left == 0 and rng.random() < cfg.event_cluster_scale:
                cluster_left = int(rng.integers(6, 36))
                cluster_boost = float(rng.uniform(1.8, 3.3))

            duration = int(rng.integers(cfg.spike_min_duration, cfg.spike_max_duration + 1))
            magnitude = float(rng.uniform(cfg.spike_min_magnitude, cfg.spike_max_magnitude))
            end = min(i + duration, n)

            x = np.linspace(0.0, 1.0, end - i)
            spike_profile = (1.0 - np.exp(-6.0 * x)) * np.exp(-2.8 * x)
            spike_profile = spike_profile / (spike_profile.max() + 1e-9)
            values[i:end] += magnitude * spike_profile

            i += max(1, duration // 3)
            continue

        i += 1

    return values, event_count


def _apply_drops(
    values: np.ndarray, rng: np.random.Generator, cfg: SimulationConfig
) -> tuple[np.ndarray, int]:
    """Apply temporary degradations and recovery tails to mimic outages."""
    n = len(values)
    drop_count = 0
    for i in range(n):
        if rng.random() < cfg.drop_probability:
            drop_count += 1
            duration = int(rng.integers(cfg.drop_min_duration, cfg.drop_max_duration + 1))
            depth = float(rng.uniform(cfg.drop_min_depth, cfg.drop_max_depth))
            end = min(i + duration, n)
            values[i:end] *= 1.0 - depth

            recovery_len = int(
                rng.integers(cfg.recovery_min_duration, cfg.recovery_max_duration + 1)
            )
            rec_end = min(end + recovery_len, n)
            if rec_end > end:
                rec = np.linspace(1.0 - depth * 0.55, 1.0, rec_end - end)
                values[end:rec_end] *= rec

    return values, drop_count


def simulate_consumption_data(cfg: SimulationConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    timestamps = _resolve_timestamps(cfg)
    n = len(timestamps)
    t = np.arange(n, dtype=float)

    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    month = timestamps.month.to_numpy()

    # Multi-part trend: linear base + compounding monthly growth.
    trend_linear = cfg.trend_per_step * t
    total_months = max(1, cfg.months + (cfg.years * 12))
    steps_per_month = max(1.0, n / total_months)
    trend_compounded = cfg.base_consumption * (
        np.power(1.0 + cfg.monthly_growth_rate, t / steps_per_month) - 1.0
    )

    hour_phase = hour / 24.0
    day_phase = weekday / 7.0
    month_phase = (month - 1) / 12.0

    daily = cfg.daily_amplitude * np.sin(2 * np.pi * hour_phase - np.pi / 2)
    weekly = cfg.weekly_amplitude * np.sin(2 * np.pi * day_phase - np.pi / 2)
    monthly = cfg.monthly_amplitude * np.sin(2 * np.pi * month_phase - np.pi / 2)

    # Typical telecom usage has stronger evening peaks than morning peaks.
    evening_peak = cfg.evening_peak_boost * np.exp(-0.5 * ((hour - 21) / 2.2) ** 2)

    weekend_multiplier = np.where(weekday >= 5, cfg.weekend_factor, 1.0)

    base_signal = (
        cfg.base_consumption + trend_linear + trend_compounded + daily + weekly + monthly + evening_peak
    )
    base_signal *= weekend_multiplier

    # Residuals are persistent over time and noisier at peak hours.
    hetero_scale = 0.55 + 0.65 * ((hour >= 18) & (hour <= 23)) + 0.15 * (weekday < 5)
    eps = rng.normal(0.0, cfg.noise_std * hetero_scale, size=n)
    ar_noise = np.empty(n, dtype=float)
    ar_noise[0] = eps[0]
    for i in range(1, n):
        ar_noise[i] = cfg.ar1_strength * ar_noise[i - 1] + eps[i]

    values = base_signal + ar_noise
    values, event_count = _add_spikes(values, rng, cfg)
    values, drop_count = _apply_drops(values, rng, cfg)

    # Consumption cannot be negative.
    values = np.clip(values, 0.0, None)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "data_consumption_gb": np.round(values, 3),
        }
    )
    df.attrs["event_count"] = event_count
    df.attrs["drop_count"] = drop_count
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate telecom data-consumption time series with trend, spikes, and drops."
    )
    parser.add_argument("--periods", type=int, default=SimulationConfig.periods)
    parser.add_argument("--months", type=int, default=SimulationConfig.months)
    parser.add_argument("--years", type=int, default=SimulationConfig.years)
    parser.add_argument("--freq", type=str, default=SimulationConfig.freq)
    parser.add_argument("--start-date", type=str, default=SimulationConfig.start_date)
    parser.add_argument("--seed", type=int, default=SimulationConfig.seed)

    parser.add_argument("--base-consumption", type=float, default=SimulationConfig.base_consumption)
    parser.add_argument("--trend-per-step", type=float, default=SimulationConfig.trend_per_step)
    parser.add_argument("--monthly-growth-rate", type=float, default=SimulationConfig.monthly_growth_rate)
    parser.add_argument("--noise-std", type=float, default=SimulationConfig.noise_std)
    parser.add_argument("--ar1-strength", type=float, default=SimulationConfig.ar1_strength)

    parser.add_argument("--daily-amplitude", type=float, default=SimulationConfig.daily_amplitude)
    parser.add_argument("--weekly-amplitude", type=float, default=SimulationConfig.weekly_amplitude)
    parser.add_argument("--monthly-amplitude", type=float, default=SimulationConfig.monthly_amplitude)
    parser.add_argument("--weekend-factor", type=float, default=SimulationConfig.weekend_factor)
    parser.add_argument("--evening-peak-boost", type=float, default=SimulationConfig.evening_peak_boost)

    parser.add_argument("--spike-probability", type=float, default=SimulationConfig.spike_probability)
    parser.add_argument("--spike-min-magnitude", type=float, default=SimulationConfig.spike_min_magnitude)
    parser.add_argument("--spike-max-magnitude", type=float, default=SimulationConfig.spike_max_magnitude)
    parser.add_argument("--spike-min-duration", type=int, default=SimulationConfig.spike_min_duration)
    parser.add_argument("--spike-max-duration", type=int, default=SimulationConfig.spike_max_duration)
    parser.add_argument("--event-cluster-scale", type=float, default=SimulationConfig.event_cluster_scale)

    parser.add_argument("--drop-probability", type=float, default=SimulationConfig.drop_probability)
    parser.add_argument("--drop-min-depth", type=float, default=SimulationConfig.drop_min_depth)
    parser.add_argument("--drop-max-depth", type=float, default=SimulationConfig.drop_max_depth)
    parser.add_argument("--drop-min-duration", type=int, default=SimulationConfig.drop_min_duration)
    parser.add_argument("--drop-max-duration", type=int, default=SimulationConfig.drop_max_duration)
    parser.add_argument("--recovery-min-duration", type=int, default=SimulationConfig.recovery_min_duration)
    parser.add_argument("--recovery-max-duration", type=int, default=SimulationConfig.recovery_max_duration)

    parser.add_argument("--output-csv", type=str, default="simulated_telecom_consumption.csv")
    parser.add_argument("--output-metadata", type=str, default="simulated_telecom_consumption_meta.json")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = SimulationConfig(
        periods=args.periods,
        months=args.months,
        years=args.years,
        freq=args.freq,
        start_date=args.start_date,
        seed=args.seed,
        base_consumption=args.base_consumption,
        trend_per_step=args.trend_per_step,
        monthly_growth_rate=args.monthly_growth_rate,
        noise_std=args.noise_std,
        ar1_strength=args.ar1_strength,
        daily_amplitude=args.daily_amplitude,
        weekly_amplitude=args.weekly_amplitude,
        monthly_amplitude=args.monthly_amplitude,
        weekend_factor=args.weekend_factor,
        evening_peak_boost=args.evening_peak_boost,
        spike_probability=args.spike_probability,
        spike_min_magnitude=args.spike_min_magnitude,
        spike_max_magnitude=args.spike_max_magnitude,
        spike_min_duration=args.spike_min_duration,
        spike_max_duration=args.spike_max_duration,
        event_cluster_scale=args.event_cluster_scale,
        drop_probability=args.drop_probability,
        drop_min_depth=args.drop_min_depth,
        drop_max_depth=args.drop_max_depth,
        drop_min_duration=args.drop_min_duration,
        drop_max_duration=args.drop_max_duration,
        recovery_min_duration=args.recovery_min_duration,
        recovery_max_duration=args.recovery_max_duration,
    )

    df = simulate_consumption_data(cfg)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    metadata = {
        "config": asdict(cfg),
        "rows": int(len(df)),
        "start_timestamp": str(df["timestamp"].iloc[0]),
        "end_timestamp": str(df["timestamp"].iloc[-1]),
        "min_consumption": float(df["data_consumption_gb"].min()),
        "max_consumption": float(df["data_consumption_gb"].max()),
        "mean_consumption": float(df["data_consumption_gb"].mean()),
        "event_count": int(df.attrs.get("event_count", 0)),
        "drop_count": int(df.attrs.get("drop_count", 0)),
    }

    output_meta = Path(args.output_metadata)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    output_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved dataset: {output_csv.resolve()}")
    print(f"Saved metadata: {output_meta.resolve()}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

