from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Tuple

from microgrid.config import DATASET_DIR, DATA_DIR


# Module-level cache (avoids re-reading large CSVs on every import)
_CACHE: dict = {}
_FC_CACHE: dict = {}


def load_raw_series(
    dataset_dir: Path = DATASET_DIR,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:

    global _CACHE
    if _CACHE:
        return (
            _CACHE["load"],
            _CACHE["solar"],
            _CACHE["wind"],
            _CACHE["solar_ec"],
            _CACHE["wind_ec"],
        )

    dataset_dir = Path(dataset_dir)

    load_5min = pd.read_csv(
        dataset_dir / "BA_load_actuals_2018.csv",
        parse_dates=["timestamp"],
        index_col="timestamp",
    )["total_load"].astype(float)

    solar_5min = pd.read_csv(
        dataset_dir / "BA_solar_actuals_Existing_2018.csv",
        parse_dates=["timestamp"],
        index_col="timestamp",
    )["total_solar"].astype(float)

    wind_5min = pd.read_csv(
        dataset_dir / "BA_wind_actuals_Existing_2018.csv",
        parse_dates=["timestamp"],
        index_col="timestamp",
    )["total_wind"].astype(float)

    solar_ecmwf = pd.read_csv(
        dataset_dir / "BA_solar_day-ahead_fcst_ecmwf_Existing_2018.csv",
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    wind_ecmwf = pd.read_csv(
        dataset_dir / "BA_wind_day-ahead_fcst_ecmwf_Existing_2018.csv",
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    _CACHE = {
        "load":     load_5min,
        "solar":    solar_5min,
        "wind":     wind_5min,
        "solar_ec": solar_ecmwf,
        "wind_ec":  wind_ecmwf,
    }

    return load_5min, solar_5min, wind_5min, solar_ecmwf, wind_ecmwf


def load_forecast_series(
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    if _FC_CACHE:
        return _FC_CACHE["load_fc"]

    data_dir = Path(data_dir)
    fc_path  = data_dir / "BA_load_day-ahead_fcst_2018.csv"

    df = pd.read_csv(fc_path, parse_dates=["forecast_time"], index_col="forecast_time")
    # Average duplicate timestamps (two 30-min sub-steps per hour)
    df = df.groupby(df.index).mean()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"

    _FC_CACHE["load_fc"] = df
    return df


def extract_day(
    series_5min: pd.Series,
    date: str,
) -> pd.Series:
    day_ts = pd.Timestamp(date, tz="UTC")
    mask   = series_5min.index.normalize() == day_ts
    return series_5min[mask]


def extract_day_df(
    df: pd.DataFrame,
    date: str,
) -> pd.DataFrame:
    """Return the sub-DataFrame for a single calendar day (UTC)."""
    day_ts = pd.Timestamp(date, tz="UTC")
    mask   = df.index.normalize() == day_ts
    return df[mask]
