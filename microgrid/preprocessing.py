from __future__ import annotations

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from microgrid.config import (
    P_GRID_FRAC,
    P_BATT_FRAC,
    SOC_CAP_HOURS,
    DATA_PROC,
    DATASET_DIR,
    processed_dir_for_date,
)
from microgrid.data_loading import (
    load_raw_series,
    load_forecast_series,
    extract_day,
    extract_day_df,
)


# Global (whole-year) scale factor - one value shared across all dates so
# battery absolute MW is fixed (standard microgrid sizing convention).
_GLOBAL_SCALE_CACHE: Dict[str, float] = {}


def compute_global_scale(dataset_dir: Path | None = None) -> float:
    """Highest hourly mean load of the year (per-unit base). Cached on disk + in memory."""
    if "scale" in _GLOBAL_SCALE_CACHE:
        return _GLOBAL_SCALE_CACHE["scale"]

    cache_file = DATA_PROC / "global_scale.json"
    if cache_file.exists():
        with open(cache_file) as f:
            v = float(json.load(f)["scale_mw"])
        _GLOBAL_SCALE_CACHE["scale"] = v
        return v

    load_5min, _, _, _, _ = load_raw_series(dataset_dir or DATASET_DIR)
    load_1h_year = load_5min.resample("1h").mean()
    pos = load_1h_year.values[load_1h_year.values > 0]
    scale = float(np.max(pos)) if len(pos) else 1.0   # annual peak hourly load

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({"scale_mw": scale, "method": "MAX_annual_hourly_load"}, f, indent=2)
    _GLOBAL_SCALE_CACHE["scale"] = scale
    print(f"[global_scale] max annual hourly load = {scale:.1f} MW")
    return scale


# Output data structure
@dataclass
class ProcessedDay:
    """All normalised 15-min arrays needed for optimisation."""
    date:       str
    scale_mw:   float                  # annual peak load - denormalisation factor

    load_n:     np.ndarray             # shape (96,) - normalised 15-min load
    solar_n:    np.ndarray             # shape (96,) - normalised 15-min solar
    wind_n:     np.ndarray             # shape (96,) - normalised 15-min wind
    solar_ec_n: np.ndarray             # shape (51, 96) - ECMWF solar upsampled to 15-min
    wind_ec_n:  np.ndarray             # shape (51, 96) - ECMWF wind upsampled to 15-min

    P_G:        float                  # per-unit grid capacity
    P_B:        float                  # per-unit battery capacity
    SOC_CAP:    float                  # per-unit battery energy capacity (pu·h)

    # Day-ahead forecast arrays (normalized, shape 24)
    # load_fc_n  : mean day-ahead load forecast
    # net_load_fc_n : load_fc_n - mean(solar_ec_n) - mean(wind_ec_n)
    load_fc_n:    np.ndarray = field(default_factory=lambda: np.zeros(24))
    net_load_fc_n: np.ndarray = field(default_factory=lambda: np.zeros(24))

    load_1h:    pd.Series  = field(default_factory=pd.Series)
    solar_1h:   pd.Series  = field(default_factory=pd.Series)
    wind_1h:    pd.Series  = field(default_factory=pd.Series)


# Main preprocessing function
def preprocess_date(
    date: str,
    dataset_dir: Path | None = None,
    force: bool = False,
) -> ProcessedDay:
 
    out_dir = processed_dir_for_date(date)
    meta_file = out_dir / "metadata.json"

    # Invalidate cache if stored scale differs from current global scale
    global_scale = compute_global_scale(dataset_dir or DATASET_DIR)
    if not force and meta_file.exists():
        try:
            with open(meta_file) as f:
                _m = json.load(f)
            if abs(float(_m.get("scale_mw", -1)) - global_scale) < 1e-3:
                return _load_processed(date, out_dir)
        except Exception:
            pass
        # Cache stale - fall through to rebuild

    # 1. Load raw series
    load_5min, solar_5min, wind_5min, solar_ecmwf, wind_ecmwf = load_raw_series(dataset_dir or DATASET_DIR)
    load_fc_hourly = load_forecast_series()  # day-ahead load forecast (8760, 99)

    # 2. Extract day
    load_day   = extract_day(load_5min,  date)
    solar_day  = extract_day(solar_5min, date)
    wind_day   = extract_day(wind_5min,  date)
    solar_ec_d = extract_day_df(solar_ecmwf, date)
    wind_ec_d  = extract_day_df(wind_ecmwf,  date)

    assert len(load_day) == 288, f"Expected 288 5-min rows, got {len(load_day)}"

    # 3. Resample 5-min -> 15-min (96 steps per day)
    load_15  = load_day.resample("15min").mean()
    solar_15 = solar_day.resample("15min").mean()
    wind_15  = wind_day.resample("15min").mean()
    assert len(load_15) == 96, f"Expected 96 15-min rows, got {len(load_15)}"

    # Keep 1h series for dataset characteristics only
    load_1h  = load_15.resample("1h").mean()
    solar_1h = solar_15.resample("1h").mean()
    wind_1h  = wind_15.resample("1h").mean()

    # 4. Align ECMWF hourly forecasts -> upsample to 15-min (96 steps)
    def align_ec_15min(ec_day: pd.DataFrame, target_idx: pd.DatetimeIndex) -> np.ndarray:
        """Interpolate ECMWF hourly members to 15-min target index. Returns (n_members, 96)."""
        aligned = (
            ec_day
            .reindex(ec_day.index.union(target_idx))
            .interpolate(method="time")
            .reindex(target_idx)
            .ffill()
            .bfill()
        )
        return aligned.values.T   # (n_members, 96)

    solar_ec_arr = align_ec_15min(solar_ec_d, load_15.index)   # (51, 96)
    wind_ec_arr  = align_ec_15min(wind_ec_d,  load_15.index)   # (51, 96)

    # 5. Per-unit normalisation
    scale_mw = compute_global_scale(dataset_dir or DATASET_DIR)

    load_n     = load_15.values  / scale_mw    # (96,)
    solar_n    = solar_15.values / scale_mw    # (96,)
    wind_n     = wind_15.values  / scale_mw    # (96,)
    solar_ec_n = solar_ec_arr    / scale_mw    # (51, 96)
    wind_ec_n  = wind_ec_arr     / scale_mw    # (51, 96)

    # 5b. Day-ahead load forecast -> upsample to 15-min
    day_ts_utc = pd.Timestamp(date, tz="UTC")
    fc_day_mask = load_fc_hourly.index.normalize() == day_ts_utc
    fc_day = load_fc_hourly[fc_day_mask]
    if len(fc_day) >= 24:
        load_fc_1h_mw = fc_day.iloc[:24].mean(axis=1).values   # (24,) MW
    else:
        load_fc_1h_mw = load_1h.values.copy()
    load_fc_n     = np.repeat(load_fc_1h_mw, 4) / scale_mw    # (96,)

    solar_fc_n    = solar_ec_n.mean(axis=0)    # (96,)
    wind_fc_n     = wind_ec_n.mean(axis=0)     # (96,)
    net_load_fc_n = load_fc_n - solar_fc_n - wind_fc_n

    # 6. Microgrid sizing (per-unit)
    P_G     = P_GRID_FRAC
    P_B     = P_BATT_FRAC
    SOC_CAP = P_B * SOC_CAP_HOURS

    # 7. Persist
    np.save(out_dir / "load_n.npy",        load_n)
    np.save(out_dir / "solar_n.npy",       solar_n)
    np.save(out_dir / "wind_n.npy",        wind_n)
    np.save(out_dir / "solar_ec_n.npy",    solar_ec_n)
    np.save(out_dir / "wind_ec_n.npy",     wind_ec_n)
    np.save(out_dir / "load_fc_n.npy",     load_fc_n)
    np.save(out_dir / "net_load_fc_n.npy", net_load_fc_n)

    # Save 15-min profiles as CSV
    hourly_df = pd.DataFrame({
        "timestamp": load_15.index.astype(str),
        "load_mw":   load_15.values,
        "solar_mw":  solar_15.values,
        "wind_mw":   wind_15.values,
        "load_n":    load_n,
        "solar_n":   solar_n,
        "wind_n":    wind_n,
    })
    hourly_df.to_csv(out_dir / "hourly_profiles.csv", index=False)

    # Metadata
    metadata = {
        "date":       date,
        "scale_mw":   scale_mw,
        "P_G":        P_G,
        "P_B":        P_B,
        "SOC_CAP":    SOC_CAP,
        "n_load_rows": int(len(load_n)),
        "n_ec_members": int(solar_ec_n.shape[0]),
    }
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"[preprocess] {date} | scale={scale_mw:.1f} MW | "
        f"P_G={P_G:.2f}pu | P_B={P_B:.2f}pu | SOC_CAP={SOC_CAP:.2f}pu·h"
    )

    return ProcessedDay(
        date=date, scale_mw=scale_mw,
        load_n=load_n, solar_n=solar_n, wind_n=wind_n,
        solar_ec_n=solar_ec_n, wind_ec_n=wind_ec_n,
        load_fc_n=load_fc_n, net_load_fc_n=net_load_fc_n,
        P_G=P_G, P_B=P_B, SOC_CAP=SOC_CAP,
        load_1h=load_1h, solar_1h=solar_1h, wind_1h=wind_1h,
    )


# Load from already-processed files
def _load_processed(date: str, out_dir: Path) -> ProcessedDay:

    with open(out_dir / "metadata.json") as f:
        meta = json.load(f)

    load_n     = np.load(out_dir / "load_n.npy")
    solar_n    = np.load(out_dir / "solar_n.npy")
    wind_n     = np.load(out_dir / "wind_n.npy")
    solar_ec_n = np.load(out_dir / "solar_ec_n.npy")
    wind_ec_n  = np.load(out_dir / "wind_ec_n.npy")

    # Load forecast arrays - regenerate if missing (old cache)
    fc_path  = out_dir / "load_fc_n.npy"
    nlf_path = out_dir / "net_load_fc_n.npy"
    if fc_path.exists() and nlf_path.exists():
        load_fc_n     = np.load(fc_path)
        net_load_fc_n = np.load(nlf_path)
    else:
        # Old cache: compute from ECMWF mean + load forecast on-the-fly
        load_fc_n, net_load_fc_n = _compute_forecast_arrays(
            date, solar_ec_n, wind_ec_n, meta["scale_mw"]
        )
        np.save(fc_path,  load_fc_n)
        np.save(nlf_path, net_load_fc_n)

    profiles_df = pd.read_csv(out_dir / "hourly_profiles.csv", parse_dates=["timestamp"])
    # Resample 15-min profiles back to 1h for dataset characteristics
    profiles_df = profiles_df.set_index("timestamp")
    load_1h  = pd.Series(profiles_df["load_mw"].resample("1h").mean().values)
    solar_1h = pd.Series(profiles_df["solar_mw"].resample("1h").mean().values)
    wind_1h  = pd.Series(profiles_df["wind_mw"].resample("1h").mean().values)

    print(f"[preprocess] {date} - loaded from cache (scale={meta['scale_mw']:.1f} MW)")

    return ProcessedDay(
        date=date,
        scale_mw=meta["scale_mw"],
        load_n=load_n, solar_n=solar_n, wind_n=wind_n,
        solar_ec_n=solar_ec_n, wind_ec_n=wind_ec_n,
        load_fc_n=load_fc_n, net_load_fc_n=net_load_fc_n,
        P_G=meta["P_G"], P_B=meta["P_B"], SOC_CAP=meta["SOC_CAP"],
        load_1h=load_1h, solar_1h=solar_1h, wind_1h=wind_1h,
    )


def _compute_forecast_arrays(
    date: str,
    solar_ec_n: np.ndarray,
    wind_ec_n: np.ndarray,
    scale_mw: float,
) -> tuple:
    """Compute load_fc_n and net_load_fc_n for a date.  Used when cache is stale."""
    load_fc_hourly = load_forecast_series()
    day_ts_utc = pd.Timestamp(date, tz="UTC")
    fc_day_mask = load_fc_hourly.index.normalize() == day_ts_utc
    fc_day = load_fc_hourly[fc_day_mask]
    n_steps = solar_ec_n.shape[1]   # 96 for 15-min, 24 for legacy
    if len(fc_day) >= 24:
        load_fc_1h = fc_day.iloc[:24].mean(axis=1).values / scale_mw
    else:
        load_fc_1h = np.ones(24)
    repeats   = n_steps // 24
    load_fc_n = np.repeat(load_fc_1h, repeats) if repeats > 1 else load_fc_1h
    solar_fc_n    = solar_ec_n.mean(axis=0)   # (n_steps,)
    wind_fc_n     = wind_ec_n.mean(axis=0)    # (n_steps,)
    net_load_fc_n = load_fc_n - solar_fc_n - wind_fc_n
    return load_fc_n, net_load_fc_n


# Dataset characteristics (for combined analysis)
def compute_dataset_characteristics(pday: ProcessedDay) -> Dict:
    """Compute descriptive statistics for one processed date."""
    load  = pday.load_1h.values if len(pday.load_1h) > 0 else pday.load_n * pday.scale_mw
    solar = pday.solar_1h.values if len(pday.solar_1h) > 0 else pday.solar_n * pday.scale_mw
    wind  = pday.wind_1h.values  if len(pday.wind_1h)  > 0 else pday.wind_n  * pday.scale_mw

    total_load      = float(load.sum())
    peak_load       = float(load.max())
    mean_load       = float(load.mean())
    load_range      = float(load.max() - load.min())

    total_solar     = float(solar.sum())
    total_wind      = float(wind.sum())
    total_renewable = total_solar + total_wind

    wind_pen  = total_wind  / max(total_load, 1e-9)
    solar_pen = total_solar / max(total_load, 1e-9)
    renew_pen = total_renewable / max(total_load, 1e-9)

    net_load      = load - solar - wind
    net_load_min  = float(net_load.min())
    net_load_max  = float(net_load.max())
    net_load_mean = float(net_load.mean())

    # 1-indexed to match df_compare/df_schedule hour numbering
    peak_hour         = int(np.argmax(load)) + 1
    min_net_load_hour = int(np.argmin(net_load)) + 1

    # Renewable variability: coefficient of variation of net load
    renew_variability = float(np.std(net_load) / max(np.mean(net_load), 1e-9))

    # Season label
    month = pd.Timestamp(pday.date).month
    if month in (12, 1, 2):
        season = "Winter"
    elif month in (3, 4, 5):
        season = "Spring"
    elif month in (6, 7, 8):
        season = "Summer"
    else:
        season = "Fall"

    # Concise character string
    wind_label  = (
        "near-zero wind" if wind_pen < 0.05
        else "low wind"    if wind_pen < 0.15
        else "moderate wind" if wind_pen < 0.30
        else "high wind"   if wind_pen < 0.50
        else "very high wind"
    )
    load_label  = "high load" if peak_load > 0.9 * pday.scale_mw else "moderate load"
    character   = f"{wind_label}, {season.lower()}, {load_label}"

    return {
        "date":                  pday.date,
        "season":                season,
        "peak_load_mw":          round(peak_load, 1),
        "mean_load_mw":          round(mean_load, 1),
        "load_range_mw":         round(load_range, 1),
        "wind_penetration":      round(wind_pen,  3),
        "solar_penetration":     round(solar_pen, 3),
        "renewable_penetration": round(renew_pen, 3),
        "net_load_min_mw":       round(net_load_min, 1),
        "net_load_max_mw":       round(net_load_max, 1),
        "net_load_mean_mw":      round(net_load_mean, 1),
        "renewable_variability": round(renew_variability, 3),
        "peak_hour":             peak_hour,
        "min_net_load_hour":     min_net_load_hour,
        "character":             character,
    }
