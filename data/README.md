# Dataset

The case study uses the ERCOT 2018 data of the **ARPA-E PERFORM datasets** produced by the
National Renewable Energy Laboratory (NREL) for the ARPA-E PERFORM program. The six files in this
folder are the system-level (balancing-authority, "BA") series for 2018, converted from the
published HDF5 files to CSV.

| | |
|---|---|
| Dataset | ARPA-E PERFORM datasets |
| Producer | National Renewable Energy Laboratory (NREL), 2022 |
| DOI | [10.25984/1891136](https://doi.org/10.25984/1891136) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Region and level | ERCOT, system (balancing-authority) level |
| Year | 2018 (ERCOT forecasts are published for 2018) |
| Time zone | UTC, as published |
| Units | MW |
| Original files | HDF5 on AWS, `s3://arpa-e-perform/` ([registry](https://registry.opendata.aws/arpa-e-perform/)) |
| Documentation | [PERFORM-Forecasts/documentation](https://github.com/PERFORM-Forecasts/documentation), [ERCOT technical report](https://www.nlr.gov/docs/fy23osti/79498.pdf) |

## Files

| File | Content | Resolution | Rows | Value columns | Range (MW) |
|---|---|---|---|---|---|
| `BA_load_actuals_2018.csv` | ERCOT system load | 5 min | 105,120 | `total_load` | 27,240 to 73,364 |
| `BA_solar_actuals_Existing_2018.csv` | solar generation, existing fleet | 5 min | 105,120 | `total_solar` | 0 to 651 |
| `BA_wind_actuals_Existing_2018.csv` | wind generation, existing fleet | 5 min | 105,120 | `total_wind` | 80 to 20,477 |
| `BA_load_day-ahead_fcst_2018.csv` | day-ahead probabilistic load forecast, 1st to 99th percentile | hourly | 17,520 | `member_0` to `member_98` | 17,894 to 79,258 |
| `BA_solar_day-ahead_fcst_ecmwf_Existing_2018.csv` | day-ahead solar forecast from the 51 ECMWF ensemble members, existing fleet | hourly | 8,760 | `M00` to `M50` | 0 to 639 |
| `BA_wind_day-ahead_fcst_ecmwf_Existing_2018.csv` | day-ahead wind forecast from the 51 ECMWF ensemble members, existing fleet | hourly | 8,760 | `M00` to `M50` | 14 to 20,493 |

The first column is the timestamp (`timestamp`, or `forecast_time` in the load forecast file).

How PERFORM builds these series (from the PERFORM documentation):

- **Load actuals** are ERCOT system load collected from the ISO.
- **Solar actuals** are simulated with NREL's System Advisor Model from the National Solar Radiation
  Database at each site and summed to the system level. **Wind actuals** are simulated from Weather
  Research and Forecasting (WRF) model meteorology in the same way. "Existing" is the fleet of sites in
  operation; PERFORM also publishes planned sites from the interconnection queue, which are not used.
- **Day-ahead forecasts** have an 11-hour lead time, a 48-hour horizon, hourly resolution and a daily
  update. Consecutive daily issues overlap, so every hour has two rows in the load forecast file.
- **Solar and wind day-ahead forecasts** come from the 51-member ECMWF ensemble.
- **Load day-ahead forecasts** come from a deep-learning ensemble (recurrent and convolutional neural
  networks and gradient boosting) turned into percentiles by an adaptive Gaussian model.

## Study days

Six days of 2018, selected to span different seasons and diverse renewable-generation and
forecast-error conditions. A study day is the UTC calendar day (288 five-minute rows). Values below are computed from the files in this
folder by `scripts/reproduce_paper.py` (`results/paper/study_days.csv`); shares are of daily load
energy.

| Date | Season | Peak load (MW) | Mean load (MW) | Net load range (MW) | Solar share | Wind share | Renewable share |
|---|---|---|---|---|---|---|---|
| 2018-03-03 | Winter | 36,970 | 33,267 | 13,540 to 28,887 | 0.005 | 0.366 | 0.371 |
| 2018-04-13 | Spring | 46,795 | 40,127 | 12,191 to 29,246 | 0.004 | 0.466 | 0.470 |
| 2018-04-14 | Spring | 43,805 | 33,641 | 10,570 to 25,146 | 0.007 | 0.512 | 0.518 |
| 2018-06-27 | Summer | 69,075 | 55,281 | 25,327 to 64,332 | 0.005 | 0.189 | 0.195 |
| 2018-07-23 | Summer | 73,111 | 58,515 | 32,704 to 70,306 | 0.005 | 0.109 | 0.113 |
| 2018-10-26 | Fall | 37,906 | 34,571 | 28,207 to 37,013 | 0.006 | 0.014 | 0.020 |

2018-07-23 is the peak-load day (Fig. 3). 2018-04-14 has the highest wind share and the lowest net
load, and 2018-10-26 has almost no wind.

## Processing in the code

`microgrid/data_loading.py` reads the files and `microgrid/preprocessing.py` builds the arrays used by
every solver:

1. Each study day is cut from the series by UTC calendar day.
2. Load, solar and wind actuals are averaged from 5 minutes to 15 minutes (96 steps per day).
3. The hourly ECMWF members are interpolated in time to the same 15-minute steps (51 x 96 per day).
4. The two load-forecast rows of each hour are averaged, the 99 percentiles are averaged into one
   hourly forecast, and each hourly value is held for its four 15-minute steps.
5. All series are divided by P_scale = 73,262.2 MW, the highest hourly mean load of 2018
   (2018-07-19 20:00 UTC), so every quantity is in per unit.
6. The 51 ECMWF members are reduced to three scenarios: k-means (k = 3, 20 initializations, random
   state 42) on each member's 15-minute solar and wind trajectory of the day, keeping the medoid
   member of each cluster with weight w_k = n_k/51 (`microgrid/scenario_generation.py`).

How the series enter the dispatch problem:

| Series | Used as |
|---|---|
| Load actuals | load L_t of every dispatch window |
| ECMWF scenarios (three medoids) | renewable generation R_t^(k) in the scenario-weighted power balance |
| Load forecast and ECMWF ensemble mean | day-ahead MILP that gives the SOC reference SOC^DA |
| Solar and wind actuals | realized generation recorded in each `df_schedule.csv` |

The processed arrays are cached in `data/processed/` on the first run; that folder is not tracked.

## Citation

B. Sergi, C. Feng, F. Zhang, B.-M. Hodge, R. Ring-Jarvi, R. Bryce, K. Doubleday, M. Rose, G. Buster
and M. Rossol, "ARPA-E PERFORM datasets," National Renewable Energy Laboratory, 2022,
doi: 10.25984/1891136.

```
@misc{sergi2022perform,
  author    = {Sergi, Brian and Feng, Cong and Zhang, Flora and Hodge, Bri-Mathias and
               Ring-Jarvi, Ross and Bryce, Richard and Doubleday, Kate and Rose, Megan and
               Buster, Grant and Rossol, Michael},
  title     = {{ARPA-E} {PERFORM} datasets},
  publisher = {National Renewable Energy Laboratory},
  year      = {2022},
  doi       = {10.25984/1891136}
}
```

The data are redistributed here under CC BY 4.0 with this attribution. The MIT license of the
repository covers the code only.
