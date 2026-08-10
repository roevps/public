# Mediterranean SST reconstruction pipeline

Run in order:

```bash
python -m pip install -r requirements.txt
python 1_download_data.py
python 2_calculate_trends.py
python 3_create_plots.py
python 4_garch_model.py
```

The pipeline uses:
- NOAA OISST v2.1 (daily)
- NOAA ERSST v5 (monthly)
- Met Office HadSST4.2 actuals median
- JMA COBE-SST2 distributed through NOAA PSL

Method:
- common Mediterranean extraction domain
- stricter Mediterranean mask during analysis
- cosine(latitude) area weighting
- 1982–2011 monthly climatology
- trends from monthly anomalies since 1982 with HAC(12) standard errors
- OISST daily 90th-percentile exceedance analysis
- OISST grid-cell trend map
- GARCH(1,1), Student-t innovations, on detrended monthly anomalies
