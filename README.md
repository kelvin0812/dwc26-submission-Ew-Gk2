# DWC26 - History Matching and Production Forecast

Our code for the ExxonMobil DataWorks Challenge 2026.

We compare all 100 provided cases with the real field production from 1998 to 2008, keep the 10 that match
best, and forecast their oil, gas and water up to 2028. We also train a Gaussian Process as a fast stand-in for
the simulator (4 parameters in, production curves out) and use it to see which parameter values fit the history.

## Setup

Needs Python 3.11 or newer.

```bash
pip install -r requirements.txt
```

The challenge data is not in this repo because the organisers don't allow it to be shared. Copy these 3 files
from the starter pack's `TestCases` folder into `TestCases/`:

- `01 Introduction.xlsx` - the 4 parameters of each case
- `02 Field Production.xlsx` - simulated and observed production, 1998-2008
- `09 Template Deliverable.xlsx` - the answer template (20-year version)

If one is missing the pipeline stops and tells you which. The grid files (03-08) are not needed.
`03 Train Cases 1.xlsx` is only used by the optional grid check in `src/explore.py`.

## Run

```bash
python src/run_pipeline.py
```

Takes about 2 minutes. You can also run a single step, e.g. `python src/step4_train_and_match.py`.

| Step | File | What it does |
|---|---|---|
| 1 | `src/step1_data.py` | Reads the parameters and production, drops the blank cells, makes quarterly tables |
| 2 | `src/step2_features.py` | Inputs (log permeability, scaled on train cases) and targets (quarterly cumulative oil and water), PCA on the curves |
| 3 | `src/step3_model.py` | The models we compare: Gaussian Process, kNN, polynomial ridge, LightGBM, MLP |
| 4 | `src/step4_train_and_match.py` | Picks the proxy (CV on train, choose on validation, test once), ranks the 100 cases by mismatch J and keeps the best 10 |
| 5 | `src/step5_forecast_and_export.py` | Back-tests the forecast method, forecasts 2008-2028, makes the plots, writes the CSV and the template |
| - | `src/explore.py` | Optional checks: parameter sensitivity and the grid comparison |

Cases 1-70 are train, 71-85 validation and 86-100 test (the organisers' split).

## Outputs

| Path | Content |
|---|---|
| `outputs/deliverable/09 Template Deliverable.xlsx` | Filled template, 10 cases, quarterly 2008-04 to 2028-01 |
| `outputs/deliverable/forecast_2008_2028.csv` | Same forecast in one table with rates and water cut (we use it in Power BI) |
| `outputs/deliverable/selected_cases.csv` | Which case is in each template sheet, its J and its 2028 totals |
| `outputs/figures/` | Plots; `07_history_match_and_forecast_summary.png` shows the whole result |
| `outputs/results/` | Tables behind the plots |

`data/` and `models/` are created when you run the pipeline. They are not committed because they contain the
challenge data (the GP keeps its training data).

## Results

- Proxy error on the 15 test cases: 0.04% (oil) and 0.09% (water) on the 2008 totals.
- Selected cases: 6, 73, 76, 97, 11, 40, 50, 90, 20, 4.
- 2028 forecast: oil 53.3-53.8 million STB, gas 19.4-19.6 million MSCF, water 62.7-65.9 million STB.
  With the other forecast methods we tested, 2028 oil ranges from 53 to 60 million STB.

Seeds are fixed, so every run gives the same numbers.
