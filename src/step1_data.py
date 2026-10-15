"""Step 1: read the case parameters (file 01) and the production data 1998-2008 (file 02).

We don't use the grid files 03-08. We checked and they're either the same for every case or just
scaled by the 4 parameters (see explore.py).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "TestCases"
PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

PARAMS = ["fault_trans", "poro_mult", "perm_mult", "aquifer_pv"]
PARAM_LABELS = {
    "fault_trans": "Fault transmissibility",
    "poro_mult": "Porosity multiplier",
    "perm_mult": "Permeability multiplier",
    "aquifer_pv": "Aquifer pore volume",
}

PRODUCTION_SHEETS = {
    "Oil production rate": "oil_rate",            # STB/d
    "Gas production rate": "gas_rate",            # MSCF/d
    "Water production rate": "water_rate",        # STB/d
    "Oil production cumulative": "oil_cum",       # STB
    "Gas production cumulative": "gas_cum",       # MSCF
    "Water production cumulative": "water_cum",   # STB
}
SERIES = list(PRODUCTION_SHEETS.values())

# organisers' files, we can't put them in the repo so they have to be copied into TestCases/
REQUIRED_FILES = ["01 Introduction.xlsx", "02 Field Production.xlsx", "09 Template Deliverable.xlsx"]


def check_data():
    missing = [f for f in REQUIRED_FILES if not (RAW / f).exists()]
    if missing:
        raise SystemExit(f"Missing challenge data in {RAW}:\n  " + "\n  ".join(missing)
                         + "\nCopy these files from the DWC26 starter pack (its TestCases folder) into TestCases/.")


def split_of(case):
    # the organisers gave us this split
    return "train" if case <= 70 else "validation" if case <= 85 else "test"


def case_number(label):
    # "Case 12" or "Case_12" -> 12
    return int(re.search(r"(\d+)", str(label)).group(1))


def load_cases():
    df = pd.read_excel(RAW / "01 Introduction.xlsx", sheet_name="Uncertainty Values", engine="calamine")
    df = df.rename(columns={
        "Item": "case",
        "Fault Transmissibility": "fault_trans",
        "Porosity Multiplier": "poro_mult",
        "Permeability Multiplier": "perm_mult",
        "Aquifer Pore Volume": "aquifer_pv",
    })
    df["case"] = df["case"].map(case_number)
    df["split"] = df["case"].map(split_of)
    return df


def load_production():
    """Each sheet has 3 header rows, then one row per simulator step and one column per case.
    The last column (RAMP_History) is the real field data."""
    src = RAW / "02 Field Production.xlsx"
    sim, hist = None, None
    for sheet, col in PRODUCTION_SHEETS.items():
        raw = pd.read_excel(src, sheet_name=sheet, header=None, engine="calamine")
        names = raw.iloc[1].tolist()
        data = raw.iloc[3:].reset_index(drop=True)
        data.columns = ["date"] + names[1:]
        data = data.copy()  # pandas complains about fragmentation without this (100+ columns)
        data["date"] = pd.to_datetime(data["date"])
        # some steps fall on the same date, so we use the row number as the id instead
        data["step"] = np.arange(len(data))

        # one column per case -> one row per case, then join the 6 series
        case_cols = [c for c in data.columns if str(c).startswith("Case_")]
        long = data.melt(id_vars=["step", "date"], value_vars=case_cols, var_name="case", value_name=col)
        long["case"] = long["case"].map(case_number)
        sim = long if sim is None else sim.merge(long, on=["case", "step", "date"])

        h = data[["step", "date", "RAMP_History"]].rename(columns={"RAMP_History": col})
        hist = h if hist is None else hist.merge(h, on=["step", "date"])

    # all cases share one time axis but each only reports on ~170 of the steps, the rest is blank
    # (went from 1,109,600 rows to 17,222 after dropping them)
    sim = sim.dropna(subset=SERIES, how="all")
    sim = sim.sort_values(["case", "step"]).reset_index(drop=True)
    sim[SERIES] = sim[SERIES].astype("float64")
    return sim, hist


def to_quarterly(sim):
    """Take the last simulator step on or before each quarter date. Exact for cumulatives."""
    last = sim.groupby(["case", "date"]).tail(1)
    quarters = pd.date_range("1998-01-01", "2008-01-01", freq="QS")
    return (last.set_index("date").groupby("case")[SERIES]
            .apply(lambda g: g.reindex(g.index.union(quarters)).ffill().loc[quarters].rename_axis("date"))
            .reset_index())


def load_observed(hist):
    """Rates and injection come from the 'Field Production' sheet, cumulatives from RAMP_History
    (the sheet doesn't have cumulatives). We checked, the rates match between the two."""
    field = pd.read_excel(RAW / "02 Field Production.xlsx", sheet_name="Field Production", engine="calamine")
    field = field.rename(columns={
        "Date": "date",
        "Oil production rate (STB/d)": "oil_rate",
        "Gas production rate (MSCF/d)": "gas_rate",
        "Water production rate (STB/d)": "water_rate",
        "Water injection rate (STB/d)": "water_inj_rate",
    }).drop(columns=["Identifier"])
    field["date"] = pd.to_datetime(field["date"])
    cums = hist.dropna(subset=["oil_cum", "gas_cum", "water_cum"])[["date", "oil_cum", "gas_cum", "water_cum"]]
    return field.merge(cums, on="date")


def read_cases():
    return pd.read_parquet(PROC / "cases.parquet").set_index("case")


def read_quarterly():
    return pd.read_parquet(PROC / "production_quarterly.parquet")


def read_observed():
    return pd.read_parquet(PROC / "observed_quarterly.parquet")


def main():
    check_data()
    PROC.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    sim, hist = load_production()
    quarterly = to_quarterly(sim)
    observed = load_observed(hist)

    cases.to_parquet(PROC / "cases.parquet", index=False)
    sim.to_parquet(PROC / "production_steps.parquet", index=False)
    quarterly.to_parquet(PROC / "production_quarterly.parquet", index=False)
    observed.to_parquet(PROC / "observed_quarterly.parquet", index=False)

    print("Step 1 done")
    print("  cases per split:", cases.groupby("split").size().to_dict())
    print(f"  simulated rows kept: {len(sim):,} | quarterly rows: {len(quarterly):,} | observed quarters: {len(observed)}")


if __name__ == "__main__":
    main()
