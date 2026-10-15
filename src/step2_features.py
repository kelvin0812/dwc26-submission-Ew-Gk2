"""Step 2: build the inputs X, the targets Y and the PCA.

X = the 4 parameters, scaled 0-1 using the train cases only. Permeability is logged because it
spans ~20x and its effect flattens out above ~4.
Y = quarterly cumulative oil and water from 1999-01 to 2008-01. No separate gas model, GOR is
basically constant (0.363) so gas = GOR x oil.
PCA squeezes each 37-point curve into 5-6 numbers, easier to learn with only 70 cases.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from step1_data import MODELS, OUTPUTS, PROC, read_cases, read_quarterly

FEATURES = PROC / "features"
PHASES = ["oil_cum", "water_cum"]
TARGET_START = "1999-01-01"
UNIT = 1e6  # million STB
INPUTS = ["poro_mult", "log_perm_mult", "aquifer_pv", "fault_trans"]

# PCA can lose at most 0.1% of the final cumulative. perm/aquifer only move oil by ~0.8%
# so we can't lose more than that. ends up as 5 components for oil, 6 for water
TOL = 0.001


def build_inputs(cases, train):
    raw = cases.assign(log_perm_mult=np.log(cases.perm_mult))[INPUTS]
    lo, hi = raw[train].min(), raw[train].max()
    scaled = (raw - lo) / (hi - lo)
    X = pd.concat([cases[["split"]], raw.add_prefix("raw_"), scaled], axis=1)
    return X, {"lo": lo.to_dict(), "hi": hi.to_dict()}


def build_targets(quarterly):
    q = quarterly[quarterly.date >= TARGET_START]
    return {p: q.pivot(index="case", columns="date", values=p) / UNIT for p in PHASES}


def quarterly_rates(Y, dates):
    """Average daily rate in each quarter from a cumulative curve."""
    days = np.diff(pd.to_datetime(dates)).astype("timedelta64[D]").astype(float)
    return np.diff(Y, axis=1) / days


def worst_error(Y_true, Y_hat):
    # worst point on each curve, as a fraction of the final value
    return np.abs(Y_hat - Y_true).max(axis=1) / np.abs(Y_true[:, -1])


def choose_n_components(Y_tr, Y_val, kmax=12):
    """Smallest number of components that keeps every train and validation curve within TOL."""
    rows = []
    for k in range(1, kmax + 1):
        pca = PCA(n_components=k).fit(Y_tr)
        rows.append({"k": k, "explained_var": pca.explained_variance_ratio_.sum(),
                     "max_err_train": worst_error(Y_tr, pca.inverse_transform(pca.transform(Y_tr))).max(),
                     "max_err_val": worst_error(Y_val, pca.inverse_transform(pca.transform(Y_val))).max()})
    table = pd.DataFrame(rows).set_index("k")
    ok = table[(table.max_err_train < TOL) & (table.max_err_val < TOL)]
    return (int(ok.index[0]) if len(ok) else kmax), table


def encode(pcas, Y):
    """curves -> scores"""
    return np.hstack([pcas[p].transform(Y[p].values) for p in PHASES])


def decode(pcas, scores):
    """scores -> curves"""
    out, i = {}, 0
    for p in PHASES:
        k = pcas[p].n_components_
        out[p] = pcas[p].inverse_transform(scores[:, i:i + k])
        i += k
    return out


def load_features():
    X = pd.read_parquet(FEATURES / "X.parquet")
    Y = {p: pd.read_parquet(FEATURES / f"Y_{p}.parquet") for p in PHASES}
    scores = pd.read_parquet(FEATURES / "Y_scores.parquet")
    info = joblib.load(MODELS / "pca.joblib")
    return X, Y, scores, info


def main():
    FEATURES.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "results").mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    cases, quarterly = read_cases(), read_quarterly()
    train = cases.split == "train"
    X, bounds = build_inputs(cases, train)
    Y = build_targets(quarterly)
    assert (Y["oil_cum"].index == X.index).all(), "X and Y case order differs"

    # GOR from the train cases, used later to get gas from oil
    q_tr = quarterly[quarterly.case.isin(cases.index[train]) & (quarterly.date >= TARGET_START)]
    gor = float((q_tr.gas_cum / q_tr.oil_cum).median())

    tr, va = X.split == "train", X.split == "validation"
    n_comp = {p: choose_n_components(Y[p][tr].values, Y[p][va].values)[0] for p in PHASES}
    pcas = {p: PCA(n_components=n_comp[p]).fit(Y[p][tr].values) for p in PHASES}  # fit on train only
    names = [f"{p.split('_')[0]}_pc{j + 1}" for p in PHASES for j in range(n_comp[p])]
    scores = pd.DataFrame(encode(pcas, Y), index=X.index, columns=names)

    # check how much PCA loses, on the curves and on the rates
    dates = list(Y[PHASES[0]].columns)
    rebuilt = decode(pcas, scores.values)
    rows = []
    for p in PHASES:
        err = pd.Series(worst_error(Y[p].values, rebuilt[p]), index=X.index)
        r_true, r_hat = quarterly_rates(Y[p].values, dates), quarterly_rates(rebuilt[p], dates)
        scale = np.abs(r_true).max(axis=1)
        rate_err = pd.Series(np.abs(r_hat - r_true).max(axis=1) / np.where(scale > 0, scale, 1), index=X.index)
        for s in ("train", "validation", "test"):
            m = X.split == s
            rows.append({"phase": p, "split": s, "components": n_comp[p],
                         "max_curve_err_%": 100 * err[m].max(), "max_rate_err_%": 100 * rate_err[m].max()})
    check = pd.DataFrame(rows)
    check.to_csv(OUTPUTS / "results" / "pca_check.csv", index=False)

    X.to_parquet(FEATURES / "X.parquet")
    for p in PHASES:
        Y[p].set_axis([d.strftime("%Y-%m-%d") for d in Y[p].columns], axis=1).to_parquet(FEATURES / f"Y_{p}.parquet")
    scores.to_parquet(FEATURES / "Y_scores.parquet")
    joblib.dump({"pcas": pcas, "dates": dates, "bounds": bounds, "gor": gor}, MODELS / "pca.joblib")

    print("Step 2 done")
    print(f"  GOR (train) = {gor:.5f} | PCA components: {n_comp}")
    print(check.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
