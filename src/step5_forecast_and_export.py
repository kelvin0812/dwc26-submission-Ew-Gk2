"""Step 5: forecast 2008-2028, save the CSV and fill in the template.

There's no data after 2008, so we extend each selected case's decline trend. To choose how, we cut
the data earlier and forecast years we already know (back-testing). Back-test 1 picks the trend
method on all 100 cases with the 2005 and 2006 cuts (2003 was too early, water had only just
broken through). Back-test 2 picks how to connect the cases to the real field, redoing the top-10
selection with only the data before the cut.
"""
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from scipy.optimize import curve_fit

from step1_data import OUTPUTS, RAW, check_data, read_observed, read_quarterly
from step2_features import PHASES, load_features
from step4_train_and_match import N_PICKS, RESULTS, make_objective, select_top_cases

FIGS, DELIVERABLE = OUTPUTS / "figures", OUTPUTS / "deliverable"
FC_DATES = pd.date_range("2008-04-01", "2028-01-01", freq="QS")   # same quarters as the template
BT_CUTS = ["2003-01-01", "2005-01-01", "2006-01-01"]
SELECT_CUTS = ["2005", "2006"]
# we only deliver tie-ins that are smooth in both cumulative and rate. raw and cum_shift are
# only back-tested. cum_shift keeps the model's rate, and with our cases that made oil jump up
# ~12% at 2008, which can't happen in a declining field
TIE_INS = ["rate_ratio", "wor_shift"]
WINDOWS = [8, 12, 16]   # how many quarters we fit the trend on
METHODS = [(m, w) for m in ("arps_wor", "liquid_wor") for w in WINDOWS]


# --- decline curves
def avg_rates(cum, dates):
    """Quarterly rates, plus mid-quarter time (years) and quarter length (days)."""
    d = pd.to_datetime(dates)
    days = np.diff(d).astype("timedelta64[D]").astype(float)
    mid = d[:-1] + (d[1:] - d[:-1]) / 2
    return np.diff(cum) / days, np.asarray((mid - d[0]).days / 365.25), days


def arps(t, qi, D, b):
    # standard Arps, b near 0 is exponential and b = 1 is harmonic
    b = np.maximum(b, 1e-6)
    return qi * (1 + b * D * t) ** (-1 / b)


def fit_trend(method, window, dates, oil, water):
    """Fit the decline trends on the last `window` quarters.

    log(WOR) vs cumulative oil is roughly a straight line in a waterflood. Oil comes from either an
    Arps decline ('arps_wor') or an exponential liquid trend split with the WOR ('liquid_wor').
    Slopes come from the window but the starting level is the last quarter, otherwise the first
    forecast quarter jumped by ~7%.
    """
    dates = pd.to_datetime(dates)[-(window + 1):]
    oil, water = oil[-(window + 1):], water[-(window + 1):]
    qo, t, _ = avg_rates(oil, dates)
    qw, _, _ = avg_rates(water, dates)
    np_mid = (oil[:-1] + oil[1:]) / 2
    wor = np.clip(qw / qo, 1e-6, None)
    slope, _ = np.polyfit(np_mid, np.log(wor), 1)
    trend = {"method": method, "wor": (slope, np.log(wor[-1]) - slope * np_mid[-1]),
             "t0": dates[0], "end": dates[-1], "start": (oil[-1], water[-1])}
    if method == "arps_wor":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = curve_fit(arps, t, qo, p0=[qo[0], 0.2, 0.5], bounds=([0, 0, 0], [10 * qo.max(), 5, 1]), maxfev=20000)
        p[0] *= qo[-1] / arps(t[-1], *p)
        trend["oil"] = p
    else:
        beta, _ = np.polyfit(t, np.log(qo + qw), 1)
        beta = min(beta, 0.0)   # don't let total liquid go up
        trend["liquid"] = (np.log(qo[-1] + qw[-1]) - beta * t[-1], beta)
    return trend


def trend_rates(trend, t_years, np_mid):
    wor = np.exp(np.polyval(trend["wor"], np_mid))
    if trend["method"] == "arps_wor":
        qo = arps(t_years, *trend["oil"])
        return qo, wor * qo
    ql = np.exp(trend["liquid"][0] + trend["liquid"][1] * t_years)
    wc = wor / (1 + wor)
    return ql * (1 - wc), ql * wc


def extrapolate(method, window, dates, oil, water, future):
    """Step forward one quarter at a time, returns cumulative oil and water."""
    trend = fit_trend(method, window, dates, oil, water)
    d = pd.to_datetime(np.concatenate([[trend["end"]], pd.to_datetime(future)]))
    o, w = [trend["start"][0]], [trend["start"][1]]
    for a, b in zip(d[:-1], d[1:]):
        days = (b - a).days
        t_mid = ((a + (b - a) / 2) - trend["t0"]).days / 365.25
        np_mid = o[-1]
        for _ in range(2):   # use the WOR at mid-quarter, 2 passes is enough
            qo, qw = trend_rates(trend, t_mid, np_mid)
            np_mid = o[-1] + qo * days / 2
        o.append(o[-1] + qo * days)
        w.append(w[-1] + qw * days)
    return np.array(o[1:]), np.array(w[1:])


# --- back-test 1: which trend method
def backtest_methods(Y, dates):
    dates = pd.to_datetime(dates)
    rows = []
    for cut in BT_CUTS:
        n = int((dates <= cut).sum())
        for i in range(Y["oil_cum"].shape[0]):
            o, w = Y["oil_cum"][i], Y["water_cum"][i]
            for m, k in METHODS:
                po, pw = extrapolate(m, k, dates[:n], o[:n], w[:n], dates[n:])
                rows.append({"cut": cut[:4], "method": m, "window": k, "case": i,
                             "oil_err_%": 100 * (po[-1] - o[-1]) / (o[-1] - o[n - 1]),
                             "water_err_%": 100 * (pw[-1] - w[-1]) / (w[-1] - w[n - 1])})
    df = pd.DataFrame(rows)
    summary = (df.groupby(["cut", "method", "window"])
               .agg(oil_bias=("oil_err_%", "mean"), oil_mae=("oil_err_%", lambda s: s.abs().mean()),
                    water_bias=("water_err_%", "mean"), water_mae=("water_err_%", lambda s: s.abs().mean()))
               .reset_index())
    summary["score"] = (summary.oil_mae + summary.water_mae) / 2
    return summary


# --- back-test 2: how to connect the cases to the real field
def tie_in(model_o, model_w, obs_o, obs_w, n_ref, how, dates):
    """Start the model curves from the real values at index n_ref-1.

    cum_shift just adds the model's increments to the real totals. rate_ratio also scales the oil
    and water rates by real/model over the last two quarters. wor_shift keeps the model's liquid but
    moves its WOR to the real one.
    """
    i = n_ref - 1
    if how == "raw":
        return model_o[:, i:], model_w[:, i:]
    inc_o = model_o[:, i:] - model_o[:, [i]]
    inc_w = model_w[:, i:] - model_w[:, [i]]
    ref_o, ref_w = model_o[:, i] - model_o[:, i - 2], model_w[:, i] - model_w[:, i - 2]
    obs_ref_o, obs_ref_w = obs_o[i] - obs_o[i - 2], obs_w[i] - obs_w[i - 2]
    if how == "rate_ratio":
        inc_o, inc_w = inc_o * (obs_ref_o / ref_o)[:, None], inc_w * (obs_ref_w / ref_w)[:, None]
    elif how == "wor_shift":
        days = np.diff(pd.to_datetime(dates)).astype("timedelta64[D]").astype(float)[i:]
        qo = np.diff(model_o[:, i:], axis=1) / days
        qw = np.diff(model_w[:, i:], axis=1) / days
        wor = qw / qo * ((obs_ref_w / obs_ref_o) / (ref_w / ref_o))[:, None]
        ql = (qo + qw) * ((obs_ref_o + obs_ref_w) / (ref_o + ref_w))[:, None]
        qo_f = ql / (1 + wor)
        inc_o = np.hstack([np.zeros((len(qo), 1)), np.cumsum(qo_f * days, axis=1)])
        inc_w = np.hstack([np.zeros((len(qo), 1)), np.cumsum((ql - qo_f) * days, axis=1)])
    return obs_o[i] + inc_o, obs_w[i] + inc_w


def error_vs_observed(fo, fw, to, tw):
    # error at 2008 as % of what the field produced after the cut
    eo = 100 * (fo[:, -1] - to[-1]) / (to[-1] - to[0])
    ew = 100 * (fw[:, -1] - tw[-1]) / (tw[-1] - tw[0])
    return {"oil_err_%": eo.mean(), "water_err_%": ew.mean(), "score": (abs(eo.mean()) + abs(ew.mean())) / 2}


def backtest_tiein(Ydf, o, best_method):
    obs_o, obs_w = o["obs"]["oil_cum"], o["obs"]["water_cum"]
    rows, paths = [], {}
    for cut in SELECT_CUTS:
        n = int((pd.to_datetime(o["dates"]) <= f"{cut}-01-01").sum())
        # same selection as step 4 but only with history before the cut
        _, top = select_top_cases({q: Ydf[q].iloc[:, :n] for q in PHASES}, make_objective(o["dates"][:n]))
        c = {q: Ydf[q].loc[top.case].values for q in PHASES}
        truth_o, truth_w = obs_o[n - 1:], obs_w[n - 1:]
        paths[cut] = {"n": n}
        for how in ["raw", "cum_shift"] + TIE_INS:
            fo, fw = tie_in(c["oil_cum"], c["water_cum"], obs_o, obs_w, n, how, o["dates"])
            paths[cut][how] = fo
            rows.append({"cut": cut, "approach": how, **error_vs_observed(fo, fw, truth_o, truth_w)})
        # for comparison: extend the real data only, no model
        po, pw = extrapolate(*best_method, o["dates"][:n], obs_o[:n], obs_w[:n], pd.to_datetime(o["dates"])[n:])
        fo, fw = np.concatenate([[obs_o[n - 1]], po])[None], np.concatenate([[obs_w[n - 1]], pw])[None]
        paths[cut]["observed_only"] = fo
        rows.append({"cut": cut, "approach": "observed_only", **error_vs_observed(fo, fw, truth_o, truth_w)})
    return pd.DataFrame(rows), paths


# --- final forecast
def final_forecast(Ydf, top, o, method, tie, gor):
    curves = {p: Ydf[p].loc[top.case].values for p in PHASES}
    dates = pd.to_datetime(o["dates"])
    obs_o, obs_w = o["obs"]["oil_cum"], o["obs"]["water_cum"]
    n = len(dates)
    all_dates = list(dates) + list(FC_DATES)
    out = []
    for r in range(len(top)):
        po, pw = extrapolate(*method, dates, curves["oil_cum"][r], curves["water_cum"][r], FC_DATES)
        mo = np.concatenate([curves["oil_cum"][r], po])[None]
        mw = np.concatenate([curves["water_cum"][r], pw])[None]
        # padding is just so the arrays line up, tie_in only looks at values up to 2008
        fo, fw = tie_in(mo, mw, np.concatenate([obs_o, po]), np.concatenate([obs_w, pw]), n, tie, all_dates)
        fo, fw = fo[0, 1:], fw[0, 1:]
        out.append(pd.DataFrame({"rank": r + 1, "case": top.case.iloc[r], "date": FC_DATES, "oil_cum_stb": fo * 1e6,
                                 "water_cum_stb": fw * 1e6, "gas_cum_mscf": gor * fo * 1e6}))
    fc = pd.concat(out, ignore_index=True)
    # cumulative can't go down
    for col in ("oil_cum_stb", "water_cum_stb", "gas_cum_mscf"):
        assert (fc.groupby("rank")[col].diff().dropna() >= -1e-6).all(), f"{col} decreases"
    return fc


def add_rates(fc, obs_end):
    fc = fc.copy()
    prev_date = fc.groupby("rank").date.shift().fillna(pd.Timestamp("2008-01-01"))
    days = (fc.date - prev_date).dt.days
    for p, start in obs_end.items():
        prev = fc.groupby("rank")[f"{p}_cum_stb"].shift().fillna(start)
        fc[f"{p}_rate_stbd"] = (fc[f"{p}_cum_stb"] - prev) / days
    fc["gas_rate_mscfd"] = fc.oil_rate_stbd * (fc.gas_cum_mscf / fc.oil_cum_stb)
    fc["water_cut"] = fc.water_rate_stbd / (fc.oil_rate_stbd + fc.water_rate_stbd)
    return fc


# --- export
def fill_template(fc):
    """Put the k-th best case into sheet 'Case k', then open the file again and check it."""
    columns = {"Gas production cumulative [MSCF]": "gas_cum_mscf",
               "Oil production cumulative [STB]": "oil_cum_stb",
               "Water production cumulative [STB]": "water_cum_stb"}

    def parse_date(v):
        # careful: most dates are text like "04/01/2008" (month first) but the last one is a real date
        return pd.Timestamp(v) if isinstance(v, datetime) else pd.Timestamp(datetime.strptime(str(v).strip(), "%m/%d/%Y"))

    def layout(ws):
        header = [c.value for c in ws[1]]
        cols = {h: header.index(h) + 1 for h in columns}
        rows = [(r, parse_date(ws.cell(r, 1).value)) for r in range(2, ws.max_row + 1) if ws.cell(r, 1).value is not None]
        return cols, rows

    wb = openpyxl.load_workbook(RAW / "09 Template Deliverable.xlsx")
    for k in range(1, N_PICKS + 1):
        ws = wb[f"Case {k}"]
        cols, rows = layout(ws)
        assert [d for _, d in rows] == list(FC_DATES), f"Case {k}: unexpected template dates"
        pick = fc[fc["rank"] == k].set_index("date")
        for r, d in rows:
            for h, src in columns.items():
                ws.cell(r, cols[h]).value = round(float(pick.loc[d, src]), 1)
    path = DELIVERABLE / "09 Template Deliverable.xlsx"
    # remove the template author's name from the file properties
    wb.properties.creator = wb.properties.lastModifiedBy = None
    wb.save(path)

    # open it again and check
    obs_2008 = read_observed().iloc[-1]
    wb = openpyxl.load_workbook(path, data_only=True)
    for k in range(1, N_PICKS + 1):
        cols, rows = layout(wb[f"Case {k}"])
        vals = pd.DataFrame({src: [wb[f"Case {k}"].cell(r, cols[h]).value for r, _ in rows] for h, src in columns.items()})
        expected = fc[fc["rank"] == k][list(columns.values())].values
        assert len(vals) == 80 and not vals.isna().any().any(), f"Case {k}: missing values"
        assert (vals.diff().dropna() >= 0).all().all(), f"Case {k}: a cumulative decreases"
        assert vals.oil_cum_stb.iloc[0] >= obs_2008.oil_cum, f"Case {k}: does not continue from 2008"
        assert np.allclose(vals.values, expected, atol=0.1), f"Case {k}: file differs from the forecast"
    return path


def export(fc, top):
    DELIVERABLE.mkdir(parents=True, exist_ok=True)
    # rank = sheet number in the template, case = original case number
    fc.to_csv(DELIVERABLE / "forecast_2008_2028.csv", index=False, date_format="%Y-%m-%d")
    end = fc[fc.date == FC_DATES[-1]].set_index("rank")
    t = top[["case", "split", "J"]].join(
        end[["oil_cum_stb", "gas_cum_mscf", "water_cum_stb", "oil_rate_stbd", "water_cut"]].add_suffix("_2028"))
    t.insert(0, "sheet", [f"Case {k}" for k in t.index])
    t.rename(columns={"J": "history_mismatch_J"}).to_csv(DELIVERABLE / "selected_cases.csv")
    return fill_template(fc)


# --- plots
def plot_backtest(paths, o):
    d = pd.to_datetime(o["dates"])
    styles = {"raw": ("#9a9a9a", "model, raw"), "cum_shift": ("#2b5c8a", "model, cumulative shift"),
              "rate_ratio": ("#e08a2c", "model, rate ratio"), "wor_shift": ("#b4461f", "model, WOR shift"),
              "observed_only": ("#2f6b45", "observed data only")}
    fig, axes = plt.subplots(1, len(paths), figsize=(15, 4.6))
    for ax, (cut, pc) in zip(axes, paths.items()):
        n = pc["n"]
        for key, (c, label) in styles.items():
            for j, path in enumerate(pc[key]):
                ax.plot(d[n - 1:], path - o["obs"]["oil_cum"][n - 1:], color=c, lw=1.2, alpha=0.8,
                        label=label if j == 0 else None)
        ax.axhline(0, color="black", lw=1.2, label="observed (zero error)")
        ax.set_title(f"Oil: forecast minus observed [MMSTB], cases selected on data to {cut}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Back-test: forecasts from the selected cases vs what the field really produced", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "05_forecast_backtest.png", dpi=125)
    plt.close(fig)


def plot_forecast(fc, o):
    d = pd.to_datetime(o["dates"])
    days = np.diff(d).astype("timedelta64[D]").astype(float)
    gor = fc.gas_cum_mscf.iloc[0] / fc.oil_cum_stb.iloc[0]
    hist = {"oil": o["obs"]["oil_cum"] * 1e6, "water": o["obs"]["water_cum"] * 1e6}
    hist["gas"] = hist["oil"] * gor
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    cmap = plt.get_cmap("tab10")
    for col, (p, unit) in enumerate((("oil", "STB"), ("gas", "MSCF"), ("water", "STB"))):
        cum_col = f"{p}_cum_{'mscf' if p == 'gas' else 'stb'}"
        rate_col = f"{p}_rate_{'mscfd' if p == 'gas' else 'stbd'}"
        axes[0, col].plot(d[1:], np.diff(hist[p]) / days, "o", ms=3, color="black", label="observed")
        axes[1, col].plot(d, hist[p] / 1e6, "o", ms=3, color="black", label="observed")
        for r, g in fc.groupby("rank"):
            axes[0, col].plot(g.date, g[rate_col], color=cmap(r - 1), lw=1.1, label=f"#{r}: case {g.case.iloc[0]}")
            axes[1, col].plot(g.date, g[cum_col] / 1e6, color=cmap(r - 1), lw=1.1)
        axes[0, col].set_title(f"{p.capitalize()} rate [{unit}/d]", fontsize=10)
        axes[1, col].set_title(f"{p.capitalize()} cumulative [million {unit}]", fontsize=10)
        for ax in axes[:, col]:
            ax.axvline(pd.Timestamp("2008-01-01"), color="0.4", ls=":", lw=1)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Forecast 2008-2028 for the 10 best-matching provided cases (dotted line = end of history)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "06_forecast_2008_2028.png", dpi=120)
    plt.close(fig)


def plot_workflow_summary(fc, top):
    """Top: all 100 cases vs the real field. Bottom: our 10 cases and their forecast."""
    q, obs = read_quarterly(), read_observed()
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), sharex=True)
    cmap = plt.get_cmap("tab10")
    for col, (p, unit) in enumerate((("gas_cum", "MSCF"), ("oil_cum", "STB"), ("water_cum", "STB"))):
        name = p.split("_")[0].capitalize()
        fc_col = "gas_cum_mscf" if p == "gas_cum" else f"{p}_stb"
        ax = axes[0, col]
        for _, g in q.groupby("case"):
            ax.plot(g.date, g[p] / 1e6, color="0.75", lw=0.6)
        ax.plot(obs.date, obs[p] / 1e6, "o", ms=3, color="black", label="observed (RAMP_History)")
        ax.set_title(f"{name} cumulative [million {unit}]: all 100 provided cases", fontsize=10)
        ax = axes[1, col]
        for i, case in enumerate(top.case):
            g = q[q.case == case]
            ax.plot(g.date, g[p] / 1e6, color=cmap(i), lw=1.1, label=f"#{i + 1}: case {case}")
            f = fc[fc["rank"] == i + 1]
            ax.plot(f.date, f[fc_col] / 1e6, color=cmap(i), lw=1.1, ls="--")
        ax.plot(obs.date, obs[p] / 1e6, "o", ms=3, color="black")
        ax.set_title(f"{name}: 10 best history matches (solid) and their 2008-2028 forecast (dashed)", fontsize=10)
        for a in axes[:, col]:
            a.axvline(pd.Timestamp("2008-01-01"), color="0.3", ls="-.", lw=1.2)
            a.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("History matching and 20-year forecast: provided cases -> 10 best matches -> forecast to 2028 "
                 "(dash-dot line = end of history, 2008)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "07_history_match_and_forecast_summary.png", dpi=120)
    plt.close(fig)


def main():
    check_data()
    warnings.filterwarnings("ignore")
    for d in (FIGS, RESULTS, DELIVERABLE):
        d.mkdir(parents=True, exist_ok=True)
    _, Ydf, _, info = load_features()
    dates, gor = info["dates"], info["gor"]
    Y = {p: Ydf[p].values for p in PHASES}
    o = make_objective(dates)

    bt1 = backtest_methods(Y, dates)
    bt1.to_csv(RESULTS / "backtest_methods.csv", index=False)
    sel1 = bt1[bt1.cut.isin(SELECT_CUTS)].groupby(["method", "window"]).score.mean().sort_values()
    best_method = (sel1.index[0][0], int(sel1.index[0][1]))

    bt2, paths = backtest_tiein(Ydf, o, best_method)
    bt2.to_csv(RESULTS / "backtest_tiein.csv", index=False)
    tie = bt2[bt2.approach.isin(TIE_INS)].groupby("approach").score.mean().idxmin()

    top = pd.read_csv(RESULTS / "selected_top10_cases.csv", index_col="rank")
    fc = final_forecast(Ydf, top, o, best_method, tie, gor)
    fc = add_rates(fc, {"oil": o["obs"]["oil_cum"][-1] * 1e6, "water": o["obs"]["water_cum"][-1] * 1e6})
    # oil is declining in 2008, so the first forecast quarter shouldn't be higher than the last real one
    last_obs_rate = 1e6 * (o["obs"]["oil_cum"][-1] - o["obs"]["oil_cum"][-2]) / \
        (pd.to_datetime(o["dates"][-1]) - pd.to_datetime(o["dates"][-2])).days
    first_rate = fc[fc.date == FC_DATES[0]].oil_rate_stbd
    assert (first_rate <= last_obs_rate).all(), "forecast oil rate jumps above the last observed rate"

    # 2028 oil with every method, shows how much the method choice matters
    sens = []
    for m, k in METHODS:
        for how in TIE_INS:
            f = final_forecast(Ydf, top, o, (m, k), how, gor)
            sens.append({"method": m, "window": k, "tie_in": how,
                         "oil_2028_mmstb_median": f[f.date == FC_DATES[-1]].oil_cum_stb.median() / 1e6})
    sens = pd.DataFrame(sens)
    sens.to_csv(RESULTS / "method_sensitivity.csv", index=False)

    path = export(fc, top)
    plot_backtest(paths, o)
    plot_forecast(fc, o)
    plot_workflow_summary(fc, top)

    end = fc[fc.date == FC_DATES[-1]]
    print("Step 5 done")
    print("  back-test 1, mean score over 2005/2006 cuts:\n", sel1.round(2).to_string())
    print(f"  -> extrapolation: {best_method}")
    print("  back-test 2:\n", bt2.round(2).to_string(index=False))
    print(f"  -> tie-in: {tie}")
    print(f"  2028 oil {end.oil_cum_stb.min() / 1e6:.2f}-{end.oil_cum_stb.max() / 1e6:.2f} MMSTB, "
          f"gas {end.gas_cum_mscf.min() / 1e6:.2f}-{end.gas_cum_mscf.max() / 1e6:.2f} MM MSCF, "
          f"water {end.water_cum_stb.min() / 1e6:.2f}-{end.water_cum_stb.max() / 1e6:.2f} MMSTB")
    print(f"  method band for 2028 oil: {sens.oil_2028_mmstb_median.min():.1f}-{sens.oil_2028_mmstb_median.max():.1f} MMSTB")
    print(f"  template filled and checked: {path.relative_to(OUTPUTS.parent)}")


if __name__ == "__main__":
    main()
