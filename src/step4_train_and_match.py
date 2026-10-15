"""Step 4: choose the proxy model, then history match.

Each model is tuned with 5-fold CV on train, picked on validation, and only then scored on test.
For history matching we score all 100 cases against RAMP_History (mismatch J) and keep the 10
lowest, like the FAQ says (Q8-Q10). After that we run the proxy on 100k random parameter sets to
see what range of values actually fits the history.
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold

from step1_data import MODELS, OUTPUTS, PARAM_LABELS, PARAMS, read_cases, read_observed
from step2_features import INPUTS, PHASES, TARGET_START, decode, load_features, quarterly_rates
from step3_model import CANDIDATES, SEED, with_scaled_targets

FIGS, RESULTS = OUTPUTS / "figures", OUTPUTS / "results"

HM_SEED = 7
N_SAMPLES = 100_000
TOP_FRAC = 0.01          # we count the best 1% as "fits the history"
N_PICKS = 10
LATE_FROM, LATE_WEIGHT = "2005-01-01", 2.0   # 2005-2008 counts double since it's closest to the forecast
# fault doesn't change production at all, so we don't cluster on it
CLUSTER_ON = ["poro_mult", "log_perm_mult", "aquifer_pv"]
# only use points where the GP uncertainty is under ~2x its test error. without this the
# optimiser kept running to empty corners where the GP is just unsure
TRUST_STD = {"oil": 0.08, "water": 0.18}   # % of the 2008 cumulative


# --- part A: train and pick the proxy
def predict_curves(model, X, pcas):
    return decode(pcas, model.predict(X))


def porosity_trend(poro, Y):
    """Quadratic fit of the 2008 totals against porosity. Porosity explains ~99%, what's left over
    is the perm/aquifer effect, which is the hard part for the proxy."""
    A = np.column_stack([np.ones_like(poro), poro, poro ** 2])
    return {p: np.linalg.lstsq(A, Y[p][:, -1], rcond=None)[0] for p in PHASES}


def after_porosity(coef, poro, Y):
    A = np.column_stack([np.ones_like(poro), poro, poro ** 2])
    return {p: Y[p][:, -1] - A @ coef[p] for p in PHASES}


def curve_metrics(Y_true, Y_pred, dates, poro=None, trend=None):
    m = {}
    for p in PHASES:
        name = p.split("_")[0]
        t, h = Y_true[p], Y_pred[p]
        final = np.abs(t[:, -1])
        m[f"{name}_curve_err_%"] = 100 * np.mean(np.sqrt(((h - t) ** 2).mean(axis=1)) / final)
        m[f"{name}_2008_err_%"] = 100 * np.mean(np.abs(h[:, -1] - t[:, -1]) / final)
        m[f"{name}_worst_case_%"] = 100 * np.max(np.abs(h - t).max(axis=1) / final)
        m[f"{name}_rate_rmse_stbd"] = 1e6 * np.sqrt(((quarterly_rates(h, dates) - quarterly_rates(t, dates)) ** 2).mean())
        if trend is not None:
            # R^2 on what's left after porosity
            rt, rh = after_porosity(trend, poro, Y_true)[p], after_porosity(trend, poro, Y_pred)[p]
            m[f"{name}_2nd_order_R2"] = 1 - ((rh - rt) ** 2).sum() / ((rt - rt.mean()) ** 2).sum()
    m["score"] = (m["oil_curve_err_%"] + m["water_curve_err_%"]) / 2
    return m


def cv_score(model, X, S, Y, pcas, dates, folds=5):
    errs = []
    for tr, te in KFold(folds, shuffle=True, random_state=SEED).split(X):
        fitted = with_scaled_targets(model).fit(X[tr], S[tr])
        pred = predict_curves(fitted, X[te], pcas)
        errs.append(curve_metrics({p: Y[p][te] for p in PHASES}, pred, dates)["score"])
    return float(np.mean(errs))


def train_proxy():
    Xdf, Ydf, scores, info = load_features()
    pcas, dates = info["pcas"], info["dates"]
    X, S = Xdf[INPUTS].values, scores.values
    Y = {p: Ydf[p].values for p in PHASES}
    idx = {s: (Xdf.split == s).values for s in ("train", "validation", "test")}
    poro = Xdf.raw_poro_mult.values
    tr = idx["train"]
    trend = porosity_trend(poro[tr], {p: Y[p][tr] for p in PHASES})

    # 1. tune with CV on train
    best = {}
    for family, settings in CANDIDATES.items():
        cv = {name: cv_score(m, X[tr], S[tr], {p: Y[p][tr] for p in PHASES}, pcas, dates) for name, m in settings.items()}
        best[family] = min(cv, key=cv.get)
        print(f"  {family:18s} best setting {best[family]:12s} CV score {cv[best[family]]:.4f}%")

    # 2-3. fit on train, score on validation and test
    rows, fitted = [], {}
    for family, name in best.items():
        model = with_scaled_targets(CANDIDATES[family][name]).fit(X[tr], S[tr])
        fitted[family] = model
        for s in ("validation", "test"):
            m = curve_metrics({p: Y[p][idx[s]] for p in PHASES}, predict_curves(model, X[idx[s]], pcas), dates,
                              poro[idx[s]], trend)
            rows.append({"family": family, "setting": name, "split": s, **m})
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "model_comparison.csv", index=False)
    val = results[results.split == "validation"].set_index("family")
    winner = val.score.idxmin()   # pick on validation, test is only for reporting

    # 4. refit on all 100. the train-only model is kept to check unseen cases later
    final = with_scaled_targets(CANDIDATES[winner][best[winner]]).fit(X, S)
    joblib.dump(final, MODELS / "proxy_gp.joblib")
    joblib.dump(fitted[winner], MODELS / "proxy_gp_train_only.joblib")

    # short length-scale means the input matters. the fault's goes to the upper bound
    gp = fitted["Gaussian Process"].regressor_.estimators_
    pd.DataFrame([dict(zip(INPUTS, np.atleast_1d(e.kernel_.k1.k2.length_scale))) for e in gp],
                 index=scores.columns).to_csv(RESULTS / "gp_lengthscales.csv")

    plot_model_comparison(results, winner)
    plot_test_cases(fitted[winner], X, Y, idx, pcas, dates, Xdf.index, winner)
    cols = ["oil_2008_err_%", "water_2008_err_%", "oil_rate_rmse_stbd", "water_rate_rmse_stbd",
            "oil_2nd_order_R2", "water_2nd_order_R2", "score"]
    print("  TEST (scored once, after choosing):\n",
          results[results.split == "test"].set_index("family")[cols].round(3).to_string())
    print(f"  winner on validation: {winner} ({best[winner]})")


# --- part B: history matching
def make_objective(dates):
    obs = read_observed()
    obs = obs[obs.date >= TARGET_START].set_index("date").loc[pd.to_datetime(dates)]
    o = {"dates": dates, "obs": {p: obs[p].values / 1e6 for p in PHASES}}
    o["obs_rate"] = {p: quarterly_rates(o["obs"][p][None], dates)[0] for p in PHASES}
    o["w"] = np.where(pd.to_datetime(dates) >= LATE_FROM, LATE_WEIGHT, 1.0)
    # divide by the max so oil, water, totals and rates all count the same
    o["scale"] = {p: np.abs(o["obs"][p]).max() for p in PHASES}
    o["scale_rate"] = {p: np.abs(o["obs_rate"][p]).max() for p in PHASES}
    return o


def mismatch(curves, o):
    """Mismatch J, lower is better. Cumulatives check the volume, rates check the shape of the decline."""
    out, w, w_rate = {}, o["w"], o["w"][1:]
    for p in PHASES:
        name = p.split("_")[0]
        e = (curves[p] - o["obs"][p]) / o["scale"][p]
        out[f"{name}_cum"] = np.sqrt((w * e ** 2).sum(1) / w.sum())
        er = (quarterly_rates(curves[p], o["dates"]) - o["obs_rate"][p]) / o["scale_rate"][p]
        out[f"{name}_rate"] = np.sqrt((w_rate * er ** 2).sum(1) / w_rate.sum())
    df = pd.DataFrame(out)
    df["J"] = df.mean(axis=1)
    return df


def unit_to_params(U, bounds):
    """Convert from the 0-1 search box back to real parameter values."""
    lo, hi = pd.Series(bounds["lo"])[INPUTS].values, pd.Series(bounds["hi"])[INPUTS].values
    raw = pd.DataFrame(lo + U * (hi - lo), columns=INPUTS)
    raw["perm_mult"] = np.exp(raw.pop("log_perm_mult"))
    return raw[PARAMS]


def gp_uncertainty(model, pcas, U):
    """GP uncertainty on the 2008 totals in %, pushed through the PCA loadings at the last date."""
    stds = np.column_stack([est.predict(U, return_std=True)[1] for est in model.regressor_.estimators_])
    stds = stds * model.transformer_.scale_
    curves = decode(pcas, model.predict(U))
    out, i = {}, 0
    for p in PHASES:
        k = pcas[p].n_components_
        sd = np.sqrt((stds[:, i:i + k] ** 2 * pcas[p].components_[:, -1] ** 2).sum(1))
        out[f"{p.split('_')[0]}_2008_std_%"] = 100 * sd / np.abs(curves[p][:, -1])
        i += k
    return pd.DataFrame(out)


def select_top_cases(Ydf, o):
    """Score every provided case on its own simulator curves and keep the 10 with the lowest J."""
    true_J = mismatch({p: Ydf[p].values for p in PHASES}, o).set_index(Ydf[PHASES[0]].index)
    ranking = read_cases()[["split"]].join(true_J).sort_values("J", kind="stable")
    top = ranking.head(N_PICKS).reset_index()
    top.index = pd.RangeIndex(1, N_PICKS + 1, name="rank")
    return ranking, top


def history_match():
    Xdf, Ydf, _, info = load_features()
    pcas, dates, bounds = info["pcas"], info["dates"], info["bounds"]
    o = make_objective(dates)

    # this is our actual answer
    ranking, top = select_top_cases(Ydf, o)
    ranking.to_csv(RESULTS / "original_cases_ranking.csv")
    top.to_csv(RESULTS / "selected_top10_cases.csv")

    # proxy part: which parameter values fit the history?
    model = joblib.load(MODELS / "proxy_gp.joblib")

    def curves_at(U):
        return decode(pcas, model.predict(U))

    # latin hypercube to cover the whole parameter space evenly
    U = qmc.LatinHypercube(d=len(INPUTS), seed=HM_SEED).random(N_SAMPLES)
    J = pd.concat([mismatch(curves_at(U[i:i + 20_000]), o) for i in range(0, N_SAMPLES, 20_000)], ignore_index=True)

    # best single point, just to compare. note vectorized=True gives candidates as columns
    de = differential_evolution(lambda u: mismatch(curves_at(u.reshape(len(INPUTS), -1).T), o)["J"].values,
                                bounds=[(0, 1)] * len(INPUTS), vectorized=True, updating="deferred",
                                popsize=40, maxiter=400, tol=1e-10, seed=HM_SEED, polish=False)

    cloud_U = U[J.J.rank(method="first") <= TOP_FRAC * N_SAMPLES]
    cloud = pd.concat([unit_to_params(cloud_U, bounds), mismatch(curves_at(cloud_U), o),
                       gp_uncertainty(model, pcas, cloud_U)], axis=1)
    cloud["trusted"] = ((cloud["oil_2008_std_%"] <= TRUST_STD["oil"])
                        & (cloud["water_2008_std_%"] <= TRUST_STD["water"]))
    cloud_U = pd.DataFrame(cloud_U, columns=INPUTS)

    # best set in each cluster, to see if something between the 100 cases fits better
    trusted = cloud[cloud.trusted]
    km = KMeans(N_PICKS, n_init=20, random_state=HM_SEED).fit(cloud_U.loc[trusted.index, CLUSTER_ON].values)
    cloud.loc[trusted.index, "cluster"] = km.labels_
    order = cloud.loc[trusted.index].groupby("cluster").J.idxmin().sort_values(key=lambda s: cloud.loc[s, "J"]).values
    proxy_sets = cloud.loc[order].reset_index(drop=True)
    proxy_sets.index = pd.RangeIndex(1, N_PICKS + 1, name="rank")
    proxy_sets.to_csv(RESULTS / "proxy_optimised_sets.csv")

    # sanity check: train-only proxy vs simulator on the 30 cases it never saw
    model_tr = joblib.load(MODELS / "proxy_gp_train_only.joblib")
    held = Xdf.split.isin(["validation", "test"])
    J_proxy = mismatch(decode(pcas, model_tr.predict(Xdf.loc[held, INPUTS].values)), o).J.values
    check = pd.DataFrame({"sim": ranking.loc[Xdf.index[held], "J"].values, "proxy": J_proxy})

    edge = ", ".join(n for n, v in zip(INPUTS, de.x) if v < 0.02 or v > 0.98)
    plot_history_match(Ydf, top, o)
    plot_parameter_constraints(cloud, read_cases().loc[top.case], unit_to_params(U, bounds))

    print("  SELECTED: 10 best of the 100 provided cases by history match (J on their own simulator curves)")
    print(top[["case", "split", "J"]].round(5).to_string())
    print(f"  supporting analysis: best proxy-optimised set J {proxy_sets.J.min():.4f} vs best provided case "
          f"J {top.J.iloc[0]:.4f}; unconstrained optimum J {de.fun:.4f} at the edge of: {edge}")
    print(f"  proxy vs simulator J on 30 unseen cases: rank correlation {check.corr(method='spearman').iloc[0, 1]:.3f}, "
          f"mean difference {(check.proxy - check.sim).abs().mean():.5f}")
    print("  good-fit parameter ranges (proxy, best 1%):\n", cloud[PARAMS].describe().loc[["min", "max"]].round(3).to_string())


# --- plots
def plot_model_comparison(results, winner):
    fams = list(dict.fromkeys(results.family))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    panels = [("oil_curve_err_%", "Oil curve error [% of 2008 cum]"),
              ("water_curve_err_%", "Water curve error [% of 2008 cum]"),
              ("oil_2nd_order_R2", "Second-order skill R² (oil, after porosity)")]
    for ax, (col, title) in zip(axes, panels):
        for k, (s, c) in enumerate((("validation", "#2b5c8a"), ("test", "#e08a2c"))):
            v = results[results.split == s].set_index("family").loc[fams, col]
            if col.endswith("R2"):
                v = v.clip(lower=-0.2)
            ax.barh(np.arange(len(fams)) + (0.2 if k else -0.2), v, height=0.38, color=c, label=s)
        ax.set_yticks(range(len(fams)), [f"{f}  (chosen)" if f == winner else f for f in fams])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="x")
    axes[0].legend()
    axes[2].set_xlim(-0.2, 1)
    fig.suptitle("Proxy model comparison (lower error / higher R² is better; chosen on validation)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "01_proxy_model_comparison.png", dpi=130)
    plt.close(fig)


def plot_test_cases(model, X, Y, idx, pcas, dates, case_ids, winner):
    te = np.where(idx["test"])[0]
    pred = predict_curves(model, X[te], pcas)
    err = np.abs(pred["water_cum"][:, -1] - Y["water_cum"][te, -1]) / Y["water_cum"][te, -1]
    order = np.argsort(err)
    d = pd.to_datetime(dates)
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for row, (j, label) in enumerate(zip([order[0], order[len(order) // 2], order[-1]], ["best", "median", "worst"])):
        for col, key in enumerate(("oil_cum", "water_cum", "rates")):
            ax = axes[row, col]
            if key == "rates":
                for p, c in (("oil_cum", "#2b5c8a"), ("water_cum", "#2f6b45")):
                    ax.plot(d[1:], 1e6 * quarterly_rates(Y[p][te[j]][None], dates)[0], color=c, lw=2,
                            label=f"{p.split('_')[0]} simulator")
                    ax.plot(d[1:], 1e6 * quarterly_rates(pred[p][j][None], dates)[0], "--", color=c, lw=1.6,
                            label=f"{p.split('_')[0]} proxy")
                title = "Rates [STB/d]"
            else:
                ax.plot(d, Y[key][te[j]], color="black", lw=2, label="simulator")
                ax.plot(d, pred[key][j], "--", color="#b4461f", lw=1.8, label="proxy")
                title = f"{key.split('_')[0].capitalize()} cumulative [MMSTB]"
            ax.set_title(f"Test case {case_ids[te[j]]} ({label}) · {title}", fontsize=9.5)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7.5)
    fig.suptitle(f"{winner}: proxy vs simulator on unseen test cases", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGS / "02_proxy_vs_simulator_test_cases.png", dpi=115)
    plt.close(fig)


def plot_history_match(Ydf, top, o):
    d = pd.to_datetime(o["dates"])
    curves = {p: Ydf[p].loc[top.case].values for p in PHASES}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    cmap = plt.get_cmap("tab10")
    for ax, (p, label, rate) in zip(axes.flat, [("oil_cum", "Oil cumulative [MMSTB]", False),
                                                 ("water_cum", "Water cumulative [MMSTB]", False),
                                                 ("oil_cum", "Oil rate [STB/d]", True),
                                                 ("water_cum", "Water rate [STB/d]", True)]):
        x = d[1:] if rate else d
        f = (lambda c: 1e6 * quarterly_rates(c, o["dates"])) if rate else (lambda c: c)
        sims = f(curves[p])
        first = p == "oil_cum" and not rate
        for i, case in enumerate(top.case):
            ax.plot(x, sims[i], color=cmap(i), lw=1.3, label=f"#{i + 1}: case {case}" if first else None)
        ax.plot(x, f(o["obs"][p][None])[0], "o", color="black", ms=3.5, label="observed" if first else None)
        ax.axvspan(pd.Timestamp(LATE_FROM), d[-1], color="#f6e4dc", alpha=0.5, zorder=0)
        ax.set_title(label)
        ax.grid(alpha=0.3)
        if rate:  # only show 2002 onwards so the differences are visible
            ax.set_xlim(pd.Timestamp("2002-01-01"), d[-1] + pd.Timedelta(days=60))
            late = sims[:, x >= pd.Timestamp("2002-01-01")]
            ax.set_ylim(late.min() * 0.9, late.max() * 1.08)
    axes[0, 0].legend(fontsize=7.5, ncol=2)
    fig.suptitle("History match: the 10 best of the 100 provided cases (own simulator curves) vs observed field. "
                 "Shaded = double-weighted last 3 years", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "03_history_match.png", dpi=120)
    plt.close(fig)


def plot_parameter_constraints(cloud, selected, params):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    for ax, p in zip(axes, PARAMS):
        bins = np.geomspace(params[p].min(), params[p].max(), 30) if p == "perm_mult" else 30
        ax.hist(params[p], bins=bins, density=True, color="0.82", label="all sampled")
        ax.hist(cloud[p], bins=bins, density=True, color="#b4461f", alpha=0.75, label="good fits (best 1%)")
        for v in selected[p]:
            ax.axvline(v, color="black", lw=0.8, alpha=0.7)
        if p == "perm_mult":
            ax.set_xscale("log")
        ax.set_title(PARAM_LABELS[p], fontsize=10)
        ax.set_yticks([])
    axes[0].legend(fontsize=8)
    fig.suptitle("What the history tells us about each parameter: narrow = constrained, flat = unknown. "
                 "Black lines = the 10 selected cases", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "04_parameter_constraints.png", dpi=120)
    plt.close(fig)


def main():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for d in (FIGS, RESULTS, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    print("Step 4A: training and model selection")
    train_proxy()
    print("Step 4B: history matching")
    history_match()


if __name__ == "__main__":
    main()
