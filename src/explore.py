"""Extra checks we did after step 1 (not needed for the main run).

They showed that porosity explains almost all (~98-99%) of the difference between cases, and
that the grid files don't add anything new.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step1_data import OUTPUTS, PARAMS, RAW, read_cases, read_observed, read_quarterly

OUT = OUTPUTS / "explore"


def plot_overview(q, obs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, col, label in zip(axes.flat, ["oil_rate", "water_rate", "oil_cum", "water_cum"],
                              ["Oil rate [STB/d]", "Water rate [STB/d]", "Oil cumulative [STB]", "Water cumulative [STB]"]):
        for _, g in q.groupby("case"):
            ax.plot(g.date, g[col], color="0.75", lw=0.6)
        ax.plot(obs.date, obs[col], "o", ms=3, color="black", label="observed")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("100 simulated cases (grey) vs the observed field (black), quarterly 1998-2008")
    fig.tight_layout()
    fig.savefig(OUT / "overview_cases_vs_observed.png", dpi=120)
    plt.close(fig)


def sensitivity(cases, q):
    last = q[q.date == q.date.max()].set_index("case")
    m = cases.join(pd.DataFrame({"oil_2008": last.oil_cum, "water_2008": last.water_cum,
                                 "watercut_2008": last.water_rate / (last.oil_rate + last.water_rate)}))
    outcomes = ["oil_2008", "water_2008", "watercut_2008"]
    # spearman so it also catches non-linear trends
    rho = m[PARAMS + outcomes].corr(method="spearman").loc[PARAMS, outcomes]

    # take the porosity effect out first, otherwise it hides everything else
    A = np.column_stack([np.ones(len(m)), m.poro_mult, m.poro_mult ** 2])
    rows = []
    for k in outcomes:
        y = m[k].values
        resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        row = {"outcome": k, "r2_porosity_only": 1 - resid.var() / y.var()}
        for p in ["perm_mult", "aquifer_pv", "fault_trans"]:
            row[f"left_over_vs_{p}"] = pd.Series(resid).corr(pd.Series(m[p].values), method="spearman")
        rows.append(row)
    return rho, pd.DataFrame(rows).set_index("outcome")


def grid_check():
    """Compare case 1 and case 2 cell by cell."""
    sheets = pd.read_excel(RAW / "03 Train Cases 1.xlsx", sheet_name=["Case 1", "Case 2"], engine="calamine")
    a, b = sheets["Case 1"], sheets["Case 2"]
    a.columns = b.columns = [c.split("(")[-1].rstrip(")").strip() if "(" in c else c.split()[0] for c in a.columns]
    active = a["PORV"] > 0   # PORV 0 (or -999.25) means the cell is inactive
    props = [c for c in a.columns if c not in ("I", "J", "K", "X", "Y", "Z")]
    same = [c for c in props if np.allclose(a.loc[active, c], b.loc[active, c], rtol=1e-4)]
    perm_ratio = b.loc[active, "PERMX"] / a.loc[active, "PERMX"]
    cases = read_cases()
    return {
        "active cells": int(active.sum()),
        "properties identical in cases 1 and 2": f"{len(same)} of {len(props)}",
        "properties that differ": [c for c in props if c not in same],
        "porosity ratio (case2/case1) vs multiplier ratio": (
            round(float((b.loc[active, "PORO"] / a.loc[active, "PORO"]).median()), 4),
            round(float(cases.loc[2, "poro_mult"] / cases.loc[1, "poro_mult"]), 4)),
        "share of cells whose permeability is scaled": round(float(((perm_ratio - 1).abs() > 0.01).mean()), 3),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases, q, obs = read_cases(), read_quarterly(), read_observed()
    plot_overview(q, obs)
    rho, adj = sensitivity(cases, q)
    rho.to_csv(OUT / "sensitivity.csv")
    adj.to_csv(OUT / "after_porosity.csv")
    pd.set_option("display.width", 200)
    print("Rank correlation, parameter vs 2008 outcome:\n", rho.round(2).to_string())
    print("\nAfter removing the porosity effect:\n", adj.round(3).to_string())
    print("\nGrid check (cases 1 vs 2):")
    # file is too big for github (>100 MB), so skip if it's not there
    if not (RAW / "03 Train Cases 1.xlsx").exists():
        print("  skipped: copy '03 Train Cases 1.xlsx' from the challenge data into TestCases/ to run it")
        return
    for k, v in grid_check().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
