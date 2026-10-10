"""v2 manuscript and appendix figures. PDF + 300 DPI PNG. No caption-duplicating titles."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynafront.figures import savefig
from dynafront import certificates as ora

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]


def fig01_clipped_tanh_phase(tanh: pd.DataFrame, dest: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    ax, ax2 = axes
    for i, (rho, g) in enumerate(tanh.groupby("rho")):
        g = g.sort_values("ratio")
        ax.plot(g["ratio"], g["mean_amp_norm"], marker="o", ms=3.5, color=COLORS[i % len(COLORS)], label=fr"tanh $\rho={rho:g}$")
        if "clipped_mean_amp_norm" in g:
            ax.plot(
                g["ratio"],
                g["clipped_mean_amp_norm"],
                marker="s",
                ms=3,
                ls="--",
                color=COLORS[i % len(COLORS)],
                alpha=0.7,
                label=fr"clip $\rho={rho:g}$" if i == 0 else None,
            )
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel(r"$\tau/\tau_c$")
    ax.set_ylabel(r"normalized amplitude")
    ax.legend(frameon=False, ncol=3, fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0)
    # onset panel: subcritical near threshold
    sub = tanh[(tanh["ratio"] < 1.0) & (tanh["r_tau"] > 0)].copy()
    if not sub.empty:
        x = np.sqrt(np.maximum(3.0 * (1.0 - sub["ratio"]), 0.0))
        y = sub["r_tau"] / sub["tau_c"]
        ax2.scatter(x, y, c=[COLORS[int(np.digitize(r, [0.2, 0.5, 0.8, 0.95])) % len(COLORS)] for r in sub["rho"]], s=18)
        xx = np.linspace(0, float(np.max(x)) if len(x) else 1, 50)
        ax2.plot(xx, xx, color="k", lw=1, ls="--", label=r"$\sqrt{3(1-\tau/\tau_c)}$")
        ax2.set_xlabel(r"$\sqrt{3(1-\tau/\tau_c)}$")
        ax2.set_ylabel(r"$r_\tau/\tau_c$")
        ax2.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    savefig(fig, dest)


def fig02_two_resolvent_quadrants(df: pd.DataFrame, dest: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    labels = []
    fix_ag, pair_ag, nacc, err = [], [], [], []
    for (sf, sc), g in df.groupby(["sign_fix", "sign_cyc"]):
        labels.append(fr"$({sf:+d},{sc:+d})$")
        fix_ag.append(float(g["fix_compat_agree"].min()))
        pair_ag.append(float(g["pair_compat_agree"].min()))
        nacc.append(int(g["n_accepted"].sum()))
        err.append(float(g["max_rel_fix_err"].max()))
    x = np.arange(len(labels))
    axes[0].bar(x - 0.2, fix_ag, 0.4, label="fixed-point compatibility", color=COLORS[0])
    axes[0].bar(x + 0.2, pair_ag, 0.4, label="alternating-pair compatibility", color=COLORS[1])
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_xlabel(r"$(\chi_{\mathrm{fix}},\chi_{\mathrm{cyc}})$")
    axes[0].set_ylim(0.95, 1.005)
    axes[0].set_ylabel("oracle agreement")
    axes[0].legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    axes[1].bar(x, err, color=COLORS[2])
    axes[1].axhline(1e-9, color="k", ls="--", lw=1)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, labels, fontsize=8)
    n_each = nacc[0] if nacc and len(set(nacc)) == 1 else None
    if n_each is not None:
        axes[1].set_xlabel(rf"$(\chi_{{\mathrm{{fix}}}},\chi_{{\mathrm{{cyc}}}})$; {n_each} accepted systems each, pooled over $d$")
    else:
        axes[1].set_xlabel(r"$(\chi_{\mathrm{fix}},\chi_{\mathrm{cyc}})$; accepted systems pooled over $d$")
        for xi, n in zip(x, nacc):
            axes[1].text(xi, err[int(xi)] * 1.4 if err[int(xi)] > 0 else 1e-12, str(int(n)), ha="center", fontsize=7)
    axes[1].set_ylabel("max relative frozen-state error")
    fig.tight_layout()
    savefig(fig, dest)


def fig03_itinerary_certificates(it: pd.DataFrame, c: pd.DataFrame, dest: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    cov = it[it["sample"] == "coverage"]
    un = it[it["sample"] == "unconditional"]
    periods = np.arange(1, 7)
    found = [int(cov[f"n_period_{p}"].iloc[0]) if not cov.empty and f"n_period_{p}" in cov else 0 for p in periods]
    axes[0].bar(periods, found, color=COLORS[0])
    axes[0].axhline(25, color="k", ls="--", lw=1)
    axes[0].set_xlabel("certified minimal period")
    axes[0].set_ylabel("coverage-sample systems")
    if not c.empty:
        order = ["crossed_sign_compatible", "crossed_no_pair", "noncrossed"]
        labs = ["stratum 1\nsign-compatible pair", "stratum 2\ncrossed, no pair", "stratum 3\nnoncrossed"]
        x = np.arange(3)
        w = 0.25
        for i, (col, lab) in enumerate(
            [("frac_period_two", "period 2"), ("frac_period_one", "period 1"), ("frac_other_period", "other finite")]
        ):
            vals = [float(c.loc[c["stratum"] == s, col].mean()) if col in c else np.nan for s in order]
            axes[1].bar(x + (i - 1) * w, vals, w, label=lab, color=COLORS[i])
        axes[1].set_xticks(x, labs, fontsize=6.5)
        axes[1].set_ylabel("mean IC fraction")
        axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    savefig(fig, dest)


def fig04_deadband_hysteresis(hys: pd.DataFrame, dest: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=False)
    col = "kind" if "kind" in hys.columns else "family"
    db = hys[hys[col] == "deadband"] if col in hys.columns else hys.iloc[0:0]
    hy = hys[hys[col] == "hysteresis"] if col in hys.columns else hys
    for i, (rho, g) in enumerate(db.groupby("rho")):
        g = g.sort_values("ratio")
        axes[0].plot(g["ratio"], g["frac_zero"], color=COLORS[i % len(COLORS)], marker="o", ms=3, label=fr"zero $\rho={rho:g}$")
        axes[0].plot(g["ratio"], g["frac_cycle"], color=COLORS[i % len(COLORS)], marker="s", ms=3, ls="--", alpha=0.8)
    axes[0].axvline(1.0, color="k", ls="--", lw=1)
    axes[0].set_xlabel(r"$h/\tau_c$ (neutral deadband)")
    axes[0].set_ylabel("basin fraction")
    for i, (rho, g) in enumerate(hy.groupby("rho")):
        g = g.sort_values("ratio")
        y = 1.0 - g["frac_at_most_one_switch"]
        axes[1].plot(g["ratio"], y, color=COLORS[i % len(COLORS)], marker="o", ms=3, label=fr"$\rho={rho:g}$")
    axes[1].axvline(1.0, color="k", ls="--", lw=1)
    axes[1].set_xlabel(r"$h/h_{\mathrm{hys}}$ (hysteresis)")
    axes[1].set_ylabel("repeated-switching fraction")
    axes[1].legend(frameon=False, ncol=2, fontsize=6.5, loc="best")
    fig.tight_layout()
    savefig(fig, dest)


def fig_delayed_regions(df: pd.DataFrame, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    markers = {
        "proved_global": "o",
        "locally_stable_globally_unresolved": "s",
        "critical_not_locally_asymptotically_stable": "D",
        "locally_unstable": "^",
    }
    for region, g in df.groupby("region"):
        ax.scatter(g["rho"], g["k"], c=g["frac_to_zero"], cmap="coolwarm_r", vmin=0, vmax=1, marker=markers.get(region, "o"), s=40, label=region.replace("_", " "), edgecolors="k", linewidths=0.3)
    xs = np.linspace(0, 0.95, 40)
    ax.plot(xs, 1 - xs, color="k", ls="--", lw=1, label=r"$k=1-\rho$")
    ax.axhline(1.0, color="0.3", ls=":", lw=1, label=r"$k=1$")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$k=\beta/\tau$")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    savefig(fig, dest)


def fig_gaussian_exact(df: pd.DataFrame, arrays_dir: Path, dest: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    trans_col = "trans_over_tauc" if "trans_over_tauc" in df.columns else "trans_level"
    err_col = "max_abs_pt_err" if "max_abs_pt_err" in df.columns else "p_t_max_abs_err"
    obs = df[(df[trans_col] == 0.0) & (np.isclose(df["rho"], 0.5))] if trans_col in df else df.iloc[0:0]
    if not obs.empty and err_col in obs:
        axes[0].scatter(obs["p_inf"], obs[err_col], color=COLORS[0])
        axes[0].set_xlabel(r"$p_\infty$")
        axes[0].set_ylabel(r"$\max_t |\hat p_t - p_t|$")
    npz = sorted(arrays_dir.glob("*.npz")) if arrays_dir.exists() else []
    if npz:
        data = np.load(npz[0])
        p_hat = data["phat"] if "phat" in data.files else data["p_hat"]
        p_th = data["p_theory"] if "p_theory" in data.files else data["p_th"]
        t = np.arange(1, p_hat.size + 1)
        axes[1].plot(t, p_th, color=COLORS[0], label="analytic $p_t$")
        axes[1].plot(t, p_hat, color=COLORS[1], ls="--", label="Monte Carlo")
        axes[1].set_xlabel("t")
        axes[1].set_ylabel(r"$p_t$")
        axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig(fig, dest)


def fig_v1_v2(cmp_df: pd.DataFrame, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.scatter(cmp_df["v1"], cmp_df["v2"], color=COLORS[0])
    lim = [float(np.nanmin(cmp_df[["v1", "v2"]].values)), float(np.nanmax(cmp_df[["v1", "v2"]].values))]
    ax.plot(lim, lim, color="k", ls="--", lw=1)
    ax.set_xlabel("v1")
    ax.set_ylabel("v2")
    fig.tight_layout()
    savefig(fig, dest)


def fig_appendix_trajectories(dest: Path) -> None:
    from dynafront.dynamics import simulate_delayed, simulate_hysteresis, simulate_scalar, simulate_tanh
    from dynafront.theory import tau_c

    rho, beta = 0.5, 1.0
    t = tau_c(rho, beta)
    T = 120
    z0 = np.array([2.5, -1.8, 0.4])
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.0), sharex=True)
    z_h, _ = simulate_scalar(z0, rho, T, "hard")
    z_c, _ = simulate_scalar(z0, rho, T, "continuous", tau=0.5 * t)
    z_th = simulate_tanh(z0, rho, 0.5 * t, T, beta=beta)
    z_d, _ = simulate_scalar(z0, rho, T, "deadband", h=0.5 * t)
    zh, _ = simulate_hysteresis(z0, np.ones(3), rho, 0.5 * (beta / (1 - rho)), T, beta=beta)
    zd = simulate_delayed(z0, z0, rho, beta / 0.3, T, beta=beta)
    series = [
        (z_h, "hard selector"),
        (z_c, r"clipped $\tau=0.5\tau_c$"),
        (z_th, r"tanh $\tau=0.5\tau_c$"),
        (z_d, r"deadband $h=0.5\tau_c$"),
        (zh, r"hysteresis $h=0.5 h_{\mathrm{hys}}$"),
        (zd, r"delayed $k=0.3$ (proved-global if $\rho+k<1$)"),
    ]
    for ax, (arr, lab) in zip(axes.ravel(), series):
        ax.plot(arr)
        ax.set_ylabel("z")
        ax.text(0.02, 0.9, lab, transform=ax.transAxes, fontsize=8)
        ax.axhline(0, color="0.85", lw=0.6)
    axes[-1, 0].set_xlabel("t")
    axes[-1, 1].set_xlabel("t")
    fig.tight_layout()
    savefig(fig, dest)


def fig_bounded_tubes(df: pd.DataFrame, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    inside = df[df["sufficient_bound"] == True]  # noqa: E712
    ax.scatter(inside["rho"], inside["frac_traj_always_in_tubes"], color=COLORS[2], label="sufficient bound")
    outside = df[df["sufficient_bound"] != True]  # noqa: E712
    if not outside.empty:
        ax.scatter(outside["rho"], outside["frac_traj_always_in_tubes"], color=COLORS[1], marker="x", label="outside sufficient set")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("fraction of trajectories always in tubes")
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, dest)
