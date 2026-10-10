"""Publication figures: PDF and PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def savefig(fig: plt.Figure, dest_stem: Path) -> None:
    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(dest_stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def phase_transition_continuous(df: pd.DataFrame, dest: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharex=True)
    for rho, g in df[df["selector"] == "continuous"].groupby("rho"):
        g = g.sort_values("ratio")
        axes[0].plot(g["ratio"], g["A_z_norm_mean"], marker="o", ms=3, label=fr"$\rho={rho}$")
        axes[1].plot(g["ratio"], g["frac_zero"], marker="o", ms=3, label=fr"$\rho={rho}$")
    for ax in axes:
        ax.axvline(1.0, color="k", ls="--", lw=1, alpha=0.7)
        ax.set_xlabel(r"$\tau/\tau_c$")
    axes[0].set_ylabel(r"normalized amplitude $A_z/\tau_c$")
    axes[0].set_title("Clipped selector: amplitude")
    axes[1].set_ylabel("fraction to zero")
    axes[1].set_title("Clipped selector: zero convergence")
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    savefig(fig, dest)


def phase_transition_deadband(df: pd.DataFrame, dest: Path) -> None:
    sub = df[df["selector"] == "deadband"]
    rhos = sorted(sub["rho"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), sharex=True)
    for rho in rhos:
        g = sub[sub["rho"] == rho].sort_values("ratio")
        axes[0].plot(g["ratio"], g["A_z_norm_mean"], marker="o", ms=3, label=fr"$\rho={rho}$")
        axes[1].plot(g["ratio"], g["frac_zero"], marker="o", ms=3)
        axes[2].plot(g["ratio"], g["frac_cycle"], marker="o", ms=3)
    for ax in axes:
        ax.axvline(1.0, color="k", ls="--", lw=1, alpha=0.7)
        ax.set_xlabel(r"$h/\tau_c$")
    axes[0].set_ylabel(r"$A_z/\tau_c$ (do not interpret alone)")
    axes[0].set_title("Deadband amplitude")
    axes[1].set_ylabel("fraction to zero")
    axes[1].set_title("Zero basin")
    axes[2].set_ylabel("fraction to hard cycle")
    axes[2].set_title("Cycling basin")
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Subcritical deadband: coexistence, not mean amplitude", y=1.02, fontsize=10)
    fig.tight_layout()
    savefig(fig, dest)


def representative_trajectories(dest: Path) -> None:
    from dynafront.dynamics import simulate_scalar
    from dynafront.theory import tau_c

    rho, beta = 0.5, 1.0
    t = tau_c(rho, beta)
    T = 80
    z0 = np.array([2.5])
    specs = [
        ("fixed +", "continuous", {"tau": 1e9}, np.array([2.5]), +1),  # handled below
    ]
    fig, axes = plt.subplots(4, 2, figsize=(8.6, 9.0), sharex=False)
    axes = axes.ravel()
    panels = [
        ("Fixed positive $a\\equiv +1$", "fixed_pos"),
        ("Fixed negative $a\\equiv -1$", "fixed_neg"),
        ("Hard selector", "hard"),
        (r"Continuous $\tau=1.5\tau_c$", "cont_super"),
        (r"Continuous $\tau=\tau_c$", "cont_crit"),
        (r"Continuous $\tau=0.5\tau_c$", "cont_sub"),
        (r"Deadband $h=0.5\tau_c$ (zero basin)", "dead_zero"),
        (r"Deadband $h=0.5\tau_c$ (cycle basin)", "dead_cycle"),
    ]
    z_fixed_p = np.empty(T + 1)
    z_fixed_n = np.empty(T + 1)
    z_fixed_p[0] = 2.5
    z_fixed_n[0] = -2.5
    for k in range(T):
        z_fixed_p[k + 1] = rho * z_fixed_p[k] - beta * 1.0
        z_fixed_n[k + 1] = rho * z_fixed_n[k] - beta * (-1.0)
    z_hard, _ = simulate_scalar(np.array([2.5, -2.5]), rho, T, "hard")
    z_super, _ = simulate_scalar(np.array([2.5, -1.2]), rho, T, "continuous", tau=1.5 * t)
    z_crit, _ = simulate_scalar(np.array([2.5, 0.4 * t]), rho, T, "continuous", tau=t)
    z_sub, _ = simulate_scalar(np.array([2.5, 0.2]), rho, T, "continuous", tau=0.5 * t)
    z_dz, _ = simulate_scalar(np.array([0.2 * t]), rho, T, "deadband", h=0.5 * t)
    z_dc, _ = simulate_scalar(np.array([3.0]), rho, T, "deadband", h=0.5 * t)
    series = [
        (z_fixed_p[:, None],),
        (z_fixed_n[:, None],),
        (z_hard,),
        (z_super,),
        (z_crit,),
        (z_sub,),
        (z_dz,),
        (z_dc,),
    ]
    for ax, (title, _), (arr,) in zip(axes, panels, series):
        for j in range(arr.shape[1]):
            ax.plot(arr[:, j], lw=1.4)
        ax.axhline(t, color="0.5", ls=":", lw=0.8)
        ax.axhline(-t, color="0.5", ls=":", lw=0.8)
        ax.axhline(0, color="0.8", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_ylabel("z")
    fig.tight_layout()
    savefig(fig, dest)


def latent_orbit_error(df: pd.DataFrame, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for regime, g in df.groupby("latent_regime"):
        gg = g.groupby("d")["rel_state_orbit_error"].median()
        ax.plot(gg.index, gg.values, marker="o", label=regime)
    ax.set_xlabel("latent dimension $d$")
    ax.set_ylabel("relative state orbit error")
    ax.set_title("Numerical verification, not an inferential claim")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, dest)


def asymmetric_strata(df: pd.DataFrame, dest: Path) -> None:
    order = ["crossed_sign_compatible", "crossed_no_pair", "noncrossed"]
    labels = ["Crossed +\nsign-compatible pair", "Crossed,\nno sign-compatible pair", "Noncrossed"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(order))
    width = 0.22
    metrics = [
        ("frac_period_two", "certified period 2"),
        ("frac_other_period", "other finite period"),
        ("frac_unresolved_period", "unresolved"),
    ]
    for i, (col, lab) in enumerate(metrics):
        vals = [float(df.loc[df["stratum"] == s, col].mean()) for s in order]
        ax.bar(x + (i - 1) * width, vals, width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("fraction of systems (mean over ICs)")
    ax.set_title("Crossed equilibria do not imply period two")
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, dest)


def gaussian_heatmaps(df: pd.DataFrame, dest: Path) -> None:
    rhos = sorted(df["rho"].unique())
    fig, axes = plt.subplots(1, len(rhos), figsize=(4.0 * len(rhos), 3.6), sharey=True)
    if len(rhos) == 1:
        axes = [axes]
    for ax, rho in zip(axes, rhos):
        g = df[df["rho"] == rho]
        pivot = g.pivot_table(index="obs_level", columns="trans_level", values="p2_share_mean")
        im = ax.imshow(pivot.values, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(pivot.shape[1]), [f"{c:g}" for c in pivot.columns])
        ax.set_yticks(range(pivot.shape[0]), [f"{c:g}" for c in pivot.index])
        ax.set_xlabel(r"transition noise / $\tau_c$")
        ax.set_title(fr"$\rho={rho}$")
    axes[0].set_ylabel(r"score noise / $\tau_c$")
    fig.colorbar(im, ax=axes, fraction=0.025, label="period-two spectral share")
    fig.suptitle("Gaussian observability (not a bounded-noise theorem check)", y=1.03, fontsize=10)
    savefig(fig, dest)
