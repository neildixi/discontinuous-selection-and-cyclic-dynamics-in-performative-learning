"""Regenerate public figures and tables from compact summaries. No simulations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dynafront import _manuscript_figures as vf
from dynafront.figures import (
    asymmetric_strata,
    gaussian_heatmaps,
    latent_orbit_error,
    phase_transition_continuous,
    phase_transition_deadband,
    representative_trajectories,
)
from dynafront.io_atomic import atomic_write_csv
from dynafront.paths import cached_dir, generated_figures_dir

PAPER_FIGURES = (
    "fig01_clipped_tanh_phase",
    "fig02_two_resolvent_quadrants",
    "fig03_itinerary_certificates",
    "fig04_deadband_hysteresis_thresholds",
)

APPENDIX_FIGURES = (
    "app_clipped_phase",
    "app_deadband_basins",
    "app_representative_hard_clipped_deadband",
    "app_latent_orbit_error",
    "app_asymmetric_strata",
    "app_gaussian_observability",
    "app_delayed_region_map",
    "app_bounded_tubes",
    "app_gaussian_exact_pt",
    "app_selector_trajectories",
)


def _csv(root: Path, name: str) -> pd.DataFrame:
    path = Path(root) / "summaries" / name
    if not path.exists():
        alt = Path(root) / "figure_data" / name
        path = alt if alt.exists() else path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def rebuild_figures(source: Path | None = None, dest: Path | None = None) -> list[Path]:
    """Write PDF and 300-DPI PNG figures from cached or reconstructed summaries.

    Default destination is ``results/generated/figures/``. Cached artifacts are
    never rewritten from this function.
    """
    source = Path(source) if source is not None else cached_dir()
    dest = Path(dest) if dest is not None else generated_figures_dir()
    dest.mkdir(parents=True, exist_ok=True)
    write_aux = source.resolve() != cached_dir().resolve()
    fig_data = source / "figure_data"
    if write_aux:
        fig_data.mkdir(parents=True, exist_ok=True)
        (source / "tables").mkdir(parents=True, exist_ok=True)

    a = _csv(source, "synthetic_a.csv")
    b = _csv(source, "synthetic_b.csv")
    c = _csv(source, "synthetic_c.csv")
    db = _csv(source, "synthetic_d_bounded.csv")
    dg = _csv(source, "synthetic_d_gaussian.csv")
    u = _csv(source, "unaligned.csv")
    th = _csv(source, "tanh.csv")
    it = _csv(source, "itinerary.csv")
    hy = _csv(source, "hysteresis.csv")
    de = _csv(source, "delayed.csv")
    gx = _csv(source, "gaussian_exact.csv")

    written: list[Path] = []

    def _stem(name: str) -> Path:
        p = dest / name
        written.append(p)
        return p

    def _aux_csv(name: str, frame: pd.DataFrame) -> None:
        if write_aux:
            atomic_write_csv(fig_data / name, frame)

    if not a.empty:
        phase_transition_continuous(a, _stem("app_clipped_phase"))
        phase_transition_deadband(a, _stem("app_deadband_basins"))
        representative_trajectories(_stem("app_representative_hard_clipped_deadband"))
        _aux_csv("synthetic_a.csv", a)
    if not b.empty:
        latent_orbit_error(b, _stem("app_latent_orbit_error"))
        _aux_csv("synthetic_b.csv", b)
    if not c.empty:
        asymmetric_strata(c, _stem("app_asymmetric_strata"))
        _aux_csv("synthetic_c.csv", c)
    if not dg.empty:
        gaussian_heatmaps(dg, _stem("app_gaussian_observability"))
        _aux_csv("synthetic_d_gaussian.csv", dg)
    if not th.empty:
        vf.fig01_clipped_tanh_phase(th, _stem("fig01_clipped_tanh_phase"))
        _aux_csv("tanh.csv", th)
    if not u.empty:
        vf.fig02_two_resolvent_quadrants(u, _stem("fig02_two_resolvent_quadrants"))
        _aux_csv("unaligned.csv", u)
    if not it.empty:
        vf.fig03_itinerary_certificates(it, c, _stem("fig03_itinerary_certificates"))
        _aux_csv("itinerary.csv", it)
    if not hy.empty:
        vf.fig04_deadband_hysteresis(hy, _stem("fig04_deadband_hysteresis_thresholds"))
        _aux_csv("hysteresis.csv", hy)
    if not de.empty:
        vf.fig_delayed_regions(de, _stem("app_delayed_region_map"))
        _aux_csv("delayed.csv", de)
    if not db.empty:
        vf.fig_bounded_tubes(db, _stem("app_bounded_tubes"))
        _aux_csv("synthetic_d_bounded.csv", db)
    if not gx.empty:
        vf.fig_gaussian_exact(gx, source / "raw" / "gaussian_exact", _stem("app_gaussian_exact_pt"))
        _aux_csv("gaussian_exact.csv", gx)
    vf.fig_appendix_trajectories(_stem("app_selector_trajectories"))
    return written
