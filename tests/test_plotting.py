"""Figure rebuild from cached summaries does not simulate the full suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dynafront.cli import cmd_figures, verify_cached_manifest
from dynafront.io_atomic import sha256_file
from dynafront.paths import cached_dir
from dynafront.plotting import rebuild_figures


def test_rebuild_paper_figures(tmp_path: Path) -> None:
    dest = tmp_path / "figures"
    written = rebuild_figures(cached_dir(), dest)
    stems = {p.name for p in written}
    for name in (
        "fig01_clipped_tanh_phase",
        "fig02_two_resolvent_quadrants",
        "fig03_itinerary_certificates",
        "fig04_deadband_hysteresis_thresholds",
    ):
        assert name in stems
        assert (dest / f"{name}.pdf").exists()
        assert (dest / f"{name}.png").exists()
        assert (dest / f"{name}.png").stat().st_size > 1000


def test_fig04_repeated_switch_drops_at_threshold() -> None:
    hy = pd.read_csv(cached_dir() / "summaries" / "hysteresis.csv")
    hy = hy[hy["kind"] == "hysteresis"]
    y = 1.0 - hy["frac_at_most_one_switch"]
    below = y[hy["ratio"] < 1.0]
    above = y[hy["ratio"] >= 1.0]
    assert (below > 0.99).all()
    assert (above < 0.01).all()


def test_rebuild_does_not_mutate_cached(tmp_path: Path) -> None:
    before = {
        p: sha256_file(p)
        for p in cached_dir().rglob("*")
        if p.is_file()
    }
    rebuild_figures(cached_dir(), tmp_path / "figures")
    after = {
        p: sha256_file(p)
        for p in cached_dir().rglob("*")
        if p.is_file()
    }
    assert before == after


def test_figures_cli_refuses_cached_dest() -> None:
    class Args:
        source = None
        dest = str(cached_dir() / "figures")

    with pytest.raises(ValueError, match="cached"):
        cmd_figures(Args())


def test_verify_cached_manifest() -> None:
    assert verify_cached_manifest() == []
