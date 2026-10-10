"""Headline numbers extracted from cached CSVs match the verified targets."""

from __future__ import annotations

from dynafront.experiments import EXPECTED_JOB_COUNTS, count_jobs
from dynafront.paths import cached_dir
from dynafront.seeds import MASTER_SEED
from dynafront.summaries import extract_headline_numbers
from dynafront.validation import compare_headlines, load_expected


def test_enumerated_full_job_count_is_2156_jobs() -> None:
    counts = count_jobs("full", MASTER_SEED)
    assert counts["total"] == 2156
    for name, n in EXPECTED_JOB_COUNTS.items():
        assert counts[name] == n
    paper = count_jobs("paper", MASTER_SEED)
    assert paper["total"] == 1434


def test_cached_headlines_match_expected() -> None:
    observed = extract_headline_numbers(cached_dir() / "summaries")
    observed["master_seed"] = MASTER_SEED
    observed["jobs_full"] = 2156
    payload = compare_headlines(observed, load_expected(), mode="full")
    failed = [r for r in payload["rows"] if r["status"] == "FAIL"]
    assert not failed, failed


def test_job_versus_system_units() -> None:
    observed = extract_headline_numbers(cached_dir() / "summaries")
    assert observed["jobs_from_summaries"] == 2156
    assert observed["unaligned_n_systems"] == 432
    assert observed["unaligned_n_jobs"] == 12
    assert observed["s1_n_systems"] == 300
    assert observed["clipped_n_widths"] == 32
    assert observed["itinerary_coverage_period_1"] == 25
    assert observed["itinerary_uncond_n_systems"] == 2000
