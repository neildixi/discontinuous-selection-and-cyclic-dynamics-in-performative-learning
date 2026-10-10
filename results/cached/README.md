# Cached results

These files are the compact public record of the verified suite. They are enough to inspect the reported numbers and to rebuild every public figure **without running simulations**.

Fresh reproductions write to `results/runs/` and never overwrite this directory. `make figures` writes local rebuilds to `results/generated/figures/` and leaves this directory unchanged.

## What the units mean

| File | What each row (or entry) counts |
|---|---|
| `summaries/synthetic_a.csv` | **Jobs / parameter cells.** Five scores \(\rho\), one hard cell each, and 32 clipped widths plus 32 deadband widths. Total 325 jobs. |
| `summaries/synthetic_b.csv` | **Jobs.** Aligned latent-orbit numerical checks (540 jobs). Supplementary; not a main-paper figure family. |
| `summaries/synthetic_c.csv` | **Systems** (one job per system). 300 systems in each of three strata, 20 initial conditions per system. |
| `summaries/unaligned.csv` | **Jobs.** 12 jobs cover 432 accepted systems (108 per resolvent-sign quadrant; dimensions 2, 5, and 20). |
| `summaries/tanh.csv` | **Jobs / selector cells.** 85 cells on the \(\tau/\tau_c\) grid. 40 of them are subcritical. |
| `summaries/itinerary.csv` | **Two sample jobs.** The unconditional job is 2,000 systems. The coverage job retains 25 examples for each primitive period 1–6 after scanning 5,995 systems. Coverage bars are a retained set, not frequencies from the 2,000-system sample. |
| `summaries/hysteresis.csv` | **Jobs / threshold cells.** Deadband and hysteresis ratios, including the identity \(h_{\mathrm{hys}}/\tau_c=(1+\rho)/(1-\rho)\). |
| `summaries/delayed.csv` | **Jobs.** Delayed-clipping region map (supplementary). |
| `summaries/synthetic_d_bounded.csv` | **Jobs.** Sufficient bounded-noise tube checks (supplementary). |
| `summaries/synthetic_d_gaussian.csv` | **Jobs.** Gaussian observability grid (supplementary; not a general escape exponent). |
| `summaries/gaussian_exact.csv` | **Jobs.** Exact Gaussian marginal checks (supplementary). |
| `summaries/headline_numbers.json` | Derived headline values computed from the CSVs. |
| `expected_results.json` | Validation targets and tolerances for those headlines. |
| `figure_data/` | Copies of the CSVs used to draw figures. |
| `figures/` | Manuscript and appendix figures as vector PDF and 300-DPI PNG. |
| `tables/` | Compact LaTeX previews of the summary tables. |
| `MANIFEST.sha256` | SHA-256 of every cached artifact. |

The complete suite contains **2,156 jobs**, not 2,156 independent systems.

## Commands

```bash
make figures
make verify-cached
python -m dynafront.cli compare --mode full --summaries results/cached/summaries
```
