# Notebooks

Four short notebooks, each importing `dynafront` rather than reimplementing the maps. Run them from the repository root after `pip install -e ".[notebooks]"`, or execute all of them with:

```bash
make notebooks
```

Kernels start in `notebooks/`. Paths to cached summaries use `dynafront.paths.repo_root()`.

| Notebook | Topic | Full-family command |
|---|---|---|
| `01_hard_threshold_cycle.ipynb` | Frozen-policy contractions versus the hard closed loop | `python -m dynafront.cli reproduce --mode paper` (family `synthetic_a`) |
| `02_selector_design.ipynb` | Hard, clipped, tanh, deadband, hysteresis | same, plus `tanh` and `hysteresis` |
| `03_multidimensional_regimes.ipynb` | Four sign regimes; local attraction | family `unaligned` |
| `04_finite_itinerary_certificates.ipynb` | Certified words versus coverage vs unconditional sampling | families `itinerary` and `synthetic_c` |

Each notebook is meant to finish in a few minutes on a laptop CPU.
