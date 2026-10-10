# Paper reproducibility appendix (copy-ready)

The manuscript source is not in this repository. Paste the following into the paper’s reproducibility appendix.

## Replacement text

The complete implementation, cached numerical results, and figure-rebuild scripts are available at:

https://github.com/neildixi/discontinuous-selection-and-cyclic-dynamics-in-performative-learning

The paper experiments use the public package interface, not the internal `experiments.run` commands.

To reproduce the paper-facing suite:

```bash
python -m dynafront.cli reproduce --mode paper --workers 2
```

To rebuild only the supplied figures from the frozen cached summaries (writes to `results/generated/figures/`, and does not overwrite `results/cached/`):

```bash
make figures
```

To check that the frozen cached artifacts match the release SHA-256 manifest:

```bash
make verify-cached
```

The complete verified suite is 2,156 jobs. On a laptop it typically finishes in a few minutes. Cached summaries are enough to inspect the reported numbers and to rebuild every public figure without rerunning simulations.
