# DynaFront

Individually stable deployment responses can still form a cycling or multistable closed loop when a threshold-based rule repeatedly selects among them.

This repository is the public implementation for the paper *Discontinuous Selection and Cyclic Dynamics in Performative Learning*, accepted as a non-archival poster at *DynaFront: Dynamics at the Frontiers of Optimization, Sampling, and Games*, a NeurIPS 2026 workshop. It is organized so that the same package code supports three workflows: inspecting cached numerical results, running short notebooks, and reproducing the experiments from one command.

## 1. Project overview

The code studies affine performative maps in which a score-dependent selector chooses among otherwise contractive responses. The aligned scalar model has an exact threshold \(\tau_c=\beta/(1+\rho)\). Clipped allocation, tanh smoothing, and a neutral deadband share the stabilization scale \(\tau_c=\beta/(1+\rho)\), while hysteresis suppresses repeated switching only at the larger threshold \(h_{\mathrm{hys}}=\beta/(1-\rho)\). In the unaligned multidimensional model the algebraic conditions for stable fixed states and for an alternating pair are independent, and attraction of a compatible pair is local rather than global. Finite switching words can be certified exactly in the asymmetric scalar model.

The repository does not include external simulators or large raw trajectories. Compact summaries are enough to rebuild every public figure.

## 2. Main finding in plain language

A learner that looks stable when a deployment rule is frozen can still oscillate once the rule is allowed to switch. In the simplest aligned setting the switch is caused by a score threshold: each frozen action pulls the score toward a point on the opposite side of the threshold, so the closed loop settles on an alternating cycle rather than a fixed point. Smoothing, deadbands, and hysteresis change the basins, but they do not remove the threshold. In higher dimension the same geometry can produce four combinations of stable fixed states and alternating cycles, and a compatible alternating pair is only a local attractor.

## 3. Repository contents

| Path | Role |
|---|---|
| `src/dynafront/` | Importable package: dynamics, selectors, theorem certificates, experiments, summaries, plotting, CLI |
| `configs/` | `quick.yaml`, `paper.yaml`, and `full.yaml` |
| `notebooks/` | Four short CPU notebooks |
| `results/cached/` | Frozen compact summaries, figure data, figures, and `expected_results.json` |
| `results/generated/` | Local figure rebuilds (`make figures`) |
| `scripts/` | Wrappers for figures, notebooks, comparison, and reproduction |
| `tests/` | Unit tests, including oracle checks that do not import the simulator |
| `docs/` | Experiment table, reproducibility notes, and supported numerical results |

Theorem predictions live in `dynafront.certificates` and do not call the simulator. Empirical classifications are produced by `dynafront.dynamics` and the experiment families.

## 4. Installation

Python 3.10 or newer. No GPU, CUDA, network service, or external dataset is required after the dependencies have been downloaded.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks,test]"
```

Pinned versions from the validated macOS ARM64 environment are in `requirements-lock.txt`. That file is not a universal lock; `appnope` is macOS-only.

## 5. Fast start using cached results

Inspect `results/cached/summaries/` and `results/cached/expected_results.json`. Then rebuild every public figure from those summaries:

```bash
make figures
```

This command does not rerun simulations. Output is PDF (embedded Type 42 fonts) plus 300-DPI PNG under `results/generated/figures/`. Frozen release copies remain in `results/cached/figures/`. Check them with:

```bash
make verify-cached
```

## 6. Runnable notebooks

```bash
make notebooks
```

See `notebooks/README.md`. Each notebook imports `dynafront` and finishes with the command that reruns the corresponding full experiment family.

## 7. Reproducing the paper experiments

Fresh runs write to a new directory under `results/runs/` and never overwrite `results/cached/`. Interrupted runs resume: completed valid checkpoints are skipped.

```bash
python -m dynafront.cli reproduce --mode quick
python -m dynafront.cli reproduce --mode paper --workers 2
python -m dynafront.cli reproduce --mode full
```

Make aliases:

```bash
make quick
make reproduce-paper
make reproduce-full
```

- **quick**: reduced-size smoke jobs for every family, then representative figures.
- **paper**: the 1,434 jobs behind the paper’s reported claims and figures.
- **full**: the complete verified suite of **2,156 jobs** (jobs, not systems), including supplementary families.

Each mode prints the seed, job count, and output directory; estimates runtime after a short benchmark; shows progress bars; reconstructs summaries; rebuilds figures; and compares headlines with `results/cached/expected_results.json`. A failed required comparison returns a nonzero exit code.

## 8. Expected runtime and hardware

On a recent laptop CPU with two workers the full 2,156-job suite typically completes in a few minutes. Notebooks are written to finish in a few minutes each. Memory use is modest: jobs are compact JSON checkpoints without raw trajectories.

`make reproduce-full` will refuse to start if a benchmark projects more than 45 minutes.

## 9. Expected outputs

A successful `paper` or `full` run writes:

- `checkpoints/<family>/*.json` — one file per job
- `summaries/*.csv` and `headline_numbers.json`
- `figures/*.pdf` and `figures/*.png`
- `reports/VALIDATION.md` — comparison with cached reference values

Headline counts that must match exactly include 2,156 full-suite jobs, 32 clipped widths, 432 unaligned systems (108 per quadrant), 300/300 period-two systems in the compatible stratum, 0/300 period-two in the incompatible crossed stratum (285 other finite period, 15 unresolved), 2,000 unconditional itinerary systems, and 25 retained coverage examples per primitive period 1–6.

## 10. Testing

```bash
python -m pytest
```

Oracle tests import `dynafront.certificates` only. Simulator tests import `dynafront.dynamics`. Summary tests read `results/cached/` and do not rerun the suite.

## 11. Reproducibility notes

- Master seed: `20260821`, derived per job with `numpy.random.SeedSequence` (never Python’s salted `hash()`).
- Job IDs are SHA-256 prefixes of canonical JSON configurations.
- Paths inside code, notebooks, and documentation are repository-relative.
- Cached summaries are the scientific record for figures. Newly generated summaries are compared with `expected_results.json` using documented exact/float tolerances.
- Attraction of the alternating pair in the general unaligned model is local, not global.
- No general Gaussian escape exponent is established.

Details: `docs/REPRODUCIBILITY.md`. Copy-ready paper appendix text: `docs/PAPER_REPRODUCIBILITY_APPENDIX.md`.

## 12. Citation

Please cite the paper:

> Neil Dixit, *Discontinuous Selection and Cyclic Dynamics in Performative Learning*, DynaFront: Dynamics at the Frontiers of Optimization, Sampling, and Games, NeurIPS 2026 Workshop (accepted poster, non-archival), 2026.

See `CITATION.cff`. Repository: [https://github.com/neildixi/discontinuous-selection-and-cyclic-dynamics-in-performative-learning](https://github.com/neildixi/discontinuous-selection-and-cyclic-dynamics-in-performative-learning).

## 13. License

The code in this repository is released under the MIT License. See `LICENSE`. The paper’s CC BY 4.0 status does not license the code.
