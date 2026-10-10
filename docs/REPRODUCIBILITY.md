# Reproducibility

## Seeds and hashing

- Master seed: `20260821` (`dynafront.seeds.MASTER_SEED`).
- Child streams use `numpy.random.SeedSequence` with a spawn key derived from the experiment name, canonical configuration hash, and seed index.
- Python’s salted `hash()` is never used.
- Job IDs are `experiment__...__` plus the first 12 hex characters of `sha256(canonical_json(config))`.

## Configurations

| Mode | File | Jobs |
|---|---|---|
| `quick` | `configs/quick.yaml` | reduced smoke jobs |
| `paper` | `configs/paper.yaml` | 1,434 |
| `full` | `configs/full.yaml` | 2,156 |

Override the output directory with `--out`. The path must not be `results/cached/` or a subdirectory of it.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks,test]"
```

After dependencies are present, installation and runs do not need network access, CUDA, or authenticated services. See `requirements-lock.txt` for the macOS ARM64 versions used in the clean-environment validation (`appnope` is macOS-only).

## Checkpointing and resume

Each job writes `checkpoints/<family>/<job_id>.json` atomically (temporary file, fsync, rename). A completed valid checkpoint is never overwritten. Manifests record pending / running / completed / failed / interrupted status.

If a laptop sleeps or the process is interrupted, rerun the same command with the same `--out` directory. Completed jobs are skipped; invalid or missing checkpoints are rerun.

```bash
python -m dynafront.cli reproduce --mode paper --out results/runs/<existing_dir> --workers 2
```

`--resume` is the default for reproduction: the runner always skips valid completed jobs in the target directory.

## Runtime

The pipeline times a representative job in each family before launching the rest and prints a projected wall-clock estimate. Progress bars report completed jobs, failures, and rolling time per job.

`make reproduce-full` passes `--skip-if-over-minutes 45`. If the projection exceeds that cap the command exits without starting the remaining full suite.

The full 2,156-job suite typically finishes in a few minutes on a laptop.

## Numerical tolerances

`results/cached/expected_results.json` lists each headline with either `compare: exact` or a float `atol`/`rtol`. Exact keys include job counts, seed, clipped width count (32), system counts, coverage bar heights, and classification counts. Orbit distances and tanh amplitude errors use the stated float tolerances. Tests and the CLI fail if a required comparison misses.

## Figure rebuild

`make figures` reads `results/cached/summaries/` (and `figure_data/`) and writes PDF + 300-DPI PNG to `results/generated/figures/`. It does not overwrite `results/cached/`. Matplotlib is configured with `pdf.fonttype = 42` so fonts are embedded as TrueType.

Check the frozen release copies with:

```bash
make verify-cached
```

Paper-facing reproduction:

```bash
python -m dynafront.cli reproduce --mode paper --workers 2
```

## Paths

All code, notebooks, configs, and docs use paths relative to the repository root discovered from `dynafront.paths.repo_root()`. Renaming the repository directory does not break imports after an editable install.
