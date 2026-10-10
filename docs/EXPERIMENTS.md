# Experiment families

Every row below is implemented once in `src/dynafront/` and used by notebooks, cached-figure rebuilds, and the automatic pipeline. A **job** is one configuration cell and one checkpoint. Do not read the 2,156-job total as 2,156 independent systems.

Master seed: `20260821`.

| Family | Jobs | Unit counted | In `paper` mode | Role |
|---|---:|---|---|---|
| `synthetic_a` | 325 | Parameter cells. 5 values of \(\rho\); 1 hard cell; 32 clipped widths; 32 deadband widths | yes | Aligned scalar phase diagram |
| `synthetic_b` | 540 | Matrix/seed cells | no (full only) | Numerical check of aligned latent orbits |
| `synthetic_c` | 900 | Systems (one job per system), 20 ICs each | yes | Three strata of asymmetric affine responses |
| `synthetic_d_bounded` | 15 | Parameter cells | no | Sufficient bounded-noise tube condition |
| `synthetic_d_gaussian` | 75 | Parameter cells | no | Gaussian observability grid; not an escape exponent |
| `unaligned` | 12 | Jobs holding 36 systems each (432 systems total; 108 per sign quadrant; \(d\in\{2,5,20\}\)) | yes | Independent \(\chi_{\mathrm{fix}}\) and \(\chi_{\mathrm{cyc}}\) conditions; local tests |
| `tanh` | 85 | Selector cells on a \(\tau/\tau_c\) grid (40 subcritical) | yes | Smooth selector sharing \(\tau_c\) |
| `itinerary` | 2 | Two sample jobs: 2,000 unconditional systems; coverage retains 25 examples per period 1–6 after scanning 5,995 systems | yes | Finite-word certificates versus simulated itineraries |
| `hysteresis` | 110 | Threshold cells for deadband \(h/\tau_c\) and hysteresis \(h/h_{\mathrm{hys}}\) | yes | Distinct thresholds \(\beta/(1+\rho)\) and \(\beta/(1-\rho)\) |
| `delayed` | 65 | Parameter cells | no | Delayed clipping regions, including an unresolved band |
| `gaussian_exact` | 27 | Parameter cells | no | Exact Gaussian marginals / finite-horizon sandwich |
| **Total (full)** | **2156** | **Jobs** | | Smoke jobs are extra and excluded from this total |
| **Total (paper)** | **1434** | **Jobs** | | Families required for reported claims and figures |

`quick` mode does not use these counts. It runs a handful of reduced-size smoke jobs from each family so that checkpointing, resume, summaries, and figure rebuild can be tested quickly.

## Strata in `synthetic_c`

| Stratum | Systems | What is certified |
|---|---:|---|
| Crossed equilibria and a sign-compatible alternating pair | 300 | Period two from all 20 tested initial conditions |
| Crossed equilibria without pair compatibility | 300 | Period two is not reached; 285 systems reach another finite period; 15 remain unresolved at the horizon |
| Non-crossed | 300 | Period one / self-consistent fixed action |

## Itinerary samples

The unconditional 2,000-system sample and the period-coverage set are different objects. Coverage bars with 25 examples per primitive period 1–6 are a retained set after scanning 5,995 systems, not empirical frequencies from the 2,000-system sample.
