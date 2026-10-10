# Supported numerical results

These statements are the public scientific record. They are computed from `results/cached/summaries/` and checked against `expected_results.json`. They are not restatements of internal review correspondence.

## Suite size

The complete verified suite contains **2,156 jobs**. That number counts configuration cells (checkpoints), not independent systems. Twelve smoke jobs exist only for pipeline testing and are excluded from 2,156. The paper-facing subset is **1,434 jobs**.

## Aligned scalar selectors

For the running example \(\rho=1/2\), \(\beta=1\), the shared threshold is \(\tau_c=\beta/(1+\rho)=2/3\). Frozen actions \(a\equiv +1\) and \(a\equiv -1\) are individually contractive. The hard selector \(a=\mathrm{sign}(z)\) with \(+1\) at zero produces the alternating scores \(\pm\tau_c\) with two-step factor \(\rho^2\).

Across the five hard-selector cells, the largest *mean-over-initial-condition* distance to that orbit is \(1.52467\times 10^{-11}\). The largest per-initial-condition mean post-burn-in distance in the original array record is \(4.58\times 10^{-11}\).

Clipped (linear smoothing with clipping) uses **32 widths** on each \(\rho\) slice, including the exact ratio \(1\). Deadband uses the same 32 ratios with threshold \(\tau_c\). Hysteresis uses the larger threshold \(h_{\mathrm{hys}}=\beta/(1-\rho)\). The identity \(h_{\mathrm{hys}}/\tau_c=(1+\rho)/(1-\rho)\) holds on the hysteresis grid. At a shared numerical threshold the selectors still differ: equality in hysteresis retains the previous action, including the \(\rho=0\) case.

## Tanh selector

Every subcritical tanh cell (40/40) converged at the adaptive horizon. The maximum amplitude error relative to the independently solved \(r_\tau\) of \((1+\rho)r=\beta\tanh(r/\tau)\) over those 40 cells is \(2.40\times 10^{-15}\). Restricting to \(\tau/\tau_c<0.98\) gives \(1.25\times 10^{-15}\). Critical cells \(\tau=\tau_c\) are censored and are not amplitude claims. The square-root onset formula is an approximation, not the oracle.

## Unaligned two-resolvent model

432 accepted systems, 108 in each strict sign quadrant of \((\chi_{\mathrm{fix}},\chi_{\mathrm{cyc}})\), at dimensions 2, 5, and 20. Oracle and simulated selector compatibility agree. Frozen-state relative error is on the order of \(10^{-15}\). Compatible alternating pairs were tested with local perturbations; **attraction is local, not global**.

## Asymmetric affine systems

All **300** crossed systems with a compatible alternating pair reached period two from all 20 tested initial conditions.

None of the **300** crossed systems without pair compatibility reached period two. Of those 300, **285** reached another finite period and **15** remained unresolved at the simulation horizon.

The non-crossed stratum (300 systems) is a period-one / self-consistent fixed-action check.

## Finite itineraries

The unconditional experiment sampled **2,000** systems. No observed strict finite itinerary lacked a matching primitive-word certificate, and certified local perturbations did not fail. The maximum itinerary state discrepancy is \(8.7315\times 10^{-11}\).

The balanced period-coverage figure retains **25** examples for each primitive period from one through six after scanning **5,995** systems. Those bars are a coverage set, not empirical frequencies from the 2,000-system sample.

## What is not claimed

- No general Gaussian escape or metastability exponent is established. Gaussian files check exact marginals, finite-horizon sandwiches, or observability, and they are supplementary.
- Bounded-noise tube success is a sufficient condition, not a necessary one.
- Delayed clipping has a proved global region, a linearly unstable region, a critical non-LAS line, and a globally unresolved band; the unresolved band is observational.
- This repository contains no external loan-simulator case study.
