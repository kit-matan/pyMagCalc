# S04 — FeI₂ at finite temperature (generalized spin dynamics)

Port of Sunny tutorial `04_GSD_FeI2.jl`.

> **This file previously said OUT OF SCOPE**, blocked on SU(N) classical dynamics.
> Gap 4 **#26** closed that, and the tutorial is ported below.

## What it computes

The **same** FeI₂ Hamiltonian as [`../S03_FeI2_SUN/`](../S03_FeI2_SUN/) — six anisotropic
exchange matrices plus a large easy-axis single-ion term, `D = 2.165` meV — but at
finite temperature, evolving CP^(N−1) coherent states instead of diagonalizing about
the ground state.

Dipole Landau–Lifshitz cannot do this model at all. FeI₂'s defining feature is a
single-ion **bound state**: a transition within the S=1 ion's three levels, which one
precessing dipole has no way to express. That is why the tutorial is "generalized"
spin dynamics.

```yaml
tasks: {minimization: true, sun_sampled_correlations: true}
```

## Two things that will bite you

**The supercell is not optional.** This is *real-space* dynamics, so the cell must be
big enough to represent the q you ask for. LSWT needs no such thing — it works in
q-space — and carrying that habit across is exactly how the #26 investigation went
wrong: run on the chemical cell, and you silently measure that small system's own
normal mode. It came out near half the LSWT band and looked precisely like a
factor-of-2 bug. The config uses FeI₂'s non-diagonal magnetic cell
`[1 0 0; 0 1 −2; 0 1 2]` extended ×4 along **a**.

**Classical renormalization is real, not an error.** Modes soften at finite kT.
Measured on a test model: 21% below the LSWT band at kT = 0.15, recovering to within
1.1% as kT → 0. Compare with LSWT only in the low-temperature limit.

## Validation

The machinery is pinned in `tests/test_sun_dynamics.py`: the CP^(N−1) equations
reduce to Landau–Lifshitz at N = 2 to **4.8e-10**, energy and every ‖Z_i‖ are
conserved to 1e-8 / 1e-12, and the low-T S(q,ω) peak sits within 1.1% of the SU(N)
LSWT band, hardening monotonically toward it on cooling.

This config *exercises* that machinery on FeI₂; its own spectrum has **not** been
compared with Sunny's. The ground state it converges to is the pinned
`E/site = −2.91893118` from `tests/test_sun.py`. Treat the finite-T spectrum as a
worked example, not a validated result — the same caveat S07 carries.

## Running it

```bash
magcalc run examples/sunny_tutorials/S04_FeI2_finiteT/config.yaml
```

About 2.5 minutes: CP^(N−1) ground-state search, then thermalization and trajectories.
