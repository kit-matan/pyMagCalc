# S06 — CP² skyrmions — BLOCKED (needs a damped SU(N) quench)

Port of Sunny tutorial `06_CP2_Skyrmions.jl`. **Not ported**, and the blocker is now
narrower than this file used to claim.

## What the tutorial does

An SU(3) model on a triangular lattice — competing J₁/J₂ exchange with anisotropy
Δ = 2.6, a field h = 15.5 and easy-plane `D = 19.0` — is **randomized and then
quenched**:

```julia
integrator = Langevin(; damping=0.05, kT=0)
randomize_spins!(sys)     # -> relax -> CP^2 skyrmion texture
```

and snapshots are taken at τ = 4, 16, 256 to show skyrmions forming.

## Why it is still blocked

Gap 4 **#26 closed the conservative CP^(N−1) dynamics** — `i dZ_i/dt = h_i Z_i`,
which provably conserves energy (tested to 1e-8) — plus Metropolis sampling. Neither
can do this tutorial:

- a **damped quench at kT = 0** is *dissipative* relaxation, and #26 built the
  energy-conserving flow;
- **Metropolis is not a substitute.** It finds equilibrium; this tutorial is about the
  metastable texture a quench leaves *behind*. A Metropolis ground state would not
  contain skyrmions.

Two pieces are therefore missing:

1. **an SU(N) damping term** — the CP^(N−1) analogue of Landau–Lifshitz–Gilbert. Worth
   doing carefully: the dipole version's damping sign was wrong on first writing
   (Gap 4 #18) and produced a magnetization of the right magnitude and the wrong sign,
   caught only against an exact reference. Its validation route is clear, though —
   the zero-damping limit must reproduce the conservative flow already pinned in
   `tests/test_sun_dynamics.py`;
2. **real-space texture output** — the tutorial's product is snapshots, not a
   spectrum, and pyMagCalc has no SU(N) texture plot.

## Not this

The neighbouring [`../S04_FeI2_finiteT/`](../S04_FeI2_finiteT/) uses the SU(N) dynamics
that *does* exist, for equilibrium finite-T spectra. It is not a stand-in for a
quench: substituting one for the other would give a folder that looks like a port and
is not one.
