# Sunny.jl tutorials, ported to pyMagCalc

Ports of the official [Sunny.jl](https://github.com/SunnySuite/Sunny.jl) tutorial
series (`examples/reference/sunny_jl/01..09`) to pyMagCalc config files.

**pyMagCalc is a linear-spin-wave-theory (LSWT) engine.** The Sunny series mixes
LSWT calculations with finite-temperature classical spin dynamics, thermal Monte
Carlo, and non-equilibrium quenches. Only the LSWT calculations map onto
pyMagCalc; the finite-*T* / real-space-dynamics tutorials are **out of scope** and
are documented as such rather than faked (they are Gap Tier 2 #5/#6 in
`GAP_STATUS.md` — not implemented). Each out-of-scope folder points at the LSWT
sibling that shares the same spin Hamiltonian.

| # | Sunny tutorial | What it computes | pyMagCalc | Folder |
|---|---|---|---|---|
| 01 | CoRh₂O₄ | LSWT dispersion + powder | ✅ ported | `S01_CoRh2O4/` |
| 02 | CoRh₂O₄ finite *T* | Langevin dynamics, S(Q,ω) at *T* | ❌ out of scope | `S02_CoRh2O4_finiteT/` → S01 |
| 03 | FeI₂ | SU(3) multi-flavor LSWT | ✅ ported | `S03_FeI2_SUN/` |
| 04 | FeI₂ finite *T* | SU(N) classical dynamics at *T* | ❌ out of scope | `S04_FeI2_finiteT/` → S03 |
| 05 | 2D Ising | thermal Monte Carlo | ❌ out of scope | `S05_Ising_MC/` |
| 06 | CP² skyrmions | non-equilibrium SU(3) quench | ❌ out of scope | `S06_CP2_skyrmions/` |
| 07 | Pyrochlore dipole-dipole | LSWT + long-range dipole (Ewald + cutoff) | ✅ ported | `S07_dipole_dipole/` |
| 08 | Momentum conventions | LSWT 1D DM+Ising chain (ω(q) sign) | ✅ ported | `S08_momentum_conventions/` |
| 09 | Disordered triangular AFM | LSWT (clean part); KPM disorder | ⚠️ partial | `S09_triangular_AFM/` |

## How the ports were validated

Following the repo rule (*a check that a wrong answer passes is not a check*),
each ported config is pinned to an **independent** result, not a self-generated
golden number:

- **S01** — the Néel diamond AFM has classical energy `-2 J s² = -2.835 meV/site`
  (exact), and the LSWT bands were cross-checked against Sunny 0.8.1.
- **S03** — the FeI₂ SU(3) spectrum is already validated band-by-band **and** in
  intensity against Sunny to < 1e-4 (`tests/test_sun.py`); this folder is the
  config bridge to it.
- **S07** — the long-range dipole-dipole engine (Ewald + truncated) is validated
  against Sunny to 1.3e-8 (`tests/test_ewald.py`); this folder exercises it on the
  pyrochlore.
- **S08** — the 1D chain has the **exact analytic** dispersion
  `ω(q) = 2 s [J ± D sin(2π q₃)]`; the ± sign flip across q → −q (the whole point
  of the tutorial) is checked directly.
- **S09** — the 120° triangular AFM has a Goldstone mode at K = [1/3,1/3,0]; the
  clean LSWT is ported. The KPM disorder-broadening (SpinWaveTheoryKPM) is not
  implemented.
