# S02 — CoRh₂O₄ at finite temperature — OUT OF SCOPE

Sunny tutorial `02_LLD_CoRh2O4.jl` samples the CoRh₂O₄ Néel antiferromagnet in
**thermal equilibrium** (Langevin / Landau-Lifshitz dynamics at T = 16 K) and
extracts the dynamical structure factor `S(Q,ω)` from classical spin-dynamics
trajectories (`SampledCorrelations`), including a static-structure-factor slice
and a powder average — all at finite *T*.

**pyMagCalc does not implement finite-temperature classical spin dynamics**
(Langevin integration, `SampledCorrelations`). This is Gap Tier 2 #5 in
`GAP_STATUS.md` — not done. pyMagCalc is a *linear* spin-wave engine: it expands
about the T = 0 classical ground state and gives the harmonic magnon spectrum, not
thermally-broadened classical correlations above/near `T_N`.

**Closest available:** the *same* CoRh₂O₄ spin model at T = 0 via LSWT —
[`../S01_CoRh2O4/`](../S01_CoRh2O4/). Its dispersion is the coherent spectrum that
the finite-*T* dynamics of this tutorial would broaden and populate thermally.
