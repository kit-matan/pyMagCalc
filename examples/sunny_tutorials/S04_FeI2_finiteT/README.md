# S04 — FeI₂ at finite temperature — OUT OF SCOPE

Sunny tutorial `04_GSD_FeI2.jl` is the finite-*T* companion to the FeI₂ SU(3)
study. It runs the **classical dynamics of SU(N) coherent states** with Langevin
coupling to a thermal bath, thermalises a 16×16×4 supercell at 2.3–3.5 K, and
measures the dynamical structure factor `S(Q,ω)` from those trajectories
(`SampledCorrelations`), including the paramagnetic phase above ordering.

**pyMagCalc does not implement finite-temperature (SU(N) or dipole) classical
spin dynamics** — Gap Tier 2 #5 in `GAP_STATUS.md`, not done. It provides the
*linear* SU(N) spin-wave spectrum at T = 0, not thermally-sampled correlations.

**Closest available:** the FeI₂ SU(3) LSWT spectrum at T = 0 —
[`../S03_FeI2_SUN/`](../S03_FeI2_SUN/), which runs the validated
`examples/materials/FeI2/config_fei2_sun.yaml` (bands + intensities vs Sunny to
< 1e-4). That is the coherent multi-flavor spectrum this tutorial thermally
broadens.
