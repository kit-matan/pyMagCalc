# Sunny.jl tutorials, ported to pyMagCalc

Ports of the official [Sunny.jl](https://github.com/SunnySuite/Sunny.jl) tutorial
series (`../../../Sunny.jl-main/examples/01..09`) to pyMagCalc config files.

**Status, audited 2026-08-04; row 06 updated 2026-08-13.** Of the nine tutorials,
**eight are ported** to Sunny or to an exact analytic result, one of those only
transitively justified (07), one is ported in part (09), and **one is blocked** on a
gap that is deliberately open (`GAP_STATUS.md` Gap 4 #16b). Spelled out per row
below rather than summarised optimistically.

| # | Sunny tutorial | What it computes | pyMagCalc | Evidence |
|---|---|---|---|---|
| 01 | CoRh₂O₄ | LSWT dispersion + powder | ✅ **ported, pinned to Sunny** | all 8 bands at 6 q-points, `test_S01_dispersion_matches_sunny_band_by_band` |
| 02 | CoRh₂O₄ finite *T* | Langevin dynamics, static S(q) at *T* | ✅ **ported** | `static_correlations` at 16 K; AFM contrast sharpens on cooling. Thermalized by Metropolis rather than Langevin — both pinned to the same exact distributions |
| 03 | FeI₂ | SU(3) multi-flavour LSWT | ✅ **ported, pinned to Sunny** | E/site, 8 bands **and** intensities < 1e-4, `test_sun.py` |
| 04 | FeI₂ finite *T* | SU(N) classical dynamics at *T* | ✅ **ported** | unblocked by Gap 4 #26; machinery pinned (N=2 → Landau–Lifshitz to 4.8e-10, low-T peak within 1.1% of the LSWT band). This config's own spectrum not compared with Sunny |
| 05 | 2D Ising | thermal Monte Carlo | ✅ **ported, pinned to Onsager** | `propose: flip` + polarized start ⇒ exactly Ising; m(T) matches Onsager to **0.05%**, E(Tc) = −√2 J to 3% |
| 06 | CP² skyrmions | non-equilibrium SU(3) quench | ✅ **ported, pinned to Sunny** | Hamiltonian to **5.4e-13** on an arbitrary coherent-state configuration; quench runs at Sunny's own L = 40 (214 s, was ~55 h) and leaves an exactly quantized, non-zero SU(3) charge. `quench.py`, `test_sun_quench.py` |
| 07 | Pyrochlore dipole-dipole | LSWT + long-range dipole | ⚠️ **ported, not pinned** | the Ewald *engine* matches Sunny to 1.3e-8 (`test_ewald.py`), but **this config's own bands have never been compared** — see below |
| 08 | Momentum conventions | LSWT 1D DM+Ising chain | ✅ **ported, exact analytic pin** | ω(q) = 2s[J ± D sin 2πq₃] including the q → −q sign flip |
| 09 | Disordered triangular AFM | LSWT + KPM with bond disorder | ⚠️ **clean part only** | 120° order pinned analytically; the **disorder is the point of the tutorial** and needs Gap 4 **#16b**, open |

## What "pinned" means here, and where it was not true

The repo rule is that a check a wrong answer passes is not a check. Two rows above
were previously described as validated when nothing asserted it:

- **S01** claimed "the LSWT bands were cross-checked against Sunny 0.8.1", but the
  config has no `magnetic_structure` (it relies on `tasks.minimization`), so the test
  helper could not drive it and it was only schema-checked. It is now pinned band by
  band. It does match Sunny exactly — but that was a fact about the code, not about
  the test suite. Note the model uses `Bond(2, 3, [0,0,0])`, d = 3.68195 Å; reading
  `Bond(1, 3, ...)` off the wrong line gives a different bond and E/site = −8.505
  instead of the correct −2.835.
- **S07** is still only *transitively* justified: the Ewald engine is pinned to Sunny,
  and this config exercises it, but nobody has compared this pyrochlore's spectrum.
  Sunny's example works in **kelvin** (J₁ = 0.304 K) and reshapes to the primitive
  cell before minimising, so a real comparison has to reconcile units and land on the
  same dipolar ground state. Until someone does that, treat S07 as a worked example,
  not as a validated result.

## Two notes on the new ports

- **S05** disables replica exchange (`swap_every: 0`) on purpose. Below Tc the Ising
  model has two degenerate broken-symmetry states, and a replica that visits high T
  and returns can come back with the opposite sign — ⟨m⟩ then averages toward zero
  (measured 0.35 against Onsager's 0.9865 at T = 1.5). Sunny's single-temperature
  `LocalSampler` has no such issue. A test records this so nobody "fixes" it.
- **S02** thermalizes by Metropolis where Sunny uses Langevin. Both sample the same
  Boltzmann distribution and each is pinned independently to exact results, so
  equilibrium averages agree though trajectories do not. `classical_dynamics.langevin_step`
  is available if you want Sunny's route.

## Why 06 and the interesting half of 09 are not here

- **06** needs a DAMPED quench (`Langevin(damping, kT=0)`), i.e. dissipative CP^(N−1)
  relaxation. Gap 4 #26 delivered the energy-CONSERVING flow plus Metropolis, and
  neither substitutes: Metropolis finds equilibrium, while this tutorial is about the
  metastable texture a quench leaves behind. It also needs real-space texture output.
- **09** needs *bond disorder in LSWT*: a large inhomogeneous supercell driven through
  the KPM engine (Gap 4 #16b). Vacancies and open boundaries landed for the classical
  samplers (#16a), but the LSWT half did not, and the tutorial is specifically about
  disorder-broadened spin waves.

Porting any of these by quietly substituting a clean calculation would produce a
folder that looks like a port and is not one. They are listed as blocked instead.

## Running them

```bash
magcalc run examples/sunny_tutorials/S01_CoRh2O4/config.yaml
```

Tests: `tests/test_sunny_tutorials.py` (S01, S02, S05, S08, S09 + schema for all),
`tests/test_sun.py` (S03), `tests/test_ewald.py` (the engine S07 relies on).
