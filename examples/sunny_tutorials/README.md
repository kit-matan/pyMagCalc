# Sunny.jl tutorials, ported to pyMagCalc

Ports of the official [Sunny.jl](https://github.com/SunnySuite/Sunny.jl) tutorial
series (`../../../Sunny.jl-main/examples/01..09`) to pyMagCalc config files.

**Status, audited 2026-08-04.** Of the nine tutorials, **four are ported and pinned
to Sunny or to an exact analytic result**, one is ported in part, and four are not
ported. Two of those four are blocked on gaps that are deliberately still open
(`GAP_STATUS.md` Gap 4 #16b and #26); two are portable now and simply have not been
done. That is spelled out per row below rather than summarised optimistically.

| # | Sunny tutorial | What it computes | pyMagCalc | Evidence |
|---|---|---|---|---|
| 01 | CoRh₂O₄ | LSWT dispersion + powder | ✅ **ported, pinned to Sunny** | all 8 bands at 6 q-points, `test_S01_dispersion_matches_sunny_band_by_band` |
| 02 | CoRh₂O₄ finite *T* | Langevin dynamics, S(Q,ω) at *T* | ⬜ **not ported** (now possible) | needs a `sampled_correlations` config; the engine exists (Gap 3 #5, Gap 4 #17/#18) |
| 03 | FeI₂ | SU(3) multi-flavour LSWT | ✅ **ported, pinned to Sunny** | E/site, 8 bands **and** intensities < 1e-4, `test_sun.py` |
| 04 | FeI₂ finite *T* | SU(N) classical dynamics at *T* | ⛔ **blocked** | needs CP^(N−1) equations of motion — Gap 4 **#26**, open |
| 05 | 2D Ising | thermal Monte Carlo | ⬜ **not ported** (now possible) | `thermal_mc` exists (Gap 3 #6); needs an Ising-like proposal / strong easy axis |
| 06 | CP² skyrmions | non-equilibrium SU(3) quench | ⛔ **blocked** | same as 04 — Gap 4 **#26**, open |
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

## Why 04, 06 and the interesting half of 09 are not here

They are not oversights and they are not hard to *fake*; they are blocked on engine
capabilities that are open by choice:

- **04 and 06** need finite-temperature dynamics of SU(N) coherent states — the
  CP^(N−1) equations of motion, which are a different integrator from the
  Landau–Lifshitz one in `classical_dynamics.py`, not a wrapper on it (Gap 4 #26).
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

Tests: `tests/test_sunny_tutorials.py` (S01, S08, S09 + schema for all),
`tests/test_sun.py` (S03), `tests/test_ewald.py` (the engine S07 relies on).
