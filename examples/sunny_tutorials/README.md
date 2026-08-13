# Sunny.jl tutorials, ported to pyMagCalc

Ports of the official [Sunny.jl](https://github.com/SunnySuite/Sunny.jl) tutorial
series (`../../../Sunny.jl-main/examples/01..09`) to pyMagCalc config files.

**Status, audited 2026-08-04; rows 06 and 09 updated 2026-08-13.** All nine
tutorials are now ported to Sunny or to an exact analytic result, **one of them only
transitively justified (07)**. Nothing is blocked. Spelled out per row below rather
than summarised optimistically.

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
| 09 | Disordered triangular AFM | LSWT + KPM with bond disorder | ✅ **ported, exact analytic pins** | 120° order as a real-space √3×√3 cell at E/site = **−0.375 exactly**, bands = the analytic ω(q) to 1e-13, big-cell S(q,ω) = minimal-cell to 1e-9; disorder broadens +12.9 % (L = 12) / +13.5 % (Sunny's L = 30). `disorder_kpm.py`, `test_s09_disorder_kpm.py` |

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

## What unblocked 06 and 09 (2026-08-13)

Both were listed as blocked here for months, and **neither was blocked on the thing
this file said it was**. Porting either by quietly substituting a clean or
equilibrium calculation would have produced a folder that looks like a port and is
not one, so they stayed listed as blocked instead — which was right, but the recorded
reasons were wrong:

- **06** was said to need a damped quench. That machinery already existed; the blocker
  was PERFORMANCE. A skyrmion is several lattice constants across, so 64–256 sites
  relax to the uniform state — a size failure that reads as a physics failure. At
  Sunny's L = 40 the CP^(N−1) derivative cost ~16 s/step (≈55 h); vectorized, it is
  8.4 ms/step and the whole run takes 214 s.
- **09** was said to need bond disorder in LSWT (Gap 4 #16b). That had already landed
  too. The real blockers were the reference state — the 120° order had to be an
  explicit real-space supercell, since the SU(N)/KPM path does not consume the
  rotating-frame `single_k` form — and, underneath it, **a bug in `sun/kpm.py`** that
  made a clean non-collinear spectrum arrive pre-broadened, so adding disorder
  appeared to NARROW it. See that folder's README.

Both are in the table above now. The lesson worth carrying: a blocked row's stated
reason ages badly, because the capability it names usually lands without anyone
revisiting the row.

## Running them

```bash
magcalc run examples/sunny_tutorials/S01_CoRh2O4/config.yaml
```

Tests: `tests/test_sunny_tutorials.py` (S01, S02, S05, S08, S09 + schema for all),
`tests/test_s09_disorder_kpm.py` (S09's supercell + disorder half),
`tests/test_sun.py` (S03), `tests/test_ewald.py` (the engine S07 relies on).
