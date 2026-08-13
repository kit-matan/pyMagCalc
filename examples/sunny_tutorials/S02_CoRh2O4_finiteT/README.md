# S02 — CoRh₂O₄ at finite temperature

Port of Sunny tutorial `02_LLD_CoRh2O4.jl`.

> **This file previously said OUT OF SCOPE.** That was correct when written —
> pyMagCalc was a pure LSWT engine. Finite-temperature classical dynamics
> (Gap Tier 2 #5) and instantaneous correlations (Gap 4 #19) have since landed, so
> the tutorial is now portable and is ported below.

## What the tutorial does

The **same** CoRh₂O₄ Hamiltonian as [`../S01_CoRh2O4/`](../S01_CoRh2O4/) — Néel
diamond antiferromagnet, `J = 0.63` meV on the nearest-neighbour bond — but at
**T = 16 K** instead of T = 0. Sunny thermalizes a 10×10×10 supercell with Langevin
dynamics and then measures three things:

1. the **instantaneous** structure factor (`SampledCorrelationsStatic`);
2. the **dynamical** `S(q,ω)` along a q-path (`SampledCorrelations`, with the
   classical-to-quantum `kT` correction);
3. a **powder average** of (2).

## What is ported (`config.yaml`)

Parts 1 and 2:

```yaml
tasks: {static_correlations: true, sampled_correlations: true}
```

- `static_correlations` is pyMagCalc's `SampledCorrelationsStatic`: the equal-time
  correlation ⟨|S(q)|²⟩ averaged over thermally sampled configurations, per site.
- `sampled_correlations` is the real-time Landau–Lifshitz `S(q,ω)`, with
  `classical_to_quantum: true` applying the same `|ω/kT| / (1 − e^{−ω/kT})` factor
  Sunny applies when you pass it `kT` (Gap 4 #17).

**Part 3 is not wired.** pyMagCalc's `powder_average` task drives the *LSWT* engine;
there is no spherical average over the classical `S(q,ω)` yet. Nothing conceptually
hard, just not built.

## Two deliberate differences from Sunny

**Thermalization: Metropolis, not Langevin.** Sunny uses its Langevin integrator;
these tasks use Metropolis. Both sample the same Boltzmann distribution, and each is
pinned *independently* to exact results — the free-spin Langevin function and the
exact classical dimer ⟨E⟩(T) (`tests/test_thermal_mc.py`, `tests/test_integrators.py`)
— so equilibrium averages agree even though the trajectories do not. If you want
Sunny's exact route, `classical_dynamics.langevin_step` is the thermostat (Gap 4 #18).

**Supercell: 4×4×4, not 10×10×10.** This sampler stores the classical energy as a
**dense quadratic form** `½ mᵀH m + bᵀm`, not a bond list, so cost grows with the
*square* of the site count:

| supercell | spins | H | memory |
|---|---|---|---|
| 4×4×4 | 512 | 1536² | 19 MB |
| 6×6×6 | 1728 | 5184² | 215 MB |
| 10×10×10 (Sunny's) | 8000 | 24000² | **4.6 GB** |

Raise it if you need finer q resolution and have the memory — but check the cost
first. An example that needs gigabytes is not a useful example.

## Units

`T = 16 K = 16 × 0.0861733 = 1.3788 meV`. pyMagCalc's classical modules take `kT` in
**meV**; passing 16 directly would silently simulate ~186 K, which looks like a
perfectly ordinary result.

## Validation

`tests/test_sunny_tutorials.py::test_S02_static_correlations_peak_at_the_antiferromagnetic_wavevector`
— CoRh₂O₄ orders Néel, so the instantaneous `S(q)` must carry more weight at the
ordering wavevector than at a generic zone-interior point, and the contrast must
**sharpen on cooling** (checked at 46 K vs the tutorial's 16 K).

The underlying machinery carries its own exact pins: the free-spin sum rule
`n_atoms·2S²/3` (perp) and `n_atoms·S²` (trace) at every q and every T for the static
estimator, and Sunny's own `c2q` formula plus detailed balance for the dynamic one.

**Absolute intensities ARE now on the LSWT/Sunny scale** (2026-08-13; this entry used
to say the opposite). Both classical estimators are normalized per chemical cell with
the 1/2π of the time transform, pinned by the equal-time sum rule
`∫dω S(q,ω) = ⟨S(q)*S(q)⟩/n_cells` at machine precision and, on a gapped low-T
ferromagnet, against the LSWT band sum to ~2 % — see
`tests/test_classical_absolute_normalization.py`. One caveat remains and it is a
lineshape one, not a scale one: no time-domain window is applied, so integrating the
`c2q`-corrected spectrum over the *whole* frequency axis picks up ~16 % of leakage
(`OPEN_WORK.md`). Integrate over the feature you care about.

## Running it

```bash
magcalc run examples/sunny_tutorials/S02_CoRh2O4_finiteT/config.yaml
```

This is diffuse scattering from a thermally disordered state, **not** a magnon
spectrum. On cooling it sharpens toward the coherent S01 dispersion.
