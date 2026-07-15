# S03 — Multi-flavor (SU(3)) spin waves of FeI₂

Port of Sunny tutorial `03_LSWT_SU3_FeI2.jl`.

FeI₂ is an effective spin-1 material with strong easy-axis single-ion anisotropy
`D = 2.165 meV`. Quadrupolar fluctuations produce a single-ion **bound state**
that dipole LSWT structurally cannot represent — you need the LSWT of SU(3)
coherent states (2-flavor bosons, `N = 2S+1 = 3`). This is exactly what
pyMagCalc's `calculation: {mode: SUN}` provides.

## Runnable, validated config

The FeI₂ SU(3) model is already shipped and validated as a material example:

```bash
magcalc run examples/materials/FeI2/config_fei2_sun.yaml
```

It uses the same anisotropic exchange matrices and `D` as the Sunny tutorial, the
non-diagonal magnetic supercell `[1 0 0; 0 1 -2; 0 1 2]` for the k = (0, −1/4, 1/4)
two-up/two-down spiral, and the CP^(N−1) ground-state search (`tasks:
{minimization: true}`) — the SU(N) ground state must be found in SU(N), never
seeded from the dipole state.

**Validation:** band-by-band **and** in intensity against Sunny 0.8.1 `:SUN` to
< 1e-4 — E/site = −2.91893118, all 8 bands (`tests/test_sun.py`). Note Sunny's own
published FeI₂ example converges to a *local* minimum (−2.35592338); the value
above is the converged ground state.

## What is not ported here

The Sunny tutorial also shows `domain_average` over the three 120°-rotated spiral
domains. pyMagCalc has domain averaging for `cross_section: perp|trace`, but not
yet in SU(N) mode (see `GAP_STATUS.md`, SU(N) "not yet: powder/domain averaging").
The single-crystal SU(3) bands — the physics the tutorial exists to demonstrate —
are fully reproduced.
