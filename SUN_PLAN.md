# SU(N) mode — implementation plan

Branch: `feature/sun-mode`.

**STATUS: the engine is implemented and all three validation gates pass.**

| gate | result |
|---|---|
| 1. S=1/2 (N=2) must be IDENTICAL to dipole LSWT | **exact, 0.0e+00** (FM and Neel chains) |
| 2. no single-ion terms: dipole bands reproduced for any S | **exact**; extras are the flat Dm>=2 multipolar modes, at k x the local exchange field |
| 3. single-ion anisotropy vs Sunny `:SUN`, mode by mode | **4.7e-07**, incl. the quadrupolar band dipole mode misses entirely; classical energy matches exactly |

Implemented: `magcalc/sun/operators.py` (N x N spin + Stevens matrices, coherent
states), `magcalc/sun/lswt.py` (`SUNModel`: H(q), dispersion, classical energy).
Tests: `tests/test_sun.py`.

### Done since
- **Config bridge**: `SUNModel.from_generic_model(model)` reuses the whole existing
  front end (structure, space-group propagation of exchange matrices, supercells) and
  swaps only the engine. Reproduces Sunny on the GATE 3 model to 4.7e-07.
- **CP^(N-1) ground state**: `SUNModel.minimize_energy` -- self-consistent local-field
  diagonalisation (the SU(N) analogue of `optmagsteep`) with random restarts. This is a
  genuine requirement, not a nicety: with anisotropy a coherent state has
  `<Sz^2> != (S n_z)^2`, so the SU(N) and dipole classical energies **differ** for a
  canted structure. Seeding SU(N) with the dipole ground state would simply be wrong.

### Still to do
- ~~FeI2~~ **DONE — FeI2 in SU(3) reproduces Sunny exactly.**

  | quantity | pyMagCalc SU(N) | Sunny `:SUN` |
  |---|---|---|
  | E/site | **-2.91893118** | **-2.91893118** |
  | 8 bands at 3 q-points | match to **< 1e-4 meV** | |

  Built from the ordinary config: pyMagCalc's spglib `ref_pair` propagation supplies the
  exchange orbits (verified against `print_symmetry_table`: 6/2/6/12/6/6, including the
  J'2a-vs-J'2b split at 9.7366 A), and only the NON-DIAGONAL magnetic cell
  `[1 0 0; 0 1 -2; 0 1 2]` is applied on top (`SUNModel._replicate`).

  **The long "discrepancy" was in the REFERENCE, not the code.** Sunny's published FeI2
  example converges to a LOCAL minimum (`E/site = -2.35592338`): it does a single
  `minimize_energy!` after `randomize_spins!`. Given 60 restarts Sunny reaches
  `-2.91893118` — exactly our value, to 8 decimals. Every "bug hunt" against the
  published number was chasing a non-converged reference.

  That is the ground-state trap this codebase keeps falling into, and it caught me here
  too. The lesson generalises: **a published reference number is not automatically a
  converged one.** The tests now pin the CONVERGED values and say so.

- ~~Runner integration~~ **DONE**: `calculation: {mode: SUN}`. Everything else
  (structure, symmetry propagation, q-path, tasks, plotting, the ground-state guards)
  is unchanged. Non-diagonal `magnetic_supercell: {matrix: [[...]]}` supported here and
  refused in dipole mode. Example: `examples/materials/FeI2/config_fei2_sun.yaml`.
- ~~SU(N) intensities~~ **DONE**: `SUNModel.structure_factor` (Colpa Bogoliubov +
  one-magnon matrix elements of the full N x N spin operators). Matches Sunny's
  `ssf_perp` on FeI2 to **4e-07** -- including the SINGLE-ION bands, which carry real
  weight (0.16, 0.31) and which dipole LSWT cannot produce at all. Normalised per site,
  as Sunny's ssf is.

### Remaining
- SU(N) powder averaging and domain averaging.
- SU(N) fitting (the dipole `FitProblem` path is unchanged).
