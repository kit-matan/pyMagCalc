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
- **FeI2 still not reproduced — but now isolated to ONE thing.**

  Sunny (`:SUN`, 4-site cell): `E/site = -2.35592338`.

  What is now VERIFIED for the FeI2 setup:
  * bond orbits from pyMagCalc's spglib `ref_pair` propagation match Sunny's
    `print_symmetry_table` exactly: coordinations 6 / 2 / 6 / 12 / 6 / 6 (including that
    d = 9.7366 splits into TWO classes of 6, with J'2a on only one -- the J'2a/J'2b
    split that the earlier hand-rolled group got wrong);
  * the J matrices are self-consistent: `sum(J_zz)` over the 38 bonds equals the
    analytic value exactly, and `trace(J)` is constant within each orbit;
  * the NON-DIAGONAL supercell replication (`SUNModel._replicate`) is correct: the
    ferromagnetic energy matches the analytic value to machine precision, and every one
    of the 152 bonds satisfies the fold invariant
    `pos[J] - pos[I] - dr in the supercell lattice`.

  What is still WRONG: evaluating Sunny's own ground state in this model gives -2.917,
  and NONE of the 16 stripe sign patterns reproduces -2.35592338.

  **The fault must be in how the anisotropic exchange matrix is TRANSFORMED onto the
  symmetry-propagated bonds** (`R J R^T`). Every check above is invariant under the
  point group -- `sum(J_zz)`, `trace(J)`, coordination, and the ferromagnetic energy all
  stay the same under a wrong rotation -- which is exactly why they passed. The stripe
  energy, by contrast, depends on the in-plane and off-diagonal parts (J1yz = -0.261).

  Next: compare pyMagCalc's propagated J1 matrices bond-by-bond against Sunny's
  (`Sunny.exchange_matrix` / the printed bond table). A likely culprit is a differing
  Cartesian convention for the trigonal a2 axis (+120 vs -120 degrees), which would
  leave every invariant intact while rotating the off-diagonal parts wrongly. NOTE this
  would be a bug in the DIPOLE engine's propagation too -- it would affect any
  anisotropic-exchange model in a trigonal setting.

- Runner integration (`mode: SUN`) and SU(N) intensities.


## Why

Dipole LSWT expands each spin as a single boson about a classical direction. That
structurally cannot represent **single-ion (multipolar) excitations** — transitions
between the local crystal-field levels of a large-S ion. The canonical case is
**FeI₂** (S=1, strong easy-axis anisotropy), whose single-ion bound state is visible in
INS and is entirely missing from a dipole-mode calculation. `examples/materials/FeI2/`
is already in the repo and is currently wrong in exactly this way.

Sunny solves this with a second engine (`HamiltonianSUN.jl`), not by extending the
dipole one. We should expect the same.

## What it is

Each site carries an N-level local Hilbert space (N = 2S+1). The reference state is a
**coherent state** |Z_i⟩ ∈ CP^{N-1} (an N-component complex vector), not a direction on
S². Excitations are N−1 boson flavours per site (the N−1 orthogonal local levels), so
the dynamical matrix is 2(N−1)L × 2(N−1)L rather than 2L × 2L.

## Work items

1. **Local operator algebra** (`magcalc/sun/operators.py`)
   - Spin matrices S^x, S^y, S^z in the spin-S representation.
   - Stevens operators as **matrices** O_k^q (we currently generate only the classical
     polynomials in `magcalc/stevens.py` — those are the large-s limit and are NOT what
     is needed here).
   - Cross-check: `stevens_matrices(s)` in Sunny for the same s.

2. **General couplings** — SU(N)'s value comes precisely from terms dipole LSWT cannot
   represent, so the bond coupling must accept a general operator sum
   (Sunny's `set_pair_coupling!`), not just a 3×3 bilinear. Biquadratic and
   general on-site anisotropy must decompose into it.

3. **Ground state on CP^{N-1}** (`magcalc/sun/groundstate.py`)
   - Energy and gradient for coherent states.
   - The existing annealer (`magcalc/annealing.py`) optimises directions on S²; it
     needs generalising to CP^{N-1}. Keep the same interface so the ground-state
     **guards** (`stability_report`, `relax_from_current`) still apply.

4. **SU(N) LSWT** (`magcalc/sun/lswt.py`)
   - Boson expansion about |Z_i⟩; build H(q) in the same Nambu convention the host
     uses. NOTE: pyMagCalc stores **g·H₂** (metric folded in) — verified in
     `magcalc/core.py::_ewald_nambu`. Match it.

5. **Intensities** — observables are the spin operators as N×N matrices; the neutron
   cross-section contracts the same way, so `contract_cross_section` should be reusable.

## Validation gate — do not merge without this

**Primary:** FeI₂ vs Sunny, mode by mode.
- The single-ion bound state must appear at the right energy and with the right
  intensity. It is absent in dipole mode, so this is a sharp, falsifiable target.
- Compare against `:SUN` mode in Sunny, not `:dipole`.

**Secondary (cheap, catches convention errors):**
- For a model with **no** single-ion terms, SU(N) must reproduce dipole-mode LSWT
  exactly. Any disagreement is a convention bug, not physics.
- S=1/2 (N=2): SU(N) is *identical* to dipole LSWT. This must hold to machine precision
  and is the fastest possible smoke test.
- Sum rules on the total spectral weight.

## Known trap

This is the one place where a subtly wrong implementation produces **plausible-looking**
spectra — the failure class this codebase has repeatedly been bitten by (the S-power
filter silently deleting quartic terms; the classical energy optimising a different
Hamiltonian than LSWT diagonalised; the mixed-spin prefactor). Every one of those was
caught by an oracle or an exact identity, never by inspection. Hold the same line here:
**Sunny agreement before shipping, not "it runs".**
