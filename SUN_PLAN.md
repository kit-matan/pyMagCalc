# SU(N) mode — implementation plan

Branch: `feature/sun-mode`. Nothing is implemented yet; this is the spec and, more
importantly, the **validation gate**.

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
