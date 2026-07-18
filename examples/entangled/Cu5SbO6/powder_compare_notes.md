# How pyMagCalc computes the map in `powder_compare.png`

The right-hand panel of [`powder_compare.png`](powder_compare.png) is produced by
[`powder_compare.py`](powder_compare.py) from pyMagCalc's **entangled-unit
(bond-operator) spin-wave engine** — not ordinary dipole LSWT. The chain, following
`powder_compare.py` → [`config.yaml`](config.yaml):

## 1. Model — a lattice of singlet dimers
`calculation.mode: entangled` with `units: [[CuA, CuB]]` treats each Cu²⁺–Cu²⁺ pair
as one effective site whose 4-dimensional Hilbert space (singlet + triplet) is
diagonalized exactly. The intradimer **J₁** becomes the on-site term (reference = the
dimer **singlet**, which has zero dipole moment), and the interdimer **J₂** (chain)
and **J₄** (interlayer) disperse the excitation.
Code: `EntangledCalculator` in `magcalc/sun/entangled.py`.

## 2. Dispersion — harmonic bond-operator (Sachdev–Bhatt) triplon
Diagonalizing the one-triplon Bogoliubov problem gives 3 degenerate modes,

    ω(q) = sqrt( J₁² − J₁·J₂·cos 2π q₁ − J₁·J₄·cos 2π (q₃ − q₁) ),

the square-root resummation of the paper's first-order formula. The two agree to first
order in J₂/J₁, J₄/J₁; beyond that pyMagCalc additionally captures the RPA-level
band-edge correction (band bottom ≈ 11.1 meV vs the linear estimate's 12 meV).

## 3. Intensity — `calculate_sqw`, `cross_section: perp`
The neutron structure factor uses the **staggered** dimer moment
Σₖ exp(i q·dₖ) Sₖ  (dₖ = the two Cu offsets within a dimer), which produces the dimer
selection rule (intensity → 0 at low |Q|) and the 1 − cos(q·d) modulation. No
`temperature` is set, so this is the T = 0 spectrum (no Bose factor).

## 4. Powder average — `powder_sample_modes` (Fibonacci sphere)
For each experimental |Q| shell, `calculate_sqw` is evaluated at **1500 directions**
on the sphere, and *each direction keeps its own mode energy and intensity* (the SpinW
`powspec` convention). This builds the true band shape and van Hove peaks instead of
collapsing the band to a blob at J₁.

## 5. Experiment layer — form factor and broadening

The Cu²⁺ magnetic form factor F(Q)² sets the |Q| intensity falloff. **pyMagCalc applies
it internally**: `calculate_sqw` → `lswt.structure_factor` multiplies by
`get_form_factor("Cu2+", |Q|)²` (Int-Tables ⟨j0⟩) because the config sets `ion: Cu2+`,
alongside the staggered dimer (1 − cos q·d) factor. So F(Q)² must **not** be applied
again in the script — doing so double-counts it and over-steepens the pyMagCalc falloff.
For an apples-to-apples comparison, `powder_compare.py` divides out pyMagCalc's built-in
form factor and reapplies the manuscript's Cu²⁺ parameterization, so both panels carry
F(Q)² exactly once with the **same** coefficients; only the dispersion (√-resummed vs
first-order) then differs. Both maps are convolved with a Lorentzian of HWHM
**ε = J₁/40**, then (for display only) regridded onto one common (|Q|, E) mesh and passed
through the **same** Gaussian filter (σ = 1.6).

## Summary
entangled-dimer harmonic LSWT → S(Q,ω) with the staggered-dimer cross-section →
Fibonacci-sphere powder average → magnetic form factor + Lorentzian broadening.

The only *physics* difference from the manuscript panel is that pyMagCalc carries the
√-resummation (and its RPA band-edge correction), whereas
[`analytic/powderINS_sim.py`](analytic/powderINS_sim.py) evaluates the first-order dimer
expansion analytically.

Reference: Piyakulworawat *et al.*, Phys. Rev. Research **8**, 013247 (2026).
