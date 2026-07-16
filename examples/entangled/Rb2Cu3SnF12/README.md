# Rb₂Cu₃SnF₁₂ — pinwheel valence-bond solid (single-dimer building block)

From **Matan et al., Nature Physics 6, 865 (2010)** — "Pinwheel valence-bond solid
and triplet excitations in the two-dimensional deformed kagome lattice" — and the
higher-resolution follow-up **Ehlers/Matan et al., Phys. Rev. B 89, 024414 (2014)**,
"Ghost modes and continuum scattering …", which resolved the two branches
(gaps **2.4 and 6.9 meV** at the zone centre) and determined their **polarizations**.

Rb₂Cu₃SnF₁₂ is an S=1/2 deformed-kagome antiferromagnet whose ground state is a
product of **singlet dimers** (the pinwheel VBS). Singlet dimers form on the
strongest bond J1 ≈ 18.6 meV, and the excitations are gapped **triplons** — exactly
the physics of `calculation.mode: entangled`.

## What this example does (exactly)

`config.yaml` models **one J1 dimer** with its out-of-plane Dzyaloshinskii–Moriya
interaction (d_z = D_z/J1 = 0.18) and a c-axis field. It reproduces the paper's
on-dimer physics analytically:

- a non-magnetic **singlet ground state** with a gapped triplon (dipole LSWT is
  blind to it — the singlet has no dipole moment);
- the **DM splits the triplet**: it raises the Stot^z = 0 branch to
  `Δ₀ = √(J1² + Dz²)` and leaves the Stot^z = ±1 branch degenerate at
  `Δ± = J1/2 + ½√(J1² + Dz²) < Δ₀` — the mechanism behind the paper's two gaps;
- a **c-axis field Zeeman-splits** the Stot^z = ±1 doublet (slope ∓2μ_B·B, g = 2)
  while Stot^z = 0 is field-independent — the 2010 paper's Fig. 4. Run
  `python field_sweep.py` to reproduce the gaps-vs-field plot (Fig. 4d) and read off
  g ≈ 2;
- the **mode polarizations** the 2014 follow-up determined: the Stot^z = ±1 doublet
  is **in-plane** (it carries S_xx, S_yy) and the Stot^z = 0 singlet is
  **out-of-plane** (only S_zz) — the entangled structure factor reproduces this exactly.

`tests/test_entangled_units.py` (`…_dimer_DM_and_field_mechanism`,
`…_triplet_polarizations`) pins all of the above.

## What it does NOT do (and why)

The **observed** excitations of Rb₂Cu₃SnF₁₂ are highly dispersive with small gaps
**Δ₁ = 2.35 and Δ₂ = 7.3 meV**. These are dispersion *minima*, far below the
J1 ≈ 18.6 meV dimer scale, produced by the **strong interdimer coupling**
(J2 = 0.95 J1, J3 = 0.85 J1, J4 = 0.55 J1) on the six-dimer deformed-kagome
**pinwheel**. Reproducing them needs both:

1. the full **12-spin pinwheel geometry** (6 dimers, the four bonds J1–J4 and the DM
   circulation), and
2. a **high-order dimer series expansion** — the paper used 8th order with
   Dlog-Padé, because J2 ≈ J1 is strong coupling.

The entangled engine works at the **harmonic bond-operator** level, which is exact
only in the *weak*-interdimer limit (see `../Cu5SbO6/`, where J2/J1 ≈ 0.36 and the
full dispersion is reproduced to machine precision). For Rb₂Cu₃SnF₁₂'s strong
coupling it would be only qualitative even with the full geometry. This example
therefore captures the **singlet-VBS + DM + field + polarization** structure of a
single pinwheel dimer — the papers' central mechanism — rather than the full
dispersion. The 2014 follow-up further reports **ghost modes** (from the 2a×2a,
48-spin enlarged cell below the 215 K structural transition) and an **8–10 meV
continuum** (a possible kagome-spinon remnant); both are outside any single-cell
dimer LSWT.

### The full pinwheel from the CIF — bond assignment by superexchange angle

Given the R-3 CIF (a = 13.877 Å, c = 20.239 Å, Cu1/Cu2 on 18f), the four in-plane
Cu–Cu bond families are 3.341, 3.497, 3.531, 3.565 Å — and **every one of them is a
perfect matching** (each Cu in exactly one bond of each family), so bond length alone
cannot identify the dimer bond. The physical criterion is the **Cu–F–Cu superexchange
angle** (J grows with angle in this family). Computing the bridging-fluorine angles
from the CIF gives:

| family | d (Å) | Cu–F–Cu angle | assignment |
|---|---|---|---|
| **J1 (dimer)** | 3.565 | **138.3°** | strongest — matches the papers' 138° |
| J2 = 0.95 J1 | 3.531 | 136.2° | |
| J3 = 0.85 J1 | 3.497 | 133.4° | |
| J4 = 0.55 J1 | 3.341 | **123.3°** | weakest — matches the papers' 124° |

(The end-point angles 138°/124° reproduce the 2010 paper exactly — a strong pin that
the geometry and assignment are right. Note the *longest* bond is the strongest: the
angle, not the distance, controls the exchange here.)

### The full dispersion: `series_dispersion.py` (linked-cluster + Dlog-Padé)

The harmonic bond-operator level is hopeless here (J2 = 0.95 J1 — strong coupling),
so `series_dispersion.py` computes the triplon dispersion with the **high-order dimer
series expansion** (`magcalc/sun/dimer_series.py`) — the same method as the papers:
linked-cluster expansion of the one-triplon effective Hamiltonian in the interdimer
couplings (out-of-plane DM d_z = 0.18 included, alternating pattern built from the
triangle circulation), resummed with Dlog-Padé.

At the Γ point (experiment: Δ₁ = 2.35/2.4 meV doublet, Δ₂ = 7.3/6.9 meV singlet),
the Stot^z = ±1 doublet series is
`18.75 − 10.11λ − 11.65λ² + 2.82λ³ + 2.80λ⁴ − 2.32λ⁵ − 0.51λ⁶` — strongly
oscillating, the strong-coupling signature that forced the papers to 8th order:

| Γ branch | order 4 | order 5 | order 6 | experiment |
|---|---|---|---|---|
| doublet (Dlog-Padé) | 2.2 ± 1.0 | 6.3 (wide) | 1.6 | **2.35 / 2.4** |
| singlet (Dlog-Padé) | 11.1 ± 0.8 | 10.2 | 9.9 ± 0.9 | **7.3 / 6.9** |

The doublet estimates bracket the measured gap with O(1 meV) scatter — the papers'
order-8 Dlog-Padé is what pins it at 2.35 — and the singlet descends monotonically
toward 7.3. A global sign flip of the DM pattern leaves every series coefficient
invariant (checked to 0.0), as the mirror symmetry requires.

Run `python series_dispersion.py [order]` (order 4 ≈ seconds, 5 ≈ a minute, 6 ≈ half
an hour). The engine itself is validated against exact diagonalization of the
alternating chain at strong coupling (`tests/test_dimer_series.py`) — the pinwheel
numbers inherit that trust; only the ORDER is the limitation, exactly as in the
papers.
