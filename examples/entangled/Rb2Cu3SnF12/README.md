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

To build the full 6-band dispersion here one would need the explicit **12-Cu
deformed-kagome coordinates and the J1–J4 bond assignments** (from the
crystallography, Morita 2008 / Ono 2009) — then `mode: entangled` with 6 units
would give the six triplon bands, to be compared against the papers' 8th-order
Dlog-Padé dispersion with the strong-coupling caveat above.
