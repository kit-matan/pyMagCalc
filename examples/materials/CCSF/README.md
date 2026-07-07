# Cs₂Cu₃SnF₁₂ — distorted spin-½ kagome antiferromagnet

Linear spin-wave theory (dispersion **and** single-crystal S(Q, ω)) for
Cs₂Cu₃SnF₁₂ in its **corrected** magnetic structure: monoclinic *P*2₁/*n*,
coplanar 120°, **negative** vector chirality.

## Run

```bash
magcalc run examples/CCSF/config_ccsf.yaml
```

Outputs (energies in meV):

| output | contents |
| ------ | -------- |
| `CCSF_disp.png` | spin-wave dispersion (6 branches) |
| `CCSF_sqw.png`  | single-crystal S(Q, ω) intensity map (Cu²⁺ form factor) |

## Web app / symmetry-driven variant

`config_ccsf_symmetry.yaml` is the same physics rebuilt for the web app
(pyMagCalc Studio, `./start_magcalc.sh` → **Import YAML**). Instead of
listing all 6 spins and 104 interaction entries explicitly, it gives the
true crystallographic cell — the 10 K *P*2₁/*n* SuperHRPD refinement in the
standard *P*2₁/*c* (b1) setting that the builder loads for space group 14 —
with only the **two inequivalent Cu sites** (`Cu1`, `Cu2`) and **15 symmetry
rules** (3 NN exchanges + 3 DM vectors + 9 J₂ orbits). The space-group
operators generate all spin positions and propagate every bond and DM
vector (axial transform `D′ = det(R)·R·D`).

It was generated and validated by `make_symmetry_config.py`:

```bash
python examples/CCSF/make_symmetry_config.py --run
```

which maps the legacy orthorhombic kagome-frame model onto the monoclinic
frame, checks that the symmetry expansion reproduces all 80 Heisenberg and
24 DM directed bonds exactly, and confirms the dispersions of the two
configs agree to machine precision (max |ΔE| = 0.000000 meV; the legacy
kagome-frame path Γ–X–M–Y–Γ maps to (0,0,0)–(0,½,0)–(0,½,½)–(0,0,½)–(0,0,0)
in the monoclinic cell, whose in-plane reciprocal axes are b\* and c\*).

## The model

Because the low-temperature structure is monoclinic, the three
nearest-neighbour bonds of each triangle are inequivalent, so the magnetic
cell contains **six** Cu sublattices with three exchanges *J*₁₁, *J*₁₂, *J*₁₃:

```text
H = Σ_<ij> [ J1(ij) Si·Sj + J1(ij) D_ij·(Si×Sj) ]  +  Σ_<<lk>> J2 Sl·Sk
```

The DM vectors are `D_ij = (0, Dy, −Dz)` and its symmetry partners, expressed
as **ratios of the local exchange** (so the DM energy is `J1(ij)·D`). The spin
model and this bond/DM pattern live in `spin_model.py`; the 120° negative-
chirality ground state (obtained once by energy minimization, E = −19.61 meV)
is baked into `spin_model.mpr()`, so no re-minimization is needed per run.

Parameter order in `hamiltonian_params`: `[J11, J12, J13, J2, Dy, Dz, H]`.

| parameter | value | meaning |
| --------- | ----- | ------- |
| `J11` | 6.89 meV | nearest-neighbour exchange, bond 1 |
| `J12` | 8.14 meV | nearest-neighbour exchange, bond 2 |
| `J13` | 11.94 meV | nearest-neighbour exchange, bond 3 |
| `J2`  | +0.21 meV | next-nearest-neighbour exchange |
| `Dy`  | 0.15 (×J) | in-plane DM ratio |
| `Dz`  | −0.262 (×J) | out-of-plane DM ratio (\|Dz\|·J̄ ≈ 2.4 meV) |
| `H`   | 0 | applied field (meV, Zeeman) |

These are the **variant-B refit** of the AMATERAS inelastic-neutron data to the
negative-chirality structure (mean exchange J̄ ≈ 9 meV). See the companion
technical report `20230319/CCSF/CCSF_P21n_G1neg_rework2026/` for the fit and its
provenance.

## What to look for

The six-branch spectrum reproduces the measured features:

* a **low-energy dispersive magnon** rising from the zone centre, gapped by
  **0.66 meV** (from the monoclinic exchange anisotropy);
* a weakly dispersive **flat (weathervane) mode** near **8.5 meV**;
* **weakly dispersive upper branches** at ~9.1, 9.3, 11.8, and 13.0 meV at the
  zone centre — the modes reported between 9 and 14 meV. Whether these are
  folded magnons (as here) or two-spinon bound states is discussed in the
  technical report.

Zone-centre energies (calc.): 0.66, 8.52, 9.10, 9.33, 11.77, 13.04 meV.

## Why this structure (and not all-in–all-out)

The 2019 powder refinement first assigned the all-in–all-out (positive-chirality,
*P*2₁′/*n*′) structure. The corrected coherent re-refinement of that data
(2026 Erratum) instead favours **negative** vector chirality in *P*2₁/*n*, which
is what this example uses. At the simplified uniform-exchange (R‑3m) level the
negative-chirality state is *gapless* (the in-plane DM cannot open a gap there);
the measured 0.7 meV gap comes from the exchange anisotropy J₁₃/J₁₁ ≈ 1.7, which
is why the full monoclinic six-sublattice model is required.

## Fitting

`spin_model.py` is compatible with pyMagCalc's `MagCalc` API, so the exchanges
can be refit to measured dispersions. Because the DM stabilizes the structure,
fit the exchanges (`J11, J12, J13, J2`) with the DM fixed, and wrap the fit in a
short outer loop that re-minimizes/recompiles between refits (the structure is
held fixed within a single fit). The self-consistent refit engine used to obtain
the parameters above is `fast_refit.py` in
`20230319/CCSF/CCSF_P21n_G1neg_rework2026/`; see also **TUTORIAL.md §4b** for the
built-in dispersion-fitting workflow and the compile-once fast evaluator.

## References

1. K. Matan *et al.*, Phys. Rev. B **99**, 224404 (2019) — magnetic structure
   (and 2026 Erratum: *P*2₁/*n* negative vector chirality).
2. K. Matan, T. Ono, S. Ohira-Kawamura, K. Nakajima, Y. Nambu, and T. J. Sato,
   Phys. Rev. B **105**, 134403 (2022) — spin dynamics.
3. T. Ono *et al.*, Phys. Rev. B **79**, 174407 (2009) — crystal structure.
