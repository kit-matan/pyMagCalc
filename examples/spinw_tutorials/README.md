# SpinW tutorials ported to pyMagCalc

This folder ports the [SpinW tutorials](https://spinw.org/tutorials/) (via their
[Sunny.jl](https://sunnysuite.github.io/Sunny.jl/) re-implementations, the
`SW*.jl` files here) to **pyMagCalc** linear-spin-wave-theory (LSWT) input.

Each port lives in its own sub-directory as a declarative `config.yaml` that you
run with:

```bash
# from the pyMagCalc repo root
python -m magcalc run examples/spinw_tutorials/SW01_FM_chain/config.yaml
```

(or `magcalc run …` if the `magcalc` console script is on your PATH). Each run
writes its dispersion / powder PNG next to the config.

---

## Conventions used in every port

pyMagCalc and SpinW/Sunny differ in a few bookkeeping conventions; these are the
rules the ports follow so the numbers match:

1. **Exchange energy counts ordered pairs with a 1/2.** pyMagCalc evaluates
   `H = (1/2) Σ_ij J_ij S_i·S_j` over *ordered* pairs, so **every bond must be
   present in both directions** (`i→j` *and* `j→i`). Distance-based
   `symmetry_rules` do this automatically (each rule expands to both
   orderings); explicit `heisenberg`/`interaction_matrix` pairs are listed in
   both directions by hand. This calibration is exact: SW01 reproduces the
   FM-chain zone-boundary energy `4S|J| = 4 meV` on the nose.

2. **No propagation vectors — explicit magnetic cells.** SpinW/Sunny describe
   k≠0 order with a propagation vector on the *chemical* cell. pyMagCalc instead
   works with an **explicit magnetic unit cell** where every site carries a spin
   direction. So Néel / doubled orders are built as real supercells:
   * a Néel chain → 2×1×1 cell (SW02),
   * a Néel square → 2×2 cell (SW04, SW10, SW11),
   * a 120° triangular order → 3×3 cell (SW12).
   k=0 orders (FM, 120° kagome) need no supercell.

3. **q-paths are in RLU of the (magnetic) cell.** When the cell is enlarged by
   `n` along an axis, the chemical-cell RLU is multiplied by `n`. Each config's
   `q_path` is chosen so the **physical** (Å⁻¹) range matches the reference
   figure; comments spell out the chemical→magnetic RLU mapping.

4. **Form factors.** pyMagCalc ships a small ion table (Cu2+, Fe2+, Fe3+, Ni2+,
   V2+/3+/4+, …). Where Sunny uses an ion pyMagCalc lacks (e.g. Cu1+), the
   nearest available ion is used; this only affects S(Q,ω)/powder *intensities*,
   never the magnon energies.

5. **No classical-to-quantum anisotropy renormalization.** Sunny's `:dipole`
   mode rescales quadratic single-ion anisotropy by `(1-1/2s)`; pyMagCalc does
   not. Ports use the physical (un-renormalized) value directly.

---

## Status table

| # | Tutorial | Status | Notes |
|---|----------|--------|-------|
| SW01 | FM Heisenberg chain            | ✅ ported, verified | zone boundary = 4S\|J\| = 4 meV |
| SW02 | AFM Heisenberg chain           | ✅ ported, verified | folded double-hump, gapless at H=0,½,1; max 2 meV |
| SW03 | Frustrated J1–J2 chain (spiral)| ✅ ported, verified | exact rotating-frame spiral at k=0.23006 (`config_spiral.yaml`, validated to ~1e-12 vs the analytic helix dispersion); also a commensurate 13-site supercell (`config.yaml`) |
| SW04 | Frustrated square lattice      | ✅ ported, verified | Néel 2×2; Goldstone cones at Γ and zone boundary |
| SW05 | Simple kagome FM               | ✅ ported, verified | flat band at top (6 meV) + gapless acoustic |
| SW06 | Complex kagome FM              | ✅ ported, verified | J3a/J3b split at equal distance handled by midpoint classification |
| SW07 | k=0 kagome AFM (120°)          | ✅ ported, verified | 120° order; J2 lifts the lower band |
| SW08 | √3×√3 kagome AFM (spiral)      | ✅ ported, verified | rotating-frame spiral on the 3-site cell (`config.yaml`, 3 clean branches) cross-checked band-by-band (~1e-8) against the exact 9-site supercell (`config_supercell.yaml`) |
| SW09 | k=0 kagome AFM + DM            | ✅ ported, verified | DM entered as per-bond matrices with **consistent triangle circulation** (= the P-3-propagated DM); flat band lifted to ~0.97 meV as in the tutorial |
| SW10 | Energy cut on square lattice   | ✅ ported | model in `config.yaml`; the 2-D constant-E cut is produced by `sw10_energy_cut.py` (S(Q,ω) on an (H,K) grid) — rings of scattering around the AFM points, as in the tutorial |
| SW11 | La₂CuO₄                        | ✅ ported, verified | raw LSWT ×1.18 ≈ 316 meV zone-boundary magnon |
| SW12 | Triangular, easy-plane SIA     | ✅ ported, verified | one gapless in-plane Goldstone + out-of-plane gap ∝√D (required two pyMagCalc bug fixes, see below) |
| SW13 | LiNiPO₄                        | ✅ ported | SG62 (4 Ni), symmetry-expanded bonds (Jbc/Jb/Jc/Jab/Jac by bond character), two-axis SIA; Cz AFM along c, ~2 meV spin gap, ~7 meV bandwidth |
| SW14 | YVO₃                           | ✅ ported | P1, 2×2×1 supercell, anisotropic 3×3 c-axis matrices (Jc1/Jc2 with DM) + −K1 Sx²; canted C-type AFM, ~11 meV bandwidth |
| SW15 | Ba₃NbFe₃Si₂O₁₄ (spiral)        | ✅ ported, verified | P321, 3 Fe, symmetry-propagated bonds; rotating-frame spiral. Chirality ϵT=−1 gives the energy minimum at kz=0.14275 (matches the tutorial's 0.1426); ϵT=+1 → kz=0 |
| SW16 | Na₂IrO₃ Kitaev honeycomb       | ✅ ported, verified | C2/m honeycomb (4 Ir), bond-dependent Kitaev + Heisenberg J1/J2/J3 as per-bond `interaction_matrix`. Validated: stripy E/site=−0.29125 (tutorial −0.2913, spins‖c); zig-zag (Fig S3 i-j) is the ground state ‖b. Not in the Sunny port set |
| SW17 | Symbolic FM chain              | ✅ ported, verified | pyMagCalc builds the LSWT Hamiltonian in SymPy; `sw17_symbolic.py` reads off E(q)=2S\|J\|(1−cos qa) from `HMat_sym`, matching SpinW's symbolic result. Not in the Sunny port set |
| SW18 | Distorted kagome (spiral)      | ✅ ported, verified | C2/m, 6 Cu, symmetry-propagated bonds; rotating-frame spiral at k=[0.7859,0,0.1070]. Bond identities fixed by matching the classical minimum: k_x=0.7873 vs 0.785902, E/site=−0.7838 vs −0.78338 |
| SW19 | Different magnetic ions        | ✅ ported, verified | mixed-spin support added: Cu(½) band ~1.4 meV, Fe(2) band ~4 meV on distinct energy scales |

Legend: ✅ ported and physically checked against the reference · ⚠️ runs but with
a documented discrepancy/limitation · ⏳ not yet ported (reason given).

---

## pyMagCalc engine fixes made while porting these tutorials

Several discrepancies were traced to real pyMagCalc bugs and **fixed in the
engine** (`magcalc/symbolic.py`, `magcalc/generic_model.py`). Pure-exchange
examples (CCSF, KFe3J) are byte-identical before/after (verified), so the fixes
are inert except where they were needed:

* **Mixed spin magnitudes (SW19).** The LSWT layer now scales each site's
  Holstein–Primakoff expansion by its own `spin_S` (site *i* uses
  `S_i = (spin_S_i / spin_S_0) · S`, a numeric ratio times the single spin
  symbol, so the S-power filtering and numerical binding are untouched).
  `GenericSpinModel.spin_magnitudes()` exposes the per-atom list.
* **Single-ion anisotropy value ignored (SW12, FeI2).** `_compute_sia_terms`
  only honoured SIA strengths given as *named parameters*; a literal
  `value: 0.2` fell through to consuming an exchange parameter positionally.
  It now honours numeric literals and parameter expressions, and applies each
  SIA entry only to the sites in its `atoms` field.
* **On-site anomalous terms dropped (SW12).** The Fourier lookup provided no
  substitution for on-site `c²`/`c†²`, so the anomalous part of a single-ion
  anisotropy was silently discarded — spuriously gapping Goldstone modes (even
  a collinear easy-plane ferromagnet). The diagonal substitutions now include
  the on-site `c²`, `c†²`, and `c c†` forms, restoring the exact gapless
  Goldstone.
* **Spiral (rotating-frame) local structure ignored (SW03/08/15/18).**
  `generate_magnetic_structure` did not handle `type: spiral`, so the spins
  defaulted along the rotation axis instead of ⊥ to it — building the LSWT on a
  non-ground state and gapping the phason. `type: spiral` now returns local-frame
  directions perpendicular to `axis` (optionally a per-site `local_directions`
  pattern, e.g. a 120° triangle). The rotating-frame spiral is now **exact**:
  validated to ~1e-12 against the analytic helix (SW03) and band-by-band against
  an explicit supercell (SW08).
* **Bonds beyond the ±1 neighbour shell were silently dropped.** The
  original-unit-cell neighbour list only spanned ±1 cell per axis, so any bond
  with `|offset| > 1` (2nd-neighbour chains, FeI2's J3 at [2,0,0] and J′₂ₐ at
  [1,2,1], the spiral chains) simply never matched and vanished from the
  Hamiltonian. The shell now **auto-extends** to cover every explicit bond
  offset (FeI2 now correctly includes J3/J′₂ₐ; CCSF/KFe3J, which use only ±1
  offsets, are unchanged).

## Remaining pyMagCalc feature gaps

* **No symmetry propagation of anisotropic exchange matrices.** Distance
  `symmetry_rules` propagate only scalar Heisenberg `J`. Full 3×3 exchange /
  DM matrices must be listed per bond (SW09/SW14 do this; the spiral material
  ports SW13/SW15/SW18 generate the full bond list via the space-group
  operations in their helper scripts) — a `dm` distance rule that auto-orients
  bonds would be a nice addition.
* **No 2-D constant-energy-cut task in the declarative runner.** SW10's cut is
  produced by a companion script (`calculate_sqw` on a q-grid); a first-class
  grid/constant-E task would remove the need for the script.
* **Intensities still use the reference spin S.** Mixed-spin *dispersions* are
  correct; the S(Q,ω) intensity prefactor uses the single reference `S`, so
  relative intensities in a mixed-spin model are approximate (per-ion form
  factors are still applied).
* **Spiral S(Q,ω) shows one branch.** The rotating-frame spiral gives the
  correct *dispersion* of the central branch; the full neutron cross-section
  also has weight at ω(q±k). For a dispersion/energy comparison this is exactly
  right; for absolute S(Q,ω) intensities, overlay the ±k-shifted branches.

## Helper-generated configs

`SW06`, `SW09`, and `SW12` have many explicit bonds / sites and were emitted by
small Python generators (kept inline in the session that produced them). The
committed `config.yaml` files are self-contained and need no generator to run.
