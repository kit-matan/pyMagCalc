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

6. **Zeeman calibration.** With `parameters: {H_mag: ..., H_dir: [...]}` the
   engine's magnon-branch splitting is `2*mu_B*H_mag` (verified against the
   exact LSWT result on SW29), i.e. **`H_mag` = B in Tesla reproduces the
   electron g=2 Zeeman** `g*mu_B*B`. The Zeeman is a global isotropic `H.S`.

7. **Spiral phases use FULL atomic positions.** pyMagCalc's rotating frame
   rotates by `2*pi*k.(r_j - r_i)` with the fractional atomic positions,
   while SpinW's `genmagstr` rotates its per-atom `S0` by cell translations
   only. Multi-site SpinW helices therefore port with back-rotated local
   directions `n_i = R(-2*pi*k.d_i, axis) . S0_i` (see SW23's generator).

8. **Symmetry model first** (see the repo-root `CLAUDE.md` for the full
   authoring rules). Atomic positions use `lattice_parameters` +
   `space_group` + `wyckoff_atoms` where the tutorial's setting allows it
   (SW05/06/07/09/13/23). Interactions use `symmetry_rules`: plain
   `distance` rules for scalar Heisenberg shells, and `ref_pair` rules
   (one reference bond, propagated by the spglib-detected group) for 3x3 /
   DM matrices (SW09, SW20, SW38) and for same-length but
   symmetry-inequivalent families (SW06's J3b, SW21's 48g a-a family).
   Explicit lists remain only where the coupling breaks the detected
   symmetry: magnetic-order-dependent J_eff (SW28), deliberately
   sub-symmetric models (SW16 Kitaev — SpinW itself sets 'sym' false
   there; SW36's matrix breaks the b=c lattice symmetry), dimerized
   same-length matrices from orbital order (SW14), and cases where the
   magnetic sublattice alone detects a HIGHER group that would merge
   inequivalent families (SW15's chirality-selecting J3/J5, SW18 —
   both conversions were tested and demonstrably change the spectrum).
   Every conversion was verified band-by-band against the explicit model.

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
| SW20 | Yb₂Ti₂O₇ (anisotropic pyrochlore) | ⚠️ ported, zero field only | Ross PRX 1, 021002 anisotropic exchange propagated to all 96 NN bonds with the Fd-3m ops (reference bond + allowed form taken from the tutorial's published getmatrix output). Classical minimization reproduces the splayed ⟨100⟩ ferromagnet ((±¼,±¼,0.935) sublattice directions). The tutorial's 5 T / 2 T spectra need an anisotropic g-tensor Zeeman (g_xy=4.32, g_z=1.8 in local ⟨111⟩ frames) — not supported, see feature gaps |
| SW21 | YIG (ferrimagnet garnet)       | ✅ ported, verified | 20-site primitive BCC cell (as the tutorial's newcell), Jad/Jdd distance rules + explicit Jaa on the 48g-midpoint a–a family only (bond 3 vs 4 — the tutorial's teaching point; the 16b variant was tested and is clearly wrong). Γ levels 0/6.4/7.5/9.7/9.8/13.0/16.3/25.6 THz match the published figure; upper optical branches within ~5% |
| SW22 | Easy-plane frustrated chain    | ✅ ported, verified | J1–J2 spiral (k=0.38497) + hard-axis SIA, rotating frame stays exact (SIA ∥ spiral axis); phason exactly gapless, out-of-plane gap at ±k = 0.7144 analytic vs 0.7155 engine, ω(½)=0.92191 matches analytic to 5 decimals |
| SW23 | Sr₃Fe₂O₇ (bilayer helix)       | ⚠️ ported | 4 Fe, 5 distance rules; helix k=(1/7,1/7,1) about [110] with the tutorial's S0 (per-site local directions back-rotated for pyMagCalc's full-position spiral convention). Tiny easy-axis D dropped (not rotation-invariant; SpinW drops it too). Like the tutorial (hermit=false), the helix is not the exact classical minimum: acoustic branch collapses at the (1∓1/7, 0.14) satellites with small imaginary parts truncated |
| SW26 | J1–J2 chain spiral             | ✅ ported, verified | k = arccos(−1/12)/2π = 0.263278; engine ≡ analytic helix (3.83108/0.95278/1.83527) to 5 decimals at every checked q |
| SW28 | Biquadratic FCC (MnO)          | ✅ ported, verified | biquadratic b(Si·Sj)² mapped exactly (at LSWT order) onto per-bond J_eff = J1 ± dJ on the parallel/antiparallel NN families, dJ = −Q/S (matches the paper convention; the naive 2bS² is the tutorial's noted "factor 2"). One supercell band ≡ the tutorial's analytic dispersion at every sampled q (diff ~1e-13). Same-length bonds with different J_eff ⇒ must stay an explicit (generated) list |
| SW29 | Easy-axis AFM chain in field   | ✅ ported, verified | Zeeman calibrated: engine splitting is 2·μB·H_mag ⇒ H_mag = B(T) gives the g=2 electron Zeeman. At 7 T < B_sf = 7.53 T the collinear AFM stays the ground state; branches ω₀(q)±0.8103 with ω₀(0)=√0.84=0.9165 — engine exact |
| SW35 | LuVO₃ + fitting                | ✅ ported, verified | G-type AFM (k=[0,1,1] ⇒ chemical cell), 2 distance rules + two-axis SIA. The fitspec part is ported with `magcalc fit` (config_fit.yaml): Jab=5.76(11), Jc=4.31(12), Kxx=0.66(6), Kyy=0.12(4) meV against the 17 measured modes (LuVO3_modes.txt); gaps 3.89/9.18 vs 3.92/8.90 meV exp.; the observed branch switches doublet across L=2.5 exactly as in the data |
| SW36 | Anisotropic-exchange FM chain  | ✅ ported, verified | J=diag(−3,−4,−5): gapped FM magnon, ω(0)=2√2, ω(½)=√288 — engine exact to 5 decimals |
| SW37 | Triangular AFM + resolution    | ✅ ported, verified | single-site 120° spiral k=(⅓,⅓,0), one distance rule; engine ≡ analytic at every checked q. Constant-FWHM S(Q,ω) broadening (0.3 meV) covers the tutorial's constant-dE panel; energy-dependent dE, dQ and Ei kinematic cuts are feature gaps |
| SW38 | Ca₂RuO₄ (T and T′ modes)       | ✅ ported, verified | S=1 checkerboard, NN matrix J=5.2/X=±1 (sign alternates between the two diagonals)/Jzz=4.68 + SIA e=1, E=21.5. Engine 38.14 (X–M flat), 12.85/44.33 at Γ — matches the published SpinW figure (~38 / ~13 / ~44) |

Not ported — nothing to port (SpinW-specific usage tutorials, no LSWT physics):

* **T24, T25, T27** do not exist in the SpinW 3.2.0 tutorial set.
* **T30** (help/documentation browsing), **T31** (spinw object internals),
  **T32** (symmetry analysis of allowed DM components on a bond — pyMagCalc
  has no symmetry-allowed-tensor analyzer; the *result* of such analysis is
  what SW09/SW20 encode), **T33** (swplot plotting demo on the Yb₂Ti₂O₇
  model of SW20), **T34** (Na₂IrO₃ symmetry analysis + bond plotting — the
  physics model itself is ported as SW16).

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
* **Distance `symmetry_rules` also only searched ±1 cell images** (SW22/26's
  2nd-neighbour chain bond at 2a and SW23's in-plane 3rd neighbour at (2,0,0)
  would silently vanish). `_expand_heisenberg_rules` now sizes the offset
  search per rule from the target distance and the cell's perpendicular
  heights (`ceil(d/h_i)`), which is inert for existing configs (extra cells
  cannot contain matches at distances the ±1 shell already covered —
  SW01/SW05 regression-checked identical).
* **`ref_pair` symmetry rules were unusable.** Three fixes in
  `config_builder.py`: (1) symmetry detection mapped spglib species by raw
  per-site labels, so any config with per-site labels (Fe0, Fe1, …) detected
  P1 — it now keys on `element`/`ion` (falling back to the digit-stripped
  label); (2) `add_symmetry_interaction` crashed on an unbound `best_dist`
  when the reference bond was given via `offset`; (3) the entry-dedup loops
  crashed (KeyError 'pair') when generic distance rules shared the
  interaction list. With these, one reference bond/matrix propagates over
  the detected group exactly like SpinW's `addcoupling` + `setmatrix`
  (validated: SW09 DM kagome, SW20 pyrochlore 96-bond orbit, SW21 48g
  family, SW38 alternating pseudo-dipolar X — all band-identical to the
  explicit lists).

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
* **No anisotropic g-tensor / per-site Zeeman.** The applied field is a
  single global `H.S` with an isotropic effective g. Tutorial 20's in-field
  Yb₂Ti₂O₇ spectra (g_xy=4.32 / g_z=1.8 in the local ⟨111⟩ frames, B∥[1-10])
  need `g_i·B` to differ between sublattices — SW20 is therefore ported at
  zero field only.
* **No biquadratic exchange term.** On collinear structures the exact
  LSWT-order workaround is the per-bond effective bilinear map used by SW28
  (`J_eff = J ± dJ` on parallel/antiparallel bonds); non-collinear
  biquadratic models have no such mapping.
* **Resolution convolution is constant-FWHM only.** `plotting.broadening_width`
  broadens the S(Q,ω) map with a fixed FWHM; sw_instrument's energy-dependent
  dE polynomial, dQ broadening and Ei kinematic limits (tutorials 21/23/37/38)
  are not implemented.

## Helper-generated configs

`SW06`, `SW09`, and `SW12` have many explicit bonds / sites and were emitted by
small Python generators (kept inline in the session that produced them). The
committed `config.yaml` files are self-contained and need no generator to run.
