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

2. **k≠0 order: propagation vector, or an explicit magnetic cell.** pyMagCalc
   supports both of SpinW/Sunny's routes, and new configs should use them
   (see the repo-root `CLAUDE.md`):
   * `magnetic_structure: {type: single_k, k: …, axis: …}` — the rotating-frame
     propagation vector (SpinW `genmagstr` 'helical' / Sunny `set_spiral!`).
     Exact at incommensurate k, and the only right choice for a true spiral.
   * `crystal_structure.magnetic_supercell: [n1, n2, n3]` (or `'auto'` to derive
     the minimal cell from a commensurate k) — the SpinW `nExt` / Sunny
     `resize_supercell` analogue. You declare the **chemical** cell and the
     engine replicates it, remaps interactions/SIA, and turns a `single_k`
     structure into the real-space commensurate pattern.

   The older ports (**SW02** Néel chain 2×1×1, **SW04/SW10/SW11** Néel square
   2×2, **SW12** 120° triangular 3×3) predate both features and hand-write the
   replicated cell as explicit `atoms_uc` + a `pattern` structure. That is
   porting history, not an engine limitation — do not copy the style into new
   configs. SW03 shows the modern form side by side: `config_spiral.yaml`
   (rotating-frame `single_k`) and `config_supercell_auto.yaml`
   (`magnetic_supercell: auto` → a 13-site cell). k=0 orders (FM, 120° kagome)
   need neither.

3. **q-path RLU follows how the cell was built.** With a hand-written magnetic
   supercell (SW02/04/10/11/12) the q-path is in RLU of the **magnetic** cell:
   enlarging by `n` along an axis multiplies the chemical-cell RLU by `n`, and
   each such config's comments spell out the chemical→magnetic mapping. With
   `magnetic_supercell` the q-path stays in **chemical** RLU (bands fold), and
   S(Q,ω) is normalized per chemical cell — the Sunny/SpinW convention. Either
   way the `q_path` is chosen so the **physical** (Å⁻¹) range matches the
   reference figure.

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
   electron g=2 Zeeman** `g*mu_B*B`. By default the Zeeman is a global
   isotropic `H.S`; declaring a per-site `g` on any atom switches to
   `mu_B * B . g_i . S_i` (anisotropic / sublattice-dependent — see
   `CLAUDE.md` §5b). An isotropic `g: 2` reproduces the default exactly.

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
| SW10 | Energy cut on square lattice   | ✅ ported, verified | 2-D constant-E cut is a first-class runner task (`tasks.energy_cut` + `energy_cut:` block in `config.yaml`) — rings of scattering around the AFM points, as in the tutorial. `sw10_energy_cut.py` kept as the hand-rolled reference; the task output is cleaner (no degenerate-point speckle) |
| SW11 | La₂CuO₄                        | ✅ ported, verified | raw LSWT ×1.18 ≈ 316 meV zone-boundary magnon |
| SW12 | Triangular, easy-plane SIA     | ✅ ported, verified | one gapless in-plane Goldstone + out-of-plane gap ∝√D (required two pyMagCalc bug fixes, see below) |
| SW13 | LiNiPO₄                        | ✅ ported | SG62 (4 Ni), symmetry-expanded bonds (Jbc/Jb/Jc/Jab/Jac by bond character), two-axis SIA; Cz AFM along c, ~2 meV spin gap, ~7 meV bandwidth |
| SW14 | YVO₃                           | ✅ ported | P1, 2×2×1 supercell, anisotropic 3×3 c-axis matrices (Jc1/Jc2 with DM) + −K1 Sx²; canted C-type AFM, ~11 meV bandwidth |
| SW15 | Ba₃NbFe₃Si₂O₁₄ (spiral)        | ✅ ported, verified | P321, 3 Fe, symmetry-propagated bonds; rotating-frame spiral. Chirality ϵT=−1 gives the energy minimum at kz=0.14275 (matches the tutorial's 0.1426); ϵT=+1 → kz=0 |
| SW16 | Na₂IrO₃ Kitaev honeycomb       | ✅ ported, verified | C2/m honeycomb (4 Ir), bond-dependent Kitaev + Heisenberg J1/J2/J3 as per-bond `interaction_matrix`. Validated: stripy E/site=−0.29125 (tutorial −0.2913, spins‖c); zig-zag (Fig S3 i-j) is the ground state ‖b. Not in the Sunny port set |
| SW17 | Symbolic FM chain              | ✅ ported, verified | pyMagCalc builds the LSWT Hamiltonian in SymPy; `sw17_symbolic.py` reads off E(q)=2S\|J\|(1−cos qa) from `HMat_sym`, matching SpinW's symbolic result. Not in the Sunny port set |
| SW18 | Distorted kagome (spiral)      | ✅ ported, verified | C2/m, 6 Cu, symmetry-propagated bonds; rotating-frame spiral at k=[0.7859,0,0.1070]. Bond identities fixed by matching the classical minimum: k_x=0.7873 vs 0.785902, E/site=−0.7838 vs −0.78338 |
| SW19 | Different magnetic ions        | ✅ ported, verified | mixed-spin support added: Cu(½) band ~1.4 meV, Fe(2) band ~4 meV on distinct energy scales |
| SW20 | Yb₂Ti₂O₇ (anisotropic pyrochlore) | ✅ ported, verified (zero field **and** in field) | Ross PRX 1, 021002 anisotropic exchange propagated to all 96 NN bonds. Zero field: splayed ⟨100⟩ ferromagnet. **In field** (`config_field.yaml`, B = 5 T ∥ [1-10]): the anisotropic g-tensor (g_xy=4.32/g_z=1.8, uniaxial about each site's own local ⟨111⟩) is now expressible, giving a fully gapped spectrum (gap 0.413, bands to 1.109 meV) with no imaginary modes. Needs `num_starts: 200` + `early_stopping: 40` — the 32-angle landscape has local minima that silently produce imaginary magnons (see the config header) |
| SW21 | YIG (ferrimagnet garnet)       | ✅ ported, verified | 20-site primitive BCC cell (as the tutorial's newcell), Jad/Jdd distance rules + explicit Jaa on the 48g-midpoint a–a family only (bond 3 vs 4 — the tutorial's teaching point; the 16b variant was tested and is clearly wrong). Γ levels 0/6.4/7.5/9.7/9.8/13.0/16.3/25.6 THz match the published figure; upper optical branches within ~5% |
| SW22 | Easy-plane frustrated chain    | ✅ ported, verified | J1–J2 spiral (k=0.38497) + hard-axis SIA, rotating frame stays exact (SIA ∥ spiral axis); phason exactly gapless, out-of-plane gap at ±k = 0.7144 analytic vs 0.7155 engine, ω(½)=0.92191 matches analytic to 5 decimals |
| SW23 | Sr₃Fe₂O₇ (bilayer helix)       | ⚠️ ported | 4 Fe, 5 distance rules; helix k=(1/7,1/7,1) about [110] with the tutorial's S0 (per-site local directions back-rotated for pyMagCalc's full-position spiral convention). Tiny easy-axis D dropped (not rotation-invariant; SpinW drops it too). Like the tutorial (hermit=false), the helix is not the exact classical minimum: acoustic branch collapses at the (1∓1/7, 0.14) satellites with small imaginary parts truncated |
| SW26 | J1–J2 chain spiral             | ✅ ported, verified | k = arccos(−1/12)/2π = 0.263278; engine ≡ analytic helix (3.83108/0.95278/1.83527) to 5 decimals at every checked q |
| SW28 | Biquadratic FCC (MnO)          | ✅ ported, verified | biquadratic b(Si·Sj)² mapped exactly (at LSWT order) onto per-bond J_eff = J1 ± dJ on the parallel/antiparallel NN families, dJ = −Q/S (matches the paper convention; the naive 2bS² is the tutorial's noted "factor 2"). One supercell band ≡ the tutorial's analytic dispersion at every sampled q (diff ~1e-13). Same-length bonds with different J_eff ⇒ must stay an explicit (generated) list |
| SW29 | Easy-axis AFM chain in field   | ✅ ported, verified | Zeeman calibrated: engine splitting is 2·μB·H_mag ⇒ H_mag = B(T) gives the g=2 electron Zeeman. At 7 T < B_sf = 7.53 T the collinear AFM stays the ground state; branches ω₀(q)±0.8103 with ω₀(0)=√0.84=0.9165 — engine exact |
| SW35 | LuVO₃ + fitting                | ✅ ported, verified | G-type AFM (k=[0,1,1] ⇒ chemical cell), 2 distance rules + two-axis SIA. The fitspec part is ported with `magcalc fit` (config_fit.yaml): Jab=5.76(11), Jc=4.31(12), Kxx=0.66(6), Kyy=0.12(4) meV against the 17 measured modes (LuVO3_modes.txt); gaps 3.89/9.18 vs 3.92/8.90 meV exp.; the observed branch switches doublet across L=2.5 exactly as in the data |
| SW36 | Anisotropic-exchange FM chain  | ✅ ported, verified | J=diag(−3,−4,−5): gapped FM magnon, ω(0)=2√2, ω(½)=√288 — engine exact to 5 decimals |
| SW37 | Triangular AFM + resolution    | ✅ ported, verified | single-site 120° spiral k=(⅓,⅓,0), one distance rule; engine ≡ analytic at every checked q. S(Q,ω) uses `plotting.resolution.de_fwhm` with the cubic fitted to the tutorial's table EN=[0..4]/dE=[0.05 0.05 0.05 0.3 0.4] (= sw_instrument 'polDeg' 3): sharp branches below ~1.5 meV, broad above, as in the tutorial. Its third mode (arbitrary-function dE) needs the Python API |
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
* **Symmetry rules could fail silently.** Found by auditing this gap list, not
  by a port. Three related holes, all of which let a run "succeed" with a term
  missing from H — a physically wrong spectrum that looks like a good result:
  (1) a non-scalar `symmetry_rules` entry given as a bare `distance` rule
  (`{type: dm, distance: 2.5, …}`, no `ref_pair`) expanded to **zero bonds** —
  `add_symmetry_interaction` subscripted `ref_pair[0]` with `ref_pair=None` and
  the `TypeError` was swallowed by a blanket `except Exception: logger.warning`
  in `generic_model.py`; (2) a distance rule that matched **no** bonds (a typo'd
  bond length) was likewise dropped without complaint; (3) only
  `_expand_heisenberg_rules` auto-sized its cell-image search — the DM / matrix /
  anisotropic expanders and the `ref_pair` reference-bond lookup were hard-coded
  to `product([-1,0,1], repeat=3)`, so any such rule beyond one cell image found
  nothing. All three now **raise** with a message naming the offending rule, the
  offset shell is sized from the target distance everywhere (shared
  `_offset_shell`), and a DM rule at 2c is found where the ±1 shell returned
  zero. Inert on every existing config: the expanded bond list of every
  `examples/**/*.yaml` is byte-identical before and after (all `ref_pair` ports
  sit at `n=[1,1,1]`, i.e. the old ±1 shell), and the 62-test suite passes.
* **Spiral S(Q,ω) had only the central branch.** The rotating-frame spiral gave
  the correct *dispersion* at ω(q), but the neutron cross-section also carries
  weight at ω(q±k), which was missing. Single-k structures now compute the
  satellites: `satellites: true` (in `magnetic_structure` or `tasks`) yields
  `3·nspins` modes, channel-major `[q−k | q | q+k]`, and S(Q,ω) uses the
  Toth & Lake three-channel projection for the correct satellite intensities.
  Default: **on** for S(Q,ω), **off** for dispersion (so an energy-only
  comparison still shows the clean central branches). Validated against Sunny's
  `SpinWaveTheorySpiral`.
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

## Intensity / experiment layer (added 2026-07, closes former gaps)

S(Q,ω), powder and energy-cut intensities now model the measurement, with
everything driven from the config (validated in `tests/test_intensity_layer.py`
— exact identities against analytic detailed balance, manual twin averages and
tensor sum rules):

* **Thermal (Bose) factor** — `calculation: {temperature: <K>}` multiplies
  every mode by |1/(1−exp(−E/kT))| (Sunny `kT` / SpinW `sw_egrid` 'T'),
  applied per (q, mode) before any powder averaging.
* **Domain / twin averaging** — `calculation.domains`, either the shorthand
  `{axis: [0,0,1], n_fold: 3}` or an explicit list of `{axis, angle, weight}`
  (the list is the complete set — include angle 0). SpinW `addtwin` analogue;
  exact for the perp/trace cross-sections (the projector transforms
  covariantly), rejected for lab-frame components. Modes concatenate
  domain-major; powder skips domains (a spherical average is
  rotation-invariant). Dispersion/fitting stay single-domain.
* **Cross-section selection** — `calculation.cross_section: perp | trace |
  xx | yy | zz | xy | ...` (Sunny ssf_perp / ssf_trace / ssf_custom
  component analogue). Off-diagonal components are signed (real part,
  unclamped).
* **Instrument resolution** — `plotting.resolution` with `de_fwhm` (scalar or
  polyval-convention polynomial FWHM(E), = SpinW `sw_instrument` 'dE'
  polyfit; see SW37), `shape` (gaussian/lorentzian), `dq_fwhm` (Gaussian |Q|
  smoothing), `ei` + `two_theta` (direct-geometry kinematic masking of powder
  maps); plus `plotting.energy_grid_step` for the map's energy grid.
* **2-D q-grid constant-energy cuts** — `tasks: {energy_cut: true}` with an
  `energy_cut:` block (`origin`, `axis1/axis2: {vec, points}`, `cuts:` list of
  `{center, fwhm}` Gaussian windows and/or `{band: [lo, hi]}` integrations).
  Writes `energy_cut_data.npz` + a panel figure; replaces SW10's companion
  script (kept as a reference).

## Hamiltonian terms (added 2026-07, closes the former "Gap 2")

All validated in `tests/test_hamiltonian_terms.py` against exact identities or
Sunny — never a golden number. See the repo-root `CLAUDE.md` §5b for the config
surface.

* **Anisotropic / per-site g-tensor** — `g` on the atom: scalar, `[gxx,gyy,gzz]`,
  a 3×3, or `{g_par, g_perp, axis}` (uniaxial about a **local** axis, which is what
  rare-earth pyrochlores need). Zeeman is `μ_B B·g_i·S_i`; an isotropic `g: 2`
  reduces *exactly* to the legacy calibrated term.
* **Full 3×3 single-ion anisotropy** (`sia_matrix`) and **Stevens operators**
  (`stevens`, `B_k^q O_k^q`, k = 2/4/6, all q). The O_k^q are the classical
  large-s polynomials generated mechanically from Sunny's `stevens_matrices(Inf)`
  (`magcalc/stevens.py`) — not hand-transcribed. Rare-earth crystal fields are now
  expressible.
* **Biquadratic exchange** (`biquadratic`) — the genuine `B (S_i·S_j)²` operator,
  so it is valid for **non-collinear** structures, unlike SW28's collinear
  `J_eff = J ± dJ` workaround. On a collinear state it reduces exactly to that map
  (`J_eff = J + 2Bσ S²`, verified to 1e-15).
* **Long-range dipole-dipole** (`dipole_dipole: {cutoff: …}`) — expanded to explicit
  3×3 bond matrices, so the rotating frame / supercell / symmetry checks all apply.
  Matches Sunny's `modify_exchange_with_truncated_dipole_dipole!` to **3e-8**.
  It is a *truncated* real-space sum (not Ewald): the dipolar sum is only
  conditionally convergent, so raise the cutoff until your answer stops moving.
* **Multi-k structures** (`type: multi_k`) — real-space superposition
  `S_i = Σ_m m_m cos(2π k_m·r_i + φ_m)` on a supercell derived by per-axis **LCM**
  over all k (`magnetic_supercell: auto`). A one-component k=½ multi_k reproduces
  the hand-built Néel chain exactly. There is no rotating-frame multi-k theory —
  SpinW and Sunny also require a supercell — so all k must be commensurate.

### Two engine bugs this uncovered (both silent, both fixed)

* **The LSWT truncation deleted every quartic term.** `_prepare_hamiltonian`
  filtered by *powers of S* (`coeff(S,1)·S + coeff(S,2)·S²`). The quadratic-boson
  part of a quartic operator (biquadratic, Stevens O_4/O_6) carries **S³**, so it
  was silently annihilated — a biquadratic or O_4^0 term would have been accepted
  and then contributed *nothing*. Truncation is now by **boson degree**, the
  physically correct LSWT criterion. Inert on every existing config (all their
  quadratic terms are S¹ / S⁰-param-linear): dispersions are byte-identical.
* **The classical energy used a different Hamiltonian than LSWT.** The g-tensor,
  and the field *direction*, were only honoured on the symbolic-parameter path.
  With numeric parameters — which is exactly what the **minimizer** uses — `H_dir`
  failed to resolve and the code fell through to a branch that ignored `g` and
  assumed **B ∥ z**. The minimizer therefore optimized a different Hamiltonian than
  LSWT diagonalized, giving a wrong ground state and imaginary magnon energies. The
  field vector is now resolved once, and the g-tensor applies on every path.



## Ground-state search (`method: anneal`)

Prefer **Monte-Carlo annealing** over the legacy random-multistart gradient descent —
it is both more reliable and cheaper:

```yaml
minimization: {enabled: true, method: anneal, num_starts: 4, n_sweeps: 2000, seed: 0}
```

* `anneal` (= `monte_carlo`) — Metropolis with geometric cooling (SpinW `anneal`;
  Sunny `LocalSampler`'s uniform/flip/delta proposal mix), finished with an L-BFGS
  polish so the result is a true stationary point. Crosses barriers.
* `steep` (= `optmagsteep`) — iterative alignment of each spin with its local field.
  Very fast, but **monotone**: it cannot escape a local minimum, so it is a polisher,
  not a global search.
* `L-BFGS-B` etc. — the legacy multistart path, kept for compatibility.

On **SW20 in field** (16 sites = 32 angles; true minimum −5.716074 meV) the legacy
path is startlingly weak — it optimizes in (θ, φ), whose polar coordinate
singularities hurt it badly:

| method | budget | result |
|---|---|---|
| L-BFGS-B | 24 starts, early_stopping 10 | **−5.338112 — WRONG**, imaginary modes at every q |
| L-BFGS-B | 200 starts, early_stopping 40 | −5.716074, reached by only **3 of 200** starts |
| **anneal** | **1 run × 500 sweeps** | **−5.716074, ~0.8 s** |
| anneal | 4 runs × 2000 sweeps | −5.716074, **4/4 runs**, reproducible across seeds |

Cross-checked against the verified multistart ground state on SW04, SW07, SW09,
SW12, SW20 and CCSF: anneal and steep agree to <1e-5 meV everywhere, and anneal is
never worse. Both SW20 configs now use `method: anneal`.

## Ground-state guards (added 2026-07, after SW20 shipped a wrong ground state)

LSWT is an expansion about a classical energy **minimum**; about anything else the
spectrum is meaningless. Until now that failure was survivable — the engine logged
per-q warnings, *discarded the imaginary part*, and completed the run, writing a
plausible-looking plot. Two independent guards now run before any task, and a
failure is a hard error (`calculation.on_imaginary: error` by default; `warn`/`off`
to relax):

1. **Imaginary-energy check** — a non-minimum with anomalous terms gives imaginary
   magnons (the SW20-in-field class).
2. **Energy audit** — perturb the structure and relax; if the energy drops, it was
   not a minimum. This catches what check 1 provably *cannot*: a stationary
   **maximum** (e.g. a ferromagnetic pattern given for an antiferromagnet) leaves the
   Bogoliubov problem diagonal, so the ± ω pairs are sorted and the upper half
   returned — the spectrum comes back real, positive and completely plausible.
   Neither guard alone is sufficient.

The imaginary check thresholds on **|Im ω| relative to the bandwidth** (default 0.5%),
not on an absolute meV value: an absolute cutoff cannot separate a real instability from
numerical noise across models whose energy scales differ by orders of magnitude. The
noise is worst exactly where it matters — at the ω ≈ 0 Goldstone modes, where the
Bogoliubov problem is singular and the minimizer's residual error leaks into the
imaginary part. SW07's 120° kagome carries ~1e-3 meV of such noise on a 2.4 meV band
(0.05% — ignored), while SW23's genuine instability is 1.5 meV on a 59 meV band (2.5% —
flagged). It also tests **the q-path you are about to plot**, not just random q: SW23's
collapse is confined to the magnetic satellites and a random sample misses it entirely
(reporting 1e-14).

The minimizer additionally reports `hits` (how many starts reached the best energy)
and warns when `hits == 1`, and `early_stopping` now defaults to `max(10, 2 × n_sites)`
instead of a flat 10.

### Why SW03 / SW18 / SW23 have no minimum (and why that is *not* an engine bug)

All three impose the **reference's** magnetic structure, which is not the classical
minimum of *this* Hamiltonian. They fail in two physically distinct ways:

* **SW03 — a commensurate approximant to an incommensurate spiral.** The true ground
  state is a helix at k = 0.23006; a 13-site supercell can only host k = n/13 = 0.23077.
  The shipped structure *is* the exact minimum **within** the 13-site cell (annealing
  confirms it, 6/6 runs, and cannot do better) — but it is not the minimum of the
  infinite chain: the exact rotating-frame treatment gives −2.062500 meV/site versus
  −2.062420 for the approximant. LSWT is right to report the residual instability
  toward the true k. **The exact route already exists**: `config_spiral.yaml` is stable
  (|Im| ~ 5e-17). This is the cost of the approximation, not a defect.
* **SW18 — the reference's k, not this model's.** The config uses the tutorial's
  k = [0.7859, 0, 0.1071]; this model's own classical minimum is at k_x = 0.7873
  (E/site −0.7838 vs −0.78338 — the status table already recorded the discrepancy).
  Expanding about the reference k leaves a small unstable acoustic mode (1.8% of the
  bandwidth). Relaxing k (`minimization` with `optimize_k: true`) removes it.
* **SW23 — a knowingly non-minimal helix.** The tutorial itself uses `hermit=false`.
  Annealing finds a state 5.8e-4 meV lower (3e-6 relative) that is *fully stable*
  (|Im| ~ 1e-14), so the instability here **is** removable — the port keeps the
  published helix on purpose.

**A structural limitation this exposes.** Both the annealer and the energy audit
optimize only *within* the magnetic cell — i.e. q = 0 of the supercell. Neither can see
an instability at q ≠ 0 (a long-wavelength re-winding spanning many cells), which is
exactly SW03's case: the energy guard correctly reports "this is a minimum", because it
*is* one, within the cell. Only the imaginary-mode check sees such instabilities. That
is why both guards are needed, and why neither is redundant.

**Turning the guards on immediately caught a real, shipped bug:** SW20's *zero-field*
`config.yaml` — marked "✅ verified" — carried a hand-written splayed-FM structure that
is **not** the ground state (E = −1.355 vs −2.220 meV, imaginary modes at 0.21 meV;
its four sublattice directions did not match the local ⟨111⟩ axes). It now minimizes
properly and is stable. Three configs are knowingly non-minimal and are marked
`on_imaginary: warn` with a reason in the header: **SW03** `config.yaml` /
`config_supercell_auto.yaml` (a commensurate 13-site approximation to an
incommensurate spiral — use `config_spiral.yaml` for the exact, stable treatment) and
**SW23** (the tutorial itself uses `hermit=false`).

## Remaining pyMagCalc feature gaps

* **Intensities still use the reference spin S.** Mixed-spin *dispersions* are
  correct; the S(Q,ω) intensity prefactor uses the single reference `S`, so
  relative intensities in a mixed-spin model are approximate (per-ion form
  factors are still applied).
* ~~Ground-state search is random multistart only.~~ **Closed** — see
  "Ground-state search" below: `method: anneal` (SpinW `anneal` / Sunny
  `LocalSampler`) and `method: steep` (SpinW `optmagsteep`) are implemented.
* ~~Dipolar sum is truncated, not Ewald.~~ **Closed** — `dipole_dipole: {method: ewald}`
  performs the full Ewald summation (real + reciprocal + surface/demagnetisation term),
  matching Sunny's `enable_dipole_dipole!` to 1e-8. The truncated variant remains
  available and is verified to converge to the Ewald result as the cutoff grows.
  Not yet supported together with a single-k rotating-frame structure (raises).

## Helper-generated configs

`SW06`, `SW09`, and `SW12` have many explicit bonds / sites and were emitted by
small Python generators (kept inline in the session that produced them). The
committed `config.yaml` files are self-contained and need no generator to run.
