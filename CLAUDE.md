# pyMagCalc — config authoring rules

When writing or editing a `config.yaml` for `magcalc run`, always prefer the
**symmetry model** over explicit listings, and always use the **compact YAML
form**. The `examples/spinw_tutorials/` ports are the reference
implementations of these rules.

## 1. Atomic positions: Wyckoff + space group

Prefer `lattice_parameters` (with `space_group`) plus `wyckoff_atoms` — one
representative per Wyckoff site; the engine expands the orbit with spglib:

```yaml
crystal_structure:
  lattice_parameters: {a: 6.0, b: 6.0, c: 8.0, alpha: 90.0, beta: 90.0, gamma: 120.0,
                       space_group: 147}
  wyckoff_atoms:
  - {label: K, pos: [0.5, 0.0, 0.0], spin_S: 1.0, ion: Cu2+}   # 3e orbit -> K0,K1,K2
```

Expanded labels are `<label>0..N` in orbit order. Fall back to explicit
`lattice_vectors` + `atoms_uc` only when the symmetry model cannot express the
structure, and say why in a comment:

- non-standard space-group settings (SW35 LuVO3);
- non-standard bases such as a primitive cell of a centred group (SW21 YIG),
  where the database ops (conventional basis) do not apply.

For magnetic supercells, do NOT hand-write the replicated cell (the old
SW02/04/10/11/12/14/28 style): declare the CHEMICAL cell and add
`magnetic_supercell: [n1, n2, n3]` (or `'auto'`) under `crystal_structure` —
see §3. Symmetry expansion runs on the chemical cell first, so
`wyckoff_atoms`/`symmetry_rules` still work.

## 2. Interactions: `symmetry_rules`

Inside `interactions:`, use `symmetry_rules` instead of explicit bond lists.
Two rule kinds:

```yaml
interactions:
  symmetry_rules:
  # (a) distance rule: scalar Heisenberg on EVERY bond of that length
  - {type: heisenberg, distance: 3.0, value: J1}
  # (b) ref_pair rule: one reference bond, propagated by the detected
  #     space group (J' = R J R^T; DM parts transform as axial vectors).
  #     Use for 3x3 matrices / DM, and for same-length but
  #     symmetry-INEQUIVALENT bond families that need different values.
  - type: interaction_matrix
    ref_pair: [Yb0, Yb4]
    offset: [0, 0, 0]
    value:
    - [-0.22, 0.01, 0.01]
    - [-0.01, -0.09, -0.29]
    - [-0.01, -0.29, -0.09]
```

Rules expand to **both bond directions** automatically (required by
pyMagCalc's `H = (1/2) Σ_ordered`); never list reverse bonds by hand alongside
rules. Distance rules auto-size their cell-image search from the target
distance, so 2nd-neighbour (2a) bonds are found.

Keep explicit bond lists ONLY when the coupling genuinely breaks the detected
crystal symmetry, with a comment saying so:

- couplings that depend on the magnetic order (SW28's biquadratic-derived
  J_eff differs on parallel vs antiparallel bonds of one orbit);
- deliberately sub-symmetric models (SW16 Kitaev — SpinW itself disables
  symmetry there; SW36's anisotropic matrix breaks the lattice's y/z symmetry);
- when spglib detects a HIGHER group than the physical one because only the
  magnetic sublattice is listed, and that extra symmetry would merge
  inequivalent families (SW15 langasite J3/J5 chirality pair, SW18 distorted
  kagome). Always verify a rule conversion band-by-band before keeping it.

`single_ion_anisotropy` entries stay as they are (`value` may be a number or
a parameter name; `axis`; `atoms` list).

## 3. Magnetic structure: manual (explicit)

The spin structure is always given manually — it is physics input, not
derivable from crystal symmetry:

- collinear / k=0 patterns: `type: pattern` with `pattern_type:
  ferromagnetic` (+ `direction`) or `generic` (+ per-site `directions`);
- incommensurate / propagation-vector: `type: single_k` with `k` (RLU),
  `axis` (rotation axis, Cartesian), and spin directions given as ONE of
  `local_directions` (rotating frame), `S0` (lab-frame cell-0 directions,
  SpinW/Sunny convention — the engine back-rotates them per site), or a
  `u`/`v` basis. `type: spiral` is a deprecated alias (same fields).

Single-k extras (validated against Sunny `SpinWaveTheorySpiral` and SpinW):

- `satellites: true` (in `magnetic_structure` or `tasks`) adds the ω(q±k)
  branches: dispersion/S(Q,w) then have `3·nspins` modes, channel-major
  `[q−k | q | q+k]`. Default: on for S(Q,w), off for dispersion. S(Q,w) uses
  the Toth & Lake three-channel projection (correct satellite intensities).
- `minimization: {enabled: true, optimize_k: true}` optimizes (k, spin
  directions) — Sunny `minimize_spiral_energy!` analogue — with a
  Luttinger-Tisza initial guess (`lt_guess`, `k_grid`), optional
  `optimize_axis: true`, and writes `optimized_structure.yaml`.
- The engine warns when the Hamiltonian is not rotationally invariant about
  `axis` (DM ∦ axis, SIA axis ∦ axis, field ∦ axis) — the rotating-frame
  method is unreliable then (`enforce_rotational_symmetry: warn|error|off`).
- `crystal_structure.magnetic_supercell: [n1, n2, n3]` (or `'auto'` to derive
  the minimal cell from a commensurate k) — SpinW `nExt` / Sunny
  `resize_supercell` analogue. The chemical cell is replicated (cell-major,
  replicas labelled `<label>@i_j_l`), interactions/SIA are remapped, and a
  `single_k` structure becomes the real-space commensurate pattern (replicas
  rotated by `R(2π k·c, axis)`, Sunny's `repeat_periodically_as_spiral`).
  q_path stays in CHEMICAL RLU (bands fold); S(Q,w) is normalized per
  chemical cell (Sunny/SpinW convention). Use for collinear k=1/2-type or
  multi-k states; prefer the rotating-frame `single_k` for true spirals
  (exact at incommensurate k, no ghost bands). Reference:
  SW03 `config_supercell_auto.yaml`.

The order of `directions`/`local_directions` follows `atoms_uc` /
Wyckoff-orbit order — after switching to `wyckoff_atoms`, re-verify the
spectrum to catch ordering mismatches. pyMagCalc's spiral phases use FULL
atomic positions (`2π k·(r_j−r_i)`); SpinW `S0` values are back-rotated
per site `n_i = R(−2π k·d_i, axis)·S0_i` — automatic with the `S0` field.

## 4. Compact YAML form — always

Vectors and matrix rows are written in flow style on one line; mappings stay
block style:

```yaml
  - {label: Ni, pos: [0.25, 0.25, 0.0], spin_S: 1.0, ion: Ni2+}
```

Generator scripts must emit through
`examples/spinw_tutorials/_compact_yaml.py` (`dump(cfg, f)`), never plain
`yaml.safe_dump` (which explodes every vector into a bullet list).

## 5. Verify every config change

Run the config and check band energies at a few q against the previous values
(or an analytic/reference result) before considering a conversion done:

```bash
python -m magcalc run examples/spinw_tutorials/SWxx_name/config.yaml
```

Zeeman convention: `parameters: {H_mag: <B in Tesla>, H_dir: [...]}` (with
both listed in `parameter_order`) reproduces the electron g=2 Zeeman —
the engine's splitting is `2·μB·H_mag`.
