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
  # (a) distance rule: scalar Heisenberg on EVERY bond of that length.
  #     `distance` WITHOUT `ref_pair` is valid ONLY for type: heisenberg.
  - {type: heisenberg, distance: 3.0, value: J1}
  # (b) ref_pair rule: one reference bond, propagated by the detected
  #     space group (J' = R J R^T; DM parts transform as axial vectors).
  #     REQUIRED for every non-scalar type (dm, interaction_matrix,
  #     anisotropic_exchange, kitaev), and for same-length but
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
rules.

Two rules the engine now **enforces with a hard error** (they used to be silent
failures — a WARNING plus a Hamiltonian quietly missing a term):

1. **`distance` without `ref_pair` is valid only for `type: heisenberg`.** A
   non-scalar rule (`dm`, `interaction_matrix`, `anisotropic_exchange`,
   `kitaev`) needs a `ref_pair`; without one it raises. (The bare-`distance`
   form *does* work if the entry is placed under `interactions.dm_interaction` /
   `interaction_matrix` / `anisotropic_exchange` directly rather than under
   `symmetry_rules` — a separate symmetry-aware expander handles those — but
   prefer `ref_pair`, which is the tested route.)

2. **A rule that matches no bonds raises.** If no pair of sites sits at the
   given `distance`, you get an error naming the rule instead of a Hamiltonian
   silently missing that interaction. Check the distance against the real bond
   lengths.

Cell-image searches are sized from the target distance everywhere (Heisenberg,
DM, matrix, anisotropic, and the `ref_pair` reference-bond lookup), so
2nd-neighbour bonds and bonds reaching past one cell image are found. Passing an
explicit `offset:` on a `ref_pair` rule skips the search entirely.

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

## 5b. Hamiltonian terms beyond bilinear exchange

All of these live under `interactions:` (dict form) and are validated by
`tests/test_hamiltonian_terms.py` against exact identities / Sunny.

```yaml
interactions:
  # Full 3x3 single-ion anisotropy tensor (only the symmetric part matters).
  sia_matrix:
  - {matrix: [[Axx, 0, 0], [0, Ayy, 0], [0, 0, Azz]], atoms: [Fe0]}

  # Crystal field: sum_kq B_k^q O_k^q. Classical (large-s) Stevens polynomials,
  # Sunny `stevens_matrices(Inf)` convention. k in {2,4,6} (even: time reversal),
  # -k <= q <= k. THE ROUTE FOR RARE EARTHS.
  stevens:
  - {B: {'2,0': B20, '4,0': B40, '4,3': B43}, atoms: [Yb0]}

  # Biquadratic B (S_i.S_j)^2. Genuine operator -- valid for NON-collinear
  # structures too (unlike SW28's collinear J_eff = J +/- dJ workaround).
  # Both bond directions, like heisenberg.
  biquadratic:
  - {pair: [A, B], rij_offset: [0, 0, 0], value: -0.037}

  # Long-range dipolar coupling. Two methods:
  dipole_dipole: {method: ewald}                  # EXACT -- prefer this
  # dipole_dipole: {method: truncated, cutoff: 20.0}   # real-space sum, Angstrom
  #
  # The dipolar sum is only CONDITIONALLY convergent: a truncated sum depends on the
  # cutoff and on the (fictitious) sample shape. `ewald` sums it exactly -- real-space
  # + reciprocal-space + the surface/demagnetisation term (`demag:`, default I/3, a
  # sphere in vacuum). Matches Sunny's `enable_dipole_dipole!` to 1e-8; the truncated
  # sum converges to it as the cutoff grows. With `truncated`, RAISE THE CUTOFF until
  # your answer stops moving.
  #
  # Ewald's A(q) is an infinite lattice sum, so it is NOT a bond list: it is added to
  # H(q) numerically, and to the classical energy via A(0) (so the minimiser optimises
  # the same Hamiltonian LSWT diagonalises). Not yet supported with a single-k
  # rotating-frame structure (each q +/- k channel needs its own A(q)) -- it raises
  # rather than quietly using the wrong one; use a magnetic_supercell instead.
  # g comes from the per-site `g`, else 2.
```

**Per-site g-tensor** goes on the atom, not in `interactions`:

```yaml
crystal_structure:
  atoms_uc:
  - {label: Yb0, pos: [...], spin_S: 0.5,
     g: {g_par: 1.8, g_perp: 4.32, axis: [1, 1, 1]}}   # uniaxial about a LOCAL axis
  # also accepted:  g: 2.0  |  g: [gxx, gyy, gzz]  |  g: [[3x3]]
```

The Zeeman is then `mu_B * B . g_i . S_i`. If NO atom declares `g`, the legacy
global isotropic term is used unchanged; an explicit isotropic `g: 2.0` reduces
to it exactly (that is the SW29 calibration, and it is asserted in the tests).

**Multi-k** (`magnetic_structure`) is REAL-SPACE and needs a commensurate cell:

```yaml
crystal_structure: {magnetic_supercell: auto}   # per-axis LCM over all k
magnetic_structure:
  type: multi_k
  components:
  - {k: [0.5, 0, 0], m: [0, 0, 1], phase_deg: 0}   # S_i = sum_m m_m cos(2pi k_m.r_i + phi_m)
  - {k: [0, 0.5, 0], m: [1, 0, 0]}
  normalize: true      # rescale each site to |S| = 1 (default)
```

There is no rotating-frame multi-k theory (SpinW and Sunny also require a
supercell), so every k must be commensurate.

Caveat that bit once: an on-site/bond term that matches **no** bonds, or an
unsupported Stevens order, RAISES -- it is never silently dropped.

### The ground state is the #1 source of silently wrong physics

LSWT is an expansion about a classical energy MINIMUM. Expand about anything else
and the spectrum is meaningless -- but it will still *look* like a spectrum. The
engine now refuses to do that: **two independent guards run before any task**, and
a failure is a hard error, not a warning.

```yaml
calculation:
  on_imaginary: error        # error (default) | warn | off  -- controls BOTH guards
  imaginary_tolerance: 1.0e-4   # meV
  energy_tolerance: 1.0e-6      # meV
```

1. **Imaginary-energy check** (`max_imaginary_energy`) -- a non-minimum with
   anomalous terms gives imaginary magnons. This is the SW20-in-field class.
2. **Energy audit** (`relax_from_current`) -- nudge the structure and relax; if the
   energy drops, it was not a minimum. This catches what guard 1 provably CANNOT:
   a stationary *maximum* (e.g. a `ferromagnetic` pattern supplied for an
   antiferromagnet) keeps the Bogoliubov problem diagonal, so `process_calc_disp`
   sorts the ±ω pairs, returns the upper half, and hands back a real, positive,
   entirely plausible spectrum. Neither guard alone is sufficient.

Set `on_imaginary: warn` **only** when the instability is understood and intended
(SW03's commensurate approximation to an incommensurate spiral; SW23, where the
tutorial itself uses `hermit=false`). Say why in a comment.

### Finding the ground state: use `method: anneal`

**Prefer Monte-Carlo annealing over multistart gradient descent.** It is both more
reliable and cheaper:

```yaml
minimization: {enabled: true, method: anneal, num_starts: 4, n_sweeps: 2000, seed: 0}
```

Methods:

* `anneal` (= `monte_carlo`) -- **the default choice.** Metropolis with a geometric
  cooling schedule (SpinW `anneal`; Sunny `LocalSampler`'s uniform / flip / delta
  proposal mix), then an L-BFGS polish so the answer is a true stationary point.
  Crosses barriers, so it does not get trapped. Optional: `n_sweeps`, `T_start`,
  `T_end` (meV; defaults derived from the coupling scale), `polish`.
* `steep` (= `optmagsteep`) -- iteratively align each spin with its local field
  (SpinW `optmagsteep`). Very fast. **Monotone: it cannot escape a local minimum**,
  so it is a polisher, not a global search. Good as a cheap first look.
* `L-BFGS-B` / `TNC` / ... -- the legacy random-multistart path. Works, but it
  optimizes in (theta, phi), whose coordinate singularities at the poles make it far
  weaker than it looks. On SW20 in field it reached the true minimum in only
  **3 of 200 starts**; annealing reaches it in **1 run out of 1**.

The SW20-in-field numbers (16 sites = 32 angles), true minimum -5.716074 meV:

| method | budget | result |
|---|---|---|
| L-BFGS-B | 24 starts, early_stopping 10 | **-5.338112 (WRONG)** -- imaginary modes at every q |
| L-BFGS-B | 200 starts, early_stopping 40 | -5.716074, hit by only 3/200 starts, ~2 s |
| **anneal** | **1 run x 500 sweeps** | **-5.716074, ~0.8 s** |
| anneal | 4 runs x 2000 sweeps | -5.716074, 4/4 runs, reproducible across seeds |

All methods report `hits` (how many runs reached the best energy) and warn when
`hits == 1`. `early_stopping` (multistart only) now defaults to
`max(10, 2 x n_sites)` instead of a flat 10. **Accept a ground state only when the
energy is reproducible across several `seed`s** -- and the guards above will catch
you if it is not.

## 5c. SU(N) mode (single-ion / multipolar excitations)

```yaml
calculation:
  mode: SUN                      # default: dipole
crystal_structure:
  magnetic_supercell: {matrix: [[1, 0, 0], [0, 1, -2], [0, 1, 2]]}   # may be NON-diagonal
tasks: {minimization: true}      # the SU(N) ground state must be found in SU(N)
```

Dipole LSWT expands each spin as ONE boson about a classical direction, and structurally
cannot represent transitions between an ion's local crystal-field levels. SU(N) gives each
site an N-level Hilbert space (N = 2S+1) with a coherent reference state, so there are
N-1 bosons per site and the single-ion (multipolar) bands appear -- with intensity.

* **FeI2** is the canonical case: `examples/materials/FeI2/config_fei2_sun.yaml`. Its
  bound state is the upper band group (3.5-4.7 meV). Validated against Sunny `:SUN`:
  E/site and all 8 bands AND their intensities match to < 1e-4.
* **The ground state differs from the dipole one** whenever an anisotropy is present: a
  coherent state has `<Sz^2> != (S n_z)^2`. Never seed SU(N) from a dipole ground state --
  run the CP^(N-1) search (`tasks.minimization: true`).
* Non-diagonal `magnetic_supercell` matrices are **only** supported in SU(N) mode; the
  dipole engine refuses them rather than silently using the chemical cell.
* Not yet: powder averaging, domain averaging.

### The two SU(N) traps, and what stops you

**1. Running a model in dipole mode that NEEDS SU(N).** With S >= 1 and an anisotropy, the
single-ion bands simply are not in the dipole spectrum -- nothing looks wrong, whole bands
are just absent. The runner now **warns** whenever S >= 1 AND an on-site anisotropy is
present (and stays quiet otherwise: S=1/2 has no multipolar levels, and without anisotropy
those modes carry no weight).

**2. Seeding SU(N) with a DIPOLE ground state.** This one is nastier, and the
imaginary-mode check is BLIND to it: such a state is normally a perfectly good LOCAL
minimum, so the magnons come out real and the spectrum looks plausible. (Measured on FeI2:
the collinear stripe is 0.048 meV/site above the true ground state and has
|Im w| ~ 5e-16.) Only an ENERGY audit catches it, so SU(N) always runs one on a supplied
structure and refuses by default. Two further barriers:
* a non-diagonal magnetic cell cannot express per-site directions in the config at all, so
  you are structurally forced through the CP^(N-1) search;
* a direction count that does not match the magnetic cell is a hard error naming the fix.

**Reference caveat, learned the hard way:** Sunny's own published FeI2 example converges
to a LOCAL minimum (E/site = -2.35592338, one `minimize_energy!` after
`randomize_spins!`). The true ground state is -2.91893118. A published reference number
is not automatically a converged one -- check it before trusting it.

## 5d. 1/S (LSWT) corrections

```yaml
tasks: {corrections: true}
corrections: {k_mesh: [24, 24, 24]}      # per-axis; use 1 on a decoupled axis
```

LSWT is the leading 1/S term. `compute_corrections` (magcalc/corrections.py) gives the two
standard next-order quantities from the SAME H(q) the dispersion uses:

* **zero-point energy** `dE` per site (add to the classical energy) -- quantum
  fluctuations lower the ground state;
* **ordered-moment reduction** `dS_i`, with `<S^z_i> = S - dS_i`.

Both are k-space integrals, done on an OFFSET (Monkhorst-Pack) grid that avoids Gamma and
the zone edge -- a magnet with a Goldstone mode has omega -> 0 there and the moment
integrand ~ 1/omega, so a Gamma-centred grid samples the divergence directly (it produced
~1e6 nonsense before the offset). The energy converges fast; the moment converges SLOWLY
across a gapless cone, so use a fine `k_mesh` (>= 64 per active axis) for dS.

Validated against Sunny 0.8.1 AND the textbook S=1/2 square-lattice Heisenberg AFM:
`dE = -0.157947 J/site` (exact to 6 dp), `dS -> 0.1966`. A ferromagnet gives exactly zero
(the classical state IS the magnon vacuum) -- the cleanest self-consistency check.

Refuses (does not return a plausible number) when the structure is not a classical
minimum: imaginary magnons, or H(q) non-positive-definite, both trigger a hard error.

## 6. Intensity / experiment layer

Applies to S(Q,ω), powder, energy-cut **and FITTING** intensities (never to energies).
`magcalc fit` reads the SAME `calculation:` block as `magcalc run`, so a fit and a
forward run model the experiment identically; `fitting:` may override any key locally.

```yaml
calculation:
  temperature: 5.0                       # K -> Bose factor per mode
  domains: {axis: [0, 0, 1], n_fold: 3}  # twins
  cross_section: perp                    # | trace | chiral | sf+ | sf- | xx | zz | xy ...
```

**Polarized / chiral.** With the polarization along q (longitudinal SF/NSF) all magnetic
scattering is spin-flip and the beams differ by the chiral term:
`M_ch = i q̂·[Σ ε_abc S^ab]`, `σ_SF^± = S_perp ∓ M_ch`. `cross_section: chiral` returns the
signed M_ch (it vanishes identically for any collinear structure, and for a cycloid when
q ⊥ the rotation axis). Sign convention pinned to Sunny — `tests/test_polarized.py`.

**Absolute normalization caveat:** pyMagCalc's S(Q,ω) is **3/4 of Sunny's**. This is a
pre-existing convention difference that affects every channel identically (verified on
`perp`), so ratios and fitted parameters are unaffected — a fit's free `scale` absorbs it.
Do not compare ABSOLUTE intensities with Sunny without this factor.

**Ignoring temperature biases fits.** Not a rounding effect: on a ferrimagnet, fitting
40 K data with a T=0 model returns J = 1.07 instead of 1.30 (a 17% error), because the
Bose factor reweights the acoustic and optic branches *relative to each other* and a free
`scale` cannot absorb that. (On a simple AFM chain no bias is even possible: I ~ 1/(J f(q)),
so J only rescales the intensity and it carries no information about J at all.)

**Mixed spin.** The S(Q,ω) prefactor is √(S_i/2) **per site**. It used to be a single
global √(S/2), making every site whose S differed from the reference wrong by √(S_i/S_ref)
— a 60% error on a Cu(½)+Fe(2) model. The Fortran backend still applies the global
factor, so it now falls back to NumPy for mixed-spin S(Q,ω).


Applies to S(Q,ω), powder, and energy-cut intensities (never to energies):

```yaml
calculation:
  temperature: 5.0                       # K -> Bose factor per mode
  domains: {axis: [0, 0, 1], n_fold: 3}  # twins; or explicit list of
                                         #   {axis, angle, weight} (include angle 0)
  cross_section: perp                    # | trace | xx | zz | xy ...
plotting:
  resolution:
    de_fwhm: [-0.0125, 0.107143, -0.141071, 0.059286]  # polyval FWHM(E),
                                         #   highest power first; or a scalar
    shape: gaussian                      # default gaussian when de_fwhm given
    dq_fwhm: 0.05                        # |Q| smoothing (1/A)
    ei: 25.0                             # direct-geometry kinematics (meV)
    two_theta: [5, 130]                  # detector coverage; masks powder maps
  energy_grid_step: 0.01                 # map energy grid (default 0.05)
```

Caveats: domains work only with `cross_section: perp|trace` (lab-frame
components of a rotated crystal would need tensor rotation — the engine
raises); powder ignores domains (spherical average is rotation-invariant);
dispersion and fitting stay single-domain. Constant-energy cuts on a 2-D
q grid: `tasks: {energy_cut: true}` +

```yaml
energy_cut:
  origin: [0, 0, 0]
  axis1: {vec: [4, 0, 0], points: 121}   # RLU span from origin
  axis2: {vec: [0, 4, 0], points: 121}
  cuts:
  - {center: 3.75, fwhm: 0.25}           # Gaussian energy window
  - {band: [3.5, 4.01]}                  # hard integration window
```

Reference: SW10 (energy_cut), SW37 (resolution polynomial).
