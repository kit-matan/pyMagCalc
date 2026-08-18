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
  # (c) kitaev: a SCALAR K plus the Cartesian spin `axis` the bond couples.
  #     Equivalent to an interaction_matrix carrying K at that diagonal entry --
  #     and propagated the same way, R K R^T, so one rule generates the whole
  #     bond-dependent set: on a cubic lattice a z-axis reference bond gives
  #     K^xx on the x bonds, K^yy on the y, K^zz on the z.
  - {type: kitaev, ref_pair: [Ir0, Ir1], offset: [0, 0, 1], value: K, axis: z}
```

Rules expand to **both bond directions** automatically (required by
pyMagCalc's `H = (1/2) Σ_ordered`); never list reverse bonds by hand alongside
rules.

`kitaev` is also available as an explicit bond list under `interactions.kitaev`
(`{pair, rij_offset, value: K, axis}`; `bond_direction` is an accepted alias for
`axis`). Both routes raise on an unresolvable `value` and on an `axis` outside
{x, y, z} -- until 2026-08 the first logged a warning and dropped the bond and
the second silently meant `z`, and the `symmetry_rules` route had no propagation
branch at all, so a documented rule expanded to **zero bonds** in silence. Any
rule type without a propagation branch now raises rather than expanding to
nothing. `tests/test_kitaev.py` pins every path to its exact `interaction_matrix`
equivalent.

To pick `ref_pair` bonds and see which matrix entries symmetry zeros or ties,
run `magcalc symmetry <config> [--max-distance Å] [--json]` — it lists the space
group, the symmetry-inequivalent bond orbits, and the allowed exchange matrix
for each (the Sunny `print_symmetry_table` analogue).

**That allowed matrix was wrong on any lattice carrying measurement noise until
2026-08-17.** `get_bond_constraints` solved `J = R J Rᵀ` symbolically after
"sanitizing" the Cartesian rotation with `np.round(R, 10)` + `nsimplify` — and
nsimplify of a 10-decimal float is not `2/3`, it is the exact rational
`1666666499/2500000000`. Such an R is not orthogonal, so the constraint admits
far fewer solutions than it should. Measured: perturbing KFe₃J's `a` by **1e-7 Å**
took its NN bond from **6 free parameters to 1**, reporting every off-diagonal as
symmetry-forbidden — i.e. "no DM allowed here" — on the config whose whole point
is its DM term. On a Materials Project NiO primitive cell (cubic to six decimals,
not exactly) the same giant rationals instead made sympy grind for **over ten
minutes without returning**. Both are now a 0.2 ms SVD null space, with R snapped
to the nearest orthogonal matrix first (spglib already accepted the op at
symprec = 1e-3, so idealizing it assumes nothing new) and coefficients snapped at
`_SYMMETRY_COEFF_TOL = 1e-5`. The reported basis is now the canonical rref one,
so free parameters may be *named* differently than before; the space they span is
the same, and `tests/test_symmetry_cli.py` pins both the unchanged forms and the
1e-7 stability that used to fail.

Three rules the engine now **enforces with a hard error** (they used to be silent
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

3. **An ambiguous reference bond raises.** A `ref_pair` without `offset` has to
   choose which periodic image is the reference. Two cases now error instead of
   guessing: (a) several images of the pair are the *same length* and the rule
   carries a direction (`dm`, `interaction_matrix`, `anisotropic_exchange`,
   `kitaev`); (b) the `distance` window spans more than one bond length, i.e.
   more than one orbit — an error for every type, scalar included. Fix either by
   pinning `offset: [u, v, w]` (the error names the candidates). A scalar
   `heisenberg` rule in case (a) is left alone: it expands to the whole orbit, so
   the bond table does not depend on which image was the reference.

   This one bit CCSF. In P2₁/n the two screw-related Cu2–Cu2 (J12) bonds from a
   site have *identical* length, and the two candidates differed by ~9e-16 Å — one
   ULP — so the old `<` comparison picked between them by floating-point rounding.
   The chosen bond sets the orientation convention for the whole orbit: the 2₁
   screw and the *n* glide both act on axial vectors as C2x = diag(1,−1,−1), so
   the other choice realizes −C2x·**D**, silently flipping the sign of `D12x`.
   Identical bond count, plausible spectrum, different Hamiltonian — caught only
   by diffing the bond table against an independent model.

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
  # NB: un-renormalized, i.e. Sunny's `:dipole_uncorrected` -- see
  # `calculation.anisotropy_renormalization` in section 5b1 below.
  stevens:
  - {B: {'2,0': B20, '4,0': B40, '4,3': B43}, atoms: [Yb0]}

  # Biquadratic B (S_i.S_j)^2. Genuine operator -- valid for NON-collinear
  # structures too (unlike SW28's collinear J_eff = J +/- dJ workaround).
  # Both bond directions, like heisenberg.
  biquadratic:
  - {pair: [A, B], rij_offset: [0, 0, 0], value: -0.037}

  # Arbitrary two-site coupling (SU(N)/entangled only; Sunny `set_pair_coupling!`).
  # `poly` is sum_n c_n (S_i.S_j)^n -- c1 = Heisenberg, c2 = biquadratic, higher n
  # = ring-exchange-like. `matrix:` takes an explicit (2S+1)^2 Hermitian operator on
  # the product space instead. Decomposed into sum_k A_k (x) B_k internally.
  # Both bond directions are listed, as for heisenberg, so the operator must be
  # SYMMETRIC under exchanging the two sites (checked; it raises otherwise).
  pair_operator:
  - {pair: [A, B], rij_offset: [0, 0, 0], poly: [0, 1.0, -0.4]}

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
  # the same Hamiltonian LSWT diagonalises).
  # g comes from the per-site `g`, else 2.
```

**Ewald with a single-k (rotating-frame) structure** works as of 2026-08-13 (it used
to refuse). Each of the three q ± k channels gets its own A(q), itself the
Toth–Lake projector combination at that momentum. Read this before trusting the
number:

- it is EXACT when A(q) is uniaxial about the spiral axis — i.e. when that axis is a
  3-fold or higher symmetry axis of the lattice and q lies along it — and when
  2k is a reciprocal-lattice vector (`k_case 2`, where the ±2k umklapp folds back
  into the same channel and is kept);
- OTHERWISE it drops the ±2k terms, which leave the {q−k, q, q+k} channel set
  entirely. That is the same approximation Sunny's `SpinWaveTheorySpiral` makes, and
  it is not small: ~10–20 % of the dipolar shift on a chain whose axis lies in the
  spiral plane. **The engine warns**, following
  `magnetic_structure.enforce_rotational_symmetry` (`warn` default, `error`, `off`),
  and names the dropped weight. Sunny's own `check_rotational_symmetry` cannot see
  this — the dipolar term sits outside `interactions_union` — so the warning has no
  counterpart there;
- for an exact answer at commensurate k, use `crystal_structure.magnetic_supercell`.
  Note that a dipolar term breaking the U(1) symmetry generally makes the spiral
  itself unstable (Sunny errors outright on such models), so the warning is usually
  telling you something about the physics, not only about the method.

Validated in `tests/test_ewald_spiral.py`: against the explicit `magnetic_supercell`
as an exact identity (both `k_case` branches, after the no-Ewald control passes) and
against Sunny 0.8.1 at incommensurate k.

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

### 5b1. Anisotropy renormalization — dipole mode is Sunny's `:dipole_uncorrected`

Dipole LSWT replaces an on-site operator by its **classical (s → ∞) polynomial**, which
overestimates a rank-k term at finite s. Sunny's DEFAULT `:dipole` mode corrects for this
(RCS, D. Dahlbom et al., arXiv:2304.03874): every rank-k Stevens coefficient is scaled by

    λ_k(s),   λ_2 = 1 − 1/(2s):   0 at s = ½,  ½ at s = 1,  ⅔ at s = 3/2  → 1

so that `:dipole` agrees with the exact `:SUN`. Its `:dipole_uncorrected` mode does not.
**pyMagCalc's dipole engine has always been the uncorrected one** — the same as SpinW, and
what every config in this repo means — and the difference is large:

| model | pyMagCalc default | Sunny `:dipole` |
|---|---|---|
| s = 1, `sia` D = −0.5 | every band +0.5 meV | matches `:SUN` |
| s = 2, `stevens` B₄⁰ | gap 13.13 meV | 1.53 meV (λ₄ = 0.09375) |

Opt in per config; it applies to `sia`, `sia_matrix`, `stevens` and `biquadratic`, in
dipole mode only (SU(N)/entangled carry the full operator and are already exact):

```yaml
calculation:
  anisotropy_renormalization: rcs      # none (default) | rcs
```

λ₂(½) = 0 is the sanity check: (S·n)² *is* a constant for s = ½, so a quadratic
anisotropy can have no effect there — the un-renormalized classical polynomial wrongly
says otherwise. For biquadratic, RCS scales the *quadrupole* part and shifts the bilinear
part by −B/2 (Sunny's `adapt_for_biquad`), not the raw coefficient. Both branches are
pinned to Sunny in `tests/test_rcs_renormalization.py`. If you want the accurate finite-s
answer and can afford it, `mode: SUN` is exact and needs no factor at all.

### The ground state is the #1 source of silently wrong physics

LSWT is an expansion about a classical energy MINIMUM. Expand about anything else
and the spectrum is meaningless -- but it will still *look* like a spectrum. The
engine now refuses to do that: **two independent guards run before any task** (and a
third, sharper one per q for `kpm_sqw`, which cannot rely on either), and a failure is
a hard error, not a warning.

```yaml
calculation:
  on_imaginary: error        # error (default) | warn | off  -- controls ALL THREE
  imaginary_tolerance: 1.0e-4      # meV, ABSOLUTE  |  guard 1 fires only if
  imaginary_rel_tolerance: 5.0e-3  # fraction of the bandwidth  |  BOTH are exceeded
  energy_tolerance: 1.0e-6         # meV per cell (per SITE in SU(N) mode)
  h2_rel_tolerance: 1.0e-6         # guard 3 (kpm_sqw): min eig H2 / ||H2||
```

1. **Imaginary-energy check** (`max_imaginary_energy`) -- a non-minimum with
   anomalous terms gives imaginary magnons. This is the SW20-in-field class.
   The two thresholds are **ANDed**, so lowering `imaginary_tolerance` alone will
   *not* make the guard fire. That is deliberate: an absolute meV cutoff cannot
   separate a real instability from numerical noise across models whose energy
   scales differ by orders of magnitude, and the noise is worst exactly where it
   matters -- at the ω ≈ 0 Goldstone modes where the Bogoliubov problem is singular
   (SW07's 120° kagome carries 1e-3 meV of noise on a 2.4 meV band). Either knob
   alone therefore SILENCES the guard; both must be exceeded to trip it.
2. **Energy audit** (`relax_from_current`) -- nudge the structure and relax; if the
   energy drops, it was not a minimum. This catches what guard 1 provably CANNOT:
   a stationary *maximum* (e.g. a `ferromagnetic` pattern supplied for an
   antiferromagnet) keeps the Bogoliubov problem diagonal, so `process_calc_disp`
   sorts the ±ω pairs, returns the upper half, and hands back a real, positive,
   entirely plausible spectrum. Neither guard alone is sufficient.
   SU(N) mode runs its OWN energy audit (`sun/adapter.py`) off the same
   `energy_tolerance` key but in meV **per site** -- it is the only thing that can
   catch a dipole-derived state pasted under `mode: SUN` (§5c).
3. **H₂(q) ⪰ 0, per q -- `tasks: {kpm_sqw: true}` only** (`SUNModel.is_stable_at` /
   `assert_stable`, `tests/test_kpm_stability.py`). Guards 1 and 2 are necessary and
   *not sufficient* for KPM, which never diagonalizes and so has no Cholesky to fail:
   both run on the reference state within its own cell, and a state can be a genuine
   in-cell minimum with an entirely real spectrum while being unstable to a
   modulation the cell cannot represent (a frustrated ferromagnetic chain whose true
   state is an incommensurate spiral — guard 1 reads |Im ω| = 0 *exactly*, guard 2
   relaxes and stays put, a `dispersion` run of it succeeds and plots a plausible
   band). H₂(q) ⪰ 0 is the exact criterion and is strictly sharper than guard 1: at a
   stationary maximum H₂ stays diagonal, so g·H₂ is entirely real. It runs at **every
   q the KPM computes**, not a sample, because the instability is q-specific — one
   shifted Cholesky (`H₂ + εI ≻ 0`, ε = `h2_rel_tolerance`·‖H₂‖) is exact and 45×
   cheaper than an eigensolve at 2D = 1800, i.e. 1–5 % of the KPM work at that q. The
   shift is not a fudge: a Goldstone mode puts an *exact* zero in H₂ at the ordering
   wavevector and at Γ, so an unshifted test would refuse every gapless magnet.

`tests/test_guard_tolerances.py` pins all of this against the exact tilt identity
ΔE(θ) = 2·J·S²·(1 − cos θ) for a Néel chain, and each knob is bracketed above and
below a drop of known size.

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
  proposal mix), then a `steepest_descent` polish that is **kept only if it lowers
  the energy** (see the warning below -- it used to be taken unconditionally).
  Crosses barriers, so it does not get trapped. Optional: `n_sweeps`, `T_start`,
  `T_end` (meV; defaults derived from the coupling scale), `polish`.
* `steep` (= `optmagsteep`) -- iteratively align each spin with its local field
  (SpinW `optmagsteep`). Very fast. **Monotone: it cannot escape a local minimum**,
  so it is a polisher, not a global search. Good as a cheap first look.
* `L-BFGS-B` / `TNC` / ... -- the legacy random-multistart path. Works, but it
  optimizes in (theta, phi), whose coordinate singularities at the poles make it far
  weaker than it looks. On SW20 in field it reached the true minimum in only
  **3 of 200 starts**; annealing reaches it in **1 run out of 1**.

The SW20-in-field numbers (16 sites = 32 angles), true minimum -9.662153 meV
(re-measured after the 2026-07 Zeeman calibration fix; the pre-fix docs quoted
-5.716074 with a half-strength field):

| method | budget | result |
|---|---|---|
| L-BFGS-B | 24 starts, early_stopping 10 | **-8.994590 (WRONG)** -- a local minimum |
| L-BFGS-B | 200 starts, early_stopping 40 | -9.662153, hit by only 3/200 starts, ~1 s |
| **anneal** | **1 run x 500 sweeps** | **-9.662153, ~0.9 s** |
| anneal | 4 runs x 2000 sweeps | -9.662153, 4/4 runs, reproducible across seeds |

All methods report `hits` (how many runs reached the best energy) and warn when
`hits == 1`. `early_stopping` (multistart only) now defaults to
`max(10, 2 x n_sites)` instead of a flat 10. **Accept a ground state only when the
energy is reproducible across several `seed`s** -- and the guards above will catch
you if it is not.

**REPRODUCIBILITY IS NOT CORRECTNESS** (2026-08-13). That last rule is necessary and
NOT sufficient, and this is the case that proves it: `anneal`'s `steepest_descent`
polish used to be taken **unconditionally**, and that step aligns each spin with the
field from everything EXCEPT itself, so it ignores the on-site block `H_ii`. With a
large single-ion term it is not a descent step at all. On the S06 skyrmion model
(easy-plane D = 19, one site per cell, so all 12 bonds fold into `H_ii`) Metropolis
found the exact minimum E = -4.644706 and the polish walked it up to **+0.520665, a
local MAXIMUM** -- on every seed, at 500 and 5000 sweeps, so `minimize_energy`
reported "4 of 4 runs hit the minimum". **A deterministic bug downstream of a
stochastic search reproduces perfectly.** What caught it was an analytic minimum in
closed form, not a repeat run. Both halves are fixed (the polish is kept only if it
lowers the energy; `steepest_descent` returns the best state it saw), and the shipped
configs were swept -- only S06 and the staging FeI2 config carry both `anneal` and an
on-site term, and FeI2's ground state was unchanged. Note the second line of defence
is the guards above, so a **direct `minimize_energy` call outside the runner**, or a
config running `on_imaginary: warn|off`, had none.

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
* **Mixed spin is supported**: sites may carry different `spin_S` (S=½ gives 1 boson,
  S=1 gives 2, S=3/2 gives 3), addressed through a per-site offsets table. `model.M`
  and `model.N` are `None` for such a cell — use `Ms`/`Ns`/`D` — so anything still
  assuming a uniform N fails loudly instead of silently using site 0's value.
* Powder averaging: supported (shared spherical average over calculate_sqw,
  `tests/test_powder_sun.py`). Not yet: domain averaging.

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

## 5e. Magnetic CIF (mCIF) import

```yaml
from_mcif: my_structure.mcif      # relative to the config file
mcif: {spin_S: 2.5, ion: Fe2+}    # optional: S and form-factor ion for every site
interactions: {...}               # you still supply the exchange
```

An mCIF encodes an experimentally-determined magnetic structure via a magnetic space
group (symmetry ops each with a time-reversal parity p = +/-1). `from_mcif` expands it
into the full magnetic cell -- lattice, per-site positions, and spin DIRECTIONS -- so you
only add `interactions`/`parameters`/`tasks`. An explicit `crystal_structure` in the same
config overrides the mCIF-derived one.

CLI: `magcalc mcif file.mcif [--out frag.yaml] [--spin-s S] [--ion Fe2+]` prints the
expanded sites, or writes a runnable config fragment. **The reverse direction exists as of
2026-08-18**: `magcalc mcif-out config.yaml [-o out.mcif] [--minimize]` writes the config's
magnetic structure back out as an mCIF, in P1 magnetic symmetry with every site listed
explicitly -- which is what FullProf's `mCIF_to_PCR`, VESTA and Bilbao accept, and what
makes the round trip exact. `--minimize` exports the minimized structure instead of the
configured one (off by default: a minimisation that can land in a different domain each
run would make the file non-deterministic).

Transforms (matching Sunny's MCIF.jl): position `r' = R r + T`; moment (an AXIAL vector)
`m' = det(R) * p * (R m)` -- invariant under spatial inversion, flipped by time reversal.
Validated against Sunny on TbSb (`tests/data_TbSb.mcif`): identical sites and directions,
including the R-centring anti-translations that make it a G-type AFM. The reader REFUSES a
file whose symops map a fixed site to two different moments (as Sunny does).

**Moment components are mu_B on UNIT crystal axes** (`moment_basis='unit_axes'`, the
default): `m_cart = mx*a^ + my*b^ + mz*c^`. Until 2026-08-18 this reader multiplied them by
the FULL lattice vectors, which is what **Sunny 0.8.1 also does** -- and it is wrong:
FullProf's own mCIF export writes the same three numbers it carries as the PCR magnetic
phase's Rx/Ry/Rz and states `spherical_modulus` = their plain Euclidean norm beside them
(Ho2BaNiO5: components (-0.1441, 0, -8.9931) in a 7.51 x 5.74 x 22.56 A cell, modulus
8.99423 -- the other reading gives 203 mu_B, which no Ho3+ ion can carry). The two differ in
DIRECTION, not only magnitude, whenever the moment has components on axes of unequal length
(0.92 deg off -c here, against 0.31 deg before), so it changed the magnetic structure LSWT
was handed. It went unnoticed because TbSb's moment lies along c ALONE, where the two bases
differ only by a positive scale factor. Pass `moment_basis='lattice_vectors'` to read a file
written under the old reading.

**Elastic magnetic intensities** live in `magcalc/diffraction.py`: `magnetic_intensity(Q,
positions, moments, ions=...)` is `|F_perp|^2` summed COHERENTLY over every magnetic site,
with the same form factors and `perp` projection the S(Q,w) layer uses.
`incoherent_intensity(..., groups=...)` computes the multi-phase answer `sum_p |F_p|^2` on
purpose, and `cross_term_fraction` measures how far apart the two are -- because entering
two inequivalent sublattices as two FullProf *phases* drops the interference between them,
converges happily, reports a respectable chi^2, and picks the wrong magnetic structure
(this is what forced an erratum on Cs2Cu3SnF12). Check any multi-phase refinement on the
reflections where that fraction is LARGE; a small one proves nothing.

## 5f. Entangled units (dimers / trimers)

```yaml
calculation: {mode: entangled}
units: [[Cu0, Cu1]]      # each unit = a list of site labels (or indices) forming a cluster
crystal_structure: {...} # the PHYSICAL spins (2 per dimer, etc.)
interactions: {...}      # intra- AND inter-unit exchange between the physical spins
```

A "unit" (dimer/trimer/tetramer) is treated as ONE effective SU(N) site whose Hilbert
space is the product of its constituents (N = prod_k (2 S_k + 1)). The intra-unit coupling
is diagonalized EXACTLY and becomes the on-site term; the reference is the unit's ground
state -- e.g. a dimer SINGLET, which has zero dipole moment, so dipole (and single-site
SU(N)) LSWT structurally cannot see its excitations. Inter-unit couplings disperse the
resulting triplon (Sunny's `EntangledSystem` analogue).

* THE case for strong intra-cluster coupling: spin-dimer magnets (a singlet ground state
  with a gapped triplon), spin ladders, etc. -- where the excitation is a transition WITHIN
  the cluster spectrum, not a spin precession.
* `units:` must partition every magnetic site exactly once, and all units must have the
  same product dimension N (hard error otherwise). Bonds within a unit fold into the
  on-site Hamiltonian; everything else becomes an inter-unit bond. A unit's members may
  carry a cell offset -- `[i, [j, [ox,oy,oz]]]` -- so a dimer can STRADDLE the cell
  boundary (real dimer coverings always have some; the Rb2Cu3SnF12 pinwheel has 4 of 18).
* The reference is exact, so there is no coherent-state minimization; an over-strong
  inter-unit coupling instead shows up as an imaginary triplon (the ground-state guard's
  imaginary check -- the dimer picture breaking down).
* The neutron structure factor uses the q-dependent STAGGERED moment sum_k e^{i q.d_k} S_k
  (d_k = constituent offset), so the dimer selection rule I(q=0)=0 and the (1-cos(q.d))
  form factor come out right (the total spin sum_k S_k alone is silent on the triplon).
* A magnetic field (`parameters: {H_mag: <Tesla>, H_dir: [...]}`) adds the Zeeman term
  gamma*mu_B*H.(sum_k S_k) to each unit, so a field splits the unit's multiplet -- e.g. a
  c-axis field Zeeman-splits the Stot^z = +/-1 dimer triplet while Stot^z = 0 is
  unchanged (`examples/entangled/Rb2Cu3SnF12/`, Matan et al., Nat. Phys. 6, 865 (2010)).
* **On-site anisotropy** (`single_ion_anisotropy`, `sia_matrix`, `stevens`) is applied
  per constituent, embedded into the unit's product space and folded into the on-site
  block BEFORE the reference state is chosen -- so an anisotropic dimer's reference is
  the anisotropic ground state. Pinned against exact diagonalization of the isolated
  unit (`tests/test_entangled_units.py`), which is the definition, not a golden number.
  These were **silently dropped until 2026-08-05**: the builder read the bilinear pair
  terms plus the Zeeman and nothing else, so a D = -5 meV anisotropy on an S=1 dimer
  changed the triplon by exactly 0.000. Note that at S=1/2 an (S.n)^2 anisotropy is a
  CONSTANT (Sz^2 = I/4) and correctly has no effect -- which is part of why the drop
  went unnoticed, since the shipped dimer examples are all S=1/2.
* Harmonic bond-operator level: EXACT in the weak-interdimer limit (Cu5SbO6), only
  qualitative at strong coupling (J2 ~ J1). For strong coupling use the SERIES:

```yaml
calculation: {mode: entangled, series_order: 5, series_resum: dlog_pade}  # | pade | sum
```

  `series_order: N` switches the DISPERSION to the high-order dimer series expansion
  (`magcalc/sun/dimer_series.py`): a linked-cluster expansion of the one-triplon
  effective Hamiltonian to order N in ALL interdimer couplings (Heisenberg + DM),
  resummed per band with Dlog-Pade (the papers' method; the spread across approximants
  is the uncertainty). Validated against exact diagonalization of the alternating
  chain -- 5e-4 J at J'/J = 0.4, better than 8 percent at the STRONG coupling
  J'/J = 0.8 -- and against the exact first-order dimer expansion of PRR 8, 013247
  (Eq. A11). Cost grows fast with order (cluster count x Hilbert 4^(N+1)): order 4-5
  is interactive, 6+ is a batch run. S(Q,w) and the ground-state guards stay on the
  harmonic model. Reference: `examples/entangled/Rb2Cu3SnF12/series_dispersion.py`
  (the full pinwheel at J2 = 0.95 J1, gaps vs the measured 2.35 / 7.3 meV).
* Powder averaging: supported (same shared spherical average as SU(N)).
  Not yet: domain averaging.

Example: `examples/entangled/dimer_chain/` -- a chain of S=1/2 dimers whose triplon
`omega(q) = sqrt(J^2 - J J' cos 2 pi q)` matches the exact bond-operator result.
Reference: `magcalc/sun/entangled.py`, `tests/test_entangled_units.py`.

## 5f2. Test suite: fast by default, FULL before merge

`pytest` runs the FAST suite (**697 of 876 collected tests**): the `slow`
marker (pytest.ini) holds the deep validations (ED oracles, convergence sweeps,
integration runs). The last full gate was **882 passed, 1 skipped** (`pytest -m ""`
from the workspace root, 2026-08-17, **12 min**, 883 collected there since it also
picks up `fMagCalc/tests`).

Both pytest.ini files carry `-n auto --dist worksteal` (pytest-xdist), which is the
whole of that 33 min -> 12 min; the fast suite went 6 m 07 s -> 4 m 37 s. **Nothing was
deselected or weakened to get it** -- same 882/1. Use **`-n0`** when you need `-s`, pdb,
or a live per-test name. The remaining floor is a single test:
`test_classical_absolute_normalization`'s 128-trajectory LL average is ~7 min of serial
Python, so the gate cannot go below it until `sampled_correlations` parallelizes its
trajectory loop (it is a plain `for it in range(n_traj)`), and its cost is the
statistical tolerance the test asserts -- do not trim `n_traj` to save time.

Rules:

- iterate with `pytest`; run a feature's deep checks with `pytest -m slow -k <name>`;
- **before merging to master, ALWAYS run `pytest -m ""` (everything)** -- the fast
  suite alone is NOT a merge gate;
- every feature must keep at least one quick pinned test OUTSIDE `slow`, so the
  fast suite still touches all code paths;
- **run it as `pytest -m "" -rs` and account for every skip.** A skip is not a pass,
  and a MODULE-level one is invisible without `-rs`.

### A skipped module is a silent hole (2026-08-16)

The gate read "3 skipped" for a month and only ONE was a real optional dependency.
`tests/test_magcalc.py` guards its imports with a `try/except ImportError` that calls
`pytest.skip(..., allow_module_level=True)` -- and its import list still named
`_calculate_K_Kd`, which `9699a86` ("sweep dead code") had deleted from `linalg.py`.
Nothing in the file used the symbol; the stale line alone **took all 24 of its tests
dark**, and the file was 4 % of the suite. Same commit, same blind spot: it also changed
`gram_schmidt` to ZERO rank-deficient columns instead of returning numpy's arbitrary
orthonormal completion, and the two tests asserting the old `Q^d Q = I` contract went
dark in the very commit that invalidated them -- so they never failed. Restoring the
module surfaced them immediately (22 passed, 2 failed) and they now pin the real
contract: one column zeroed, `Q^d Q` = the projector onto the survivors, survivors still
spanning the input's column space, warning logged.

`fMagCalc/tests/test_pymagcalc_integration.py` was the same shape one repo over -- it
skipped on `examples/KFe3J`, a path the examples reorg had moved to
`examples/materials/KFe3J`, so the Fortran-vs-NumPy S(Q,w) oracle
(`|E - E_oracle| < 1e-8`) had not run since. **A skipif whose condition is a filesystem
path rots silently when the file moves**; prefer one that fails loudly, or assert the
path exists in a cheap always-on test.

The surviving skip is `seekpath` and it is structural: `tests/test_cell_utils.py:146`
needs the package PRESENT (it checks seekpath's own spacegroup determination against
spglib's) and `:161` needs it ABSENT (it checks the ImportError is actionable), so
exactly one of the pair always skips. `pip install seekpath` swaps which -- worth it, as
it trades a message check for a genuine third-party oracle. The count cannot reach zero.

## 5g. Beyond LSWT: diffuse, thermal, and dynamical methods

Four tasks for regimes LSWT (an expansion about an ordered state) does not cover.
The first three are CLASSICAL and paramagnetic-friendly, so when run alone they
auto-skip the LSWT ground-state guard (no ordered state required).

```yaml
tasks: {scga: true}                 # paramagnetic diffuse S(q) above T_N
scga: {temperature: 1.5, mesh_density: 20, cross_section: perp}   # temperature = kT (meV)

tasks: {thermal_mc: true}           # finite-T thermodynamics <E>,C,M,chi vs T
thermal_mc: {temperatures: [0.2,0.5,1,2,4], supercell: [6,6,1], n_sweeps: 4000, n_equil: 1500}

tasks: {sampled_correlations: true} # classical-dynamics S(q,w) (full thermal lineshape)
sampled_correlations: {temperature: 0.5, supercell: [16,1,1], dt: 0.02, n_steps: 2048, n_traj: 8}

# Site-level disorder, available to ALL the real-space classical samplers below
# (thermal_mc, sampled_correlations, static_correlations, wang_landau). A vacancy
# DELETES the site's rows/columns from the classical energy, so it removes every bond
# it took part in; `periodic: false` on an axis drops the bonds that wrap it.
# LSWT does not support disorder (its front end is symbolic and per-cell).
thermal_mc: {supercell: [8,8,1], disorder: {vacancy_concentration: 0.1, seed: 0},
             periodic: [true, true, false]}

tasks: {wang_landau: true}          # density of states g(E): ONE run, every temperature
wang_landau: {supercell: [4,4,1], temperatures: [0.25,0.5,1,2,4], n_bins: 100, f_final: 1.0e-6}

calculation: {mode: SUN}            # KPM: Chebyshev S(q,w), no diagonalization (large cells)
tasks: {kpm_sqw: true}
kpm: {e_min: 0, e_max: 10, e_step: 0.05, fwhm: 0.1, tol: 0.02}    # or moments: N
```

* **SCGA** (`magcalc/scga.py`) -- self-consistent Gaussian approximation. `S(q) = kT
  (lambda + J(q))^{-1}` with the same Fourier exchange matrix as the LT guard and a
  single Lagrange multiplier lambda from the spin sum rule. Single symmetry class only
  (Bravais + kagome/pyrochlore-type); refuses inequivalent sublattices. Validated vs
  Sunny 0.8.1 SCGA (square + kagome) and the exact classical-chain closed form.
* **Thermal MC** (`magcalc/thermal_mc.py`) -- parallel-tempering Metropolis on a PBC
  supercell built from `spin_interactions` + `_resolve_field`, same classical energy
  `1/2 m^T H m + b^T m` as `annealing`. Validated vs the Langevin function (free spins
  in field) and the exact classical dimer ⟨E⟩(T), C(T).
  **ON-SITE ANISOTROPY WAS MISSING FROM THIS BUILDER UNTIL 2026-08-15.** `H` came
  from the bond list alone, so `single_ion_anisotropy` / `sia_matrix` / `stevens`
  never reached **`thermal_mc`, `wang_landau`, `static_correlations` or the classical
  `sampled_correlations`** -- an anisotropic model was silently sampled as
  exchange-only in all four (measured on S06: `H_zz` was the bare exchange sum, with
  D = 19 simply absent), while the docstring already called `H` "the
  exchange/anisotropy Hessian". Every test here was bond-only, so the property had
  never been false in the suite -- the `ref_pair` / `steepest_descent` shape again.
  `thermal_mc.onsite_quadratic` now reads the terms from the MODEL'S OWN assembly
  (`_compute_sia_terms`, so targeting, parameter resolution and the RCS
  renormalization cannot drift from LSWT) and extracts (H, b) by exact numeric
  probing. Pinned against the closed-form single-spin partition function in
  `tests/test_thermal_mc.py`; 5 of its 6 new tests were confirmed to FAIL before.
  **A Stevens term of rank k >= 4 now RAISES here**: its classical polynomial is
  quartic/sextic, so `E = 1/2 m^T H m + b^T m` cannot carry it at all -- use
  `mode: SUN`. Refusing beats sampling the model without it.
* **Wang-Landau** (`magcalc/thermal_mc.py`) -- flat-histogram sampling of g(E), so
  the whole T sweep is post-processing rather than one simulation per T. Validated on
  the classical dimer, whose g(E) is EXACTLY FLAT in closed form.
* **SampledCorrelations** (`magcalc/classical_dynamics.py`) -- Landau-Lifshitz on
  thermalized states, S(q,w) by space-time FFT. Thermalize by Metropolis or by the
  `langevin_step` thermostat; measure with `integrator='rk4'` or `'midpoint'` (the
  implicit midpoint rule is symplectic: energy drift 1e-12 vs RK4's 8e-5 over a long
  run, and |S| conserved exactly without renormalizing). `suggest_timestep` picks dt
  from the largest local field.
  MIND THE SIGN if you touch the damping: `local_field` returns the energy GRADIENT
  G = dE/dS, not the field B = -G, so the Landau-Lifshitz damping is
  +(lambda/S) S x (S x G). The textbook sign relaxes spins AWAY from the minimum and
  produces a magnetization of the right magnitude and the wrong sign.
  Validated: Larmor omega = g mu_B B, RK4 energy conservation, and the low-T
  ferromagnet peaks fall on the exact LSWT dispersion.
  **ABSOLUTE INTENSITIES ARE COMPARABLE WITH LSWT as of 2026-08-13**, and the
  LINESHAPE can be windowed as of 2026-08-15 (`window: cosine`, opt-in -- pair it
  with `subtract_elastic: true`) -- see section 6a below.
* **KPM** (`magcalc/sun/kpm.py`) -- para-unitary Chebyshev expansion of the LSWT
  spectral function (Lane et al. / Sunny's `SpinWaveTheoryKPM`); O(D*M) matvecs, no
  eigensolve, for large SU(N)/entangled cells. Validated: converges to the engine's
  own exact `structure_factor` as the moment count grows, on a NON-COLLINEAR
  supercell and in the antisymmetric (`chiral`) channel as well as the symmetric
  ones -- two bugs lived in exactly the gap between those (2026-08-13, see
  `tests/test_kpm.py`), and both were invisible to a collinear test.
  **KPM NEVER DIAGONALIZES, so by itself it cannot notice it is expanding about a
  non-minimum**: no Cholesky, hence no positive-definiteness failure, hence nothing
  for `on_imaginary` to catch. It returns a smooth, plausible S(q,w) about a saddle
  or a maximum. The check therefore has to be made explicitly, and is -- guard 3
  above, `min eig H2(q) >= 0` at **every q computed**, run by the `kpm_sqw` task and
  by `examples/sunny_tutorials/S09_triangular_AFM/disorder_kpm.py`, both through the
  same `SUNModel.assert_stable(qs)`. **A model built or perturbed in a SCRIPT must
  call it itself** (`apply_bond_disorder` and friends are Python-only, so the runner
  never sees those models):

  ```python
  model.assert_stable(qs_cart)                 # raises unless H2(q) >= 0 at every q
  model.is_stable_at(q, hmat=H)                # one shifted Cholesky, reuses your H
  model.min_h2_eigenvalue(q)                   # how negative, for the report
  ```

  At Sunny's own disorder strength (sigma = 1/3) this refuses, and correctly: the
  relaxed 120-degree state is genuinely not a minimum there.

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
signed M_ch. Sign convention pinned to Sunny — `tests/test_polarized.py`.

Careful with "chiral vanishes for a collinear structure": that is true **per band only
when P ∥ q**. A collinear magnet's two magnons are degenerate and *oppositely handed*, so
for a general P the chirality is non-zero band by band and cancels only in the **band
sum** — and how it splits between the degenerate pair is basis-dependent, so it is not an
observable at all (Sunny and pyMagCalc split it differently, both correctly).

**Arbitrary polarization axes and Blume–Maleev frames** (`tests/test_polarization_frames.py`):

```yaml
calculation:
  cross_section: {polarization: [0, 0, 1], channel: sf}   # sf | sf+ | sf- | nsf
  # or a Blume-Maleev frame component (Sunny `ssf_custom_bm`):
  cross_section: {bm: {u: [1, 0, 0], v: [0, 1, 0]}, component: '23'}
```

`u`/`v` (or `normal`) are **Cartesian** lab vectors, as `domains.axis` is. The BM axes
follow Sunny: `e1 = q̂`, `e3 = the scattering-plane normal`, `e2 = e3 × q̂`; q outside the
plane is a hard error, checked up front rather than inside the per-q pool workers (where
it would come back as an all-NaN map). `P ∥ q` reproduces the plain `sf+`/`sf-` strings
exactly, and `SF + NSF = perp` for any P.

**Absolute normalization: pyMagCalc's S(Q,ω) EQUALS Sunny's.** Pinned band-by-band on a
ferromagnet (S = ½, 1, 2), a Néel antiferromagnet and a non-collinear helix by
`tests/test_absolute_normalization.py` + `tests/test_polarized.py`. This entry used to
say the opposite — "pyMagCalc's S(Q,ω) is 3/4 of Sunny's, a pre-existing convention
difference; do not compare absolute intensities" — which was wrong. The 4/3 lived in
hardcoded reference numbers in `test_polarized.py`, and that test compared only the
ratio chiral/perp, in which an overall factor cancels; the caveat then explained the
leftover away. A clean constant factor is a bug until proven otherwise (GAP_STATUS says
exactly this), so it should never have been documented as a convention.

The one real difference, which is *not* an overall factor: Sunny's `ssf_perp` applies
the g-tensor by DEFAULT, i.e. it measures moments (g·S) and is 4× ours at g = 2.
Compare against `ssf_perp(sys; apply_g=false)`.

**Ignoring temperature biases fits.** Not a rounding effect: on a ferrimagnet, fitting
40 K data with a T=0 model returns J = 1.07 instead of 1.30 (a 17% error), because the
Bose factor reweights the acoustic and optic branches *relative to each other* and a free
`scale` cannot absorb that. (On a simple AFM chain no bias is even possible: I ~ 1/(J f(q)),
so J only rescales the intensity and it carries no information about J at all.)

**Mixed spin.** The S(Q,ω) prefactor is √(S_i/2) **per site**. It used to be a single
global √(S/2), making every site whose S differed from the reference wrong by √(S_i/S_ref)
— a 60% error on a Cu(½)+Fe(2) model. The Fortran backend still applies the global
factor, so it now falls back to NumPy for mixed-spin S(Q,ω).

**Mixed spin, the CLASSICAL half** (fixed 2026-08-18). `spin_magnitude` is the
*reference* S — the first atom's `spin_S`, which is what `S_sym` binds to, with each
site's ratio already inside H(q). The ground-state search used it as a **length** as
well, moving on |m_i| = S_ref for every site. That is not a rescaling: the classical
energy is a quadratic form in free Cartesian components (already right for mixed spins)
and the lengths enter *only* through that constraint, so a uniform radius changes which
state minimises whenever the minimising directions depend on the lengths. An AFM trimer
with S = (1, 1, ½) closes its moment triangle at 151.0°/104.5°, E = −1.125 J; the old
code returned the 120° state at E = −1.5 J, which is not a state of that Hamiltonian at
all (evaluated honestly it is −1.0). Worse, `relax_from_current` scored the *correct*
structure the same wrong way and "relaxed" it downhill — so supplying the right answer
by hand was rejected too, and the model could not be run at any setting. `S_val`
(binds `S_sym`) and `S_vec` (the |m_i| = S_i constraint) are now separate everywhere;
`MagCalc._classical_spin_lengths` is the single place that decides the latter.
`tests/test_mixed_spin_classical.py` pins it on the closed-triangle law of cosines,
and pins that a uniform model is bit-identical seed-for-seed.

`thermal_mc` / `sampled_correlations` still **refuse** mixed spins outright
(`NotImplementedError`) — the sampler assumes one |m_i| throughout. 1/S `corrections`
and `scga` need nothing: corrections' outputs are per-site quantities of the Bogoliubov
transform (the moment reduction is ⟨a†_i a_i⟩, which carries no explicit S), verified
against the decoupled-sublattice identity, and `scga` reads `spin_magnitudes()` directly.


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

## 6a. The CLASSICAL S(q,ω) is on that same absolute scale (2026-08-13)

`sampled_correlations`, `static_correlations` and the SU(N)
`sun_sampled_correlations` are all normalized the way `calculate_sqw` and Sunny are:

    S^ab(q,ω) = (1/2π) ∫dt e^{-iωt} ⟨S^a(q,0)* S^b(q,t)⟩ / n_cells

**per CHEMICAL CELL, with the 1/2π of the time transform.** This entry used to say
the opposite — "the shape is validated, the scale is not; do not read an absolute
intensity off this path" — and the missing scale was two ordinary bugs, not a
convention:

* the time FFT was never normalized, so every classical S(q,ω) was **2π/dt too
  large** — 314× at the default `dt: 0.02`, and *proportional to 1/dt*, so refining
  the time step moved the intensity;
* the spatial sum was divided by the **site** count instead of the **cell** count, so
  a cell with `n_atoms` magnetic atoms came out `n_atoms` too small. Invisible to
  every test, because every classical model tested here had one site per cell.

The oracle is an exact identity, not a number: `∫dω S(q,ω) = ⟨S(q)*S(q)⟩/n_cells`,
checked against the same trajectory at machine precision on a one-site AND a two-site
cell (`tests/test_classical_absolute_normalization.py`, which also pins the free-spin
per-cell sum rule and the grid independence the missing 1/dt broke). The physical
close of the loop: a gapped low-T ferromagnet's `c2q`-corrected classical intensity
equals the LSWT band sum to ~2 %, and that band sum is pinned to Sunny in
`tests/test_absolute_normalization.py`.

### The time-domain window (2026-08-15) — OPT-IN, and pair it with `subtract_elastic`

```yaml
sampled_correlations:     {window: cosine, subtract_elastic: true}
sun_sampled_correlations: {window: cosine, subtract_elastic: true}
# default for both: window: rectangular, subtract_elastic: false
```

Nothing tapered the time correlation before the ω transform, i.e. a RECTANGULAR
window was implied, whose sidelobes fall only as 1/(ω−ω₀)² — and `c2q` grows LINEARLY
in ω out to the Nyquist frequency π/dt, 157 meV at `dt: 0.02`, so leakage off a 4 meV
band is amplified across a 40× wider axis (item 4 measured +16 % on the whole-axis
integral). `window: cosine` is Sunny's fix for exactly this.

**What it costs is exactly one bin, and that is provable rather than asserted:**
cos²(x) = ½ + ¼e^{2ix} + ¼e^{−2ix}, so windowing the correlation is IDENTICAL to
convolving the spectrum with the 3-point kernel [¼, ½, ¼] — one bin Δω = 2π/T of Hann
broadening. Two consequences follow and are pinned in
`tests/test_classical_window.py`: the kernel is non-negative, so the windowed S(q,ω)
cannot dip below zero where the raw one was positive; and it sums to 1, so **the
two-sided ω-integral and every sum rule are preserved to machine precision** — which
is also why `tests/test_classical_absolute_normalization.py` is structurally blind to
this change and the identity above had to be the oracle instead.

**WHY IT IS OPT-IN HERE AND SUNNY'S DEFAULT THERE — measured, not preference.** That
same one-bin smear lands on the ELASTIC delta of an ordered magnet, and `c2q` is 1 at
ω = 0 but |ω|/kT one bin away — **31** at kT = 0.005 with Δω = 0.153 meV. On the
gapped ferromagnetic chain (L = 24, q = 0.15; LSWT `perp` band sum 0.5):

| window | `subtract_elastic` | whole-axis / LSWT | first inelastic bin |
|---|---|---|---|
| rectangular | false | 1.55 | 0.00006 |
| rectangular | true | 1.40 | 0.00006 |
| cosine | false | **2.60** | **9.10** ← 18× the whole band sum, from one bin |
| cosine | true | 1.40 | 0.00005 |

The two windows agree once the delta is removed — it *was* the entire difference. So
turn the window on for lineshape work, and turn `subtract_elastic` on with it.

**Forgetting the pairing is REPORTED, not left to this page** (2026-08-16,
`classical_dynamics.check_elastic_leakage`, on both entry points and both config
blocks). The warning fires only when all four hold — `window: cosine`,
`subtract_elastic: false`, `classical_to_quantum` on, and the amplification c2q(Δω) ≥ 2
(kT ≲ Δω/1.6, which is the condition that makes it dangerous) — and it names the factor
it triggered on. It reports rather than deciding: making `cosine` imply
`subtract_elastic` would silently change what the config asked for.

```yaml
sampled_correlations: {window: cosine, on_elastic_leakage: warn}   # warn | error | off
```

One thing left to know when you read an absolute number off this path:

* **`classical_to_quantum` does not make a classical spin quantum.** It fixes the
   Bose weighting; the classical spin still has |S| = S where the quantum one has
   S(S+1), and a classical mode still softens at finite kT (21 % at kT = 0.15 on the
  SU(N) chain). Compare at kT ≪ ω, and mind that a gapless magnet in a small cell
  has a wandering order parameter that inflates the transverse weight (45 % on an
  L = 20 Heisenberg chain at kT = 0.02) — a finite-size effect that reads exactly
  like a normalization error.

**The CP^(N−1) sampler adapts its step size (2026-08-15).** `sigma` does not change
the stationary distribution, only how fast the chain reaches it — which made a fixed
`sigma` formally harmless and practically decisive: at low kT nearly every proposal
was rejected, the chain barely moved, and `sun_sampled_correlations` returned the
spectrum of its starting state. Measured: the SU(N) classical intensity swung
**0.30 → 1.63** on `therm_sweeps`/`sigma` alone. It is now tuned toward
`target_acceptance` (~0.5) over the first half of the thermalization and held fixed
for the second half (adapting while measuring would break detailed balance), and the
residual energy drift over that fixed half is checked against its own fluctuation:

```yaml
sun_sampled_correlations:
  adapt_sigma: true          # default
  target_acceptance: 0.5
  on_unequilibrated: warn    # warn (default) | error | off
```

Pinned in `tests/test_sun_sampler_equilibration.py` against the sampler's OWN
partition function in closed form — for a decoupled site the coherent-state energy is
Σ_i a_i |z_i|², the Fubini–Study measure makes |z|² uniform on the simplex, so
Z(β) = Σ_i e^{−βa_i} / Π_{j≠i} β(a_j − a_i) exactly. That pins the sampler with no
spectrum and no reference code in the way.
