# Plan — closing the 12 remaining Sunny gaps (Gap 4)

Companion to `GAP_STATUS.md` §"Gap 4 — Still open" (audit of 2026-08-03). One entry
per open item, ordered into phases by value/effort. Effort figures are estimates in
working days for someone who already knows this codebase; treat the *ordering* as the
firm part and the numbers as indicative.

**Keep this file updated when an item moves, and delete it when the table is empty.**

> **Status 2026-08-04.** All four phases worked through. Closed: #17, #19, #23, #27
> (Phase 1); #25, #21, #24a (Phase 2); #18, #22 (Phase 3); #16a, #16b, #26 (Phase 4).
> Open: **#24b** (estimate revised 3 d → 1 w, derivation written up below) and the
> classical S(q,ω) absolute normalization opened by #17. **#20 closed 2026-08-04.**
>
> Two SHIPPED BUGS were found along the way, both in field handling and both silent:
> the Zeeman term was dropped entirely in `mode: SUN`, and `H_dir` was flattened to a
> scalar so every field in every engine was forced along +z. Neither was visible to a
> suite in which no model applies a field off the z axis. See GAP_STATUS's trap list.
>
> Tutorials 06 and 09 remain unported, but **no longer for want of engine
> capability** — both are now reference-state problems. See GAP_STATUS.

---

## The rule every item obeys

`GAP_STATUS.md` closes with the project's institutional memory: the engine has
repeatedly produced plausible-but-wrong spectra, and *every* one was caught by an
independent oracle or an exact identity, never by inspection. The 2026-08-03 audit
added two more failure modes that are specific to this list:

- **A self-consistent test is not a test.** The form-factor table was wrong for as
  long as it was, because the only check compared `I_ion/I_bare` against
  `get_form_factor(...)**2` — the same wrong number on both sides. For every item
  below, the "Oracle" line must be something *outside* the code being added.
- **A documented convention is not an excuse.** The "S(Q,ω) is 3/4 of Sunny's"
  caveat survived because it explained a discrepancy away. If an item lands with a
  clean constant factor between us and Sunny, that is a bug until proven otherwise.

Two more rules carried over from `CLAUDE.md`:

- every item keeps at least one **fast** pinned test (outside the `slow` marker), so
  the 4-minute suite touches the new code path;
- `pytest -m ""` (from the workspace root) is the merge gate, not `pytest`.

**Refuse before you approximate.** Several items below are large. Where an item is
deferred, the engine must raise a `NotImplementedError` naming the alternative rather
than compute something plausible — the pattern now in
`sun/lswt.py::_reject_unsupported_terms`. Silence is the failure mode this project
exists to avoid.

---

## Phase 1 — quick wins (~1 week total) — ✅ DONE (2026-08-03)

Small, self-contained, and each removes a "not supported" from a path users already
hit. Do these first; they are also the cheapest way to re-familiarize with the
relevant modules.

### #17 Classical-to-quantum correction for `sampled_correlations` — ✅ DONE (~1 day)

**What.** `magcalc/classical_dynamics.py` returns the classical S(q,ω), which is not
on the quantum intensity scale. Sunny multiplies by the correspondence factor
`|ω/kT| / (1 − e^{−ω/kT})` (`SampledCorrelations/DataRetrieval.jl:128`). Add it,
keyed off the existing `sampled_correlations.temperature`, with an explicit
`classical_to_quantum: true|false` escape hatch. Also add
`set_spin_rescaling_for_static_sum_rule!`'s κ as a config knob.

**Where.** `classical_dynamics.py` (intensity assembly), `runner.py` (plumb the key).

**Oracle.** Sunny `intensities(sc, qpts; energies, kT)` on the same thermalized
model; and the low-T limit, where the corrected classical S(q,ω) must approach the
LSWT intensities the engine computes independently (those are themselves pinned to
Sunny, so this is a genuine cross-check, not a self-comparison).

**Risk.** Low. Pure post-processing; no change to any existing number when the key
is absent.

### #23 Domain averaging in SU(N) / entangled — ✅ DONE (~1 day)

**What.** `sun/adapter.py:155` and `sun/entangled.py:274` raise. The dipole path
already does this at `core.py:2214-2243` by rotating q per domain and averaging with
weights — the logic is q-space only and applies unchanged.

**Where.** Lift the `_parse_domains` + averaging loop out of
`core.calculate_sqw` into a shared helper (`numerical.py`), and call it from both
SU(N) and entangled `calculate_sqw`. Keep the existing guard that refuses lab-frame
components (`xx`, `zz`, …) — rotating q alone would be silently wrong there, and
that reasoning is unchanged.

**Oracle.** Exact identity, no Julia needed: build the *rotated structure*
explicitly, compute its S(q,ω), average by hand, and require equality with the
domain-averaged result. Plus Sunny `domain_average`.

**Risk.** Low, provided the cross-section guard is carried over rather than
reimplemented.

### #19 Static / energy-integrated correlations — ✅ DONE (~2 days)

**What.** Two distinct things sharing a name in Sunny:

1. `intensities_static` for LSWT — the energy-integrated S(q), i.e. the band sum of
   what `calculate_sqw` already returns. Expose as `tasks: {static_sqw: true}`.
2. `SampledCorrelationsStatic` — instantaneous classical correlations
   `⟨|S(q)|²⟩` averaged over Metropolis samples, with no dynamics at all. This is
   `thermal_mc`'s sampler plus one FFT; it does not need `classical_dynamics.py`.

**Where.** (1) `core.py`/`runner.py`; (2) a new small entry point in
`thermal_mc.py`, reusing `build_supercell`.

**Oracle.** (1) Sunny `intensities_static`, plus the total-moment sum rule
`∫ S(q) d³q ∝ S(S+1)`-type identity. (2) Sunny `SampledCorrelationsStatic`, plus the
high-T limit where S(q) must go flat, plus agreement with `scga.py`'s S(q) in the
regime where SCGA is valid (two independent in-repo routes to the same quantity).

**Risk.** Low. Note (1) is *not* the same as (2) — do not let one test cover both.

### #27 Crystal utilities — ✅ DONE (~2 days)

**What.** `primitive_cell`, `standardize`, `subcrystal` (all thin spglib wrappers —
spglib is already a dependency) and `print_irreducible_bz_paths`. Expose through the
existing `magcalc symmetry` CLI rather than inventing a new command.

**Where.** `magcalc/cif_utils.py` + `magcalc/cli.py`; `MagCalcConfigBuilder` already
provides the structure handle (`config_builder.py`).

**Oracle.** Sunny's `primitive_cell`/`standardize` on the same CIFs, and spglib's own
`standardize_cell` for the round-trip identity (standardize∘primitive is idempotent).
For BZ paths, `seekpath` is the natural third-party check — prefer depending on it to
porting Sunny's path tables by hand.

**Risk.** Low, but resist scope creep: this is plumbing, not a new engine.

---

## Phase 2 — parity for work you would publish — ✅ 3 of 4 DONE (2026-08-03)

These affect results, not convenience. Each is contained but touches validated code,
so each needs its identity test *before* the refactor, not after.

### #25 Blume–Maleev / arbitrary polarization frames — ~2 days

**What.** Cross-sections are currently P ∥ q only (`perp`, `trace`, `chiral`, `sf±`,
components). Sunny's `ssf_custom_bm` supports an arbitrary polarization axis in the
BM frame. Extend the `cross_section` spec to accept
`{polarization: [x,y,z], channel: sf|nsf}`.

**Where.** `numerical.py::contract_cross_section` (line 60) is the single contraction
point — the whole change should land there plus the validator in
`core.py:2196-2204`.

**Oracle.** Sunny `ssf_custom_bm`. Plus two exact identities that fail loudly: with
**P ∥ q** the new path must reproduce the existing `sf±` bit-for-bit, and
`SF + NSF` must equal the unpolarized total for any P.

**Risk.** Low–medium. The trap is the sign/handedness convention, which is exactly
what `tests/test_polarized.py` was built to pin — extend it rather than starting a
new file.

### #21 General pair couplings in SU(N) — ~3 days

**What.** Arbitrary two-site operators (Sunny `set_pair_coupling!`). **Most of this
already exists**: the audit added the operator-pair machinery for biquadratic, so
`SUNModel` already accepts a per-site operator list and an `(n_ops_i, n_ops_j)`
coupling matrix. What is missing is the front end: parse an operator (a polynomial in
`S_i`, `S_j`, or an explicit N²×N² matrix) and decompose it into `Σ_k A_i^k ⊗ B_j^k`
— an SVD of the reshaped tensor, which is Sunny's `svd_tensor_expansion`.

**Where.** New `interactions.pair_operator` in `schema.py`; decomposition helper in
`sun/operators.py`; wire into `sun/lswt.py::from_generic_model` next to the
biquadratic block.

**Oracle.** Sunny `set_pair_coupling!`. Plus the internal exact identity that costs
nothing: `(S_i·S_j)²` entered as a *general* operator must reproduce the dedicated
biquadratic path to machine precision — if the decomposition is wrong, that fails.

**Risk.** Medium. Keep the `n_ops` growth opt-in as biquadratic does; a general
operator on a large N is expensive.

### #24a Mixed-spin SU(N) — ✅ DONE (~3 days)

**What.** `sun/lswt.py:130` raises when sites have different N. The engine assumes a
uniform `M = N − 1` and computes block offsets as `i*M`.

**Where.** Replace the scalar `M` with a per-site `M_i` and a precomputed offset
array `offs[i]`; `D = Σ M_i`. Touches `_prepare`, `hamiltonian`, `_bogoliubov`,
`structure_factor`, and `sun/kpm.py` — all of which index with `i*M`.

**Oracle.** Exact identity first, Sunny second: a model of two **decoupled**
sublattices with S=½ and S=1 must return exactly the two independent single-species
spectra and intensities. That catches every offset/normalization slip. Then a real
mixed-spin ferrimagnet against Sunny `:SUN`.

**Risk.** Medium–high — this is the validated core. Do the decoupled-sublattice test
*first*, confirm it passes on the current uniform-N code, and only then refactor.

### #24b Ewald + rotating-frame single-k — **DONE 2026-08-13**

**Status: implemented, validated, shipped.** `tests/test_ewald_spiral.py`. Read the
correction at the end of this section before the plan text below it: transcribing
Sunny's expression, as steps 1–4 tell you to, produces a WRONG Hamiltonian in
pyMagCalc's Fourier gauge, and the wrong one passes every test that existed at the
time. The plan below is kept because its reading of Sunny is right and its two-stage
oracle is the one that worked; only the transcription step was too naive.

**Status when written: not implemented, but no longer a derivation problem.** Sunny.jl is MIT
licensed and in-repo at `../Sunny.jl-main`; this project already takes conventions
from it (the Stevens table in `stevens.py` was generated from it). Take the METHOD
from `src/Spiral/SpinWaveTheorySpiral.jl` rather than re-deriving — my own attempts
produced three wrong characterizations in a row (see the history at the end).

**THE KEY INSIGHT, and it makes the item much smaller: Ewald is not special-cased at
all.** `fourier_bilinear_interaction!` (line 54) builds the Fourier bilinear matrix
`Jq` from the exchange bonds and then simply ADDS the dipolar term into the same
matrix (line 78):

```julia
if !isnothing(sys.ewald)
    Aq = precompute_dipole_ewald_at_wavevector(cryst, (1,1,1), demag, -q_reshaped) * μ0_μB²
    for i in 1:Na, j in 1:Na
        Jq[i, j] += gs[i]' * Aq[i, j] * gs[j]          # note the g-tensors
    end
end
```

Everything downstream — all the rotating-frame channel algebra — then operates on
`Jq` without knowing or caring that part of it came from an infinite lattice sum. So
there is no separate "Ewald channel machinery" to build. pyMagCalc already has both
halves: `core._ewald_A(q_rlu)` is the analogue of `precompute_dipole_ewald_at_wavevector`,
and `_ewald_g()` supplies the g-tensors.

**CORRECTED 2026-08-12 — the three/five split was written down BACKWARDS here.**
Re-read `SpinWaveTheorySpiral.jl` lines 129–138. With `R2 = axis axisᵀ` and
`R1 = (I − i[axis]× − R2)/2`, the branch is on `k_case`, and it is:

```julia
# k_case 2  (2k integer -- the satellites COINCIDE):  FIVE terms
J = R2*J(q)*R2 + conj(R1)*J(q+k)*conj(R1) + R1*J(q−k)*R1
                + R1*J(q+k)*conj(R1) + conj(R1)*J(q−k)*R1

# k_case 3  (generic incommensurate -- the COMMON case):  THREE terms
J = R2*J(q)*R2 + conj(R1)*J(q+k)*conj(R1) + R1*J(q−k)*R1

# k_case 1  (k integer): no projection at all, J = J(q)
```

The earlier text here said "FIVE terms … dropping to three when the satellites
coincide", which is the exact inverse: the five-term form with the two cross terms
is the `k_case 2` special case, and the generic incommensurate case is the plain
three-term one. Implementing from the old description would have put the cross
terms into the common branch — a wrong Hamiltonian that still diagonalizes and
still produces a plausible spectrum. `k_case` is defined identically on both sides
(`SpiralEnergy.jl:12` / `generic_model.py:236`): 1 = k integer, 2 = 2k integer,
3 = otherwise.

My guesses were wrong in both directions — first three terms (assuming commutation),
then nine (assuming none of it collapses). The cross terms `R1 J(q+k) R1*` and
`R1* J(q−k) R1` are exactly the ones a commutation-based derivation drops and a naive
9-term expansion over-counts.

**THE INJECTION POINT IS EXACT** (checked 2026-08-04). `core._ewald_nambu(q_cart)`
already does, for the non-spiral path, precisely what Sunny's
`fourier_bilinear_interaction!` does for the Ewald half:

    Jq = exchange_from_A(self._ewald_A(q_rlu), self._ewald_g())     # = gs[i]' Aq gs[j]
    J0 = exchange_from_A(self._ewald_A(zeros(3)), self._ewald_g())

and then builds the standard Nambu blocks from `Jq`/`J0`. So the whole change is a
`_ewald_nambu_spiral(q_cart, k, axis)` that computes the SAME blocks from the
rotating-frame combinations

    # generic incommensurate (k_case 3) -- THREE terms:
    Jrot = R2 Jq(q) R2 + R1* Jq(q+k) R1* + R1 Jq(q-k) R1
    J0rot= R2 J0(0) R2 + R1* J0(+k) R1* + R1 J0(-k) R1
    # k_case 2 additionally carries + R1 Jq(q+k) R1* + R1* Jq(q-k) R1 (and same for J0)

(projectors acting on the 3x3 spin indices of each (i,j) block), and then hands the
result to the three-channel worker the way `dip_pairs` is already handed to it in
`calculate_sqw`. Estimated ~1 hour with the formula in hand; the existing refusal in
`core.py` names the exact spot.

**Implementation sketch for pyMagCalc.**

1. Build the rotating-frame `J(q)` including the Ewald contribution — i.e. make the
   single-k channel evaluation consume a Fourier matrix that already has `A(q)`
   folded in, mirroring `fourier_bilinear_interaction!`. Mind the g-tensors.
2. Apply the five-term projector combination at `q`, `q±k`, plus the same for the
   `q = 0` on-site term (`J0`, line 134 — it is built identically from `J(0)`,
   `J(±k)`).
3. Keep the `k_case` branch: three terms when 2k is a reciprocal-lattice vector.
4. Delete the refusal in `core.py`.

**Oracle, unchanged and still the right gate:** at commensurate k the same physics is
reachable via `magnetic_supercell`, which already supports Ewald, so the result must
agree band-for-band. Do that first. Then — and this is worth the extra step, since
the incommensurate case has no independent check — compare directly against Sunny's
`SpinWaveTheorySpiral` with `enable_dipole_dipole!` at incommensurate k.

**History, as a caution.** This item was estimated at 3 days, then a week, then
"possibly structurally invalid", then "nine terms" — four characterizations, each
from a closer look that changed its nature rather than its size, and three of them
wrong. Ten minutes reading the reference implementation settled it. The lesson is not
subtle: when a validated implementation of the same physics is sitting in the repo as
an oracle, read it before deriving.

---

**WHAT ACTUALLY HAPPENED (2026-08-13), i.e. the fifth correction.**

Steps 1–4 above were carried out verbatim in `3a986ac`, and the result was WRONG.
Reading Sunny correctly is necessary and not sufficient: **the two codes phase `A(q)`
differently**, and the projector algebra is not invariant under that.
`ewald.dipole_ewald_at_q` multiplies by `exp(i q·dr)` over the FULL bond vector (to
match pyMagCalc's symbolic `H`), where Sunny phases over lattice translations only.
Writing the `P_a A P_b` decomposition of `U(-θ_i) A U(θ_j)` with
`P ∈ {R2, R1, R1*}` of charge `q_P = 0, +1, -1`, summing over cells and folding in
that gauge, the coefficient of `P_a A P_b` is

    exp(i (q_b - q_a) k·r_i) · A(q + q_b k)

so in pyMagCalc's convention **R1 pairs with q+k and R1\* with q−k — the mirror of
Sunny's** — and the `k_case 2` cross terms carry a per-ROW phase `exp(±2i k·r_i)`
(their absolute-phase factor is unity because 2k is a RLV, but the intra-cell part of
the full-bond gauge survives). It is not symmetric in (i, j) and must not be
symmetrized.

**Why this was invisible.** The mirrored assignment is *identical* whenever `A(q)` is
uniaxial about the spiral axis (then `R1 A R1* = 0`), and identical for any one-site
cell (`r_i = 0`). Both hold in every model this repo already used, so the wrong
version reproduced the `magnetic_supercell` answer to 1e-15 on the first oracle model
tried. It took a cell with TWO sites at an intra-cell offset AND the Sunny `S0`
spin-direction convention (the position-spiral `local_directions` convention has zero
relative phase in the rotating frame and also hides it) to separate them.

**What the two-stage oracle actually needed.**

1. The commensurate-k-vs-supercell identity is exact only where the method is —
   `k_case 2` always, and `k_case 3` only when `A(q)` is uniaxial about the axis. The
   test models therefore put k and q along a 4-fold axis. The `k_case 3` three-term
   form otherwise drops the ±2k umklapp, which is a real ~10–20 % error on the
   dipolar part, now warned about by `core._check_ewald_spiral_validity`.
2. `k_case 2` is the ONLY way to test the cross terms: with a uniaxial `A` they are
   exactly zero, so a non-uniaxial lattice is required, and 2k ∈ RLV is what makes
   the supercell identity exact there anyway.
3. Sunny at incommensurate k agreed to 1.3e-8 once the gauge was right. Note Sunny
   *refuses* (PosDefException) most non-uniaxial spiral + dipole models — a U(1)-
   breaking dipolar term generally destabilizes the spiral — which is a second reason
   the useful test regime is the uniaxial one.

The earlier note that the harness "does not agree even with Ewald off" was itself
wrong: the no-Ewald control is exact to 1e-15 at commensurate k, including for a
state that is not the ground state. That attempt had imposed a 120° spiral on a plain
AFM chain (whose spiral minimum is k = 1/2, not 1/3) and, separately, risked
comparing the two descriptions at q converted through *different* B-matrices — the
supercell path keeps q in CHEMICAL RLU, so mixing them shifts q by a factor of the
supercell dimension.

---

## Phase 3 — new machinery — ✅ #18, #22 DONE (2026-08-04); #20 deferred

Genuinely new code rather than extensions. Independent of each other; can be done in
any order or in parallel.

### #18 Langevin thermostat + `ImplicitMidpoint` + `suggest_timestep` — ✅ DONE (~3 days)

**What.** `classical_dynamics.py` thermalizes by Metropolis and evolves with undamped
RK4. Add (a) Langevin dynamics (damping λ + noise) as an alternative thermalizer,
(b) `ImplicitMidpoint`, which conserves energy and |S| *exactly* rather than to
O(dt⁴), for the measurement trajectory, and (c) a `suggest_timestep` helper from the
maximum local field.

**Where.** `classical_dynamics.py`, alongside the existing `_deriv`/RK4.

**Oracle.** Already in the repo, which is what makes this cheap: `tests/test_thermal_mc.py`
pins the exact classical results Langevin must reproduce — N free spins in a field give
the Langevin function `−L(βgμ_B|B|S)`, and the classical Heisenberg dimer has closed-form
`⟨E⟩(T)` and `C(T)`. A thermostat that does not sample Boltzmann fails those.
For ImplicitMidpoint: energy drift must be *bounded* over a long run where RK4's drifts
secularly.

**Risk.** Low–medium. ImplicitMidpoint needs a fixed-point iteration per step; get its
convergence tolerance into the config rather than hardcoding it.

### #22 Wang–Landau — ✅ DONE (~3 days)

**What.** The one Tier-2 remnant. Flat-histogram sampling of the density of states
g(E), then thermodynamics from g(E) at any T in one run.

**Where.** `thermal_mc.py`, reusing `build_supercell` and the existing Metropolis
proposal machinery.

**Oracle (CORRECTED in flight).** Beale's exact 2-D Ising g(E) does NOT apply: these
are continuous classical Heisenberg spins, not Ising. The exact result that does is
better, because it pins g(E) itself: for ONE classical dimer, E = J S^2 cos(theta)
and cos(theta) is uniform for random unit vectors, so **g(E) is exactly constant** on
[-JS^2, +JS^2]. Then: C(T) reconstructed from
g(E) must match the parallel-tempering C(T) that `test_thermal_mc.py` already
validates, over the whole T range.

**Risk.** Low. Self-contained, and the oracle is unusually strong.

### #20 Experiment-data binning — ✅ DONE (2026-08-04, ~2 h)

**What.** Bin computed S(q,ω) onto an experimental histogram grid, and read NeXus
files. Today `fitting.py::load_fit_data` (line 73) reads CSV only via `np.loadtxt`.

**Where.** New `magcalc/binning.py`; extend `load_fit_data` to dispatch on extension;
`h5py` becomes a dependency (optional, imported lazily — a missing h5py must raise a
clear "install h5py to read .nxs" rather than a traceback).

**Oracle.** Sunny `load_nxs` + `BinningParameters` on the same file. Plus the
conservation identity: total counts must be preserved under rebinning, and binning a
model onto a grid then integrating must equal the direct integral.

**Risk.** Low technically, but **check the value first**. The existing CSV fitting path
may already cover how this group actually gets data out of its reduction pipeline; if
so this drops to Phase 4. Ask before building.

---

## Phase 4 — ✅ #16 step 1 DONE (2026-08-04); the rest still gated

Both are multi-week. Neither is on the critical path for anything currently in
`examples/`. Do not start either without a specific calculation that requires it.

### #16 Site-level inhomogeneity — ✅ BOTH STEPS DONE (2026-08-04)

**What.** Vacancies, per-site fields/couplings, open boundaries (Sunny
`to_inhomogeneous`, `set_vacancy_at!`, `set_field_at!`, `set_exchange_at!`,
`remove_periodicity!`). Blocks all dilution/disorder work; Sunny's example 09 is
built on it.

**Where — and why it splits in two.** pyMagCalc's LSWT front end is symbolic, per
unit cell, with periodic bonds; per-site disorder is structurally foreign to it. But
the **classical** modules already build explicit real-space supercells:
`thermal_mc.py::build_supercell` (line 40) returns `(H, b, N, S, pos)`. Vacancies are
then just zeroed rows/columns of `H` and per-site overrides of `b` — contained and
cheap. So:

1. **Classical first** (~3 days): a `disorder:` block (site list, or concentration +
   seed) applied at supercell construction, shared by `thermal_mc`,
   `classical_dynamics` and `annealing`; plus `periodic: [bool,bool,bool]` for open
   boundaries.
2. **LSWT after** (~1 week): disorder needs a large supercell and no eigensolve —
   which is exactly what `sun/kpm.py` is for. This is Sunny example 09's recipe, and
   the KPM engine is already validated, so the work is the disordered-supercell
   builder, not new spectral machinery.

**Oracle.** The dilution limit x→0 must reproduce the clean result exactly (and
x→1 the empty one). A single vacancy in a large cell against Sunny's example-09
setup. Self-averaging: results must be stable across disorder seeds — report the
spread, and *fail* if it exceeds a tolerance, rather than quietly returning one
realization.

**Risk.** High, mostly in scope. Ship step 1 and stop if that answers the question.

### #26 SU(N) classical dynamics — ✅ DONE (2026-08-04, incl. dissipative quench)

**What.** Finite-T classical dynamics for entangled units. Requires evolving SU(N)
coherent states under the CP^(N−1) equations of motion — a different integrator from
the dipole Landau–Lifshitz one in `classical_dynamics.py`, not a wrapper around it.

**Oracle.** The dipole limit: for N=2 the CP^(N−1) dynamics must reduce *exactly* to
the existing Landau–Lifshitz result — the same "S=½ SU(N) ≡ dipole" gate that
load-bears in `tests/test_sun.py`. Then Sunny.

**Outcome.** The equations of motion turned out to be the cheap part -- the generator
is `SUNModel.local_field`, already built for the ground-state search, so propagating
instead of minimizing is a few dozen lines. The N=2 dipole-limit gate passes to
4.8e-10, which is the strong result here.

What did NOT close is the finite-T S(q,ω) built on top: it peaks at roughly HALF the
SU(N) LSWT energy (1.95x, stable across kT, with real spectral weight, so not noise).
The EOM and integrator are independently verified, so the defect lies between the
trajectory and the spectrum -- CP^(N-1) thermal sampling, the moment operator, or a
genuine factor in the N > 2 correspondence. Left as a visible xfail rather than
guessed at. **S04 and S06 remain blocked** on resolving it.

---

## Summary table

| Phase | # | Item | Est. | Risk | Primary oracle |
|---|---|---|---|---|---|
| ✅ 1 | 17 | Classical→quantum correction | 1 d | low | Sunny `intensities(...; kT)`; low-T → LSWT |
| ✅ 1 | 23 | Domain averaging in SU(N)/entangled | 1 d | low | hand-rotated structure (exact identity) |
| ✅ 1 | 19 | Static / energy-integrated correlations | 2 d | low | Sunny `intensities_static`; SCGA agreement |
| ✅ 1 | 27 | Crystal utilities + BZ paths | 2 d | low | spglib round-trip; seekpath |
| ✅ 2 | 25 | Blume–Maleev polarization frames | 2 d | low–med | Sunny `ssf_custom_bm`; P∥q reduces to `sf±` |
| ✅ 2 | 21 | General pair couplings | 3 d | med | biquadratic via the general path (exact) |
| ✅ 2 | 24a | Mixed-spin SU(N) | 3 d | med–high | decoupled sublattices (exact) |
| 2 | 24b | Ewald + rotating-frame single-k | **1 w** | med–high | commensurate k vs supercell (exact) |
| ✅ 3 | 18 | Langevin / ImplicitMidpoint | 3 d | low–med | existing exact Langevin-function tests |
| ✅ 3 | 22 | Wang–Landau | 3 d | low | Beale's exact 2-D Ising g(E) |
| ✅ 3 | 20 | NeXus binning | 4 d | low | Sunny `load_nxs`; count conservation |
| ✅ 4 | 16a | Vacancies + open boundaries (classical) | 3 d | — | exact restriction identity; analytic bond counts |
| ✅ 4 | 16b | Bond disorder in LSWT (via KPM) | ~2 h | — | σ=0 bit-identical; Hermiticity 9e-16; spread monotone in σ |
| ✅ 4 | 26 | SU(N) classical dynamics | ~1 d | — | N=2 reduces to Landau–Lifshitz to 4.8e-10; low-T S(q,ω) within 1.1% of the LSWT band |

Phase 1 landed 2026-08-03 (see GAP_STATUS.md for what each was pinned to).
Phases 2–3 total roughly four more weeks and close 7 of the 9 remaining line
items (#24 is two).
Phase 4 is deliberately open-ended.

## Non-goals

- **Not** matching Sunny's API surface for its own sake. The config file is this
  project's interface (`CLAUDE.md`, "config as single source"); an item is closed when
  a YAML key does the job, not when a Python function is named like a Julia one.
- **Not** GUI editors for these. The Studio passes unknown config blocks through
  verbatim (`tests/test_gui_passthrough.py` pins this), so every item lands usable
  from the apps without UI work. Add editors later, per demand.
- **Not** the Fortran backend. `fMagCalc` consumes pyMagCalc as a read-only oracle and
  falls back to NumPy; none of these items should touch it.
