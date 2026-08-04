# Plan — closing the 12 remaining Sunny gaps (Gap 4)

Companion to `GAP_STATUS.md` §"Gap 4 — Still open" (audit of 2026-08-03). One entry
per open item, ordered into phases by value/effort. Effort figures are estimates in
working days for someone who already knows this codebase; treat the *ordering* as the
firm part and the numbers as indicative.

**Keep this file updated when an item moves, and delete it when the table is empty.**

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

### #24b Ewald + rotating-frame single-k — ~1 week (REVISED UP)

**Status: not started. The estimate below was wrong and is corrected here.**

**What the plan originally said.** "The rotating frame builds three channels (q−k, q,
q+k) and each needs its own `A(q)`; `core._ewald_nambu` builds one." That reads like
plumbing. It is not.

**What the code actually does.** The rotating frame is baked into the SYMBOLIC
Hamiltonian: `generic_model` forms the effective per-bond coupling
`R_i^T J_ij R_j = R(φ)` (see `generic_model.py:1690-1728`) *before* the Fourier
transform, and the three-channel worker then just evaluates that symbolic H at
`q ± k` (`numerical.py`, `calculate_sqw_spiral_single_q`). The Ewald term cannot join
that route: `A(q)` is an infinite lattice sum added NUMERICALLY in the LAB frame
(`core._ewald_nambu`), so it never sees the rotation.

Rotating it means transforming the real-space dipolar coupling per pair,
`R_i^T A(r_ij + R) R_j`, and only then summing over images. Because `R_j` depends on
`k · (r_j + R)`, that sum carries `e^{±i k·R}` factors: the rotated dipolar Fourier
matrix for one channel is a **projector-weighted combination of `A(q_c)`,
`A(q_c + k)` and `A(q_c − k)`**, not `A(q_c)` alone. So it is a derivation (the
dipolar analogue of Toth & Lake's three-channel exchange result), plus an Ewald sum
at three shifted arguments per channel, plus the demagnetization/surface term needing
its own treatment in the rotating frame.

**Oracle (unchanged, and it is a good one).** At a COMMENSURATE k the same physics is
reachable through `magnetic_supercell`, which already supports Ewald. The
rotating-frame and supercell answers must agree band for band; if they do at several
commensurate k, the incommensurate case is sound. Do not ship the incommensurate path
without that.

**Until then** the engine refuses honestly (`core.py`, "Ewald dipole-dipole is not yet
supported together with a single-k (rotating-frame) structure"), and the message
already names both workarounds: a `magnetic_supercell`, or
`dipole_dipole.method: truncated`.

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

### #20 Experiment-data binning (`BinningParameters`, `load_nxs`) — ~4 days

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

### #16 Site-level inhomogeneity — step 1 ✅ DONE (~3 days); step 2 still open

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

### #26 SU(N) classical dynamics — ⚠️ PARTIAL (2026-08-04)

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
| ⏸ 3 | 20 | NeXus binning | 4 d | low | Sunny `load_nxs`; count conservation |
| ✅ 4 | 16a | Vacancies + open boundaries (classical) | 3 d | — | exact restriction identity; analytic bond counts |
| 4 | 16b | Disorder in LSWT (via KPM) | 1 w | high | Sunny example 09 |
| 4 | 26 | Entangled classical dynamics | 1–2 w | high | N=2 reduces to Landau–Lifshitz (exact) |

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
