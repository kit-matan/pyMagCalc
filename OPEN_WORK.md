# Open work — pick-up notes

Last updated **2026-08-13**, `master` at `00083a6` (pushed).

**Full gate GREEN on this tree: 672 passed, 3 skipped** (`pytest -m ""` from the
workspace root, 39:06). Trail: 621 (2026-08-12 baseline) → 635 after the
provenance + shadow-guard work (+5 `test_install_provenance`, +8
`test_shadow_guard`, +1 newly-discovered CCSF config) → 637 with the two
`test_ccsf_fit_roundtrip` tests → 650 with item 1's `test_ewald_spiral` → **672**
with item 2's (**+22**: 14 `test_sun_quench`, 5 S06 in `test_sunny_tutorials`,
3 `test_annealing` for item 9). Both `chore/open-work-housekeeping` and
`feat/open-work-followups` were fast-forward merged into `master` after a green
gate; both are level with it and can be pruned.

Items 1, 2, 9 and 3 are committed on `feat/s06-cp2-skyrmions`, **5 commits ahead of
`master`** (which is at `f819694`), unmerged and unpushed. Item 3 is two of them:
`5812975` the KPM fix (`magcalc/sun/kpm.py`, `tests/test_kpm.py`) and `ff9fbfe` the
S09 port — the fix stands on its own and is worth reading separately, since it
changes every KPM spectrum of a non-collinear model and every `cross_section:
chiral` KPM result.

**The full gate has NOT been re-run since item 3.** What has: the fast suite (568
passed, 2 skipped, 11:54) and `-m ""` on the four touched suites (35 passed, 6:16).
Run `pytest -m ""` from the workspace root before merging.

This file is the "what to do next" companion to `GAP_STATUS.md`, which is the
authoritative record of *what is done and how it was validated*. Read
`CLAUDE.md` first (config-authoring rules and the engine's hard errors), then
`GAP_STATUS.md` — in particular its closing section, "How things were validated
(and the recurring trap)", which is why the notes below all name an oracle.

**The house rule that shapes every item here:** a check a wrong answer passes is
not a check. Every fix needs an independent oracle — Sunny (in-repo at
`../Sunny.jl-main`), SpinW, an exact analytic identity, or exact
diagonalization — never a self-generated golden number, and never a published
number that has not been re-converged.

---

## Status at a glance

Every section below opens with a `**Status:**` line; this is the index. `PARTIAL`
means the hard question is answered but an action is still outstanding — those
are the easiest to mistake for done.

| # | Item | Status |
|---|---|---|
| 1 | Gap #24b — Ewald + rotating-frame single-k | **DONE** 2026-08-13 — oracle built, bug found, refusal lifted |
| 2 | Sunny S06 — skyrmion lattice | **DONE** 2026-08-13 — ported at L = 40; the size WAS the blocker |
| 3 | Sunny S09 — disorder + KPM | **DONE** 2026-08-13 — ported; found a KPM bug and a stability limit |
| 4 | Classical S(q,ω) absolute normalization | **OPEN** — shape pinned, scale not |
| 5 | Coverage follow-ups | **PARTIAL** — audit's 4 items done; 2 follow-ups + discovery shape open |
| 6 | `minimization.tolerance` silently ineffective | **DONE** 2026-08-12 |
| 7 | FeI2 dipole ground state | **PARTIAL** — physics answered; `examples/materials/FeI2` fix open |
| 8 | Studio open→run limits | **OPEN** — 2 items |
| 9 | `anneal`'s polish could return a MAXIMUM | **DONE** 2026-08-13 — fixed + swept; a RELATED defect is open |
| 10 | KPM has no ground-state guard | **OPEN** — opened by item 3 |
| A | `pytest.ini` collection scope | **DONE** 2026-08-12 |
| B | Engine provenance — `magcalc where` | **DONE** 2026-08-12 |
| C | Interpreter-startup shadow guard | **DONE** 2026-08-12 — `magcalc guard`; nothing outstanding |
| D | Stale OneDrive trees deleted | **DONE** 2026-08-12 — both rescued-file follow-ups closed |
| E | `mu_B` → `constants.py` | **DONE** 2026-08-12 — dipolar prefactor folded in too |

**All four small items that used to hang off C/D/E are now CLOSED** (2026-08-12):

- **C** — `magcalc guard [--install|--uninstall]`. The source moved into the
  package (`magcalc/_shadow_guard.py`), which fixed a hole bigger than the one
  logged: `tools/` is not shipped by a non-editable `pip install`, so a wheel user
  could not install the guard *at all*. Still per-interpreter by design — a fresh
  venv starts unprotected and `magcalc where` says so. A build hook that writes to
  site-packages remains deliberately out of scope.
- **D** — `tests/test_ccsf_fit_roundtrip.py` pins the rescued fit demo; and the
  rescued aCVO model is tracked at
  `examples/materials/aCVO/legacy_spin_model_sf_2024.py`.
- **E** — both dipolar prefactors now live in `constants.py`. **Their values were
  NOT reconciled**, and that is the point: `MU0_MUB2_MEV_A3/(4π) = 0.053681511`
  but `DIPOLE_PREFACTOR_MEV_A3 = 0.05368216` — 1.2e-5 RELATIVE apart, truncated
  independently. Deriving either from the other is an accuracy change, not a
  refactor: `test_truncated_sum_converges_to_ewald` asserts to 1e-4 absolute on a
  ~4 meV band, where that shift is ~5e-5 — the same order as the tolerance. A
  comment in `generic_model.py` claimed the division was exact; it is now
  corrected. Reconciling them deserves its own commit and its own oracle run.

---

## 1. Gap #24b — Ewald + rotating-frame single-k

**Status: DONE 2026-08-13.** `dipole_dipole: {method: ewald}` now works with a
`single_k` structure; the refusal in `core.py` is gone. Validated in
`tests/test_ewald_spiral.py` (12 tests, all fast). Nothing outstanding — the
notes below are kept because the *way* it failed is the reusable part.

**Building the oracle found a real bug in the machinery**, which is what the
refusal existed for. The three-term combination written in `3a986ac` was a
faithful transcription of Sunny `SpinWaveTheorySpiral.jl:129-138` — and wrong,
because **the two codes phase `A(q)` differently and the projector algebra is not
invariant under that regauging**: `ewald.dipole_ewald_at_q` multiplies by
`exp(i q·dr)` over the FULL bond vector (matching pyMagCalc's symbolic `H`),
Sunny phases over lattice translations only. Decomposing `U(-θ_i) A U(θ_j)` into
`P_a A P_b` with `P ∈ {R2, R1, R1*}` of charge `0, +1, -1` and summing over cells,
the coefficient of `P_a A P_b` is `exp(i (q_b - q_a) k·r_i) · A(q + q_b k)`, so:

- **R1 pairs with q+k and R1\* with q−k** — the mirror of Sunny's assignment;
- the `k_case 2` cross terms carry a **per-ROW** phase `exp(±2i k·r_i)`. It is not
  symmetric in (i, j); symmetrizing it fails.

**Why the bug was invisible, which is the transferable lesson.** The two
assignments are *identical* whenever `A(q)` is uniaxial about the spiral axis
(then `R1 A R1* = 0`) and for any one-site cell (`r_i = 0`). Every model in this
repo is one or the other. Separating them needs a cell with two sites at an
intra-cell offset AND the Sunny `S0` spin-direction convention — the
`local_directions` convention gives zero relative spiral phase in the rotating
frame and hides it just as well. The first oracle model built reproduced the
supercell answer to 1e-15 **with the bug in**.

**The oracle, in the order it had to be built.**

1. *No-Ewald control.* At commensurate k the rotating frame and the explicit
   `magnetic_supercell` are two cell choices for one lattice and one spin
   configuration, so the sorted band sets agree at the same **Cartesian** q. This
   holds to 1e-15 — including for a state that is NOT the ground state, since it is
   an algebraic identity, not a physical one.
2. *Ewald, in the regime where the identity is entitled to hold*: `k_case 2`
   always, and `k_case 3` only when `A(q)` is uniaxial about the axis. The test
   models put k and q along a 4-fold axis to arrange that, with the two sites at
   z = 0 and 0.3 so inversion is broken and the R1/R1\* order is observable.
3. *`k_case 2` with a NON-uniaxial `A`* — the only way to test the cross terms at
   all, since a uniaxial `A` makes them exactly zero, and 2k ∈ RLV is what keeps
   the supercell identity exact there.
4. *Sunny 0.8.1 at incommensurate k*, where no supercell exists: 1.3e-8.

**The earlier "the harness is broken" note was itself wrong.** The no-Ewald
control is exact; the 9e-4 residual seen on a chain at k = 1/3 was physics — the
±2k umklapp that the `k_case 3` form legitimately drops when `A(q)` is not
uniaxial about the axis. That approximation is real and not small (~10–20 % of the
dipolar shift), is the same one Sunny makes, and Sunny's `check_rotational_symmetry`
cannot see it because the dipolar term lives outside `interactions_union`.
`core._check_ewald_spiral_validity` now measures the dropped weight directly and
warns, following `magnetic_structure.enforce_rotational_symmetry`. Two traps in
that earlier attempt, worth keeping: the 120° state it used was not the ground
state of the plain AFM chain it was imposed on (k = 1/2 is), and the supercell path
keeps q in CHEMICAL RLU, so converting q through the supercell's own B-matrix
shifts it by the supercell dimension.

**Also worth knowing:** a U(1)-breaking dipolar term generally destabilizes the
spiral itself — Sunny raises `PosDefException` on most non-uniaxial spiral +
`enable_dipole_dipole!` models. So the warning above is usually telling you
something about the physics, not only about the method.

**Still true:** this blocks no shipped example — nothing in `examples/` combines
`single_k` with `dipole_dipole: {method: ewald}`. It was capability parity.

---

## 2. Sunny tutorial S06 — skyrmion lattice

**Status: DONE 2026-08-13.** Ported at Sunny's own L = 40, as a real quench —
`examples/sunny_tutorials/S06_CP2_skyrmions/{config.yaml, quench.py}`, with that
folder's README carrying the numbers. Nothing outstanding.

**Of the two questions listed here, the answer was #2, and #1 was innocent.** Worth
keeping, because the cheap-and-likely one was the wrong lead:

1. **The J2 bond shell was correct all along.** `Bond(1,1,[1,2,0])` is |a₁+2a₂| = √3
   with coordination 6, and `magcalc symmetry` finds exactly Sunny's shells
   (1.0 ×6, √3 ×6, 2.0 ×6). Now pinned (`test_S06_second_neighbour_shell_is_the_
   sqrt3_bond`), so it cannot come back as a suspect.
2. **System size was the blocker, via performance, by three orders of magnitude.**
   A skyrmion is several lattice constants across, so 64–256 sites cannot hold a
   liquid of them and relaxes to the uniform state — which reads as a physics
   failure and is a size failure. L = 40 cost ~16 s/step (≈55 h for the tutorial's
   25 600 steps) because `SUNModel.local_field(i, ·)` scans the whole bond list per
   site, i.e. O(sites²). `sun/dynamics.py` now sums the bond list once and forms
   only h_i|Z_i⟩, never the matrices h_i: **8.4 ms/step at 1600 sites, the full run
   in 214 s.** (`_replicate`'s cell lookup was then the bottleneck at 36 s; it is a
   dict now, 3.4 s.)

**The result.** The quench leaves an exactly quantized, non-zero SU(3) charge
(+12 → −4 → −6 at τ = 4, 16, 256) on a quadrupolar background, reproducing the
tutorial's figure. The Hamiltonian is pinned to **Sunny 0.8.1 at 5.4e-13**, via the
energy of an *arbitrary* coherent-state configuration rather than a ground state —
the latter would be far weaker, since the state relaxes to fit whatever Hamiltonian
it is handed.

**Three things this port turned up, none of which were the thing being looked for:**

- **The whole dissipative-quench API had NO test** — `damped_deriv`, `damped_step`,
  `quench`, `topological_charge`, `triangulate_lattice`. The "validated by
  dE/dt = −2λ·Var(h) to 5e-6" note in the old S06 README was a one-off measurement
  at a terminal, never pinned. `tests/test_sun_quench.py` is now that oracle (14
  tests, all exact identities), written *before* the derivative was rewritten.
- **`topological_charge` was undefined on exactly this model and did not say so.**
  It normalizes every spin, so the quadrupolar |m=0⟩ background (⟨S⟩ ≈ 0, most of
  the area here) contributed arbitrary directions and still returned a
  quantized-looking number. It now refuses, and `sun_topological_charge` — the CP^(N−1)
  Berry phase, which is what Sunny's tutorial actually plots — is the right tool. It
  is pinned to the dipole one by the exact N = 2 identity.
- **`method: anneal` could return a local MAXIMUM with full consensus** — see item 9.

**Do not "port" it by substituting an equilibrium calculation for the quench.**
That produces a folder that looks like a port and is not one. (The uniform SU(3)
ground state here is the non-magnetic |m=0⟩ state at E/site = 0; the skyrmions are
metastable and a minimizer destroys them.)

---

## 3. Sunny tutorial S09 — disorder + KPM on the triangular lattice

**Status: DONE 2026-08-13.** `examples/sunny_tutorials/S09_triangular_AFM/` now has
`config_supercell.yaml` (the 120° order as an explicit √3×√3 SU(N) cell) and
`disorder_kpm.py` (the tutorial's disorder + KPM protocol), pinned by
`tests/test_s09_disorder_kpm.py` (9 tests). Two things came out of it that were not
what the item was about — one of them a bug in shipped code, the other a physics
result that changes how the KPM feature should be used.

**The structure-geometry half went exactly as this item predicted.** The √3×√3 cell
is `magnetic_supercell: {matrix: [[1,1,0],[-1,2,0],[0,0,1]]}` — columns a₁−a₂ and
a₁+2a₂, both with k·A ∈ ℤ — and gives E/site = **−0.375 exactly**, with the bands
reproducing the analytic ω(q) = 3JS√[(1−γ)(1+2γ)] folded into {q−k, q, q+k} to 1e-13.
Worth keeping: the TRANSPOSED matrix has the same |det| = 3, cannot host the order
(k·(a₁+a₂) = 2/3), and returns a frustrated collinear state at +0.0833 — with a
perfectly plausible dispersion. Commensurability is the criterion; the energy is how
you see it. (Pinned, so it cannot come back.)

**The recorded diagnosis was wrong, and that is the transferable part.** This item
said the narrowing was "what expanding about a non-minimum buys you". The placeholder
reference state WAS wrong — but it was a ferromagnet at E/site = +0.75, not the
−0.3333 recorded here, and **fixing it did not fix the narrowing**. The narrowing was
a bug in `magcalc/sun/kpm.py`:

- the Chebyshev recursion ran on `D̂` where the structure factor needs `conj(D̂)`.
  The two are identical whenever `D̂` is real — every collinear, inversion-symmetric
  model, which is everything `tests/test_kpm.py` covered — so it survived until a
  **non-collinear supercell**: on the clean 81-site 120° cell it put ~5 % of the
  intensity onto bands that carry none, at LOW energy. A clean spectrum that arrives
  pre-broadened makes real disorder look like narrowing;
- found next to it: the moments are assembled from the annihilation block of the
  Nambu basis while `structure_factor` uses the creation block, and the two carry
  complex-conjugate weight matrices. The symmetric channels cannot see it;
  **`cross_section: chiral` came back sign-reversed** on a plain ferromagnet
  (relative error exactly 2.0, the signature of a flip).

Both are fixed, and `tests/test_kpm.py` now covers the two shapes the old suite
structurally could not — a non-collinear supercell and an antisymmetric channel.
Both new tests were confirmed to FAIL on the pre-fix code.

**A ground-state result that limits the feature, not just this port.** At Sunny's own
disorder strength σ = 1/3 the relaxed 120° state is **not a classical minimum**:
min eig H₂ reaches −1e-2 and |Im ω| reaches 0.16 meV on a 1.591 meV band. The
relaxation is not at fault — Metropolis annealing from three temperatures and a
damped CP^(N−1) quench return the same state to 8 decimals, and relaxing in cells of
2×, 3×, 4× the disorder period does not lower the energy. Disorder is destabilizing
the 120° order itself. It is stable for σ ≲ 0.1, which is what the port ships
(broadening +12.9 % at L = 12, +13.5 % at Sunny's L = 30, seed spread 0.4–1.2 %).

**The general point: KPM is the one path here with NO ground-state guard**, because
it never diagonalizes — no Cholesky, so no positive-definiteness failure, so nothing
for `on_imaginary` to catch. It returns a smooth, plausible S(q,ω) about a saddle.
`disorder_kpm.py` checks H₂ ⪰ 0 itself and refuses by default. **Wiring that check
into the runner's `kpm_sqw` task is the obvious follow-up and was NOT done** — see
item 10.

---

## 4. Classical S(q,ω) absolute normalization

**Status: OPEN.** Opened by #17. The classical-dynamics path's overall scale has never been
reconciled with the LSWT/Sunny one — **shape is pinned, scale is not**.

Treat a clean constant factor as a bug until proven otherwise. `GAP_STATUS.md`
records a case where a 4/3 was documented as a "convention difference" for
months and turned out to be wrong reference numbers in a test that only compared
a ratio, in which any overall factor cancels.

Natural oracle: the low-T classical ferromagnet, whose peaks already fall on the
exact LSWT dispersion (`tests/test_classical_to_quantum.py`,
`classical_to_quantum_factor`) — extend from position to weight.

---

## 5. Coverage follow-ups

**Status: PARTIAL.** The audit's own four items are done; two follow-ups and the
discovery-shape problem remain open.

The 2026-08-04/05 audit closed its four items (config smoke test, `kitaev`,
guard tolerances, combination matrix — see `GAP_STATUS.md` §"Config-surface
coverage audit"). Where the two recorded follow-ups stand:

- **Enumerate config keys from the CODE, not the docs.** STILL OPEN. The audit
  swept *documented* keys against `tests/`, so
  `calculation.imaginary_rel_tolerance` — in neither the docs nor the tests —
  was invisible to the very process meant to find gaps. Sweeping
  `calc_config.get(...)` call sites has been done for the `calculation:` block
  (clean, apart from internal `cache_file_base`); the other blocks (`tasks`,
  `plotting`, `minimization`, `scga`, `thermal_mc`, …) have not.
- **Escalate a whitelist of WARNINGs in the config smoke test.** STILL OPEN (it
  fails on ERROR log records only), but item 6's case no longer argues for it:
  that failure is now a hard error, so the smoke test catches it as an ERROR.
  What changed in its favour is the *prerequisite*: escalating warnings is only
  viable once benign ones stop firing routinely, and two that fired on
  correctly-written configs are gone (2026-08-12) — `num_starts <
  early_stopping` no longer warns for the Monte-Carlo methods, where
  `early_stopping` is meaningless and a handful of runs is the recommendation;
  and `plt.show()` on a non-interactive backend no longer warns, because
  `plotting.show_plot_if_possible()` does not call it.

**Discovery, not just execution (2026-08-12).** The smoke test's glob is
`examples/*/*/config*.yaml`, and `examples/fitting/fit_dispersion.yaml` is one
directory shallow AND not named `config*` — invisible on both counts. It went on
shipping as TUTORIAL.md's `magcalc fit` example with every bond listed in one
direction only (halving each J) and no `magnetic_structure` at all (expanding
about a stationary maximum), while its own "recovers the true values" check
passed, because the shipped data had been generated from that same broken model.
An `EXTRA` list now covers it, and the blanket `future_exmaples` exclusion is
gone — that exclusion is why its FeI2 config sat 2.5 meV/site above the ground
state (item 7).

**Mind the two numbers.** `examples/future_exmaples/` is gitignored
(`.gitignore:35`), so its four configs exist only in a working tree that has them:
coverage is **55 configs here, 51 on a fresh clone** — every runnable config under
`examples/` except the four in `SKIP`. Dropping the exclusion is therefore a
no-op for CI and only helps whoever has the staging directory locally. If those
configs are worth protecting from rot, they have to be tracked first (`git add
-f`, or un-ignore the directory); until then "staging is covered" is true only on
one machine. Re-check either number with

    python -c "import sys; sys.path.insert(0,'tests'); import test_config_smoke as t; print(len(t._configs()))"

after adding examples. The remaining shape problem is that discovery is still a
glob plus a hand-list: a config named neither `config*.yaml` nor listed in
`EXTRA` is still invisible.

---

## 6. ~~Loose end — `minimization.tolerance` is silently ineffective~~

**Status: DONE 2026-08-12.** Nothing outstanding.

Resolved both ways at once, because the swallow was the worse half of it.

- **A failed minimization is now a hard error** (`runner.py`, was
  `logger.warning("Optimization attempt using MagCalc failed: …")`). Carrying on
  meant expanding LSWT about a structure that was never minimized — and with
  `on_imaginary: warn/off` that returns a plausible spectrum and exit code 0,
  which is the house's #1 hazard class. The message names the likely cause
  (method-specific `minimization` keys) rather than blaming the magnetic
  structure downstream.
- **`tolerance` → `tol` and `max_iterations` → `options: {maxiter: …}`** for the
  gradient methods, with `float()` coercion: `tolerance: 1e-6` is a *string* to
  PyYAML (its float pattern needs a decimal point), which is how both configs
  that set one are written.

Both `examples/materials/FeI2/config_fei2.yaml` and
`examples/future_exmaples/CoRh2O4/config_corh2o4.yaml` had been running with
**no minimization at all**; they now actually minimize and still pass the
ground-state guards. The strict error also surfaced the Studio bug that motivated
the sweep — the apps injected the anneal-only `n_sweeps` into `method: TNC`
configs (see `GAP_STATUS.md`, "Open a config, press Run").

---

## 7. FeI2 dipole — the "needs investigation" note now has an answer

**Status: PARTIAL.** The physics question is ANSWERED (below). Two actions remain
open: `examples/materials/FeI2/config_fei2.yaml` still carries `on_imaginary:
warn` and cannot be fixed without first separating its Hamiltonian from its
structure; and the corrected `examples/future_exmaples/FeI2` config is gitignored,
so it exists in no commit.

`examples/materials/FeI2/config_fei2.yaml` opens with

    # on_imaginary: warn -- the supplied structure is not the exact classical minimum
    # of this Hamiltonian. Flagged by the ground-state guard; needs investigation.

**The answer (2026-08-12):** FeI2 orders as a COLLINEAR 2-up-2-down stripe at
k = (0, −1/4, 1/4), and no rotating-frame `single_k`/`propagation_vector` form can
represent it — that form rotates each successive cell by a fixed angle, giving
up / in-plane / down / in-plane at k = 1/4. On the real-space `magnetic_supercell:
[1, 4, 4]` (16 sites), annealing reaches **E = −46.372796 meV per cell =
−2.898300 meV/site**, reproducibly (3–4 of 4 runs hit it at seeds 0, 1, 2, 7).
That is the value the guard was reporting as unreachable, and the config's
declared structure sits ~2.5 meV/site above it. Note this is the DIPOLE minimum;
the SU(N) ground state is −2.91893118 meV/site, and with an anisotropy present
the two genuinely differ (CLAUDE.md §5c) — they are not meant to agree.

`examples/future_exmaples/FeI2/config_fei2.yaml` has been fixed this way and now
runs with the guards at their default `error` — but that directory is **gitignored**
(`.gitignore:35`), so the fix lives in the working tree only and is not in any
commit. Either track it (`git add -f`) or accept that it will be lost on a clean
checkout; the physics above is recorded here precisely so it survives either way.

**Why `examples/materials/FeI2/config_fei2.yaml` was NOT changed with it:**
`tests/test_sun.py` uses that file as its Hamiltonian source and builds its own
SU(N) supercell from it (`SUNModel.from_generic_model(m, supercell=MSUPER, ...)`),
reading `m.config["crystal_structure"]["lattice_vectors"]` to form the chemical
reciprocal basis. Adding `magnetic_supercell` there would silently hand those
Sunny-validated comparisons (E/site to 1e-6, bands and intensities to 1e-4) a
16-site cell and the wrong basis. Doing it properly means separating the
Hamiltonian from the structure in that config — split the interactions into a
fragment both configs include, or point `test_sun.py` at a structure-free copy —
and then re-running `pytest tests/test_sun.py -m ""` to confirm the Sunny numbers
are untouched. Until that is done, leave the `on_imaginary: warn` in place.

---

## 8. Studio — the two limits left after the 2026-08-12 open→run fix

**Status: OPEN — two items.**

`gui/src/lib/configIO.js` (web) and `MagCalcConfig.backendInput` (native) are now
two implementations of one rule — *the file is the base, write only real edits
over it* — kept honest by `tests/test_gui_roundtrip.py`, which drives the web one
over all 59 shipped configs and runs four of them against their own CLI run band
for band. Two things that rule does not reach:

1. **Relative paths from the web app.** A run happens in the opened file's
   directory only when the client sends `config_dir`. The native app can (it has
   a real `URL`); the browser's File System Access API exposes only
   `handle.name`, so a web-app run of a config with `from_mcif:` /
   `fitting.data_file:` / `cif_file:` fails with FileNotFoundError. It fails
   *loudly*, which is the acceptable end of the trade, but the fix is real: route
   the web app's Open through the server's `/load-config` (which returns an
   abspath and is currently dead code) behind a "recent files" picker, then send
   `config_dir` like the native app does.
2. **No second implementation is tested.** The Swift side has no equivalent of
   `roundtrip.test.mjs`; it compiles (`xcodebuild … MagCalcStudio-macOS`) and was
   fixed by inspection against the JS. The cheap oracle is the JS emitter itself:
   a small XCTest that loads the same example configs and diffs
   `backendInput()` against `node gui/tests/emit_run_config.mjs`. Until that
   exists, treat any change to one side as owing a matching change to the other.

---

## 9. `anneal`'s polish could hand back a local MAXIMUM, unanimously

**Status: DONE 2026-08-13** (fixed in `magcalc/annealing.py`, pinned in
`tests/test_annealing.py`; shipped configs swept, none affected). **A SEPARATE and
still-open defect was found next to it** — `thermal_mc.build_supercell` carries no
single-ion anisotropy at all; see the bottom of this item.

Found while writing the S06 config, not by looking for it. `method: anneal` is the
DOCUMENTED DEFAULT ground-state search (CLAUDE.md: "Prefer Monte-Carlo annealing…
more reliable and cheaper"), and it ended with a `steepest_descent` polish whose
result was taken **unconditionally**:

```python
if polish_steep:
    m_best, e_best = steepest_descent(m_best, H, b, c, S, n)
return m_best, e_best
```

`steepest_descent` aligns each spin with the field from everything EXCEPT itself, so
it ignores the on-site block `H_ii`. Its own docstring said so ("not exactly
optimal; the caller's L-BFGS polish cleans that up") — but the caller applies no
L-BFGS polish, it applies *this*, and then trusts it.

On S06 (easy-plane D = 19, one site per cell, so all 12 bonds fold into `H_ii`):

| | E | m_z |
|---|---|---|
| Metropolis, no polish | **−4.644706** (the exact minimum) | 0.4515 |
| after the polish | **+0.520665** (a local MAXIMUM) | 1.0000 |

on **every seed**, at 500 and 5000 sweeps — so `minimize_energy` reported
"4 of 4 runs hit the minimum", which is precisely the acceptance criterion the docs
prescribe ("Accept a ground state only when the energy is reproducible across
several `seed`s"). Reproducibility certified the wrong answer. The pole is a fixed
point of the field-alignment iteration (there the only field is the Zeeman one), and
it is a stationary point of the constrained problem, so a projected gradient will not
leave it either.

Both halves are fixed: the polish is kept only if it lowers the energy, and
`steepest_descent` returns the best state it saw, which is the monotonicity its
docstring always claimed.

**Why nothing caught it.** `test_steepest_descent_is_monotone` exists — and runs on
an AFM chain, whose `H_ii` is zero, so it passed for as long as the property was
false. The house rule names this exact shape: a check a wrong answer passes is not a
check. The new tests use S06's landscape, whose minimum is available in CLOSED FORM
(`c* = h/(C−A)`, `E* = A/2 − h²/(2(C−A))`), so they are pinned to algebra rather
than to a recorded number.

**How far did this reach? Swept, and the answer is: no shipped config.** Of the 8
configs using `anneal`/`steep`, exactly two also carry an on-site term
(`single_ion_anisotropy` / `sia_matrix` / `stevens` / `anisotropy_matrix`), which is
the trigger: S06 itself, and `examples/future_exmaples/FeI2/config_fei2.yaml`. FeI2
was re-run after the fix and returns **−46.372796 meV/cell, unchanged** — the value
item 7 records — so its ground state was never affected and that item's physics
stands. Nothing else could have been.

Worth keeping in mind for calls made OUTSIDE the runner, though: the ground-state
guards are what would have caught this downstream (a maximum is exactly what guard 2
exists for), so a direct `minimize_energy` call, or a config running
`on_imaginary: warn|off`, had no second line of defence.

**Related, not yet chased:** `thermal_mc.build_supercell` builds `H` purely from
`spin_interactions` (bonds), so it carries **no single-ion anisotropy at all** —
measured directly on S06, where `H_zz` came back as the bare exchange sum and D = 19
was simply absent, while its docstring calls `H` "the exchange/anisotropy Hessian".
That builder feeds `thermal_mc`, `wang_landau`, `static_correlations` and the
classical `sampled_correlations`. It is a *different* builder from the annealer's
(`MagCalc._extract_classical_quadratic`, which is correct — verified), so this is a
separate defect and was NOT fixed here. It deserves its own item, its own oracle
(the exact single-spin-in-a-field-plus-anisotropy partition function is closed form)
and its own commit.

---

## 10. KPM has no ground-state guard, and cannot grow one for free

**Status: OPEN.** Opened by item 3.

Every other spectrum path here refuses to expand about a non-minimum: the Cholesky in
`_bogoliubov` fails on a non-positive-definite `H2`, and `on_imaginary` turns that
into a hard error. **KPM never diagonalizes** -- that is the entire point of it -- so
it has no such failure mode and returns a smooth, plausible S(q,w) about a saddle or
a maximum. Item 3 met this for real: at Sunny's own disorder strength the relaxed
120-degree state has min eig H2 = -1e-2 and |Im w| = 0.16 meV on a 1.591 meV band,
and the KPM map looks fine.

`examples/sunny_tutorials/S09_triangular_AFM/disorder_kpm.py` does the check itself
(`eigvalsh(H2) >= 0` over the q-path, refuse by default). The runner's `kpm_sqw` task
does NOT.

**Why it is not a two-line fix.** The runner's guard runs once, on the calculator's
own reference state, before any task -- and for `kpm_sqw` that is the right place
only when the model came from the config. The interesting KPM models are exactly the
ones built or perturbed in a script (`apply_bond_disorder` is Python-only), which the
runner never sees. So the useful shape is probably a cheap, reusable
`SUNModel.is_stable_at(q)` / `assert_stable(qs)` that both the runner and a script
call, rather than a runner-side check alone.

**Cost matters here.** A full `eigvalsh` per q is O(D^3) and would undo KPM's whole
advantage on the large cells it exists for. Sampling a handful of q is what the
existing `stability_report`/`max_imaginary_energy` already do, and is probably right
-- but note the instability is q-SPECIFIC: on the 9x9 S09 cell, scanning only 4
generic q found it on 1 disorder realization in 3, while a 40-point path found it on
2 of 3. A guard that samples too thinly is the "check a wrong answer passes" shape
again. The honest cheap version may be a Lanczos/power estimate of the smallest
eigenvalue of H2 rather than a full solve.

---

## Also worth knowing

### Standing facts (no action)

- **Branches.** `master` is the default (there is no `main`). On 2026-08-12 the
  merged feature branches (`feat/gap4-phase1..4`, `feat/gap4-26-sun-dynamics`,
  `fix/sunny-parity-audit`, `docs/cu5sbo6-powder-comparison`,
  `docs/rb2cu3snf12-order8`, `test/coverage-audit-items-2-4`) were pruned after
  checking each tip was an ancestor of `origin/master` with nothing unpushed —
  no history was lost, every commit is reachable from `master`. Items A–E below
  were developed on `chore/open-work-housekeeping`, and the item-1 machinery plus
  the C/D/E follow-ups on `feat/open-work-followups`. Both were **fast-forward
  merged into `master`** after a green full gate (2026-08-12 and 2026-08-13); both
  are level with `master` and can be pruned.
- **The merge gate is `pytest -m ""`**, and from the *workspace root* plain
  `pytest` is already the full suite (the root `pytest.ini` deliberately does
  not inherit `-m "not slow"`). It takes ~30 min unloaded and can take 2.5 h on
  a busy machine. Don't pipe it through `tail` — that hides all progress until
  it exits.

### A. `pytest.ini` collection scope — **DONE 2026-08-12**

- **`pyMagCalc/pytest.ini` now scopes collection too.** Only
  the *root* config had `testpaths`/`norecursedirs`, so bare `pytest` run from
  inside `pyMagCalc/` — the documented iteration command — walked the whole
  project and **died during collection** on three stale scratch scripts in
  `archive/cleanup_20251224/` (missing CIFs, an outdated `MagCalcConfigBuilder`
  signature). It exited having run *nothing*, which is easy to misread as a fast
  clean run. It collected 518 tests at the time of the fix; after items B and C
  added their tests the fast suite is **524 passed, 2 skipped, 105 deselected**
  (~8 min).

### B. Engine provenance — `magcalc where` — **DONE 2026-08-12**

- **Which engine is running is now self-reporting.** This entry
  used to be a manual tip — "a stale OneDrive copy of this tree exists; if
  `magcalc` behaves inexplicably, check `python -c "import magcalc;
  print(magcalc.__file__)"`" — which only helps once you already suspect it. Three
  things changed:
  - `magcalc/provenance.py` + **`magcalc where`**, which prints the running
    package, its git HEAD, and *any other importable copy*. The last part is what
    the manual check cannot do.
  - **Every `magcalc run` logs the engine path as its first line**, so the record
    of a confusing run carries its own explanation.
  - `tests/test_install_provenance.py` pins it in the FAST suite, and pins the
    mechanism itself, so a future packaging change that fixes it will say so.

  The mechanism, since the obvious mental model is wrong: `pip install -e .`
  *appends* its finder to `sys.meta_path`, i.e. **after** `PathFinder`, so
  anything on `sys.path` beats the editable install. `sys.path[0]` is the cwd for
  `python -c`/`-m`, pytest's rootdir for `pytest`, and the script's own directory
  for `scripts/` helpers — all three verified to shadow. (The `magcalc` console
  script gets `bin/` instead, so it is immune to cwd shadowing but not to
  `PYTHONPATH`.) The first version of the detector scanned `sys.path` only and got
  the worst case backwards: when a stale copy wins, the live tree is reachable
  *only* through the meta-path finder and is absent from `sys.path`, so the scan
  saw one copy and reported all clear. It now unions the imported package, the
  `sys.path` entries, and the editable installs' declared roots.

  **That left one hole, now closed** (see the next entry): none of the above runs
  when a stale copy wins *outright*, because that copy has no `provenance.py`.

### C. Interpreter-startup shadow guard — **DONE 2026-08-12** (one small item open)

- **The guard** — `tools/magcalc_shadow_guard.py` + `tools/install_shadow_guard.py`. This is the
  half the in-package detector structurally cannot cover, and it also removes the
  "second checkout silently re-arms the hazard" caveat: the guard lives in
  site-packages, *outside every `magcalc` copy*, so it reports no matter which one
  wins, including a brand-new checkout.

  ```bash
  python tools/install_shadow_guard.py            # install (per interpreter)
  python tools/install_shadow_guard.py --status   # / --uninstall
  ```

  `magcalc where` now states whether it is active, so the protection level is
  never a guess. `MAGCALC_SHADOW_GUARD=off` silences it for the legitimate case
  (a git worktree, a deliberate version comparison). Cost: **254 µs** per
  interpreter startup, plus one string compare per import.

  **THE TIMING TRAP, and the reason the guard is not a simple startup check.**
  The obvious implementation — survey `sys.path` when the `.pth` runs — is blind
  to the main hazard. At `.pth` execution time `sys.path[0]` is **not yet the
  working directory**: for `python -c` the `''` entry is prepended *after* site
  initialisation. The first version did exactly that, passed a PYTHONPATH test,
  and reported all clear when run from inside a stale checkout. The check is
  therefore **deferred** to the moment `magcalc` is imported, via an observer
  parked at the front of `sys.meta_path` whose `find_spec` surveys and then
  always returns `None`. `tests/test_shadow_guard.py` pins this with a throwaway
  venv carrying two real `.pth` files — the guard, and an eager probe that
  records what a startup check *would* have concluded (`EAGER=False`).

  Second non-obvious detail: stderr alone is not enough. Under `pytest` the
  import happens inside the capture, so the banner is buffered and shown only if
  something else fails — and "ran the suite inside a stale checkout" is exactly a
  case where everything passes. The guard therefore also raises a real
  `MagcalcShadowWarning`, which lands in pytest's warnings summary on a green run.

  **STILL OPEN.** Nothing installs the guard automatically, so a **fresh venv
  starts unprotected**. `magcalc where` says so, which is the cheap mitigation;
  wiring it into `pip install -e .` is the obvious next step and was left out
  deliberately (a build hook that writes to site-packages is its own hazard).

### D. Stale OneDrive trees — **DONE 2026-08-12** (two follow-ups open)

- **Both stale trees are DELETED** (~2.6 GB). They were
  `~/Library/CloudStorage/OneDrive-MahidolUniversity/research/magcalc_archived/`
  (HEAD `e1b3f3b`, 2026-07-06) and `~/OneDriveMU_20250225_MacBook/research/magcalc/`
  (HEAD `2a29d23`, 2024-03-13, a Feb-2025 machine backup). Both HEADs were verified
  **ancestors of live `master`**, and the archived copy's ~200 "modified" files were
  a OneDrive mode-only artifact (index `100644` vs on-disk `rwx`), not content.

  Tracked-content redundancy was not sufficient to justify deleting, and checking
  only that would have destroyed work: **6 UNTRACKED files existed nowhere else**,
  and were rescued first.
  - `examples/materials/CCSF/` gained a complete **fit round-trip demo** —
    `make_fit_data.py` synthesizes dispersion data from the frozen 120° ground state
    at J1 = 13.3, J2 = −0.24, and `magcalc fit config_ccsf_fit.yaml` recovers
    13.286 ± 0.009 / −0.237 ± 0.004 (`CCSF_fit_report.txt`). It lands next to the
    `config_ccsf.yaml` it reads, so it is runnable as-is. This is a genuine
    end-to-end check of the fitting path — an exact-identity oracle in the sense
    GAP_STATUS.md means. **STILL OPEN:** it is pinned by no test of its own. It is
    only exercised by `test_config_smoke.py`, which asserts the run produces no
    ERROR records — not that the fit RECOVERS J1 = 13.3, J2 = −0.24. A ~10-line
    test asserting the recovered values would make it the oracle it deserves to be.
    (The two fit outputs are gitignored, since the smoke test rewrites them in the
    merge gate; the expected values live in the config's header comment.)
  - `archive/legacy/aCVO_2024_snapshot/spin_model_sf.py` — the legacy hand-written
    α-Cu₂V₂O₇ model, reference only. **STILL OPEN:** the workspace root is not a
    git repository and `pyMagCalc/archive/` is gitignored, so this file — which
    existed nowhere else — is again in an unversioned directory. It needs a
    deliberate home if it is worth keeping.

  The other ~8900 untracked files were a `gui/node_modules.onedrive-bak/` copy.

### E. `mu_B` consolidation — **DONE 2026-08-12** (one small item open)

- **`mu_B` now lives in `magcalc/constants.py`.** It used to be
  a `5.788e-2` literal in six modules (`generic_model` ×2, `spiral_opt`,
  `thermal_mc`, `sun/lswt`, `sun/entangled`, `sun/dimer_series`), four of them
  function-locals, plus a seventh copy emitted into generated models by
  `utils/generate_spin_model.py`. All seven now import `MU_B` /
  `GAMMA_ELECTRON`; the value is unchanged (`5.788e-2`, the deliberate
  four-figure CODATA truncation every pinned Zeeman number was measured
  against — moving it to full precision shifts every in-field energy by 6.6e-5
  relative and is not a free accuracy fix).
  `test_every_engine_uses_the_same_bohr_magneton` was rewritten to match: it
  asserts each engine binds the *same object* (`is`, not `==`) and greps the
  package for a re-typed literal. That second half is the one that matters — a
  stray `mu_B = 5.788e-2` back inside a function would restore the original
  hazard while the identity check still passed.
- **STILL OPEN — the dipolar prefactor is still duplicated.** `ewald.MU0_MUB2_MEV_A3 =
  0.6745817653` and `generic_model.DIPOLE_PREFACTOR_MEV_A3 = 0.05368216` are the
  same Sunny constant with and without the 4π. Both are module-level and
  cross-referenced in comments, so they are far less drift-prone than the `mu_B`
  locals were, but folding them into `constants.py` is the obvious next step.
