# Open work — pick-up notes

Last updated **2026-08-12**, `master` at `21ecba2` (pushed).

**Full gate GREEN on this tree: 635 passed, 3 skipped** (`pytest -m ""` from the
workspace root, 40:48). That is +14 on the 2026-08-12 figure of 621 — 5 from
`test_install_provenance`, 8 from `test_shadow_guard`, and 1 config the smoke test
newly discovers (the rescued CCSF fit demo). `chore/open-work-housekeeping` was
fast-forward merged into `master` after that run and is now level with it.

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
| 1 | Gap #24b — Ewald + rotating-frame single-k | **OPEN** — machinery written + refused-by-default; blocked on building the ORACLE |
| 2 | Sunny S06 — skyrmion lattice | **OPEN** — blocked on the reference state |
| 3 | Sunny S09 — disorder + KPM | **OPEN** — blocked on structure geometry |
| 4 | Classical S(q,ω) absolute normalization | **OPEN** — shape pinned, scale not |
| 5 | Coverage follow-ups | **PARTIAL** — audit's 4 items done; 2 follow-ups + discovery shape open |
| 6 | `minimization.tolerance` silently ineffective | **DONE** 2026-08-12 |
| 7 | FeI2 dipole ground state | **PARTIAL** — physics answered; `examples/materials/FeI2` fix open |
| 8 | Studio open→run limits | **OPEN** — 2 items |
| A | `pytest.ini` collection scope | **DONE** 2026-08-12 |
| B | Engine provenance — `magcalc where` | **DONE** 2026-08-12 |
| C | Interpreter-startup shadow guard | **DONE** 2026-08-12 — auto-install still open |
| D | Stale OneDrive trees deleted | **DONE** 2026-08-12 — 2 rescued-file follow-ups open |
| E | `mu_B` → `constants.py` | **DONE** 2026-08-12 — dipolar prefactor still duplicated |

**The small open items are easy to lose**, so they are also collected here. Each
lives inside an otherwise-`DONE` entry:

- **C** — nothing installs the shadow guard automatically; a fresh venv starts
  unprotected.
- **D** — the rescued CCSF fit demo is pinned by no test of its own; and
  `archive/legacy/aCVO_2024_snapshot/` sits in an unversioned directory.
- **E** — `ewald.MU0_MUB2_MEV_A3` and `generic_model.DIPOLE_PREFACTOR_MEV_A3`
  are still two spellings of one Sunny constant.

---

## 1. Gap #24b — Ewald + rotating-frame single-k

**Status: OPEN.** Still the highest value-per-effort item, but the "~1 hour"
estimate was wrong and the written-up formula was wrong. Both corrected 2026-08-12,
BEFORE any code was written — read the two findings below first.

**FINDING 1 — the projector algebra in `GAP4_PLAN.md` was INVERTED.** It said
"FIVE terms … dropping to three when the satellites coincide". The reference says
the opposite (`SpinWaveTheorySpiral.jl:129–138`): the five-term form, with the two
cross terms, is the **`k_case 2`** special case (2k integer, satellites coincide),
and the **generic incommensurate `k_case 3`** — the common case — is the plain
three-term `R2·J(q)·R2 + R1*·J(q+k)·R1* + R1·J(q−k)·R1`. Implementing from the old
text would have put the cross terms into the common branch: a wrong Hamiltonian
that still diagonalizes and still yields a plausible spectrum. `GAP4_PLAN.md` is
now fixed. Note this is the **fourth** wrong characterization of this item, and the
first one to survive *inside the document that exists to prevent them*.

**FINDING 2 — the item is bigger than "add a term", because pyMagCalc's single-k
scheme is structurally different from Sunny's.** Sunny builds ONE rotating-frame
`J` per branch from the lab-frame `J(q)`, `J(q±k)` via the projectors. pyMagCalc
instead evaluates its *symbolic* `H` at three shifted momenta `q−k, q, q+k`
(`numerical.process_calc_Sqw_single_k`) and applies `spiral_channel_tensors` to the
**correlation tensors**, not to `H`. That shortcut is only valid because the engine
*requires* the Hamiltonian to be rotationally invariant about the spiral axis —
`generic_model.py:1630`, `enforce_rotational_symmetry`, mirroring Sunny's
`check_rotational_symmetry`. Under that assumption the projector combination
collapses to `J(q)` and evaluating at shifted momenta is exact.

**The dipolar tensor does not satisfy that assumption.** `A(q)` is fixed by lattice
geometry (its `r̂r̂` structure) and is not uniaxial about an arbitrary spiral axis.
So Ewald cannot ride the existing shortcut: it needs the real projector combination
applied per channel, `_ewald_nambu_spiral(q_c, k, axis, k_case)` for each of
`q_c ∈ {q−k, q, q+k}`, built from `J(q_c)`, `J(q_c±k)` and the `q=0` on-site `J0`
combination — and wired through the dispersion path, the S(Q,ω) worker (across the
multiprocessing boundary, so it must pickle), and the classical energy the
ground-state guard minimises. The formula is an hour; the wiring is not.

The rest of the writeup in `GAP4_PLAN.md` §"#24b … METHOD FOUND IN SUNNY" is
sound — read it, do not re-derive.

**PROGRESS 2026-08-12 — the machinery is written; the ORACLE IS NOT, and that is
the whole remaining item.** `core._ewald_J_lab`, `_spiral_projectors`,
`_ewald_J_rot` (three/five-term, branching on `k_case`) and a spiral-aware
`_ewald_nambu` exist, and `numerical.process_calc_Sqw_single_k` now takes a
per-channel `h_dip`. The non-spiral Ewald path is unaffected — all 9
`tests/test_ewald.py` pass, including the Sunny-pinned `test_ewald_matches_sunny`.

**It is still refused by default**, behind `dipole_dipole: {allow_single_k: true}`,
because it is unvalidated. Do not remove that refusal until the oracle passes.

**Where the oracle broke, so the next attempt does not repeat it.** The plan was
"commensurate k must equal the explicit `magnetic_supercell` calculation". Two
traps found:

1. **k = 1/2 is the wrong commensurate case.** It is `k_case 2`, and the engine
   itself warns that a helical description of a *collinear* structure may double
   count. Use a non-collinear commensurate k — k = 1/3 (120°, `k_case 3`) also has
   the advantage of exercising the common three-term branch.
2. **The comparison does not hold even with Ewald switched OFF.** Running k = 1/3
   single-k (`satellites=True`, 3 modes) against `magnetic_supercell: [3,1,1]`
   (3 modes) on a plain Heisenberg chain gave `[0, 0.707, 1.041]` vs
   `[0.382, 1.791, 3.827]` — a 2.8 meV disagreement with **no dipolar term at
   all**. So the harness, not the dipolar code, is wrong: the band-set
   correspondence between the rotating-frame and supercell descriptions is not the
   naive "sorted energies are equal" at the same chemical q, and/or the 120°
   structure used is not the ground state of that chain (an AFM chain wants
   k = 1/2, not 1/3).

   **Establish the no-Ewald control FIRST** — pick a model whose spiral genuinely
   is the ground state (or minimize it), and work out the correct q-correspondence
   and normalization between the two descriptions. Only when the control agrees
   does the Ewald comparison mean anything. Doing it the other way round is how a
   wrong Hamiltonian gets blessed by a broken oracle. Three earlier attempts at
re-deriving it produced three wrong characterizations in a row (three terms,
then "structurally invalid", then nine terms); reading
`../Sunny.jl-main/src/Spiral/SpinWaveTheorySpiral.jl` settled it.

Two facts that make the item small:

- **Ewald is not special-cased.** Sunny's `fourier_bilinear_interaction!` builds
  `Jq` from the exchange bonds and then adds the dipolar `Aq` into the *same*
  matrix (mind the g-tensors). Everything downstream operates on `Jq` without
  knowing part of it came from an infinite lattice sum, so there is no separate
  "Ewald channel machinery" to build.
- **The projector algebra is FIVE terms**, with `R2 = axis·axisᵀ` and
  `R1 = (I − i[axis]× − R2)/2`:

      J = R2·J(q)·R2 + R1*·J(q+k)·R1* + R1·J(q−k)·R1
                     + R1·J(q+k)·R1* + R1*·J(q−k)·R1

  The two cross terms are exactly what a commutation-based derivation drops and
  a naive 9-term expansion over-counts. Keep Sunny's `k_case` branch: it drops
  to three terms when 2k is a reciprocal-lattice vector.

**Injection point (verified 2026-08-04).** `core._ewald_nambu(q_cart)` already
does for the non-spiral path exactly what `fourier_bilinear_interaction!` does
for the Ewald half. The change is a `_ewald_nambu_spiral(q_cart, k, axis)`
building the same Nambu blocks from the rotating-frame combinations above (both
for `Jq` and for the `q=0` on-site `J0`), handed to the existing three-channel
worker the way `dip_pairs` already is in `calculate_sqw`. Then delete the
refusal in `core.py` — it names the exact spot.

**Oracle, two-stage:** for a *commensurate* k the rotating-frame answer must
equal the explicit `magnetic_supercell` calculation exactly; that path already
supports Ewald, so it is a self-contained exact check needing no external
reference. Then cross-check an incommensurate k against Sunny.

**Note:** this blocks no shipped example — nothing in `examples/` combines
`single_k` with `dipole_dipole: {method: ewald}`. It is capability parity, not a
broken user path.

---

## 2. Sunny tutorial S06 — skyrmion lattice

**Status: OPEN — blocked on the reference state, not on engine capability.** Everything it
needs exists and was validated in isolation (#26's dissipative quench and the
Berg–Lüscher topological charge).

Symptom: the quench relaxes to a uniformly polarized state (Q = 0) instead of a
skyrmion lattice. With the field sign matched to Sunny's `g = −1`, ⟨Sz⟩ = +0.45
as expected, so the Hamiltonian is right.

Two open questions, in the order worth testing:

1. **The second-neighbour triangular bond shell.** Does it match Sunny's
   `Bond(1,1,[1,2,0])`? A wrong J2 shell suppresses exactly the frustration that
   sets the skyrmion scale — and would look *precisely* like this symptom. Check
   with `magcalc symmetry <config> --max-distance …` against Sunny's
   `print_symmetry_table`. Cheap, and the most likely culprit.
2. **System size.** Sunny uses L = 40 (1600 sites); this port has run 64–256.

If size turns out to matter, the derivative is the bottleneck: it currently
scales as sites² and needs vectorizing before 1600 sites is interactive.

**Do not "port" it by substituting an equilibrium calculation for the quench.**
That produces a folder that looks like a port and is not one.

---

## 3. Sunny tutorial S09 — disorder + KPM on the triangular lattice

**Status: OPEN — blocked on structure geometry.** Needs the 120° order as an explicit
REAL-SPACE √3×√3 supercell: the clean config uses the rotating-frame `single_k`
method, which the SU(N)/KPM path does not consume.

The current placeholder gives E/site = −0.3333 against the exact −0.375, i.e.
the basis is wrong, and the consequence is measurable and diagnostic: adding
disorder **narrowed** the KPM width instead of broadening it, which is what
expanding about a non-minimum buys you.

Fix is to build the √3×√3 cell explicitly with the three sublattice directions
at 120°, and confirm E/site = −0.375 *before* looking at any spectrum.

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

## Also worth knowing

### Standing facts (no action)

- **Branches.** `master` is the default (there is no `main`). On 2026-08-12 the
  merged feature branches (`feat/gap4-phase1..4`, `feat/gap4-26-sun-dynamics`,
  `fix/sunny-parity-audit`, `docs/cu5sbo6-powder-comparison`,
  `docs/rb2cu3snf12-order8`, `test/coverage-audit-items-2-4`) were pruned after
  checking each tip was an ancestor of `origin/master` with nothing unpushed —
  no history was lost, every commit is reachable from `master`. Items A–E below
  were developed on `chore/open-work-housekeeping` and **fast-forward merged into
  `master` on 2026-08-12** after a green full gate; that branch is now level with
  `master` and can be pruned.
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
