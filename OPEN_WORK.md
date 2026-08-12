# Open work — pick-up notes

Last updated **2026-08-12**, master at `4010121` **plus uncommitted working-tree
changes** (Studio open→run parity, the `examples/fitting` correction, and the
smoke-test coverage/backend changes below).

Full gate green on the current tree: **621 passed, 3 skipped**
(`pytest -m ""` from the workspace root, 43 min). That is +21 on the 2026-08-05
baseline of 600 — 12 from the Studio open→run work (`test_atom_mode_explicit`,
`test_gui_roundtrip`, two added to `test_gui_passthrough`), 4 from
`test_fit_example`, and 5 configs the smoke test had never discovered.

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

## 1. Gap #24b — Ewald + rotating-frame single-k

**Ready to implement. ~1 hour with the formula in hand.** This is the highest
value-per-effort item open.

The method is written up in full in `GAP4_PLAN.md` §"#24b … METHOD FOUND IN
SUNNY" — read that section, do not re-derive. Three earlier attempts at
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

**Blocked on the reference state, not on engine capability.** Everything it
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

**Blocked on structure geometry.** Needs the 120° order as an explicit
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

Opened by #17. The classical-dynamics path's overall scale has never been
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

## 6. ~~Loose end — `minimization.tolerance` is silently ineffective~~ (DONE 2026-08-12)

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

- **Branches.** `master` is the default (there is no `main`), and as of
  2026-08-12 it is the *only* branch, local and remote. The merged feature
  branches (`feat/gap4-phase1..4`, `feat/gap4-26-sun-dynamics`,
  `fix/sunny-parity-audit`, `docs/cu5sbo6-powder-comparison`,
  `docs/rb2cu3snf12-order8`, `test/coverage-audit-items-2-4`) were pruned after
  checking each tip was an ancestor of `origin/master` with nothing unpushed —
  no history was lost, every commit is reachable from `master`.
- **The merge gate is `pytest -m ""`**, and from the *workspace root* plain
  `pytest` is already the full suite (the root `pytest.ini` deliberately does
  not inherit `-m "not slow"`). It takes ~30 min unloaded and can take 2.5 h on
  a busy machine. Don't pipe it through `tail` — that hides all progress until
  it exits.
- **Run from `pyMagCalc/`.** A stale OneDrive copy of this tree exists; if
  `magcalc` behaves inexplicably, check
  `python -c "import magcalc; print(magcalc.__file__)"`.
- **`mu_B` now lives in `magcalc/constants.py`** (done 2026-08-12). It used to be
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
- **Still duplicated: the dipolar prefactor.** `ewald.MU0_MUB2_MEV_A3 =
  0.6745817653` and `generic_model.DIPOLE_PREFACTOR_MEV_A3 = 0.05368216` are the
  same Sunny constant with and without the 4π. Both are module-level and
  cross-referenced in comments, so they are far less drift-prone than the `mu_B`
  locals were, but folding them into `constants.py` is the obvious next step.
