# Open work — pick-up notes

Last updated **2026-08-05**, master at `59aa035`, full gate green
(`pytest -m ""` from the workspace root: **600 passed, 3 skipped**).

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
coverage audit"). Two follow-ups were recorded rather than done:

- **Enumerate config keys from the CODE, not the docs.** The audit swept
  *documented* keys against `tests/`, so `calculation.imaginary_rel_tolerance` —
  in neither the docs nor the tests — was invisible to the very process meant to
  find gaps. Sweeping `calc_config.get(...)` call sites has been done for the
  `calculation:` block (clean, apart from internal `cache_file_base`); the other
  blocks (`tasks`, `plotting`, `minimization`, `scga`, `thermal_mc`, …) have
  not.
- **Escalate a whitelist of WARNINGs in the config smoke test.** It currently
  fails on ERROR log records only. Item 6 below is exactly the kind of thing
  that would catch.

---

## 6. Loose end — `minimization.tolerance` is silently ineffective

`examples/materials/FeI2/config_fei2.yaml:112` sets `minimization.tolerance:
1e-5`. `runner.py:490` sweeps unrecognized keys of the `minimization:` block
into `min_kwargs`, forwards them through `MagCalc.minimize_energy(**kwargs)`
(`core.py:2655`) and on into `scipy.optimize.minimize`, which takes `tol`, not
`tolerance`. The run logs

    WARNING  Optimization attempt using MagCalc failed:
             minimize() got an unexpected keyword argument 'tolerance'

and carries on — so the minimization silently does not happen with the requested
tolerance.

Decide between mapping `tolerance` → `tol` and rejecting unknown `minimization:`
keys outright. The second is more in keeping with the engine's hard-error
policy, but check the other shipped configs for the same key first. Whichever
way, the smoke test should be made to catch it (item 5).

---

## Also worth knowing

- **Branches.** `master` is the default (there is no `main`). Several feature
  branches remain on the remote after merging: `feat/gap4-phase1..4`,
  `feat/gap4-26-sun-dynamics`, `fix/sunny-parity-audit`,
  `docs/cu5sbo6-powder-comparison`, `docs/rb2cu3snf12-order8`,
  `test/coverage-audit-items-2-4`. Consolidating/pruning them is unfinished
  housekeeping.
- **The merge gate is `pytest -m ""`**, and from the *workspace root* plain
  `pytest` is already the full suite (the root `pytest.ini` deliberately does
  not inherit `-m "not slow"`). It takes ~30 min unloaded and can take 2.5 h on
  a busy machine. Don't pipe it through `tail` — that hides all progress until
  it exits.
- **Run from `pyMagCalc/`.** A stale OneDrive copy of this tree exists; if
  `magcalc` behaves inexplicably, check
  `python -c "import magcalc; print(magcalc.__file__)"`.
- **`mu_B = 5.788e-2 meV/T` is a magic number duplicated in six modules**
  (`generic_model` ×2, `spiral_opt`, `thermal_mc`, `sun/lswt`, `sun/entangled`,
  `sun/dimer_series`). `tests/test_combination_matrix.py` pins them equal by
  reading the literals out of the sources. Consolidating them into one constant
  would be a real improvement; just keep that test working.
