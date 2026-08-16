# Open work — pick-up notes

Last updated **2026-08-16**. The 2026-08-13 session's work (items 1–10, A–E) is on
`master` at `f848853`; the 2026-08-15 session closed **every remaining open item —
5, 7, 11, 12, 13, 14 and C** — and opened one new one (15, found inside item 12), which
2026-08-16 closed. **Nothing is open.** None of it is committed; the working tree
carries all of it.

**Full gate GREEN on this tree: 837 passed, 3 skipped, 0 failed** (`pytest -m ""` from
the workspace root, 2026-08-16, 45:46 — the machine was contended, load average 56 at
the peak, so that time is an upper bound; the 2026-08-15 run of the same gate took
30:22 at 830 passed). **NOT COMMITTED**: the work is in the working tree, on `master`.
Branch before committing.

(The one-test accounting note from 2026-08-15 has resolved itself: `--collect-only`
now reports 832 from inside `pyMagCalc/` and 839 from the workspace root, exactly
837 + 3 skipped − 1. The cause was as suspected — `test_config_smoke` parametrizes over
a filesystem glob of `examples/`, so its count moves by design.)

Trail: 621 (2026-08-12 baseline) → 635 after the provenance + shadow-guard work → 637
with `test_ccsf_fit_roundtrip` → 650 with item 1's `test_ewald_spiral` → 672 with
item 2's → 790 with items 10, 8 and 4 → 830 with items 11, 12, 13, 14, 5, 7 and C
(+40: 6 `test_thermal_mc`, 10 `test_classical_window`, 12
`test_sun_sampler_equilibration`, 4 `test_config_key_coverage`, 3 `test_shadow_guard`,
2 `test_config_smoke`, and one each in `test_ewald`, `test_kpm_stability`,
`test_sun`) → **837** with item 15 (+7, all in `test_classical_window`, which is now 17
tests). 832 collect from inside `pyMagCalc/`, of which 654 are the fast default suite.

Items 1, 2, 9, 3, 10, 8 and 4 were committed on `feat/s06-cp2-skyrmions`, pushed, and
fast-forward merged into `master`. Item 3 is two commits: `5812975` the KPM fix
(`magcalc/sun/kpm.py`, `tests/test_kpm.py`) and **`91ae993`** the S09 port — the fix
stands on its own and is worth reading separately, since it changes every KPM spectrum
of a non-collinear model and every `cross_section: chiral` KPM result. (An earlier
draft of this file cited `ff9fbfe` for the S09 port; that commit exists only in the
reflog and is NOT reachable from `master`. `91ae993` is the one on `master`.)

**Four merged branches are still lying around and can all be pruned**, each verified
level with or an ancestor of `master`: `feat/s06-cp2-skyrmions` (= `master`),
`docs/gate-637` (`f819694`), `feat/open-work-followups` (`00083a6`) and
`chore/open-work-housekeeping` (`21ecba2`).

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

**If you are picking this file up cold, read only the OPEN rows** — as of 2026-08-16
there are none. Everything else is kept for the traps it documents, not because
anything is left in it. Note the pattern in how the last few arrived: items 11–14 were
all found *inside* the previous session's work and written up rather than fixed in the
same commit, and item 15 was found inside item 12 the same way. That is the intended
flow — the alternative is a commit that changes several unrelated numbers at once and
cannot be reviewed. The next item here will almost certainly be found the same way, so
add it rather than folding it into whatever you are doing.

### Open

| # | Item | Status |
|---|---|---|
| — | nothing open | — |

### Closed 2026-08-16

| # | Item | Status |
|---|---|---|
| 15 | The classical window needs `subtract_elastic`, and nothing enforces the pairing | **DONE** — reported, with the amplification named; `on_elastic_leakage` |

### Closed 2026-08-15

| # | Item | Status |
|---|---|---|
| 5 | Coverage follow-ups | **DONE** — all three pieces; the escalation found 7 configs on a deprecated key |
| 11 | `thermal_mc.build_supercell` carries NO single-ion anisotropy | **DONE** — fixed; 5 of 6 new tests confirmed failing before |
| 12 | No time-domain window on the classical S(q,ω) | **DONE** — added as OPT-IN; the measurement is why it is not the default |
| 13 | CP^(N−1) sampler does not equilibrate at low kT | **DONE** — step size adapts; pinned to a closed-form partition function |
| 14 | The two dipolar prefactors are duplicated and NOT reconciled | **DONE** — derived from one constant; Ewald gate re-run |
| 7 | FeI2 dipole ground state | **DONE** — config fixed, `on_imaginary: warn` gone, Sunny pins intact |
| C | Interpreter-startup shadow guard | **DONE** — armed on first CLI use, not at build time |

### Closed earlier

| # | Item | Status |
|---|---|---|
| 1 | Gap #24b — Ewald + rotating-frame single-k | **DONE** 2026-08-13 — oracle built, bug found, refusal lifted |
| 2 | Sunny S06 — skyrmion lattice | **DONE** 2026-08-13 — ported at L = 40; the size WAS the blocker |
| 3 | Sunny S09 — disorder + KPM | **DONE** 2026-08-13 — ported; found a KPM bug and a stability limit |
| 4 | Classical S(q,ω) absolute normalization | **DONE** 2026-08-13 — two bugs, not a convention |
| 6 | `minimization.tolerance` silently ineffective | **DONE** 2026-08-12 |
| 8 | Studio open→run limits | **DONE** 2026-08-13 — the Swift emitter had 4 live defects |
| 9 | `anneal`'s polish could return a MAXIMUM | **DONE** 2026-08-13 |
| 10 | KPM has no ground-state guard | **DONE** 2026-08-13 — guarded per q |
| A | `pytest.ini` collection scope | **DONE** 2026-08-12 |
| B | Engine provenance — `magcalc where` | **DONE** 2026-08-12 |
| D | Stale OneDrive trees deleted | **DONE** 2026-08-12 |
| E | `mu_B` → `constants.py` | **DONE** 2026-08-12 — the prefactor question was item 14 |

---

## What was run (2026-08-15)

**The merge gate passed: `pytest -m ""` from the workspace root → 830 passed, 3
skipped, 0 failed, 30:22.** That is the authoritative result; the per-file runs below
are only the order in which things were checked while the work was in progress.

| file | result |
|---|---|
| `test_thermal_mc.py` | 9 passed (`-m ""`) — item 11 |
| `test_classical_window.py` | 10 passed (`-m ""`) — new, item 12 |
| `test_classical_absolute_normalization.py` + `test_classical_dynamics.py` + `test_sun_dynamics.py` + the above | 32 passed (`-m ""`) |
| `test_wang_landau.py` | 11 passed |
| `test_ewald.py` + `test_ewald_spiral.py` | 21 passed (`-m ""`) — the item-14 oracle re-run |
| `test_sun.py` | 26 passed — the Sunny-pinned FeI2 comparisons, after item 7 |
| `test_shadow_guard.py` + `test_install_provenance.py` | 16 passed (`-m ""`) — item C |
| `test_sun_sampler_equilibration.py` | 12 passed (`-m ""`) — new, item 13 |
| `test_config_key_coverage.py` | 4 passed — new, item 5 |
| `test_config_smoke.py -m ""` | 60 passed (17:53) in HARVEST mode; the escalating version is in the gate above |
| `examples/materials/FeI2/config_fei2.yaml` | runs clean end to end, guard at default `error`, max abs Im ω = 2.4e-15 |

**What was NOT measured, and why it is named rather than estimated.** The machine sat
at load average 250–340 all session (other users' processes), and two supporting
measurements were abandoned after ~20 minutes of wall clock each with under 16 s of
CPU accrued: the before/after truncated-vs-Ewald residual table (item 14) and the
end-to-end SU(N) intensity swing (item 13). Neither is the deciding evidence for its
item — item 14 rests on the re-run Ewald gate, item 13 on an exact partition function
— but both are worth taking on a quiet machine. See also the process-pool note under
"Standing facts".

**The four small items that used to hang off C/D/E are all CLOSED** — three on
2026-08-12, and the fourth (the dipolar prefactor, item 14) on 2026-08-15:

- **C** — `magcalc guard [--install|--uninstall]`, with the source inside the package
  (`magcalc/_shadow_guard.py`) rather than in `tools/`, which a non-editable
  `pip install` does not ship — that hole was bigger than the one logged, since a
  wheel user could not install the guard *at all*. The "a fresh venv starts
  unprotected" remainder was closed 2026-08-15 by arming it on the first CLI command;
  see the C entry below for why that is a run-time and not a build-time hook.
- **D** — `tests/test_ccsf_fit_roundtrip.py` pins the rescued fit demo, and it
  asserts the right thing: the values that GENERATED the synthetic data
  (J1 = 13.3, J2 = −0.24, recorded in the CSV header), not the fit output observed
  when the demo was rescued — so it cannot certify a drifted fit as correct. The
  rescued aCVO model is tracked at
  `examples/materials/aCVO/legacy_spin_model_sf_2024.py`; the copy at the workspace
  root, `archive/legacy/aCVO_2024_snapshot/spin_model_sf.py`, is byte-identical to it
  apart from the 19-line provenance header, i.e. a redundant duplicate rather than the
  only copy.
- **E** — both dipolar prefactors moved into `constants.py`, which removed the
  duplication across modules and left a numerical disagreement of 1.2e-5 relative
  standing in plain sight. That is item 14, and it is closed: the 4-pi-reduced
  constant is now DERIVED from the other rather than typed.

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
`disorder_kpm.py` checks H₂ ⪰ 0 itself and refuses by default. **That check is now the
engine's**, shared by the script and the runner's `kpm_sqw` task and applied at every
q either of them computes — item 10, closed the same day.

---

## 4. Classical S(q,ω) absolute normalization

**Status: DONE 2026-08-13.** All three classical estimators —
`classical_dynamics.sampled_correlations`, `thermal_mc.static_correlations` and the
CP^(N−1) `sun/dynamics.sampled_correlations` — are now on the LSWT/Sunny absolute
scale, per chemical cell with the 1/2π of the time transform. Pinned in
`tests/test_classical_absolute_normalization.py` (10 fast + 1 slow); **9 of the 10
were confirmed to FAIL on the pre-fix code.** One follow-up is open, at the bottom.

**The advice this item gave itself was right, and it was two bugs, not a convention.**

1. **The time FFT was never normalized.** S(q,ω) is (1/2π)∫dt e^{−iωt}C(t); the code
   kept the bare `np.fft.fft` sum, i.e. **2π/dt too large — 314× at the default
   dt = 0.02, and PROPORTIONAL TO 1/dt**. Refining the time step moved the intensity,
   and nothing compared two grids.
2. **The spatial sum was divided by the SITE count, not the CELL count.** LSWT is per
   chemical cell in both codes (`MagCalc.supercell_ncells`, `SUNModel.n_cells`,
   Sunny's `1/√prod(sys.dims)`), so a two-atom cell scatters twice as much. **Every
   classical model in `tests/` has one site per cell**, where the two divisors are
   identical — so this was invisible by construction, not by accident. That is the
   same shape as the `ref_pair` and `steepest_descent` traps: a property that is only
   *false* on inputs no test reaches.

**The oracle is an exact identity, and it had to be, because the physical check is
too noisy to pin a scale.** ∫dω S^ab(q,ω) = ⟨S^a(q)\*S^b(q)⟩/n_cells is the defining
property of S(q,ω), so checking it against the SAME trajectory needs no reference
code and no fitted constant: it holds to **1e-14**, on a one-site and a two-site cell,
in `trace` and `perp`, in the dipole and the CP^(N−1) path. Both factors move it (the
2π/dt scales one side, the divisor scales it by n_atoms), so one identity catches
both. `dynamical_structure_factor` grew a `two_sided` option so the two-sided integral
the sum rule is a statement about is reachable from the shipped function, and the
SU(N) transform was split out of `sampled_correlations` for the same reason.

**Closing the loop with LSWT took three tries, and the two failures are the useful
part.** The target is exact: for a ferromagnet, equipartition gives
⟨|s^y_q|²⟩/n_cells = S·kT/ω_q, and c2q(ω_q) → ω_q/kT, so ∫₀^∞ → S/2 — precisely the
LSWT band sum. Measured:

| what was integrated | ratio to LSWT |
|---|---|
| plain Heisenberg chain, L = 20, kT = 0.02, whole axis | **1.45**, converged (n_traj 16 and 64 agree) |
| gapped chain (20 T field), L = 32, kT = 0.005, whole axis | 1.16 |
| the same, ±1 meV around each LSWT band | **1.015** (0.96–1.07 per q) |

- The 1.45 is **physics**: in 1D with a Goldstone mode Σ_q 1/ω_q diverges, the order
  parameter direction wanders over the trajectory, and lab-frame "transverse" weight
  is inflated. It converges only as kT → 0 AND L → ∞, and it reads exactly like a
  normalization error. A field gapping the mode removes it. (Mind the Zeeman sign
  while doing that: pyMagCalc's is +μ_B B·g·S, so the moments settle ANTIparallel to
  B — getting it backwards puts the structure at a stationary maximum and returns
  bands |gμ_B B − ω_q|, which look perfectly reasonable.)
- The remaining 16 % is **spectral leakage, not scale**: no time-domain window is
  applied, so a rectangular one is implied, whose sidelobes fall only as 1/(ω−ω₀)²
   — while c2q grows LINEARLY in ω out to the Nyquist frequency π/dt = 157 meV on a
  4 meV band. Sunny multiplies its real-time correlations by a cosine window for
  exactly this reason.

**Two things this item found and deliberately did not fix**, each now an item of its
own because each changes numbers this item's tests are insensitive to:

- **no time-domain window → item 12.** The scale is right; the lineshape still
  implies a rectangular window, which is where the residual 16 % above comes from.
- **the CP^(N−1) sampler does not equilibrate at low kT → item 13.** It is why the
  SU(N) half of this item is pinned by the exact sum rule and by grid independence
  rather than against LSWT.

---

## 5. Coverage follow-ups

**Status: DONE 2026-08-15.** All three open pieces are closed.

**(a) Enumerate config keys from the CODE, not the docs — DONE.**
`tests/config_keys.py` parses the package with `ast` and records every
`<block>.get("key")` / `<block>["key"]` where the block is a local bound to a config
section; `tests/test_config_key_coverage.py` asserts every one of them appears in a
shipped config or a test, with an explicit reasoned `ALLOWED` list for the rest.
**194 keys across 21 blocks**, of which 20 were unexercised.

It paid for itself on the first run, and in exactly the predicted shape: it found
**`calculation.h2_rel_tolerance`** — guard 3's threshold, added by item 10 three days
earlier, documented prominently in CLAUDE.md, read by `runner.py`, and named by NO
test and NO config. That is `calculation.imaginary_rel_tolerance` again, one item
later, and a docs-first sweep could not have found it either time. It is covered now
(`test_kpm_stability.py::test_h2_rel_tolerance_reaches_the_guard_from_the_config`,
bracketed either side of a known instability rather than checked one-sided).

Two things about the audit worth knowing before extending it. The AST walk is a
heuristic — it fails in the SAFE direction (a key it cannot see is one nothing else
would have seen either; a spurious entry surfaces as a loud "covered nowhere" and gets
deleted by hand). And `exercised_keys()` EXCLUDES the audit's own two files: without
that, `test_config_key_coverage.py` names every key it excuses, the text sweep finds
those names, and the audit certifies exactly the keys it was told to ignore. That
self-fulfilling version was written first and passed.

**(b) Escalate WARNINGs in the config smoke test — DONE.**
`tests/test_config_smoke.py` collects at WARNING level and fails on any warning not
matching `ALLOWED_WARNINGS`, with a harvest mode (`MAGCALC_SMOKE_HARVEST=<path>`) that
records instead of asserting — so the list was built from a real run of all 58 configs
rather than guessed, and can be rebuilt the same way when a config is added.

**The whole surface is four distinct messages**, which is why escalating was viable at
all once the two benign ones were removed in 2026-08-12:

| message | configs | verdict |
|---|---|---|
| the dipole-mode SU(N) advisory | 9 | expected — allow-listed |
| "is NOT a classical energy minimum" | 3 | expected — each carries a deliberate `on_imaginary: warn` |
| "Magnon energies are IMAGINARY" | 4 | same, plus SW03's commensurate approximation |
| `type: spiral` is deprecated | 1 | **a real finding — fixed, see below** |

**The escalation caught something on its first run, and it was not the warning it
looked like.** SW08 warned that `magnetic_structure: {type: spiral}` is deprecated —
but SEVEN shipped configs used that spelling (SW08, SW15, SW18, SW22, SW23, SW26,
SW37) and only one warned, because `_LEGACY_MS_WARNED` in `generic_model.py` fires the
deprecation **once per PROCESS**. Under pytest it therefore attaches itself to whatever
runs first, and the suite randomizes order — so a warning-based check on it would have
been flaky, and a warning-based check is not the right tool. All seven are migrated to
`type: single_k`, which is safe BY CONSTRUCTION (that branch of
`normalize_magnetic_structure` rewrites `cfg['type']` and nothing else) and is pinned
as an exact identity rather than by re-verifying seven spectra
(`test_spiral_and_single_k_normalize_identically`). A separate test greps every shipped
config for both deprecated spellings, which is order-independent
(`test_no_shipped_config_still_uses_a_deprecated_structure_type`).

**(c) Discovery is no longer a glob plus a hand-list — DONE.** The criterion is now
what a config IS, not what it is called: any `*.yaml` at any depth under `examples/`
that parses to a mapping carrying `crystal_structure` or `from_mcif`. The `EXTRA` list
is gone. The three `*_fit_params.yaml` outputs are excluded by the same content test
with no name-based special case.

That shape mattered: `examples/fitting/fit_dispersion.yaml` was one directory shallow
AND not named `config*`, so it was invisible on both counts — and went on shipping as
TUTORIAL.md's `magcalc fit` example with every bond listed in one direction only
(halving each J) and no `magnetic_structure` at all (expanding about a stationary
maximum), while its own "recovers the true values" check passed because the shipped
data had been generated from that same broken model. Adding it to a list fixed that
one file and left the shape. **Cross-checked when the content test landed: it
discovers EXACTLY the same 58 configs**, so it is a change of shape, not of coverage.

**Mind the two numbers.** `examples/future_exmaples/` is gitignored, so its configs
exist only in a working tree that has them: coverage is **58 configs here, 54 on a
fresh clone**. Item 7's fix moved the corrected FeI2 physics into the tracked
`examples/materials/FeI2/config_fei2.yaml`, so the staging FeI2 config is now a
redundant near-duplicate and the smoke test runs both (~4.5 min each). Deleting the
staging copy would be reasonable and was left to the owner. Re-check the numbers with

    python -c "import sys; sys.path.insert(0,'tests'); import test_config_smoke as t; print(len(t._configs()))"

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

## 7. FeI2 dipole — closed

**Status: DONE 2026-08-15.** `examples/materials/FeI2/config_fei2.yaml` no longer
carries `on_imaginary: warn`; it runs with the guards at their default `error` and
reaches max |Im ω| = 2.4e-15 meV. `tests/test_sun.py` is green (26 passed), including
the Sunny-pinned E/site and band comparisons this item was afraid of moving.

**The physics (recorded 2026-08-12, unchanged).** FeI2 orders as a COLLINEAR
2-up-2-down stripe at k = (0, −1/4, 1/4), and no rotating-frame
`single_k`/`propagation_vector` form can represent it — that form rotates each
successive cell by a fixed angle, giving up / in-plane / down / in-plane at k = 1/4.
On the real-space `magnetic_supercell: [1, 4, 4]` (16 sites), annealing reaches
**E = −46.372796 meV per cell = −2.898300 meV/site**, reproducibly. The config's old
declared structure sat ~2.5 meV/site above it. This is the DIPOLE minimum; the SU(N)
ground state is −2.91893118 meV/site, and with an anisotropy present the two genuinely
differ (CLAUDE.md §5c) — they are not meant to agree.

**What unblocked it: separating the two roles WITHOUT a second copy of the exchange
table.** The obstacle was real — `tests/test_sun.py` uses this file as its Hamiltonian
source and builds its own non-diagonal SU(N) supercell from it, so giving the config a
magnetic cell would have silently handed those Sunny-validated comparisons a 16-site
cell and the wrong reciprocal basis. The fix is `_fei2_cfg()` in that test: it loads
the file and strips `crystal_structure.magnetic_supercell`, `magnetic_structure` and
`minimization` in memory, so the tests see the CHEMICAL cell. A structure-free second
file would have duplicated a 7-rule exchange table — the drift hazard that produced
item 14. `test_the_fei2_config_ships_the_magnetic_cell_and_the_tests_strip_it` pins
BOTH halves in one place, because neither failure announces itself in a spectrum: lose
the supercell and the shipped config is a Hamiltonian with no ground state; lose the
strip and the Sunny comparisons quietly run on 64 sites.

**The gitignored staging copy is no longer the only home for this.**
`examples/future_exmaples/FeI2/config_fei2.yaml` carried the corrected config and
existed in no commit; its content is now the tracked
`examples/materials/FeI2/config_fei2.yaml`. The staging file was left in place (it is
the owner's) and is now a redundant near-duplicate that the smoke test also runs — see
item 5.

## 8. Studio — the two limits left after the 2026-08-12 open→run fix

**Status: DONE 2026-08-13.** Both are closed. Nothing outstanding; the notes below
are what the second one turned up, which was much more than the item expected.

**1. Relative paths from the web app — done as written.** The Open button now goes
through the server: `GET /recent-configs` + `POST /browse-configs` (a directory
walk, since the recent list starts empty) feed a picker, `POST /load-config`
returns the **abspath**, and the app sends `config_dir` on every run exactly as
the native app does. `from_mcif:` / `fitting.data_file:` / `cif_file:` therefore
resolve. Save on a server-opened file goes back through `/save-config` (no
writable browser handle exists for it), and the two browser routes ("Load YAML",
and Open when the backend is unreachable) stay, clearing `config_dir` so they keep
the old project-root behaviour rather than inheriting a stale directory.

Pinned by `tests/test_gui_relative_paths.py` (4 tests, fast), whose oracle is the
CLI: the shipped `examples/materials/mcif` config — the one config that reaches a
sibling file by relative path — opened and run through the server reproduces
`magcalc run <that file>` to 1e-12, and **the same payload aimed at a directory
without the mCIF fails**, which is the control that a server ignoring `config_dir`
could not pass. One incidental cleanup made that control possible without writing
into the real checkout: `/run-calculation` recomputed `project_root` as a local,
so nothing could redirect the default run directory; it reads the module-level
global now.

**2. The Swift emitter had no test — and disagreed with the web one on ALL 58
shipped configs.** The item called this "no second implementation is tested" and
expected drift; what it was, was four separate live defects. The tool built to
find them is `magcalc-emit-config`, a third xcodegen target compiling the SAME
`Sources/Models` headless (`native/MagCalcStudio/Tools/EmitRunConfig/`), diffed
against `node gui/tests/emit_run_config.mjs` by
`tests/test_native_emitter_parity.py` (4 configs fast + all 58 slow + 3 direct
pins). Every diff below was measured before and after.

- **`mergeEdits` read "the file has no such block" as "emit the app's whole
  struct".** The web app's starts from `clone(fileBlock) || {}` and skips
  untouched defaults. So opening a config with no `minimization:` added `method:
  anneal, n_sweeps: 2000, num_starts: 4, early_stopping: 10` to the run, and one
  with no `fitting:` gained a placeholder fit. This is item 6's class exactly —
  an injected anneal-only key is what crashed the minimizer there, after which
  the run died at the ground-state guard blaming the magnetic structure. 54 of
  58 configs.
- **`fitting.data_file`, `vary`, `bounds`, `scale`, `background`,
  `energy_broadening` and `minimization.n_sweeps` had NO import branch but WERE
  re-emitted from the struct.** A struct field that is written but never read is
  strictly worse than one that is missing: it silently overwrites. Opening
  `examples/fitting/fit_dispersion.yaml` and pressing Fit sent `data_file: ""`,
  `vary: []`, `bounds: {}` — it fitted nothing.
- **The crystal block was re-emitted from the file verbatim**, so every edit made
  in the native Structure panel was discarded after opening a config (the web app
  fixed the same bug in `buildStructPayload` earlier). Switching to the editor's
  atoms then required somewhere to keep the per-site keys the app does not
  model — `g` (32 sites across the examples; it IS the Zeeman term), `charge`,
  `wyckoff`, `species` — hence `WyckoffAtom.extras`, and `SymmetryInteraction`
  needed the same for `name` plus an OPTIONAL `distance` (it was defaulting to
  0 and emitting `distance: 0.0` on `ref_pair` + `offset` rules).
- **Smaller, all real:** `parameter_order` was emitted only when the file carried
  one; the global `S` was not stripped; `cache_mode: auto` was not supplied; the
  `calculate_dispersion`/`calculate_sqw_map` aliases were absent (harmless — the
  runner defaults both to True — but a difference).

**Why parity is a legitimate oracle here** (it looks like comparing two
unverified things): the JS side is itself pinned to `magcalc run` by
`test_gui_roundtrip.py`, band for band on four configs, so "equals the JS
emitter" chains back to the CLI. What parity cannot catch is both sides being
wrong the same way, so the three losses above are ALSO pinned directly against
the file (`test_native_keeps_the_fitting_block_the_file_declared`,
`…keeps_per_site_keys_it_does_not_model`,
`…does_not_invent_a_block_the_file_omitted`).

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

**Related, not yet chased → item 11:** `thermal_mc.build_supercell` carries no
single-ion anisotropy at all. It is a *different* builder from the annealer's
(`MagCalc._extract_classical_quadratic`, which is correct — verified), so nothing in
this item is affected by it.

---

## 10. KPM has no ground-state guard, and cannot grow one for free

**Status: DONE 2026-08-13.** `SUNModel.is_stable_at(q)` / `assert_stable(qs)` /
`min_h2_eigenvalue(q)` (`magcalc/sun/lswt.py`), called by the runner's `kpm_sqw` task
at EVERY q it computes and by `disorder_kpm.py` -- one implementation, one criterion.
Pinned in `tests/test_kpm_stability.py` (26 tests, all fast). Nothing outstanding;
the notes below are what the answer turned on.

**The cost objection dissolved, and that decided the design.** The question is BINARY
-- is H2(q) positive definite -- so it does not need an eigensolve at all: a Cholesky
decides it exactly and is **45x cheaper than `eigvalsh` at 2D = 1800, 65x at 3200**
(73 ms vs 3.3 s). Against KPM's own 1.7 s/q at S09's L = 30, and sharing the g H2
build via the new `kpm_sqw(..., hmat=)`, the guard costs **1.3 % of a KPM q at
2D = 288 and 4.9 % at 2D = 1800**, measured end to end. So there was no
need to sample q thinly, which is the part that would have been dangerous: the
instability is q-specific (4 generic q find it on 1 realization in 3, a 40-point path
on 2 of 3), and a thin sample is the "a check a wrong answer passes" shape again.
Check every q you compute. The Lanczos idea recorded here was measured too (m = 60
caught every negative eigenvalue at 18 % of the KPM cost) and dropped: its error is
one-sided in the WRONG direction -- Ritz values bound lambda_min from ABOVE, so it can
report a genuinely unstable state as stable.

**The threshold is a shifted Cholesky, and the shift is not a fudge.** `H2 + eps I`
positive definite <=> min eig H2 > -eps. Without a shift the guard refuses every
gapless magnet: a Goldstone mode puts an EXACT zero eigenvalue in H2 at the ordering
wavevector, and a ferromagnet's H2 is identically zero at Gamma -- which is in every
path ever plotted. For the same reason the scale multiplying `h2_rel_tolerance`
(default 1e-6) is **q-independent** (`_reference_h2_scale`, measured once at two
generic q sized from the model's own bond lengths): a purely per-q relative threshold
is zero exactly where the band touches zero, and inconsistent from q to q everywhere
else. The default was measured, not chosen -- S09's 144-site cell over a 37-point path
through Gamma and K:

| state | min eig H2 | relative |
|---|---|---|
| clean, exact 120-degree | -3e-15 | 7e-16 |
| sigma = 0.1 disorder, relaxed | -2e-10 | 5e-11 |
| sigma = 1/3 disorder, relaxed | -3e-3 | 5e-4 |

seven orders between the noise floor and a real instability, so the tolerance sits
five orders above one and two and a half below the other.

**The guard is not redundant with the two that already ran, and the control says so.**
A frustrated FERROMAGNETIC chain (J1 < 0, J2 > |J1|/4, whose true state is an
incommensurate spiral) is a genuine minimum WITHIN its one-site cell, and its
spectrum is entirely real -- max |Im w| = 0 exactly. The energy audit relaxes and
stays put; the imaginary check reads zero; a `dispersion` run of that config succeeds
and plots a plausible band (pinned as a test, so it stays a control). Only H2(q) >= 0
knows better. It is also strictly sharper than the imaginary check in general: at a
stationary MAXIMUM H2 stays diagonal, so g H2 is entirely real. (The dipole engine's
Luttinger-Tisza guard would catch the chain -- it does not run in SU(N) mode.)

**Two things found in passing.** `kpm_sqw` was missing from the runner's `_lswt_tasks`,
so `{kpm_sqw: true, thermal_mc: true}` silenced the up-front guards for the whole run
(fixed, pinned). And at a q where H(q) is identically zero -- a ferromagnet at Gamma
-- the spectral bound gamma was 0, so `D-hat/gamma` produced a whole **NaN q-column**,
silently, into the saved .npz, the plot, and the logged intensity range. It is finite
now and warns: the value there is 0, not the q -> 0 limit, because the +/-omega poles
coincide and cancel. There is no oracle at that single q (the exact path cannot be
evaluated there either -- the Cholesky in `_bogoliubov` fails on the Goldstone zero),
so the warning says "read that column as undefined" rather than inventing a number.

---

## 11. `thermal_mc.build_supercell` carried NO single-ion anisotropy

**Status: DONE 2026-08-15.** `magcalc/thermal_mc.py::onsite_quadratic` +
`tests/test_thermal_mc.py` (6 new tests; **5 of the 6 were confirmed to FAIL on the
pre-fix code**, the sixth being the RCS one, which passed vacuously against a zero
matrix and was strengthened so it cannot).

**What it was.** `build_supercell` assembled the classical `H` from `spin_interactions`
alone — a BOND table — so `single_ion_anisotropy` / `sia_matrix` / `stevens` never
entered it, while its own docstring called `H` "the exchange/anisotropy Hessian".
Measured on a bond-free model with D = 2.5: `H` came back with **exactly zero nonzero
entries**. It fed `thermal_mc`, `wang_landau`, `static_correlations` and the classical
`sampled_correlations`, so an anisotropic magnet was silently sampled as exchange-only
in all four.

**How it is fixed, and why not by re-implementing.** `onsite_quadratic` calls the
model's OWN `_compute_sia_terms` (which already folds in `_compute_sia_matrix_terms`
and `_compute_stevens_terms`) on symbolic components, then extracts (H, b) by the same
exact probing identities `MagCalc._extract_classical_quadratic` uses. So anisotropy
targeting, parameter resolution and the RCS renormalization
(`calculation.anisotropy_renormalization`) cannot drift from what LSWT diagonalizes —
a second transcription of the on-site rules is precisely the kind of duplicate that
produced the two dipolar prefactors (item 14).

**A limit, made loud rather than silent.** A Stevens term of rank k ≥ 4 has a
quartic/sextic classical polynomial and simply cannot live in `E = ½mᵀHm + bᵀm`. It now
RAISES, naming `mode: SUN` as the route that carries the full operator. Rank 2 (`sia`,
`sia_matrix`, `stevens` k = 2) is exactly quadratic and is carried exactly.

**The oracle is the exact single-spin partition function**, as this item predicted: one
classical spin in a field with a uniaxial D has E(u) = D S²u² + bSu, so ⟨E⟩ and ⟨m_z⟩
are 1-D quadratures — the same shape the Langevin and classical-dimer tests here
already use. The structural pin is sharper still: on a D-only model the on-site block
must be exactly 2D·nnᵀ, which is the defect stated as a number.

**Blast radius, swept: no shipped config was affected.** Of the four configs that use a
sampler fed by this builder, only `examples/sunny_tutorials/S04_FeI2_finiteT` carries an
on-site anisotropy — and it uses `sun_sampled_correlations`, i.e. the CP^(N−1) path
through `sun/lswt.py`, which builds its on-site terms itself. Nothing else could have
been.

---

## 12. Time-domain window on the classical S(q,ω)

**Status: DONE 2026-08-15, as an OPT-IN** — `classical_dynamics.lag_window` /
`_window_correlation`, shared by the dipole and CP^(N−1) transforms,
`window: cosine|rectangular` on both `sampled_correlations:` and
`sun_sampled_correlations:`. Pinned in `tests/test_classical_window.py` (10 tests).
**The default stays `rectangular`, and that decision is a measurement — see below and
item 15.**

**The implementation is Sunny's, and it reduces to one line of algebra.** The window is
applied to the CORRELATION in the lag domain (Blackman–Tukey, as Sunny does), not as a
taper on the trajectory. `w(Δt) = cos²(π|Δt|/n_t)`, and

    cos²(x) = ½ + ¼e^{2ix} + ¼e^{−2ix}

so windowing the correlation is **IDENTICAL to convolving the spectrum with the
3-point kernel [¼, ½, ¼]** — one bin Δω = 2π/T of Hann broadening, exactly. Measured
agreement with that identity: **1.1e-16** on a spectrum of scale 1.9. Everything else
follows from it rather than being observed:

- the kernel is non-negative → a windowed S(q,ω) cannot dip below zero where the raw
  periodogram was positive (a general lag window CAN);
- it sums to 1 → the **two-sided** ω-integral, hence every sum rule, is preserved to
  machine precision.

**That second point is why this item needed its own oracle.** This item predicted that
`tests/test_classical_absolute_normalization.py` would be insensitive to the change,
and it is — *exactly* so, not approximately. The identity above is the oracle instead.

**WHAT THE ITEM GOT WRONG, and it is the important part.** The item proposed the
window as a fix for the ~16 % whole-axis overshoot and offered the default as a
preference. It is not a preference. The same one-bin smear lands on the ELASTIC delta
of an ordered magnet, and `classical_to_quantum_factor` is 1 at ω = 0 but |ω|/kT one
bin away — **31** at kT = 0.005 with Δω = 0.153 meV. On the gapped ferromagnetic chain
(L = 24, n_traj = 16, q = 0.15; LSWT `perp` band sum 0.5):

| window | `subtract_elastic` | whole-axis / LSWT | first inelastic bin |
|---|---|---|---|
| rectangular | false | 1.55 | 0.00006 |
| rectangular | true | 1.40 | 0.00006 |
| cosine | false | **2.60** | **9.10** — 18× the entire band sum, from ONE bin |
| cosine | true | 1.40 | 0.00005 |

The two windows agree once the delta is removed: it *was* the whole difference. So the
window makes the whole-axis integral WORSE, not better, on an ordered magnet — unless
the elastic line is removed first. `subtract_elastic` was therefore added to the dipole
path too (the SU(N) path had it all along), and `window: cosine` +
`subtract_elastic: true` is the combination that behaves.

**Do not compare the table above with item 4's 1.16 / 1.015.** That was measured at
L = 32, n_traj = 128, n_steps = 4096 and five q; this is a cheaper setting (L = 24,
n_traj = 16, n_steps = 2048) forced by machine load, so its absolute ratios carry
finite-size and statistical error that item 4's do not. What it establishes is the
RELATIVE effect of the window, which is what this item is about. Re-measuring the
absolute table at item 4's setting is worth doing and was not done.

---

## 13. The CP^(N−1) sampler did not equilibrate at low kT

**Status: DONE 2026-08-15.** `sun/dynamics.thermalize` adapts the Metropolis step size
and reports what it did (`ThermalizeInfo`); `sun_sampled_correlations:` gains
`adapt_sigma`, `target_acceptance`, `on_unequilibrated`. Pinned in
`tests/test_sun_sampler_equilibration.py` (12 tests).

**The design, and why it is split in two halves.** The step size is tuned toward
`target_acceptance` (~0.5) over the FIRST half of the sweeps and then held FIXED for
the second half — adapting while measuring would break detailed balance, and the
equilibration verdict has to be made on a chain whose proposal is not moving. The
adaptation gain decays as 1/√k so it settles rather than oscillating. The upper clip on
`sigma` is 10 and is not a limitation: the proposal is `Z + σv` RENORMALIZED, so by
σ ≈ 10 the candidate is a uniform draw on the sphere, i.e. an independence sampler —
which is why at high kT the target acceptance is simply unreachable (even a uniform
draw is accepted ~83 % of the time) and the adaptation saturates instead of running
away.

**The oracle is the sampler's own partition function, in closed form**, which is what
makes this a test of the SAMPLER with no spectrum and no reference code in the way. For
a decoupled site with on-site A = diag(a₁..a_N) the coherent-state energy is
Σᵢ aᵢ|zᵢ|², and the Fubini–Study measure makes |z|² uniform on the simplex, so

    Z(β) = ∫_simplex e^{−β Σ aᵢpᵢ} dp = Σᵢ e^{−βaᵢ} / Π_{j≠i} β(a_j − aᵢ)

exactly (checked against a 2-D quadrature before being used). ⟨E⟩ = −d lnZ/dβ.

**Measured, at kT = 0.05 / 0.2 / 1.0 on 6 decoupled S = 1 sites:**

| | ⟨E⟩ vs exact | acceptance |
|---|---|---|
| fixed σ = 0.02, kT = 1 | **+32 %** (−2.57 vs −3.78) | 0.995 — accepts everything, moves nowhere |
| fixed σ = 0.02, kT = 0.05 | +1.5 % | 0.894 |
| adapted, from σ = 0.02 | within 1.5 % at every kT | 0.47–0.83 |
| adapted, from σ = 0.5 | agrees with the above to 4 % | 0.46–0.83 |

The last two rows are the property the item named — the answer no longer depends on
where `sigma` started — and the drift diagnostic flags exactly the three broken runs
and none of the good ones.

**Not re-measured: the end-to-end 0.30 → 1.63 intensity swing** this item quoted from
item 4. A partial run (n_traj = 4, one q) narrowed it from a factor 1.76 to a factor
1.16 before the machine load made the full comparison unaffordable. The sampler itself
is now pinned against an exact result, which is the stronger claim; the end-to-end
number is worth re-taking.

---

## 14. The two dipolar prefactors are reconciled

**Status: DONE 2026-08-15.** `magcalc/constants.py`:

    MU0_MUB2_MEV_A3 = 0.6745817653324668            # Sunny Units.jl, full precision
    DIPOLE_PREFACTOR_MEV_A3 = MU0_MUB2_MEV_A3 / (4*pi)   # DERIVED, not typed

**Which value is right was not a matter of taste.** μ₀μ_B² is a physical constant,
Sunny states it to full double precision (`Sunny.jl-main/src/Units.jl`,
`vacuum_permeability`), and 4π is exact. The old `DIPOLE_PREFACTOR_MEV_A3 = 0.05368216`
was **not a truncation of the Ewald constant at all** — μ₀μ_B²/4π is 0.05368151123615953,
so it was 1.2e-5 relative too large. The 4-pi-reduced constant is now derived and cannot
drift again; `tests/test_ewald.py::test_the_two_dipolar_prefactors_are_one_constant`
pins the derivation rather than the digits.

**Why this is NOT the `MU_B` case, which is the distinction worth keeping.** `MU_B`'s
four-figure truncation is load-bearing because every pinned Zeeman number in the repo
was MEASURED against it. Nothing was ever pinned against 0.05368216: the only test of
the truncated sum is a comparison with the Ewald path, which used the other constant.
There was no reference to preserve, only an inconsistency — so this is an accuracy fix,
where moving `MU_B` would not be.

**The oracle was re-run, not the tolerance widened**, which is what this item asked
for: `pytest tests/test_ewald.py tests/test_ewald_spiral.py -m ""` → **21 passed**,
including `test_truncated_sum_converges_to_ewald` (1e-4 absolute on a ~4 meV band,
where this shift is ~5e-5) and `test_ewald_classical_energy_matches_sunny` (1e-9
against Sunny's `energy_per_site`, which `MU0_MUB2_MEV_A3`'s own 4.8e-11 move had to
survive and did).

**Not measured: the before/after residual table** (max |truncated − Ewald| at cutoffs
12/30/45). The run was abandoned at ~20 min of wall clock with almost no CPU time
accrued — the cutoff-30+ real-space sums build a huge bond list and the per-q pool
spends its time spawning workers on a machine at load 300. The gate above is the
decisive evidence; the table would only have been colour.

---

## 15. `window: cosine` is a trap without `subtract_elastic` — now reported

**Status: DONE 2026-08-16.** `classical_dynamics.check_elastic_leakage`, called by BOTH
`classical_dynamics.sampled_correlations` and `sun/dynamics.sampled_correlations`, with
`on_elastic_leakage: warn|error|off` on both config blocks. Pinned in
`tests/test_classical_window.py` section 5 (7 new tests, 17 in the file).

**What it was.** The measurement in item 12: on an ordered magnet, `window: cosine`
alone puts the smeared elastic delta into the first inelastic bin, where `c2q`
multiplies it by |ω|/kT — 9.10 in a spectrum whose entire LSWT band sum is 0.5.
`subtract_elastic: true` removes it completely. All three switches are independent
booleans defaulting to false, so the dangerous combination was one keystroke away and
said nothing.

**Option 1 of the three, as this item predicted** — warn rather than imply, because
making `cosine` imply `subtract_elastic` would silently change what the config asked
for, and because leaving it to the docs is exactly the shape of hazard the house rules
say not to leave to documentation. What made it implementable is that the trigger is a
COMPUTED NUMBER, not a guess:

    amplification = c2q(Δω),   Δω = 2π/T the energy grid step

so the warning names the factor it is triggering on. It fires only when all four hold:
`window: cosine`, `subtract_elastic: false`, `classical_to_quantum` on, and
amplification ≥ 2 (i.e. kT ≲ Δω/1.6). Below that threshold the smear costs no more than
the one bin of Hann broadening `lag_window` documents — which is the point of using the
window — so warning there would only train the user to ignore it.

**Engine-level, not runner-level**, for item 10's reason: the S09-style scripts drive
`sampled_correlations` from Python and never see the runner. The runner passes the key
through on both blocks, and `test_the_guard_is_on_by_default_and_reachable_from_a_config`
drives a real config through `run_calculation` to pin that half.

**Oracle: item 12's four rows.** `check_elastic_leakage` must fire on
(cosine, subtract_elastic=false, kT=0.005) and stay quiet on the other three — the rows
that behaved — and the number in the message is checked against
`classical_to_quantum_factor` itself (an identity, not a constant retyped into the test),
which at item 12's grid reproduces the ~31 the item measured. The kT condition is
bracketed either side of the threshold on the factor rather than on a chosen kT.

**Gate:** the merge gate was run, not only the targeted files. `pytest -m ""` from the
workspace root → **837 passed, 3 skipped, 0 failed, 45:46** (2026-08-16), against the
2026-08-15 baseline of 830 passed, 3 skipped — +7 is exactly this item's new tests. The
targeted run first — `test_classical_window.py`, `test_config_key_coverage.py`,
`test_sun_dynamics.py`, `test_classical_dynamics.py`, `test_classical_to_quantum.py`,
`test_sun_sampler_equilibration.py`, `test_classical_absolute_normalization.py` at
`-m ""` → 65 passed (3:30). The 45-minute wall clock against the baseline's 30 is
machine contention (load average peaked at 56 mid-run, and the per-q pool phase is
where it went), not new work in the suite.

---

## Also worth knowing

### Standing facts (no action)

- **Branches.** `master` is the default (there is no `main`). On 2026-08-12 the
  merged feature branches (`feat/gap4-phase1..4`, `feat/gap4-26-sun-dynamics`,
  `fix/sunny-parity-audit`, `docs/cu5sbo6-powder-comparison`,
  `docs/rb2cu3snf12-order8`, `test/coverage-audit-items-2-4`) were pruned after
  checking each tip was an ancestor of `origin/master` with nothing unpushed —
  no history was lost. Items A–E were developed on `chore/open-work-housekeeping`,
  the item-1 machinery plus the C/D/E follow-ups on `feat/open-work-followups`, and
  items 1/2/3/4/8/9/10 on `feat/s06-cp2-skyrmions`; all were **fast-forward merged
  into `master`** after a green full gate. **Four merged branches are still present
  and every one is an ancestor of `master`, so all four can be pruned:**
  `feat/s06-cp2-skyrmions`, `docs/gate-637`, `feat/open-work-followups`,
  `chore/open-work-housekeeping`. **The 2026-08-15 work is uncommitted, on `master`
  itself** — branch it before committing.
- **The merge gate is `pytest -m ""`**, and from the *workspace root* plain
  `pytest` is already the full suite (the root `pytest.ini` deliberately does
  not inherit `-m "not slow"`). It takes ~30 min unloaded and can take 2.5 h on
  a busy machine. Don't pipe it through `tail` — that hides all progress until
  it exits.
- **Beware the runner's process pools when you write a measurement script.** Two
  supporting measurements were abandoned this session after ~20 minutes of wall clock
  each with under 16 s of CPU accrued: the per-q pool was spawning short-lived workers
  faster than a loaded machine could start them. A harvest script that `chdir`s is
  worse still — multiprocessing's spawn re-executes the main module by path, so the
  workers died on a relative `sys.path` entry and were respawned in a loop (that is
  what took the load average past 300). Use absolute paths and an
  `if __name__ == "__main__":` guard, or drive one config per subprocess.

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

### C. Interpreter-startup shadow guard — **DONE 2026-08-15**

- **The guard** — `magcalc/_shadow_guard.py`, installed by `magcalc guard --install`
  (`tools/install_shadow_guard.py` still works for a source checkout; the module lives
  INSIDE the package because `tools/` is not shipped by a non-editable
  `pip install`, so a wheel user could not install the guard at all). It is the half
  the in-package detector structurally cannot cover: the guard sits in site-packages,
  *outside every `magcalc` copy*, so it reports no matter which one wins — including a
  brand-new checkout.

  ```bash
  magcalc guard              # report status (per interpreter)
  magcalc guard --install    # / --uninstall
  ```

  `magcalc where` states whether it is active, so the protection level is never a
  guess. `MAGCALC_SHADOW_GUARD=off` silences it for the legitimate case (a git
  worktree, a deliberate version comparison). Cost: **254 µs** per interpreter
  startup, plus one string compare per import.

  **THE TIMING TRAP, and the reason the guard is not a simple startup check.** The
  obvious implementation — survey `sys.path` when the `.pth` runs — is blind to the
  main hazard. At `.pth` execution time `sys.path[0]` is **not yet the working
  directory**: for `python -c` the `''` entry is prepended *after* site
  initialisation. The first version did exactly that, passed a PYTHONPATH test, and
  reported all clear when run from inside a stale checkout. The check is therefore
  **deferred** to the moment `magcalc` is imported, via an observer parked at the
  front of `sys.meta_path` whose `find_spec` surveys and then always returns `None`.
  `tests/test_shadow_guard.py` pins this with a throwaway venv carrying two real
  `.pth` files — the guard, and an eager probe that records what a startup check
  *would* have concluded (`EAGER=False`).

  Second non-obvious detail: stderr alone is not enough. Under `pytest` the import
  happens inside the capture, so the banner is buffered and shown only if something
  else fails — and "ran the suite inside a stale checkout" is exactly a case where
  everything passes. The guard therefore also raises a real `MagcalcShadowWarning`,
  which lands in pytest's warnings summary on a green run.

- **CLOSED 2026-08-15: the guard now ARMS ITSELF on the first `magcalc` CLI use.**
  `magcalc.cli._arm_shadow_guard`, called from an `@app.callback()` that runs before
  every subcommand: if the two files are already there it returns after two `stat`
  calls, otherwise it installs them and prints one line to stderr. A fresh venv is
  therefore protected from its first command, which is what "nothing installs it" was
  about.

  **Why run-time and not build-time.** Wiring it into `pip install -e .` was rejected
  earlier for a good reason and that reason still stands: a build hook that writes
  into site-packages can break the install itself (sandboxed builds, read-only or
  root-owned prefixes, cross-built wheels). Doing it on the first CLI run keeps the
  effect and drops the hazard — it is the user's own interpreter, it is recoverable,
  and every path is swallowed. `MAGCALC_SHADOW_GUARD=off` suppresses the install as
  well as the guard, so the deliberate-second-checkout workflow is untouched.

  **The test found a live defect in the first version of this**, which is worth
  keeping: `_shadow_guard_install.site_packages()` raises `SystemExit` when it cannot
  find purelib, and `except Exception` does NOT catch that — so an unusual prefix
  would have killed the calculation over a diagnostic, the exact thing the guard's own
  design rules forbid. It catches `(Exception, SystemExit)` now, and still lets
  `KeyboardInterrupt` through. Pinned by
  `test_arming_never_fails_the_command`, plus two throwaway-venv tests
  (`test_first_cli_run_arms_the_guard_in_a_fresh_environment`, which also checks the
  notice is NOT reprinted on the second run, and
  `test_the_env_var_opt_out_also_suppresses_the_install`).

### D. Stale OneDrive trees — **DONE 2026-08-12** (both follow-ups now closed)

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
    GAP_STATUS.md means. **CLOSED 2026-08-12** by
    `tests/test_ccsf_fit_roundtrip.py`. It had been exercised only by
    `test_config_smoke.py`, which asserts the run produces no ERROR records — not
    that the fit RECOVERS anything. The new test asserts against **the values that
    generated the data** (J1 = 13.3, J2 = −0.24, read from the CSV header), not
    against the output observed at rescue time (13.2860735, −0.2372…), so a fit that
    drifts cannot certify itself; a second test checks the header still records those
    generating values, so the two cannot silently diverge. (The two fit outputs stay
    gitignored, since the smoke test rewrites them in the merge gate.)
  - `archive/legacy/aCVO_2024_snapshot/spin_model_sf.py` — the legacy hand-written
    α-Cu₂V₂O₇ model, reference only. **CLOSED 2026-08-12:** it is tracked at
    `examples/materials/aCVO/legacy_spin_model_sf_2024.py`, with a provenance header
    explaining why it lives there rather than in `archive/` (the workspace root is not
    a git repository and `pyMagCalc/archive/` is gitignored, so there was nowhere else
    it could be version-controlled at all). The root `archive/` copy is byte-identical
    below that header and is now a duplicate, not the only copy.

  The other ~8900 untracked files were a `gui/node_modules.onedrive-bak/` copy.

### E. `mu_B` consolidation — **DONE 2026-08-12** (the prefactor question is item 14)

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
- **The dipolar prefactors were folded in too, and NOT reconciled — see item 14.**
  `MU0_MUB2_MEV_A3 = 0.6745817653` and `DIPOLE_PREFACTOR_MEV_A3 = 0.05368216` both
  live in `constants.py` now (`ewald.py` and `generic_model.py` import them), so the
  duplication this entry was about is gone. What remains is that they are the same
  Sunny constant with and without the 4π and **their values disagree at 1.2e-5
  relative**, which is a numerical question rather than a refactor.
