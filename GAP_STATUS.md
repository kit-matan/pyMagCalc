# pyMagCalc ↔ SpinW/Sunny feature-gap status

A running record of the gaps between pyMagCalc and SpinW/Sunny, what has been closed, and
what remains — written so a future session (AI or human) can pick up without re-deriving
the context. Keep it updated when a gap moves.

Reference oracle: **Sunny.jl 0.8.1** is checked out in-repo at
`../Sunny.jl-main`, and Julia 1.12 + Sunny 0.8.1 are installed. Use them (and textbook
analytic results) to validate every new feature — see "How things were validated" below.

Status: **merged to `master`** (PR #2). All of the below — Gap 1, Gap 2, Ewald
(Gap 3 #7), SU(N) (Gap 3 #1), 1/S corrections (#8), mCIF (#12), the Studio web +
native apps, and the Sunny.jl tutorial ports — now live on the default branch. The
two former development branches (`feature/sun-mode`, `feature/gap-closure-ewald`)
have been consolidated and retired; `feature/gap-closure-ewald` was fully contained
in the merge and is deleted.

Test suite: 208 tests (`python -m pytest tests`). Every new feature has a test that
pins it to an **independent reference** (Sunny, or an exact analytic identity), never a
self-generated golden number.

---

## Gap 1 — Intensity / experiment layer — ✅ CLOSED

The neutron-intensity / experiment surface. Config keys under `calculation:` and
`plotting:`, plus a task.

| Item | Status | Key(s) | Validated against |
|---|---|---|---|
| Bose thermal factor | ✅ | `calculation.temperature` (K) | Sunny `kT`; detailed balance |
| Domain / twin averaging | ✅ | `calculation.domains` (`{axis,n_fold}` or list) | manual twin average; exact for perp/trace |
| Cross-section selection | ✅ | `calculation.cross_section` (perp/trace/xx/…) | tensor sum rules |
| Instrument resolution | ✅ | `plotting.resolution` (`de_fwhm` poly, `dq_fwhm`, `ei`, `two_theta`, `shape`), `plotting.energy_grid_step` | SpinW `sw_instrument` (SW37) |
| 2-D constant-energy cuts | ✅ | `tasks.energy_cut` + `energy_cut:` block | SW10 |
| Polarized / chiral cross-sections | ✅ | `cross_section: chiral` / `sf±` | Sunny, proper-screw helix |

Tests: `tests/test_intensity_layer.py`, `tests/test_polarized.py`.

---

## Gap 2 — Hamiltonian terms — ✅ CLOSED

Terms beyond bilinear exchange. All under `interactions:` except the g-tensor (per atom)
and multi-k (magnetic_structure).

| Item | Status | Key | Validated against |
|---|---|---|---|
| 3×3 single-ion anisotropy | ✅ | `interactions.sia_matrix` | reduces to uniaxial `sia` |
| Stevens operators O_k^q (k=2/4/6) | ✅ | `interactions.stevens` | table generated from Sunny `stevens_matrices(Inf)`; O_2^0 identity |
| Anisotropic per-site g-tensor | ✅ | atom `g:` (scalar/diag/3×3/`{g_par,g_perp,axis}`) | g=2 ≡ legacy Zeeman; SW20 in field |
| Biquadratic exchange | ✅ | `interactions.biquadratic` | exact collinear map to 1e-15 |
| Long-range dipole-dipole (truncated) | ✅ | `interactions.dipole_dipole: {method: truncated, cutoff}` | Sunny truncated sum to 3e-8 |
| Multi-k structures | ✅ | `magnetic_structure: {type: multi_k}` + supercell (per-axis LCM) | one-component k=½ ≡ Néel chain |

**Two latent engine bugs fixed while doing this** (see CLAUDE.md; both were silent,
plausible-but-wrong):
1. The LSWT truncation filtered by *powers of S*, silently deleting every quartic term
   (biquadratic, Stevens O_4/O_6, whose quadratic-boson part carries S³). Now truncates
   by **boson degree**. Verified inert on all prior configs (dispersions byte-identical).
2. The classical energy ignored the g-tensor and assumed **B ∥ z** on the numeric-param
   path — which the minimizer uses — so it optimized a *different* Hamiltonian than LSWT
   diagonalized (wrong ground state, imaginary magnons). Field vector now resolved once.

Tests: `tests/test_hamiltonian_terms.py`.

---

## Gap 3 — Beyond dipole LSWT

### Tier 1 (affects results you would publish) — ✅ ALL DONE

| # | Item | Status | Key | Validated against |
|---|---|---|---|---|
| 1 | **SU(N) mode** | ✅ | `calculation.mode: SUN` | Sunny `:SUN` — FeI₂ energy, all 8 bands, AND intensities to 4e-7 |
| 2 | Fitting sees temperature/domains/cross_section | ✅ | (auto from `calculation:`) | ignoring T biased J by 17% |
| 3 | Mixed-spin intensity prefactor | ✅ | (per-site √(S_i/2)) | decoupled-sublattice identity (was 60% error) |
| 4 | Polarized / chiral cross-sections | ✅ | `cross_section` | Sunny (also listed under Gap 1) |

**SU(N) detail** (`magcalc/sun/`, `tests/test_sun.py`): a second LSWT engine (as in
Sunny). Each site carries an N=2S+1 level Hilbert space with a coherent state; N-1 bosons
per site; captures single-ion (multipolar) excitations dipole LSWT structurally cannot.
Validation gates, in order of how loudly they fail:
- **Gate 1** — S=1/2 (N=2) is *identical* to dipole LSWT (0.0e+00). Any convention error
  (phase, Bogoliubov metric, factors, on-site mean field) fails here.
- **Gate 2** — no single-ion terms: dipole bands reproduced; extras are flat Δm≥2 modes.
- **Gate 3** — single-ion anisotropy vs Sunny `:SUN` (4.7e-07), incl. the quadrupolar band.
- **FeI₂** — E/site −2.91893118, 8 bands + intensities to 4e-4, via the config bridge +
  a non-diagonal magnetic supercell + the CP^(N-1) ground-state search.

Runs on 38/47 example configs; the rest **refuse honestly** (incommensurate/spiral/auto-
supercell → not supported; mixed-spin → not yet; frustrated GS search not converging →
guard refuses). Never silently wrong.

### Tier 2 (capability parity) — ✅ ALL DONE (Wang–Landau closed by Gap 4 #22)

| # | Item | Status | Notes |
|---|---|---|---|
| 5 | **Finite-T classical dynamics (SampledCorrelations)** | ✅ | `tasks: {sampled_correlations: true}` + `sampled_correlations: {temperature, supercell, dt, n_steps, n_traj, therm_sweeps}`. Real-time Landau–Lifshitz dynamics (undamped RK4, dS/dt=−S×B) on Metropolis-thermalized states; S(q,ω)=⟨\|Σ_r e^{−iq·r}S_r(t)\|²⟩ by space-time FFT, whole q-path × energy grid in one shot. Validated: single-spin Larmor ω=gμ_B B; RK4 energy conservation (drift ~1e-8); the low-T ferromagnet S(q,ω) peaks fall on the EXACT LSWT magnon dispersion the engine computes (<5%, →0 as T→0). `magcalc/classical_dynamics.py`, `tests/test_classical_dynamics.py` |
| 6 | **Thermal Monte-Carlo (parallel tempering)** | ✅ | `tasks: {thermal_mc: true}` + `thermal_mc: {temperatures, supercell, n_sweeps, n_equil}`. Finite-T thermodynamics on an explicit PBC supercell (built from `spin_interactions` + `_resolve_field`, same classical energy ½mᵀHm+bᵀm the minimizer uses); replica-exchange Metropolis over a T-ladder. Reports ⟨E⟩/N, C/N=Var(E)/(NkT²), magnetization, susceptibility. Validated: N free spins in a field = Langevin −L(βgμ_B·B·S) exactly; classical Heisenberg dimer ⟨E⟩(T)=−JS²L(βJS²) and C(T) from the exact partition function; parallel tempering ≡ independent single-T Metropolis. Wang–Landau still not done. `magcalc/thermal_mc.py`, `tests/test_thermal_mc.py` |
| 7 | **Ewald dipole-dipole** | ✅ | `dipole_dipole: {method: ewald}`; Sunny to 1.3e-8; truncated→Ewald convergence (needs no Julia). `tests/test_ewald.py` |
| 8 | **LSWT 1/S corrections** | ✅ | `tasks.corrections`; Sunny + textbook square AFM: dE=−0.157947, dS→0.1966. `magcalc/corrections.py`, `tests/test_corrections.py` |
| 9 | **SCGA (paramagnetic diffuse scattering)** | ✅ | `tasks: {scga: true}` + `scga: {temperature, mesh_density, cross_section}`. Self-consistent Gaussian approximation: classical spins, hard length constraint softened to a global Lagrange multiplier λ, static S(q) = kT·pref†(λ+J(q))⁻¹pref with λ from the spin sum rule. SAME `fourier_exchange_matrix` J(q) as the LT guard; single-λ (one symmetry class — Bravais + kagome/pyrochlore-type, refuses inequivalent sublattices). Above T_N, so the LSWT ground-state guard is auto-skipped for a pure-SCGA run. Validated: exact classical-chain closed form (λ=√(4J²+(3kT/S²)²), S(q)=3kT/(λ+2Jcosq)) to 1e-9; **Sunny 0.8.1 SCGA** on square-lattice AND kagome AFM — λ and S(q) to 6 digits (matches `ssf_perp`, apply_g, (2/3)Tr at q→0); sum rule + high-T flat limit. `magcalc/scga.py`, `tests/test_scga.py` |
| 10 | **KPM (Chebyshev spectral S(q,ω))** | ✅ | `tasks: {kpm_sqw: true}` + `kpm: {e_min,e_max,e_step,fwhm,moments/tol}` (SU(N)/entangled modes). Para-unitary Chebyshev expansion of the one-magnon spectral function of the dynamical matrix D̂=gH₂ (=`SUNModel.hamiltonian`), following Lane et al. (arXiv:2312.08349) — Sunny's `SpinWaveTheoryKPM` method. Iterated matvecs only, no eigensolve: O(D·M) per q for large disordered/near-incommensurate cells. Validated against the engine's OWN exact diagonalization (`structure_factor`): as M grows KPM S(q,ω) → the exact Gaussian-broadened spectrum (rel L1 < 1e-3, integrated intensity to 1e-3); spectral bound γ verified to enclose the spectrum. `magcalc/sun/kpm.py`, `tests/test_kpm.py` |
| 11 | **Entangled units** | ✅ | `calculation.mode: entangled` + `units:` -- a cluster (dimer/trimer/...) becomes ONE effective SU(N) site (N = product Hilbert dim). Intra-unit coupling diagonalized exactly (reference = the unit ground state, e.g. a dimer SINGLET -- zero dipole, invisible to dipole/single-site SU(N) LSWT); excitations are the triplons. Generalized the SU(N) engine (per-site operators, generalized bond couplings, q-dependent staggered moment). Validated: isolated dimer flat triplon at omega=J; coupled-dimer chain omega=sqrt(J^2-JJ'cos) (bond-operator) to 7e-16; dimer structure factor (1-cos(q.d)) with the I(q=0)=0 selection rule. Optional Zeeman field (gamma*mu_B*H.sum_k S_k per unit) splits the multiplet. REAL MATERIALS: `examples/entangled/Cu5SbO6/` reproduces the J1-J2-J4 dimer expansion of Piyakulworawat et al., PRR 8, 013247 (2026) -- triplon band ~11-21 meV, dispersion = the bond-operator resummation of the paper's Eq. (A11), structure-factor selection rule Eq. (A14). `examples/entangled/Rb2Cu3SnF12/` -- the pinwheel-VBS single-dimer building block (Matan et al., Nat. Phys. 6, 865 (2010)): DM splits the Stot^z=0 from the +/-1 triplet, a c-axis field Zeeman-splits +/-1 (Fig. 4); the full deformed-kagome dispersion needs the 6-dimer geometry + high-order series expansion (strong coupling), beyond harmonic bond-operator. `magcalc/sun/entangled.py`, `examples/entangled/dimer_chain/`, `tests/test_entangled_units.py` |
| 12 | **mCIF / magnetic space groups** | ✅ | `from_mcif:` + CLI `magcalc mcif`; Sunny on TbSb: identical sites + directions. `magcalc/mcif.py`, `tests/test_mcif.py` |

### Tier 3 (plumbing / cheap wins)

| # | Item | Status | Notes |
|---|---|---|---|
| 13 | GS search sees q≠0 instabilities | ✅ | Luttinger-Tisza ordering-vector guard (`spiral_opt.ordering_wavevector` + a 3rd runner guard). Catches a q≠0 spiral GS the in-cell anneal/energy-audit provably cannot reach AND whose k=0 magnon spectrum comes back real-positive (blind to both older guards). Validated on the J1-J2 chain: LT k* = analytic `arccos(-J1/4J2)/2π` = 0.230053 to 1e-6; a FM supplied for it is now flagged with k* + the single_k/supercell fix. Zero false positives across all example configs. `tests/test_q_neq_0_instability.py` |
| 14 | Expose symmetry analyzer as CLI | ✅ | `magcalc symmetry <config> [--max-distance] [--json]` — space group, symmetry-inequivalent bond orbits, and the symmetry-ALLOWED exchange matrix per bond (Sunny `print_symmetry_table` analogue). New reusable `MagCalcConfigBuilder.from_config`. Validated: P4/mmm NN bond forced diagonal (analytic); Yb2Ti2O7 allowed form == the physical SpinW/Sunny matrix's zero/tie pattern. `tests/test_symmetry_cli.py` |
| 15 | Broken `aCVO/config.yaml` (+ `KFe3J/config.yaml`) | ✅ | both were legacy `python_model_file` configs superseded by `config_acvo.yaml` / `config_kfe3j.yaml`; **retired** (untracked + git-ignored, kept locally). Fixed 2 general runner bugs kept: clear error for missing `crystal_structure`; `hasattr(model,'minimize')` no longer matches imported scipy. `tests/test_config_robustness.py` |

---

## Delivered alongside, not on the original gap list

- **High-order dimer series expansion + Dlog-Padé** (`magcalc/sun/dimer_series.py`):
  the linked-cluster method of Matan et al. (Nat. Phys. 6, 865 (2010) / PRB 89,
  024414 (2014)) for STRONG-coupling dimer magnets, where the harmonic entangled
  engine fails. Numerical Bloch/des Cloizeaux PT per cluster, subcluster subtraction
  (cluster additivity asserted numerically), per-band eigenvalue series, Dlog-Padé
  with the approximant spread as the uncertainty. Config:
  `calculation: {mode: entangled, series_order: N}`. Validated: alternating-chain ED
  (5e-4 J at J'/J=0.4; <8% at the gap at J'/J=0.8, order 7), PRR Eq. (A11) exact at
  order 1, λ^(n+1) error scaling vs exact diagonalization. On the Rb2Cu3SnF12
  pinwheel (J2=0.95 J1, dz=0.18, bond families assigned by Cu-F-Cu angle from the
  CIF — 138.3°/123.3° matching the papers' 138°/124°): the Γ doublet lands at
  2.2±1.0 meV (order 4, Dlog-Padé) vs the measured 2.35 meV; the papers' order 8
  remains the converged answer. `tests/test_dimer_series.py`,
  `examples/entangled/Rb2Cu3SnF12/series_dispersion.py`.
- **Monte-Carlo / annealing ground-state minimizer** (`magcalc/annealing.py`): SpinW
  `anneal` (Metropolis + cooling, Sunny `LocalSampler` proposal mix) and `optmagsteep`
  (`method: steep`). On SW20-in-field, multistart L-BFGS hit the true minimum in only
  3/200 starts; annealing finds it in 1 run. Now the recommended `minimization.method`.
- **Two-part ground-state guard** (runner, `on_imaginary: error|warn|off`): an
  imaginary-mode check AND an energy audit (perturb-and-relax). The run FAILS on a
  non-minimum instead of drawing a plausible-but-wrong spectrum. Caught a real shipped
  bug (SW20 zero-field was not at its ground state). The energy audit is necessary
  because a stationary *maximum* (or a wrong SU(N) reference) returns a real, positive,
  plausible spectrum the imaginary check cannot see.
- **Studio (web + native) controls**: minimization method picker, Ground-State Check,
  LSWT engine (dipole/SU(N)/**entangled** incl. units + dimer series order),
  temperature, cross-section, 1/S corrections task, and the four beyond-LSWT tasks
  (**SCGA / thermal MC / SampledCorrelations / KPM**) with per-task settings — each
  produces a plot the apps display (runner-side `plot_scga`/`plot_thermal_mc`/
  `plot_sqw_grid`). The server passes ALL beyond-LSWT blocks through verbatim
  (`tests/test_gui_passthrough.py` pins this — they used to be silently dropped by a
  whitelist). Still config/CLI-only: the new interaction *types* (biquadratic/
  Stevens/3×3 SIA/dipole-dipole/g-tensor/multi-k) and the energy-cut/resolution
  editors (the blocks pass through if present; there is just no UI editor).

---

## Gap 4 — parity with Sunny 0.8.1 (audit 2026-08-03)

A sweep of Sunny 0.8.1's full export list against the engine. Everything below was
checked in code and, where a number was in question, measured. Implementation plan
and per-item oracles: `GAP4_PLAN.md`.

### Fixed by the audit itself (silent wrongness) — ✅ ALL DONE

| Item | Was | Now |
|---|---|---|
| Magnetic form-factor table | invented Q-dependence, up to +113% intensity error at 5 Å⁻¹ | generated from Sunny; f(Q) pinned per ion, `<j2>` branch added |
| Form-factor ion resolution | fell back to the SITE LABEL, so `Fe1`/`Cu2` silently became Fe¹⁺/Cu²⁺; `charge` was dropped by the Wyckoff expansion, so FeI₂ asked for NEUTRAL iron | `ion`, else element+charge, else neutral element, else none |
| `biquadratic` in SU(N) | silently dropped | exact via operator-pair couplings; matches Sunny `:SUN` |
| `biquadratic` in entangled, Ewald `dipole_dipole` in SU(N)/entangled | silently dropped | hard error naming the alternative |
| Anisotropy renormalization | undocumented mismatch with Sunny's default `:dipole` | `calculation.anisotropy_renormalization: rcs`, both branches pinned |
| "S(Q,ω) is 3/4 of Sunny's" | false caveat blocking absolute comparison | retired; absolute scale pinned |

### Phase 1 (quick wins) — ✅ ALL DONE

| # | Item | Status | Key | Validated against |
|---|---|---|---|---|
| 17 | Classical→quantum correction | ✅ | `sampled_correlations.classical_to_quantum` (default on) | Sunny's `c2q` formula to 1e-9; detailed balance \|c2q(ω)/c2q(−ω)\| = e^{ω/kT}; FM weight becomes q-independent (raw spread 213% → 29%) |
| 19 | Static / energy-integrated correlations | ✅ | `tasks.static_sqw` (LSWT) and `tasks.static_correlations` (classical) | Sunny `intensities_static` to 6e-9; free-spin sum rule 2S²/3 (perp) and S² (trace) EXACTLY, at every q and T |
| 23 | Domain averaging in SU(N)/entangled | ✅ | `calculation.domains` (all three engines now) | rotate the crystal explicitly: S_rot(q) = S(Rᵀq) to 3e-15; average = weighted sum of separately built twins |
| 27 | Crystal utilities + BZ paths | ✅ | `magcalc symmetry --cells / --species / --bz-path` | analytic bcc & rocksalt primitive cells (a³/2, a³/4); standardization idempotent; `seekpath` optional |

**Notes.** #17: the correction fixes the SHAPE of the classical S(q,ω); its absolute
normalization against LSWT is still unreconciled (measured ≈220 where LSWT gives S=1
on a low-T ferromagnet) — do not read absolute intensities off that path yet, and see
the open item below. #23: intensity is compared per DEGENERATE MULTIPLET, since inside
a degenerate subspace the split between individual bands is basis-dependent while the
multiplet sum is an observable. #27: `seekpath` is an optional dependency; without it
`--bz-path` raises an actionable ImportError and the rest still works.

### Phase 2 (parity for work you would publish)

| # | Item | Status | Key | Validated against |
|---|---|---|---|---|
| 25 | Blume–Maleev / arbitrary polarization | ✅ | `cross_section: {polarization, channel}` or `{bm: {u, v}, component}` | Sunny `ssf_custom_bm` (8 components, screw + cycloid) and `ssf_custom` NSF/SF/chiral at general P; P ∥ q reproduces `sf±` bit-for-bit; SF + NSF = perp |
| 21 | General pair couplings | ✅ | `interactions.pair_operator` (`poly` or `matrix`) | biquadratic via the general path == the dedicated path to 1e-10; Sunny `set_pair_coupling!` on c₁(S·S)+c₂(S·S)²+c₃(S·S)³, 3 coefficient sets |
| 24a | Mixed-spin SU(N) | ✅ | (automatic; per-site `spin_S`) | decoupled-sublattice identity for (½,1), (1,3/2), (½,3/2) — spectrum AND intensities are exactly the union of the two independent problems |
| 24b | Ewald + rotating-frame single-k | ⬜ | (refuses honestly) | **estimate revised 3 d → 1 w**: the rotating frame is baked into the SYMBOLIC Hamiltonian per bond (`R_i^T J_ij R_j`), while Ewald's A(q) is a numeric LAB-frame lattice sum. Each channel needs a projector-weighted combination of A(q_c), A(q_c±k) — a derivation, not plumbing. See GAP4_PLAN.md |

### Phase 3 (new machinery)

| # | Item | Status | Key | Validated against |
|---|---|---|---|---|
| 18 | Langevin / `ImplicitMidpoint` / `suggest_timestep` | ✅ | `classical_dynamics.langevin_step`, `evolve(..., integrator='midpoint')` | exact free-spin Langevin function (the oracle the Metropolis sampler is already pinned to); midpoint energy drift 1e-12 vs RK4's 8e-5 and \|S\| exact without renormalizing |
| 22 | Wang–Landau | ✅ | `tasks.wang_landau` + `wang_landau: {temperatures, supercell, n_bins, f_final}` | classical dimer's g(E) is EXACTLY FLAT (closed form); reconstructed ⟨E⟩(T) matches −JS²L(βJS²) to 0.01 across T; agrees with parallel tempering |
| 20 | Experiment-data binning | ⬜ | — | deferred pending demand — see below |

**#20 is deliberately not done.** `GAP4_PLAN` flagged it "check the value first": the
existing `fitting.load_fit_data` CSV path may already cover how data leaves the
reduction pipeline here, in which case NeXus import is work for nobody. It needs a
one-line answer from the group before it is worth 4 days.

### Phase 4 (large; gated on a real calculation needing them)

| # | Item | Status | Key | Validated against |
|---|---|---|---|---|
| 16a | Vacancies + open boundaries, **classical** | ✅ | `disorder: {vacancy_concentration, seed}` or `{vacancies: […]}`, `periodic: [b,b,b]` on any of `thermal_mc` / `sampled_correlations` / `static_correlations` / `wang_landau` | exact identities: x→0 is bit-identical to clean; a vacancy is exactly the restriction of H to the survivors; analytic bond counts (32 periodic vs 24 open on 4×4); self-averaging across seeds |
| 16b | Disorder in **LSWT** | ⬜ | — | needs a large disordered supercell + the existing KPM engine (Sunny example 09's recipe). ~1 week |
| 26 | Entangled classical dynamics | ⬜ | — | needs CP^(N−1) equations of motion — a different integrator, not a wrapper. ~1–2 weeks |

**Why 16a stops here.** GAP4_PLAN says "ship step 1 and stop if that answers the
question", and it plausibly does: dilution thermodynamics, open-boundary/finite-size
effects and diluted S(q,ω) are all reachable now. 16b buys LSWT *spectra* of a
disordered system, which is a different question and should wait until one is asked.

### Phase 4 — still open

Ordered by how much they cost. None is silently wrong — each either refuses or is
simply absent.

| # | Item | Phase | Sunny | Notes |
|---|---|---|---|---|
| 21 | General pair couplings | 2 | `set_pair_coupling!` | SU(N) covers bilinear + biquadratic; the operator-pair machinery is already there, only the front end (tensor SVD) is missing |
| 24a | Mixed-spin SU(N) | 2 | — | `sun/lswt.py` requires a uniform N; needs per-site block offsets |
| 24b | Ewald + rotating-frame single-k | 2 | — | each q ± k channel needs its own A(q) |
| 26 | Entangled classical dynamics | 4 | `EntangledSampledCorrelations` | needs CP^(N−1) equations of motion |
| — | Classical S(q,ω) absolute normalization | 3 | — | opened by #17: the classical path's overall scale has never been reconciled with the LSWT/Sunny one. Shape is pinned, scale is not |

Convention difference, not a gap: Sunny's `ssf_perp` applies the g-tensor by
default (4× at g = 2). pyMagCalc's S(Q,ω) is spin-only = `apply_g=false`.

---

## Sunny tutorial ports — audited 2026-08-04

`examples/sunny_tutorials/README.md` has the per-tutorial detail. Summary: of Sunny's
nine examples, **four are ported and pinned** (01, 03, 08 and the clean part of 09),
**one is ported but only transitively justified** (07 — the Ewald engine is pinned to
Sunny, that config's own bands are not), and **four are not ported**. Two of those
four (02, 05) are unblocked and simply not done; two (04, 06) plus the disordered half
of 09 are blocked on Gap 4 #26 and #16b.

The audit found the README was stale in BOTH directions: it called 02/04/05/06 "out of
scope" (Tier 2 has since implemented classical dynamics and thermal MC, so 02 and 05
are now portable), and it claimed S01's bands were "cross-checked against Sunny" when
nothing asserted it — S01 has no `magnetic_structure`, so the test helper could not
drive it and it was schema-checked only. It does match Sunny exactly, band for band,
and now says so in a test.

---

## How things were validated (and the recurring trap)

The single most important lesson from this work, stated for the next session:

**A check that a wrong answer passes is not a check.** The engine has repeatedly produced
plausible-but-wrong spectra that looked fine:
- the S-power filter silently deleting quartic terms;
- the classical energy optimizing a different Hamiltonian than LSWT diagonalized;
- a mixed-spin intensity off by 60% (a constant factor, easy to wave through);
- SU(N) intensities off by exactly ×L (per-site normalization);
- a **stationary maximum** returning a real, positive spectrum (invisible to the
  imaginary-mode check — only the energy audit catches it);
- a dipole-derived state used as an SU(N) reference (a good local minimum, so again
  invisible to the imaginary check);
- the SU(N) one-magnon amplitudes DOUBLE-counted intracell phases (the engine's H(q)
  is built in the full-position gauge, so the bosons already carry their positions)
  and normalized per SITE instead of per CELL REPLICA. Both were invisible because
  every intensity validation to that point had L == n_cells and r_i-phases that
  cancel (Gate 3: one site; FeI2: supercell of one atom) -- dispersions were exact
  while the S=1/2 AFM chain's zone-boundary intensity was ~60x too weak (the
  (u-v)^2 combination instead of (u+v)^2). Caught by the powder work; pinned by
  Gate 1b (`tests/test_powder_sun.py`) against Sunny to 2e-8;
- the powder average reported mode ENERGIES averaged over the sphere -- fine for
  near-flat bands (where it was calibrated), but it collapses a dispersive band to
  its center: Cu5SbO6's 10 meV triplon band became a ~1 meV blob at J1, in
  contradiction with the published powder spectrum (PRR 8, 013247 Fig. 5). Powder
  plots and fits now use SAMPLE-RESOLVED modes (each sphere direction keeps its own
  energies -- SpinW `powspec` convention); pinned analytically by the exact dimer
  interference factor 1 - sin(Qd)/(Qd) and by the paper's peaks
  (`tests/test_powder_binned.py`);
- GenericSpinModel.__init__ RESET `_ion_list = []` AFTER `_load_structure` had
  populated it, so `ion_list()` was empty for every config and the magnetic form
  factor was silently dropped from ALL intensities (dipole, SU(N), entangled).
  Invisible to every Sunny/SpinW cross-check -- both sides were computed
  form-factor-free -- and caught PHYSICALLY: the Cu5SbO6 powder map carried far
  too much weight at high |Q| vs PRR 8, 013247 Fig. 5. Pinned by
  `tests/test_form_factor.py` (ion_list survives construction; I_ion/I_bare ==
  f(Q)^2 exactly; the dimer powder modulation (1-sin(Qd)/Qd) f(Q)^2).
- a `ref_pair` symmetry rule without an explicit `offset` chose its reference bond
  by FLOATING-POINT NOISE. In CCSF's P2_1/n cell the two screw-related Cu2-Cu2
  (J12) images are the same length to ~9e-16 A -- one ULP -- and the resolver's
  `<` took whichever won that rounding. The reference bond fixes the orientation
  convention for the whole orbit (2_1 and the n glide both act on axial vectors as
  C2x = diag(1,-1,-1)), so the other choice realizes -C2x.D: `D12x` silently flips
  sign. Same 24 bonds, same J's, plausible spectrum, different Hamiltonian -- and
  the config's meaning was not even stable across BLAS/coordinate perturbations.
  Invisible to every self-consistent check; caught only by diffing the expanded
  bond table against an independently derived model. The resolver now enumerates
  all candidates and RAISES on a tie (directional rules) or on a `distance` window
  spanning two orbit lengths (all rules); pinned by
  `tests/test_ref_bond_ambiguity.py`.

- the magnetic form-factor COEFFICIENT TABLE was not the tabulated one. Every
  entry was normalized so that f(0) = 1 -- so a Q -> 0 check passed -- but the
  Q-dependence was invented: against Sunny (the standard P. J. Brown / Int-Tables
  <j0>) the intensity error reached +22% at |Q| = 2.5 A^-1, +53% at 3.8 and +113%
  at 5 for Mn2+, with Cu2+, Fe2+, Ni2+ and Co2+ comparably wrong. Straight onto
  the Cu5SbO6 powder work, which is Cu2+ out to 5 A^-1. Invisible because the only
  test compared I_ion/I_bare against `get_form_factor(...)**2` -- SELF-CONSISTENT
  BY CONSTRUCTION, the same wrong f(Q) on both sides. The table is now generated
  mechanically from Sunny (as `stevens.py` is) and `tests/test_form_factor.py`
  pins f(Q) ITSELF for 10 ions x 7 |Q| values, plus the <j2> (g != 2) branch.
- `interactions.biquadratic` and `dipole_dipole: {method: ewald}` were SILENTLY
  DROPPED in SU(N) and entangled mode: `from_generic_model` reads only
  (Jex, DM, Kex) + sia/sia_matrix/stevens, so the term never reached the
  Hamiltonian. Measured: adding a biquadratic B = -0.4 changed the SU(N) spectrum
  by exactly 0.0 (control: an SIA moved it 0.9 meV). The Ewald case even LOGGED
  "Ewald summation (no real-space bonds generated)" while nothing consumed it.
  Biquadratic is now expanded exactly in SU(N) via the engine's generalized
  operator-pair couplings -- (S_i.S_j)^2 = sum_ab (S_i^a S_i^b)(S_j^a S_j^b), so
  n_ops goes 3 -> 12 -- and matches Sunny `:SUN` band-for-band; the rest raise.
  `tests/test_sun_missing_terms.py`.
- **a "convention difference" that was a wrong reference number.** CLAUDE.md,
  this file and `test_polarized.py` all stated that pyMagCalc's absolute S(Q,w)
  is 3/4 of Sunny's, and told the reader not to compare absolute intensities.
  It is 1.0 -- verified band-by-band on a ferromagnet (S = 1/2, 1, 2), a Neel
  antiferromagnet in two orientations, and the non-collinear helix. The 4/3 was
  in hardcoded `SUNNY_PERP`/`SUNNY_CHIRAL` values, and the test compared only the
  RATIO chiral/perp, in which any overall factor cancels -- so the number was
  never checked, and the caveat then explained the leftover away permanently.
  This is the same rule as everywhere else on this list, applied to ourselves:
  **a clean constant factor is a bug until proven otherwise, and "it's a
  convention" is not a proof.** Now pinned absolutely by
  `tests/test_absolute_normalization.py`.

Every one was caught by an **independent oracle or an exact identity**, never by
inspection. So: validate against Sunny (in-repo) or a textbook analytic result; prefer
identities (decoupled-sublattice sum, S=1/2 SU(N)≡dipole, ferromagnet has zero
corrections) that fail loudly; and be suspicious of a discrepancy that is a *clean
constant factor*.

**A published reference number is not automatically a converged one.** Sunny's own FeI₂
example converges to a *local* minimum (E/site −2.35592338, one `minimize_energy!`); the
true ground state (−2.91893118) needs restarts. Days were spent "debugging" against a
non-converged reference. Re-minimize the reference before trusting it.

Key conventions worth knowing before touching the engine:
- pyMagCalc stores `HMat(q) = g · H2(q)` with H2 Hermitian, g = diag(I,−I); eigenvalues
  are the ±ω pairs. (The name `_build_TwogH2_matrix` is just a name.)
- Bonds are listed in BOTH directions with NO 1/2 on the hopping — this is how
  `H = (1/2) Σ_ordered` is encoded. The on-site (mean-field / local-field) term is the
  q=0 sum, NO phase (a phase there makes a ferromagnet's H(q) cancel to zero).
- The dipole ground state ≠ the SU(N) ground state when an anisotropy is present
  (a coherent state has ⟨Sz²⟩ ≠ (S n_z)²). Find the SU(N) GS in SU(N).
