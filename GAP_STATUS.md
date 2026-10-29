# pyMagCalc ↔ SpinW/Sunny feature-gap status

A running record of the gaps between pyMagCalc and SpinW/Sunny, what has been closed, and
what remains — written so a future session (AI or human) can pick up without re-deriving
the context. Keep it updated when a gap moves.

Reference oracle: **Sunny.jl 0.8.1** is checked out in-repo at
`../Sunny.jl-main`, and Julia 1.12 + Sunny 0.8.1 are installed. Use them (and textbook
analytic results) to validate every new feature — see "How things were validated" below.

Status, **2026-08-15**. Gap 1, Gap 2, Gap 3 (SU(N), Ewald, entangled units, KPM,
SCGA, thermal MC, 1/S corrections, mCIF) and **all of Gap 4** are closed, and all
nine Sunny tutorials are ported. Everything through 2026-08-13 is on `master`
(`f848853`); the 2026-08-15 follow-up work is in the working tree and **has not had a
full gate run** — see `OPEN_WORK.md`, "What has and has not been run". That session
closed the four follow-ups the 2026-08-13 work had opened (the classical time-domain
window, `thermal_mc.build_supercell`'s missing single-ion anisotropy, the CP^(N−1)
sampler's step size, the two dipolar prefactors) plus FeI₂'s `on_imaginary: warn` and
the shadow guard's installation. This file records what is *done and how it was
validated*; what is still open is deliberately not here — it is `OPEN_WORK.md`, the
"what to do next" companion.

Test suite: **1735 passed, 2 skipped** on the last full gate (`pytest -m ""` from the
workspace root, 2026-10-22, 14 min 34 s, idle machine). 986 tests collect from inside
`pyMagCalc/`, of which 807 are the fast default suite. The merge gate is
`pytest -m ""`; plain `pytest` is not. Every feature has a test pinned to an
**independent reference** (Sunny, or an exact analytic identity), never a
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
| 3 | Mixed-spin intensity prefactor | ✅ | (per-site √(S_i/2)) | decoupled-sublattice identity (was 60% error); the alternating-spin chain's closed form, energies AND per-band intensities to 1e-10, under `backend: numpy` and `fortran` (K6, 2026-10-27: fMagCalc carries the per-site factor, no NumPy fallback) |
| 3b | Mixed-spin **classical ground state** | ✅ | (per-site \|m_i\| = S_i) | closed-triangle law of cosines: AFM trimer S = (1,1,½) minimises at 151.0°/104.5°, E = −1.125 J, not the 120°/−1.5 J a uniform-length search returns |
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

Ran on 38 of the 47 example configs that shipped when this was measured; the rest
**refuse honestly** (incommensurate/spiral/auto-supercell → not supported; mixed-spin
→ since closed by Gap 4 #24a; frustrated GS search not converging → guard refuses).
Never silently wrong. (The figure has not been re-measured: the tree now ships **58**
runnable configs, 54 of them tracked — see the coverage audit below.)

### Tier 2 (capability parity) — ✅ ALL DONE (Wang–Landau closed by Gap 4 #22)

| # | Item | Status | Notes |
|---|---|---|---|
| 5 | **Finite-T classical dynamics (SampledCorrelations)** | ✅ | `tasks: {sampled_correlations: true}` + `sampled_correlations: {temperature, supercell, dt, n_steps, n_traj, therm_sweeps}`. Real-time Landau–Lifshitz dynamics (undamped RK4, dS/dt=−S×B) on Metropolis-thermalized states; S(q,ω)=⟨\|Σ_r e^{−iq·r}S_r(t)\|²⟩ by space-time FFT, whole q-path × energy grid in one shot. Validated: single-spin Larmor ω=gμ_B B; RK4 energy conservation (drift ~1e-8); the low-T ferromagnet S(q,ω) peaks fall on the EXACT LSWT magnon dispersion the engine computes (<5%, →0 as T→0). Extended since: Langevin thermalization + the symplectic implicit-midpoint integrator + `suggest_timestep` (Gap 4 #18), and the **absolute intensity scale**, which was two bugs and is now equal to LSWT's and Sunny's (Gap 4 #17b, 2026-08-13). `magcalc/classical_dynamics.py`, `tests/test_classical_dynamics.py`, `tests/test_classical_absolute_normalization.py` |
| 6 | **Thermal Monte-Carlo (parallel tempering)** | ✅ | `tasks: {thermal_mc: true}` + `thermal_mc: {temperatures, supercell, n_sweeps, n_equil}`. Finite-T thermodynamics on an explicit PBC supercell (built from `spin_interactions` + `_resolve_field`, same classical energy ½mᵀHm+bᵀm the minimizer uses); replica-exchange Metropolis over a T-ladder. Reports ⟨E⟩/N, C/N=Var(E)/(NkT²), magnetization, susceptibility. Validated: N free spins in a field = Langevin −L(βgμ_B·B·S) exactly; classical Heisenberg dimer ⟨E⟩(T)=−JS²L(βJS²) and C(T) from the exact partition function; parallel tempering ≡ independent single-T Metropolis. Wang–Landau closed separately by Gap 4 #22. **KNOWN DEFECT (2026-08-13, still open):** `build_supercell` assembles `H` from `spin_interactions` alone, so it carries **no single-ion anisotropy** — measured on S06, where `H_zz` came back as the bare exchange sum and D = 19 was simply absent, while the docstring calls `H` "the exchange/anisotropy Hessian". It feeds `thermal_mc`, `wang_landau`, `static_correlations` and the classical `sampled_correlations`. It is a *different* builder from the annealer's (`MagCalc._extract_classical_quadratic`, verified correct), so nothing above is affected. `OPEN_WORK.md` item 11. `magcalc/thermal_mc.py`, `tests/test_thermal_mc.py` |
| 7 | **Ewald dipole-dipole** | ✅ | `dipole_dipole: {method: ewald}`; Sunny to 1.3e-8; truncated→Ewald convergence (needs no Julia). `tests/test_ewald.py` |
| 8 | **LSWT 1/S corrections** | ✅ | `tasks.corrections`; Sunny + textbook square AFM: dE=−0.157947, dS→0.1966. `magcalc/corrections.py`, `tests/test_corrections.py` |
| 9 | **SCGA (paramagnetic diffuse scattering)** | ✅ | `tasks: {scga: true}` + `scga: {temperature, mesh_density, cross_section}`. Self-consistent Gaussian approximation: classical spins, hard length constraint softened to a global Lagrange multiplier λ, static S(q) = kT·pref†(λ+J(q))⁻¹pref with λ from the spin sum rule. SAME `fourier_exchange_matrix` J(q) as the LT guard; single-λ (one symmetry class — Bravais + kagome/pyrochlore-type). It refuses inequivalent sublattices on the PER-SUBLATTICE sum-rule residual, not on the symmetry labelling (guard added 2026-09-10; before that it silently used one λ — see the bug sweep below). Above T_N, so the LSWT ground-state guard is auto-skipped for a pure-SCGA run. Validated: exact classical-chain closed form (λ=√(4J²+(3kT/S²)²), S(q)=3kT/(λ+2Jcosq)) to 1e-9; **Sunny 0.8.1 SCGA** on square-lattice AND kagome AFM — λ and S(q) to 6 digits (matches `ssf_perp`, apply_g, (2/3)Tr at q→0); sum rule + high-T flat limit. `magcalc/scga.py`, `tests/test_scga.py` |
| 10 | **KPM (Chebyshev spectral S(q,ω))** | ✅ | `tasks: {kpm_sqw: true}` + `kpm: {e_min,e_max,e_step,fwhm,moments/tol}` (SU(N)/entangled modes). Para-unitary Chebyshev expansion of the one-magnon spectral function of the dynamical matrix D̂=gH₂ (=`SUNModel.hamiltonian`), following Lane et al. (arXiv:2312.08349) — Sunny's `SpinWaveTheoryKPM` method. Iterated matvecs only, no eigensolve: O(D·M) per q for large disordered/near-incommensurate cells. Validated against the engine's OWN exact diagonalization (`structure_factor`): as M grows KPM S(q,ω) → the exact Gaussian-broadened spectrum (rel L1 < 1e-3, integrated intensity to 1e-3); spectral bound γ verified to enclose the spectrum. **TWO BUGS FIXED 2026-08-13**, both invisible to the collinear square-lattice AFM that was the whole suite: (a) the Chebyshev recursion ran on D̂ where the structure factor needs conj(D̂) — identical whenever D̂ is real, but on a clean NON-COLLINEAR 81-site supercell it put ~5 % of the intensity on bands carrying none, at low energy, so a clean spectrum arrived pre-broadened; (b) the moments come from the annihilation block of the Nambu basis while `structure_factor` uses the creation block, whose weight matrices are complex conjugates — no effect on the symmetric channels, but `cross_section: chiral` came back SIGN-REVERSED on a plain ferromagnet. Now pinned on a non-collinear supercell and in the chiral channel. **KPM never diagonalizes, so it grows no ground-state guard of its own** — no Cholesky to fail, nothing for `on_imaginary` to catch — and returns a smooth, plausible spectrum about a saddle or a maximum. **Guarded 2026-08-13** (`SUNModel.is_stable_at` / `assert_stable` / `min_h2_eigenvalue`, run by the `kpm_sqw` task and by any script, see the Tier-3 #16 row). One more silent failure fixed with it: at a q where H(q) is IDENTICALLY zero (a ferromagnet at Γ, i.e. in every path) the spectral bound γ was 0 and `D̂/γ` returned a whole **NaN column**, into the saved map, the plot and the logged intensity range; it is now finite and warns that the value there is 0 rather than the q→0 limit. `magcalc/sun/kpm.py`, `tests/test_kpm.py`, `tests/test_kpm_stability.py`, `tests/test_s09_disorder_kpm.py` |
| 11 | **Entangled units** | ✅ | `calculation.mode: entangled` + `units:` -- a cluster (dimer/trimer/...) becomes ONE effective SU(N) site (N = product Hilbert dim). Intra-unit coupling diagonalized exactly (reference = the unit ground state, e.g. a dimer SINGLET -- zero dipole, invisible to dipole/single-site SU(N) LSWT); excitations are the triplons. Generalized the SU(N) engine (per-site operators, generalized bond couplings, q-dependent staggered moment). Validated: isolated dimer flat triplon at omega=J; coupled-dimer chain omega=sqrt(J^2-JJ'cos) (bond-operator) to 7e-16; dimer structure factor (1-cos(q.d)) with the I(q=0)=0 selection rule. Optional Zeeman field (gamma*mu_B*H.sum_k S_k per unit) splits the multiplet. REAL MATERIALS: `examples/entangled/Cu5SbO6/` reproduces the J1-J2-J4 dimer expansion of Piyakulworawat et al., PRR 8, 013247 (2026) -- triplon band ~11-21 meV, dispersion = the bond-operator resummation of the paper's Eq. (A11), structure-factor selection rule Eq. (A14). `examples/entangled/Rb2Cu3SnF12/` -- the pinwheel-VBS single-dimer building block (Matan et al., Nat. Phys. 6, 865 (2010)): DM splits the Stot^z=0 from the +/-1 triplet, a c-axis field Zeeman-splits +/-1 (Fig. 4); the full deformed-kagome dispersion needs the 6-dimer geometry + high-order series expansion (strong coupling), beyond harmonic bond-operator. `magcalc/sun/entangled.py`, `examples/entangled/dimer_chain/`, `tests/test_entangled_units.py` |
| 12 | **mCIF / magnetic space groups** | ✅ | `from_mcif:` + CLI `magcalc mcif`; Sunny on TbSb: identical sites + directions. `magcalc/mcif.py`, `tests/test_mcif.py` |

### Tier 3 (plumbing / cheap wins)

| # | Item | Status | Notes |
|---|---|---|---|
| 13 | GS search sees q≠0 instabilities | ✅ | Luttinger-Tisza ordering-vector guard (`spiral_opt.ordering_wavevector` + a 3rd runner guard). Catches a q≠0 spiral GS the in-cell anneal/energy-audit provably cannot reach AND whose k=0 magnon spectrum comes back real-positive (blind to both older guards). Validated on the J1-J2 chain: LT k* = analytic `arccos(-J1/4J2)/2π` = 0.230053 to 1e-6; a FM supplied for it is now flagged with k* + the single_k/supercell fix. Zero false positives across all example configs. `tests/test_q_neq_0_instability.py` |
| 14 | Expose symmetry analyzer as CLI | ✅ | `magcalc symmetry <config> [--max-distance] [--json]` — space group, symmetry-inequivalent bond orbits, and the symmetry-ALLOWED exchange matrix per bond (Sunny `print_symmetry_table` analogue). New reusable `MagCalcConfigBuilder.from_config`. Validated: P4/mmm NN bond forced diagonal (analytic); Yb2Ti2O7 allowed form == the physical SpinW/Sunny matrix's zero/tie pattern. `tests/test_symmetry_cli.py` |
| 15 | Broken `aCVO/config.yaml` (+ `KFe3J/config.yaml`) | ✅ | both were legacy `python_model_file` configs superseded by `config_acvo.yaml` / `config_kfe3j.yaml`; **retired** (untracked + git-ignored, kept locally). Fixed 2 general runner bugs kept: clear error for missing `crystal_structure`; `hasattr(model,'minimize')` no longer matches imported scipy. `tests/test_config_robustness.py` |
| 16 | Ground-state guard for KPM (`H₂(q) ⪰ 0`) | ✅ | **2026-08-13.** KPM was the one spectrum path with no ground-state guard, structurally: it never diagonalizes, so there is no Cholesky to fail and no imaginary energy to report, and it returns a smooth plausible S(q,ω) about a saddle. Now `SUNModel.is_stable_at(q)` / `assert_stable(qs)` / `min_h2_eigenvalue(q)`, called by the runner's `kpm_sqw` task at **every q it computes** and by `disorder_kpm.py` — one shared implementation, one criterion. Two decisions worth keeping: (a) the test is a SHIFTED CHOLESKY (`H₂+εI ≻ 0`, ε = `calculation.h2_rel_tolerance`·‖H₂‖, default 1e-6), exact and 45× cheaper than `eigvalsh` at 2D=1800 (65× at 3200) — 1.3–4.9 % of the KPM cost at that q (2D = 288 → 1800) with the `hmat=` build shared, which is what makes checking every q affordable, and the instability IS q-specific (4 generic q find it on 1 S09 realization in 3, a 40-point path on 2 of 3); (b) the shift and its **q-INDEPENDENT** scale are not fudges — a Goldstone mode puts an exact zero in H₂ at the ordering wavevector, and a ferromagnet's H₂ is identically zero at Γ, so a purely relative per-q threshold is zero exactly where every path starts. Threshold measured, not chosen: on S09's 144-site cell over a 37-point path, a correctly relaxed state floors at min eig H₂ = −2e-10 (rel 5e-11) while σ=1/3 gives −3e-3 (rel 5e-4) — seven orders apart. Validated in `tests/test_kpm_stability.py` against (i) the closed form min eig H₂ = −S·J·(z−Σcos q·δ) for the ferromagnetic state of an antiferromagnet, which is minus the engine's own validated FM dispersion; (ii) `eigvalsh` on the same matrices, verdict by verdict; (iii) the exact classical boundary J₂ = \|J₁\|/4 of the frustrated FM chain. That chain is also the control that says the guard is not redundant: it is a genuine in-cell minimum with \|Im ω\| = 0 exactly, so guards 1 and 2 pass it and a `dispersion` run plots a plausible band. `kpm_sqw` also joined `_lswt_tasks`, so pairing it with a classical task no longer silences the up-front guards. `magcalc/sun/lswt.py`, `magcalc/runner.py` |

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
  2.2±1.0 meV (order 4, Dlog-Padé) vs the measured 2.35 meV, and the series has been
  run to the papers' order 8 (the order-by-order table is in
  `examples/entangled/Rb2Cu3SnF12/README.md`). `tests/test_dimer_series.py`,
  `examples/entangled/Rb2Cu3SnF12/series_dispersion.py`. Under `kernels: rust` each
  cluster's PT runs in rsMagCalc ("Rust kernels K7" below).
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
- **Open a config, press Run == `magcalc run <that file>`** (2026-08-12). The server
  half was pinned; the EDITOR half — what you hold after you OPEN a file — was not,
  and it rebuilt the config from a whitelist of blocks it recognised. Seven ways a
  config that ran from the CLI ran differently, or not at all, in the apps:
  `atom_mode: explicit` (which the apps add and the CLI omits) emptied an `atoms_uc`
  cell AND switched off distance-rule expansion, so `symmetry_rules` produced zero
  bonds; a `magnetic_structure` without an explicit `enabled: true` was DELETED
  (the runner defaults that key to True, the editor to False — ZnCVO's bands moved
  3.54/3.59/11.00 → 0.62/10.94/11.10/21.42 meV and the run still exited 0);
  `tasks: {fit/energy_cut/static_correlations/…}` and `from_mcif`/`units`/
  `experiment` were dropped; `output`/`fitting` were replaced by UI placeholders;
  the anneal-only `minimization.n_sweeps` was injected into gradient-method configs
  (`minimize() got an unexpected keyword argument`); parameters were rounded to 5
  decimals, degrading every fitted value on open. The rule now is **the file is the
  base**: the whole document is kept and only genuinely-edited keys are written over
  it (`gui/src/lib/configIO.js`, mirrored in `MagCalcConfig.backendInput`). Pinned by
  `tests/test_gui_roundtrip.py` (every config under `examples/` through the app's
  open→run transform — 62 in this tree, a broader glob than the smoke test's 58 —
  plus four run end-to-end against their own CLI run, band for band) and
  `tests/test_atom_mode_explicit.py`. The native app's Run additionally posted the
  pre-`config-as-source` `{"data": …}` envelope, which the server had rejected with
  400 since a67f7cf, and fetched plots through the browser's `/api` dev-proxy prefix.
  Relative references (`from_mcif`, `fitting.data_file`, `cif_file`) now resolve
  because the run happens in the opened file's own directory — clients send
  `config_dir`. The **web app can now send it too** (2026-08-13): its Open goes
  through the server (`/load-config`, which returns an abspath) behind a
  recent-files + browse picker, since the File System Access API exposes only
  `handle.name` and no browser dialog can supply a path. Save on such a file goes
  back through `/save-config`, and the browser-side Open routes remain as the
  fallback with `config_dir` cleared (previous project-root behaviour, still
  loud). Pinned by `tests/test_gui_relative_paths.py`, whose oracle is the CLI —
  the shipped `examples/materials/mcif` config, opened and run through the server,
  reproduces `magcalc run` band for band, and the same payload aimed at a
  directory without the mCIF fails.
- **The two emitters are now checked against each other** (2026-08-13). The web
  app's `configIO.js` had `tests/test_gui_roundtrip.py`; the Swift
  `MagCalcConfig.backendInput` had nothing, and disagreed with it on **all 58**
  shipped configs. Root cause of most of it: Swift's `mergeEdits` treated "the
  file has no such block" as "emit the app's whole struct", so opening a config
  with no `minimization:` added `method: anneal, n_sweeps: 2000, …` to the run and
  one with no `fitting:` gained a placeholder fit — the injected-default class
  again. Three more, each a real loss: `fitting`'s `data_file`/`vary`/`bounds`/
  `scale`/`background` and `minimization.n_sweeps` had NO import branch but were
  re-emitted from the struct, so opening a fitting config and pressing Fit wrote
  the app's blanks over the real ones; the crystal block was re-emitted from the
  file verbatim, so every Structure-panel edit was discarded; and per-site keys the
  app does not model (`g`, `charge`, `wyckoff`, `species` — `g` is the Zeeman
  term) had nowhere to live. A third target, `magcalc-emit-config`, compiles the
  same `Sources/Models` headless, and `tests/test_native_emitter_parity.py` diffs
  it against `node gui/tests/emit_run_config.mjs` config by config. Parity is the
  oracle because the JS side is itself pinned to `magcalc run`, so it chains back
  to the CLI; the three specific losses are pinned directly against the file as
  well, since parity alone cannot catch both sides being wrong together.
  The rule extends to `parameters`: the editor always carries `H_mag`/`H_dir`, so
  an opened file that declares no field used to gain `H_mag: 0, H_dir: [0, 0, 1]`
  — a zero-magnitude Zeeman term the CLI never adds. It is numerically nothing and
  still moved CoRh2O4's bands by 6e-14 and S07's by 3e-15; with it dropped, every
  measured config agrees with its CLI run to **exactly 0.0**. (Both were first
  written off as JSON int/float round-off — `2.0` does come back as `2` through any
  browser client — but bisecting one config key at a time named the real cause.
  CLI-vs-CLI is bit-identical on both, so the discrepancy was never run-to-run
  noise.)
- **`examples/fitting/fit_dispersion.yaml` was fitting a doubly-wrong model**
  (2026-08-12). TUTORIAL.md's `magcalc fit` example listed each exchange bond in
  ONE direction only — the engine sums `H = ½ Σ` over *ordered* pairs, so every J
  was half its stated value (measured: ω(h=½) = 2.000 against the analytic 4.000)
  — and declared no `magnetic_structure`, so with J1, J2 > 0 (AFM) it expanded
  about the all-parallel state, a stationary **maximum**. Its own check ("recovers
  J1 = 1.0, J2 = 0.3") passed at 1.0002/0.2967 because `disp_data.txt` had been
  generated *by the engine from that same broken model* — self-consistent garbage,
  the exact anti-pattern the closing section of this file is about. Now: both bond
  directions, Néel via `magnetic_supercell: auto`, and the data regenerated from
  the ANALYTIC branch `ω = 2S√((J1+J2)² − (J1cos2πh + J2cos2πk)²)` by a committed
  generator, so the fit tests the engine against something external. Recovers
  J1 = 0.9988 ± 0.011, J2 = 0.3035 ± 0.005, reduced χ² = 0.52.
  `tests/test_fit_example.py` pins the engine to that analytic branch AND asserts
  the shipped data is analytic, so the circularity cannot return.
- **`show_plot: true` wedged unattended runs** (2026-08-12). `plt.show()` on an
  interactive backend blocks until a human closes the window; seven shipped
  configs set the flag, so `magcalc run` on any of them from a script or cron hung
  at ~0% CPU with no output, no error and nothing for CI to time out on (it cost
  ~30 min of a config sweep to spot, since the process looks alive). All ten
  `plt.show()` sites now go through `plotting.show_plot_if_possible()`, which
  no-ops on a non-interactive backend; the smoke test additionally pins
  `matplotlib.use("Agg")` rather than relying on the environment's accident.
- **…and the backend check alone was not enough** (2026-08-18). Windows still
  appeared during gate runs, intermittently. Cause: **`import pyalps` RESETS
  matplotlib's backend from Agg to the interactive `macosx`**, beating both
  `$MPLBACKEND` and an earlier `matplotlib.use("Agg", force=True)`. magpipe's
  `test_alps.py`/`test_thermo.py` import it, so under `-n auto --dist worksteal`
  only the workers that drew those modules were poisoned — and only the config
  runs landing in those workers afterwards opened a window, which is why a clean
  gate proved nothing. `show_plot_if_possible` now also honours **`MAGCALC_NO_GUI`
  and `PYTEST_CURRENT_TEST`, independently of the backend**: a test session is a
  TREE of processes and the `show_plot` configs execute in `magcalc run`
  children, which inherit an environment but not a backend. A `conftest.py` in
  each test root sets both and restores the backend before every test
  (`MAGCALC_TEST_GUI=1` opts out); `tests/test_no_gui_guard.py` pins it, and
  checks pyalps's behaviour in a SUBPROCESS because in-process a second
  `import pyalps` is a no-op that re-flips nothing.
- **The GUI spawned calculations that could hang the same way** (2026-08-18).
  `gui/server.py` forces Agg for ITSELF, but launched its calculation subprocess
  with no `env=`, so a `show_plot` config blocked forever on a native window —
  from the UI, a run that starts and then hangs with `/stop-calculation` the only
  way out. It now passes `MPLBACKEND=Agg` and `MAGCALC_NO_GUI=1` to the child
  (`tests/test_gui_output_dir.py`). The native app launches this same server, so
  one fix covers both clients.

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
| 19 | Static / energy-integrated correlations | ✅ | `tasks.static_sqw` (LSWT — the RUNNER branch raised NameError and did nothing until 2026-09-10; see the bug sweep) and `tasks.static_correlations` (classical) | Sunny `intensities_static` to 6e-9; free-spin sum rule 2S²/3 (perp) and S² (trace) EXACTLY, at every q and T |
| 23 | Domain averaging in SU(N)/entangled | ✅ | `calculation.domains` (all three engines now) | rotate the crystal explicitly: S_rot(q) = S(Rᵀq) to 3e-15; average = weighted sum of separately built twins |
| 27 | Crystal utilities + BZ paths | ✅ | `magcalc symmetry --cells / --species / --bz-path` | analytic bcc & rocksalt primitive cells (a³/2, a³/4); standardization idempotent; `seekpath` optional |
| 17b | Classical S(q,ω) **absolute** normalization | ✅ 2026-08-13 | (automatic; `classical_dynamics`, `thermal_mc`, `sun/dynamics`) | the equal-time sum rule ∫dω S = ⟨S(q)\*S(q)⟩/n_cells against the SAME trajectory, to 1e-14, on a one-site AND a two-site cell and in the dipole AND CP^(N−1) paths; the free-spin per-cell identity n_atoms·2S²/3; grid independence in dt; and the LSWT band sum (itself pinned to Sunny) on a gapped low-T ferromagnet, to 1.5 % |

**Notes.** #17: the correction fixes the SHAPE of the classical S(q,ω); its absolute
normalization was reconciled separately on **2026-08-13** — see #17b, which this note
used to defer to. #17b: the "≈220 where LSWT gives S = 1" recorded here was two
ordinary bugs, not a convention — **(a)** the time FFT was never normalized, so the
result was 2π/dt too large (314× at `dt: 0.02`) *and proportional to 1/dt*, and
**(b)** the spatial sum was divided by the SITE count where LSWT and Sunny divide by
the CELL count, so a multi-atom cell was a further `n_atoms` too small. (b) is the
instructive one: every classical model in `tests/` had one site per cell, where the
two divisors are identical, so no existing test could see it — and the shape tests
that did exist all compare ratios, in which any overall factor cancels. The remaining
~16 % when the *whole* frequency axis is integrated is spectral leakage, not scale:
no time-domain window is applied (Sunny uses a cosine one) and `c2q` weights the
sidelobes linearly in ω out to π/dt. #23: intensity is compared per DEGENERATE MULTIPLET, since inside
a degenerate subspace the split between individual bands is basis-dependent while the
multiplet sum is an observable. #27: `seekpath` is an optional dependency; without it
`--bz-path` raises an actionable ImportError and the rest still works.

### Phase 2 (parity for work you would publish)

| # | Item | Status | Key | Validated against |
|---|---|---|---|---|
| 25 | Blume–Maleev / arbitrary polarization | ✅ | `cross_section: {polarization, channel}` or `{bm: {u, v}, component}` | Sunny `ssf_custom_bm` (8 components, screw + cycloid) and `ssf_custom` NSF/SF/chiral at general P; P ∥ q reproduces `sf±` bit-for-bit; SF + NSF = perp |
| 21 | General pair couplings | ✅ | `interactions.pair_operator` (`poly` or `matrix`) | biquadratic via the general path == the dedicated path to 1e-10; Sunny `set_pair_coupling!` on c₁(S·S)+c₂(S·S)²+c₃(S·S)³, 3 coefficient sets |
| 24a | Mixed-spin SU(N) | ✅ | (automatic; per-site `spin_S`) | decoupled-sublattice identity for (½,1), (1,3/2), (½,3/2) — spectrum AND intensities are exactly the union of the two independent problems |
| 24b | Ewald + rotating-frame single-k | ✅ 2026-08-13 | `dipole_dipole: {method: ewald}` + `magnetic_structure: {type: single_k}` | `tests/test_ewald_spiral.py`: (a) exact identity against the explicit `magnetic_supercell` at commensurate k, both `k_case` branches, established only after the same identity passes with the dipolar term OFF; (b) Sunny 0.8.1 `SpinWaveTheorySpiral` + `enable_dipole_dipole!` at incommensurate k, 1.3e-8. The projector combination is the **mirror** of Sunny's (R1 with q+k), because `ewald.py` phases `A(q)` over the FULL bond vector where Sunny uses lattice translations only; the `k_case 2` cross terms carry a per-row `exp(±2i k·r_i)`. `k_case 3` drops the ±2k umklapp unless `A(q)` is uniaxial about the axis — same approximation as Sunny, and `_check_ewald_spiral_validity` warns |

### Phase 3 (new machinery)

| # | Item | Status | Key | Validated against |
|---|---|---|---|---|
| 18 | Langevin / `ImplicitMidpoint` / `suggest_timestep` | ✅ | `classical_dynamics.langevin_step`, `evolve(..., integrator='midpoint')` | exact free-spin Langevin function (the oracle the Metropolis sampler is already pinned to); midpoint energy drift 1e-12 vs RK4's 8e-5 and \|S\| exact without renormalizing |
| 22 | Wang–Landau | ✅ | `tasks.wang_landau` + `wang_landau: {temperatures, supercell, n_bins, f_final}` | classical dimer's g(E) is EXACTLY FLAT (closed form); reconstructed ⟨E⟩(T) matches −JS²L(βJS²) to 0.01 across T; agrees with parallel tempering |
| 20 | Experiment-data binning | ✅ | `magcalc/binning.py`: `BinningParameters`, `bin_mode_list`, `rebin`, `load_nxs`; `fitting.load_fit_data` dispatches on `.nxs`/`.nxspe`/`.h5` | exact identities, no oracle needed: weight conserved to 1e-12; a delta lands in the bin containing it; `rebin(fine) == bin(coarse)` exactly. `load_nxs` pinned by a round trip (reader ⟷ writer) — which is what that proves and all it proves |

**#20 note.** The binning half needs no extra dependency and is pinned by exact
identities. `load_nxs` needs `h5py` (present here) and is pinned only by a ROUND TRIP
through a file this module writes: that proves reader and writer agree, not that any
given instrument's NeXus dialect is understood — there are many. `nxs_report` lists a
file's datasets for when the reader cannot find a histogram. Point it at a real file
from the reduction pipeline before trusting it.

### Phase 4 (large; gated on a real calculation needing them)

| # | Item | Status | Key | Validated against |
|---|---|---|---|---|
| 16a | Vacancies + open boundaries, **classical** | ✅ | `disorder: {vacancy_concentration, seed}` or `{vacancies: […]}`, `periodic: [b,b,b]` on any of `thermal_mc` / `sampled_correlations` / `static_correlations` / `wang_landau` | exact identities: x→0 is bit-identical to clean; a vacancy is exactly the restriction of H to the survivors; analytic bond counts (32 periodic vs 24 open on 4×4); self-averaging across seeds |
| 16b | Bond disorder in **LSWT** | ✅ | `sun.lswt.apply_bond_disorder(model, sigma, seed)` on a supercell, then `sun/kpm.py` | σ=0 bit-identical to clean; Hermiticity preserved to 9e-16 (both bond directions get ONE draw); band spread grows monotonically with σ; self-averaging across seeds. `tests/test_bond_disorder.py` |
| 26 | SU(N) / entangled classical dynamics | ✅ | `magcalc/sun/dynamics.py` | Conservative flow: N=2 reproduces Landau–Lifshitz to **4.8e-10**, energy and \|Z_i\| conserved to 1e-8 / 1e-12, low-T S(q,ω) within **1.1%** of the SU(N) LSWT band and hardening monotonically on cooling. Dissipative quench (`damped_step`, `quench`): `dE/dt = −2λ·Var(h)` verified to **5e-6**, sign derived analytically. Plus Berg–Luescher `topological_charge` |

**Why 16a stops here.** GAP4_PLAN says "ship step 1 and stop if that answers the
question", and it plausibly does: dilution thermodynamics, open-boundary/finite-size
effects and diluted S(q,ω) are all reachable now. 16b buys LSWT *spectra* of a
disordered system, which is a different question and should wait until one is asked.

### Still open, updated 2026-08-13

**Every numbered Gap 4 item is closed** — #24b and the classical-normalization item,
the last two, landed 2026-08-13 — **and so is every follow-up those closures opened**,
the last of them (the classical window's `subtract_elastic` pairing) on 2026-08-16.
Follow-up work lives in `OPEN_WORK.md` with its injection point and its oracle; the
table here is only the index, and it is currently all strikethrough.

| Item | Where | Why it is open |
|---|---|---|
| ~~Classical S(q,ω) absolute normalization~~ | — | ✅ **CLOSED 2026-08-13** — Phase 1 row #17b |
| ~~Ewald + rotating-frame single-k (#24b)~~ | — | ✅ **CLOSED 2026-08-13** — Phase 2 row #24b |
| ~~No time-domain window on the classical S(q,ω)~~ | — | ✅ **CLOSED 2026-08-15** — `window: cosine` added (Sunny's lag window), OPT-IN. Pinned by the exact identity cos²(x) = ½ + ¼e^{2ix} + ¼e^{−2ix}, i.e. windowing ≡ convolving with [¼, ½, ¼], to 1.1e-16. It is not the default here because that same one-bin smear hits the elastic delta, which `c2q` then multiplies by ω/kT = 31 — measured 9.10 in one bin against a whole band sum of 0.5. Pair it with the new `subtract_elastic`. `OPEN_WORK.md` items 12 and 15 |
| ~~`thermal_mc.build_supercell` carries no single-ion anisotropy~~ | — | ✅ **CLOSED 2026-08-15** — `thermal_mc.onsite_quadratic` reads the terms from the model's own `_compute_sia_terms` and extracts (H, b) by exact probing; rank-k ≥ 4 Stevens now RAISES rather than vanishing. Pinned to the closed-form single-spin partition function; 5 of 6 new tests confirmed failing before. No shipped config was affected. `OPEN_WORK.md` item 11 |
| ~~The CP^(N−1) sampler does not equilibrate at low kT~~ | — | ✅ **CLOSED 2026-08-15** — the Metropolis step size is tuned to a target acceptance over the first half of the thermalization and held fixed for the measured half, and the residual energy drift is reported (`on_unequilibrated`). Pinned against the sampler's OWN partition function in closed form (Fubini–Study ⇒ \|z\|² uniform on the simplex). The fixed-step sampler was 32 % wrong at kT = 1 with 99.5 % acceptance. `OPEN_WORK.md` item 13 |
| ~~The two dipolar prefactors disagree at 1.2e-5~~ | — | ✅ **CLOSED 2026-08-15** — `DIPOLE_PREFACTOR_MEV_A3` is now DERIVED as `MU0_MUB2_MEV_A3/(4π)` from Sunny's full-precision constant, instead of an independently typed 0.05368216. Taken as a deliberate accuracy change and the Ewald oracle re-run (21 passed), not a widened tolerance. `OPEN_WORK.md` item 14 |
| ~~Config-surface coverage~~ | — | ✅ **CLOSED 2026-08-15**, all three follow-ups. Keys are enumerated from the CODE (`tests/config_keys.py`, 194 keys / 21 blocks — it immediately found `calculation.h2_rel_tolerance`, documented and read by the runner but named by no test, the `imaginary_rel_tolerance` shape repeating); discovery is by CONTENT rather than a glob plus a hand-list; and the smoke test now FAILS on any warning outside a reasoned allow-list. The escalation's first run found 7 shipped configs on the deprecated `type: spiral`, of which only one ever warned (the deprecation fires once per process) — all migrated, pinned by an exact identity and by an order-independent grep. `OPEN_WORK.md` item 5 |
| ~~`window: cosine` is a trap without `subtract_elastic`~~ | — | ✅ **CLOSED 2026-08-16** — `classical_dynamics.check_elastic_leakage`, on both samplers and both config blocks (`on_elastic_leakage: warn\|error\|off`). It REPORTS rather than deciding — implying `subtract_elastic` would silently change what the config asked for — and it triggers on a computed number, the amplification c2q(Δω) at the first bin, which it names; quiet unless it reaches 2 (kT ≲ Δω/1.6), since below that the window costs only the one bin of Hann broadening it advertises. Oracle: item 12's four rows, with the factor checked against `classical_to_quantum_factor` itself. `OPEN_WORK.md` item 15 |

Two of the Gap 4 closures (#16b, #26) delivered *capability* that the corresponding
Sunny tutorials did not exercise for months; **both are now exercised** (S06 and S09,
both 2026-08-13), and each port found something the capability's own tests had missed
— a 3-orders-of-magnitude performance wall in the CP^(N−1) derivative for #26, and two
bugs in `sun/kpm.py` for #16b. The tutorial table below tracks port status separately
and deliberately does not inherit the engine's status: that separation is what made
both gaps visible.

Convention difference, not a gap: Sunny's `ssf_perp` applies the g-tensor by
default (4× at g = 2). pyMagCalc's S(Q,ω) is spin-only = `apply_g=false`.

---

## Sunny tutorial ports — 9 of 9, updated 2026-08-13

`examples/sunny_tutorials/README.md` has the per-tutorial detail and is the
authoritative row-by-row record.

| ported & pinned | ported, not pinned | not ported |
|---|---|---|
| 01, 02, 03, 04, 05, **06**, 08, **09** | 07 (Ewald *engine* pinned to Sunny at 1.3e-8; this config's own spectrum never compared) | — |

**All nine are ported as of 2026-08-13**, 06 and 09 being the last two. The one
remaining caveat is 07, which is only *transitively* justified: the engine it
exercises is pinned, the pyrochlore's own bands are not. Sunny's example works in
kelvin (J₁ = 0.304 K) and reshapes to the primitive cell before minimizing, so a real
comparison has to reconcile units and land on the same dipolar ground state. Treat
S07 as a worked example, not as a validated result.

**06 is PORTED as of 2026-08-13**, at Sunny's own L = 40 and as a real quench.
Pinned to Sunny 0.8.1 at **5.4e-13** — the energy of an ARBITRARY coherent-state
configuration, which exercises both exchange shells, the Delta anisotropy, the
easy-plane single-ion term and the Zeeman sign in one number. A ground-state energy
would have been a much weaker check, because the state relaxes to fit whatever
Hamiltonian it is handed.

Of the two candidate causes recorded here on 2026-08-04, **the bond shell was
innocent and the system size was the answer** — via performance, by three orders of
magnitude. `SUNModel.local_field(i, .)` scanned the whole bond list per site, so the
CP^(N-1) derivative cost O(sites^2): ~16 s/step at 1600 sites, ~55 h for the
tutorial's 25600 steps. Summing the bond list once and forming only h_i|Z_i> (never
the matrices h_i) gives **8.4 ms/step, the whole run in 214 s**. The quench then
leaves an exactly quantized, non-zero SU(3) charge (+12 → −4 → −6 at tau = 4, 16,
256), reproducing the tutorial's figure.

Two things the port exposed, both of the recurring shape (see "How things were
validated" below):

- the entire dissipative-quench API — `damped_deriv`, `quench`,
  `topological_charge`, `triangulate_lattice` — **had no test at all**; the recorded
  "validated by dE/dt = -2 lambda Var(h) to 5e-6" was a one-off terminal
  measurement. `tests/test_sun_quench.py` is now that oracle, written before the
  derivative was rewritten;
- `topological_charge` was **undefined on this very model and did not say so**: it
  normalizes every spin, and most of the area here is the quadrupolar |m=0> state
  with <S> ~ 0, so it returned quantized-looking noise. It refuses now, and
  `sun_topological_charge` (the CP^(N-1) Berry phase Sunny actually plots) is pinned
  to it by the exact N = 2 identity.

- **09 is DONE (2026-08-13)** — `config_supercell.yaml` (the 120-degree order as an
  explicit sqrt3 x sqrt3 SU(N) cell, E/site = -0.375 exactly) plus `disorder_kpm.py`.
  **The diagnosis recorded here was wrong, and that is the part worth keeping.** It
  read "disorder NARROWED the KPM width instead of broadening it, which is what
  expanding about a non-minimum buys", blaming the ferromagnetic placeholder. The
  placeholder *was* wrong (E/site = +0.75 where the ground state is -0.375) — and
  replacing it with the exact 120-degree state did NOT fix the narrowing. The
  narrowing was a **bug in `sun/kpm.py`** (see the KPM row of the table above): the
  clean spectrum arrived pre-broadened, so real disorder looked like narrowing. A
  wrong reference state is the house's most common explanation for a wrong spectrum
  and it was the wrong one here; the second suspect only surfaced because the fixed
  reference state left the symptom in place.

Neither was ported by substituting a clean or equilibrium calculation for the
disordered or quenched one it is actually about. That would produce a folder that
looks like a port and is not one.

The 2026-08-03 audit also found the README stale in both directions and S01 claiming
a Sunny cross-check that nothing asserted; both corrected.

---

## Config-surface coverage audit (2026-08-04)

Prompted by two shipped field bugs that a thorough suite never saw. The suite's
*rigour* and its *coverage* had diverged: Gate 1/2/3 plus FeI2 bands and intensities
against Sunny to 1e-4 is genuinely strict, and no model in it applied a field off the
z axis, so both bugs lived there indefinitely.

Mechanically checking every documented config key against `tests/`:

**11 of 69 never appear in a test.**

| key | in examples | note |
|---|---|---|
| `interactions.kitaev` | 0 | an entire interaction TYPE, in no example; one test, tautological (see below) |
| `tasks.powder_average` | 9 | 9 configs use it; the tests call `powder_sample_modes` directly |
| `tasks.export_csv` | 4 | — |
| `tasks.sun_sampled_correlations` | 2 | added today; its wiring bug was caught by hand, not by a test |
| `crystal_structure.from_mcif` | 1 | `test_mcif.py` tests the READER, not the config key |
| `plotting.two_theta`, `plotting.energy_grid_step` | 1 | resolution/kinematics knobs |
| `calculation.imaginary_tolerance`, `.energy_tolerance` | 0 | the guards' thresholds |
| `magnetic_structure.optimize_k` | 2 | spiral-k optimization |
| `calculation.series_resum` | 0 | Dlog-Pade resummation選択 |

**THE PATTERN, and it is the actionable finding.** Most of these are not untested
physics — they are untested *config paths* to tested physics. `powder_average`,
`from_mcif` and the powder/mCIF suites are the clearest cases: the function is pinned
hard, and nothing checks that the YAML key reaches it. That is exactly how a wiring
bug survives, and it is not hypothetical: `sun_sampled_correlations` shipped today
referencing a variable named `calc` where the runner calls it `calculator`, which no
unit test could see and one manual run caught immediately.

**A second dimension this audit cannot see: COMBINATIONS.** Both field bugs were
present-key/absent-combination failures — `H_mag` appears in tests, but never
together with `mode: SUN`, and never with a direction off z. A key-level audit gives
a lower bound on the gap, not the size of it. Worth extending to a cross-product over
the axes that actually interact: engine mode x field x anisotropy x structure type.

**Suggested order**, cheapest and highest-risk first: a smoke test that RUNS each
example config through the runner (catches every wiring bug at once, needs no
physics); then `kitaev`; then the guard tolerances; then the combination matrix.

### Progress

**1. Config smoke test — ✅ DONE.** `tests/test_config_smoke.py` runs every shipped
config — **58 in this tree, 54 on a fresh clone** (the four under
`examples/future_exmaples/` are gitignored, so "staging is covered" is true on one
machine only) — and fails on an ERROR *log record*, not just an exception (the runner catches
and logs, so exception-only assertions see nothing). It immediately found a live bug:
the deprecated `propagation_vector -> single_k` mapping inserts `cone_angle_deg: None`
explicitly, and one of the two read sites lacked the `or 0.0` guard, so `float(None)`
raised — **every legacy `propagation_vector` config had been running with no magnetic
structure at all**, logged and carried on. FeI2 was the config that exposed it.

**2. `kitaev` — ✅ DONE.** `tests/test_kitaev.py`, 19 tests, pinned to the type's
EXACT `interaction_matrix` equivalent (K on one Cartesian diagonal entry), so no
external oracle is needed. The audit table above said "0 tests"; the truth was worse
than "none" — there was one, `test_new_interactions.py::test_kitaev_interaction`,
asserting

    assert any(str(s) in ['kx','ky','kz'] for s in hm.free_symbols) or len(...) >= 0

whose `or` clause is `len(...) >= 0`, always true. The only live assertion was "H is
not identically zero". **A test that cannot fail reads as coverage in every audit,
including this one** — the key-grep counted it as absent for the wrong reason and got
the right answer by luck. It has been rewritten around a K-linearity identity.

Three silent-drop bugs were sitting behind that gap, all now hard errors:

- an unresolvable `value` logged a WARNING and `continue`d, dropping the bond;
- an unrecognised `axis` fell through `.get(axis, 2)` to **z**, so `axis: c` or any
  typo silently built a z-Kitaev term;
- **`type: kitaev` under `symmetry_rules` had no propagation branch at all.** The rule
  ran the reference-bond search, looped over all 48 symmetry ops, and added ZERO
  bonds in silence — while CLAUDE.md §2 documented `ref_pair` as *required* for
  exactly this type. Measured on simple cubic: the `interaction_matrix` rule adds 6
  bonds, the equivalent `kitaev` rule added none.

The third is now implemented (a Kitaev term is converted to its diagonal matrix and
propagated by the tested `R J R^T` path — on cubic, one z-axis reference bond
correctly generates K^xx on the x bonds, K^yy on the y, K^zz on the z), and the
dispatch grew an `else: raise`, so no future rule type can vanish the same way.
**The root cause was structural: an if/elif chain over rule types with no else.**

**3. Guard tolerances — ✅ DONE.** `tests/test_guard_tolerances.py`, 13 tests. The
guards themselves were well covered; nothing checked that their thresholds are READ
or that they move in the right direction. The model is a Néel chain tilted by θ, which
gives an exact handle — ΔE(θ) = 2·J·S²·(1 − cos θ), verified at θ = 2°, 5°, 10° — so
each knob is bracketed above and below a drop of *known* size instead of a golden
number. The same structure carries imaginary magnons, so one model exercises both
guards and loosening the energy audit visibly hands off to the imaginary one.

No bug in the tolerances: all four read sites work. Two things did come out of it:

- **`calculation.imaginary_rel_tolerance` is a third tolerance documented NOWHERE** —
  not CLAUDE.md, not TUTORIAL.md, not `schema.py` — and guard 1 fires only when the
  absolute AND relative thresholds are both exceeded, so lowering
  `imaginary_tolerance` alone cannot make it fire. Now documented, and the AND is
  pinned (an AND→OR mutation fails two tests).
- **A key-level audit is structurally blind to undocumented keys.** The audit
  enumerated *documented* config keys and checked them against `tests/`; a knob that
  is in neither place is invisible to exactly the process meant to find gaps. This is
  the same shape as the `kitaev` finding one item earlier — there, a test that could
  not fail counted as coverage. **Both times the audit's own instrument was the blind
  spot, not the thing being audited.** Any future coverage sweep should enumerate from
  the CODE (`calc_config.get(...)` call sites) as well as from the docs.

Verified by MUTATION, not just by passing: hardcoding `energy_tolerance` fails 4
tests, hardcoding `imaginary_tolerance` fails 1, and flipping the guard's AND to OR
fails 2. A tolerance test that passes when the knob is ignored would be worth nothing.

**4. Combination matrix — ✅ DONE.** `tests/test_combination_matrix.py`, 26 tests over
engine mode x field (magnitude and direction) x anisotropy x structure type. Built on
identities that hold across the whole grid rather than per-cell reference numbers:
rotational invariance of an isotropic Hamiltonian; S=1/2 SU(N) ≡ dipole LSWT; supercell
band folding ({omega(q), omega(q+1/2)}, exact to 1e-15); the closed-form dimer triplon
sqrt(J^2 - J J' cos 2pi q) and its Zeeman splitting.

**It found a live silently-dropped term.** `single_ion_anisotropy`, `sia_matrix` and
`stevens` never reached the ENTANGLED engine: `build_entangled_model` assembled each
unit's on-site block from `_pair_matrix(Jex, DM, Kex, ...)` plus the Zeeman and read no
on-site anisotropy at all, while `_reject_unsupported_terms` let them through — its
docstring says the builder reads "the on-site SIA/Stevens terms", which is true of
plain SU(N) (`lswt.from_generic_model`) and was never true here. Measured: D = -5 meV
on an S=1 dimer moved the triplon by EXACTLY 0.000. Now implemented (embedded per
constituent, folded in before the reference state is chosen) and pinned against exact
diagonalization of the isolated unit — matching to ~1e-14 for easy-axis, easy-plane,
with a competing field, and at S=3/2.

Why no earlier audit could see it: every one of those keys is used in many dipole and
SU(N) tests, so a key-level sweep counts them covered. It is a mode x anisotropy
*combination* that was missing — the exact dimension this item exists to cover.
It also hid behind real physics: at S=1/2 an (S.n)^2 anisotropy IS a constant, so it
correctly has no effect, and every shipped entangled example is S=1/2.

Both historical field bugs were re-introduced as mutations to confirm the grid catches
them: dropping the SU(N) Zeeman fails 5 tests (including a non-slow one), and forcing
the field to +z fails 6. A third mutation (desyncing one module's mu_B) fails the
constant-consistency test, which reads the literal out of all six files that define it.

**A test of mine was vacuous, and the mutation run is what said so.** The first
entangled invariance test used the ISOTROPIC dimer — whose reference is a singlet with
no ordered moment, so its spectrum depends on |H| alone and the test survived the
forced-to-+z mutation untouched. It now uses an anisotropic unit, where the direction
is observable. Writing a test is not evidence that it can fail; running the bug against
it is.

**Method note:** the first version of this audit reported 22 keys, because `\b`
inside a character class is a backspace, not a word boundary. A confidently wrong
number from a five-line script — the same failure mode as everything else on the trap
list above, and the reason the figures here were re-derived before being written down.

---

## How things were validated (and the recurring trap)

The single most important lesson from this work, stated for the next session:

**A check that a wrong answer passes is not a check.** The engine has repeatedly produced
plausible-but-wrong spectra that looked fine:
- the S-power filter silently deleting quartic terms;
- the classical energy optimizing a different Hamiltonian than LSWT diagonalized;
- a mixed-spin intensity off by 60% (a constant factor, easy to wave through);
- the classical ground-state search constraining |m_i| = S_ref on a MIXED-spin model,
  so it minimised a Hamiltonian nobody asked for — and then the energy audit, sharing
  the same assumption, **rejected the correct structure**, leaving no setting at which
  the model would run. Note the shape of it: the failure surfaced as an imaginary-mode
  error pointing at `num_starts`, which can never fix it. A guard that fires for the
  wrong reason costs more than one that does not fire at all;
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
- **the mCIF reader multiplied moment components by the FULL lattice vectors**
  (`m_cart = m @ A`), so a 3 mu_B moment in a 4 A cell came back as 12 -- and,
  what actually matters because LSWT consumes only the direction, a moment with
  components on axes of UNEQUAL length came back pointing somewhere else.
  Ho2BaNiO5, the example FullProf itself ships: components (-0.1441, 0, -8.9931)
  in a 7.51 x 5.74 x 22.56 A cell lie 0.92 deg off -c, and this reader had them
  at 0.31 deg, carrying 203 mu_B -- which no Ho3+ ion can. The components are
  mu_B on UNIT crystal axes: FullProf writes those same three numbers as the PCR
  magnetic phase's Rx/Ry/Rz (refined within +/-10 mu_B) and states
  `spherical_modulus 8.99423` beside them, their plain Euclidean norm.
  **Sunny 0.8.1 does it the way we did** (`MCIF.jl`: `mu_new = supervecs *
  mu_new`), which is exactly why the cross-check passed -- and the in-repo
  validation file, TbSb, has its moment along c ALONE, where the two readings
  differ by a positive scale factor and therefore not at all in direction. *A
  file that cannot distinguish two conventions validates neither.* `read_mcif`
  now takes `moment_basis` (default `unit_axes`, the FullProf/Bilbao reading),
  the old reading is kept so files written under it stay readable, and
  `tests/test_diffraction.py` pins the convention against the file FullProf
  wrote rather than against ourselves.

- **the magnetic field was SILENTLY DROPPED in `mode: SUN`.**
  `SUNModel.from_generic_model` built its on-site terms from sia / sia_matrix /
  stevens and nothing else, so the Zeeman term never reached the Hamiltonian: every
  SU(N)-in-field calculation quietly solved the ZERO-FIELD problem. The entangled
  engine had always applied it, so this was an oversight, not a choice. Fixed and
  pinned by extending the load-bearing gate -- S=1/2 SU(N) is IDENTICAL to dipole
  LSWT -- to finite field, which checks the term's presence and its sign at once
  against an engine whose Zeeman convention is separately pinned. Agreement: 0.0.
- **and every field was forced along +z.** `_resolve_param_map` FLATTENS
  vector-valued parameters, so `H_dir: [1, 0, 0]` came back as the scalar 1.0;
  `_resolve_field` tested `isinstance(h_dir, list)` on that scalar, the test failed,
  and it fell through to a hardcoded `[0, 0, H]`. Wider than the first bug --
  `_resolve_field` is shared by SU(N), entangled, thermal_mc and annealing -- and
  nastier, because the field was PRESENT and of the right MAGNITUDE, merely pointing
  the wrong way. `tests/test_sun_zeeman.py`.

  Two lessons, both general:

  1. **A field-free test suite cannot see a dropped field term**, however many tests
     it has. The SU(N) suite is thorough -- Gate 1/2/3, FeI2 bands AND intensities
     against Sunny to 1e-4 -- and not one of its models applies a field. Both bugs
     surfaced only from porting Sunny's skyrmion tutorial, where a field competing
     with an easy-plane anisotropy IS the mechanism. Count coverage in physics
     exercised, not in tests passing.
  2. **Agreement between two implementations is not evidence when they share the
     faulty helper.** "SU(N) == dipole for four field directions" would have PASSED
     while the direction bug was live, because both engines were equally wrong. The
     test therefore also asserts that a transverse field differs from a z field.

- **`type: kitaev` under `symmetry_rules` expanded to ZERO bonds, in silence.**
  `add_symmetry_interaction` dispatches on rule type through an if/elif chain with
  branches for `dm`, `heisenberg`, `anisotropic_exchange` and `interaction_matrix` —
  **and no `else`.** A `kitaev` rule therefore ran the whole reference-bond search,
  looped over all 48 symmetry ops of the cell, matched no branch, and added nothing;
  the run succeeded with the interaction simply absent from H. CLAUDE.md §2 has
  listed `kitaev` among the types for which `ref_pair` is REQUIRED the entire time.
  Two smaller silent drops sat alongside it in the explicit-bond path: an unresolved
  `value` warned and `continue`d, and an unrecognised `axis` fell through
  `.get(axis, 2)` to z, so `axis: c` built a z-Kitaev term without comment.

  Three lessons:

  1. **A dispatch over a closed set of types needs an `else` that raises.** This is
     the same shape as the SU(N) missing-terms bug above (`from_generic_model` reads
     the terms it knows and ignores the rest) — the second time an unhandled case
     has meant "drop it quietly". Both now raise.
  2. **A test that cannot fail reads as coverage.** `kitaev` did have a test, whose
     live content was "H is not identically zero" after an `or len(...) >= 0`
     tautology neutralized the real assertion. It passed for a wrong axis, a wrong
     sign, and a term propagated to zero bonds by a different code path.
  3. **Documentation is not evidence that a path works.** The `symmetry_rules`
     kitaev route was documented in detail and had never once been executed. Prefer
     an oracle the feature cannot supply itself: here the exact
     `interaction_matrix` equivalent, which needs no external reference at all.

- **on-site anisotropy was SILENTLY DROPPED in `mode: entangled`.**
  `build_entangled_model` folded the bilinear pair terms and the Zeeman into each
  unit's on-site block and never read `single_ion_anisotropy` / `sia_matrix` /
  `stevens`. `_reject_unsupported_terms` did not catch it because its docstring's
  premise — that the builder reads "the on-site SIA/Stevens terms" — describes
  `lswt.from_generic_model`, not this engine. D = -5 meV on an S=1 dimer moved the
  triplon by exactly 0.000. Now implemented and pinned against exact diagonalization
  of the isolated unit (~1e-14, incl. S=3/2 and a competing field).

  Two lessons:

  1. **The guard against silent drops was itself guarded by a stale comment.** A
     rejection list that enumerates what an engine "does not support" has to be
     derived from what the builder actually reads, or it drifts into permitting
     exactly what it exists to forbid. This is the third instance on this list of a
     term vanishing because a dispatch or a filter did not know about it.
  2. **Physics can hide a bug from the very test that would find it.** At S=1/2 an
     (S.n)^2 anisotropy IS a constant and correctly has no effect — and every shipped
     entangled example is S=1/2, so even a careful "does the anisotropy change the
     spectrum?" check written against them would have concluded, correctly and
     uselessly, "no". Only S >= 1 discriminates.

- **the rotating-frame Ewald projector algebra was transcribed from Sunny, and the
  transcription was wrong** (#24b, caught 2026-08-13 before it shipped — the refusal
  had deliberately been left in place until an oracle existed). `ewald.py` phases
  `A(q)` over the FULL bond vector, matching pyMagCalc's symbolic `H`; Sunny phases
  over lattice translations only. The projector combination is not invariant under
  that regauging: in pyMagCalc's convention **R1 pairs with q+k, the mirror of
  Sunny's**, and the `k_case 2` cross terms pick up a per-ROW `exp(±2i k·r_i)`.

  Three lessons, the first one new:

  1. **Reading the reference implementation correctly is necessary and not
     sufficient.** GAP4_PLAN's instruction — read Sunny, don't re-derive — was right
     and had already corrected four wrong characterizations of this item. The fifth
     error came from following it *literally*: two codes can implement the same
     physics in gauges that differ by a phase the algebra does not commute with.
  2. **The wrong version passed every test that existed.** The mirrored assignment
     is identical whenever `A(q)` is uniaxial about the spiral axis (then
     `R1 A R1* = 0`) and for any one-site cell (`r_i = 0`) — true of every model in
     the repo. Separating them needed a cell with two sites at an intra-cell offset
     AND the Sunny `S0` spin-direction convention; the `local_directions` convention
     has zero relative spiral phase in the rotating frame and hides it too. The first
     oracle model tried reproduced the supercell answer to 1e-15 *with the bug in*.
  3. **Check where the oracle is entitled to hold.** The rotating-frame-vs-supercell
     identity is exact only where the method is: always at `k_case 2`, and at
     `k_case 3` only for a uniaxial `A(q)`. A 9e-4 residual on a chain was read as
     "the harness is broken" when it was the ±2k umklapp the theory legitimately
     drops. That distinction is now a warning
     (`core._check_ewald_spiral_validity`), not a footnote — and Sunny has no
     counterpart to it, because its `check_rotational_symmetry` never sees the
     dipolar term.

- **`magcalc symmetry`'s allowed exchange matrix collapsed on any lattice with
  measurement noise in it** (2026-08-17, surfaced by `magpipe`'s Phase 4). The
  constraint `J = R J Rᵀ` is linear in the nine entries of J, i.e. a null space, and
  `get_bond_constraints` handed it to `sp.solve` after "sanitizing" the Cartesian
  rotation with `np.round(R_cart, 10)` + `nsimplify`. But `R_cart = A R_frac A⁻¹`
  inherits the lattice's noise, and nsimplify of a 10-decimal float is not `2/3` — it
  is `1666666499/2500000000`. That R is **not orthogonal**, so the constraint has far
  fewer solutions than it should. Measured: perturbing KFe₃J's `a` by **1e-7 Å** took
  its NN bond from **6 free parameters to 1**, every off-diagonal reported as
  symmetry-forbidden — "no DM allowed on this bond", on the config whose whole point
  is its DM term. On a Materials Project NiO primitive cell (cubic to six decimals,
  not exactly) the same giant rationals instead ran sympy for **> 10 minutes without
  returning**, which is how it was noticed at all. Now a 0.2 ms SVD null space, with R
  snapped to the nearest orthogonal matrix first (spglib had already accepted the op
  at symprec = 1e-3) and coefficients snapped at 1e-5.

  Two lessons:

  1. **Symmetry is discrete, so its answer must not move continuously with the
     input.** That is a property the code can be tested against directly, with no
     oracle at all, and it is now the test: perturb the lattice below any
     experimental precision and require the allowed form to be *identical*. Every
     previous test used a hand-typed lattice (`a: 6.0`) whose rotations come out
     exact, so the whole failure mode was outside the suite's reach — a real CIF was
     needed to reach it, and nothing in the suite used one.
  2. **"Sanitizing" floats into exact numbers invents information.** Rounding then
     `nsimplify` looks like it is cleaning the input up; what it actually does is
     assert 10 digits of exactness the number never had, and then propagate that
     assertion through exact arithmetic. The tolerance belonged in the *solve*, where
     it is now explicit, not in a preprocessing step that hides it.

Every one was caught by an **independent oracle or an exact identity**, never by
inspection. So: validate against Sunny (in-repo) or a textbook analytic result; prefer
identities (decoupled-sublattice sum, S=1/2 SU(N)≡dipole, ferromagnet has zero
corrections) that fail loudly; and be suspicious of a discrepancy that is a *clean
constant factor*.

**REPRODUCIBILITY IS NOT CORRECTNESS** (2026-08-13, `anneal`, OPEN_WORK item 9). The
documented acceptance criterion for a ground state here is "the energy is
reproducible across several `seed`s" — and `anneal` returned a local **MAXIMUM** on
4 of 4 seeds, reporting exactly that consensus. Its Metropolis phase found the true
minimum every time; the `steepest_descent` polish that ran afterwards, whose result
was taken unconditionally, walked away from it (that step ignores the on-site block
`H_ii`, so with a large single-ion term it is not a descent step at all). A
deterministic bug downstream of a stochastic search reproduces perfectly. What
caught it was an **analytic minimum in closed form**, not a repeat run.

The companion lesson is where the pre-existing test was pointed:
`test_steepest_descent_is_monotone` had asserted that exact property for months, on
an AFM chain — whose `H_ii` is zero, i.e. precisely the case where the property is
trivially true. **A test aimed at the easy case does not cover the hard one**, and it
reads as coverage.

**A published reference number is not automatically a converged one.** Sunny's own FeI₂
example converges to a *local* minimum (E/site −2.35592338, one `minimize_energy!`); the
true ground state (−2.91893118) needs restarts. Days were spent "debugging" against a
non-converged reference. Re-minimize the reference before trusting it.

### A bug sweep, 2026-09-10 — seven defects, and what hid each one

Full gate after the sweep: **1681 passed, 3 skipped** (`pytest -m ""` from the
workspace root, 2026-09-10, 19:01, idle machine). The three skips are all named
reasons and none is new: `seekpath` absent (one of the mutually exclusive pair at
`tests/test_cell_utils.py:146`/`:161`), `test_alps.py:187` skipping BECAUSE ALPS is
installed, and magpipe's `test_magref.py:568`.

Each fix carries a test verified to FAIL against the old code (#7 excepted: an
unreferenced branch in a legacy example script, covered by `ruff --select F821`).

| # | Defect | What hid it | Pinned by |
|---|--------|-------------|-----------|
| 1 | `tasks: {static_sqw: true}` passed an undefined name (`calc`, not `calculator`) and raised NameError on EVERY run — inside `except Exception: logger.exception(...)`, so the task was silently absent while documented (TUTORIAL 4g) and marked ✅ (#19 above) | `tests/test_static_correlations.py` called `static_intensities` DIRECTLY. The function was right; the runner branch that reaches it had never executed | `tests/test_static_correlations.py::test_runner_static_sqw_task_actually_runs` |
| 2 | The general-polarization chiral term was contracted from the UNPROJECTED correlation tensor, so a polarization component across **q** carried spurious chirality — SF⁺ ≠ SF⁻ for a cycloid with P ⊥ q, where the term vanishes identically, and the sign could flip | The pinned Sunny reference for a general P was generated on the proper SCREW, whose chiral vector is already ∥ q — exactly where the correct and the incorrect expressions agree. Now pinned by the exact identity chiral(P) = (P̂·q̂)·chiral(q̂) on a screw, a cycloid AND a 45°-tilted helix | `tests/test_polarization_frames.py::test_chiral_term_only_sees_the_longitudinal_polarization` (screw / cycloid / tilted), `::test_a_cycloid_has_no_chirality_across_q` |
| 3 | SCGA used ONE λ for genuinely inequivalent sublattices, which #9 above already claimed it refused. A mixed-spin chain (S = 1, 5/2) returned an S(q) whose two sublattice sum rules were off by +262% and −42%, with a TOTAL residual of 3e-16 | Every SCGA test is one symmetry class (chain, square, kagome). The refusal is now decided by the per-sublattice sum-rule residual itself, not by the symmetry labelling — symmetry-equivalent sites land at ~1e-16 | `tests/test_scga.py::test_inequivalent_sublattices_are_refused`, `::test_the_refusal_reports_the_actual_violation`, `::test_one_symmetry_class_satisfies_every_sublattice_rule_exactly` |
| 4 | `magcalc mcif-out --minimize` raised TypeError on every config (`GenericSpinModel.minimize_energy(p_num)` called with no argument), and the minimizer it called builds its energy from the BOND table plus a global g = 2 Zeeman — so `sia`/`sia_matrix`/`stevens` and per-site g-tensors were absent from it | The flag was added with the export and never run. Now routed through `MagCalc.minimize_energy` (the model's own Hamiltonian), pinned on an easy-axis chain SEEDED along the degenerate z | `tests/test_diffraction.py::test_mcif_out_minimize_uses_the_full_hamiltonian` |
| 5 | The spiral optimizer's on-site term was a hand-rolled parser over `type: 'sia'` only, so `sia_matrix` and `stevens` (k = 2 included) reached it as NOTHING. It minimized an exchange-only energy and committed that structure for LSWT to expand about | Identical in cause and shape to the `build_supercell` defect of 2026-08-15: a bond table carries no on-site term, and every spiral test was bond-only. Now delegates to `thermal_mc.onsite_quadratic` — bit-identical for the axial spelling, and it RAISES for Stevens k ≥ 4 rather than dropping it | `tests/test_single_k.py::test_optimizer_sees_onsite_anisotropy_however_it_is_spelled`, `::test_optimizer_refuses_a_stevens_term_it_cannot_represent` |
| 6 | `write_mcif` emitted colliding labels (`Cu1, Cu1, Cu11` → `Cu1, Cu11, Cu11`), and `read_mcif` then rejected the file blaming a magnetic stabiliser — a writer producing files its own reader refuses | It was KNOWN, and `magpipe/tests/test_magndata.py` used it as the FIXTURE for "an unreadable entry is reported". A fixture that depends on a bug disappears with the bug and takes its check along; that test now hand-writes the colliding file | `tests/test_diffraction.py::test_write_mcif_labels_are_unique` |
| 7 | `NameError` in a log message in `examples/materials/aCVO/sw_cvo.py` (legacy standalone script) | Unreferenced branch; found by `ruff --select F821` | `ruff --select F821` |

### Per-site g-tensors with parameter entries, 2026-10-13

Found while modelling α-Cu₂V₂O₇ with a symmetry-related off-diagonal g_ac (the 16 Cu sites need ±g_ac). The g-tensor
resolver built its parameter map from the parameter NAMES (`_resolve_param_map(self.parameter_order)`), so every
name in a `g:` entry resolved to its own string. Each test below FAILED against the old code.

| Defect | What hid it | Pinned by |
|--------|-------------|-----------|
| Bare names became free sympy symbols never tied to the parameter (wrong bands); `-g` raised (unary minus on a str); `-1*g` silently became `''` (str repetition) and then "could not parse ''"; in the classical-energy path the unbound symbol reached the minimizer (L-BFGS-B: "cannot determine truth value of Relational"; `anneal`: "quadratic form could not be extracted", i.e. annealing was refused for every parameterised g). Fix: `_resolve_g_tensors(param_map)` takes the caller's map (symbols on the LSWT path, numbers on the classical path); without one it uses the configured numeric values | Every g-tensor test used NUMERIC entries, although the docstring advertised "numbers, parameter names, or expressions" | `tests/test_hamiltonian_terms.py::test_gtensor_parameter_entries_match_numeric` and `::test_gtensor_parameter_entries_in_classical_energy` (bare names, arithmetic, negated name) |
| `scga._g_tensors` caught EVERY resolver exception and fell back to g = 2·I, so a parameterised g-tensor was silently ignored in SCGA. Now only a model without the resolver falls back | `except Exception` around the call | `tests/test_hamiltonian_terms.py::test_scga_g_tensors_resolve_parameters` |

Check on the α-CVO model (16 sites, H = 14 T ∥ c, g_ac = 0.20 as named parameters): `anneal` now runs and reaches
E = −41.307948 meV in 6/8 runs, identical to the numeric-g L-BFGS-B result (best energy hit by 1–3 of 40 starts).

**The pattern across #1, #4 and #5.** All three are the SAME failure of coverage: a
function validated directly while the PATH that reaches it was never executed. #1's
runner branch, #4's CLI flag, #5's on-site term. `tests/test_corrections.py`'s
`run_calculation(str(p))  # must not raise` is the shape to distrust — the runner
swallows a task's exception, so "does not raise" asserts nothing about the task. The
new runner test asserts on the success line and the ABSENCE of the failure line.

### Symmetry propagation of a forbidden component, 2026-10-18

Found while mapping the DM canting pattern of α-Cu₂V₂O₇ (Fdd2, `examples/materials/aCVO`). `add_symmetry_interaction`
propagates a `ref_pair` value over the space group, D' = det(R)·R·D, J' = R J Rᵀ, and kept the FIRST operation
that reached each bond. Several operations reach every bond of an orbit, and they agree on its value only if the
value is invariant under the bond's little group (bond-preserving: D → D; bond-reversing: D → −D, J → Jᵀ) — which
is exactly Moriya's rules. When they disagree, the first-hit rule picked the sign of the forbidden component by
spglib's enumeration order, bond by bond.

| Defect | What hid it | Pinned by |
|--------|-------------|-----------|
| The J2 bond Cu0–Cu13 (3.97512 Å) has the 2-fold axis along c through its midpoint, so D_z is forbidden; a `Dx,Dy,Dz` rule still gave D_z = ±1 on all 8 J2 bonds, with signs disagreeing with a hand propagation on 8 of 16 directed entries and **not invariant under the F-centring translations** (32 of 48 translated entries differed). The minimized ground state then broke translation symmetry (spins on centring-related sites differed by up to 0.12). Fix: two passes — record what EVERY operation does to the reference bond, refuse (`SymmetryConflictError`, a `ValueError`) if operations landing on the same directed bond disagree, then emit in the old first-hit order, so an allowed value's table is bit-identical | Right bond count, plausible spectrum; every shipped rule puts a symmetry-allowed value on its bond (aCVO's J1 is `Dx,0,0`), so no config and no test ever propagated a forbidden one — the `ref_pair`/`steepest_descent`/on-site-anisotropy shape again: the property was never false in the suite | `tests/test_symmetry_propagation_consistency.py`, 17 tests. Oracle: the 16 Fdd2 operations written out by hand (cosets × F-centring), not spglib's list — it finds the J2 D_z conflict itself, and the engine's J1, J3 and in-plane-J2 tables must equal it entry for entry, be antisymmetric and be centring-invariant. The shipped aCVO model (32 DM + 32 anisotropic + 80 Heisenberg entries) is pinned to the same hand table. A full `interaction_matrix` on J2 pins the general little-group condition (symmetric xy block allowed, symmetric xz forbidden) |

Two things to know before relying on it. The check is numeric, on random samples of a symbolic value (a forbidden
`Dz` is caught whatever number Dz is later given), at 1e-8 of the largest entry on an exact lattice — and it is
widened to 100× the lattice's own noise, the non-orthogonality of the Cartesian operations, because on a CIF that
is cubic to six decimals every value propagates slightly differently through different operations and a fixed
tight tolerance would call measurement noise a symmetry conflict (pinned on P23 with c = a(1+1e-6): the
noise passes, a 1 % transverse component is still refused). And the distance-rule expanders (`_expand_dm_rules`,
`_expand_interaction_matrix_rules`) swallow propagation errors into a logged error plus a "loop breaker" that
drops the orbit; a `SymmetryConflictError` is re-raised through them, the rest of that swallow is unchanged and
is the same silent-drop shape as above. All 47 shipped configs with bond rules still build.

Full gate on 7e2dfd9 (`MAGCALC_NO_GUI=1 pytest -m "" -rs` from the workspace root, 2026-10-18, idle machine):
**1704 passed, 1 failed, 3 skipped, 1708 collected, 16 min 23 s**, no crashed xdist worker. The failure is this check
doing its job one repo over: `magpipe/tests/test_excite.py::test_a_single_k_ansatz_flags_a_dm_that_breaks_its_spiral_axis`
sets `D = (1, 0, 0)` on both orbits of a one-site 3 x 12 x 12 A chain fixture — P4/mmm, 16 operations, and the NN
midpoint is an inversion centre, so Moriya forbids every DM component there. Before, `add_symmetry_interaction` kept
the value silently and magpipe's planner reached its single-k branch; now `add_symmetry_interaction` raises
`SymmetryConflictError` ("the value (D1x, D1y, D1z) is not invariant under the bond's own site symmetry, which forbids
its components x, y, z ..."), `plan()` catches it as "no Luttinger-Tisza ansatz" and seeds k = 0, and the test's
`p.k is not None` fails. The same test passes against e5cacd3. Resolve it in magpipe (a fixture whose bond admits a
DM, or an assertion on the refusal), not by weakening the check. One thing the gate measured about the fix itself:
through the config route the typed exception does not survive — `GenericSpinModel`'s `symmetry_rules` loop
(`generic_model.py:550`, pre-existing) catches every `Exception` and re-raises a plain `ValueError("Failed to expand
symmetry rule {rule}: ...")`, so a caller's `except SymmetryConflictError` sees nothing; only a direct
`add_symmetry_interaction` call gets the subclass. The three skips, by name: `pyMagCalc/tests/test_cell_utils.py:146` (`seekpath` absent — the structural
pair), `magpipe/tests/test_alps.py:187` (ALPS is installed — the structural fallback), and
`magpipe/tests/test_magref.py:568` (an in-test skip since magpipe `7bad01b`, 2026-08-20: `pb_ho.pcr` refuses
`freeze_all_except_moments(2)`, so the data-file-name guard is never exercised — a skip-on-exception, which this
repo's rules call a hole; it is magpipe's to close).

**Resolved 2026-10-22.** magpipe `4e4bce5` moved that test to a two-site zigzag chain (Pmma) whose second-neighbour
bond admits exactly D ∥ y, and fixed the emitter defect the check exposed: it wrote all three DM components as fit
parameters even where the allowed form pins one to zero, so every magpipe DM on a constrained bond would have been
refused. magpipe `283f694` closed the `test_magref.py:568` skip. Full gate on master f0bbe3d (this fix merged with
the rule-value parser fix a87c8a8 below, magpipe at 283f694), 2026-10-22, idle machine: **1735 passed, 2 skipped,
0 failed, 1737 collected, 14 min 34 s**, no crashed xdist worker. The two skips are the structural `seekpath` and
ALPS ones. The merge is also what makes `sqrt(3)*D1x/3`, the allowed kagome DM, pass this check: before the
parser fix it reached the check as an unbound `sqrt3` and was refused as a conflict.

### Luttinger–Tisza guard: J(k) from J(R), the grid batched, 2026-10-29 — 12–19×, k* bit-identical

A profile of every shipped example config under `kernels: rust` and `backend: fortran`
found the q≠0 guard (#13) to be the largest Python cost left in a plain dipole run.
`ordering_wavevector` built the 3N × 3N Fourier exchange matrix with a Python loop over
N² site pairs, at every point of a 24³ grid plus the Nelder–Mead refine: about 14 000
builds, 1.7 ms each at 16 sites.

`spiral_opt._fourier_exchange` now compresses the OUC bonds once into one dense block
J(R) per distinct cell translation, J(k) = Σ_R J(R) e^{2πik·R}: 9 translations on
CoRh₂O₄, 17 on α-Cu₂V₂O₇. It evaluates a whole stack of k with one phase matrix and a
`tensordot`. The grid is scanned in ~32 MB batches through one stacked `eigvalsh`, and
the pick keeps the loop's rule exactly: the first point in product order that beats the
best by more than 1e-12. `luttinger_tisza_guess` (also the spiral optimizer's initial
guess) and `ordering_wavevector`'s `_eigmin` both use it. This is NumPy, not a compiled
kernel: the loop was Python-bound only because it was written per pair.

`ordering_wavevector`, old against new in one process on a loaded machine (best of two):

| model | sites | before | after |
|---|---|---|---|
| YIG (SW21) | 20 | 46.6 s | 3.3 s |
| α-Cu₂V₂O₇ | 16 | 29.4 s | 2.5 s |
| S07 pyrochlore | 16 | 34.0 s | 2.2 s |
| CoRh₂O₄ (S01) | 8 | 6.5 s | 0.34 s |
| KFe₃J (2-D grid) | 9 | 0.54 s | 0.12 s |

On ten example models (those five, ZnCVO and SW03/04/07/09), k*, λ_min, λ(0) and the gain
`ordering_wavevector` returns are bit-identical before and after. The raw, unsnapped
`luttinger_tisza_guess` k moves by 1e-8 to 2e-5 on six of them. That is Nelder–Mead
landing elsewhere on a flat minimum, or on an axis with no bonds, after rounding-level
changes in J(k); `ordering_wavevector` already snaps it at 1e-3.

**Oracles** (`tests/test_luttinger_tisza.py`, fast suite):

* the per-pair loop, kept verbatim in the test: J(k) to 1e-13 at 40 random k, as a stack
  and singly, on CoRh₂O₄ (8 sites, 216 OUC images), the DM kagome SW09 and the SW03 chain.
  On the same three, the unrefined grid pick equals the loop's point exactly;
* the triangular antiferromagnet, eigmin J(k) = −3 J_aa at K = (1/3, 1/3), with
  isotropic, easy-plane and easy-axis exchange;
* the DM chain, whose in-plane block 2J cos q ± 2D sin q has its minimum
  −2√(J² + D²) at q = π − atan|D/J|. This is the one check on the antisymmetric, complex
  part of J(k).

Two mutants were checked: the conjugate phase e^{−2πik·R}, and J(R) blocks stored
transposed. Both fail the file.

**Gate** (2026-10-29, workspace root, `MAGCALC_NO_GUI=1 python -m pytest -m "" -rs`,
pyMagCalc `bbadb64` plus this change, fMagCalc `4a5ff4b`, rsMagCalc `1d2532d`, magpipe
`e92b2c7`): **2405 passed, 2 skipped, 0 failed, 15 min 38 s**. The 13 added tests are
`tests/test_luttinger_tisza.py`'s, and the two skips are the structural pair.

### mCIF import reads the ion; a bare `+` is a charge of one, 2026-10-28

`read_mcif` never read `_atom_site_type_symbol`, so a structure imported from an mCIF
had no magnetic form factor unless the caller named an ion: `from_mcif` without
`mcif: {ion: ...}`, `magcalc mcif` without `--ion`, and magpipe's MAGNDATA ingest all
computed S(Q,ω) with f(Q) = 1, even on a file that said `Tb3+`. Each site now carries
`type` (the symbol as written) and `ion`. The ion is set when the symbol states an
integer charge: `Mn3+`, sign-first `Fe+3`, and `Cu+` as Cu1+. A bare element, a
label-like `Mn1`, a non-integer `Fe2.5+` or a placeholder gives none, since nothing
guesses (`mcif.ion_from_type_symbol`). `mcif_to_config_fragment` uses the file's ion
unless one is passed, which still wins for every site.

**Swept over MAGNDATA's 2414 deposits** against the committed reader:

* the same 68 refusals, and not one site position or moment changed;
* 544 sites state a charge, 88 of them sign-first;
* every one names the element its label does.

28 014 of the 28 558 sites give only the element, so for most MAGNDATA files the ion
still has to come from elsewhere: state it with `mcif: {ion: ...}` when intensities
matter.

**A separate one-line bug, found beside it.** `form_factors._normalize_ion('Cu+')`
returned `Cu0`, the neutral atom's form factor: a sign with no digit fell through to
charge 0. It is now `Cu1`.

Pinned in `tests/test_mcif.py`: nine type-symbol cases, TbSb's six `Tb3+` sites, the
fragment taking the file's ion unless one is given (read back through
`GenericSpinModel.ion_list`), and a file without the column giving none. The bare-sign
fix is pinned in `tests/test_form_factor.py`. Gate (workspace root, with magpipe's
schema 0.6 alongside): **2392 passed, 2 skipped, 0 failed, 16 min 22 s**.

### Rust kernels K7, 2026-10-29 — the dimer series' cluster PT behind `kernels: rust`

Not one of the planned six: the plan had excluded `sun/dimer_series.py` as "symbolic", which it never was. The
2026-10-28 profile of all 23 shipped example configs under `kernels: rust` + `backend: fortran` found it the one loop
left that was Python-bound. rsMagCalc `9061710` (`src/dimer_series.rs`) carries `block_effective_series` (the Bloch
recursion with column-dependent resolvents and the des Cloizeaux Hermitization) and `_cluster_pt` as
`cluster_pt_batch`: the pair operators embedded in the cluster's nloc^m product space, the ground state and each
one-particle total-Sz sector run separately, rayon over the clusters of one batch. `hopping_series` under
`kernels="rust"` hands each cluster size's new topologies to it in one batch, for every size, in place of the spawn
pool. The enumeration, `_struct`, the linked-cluster subtraction, `eigenvalue_series` and Dlog-Padé stay in Python.
`DimerSeriesModel(..., kernels=, n_workers=)` and its two constructors take the selection; `mode: entangled` with
`series_order` passes `calculation.kernels` and `calculation.n_workers` through (before, `n_workers` never reached the
series, whose NumPy pool always took cores − 1). A kernel that raises is logged once by `kernels.fell_back` and the
NumPy PT answers for the rest of that model.

**Profiled before it was written** (cProfile, Rb₂Cu₃SnF₁₂ order 5, `hopping_series`, `os.cpu_count` patched to 2 so
the NumPy path is serial; pyMagCalc `bbadb64`): 38.4 s under the profiler, of which `_cluster_pt` 32.7 s (85 %).
Inside it the compiled part was scipy's sparse products, ~11 s (`csr_matvecs` 8.3 s, CSR additions and conversions
~2.5 s); the rest — `block_effective_series`' many small dense products, `_embed_pair_op`, the sector slicing — was
Python-bound: **~70 % of the run**, above the stop-at-half bar.

**Two Python-side changes, under both kernel sets, exact.** With the PT compiled, the bookkeeping came next (order 6:
kernel 24 s, Python 22 s). `_struct` is memoized per form: the subtraction asks for every connected subcluster of
every cluster, and at order 5 41 789 asks name only 3 501 distinct forms. The subtraction of a subcluster's weight is
one fancy-indexed operation over all orders instead of an `np.ix_` per order — the same subtraction of the same
numbers, element by element. NumPy's hoppings and band series are bit-identical to `bbadb64`'s (`==`, orders 5–7).
After both, order 6 under `rust` takes 21.7 s: 15.2 s in the kernel, 1.3 s in `_struct`, 0.2 s in the
enumeration and 5.1 s in the rest of the Python (the subtraction and the assembly). At 30 % Python, the
bookkeeping stays in Python, as scoped.

**Exactness.** The kernel's arithmetic is NumPy's term by term, with savings that change no number: V·Ω_m computed
once per m and reused (NumPy recomputes it inside every M_k), Ω_0^†X as a row gather (Ω_0 is the model-space
injection, and Ω_0^†Ω_{i≥1} vanishes exactly), the Gram terms of S_k and of M_k's diagonal part summed in adjoint
pairs, one sparse V per Sz sector built directly on that sector's states, the all-zero rows of the low-order Ω
skipped. Rounding is another matter, and was measured rather than assumed. The yardstick is NumPy against ITSELF
with a cluster's nodes relabelled (the same physics, every sum in another order): on 40 order-6 six-link clusters of
Rb₂Cu₃SnF₁₂ that spread is 3e-14 on deltas of size ~1.3. A first kernel, with one running sum over the thousands of
rows of each Gram term, sat 1.6e-13 from NumPy (up to 5× the spread); summed in blocks of 64 rows, as BLAS does, it
sits 5e-14 (1.3×), at no cost in speed. That is not a general accuracy claim — on a synthetic 20 000-state problem
both variants lie inside NumPy's own spread — only that the kernel is now as close to NumPy as NumPy is to itself.

The subtraction then amplifies whatever rounding there is: a six-link cluster's linked weight at order 6 is ~100×
smaller than its raw PT weight, and each hopping sums thousands of them. So the hoppings agree to 2e-12 at orders
≤ 5 and to 2e-11 at order 6 (scale 1.6, i.e. 1.3e-11 relative); the Γ band series of the pinwheel against unmodified
NumPy, per order λ⁰…λ⁷ at order 7: 0, 0, 1e-14, 3e-14, 3e-13, 2e-12, 2e-11, 6.5e-11 (the first kernel: 5.6e-10 at
λ⁷). Every resummed Γ number of the example's README table is reproduced: order 6 doublet Dlog-Padé 1.59 / sum
−0.23, order 7 2.395 / 2.801 (README: 1.6; 2.40 / 2.80). With and without Sz sectors the kernel's hoppings are
bit-identical (NumPy's differ by 5e-12 at order 6).

**Pinned to.** `tests/test_dimer_series.py` runs every oracle under both kernel sets through the `kernels` fixture,
and each `rust` case asserts the kernel ran rather than fell back: the PT engine against exact diagonalization
(error ratio between two λ inside (2ⁿ, 2ⁿ⁺²) at order 5), the chain's analytic first order, momentum-resolved ED of
the alternating chain at J'/J = 0.4 (order 6, < 5e-4 J; slow) and 0.8 (order 7 + Dlog-Padé, gap within 8 %; slow),
PRR 8, 013247 Eq. (A11) on Cu₅SbO₆ through GenericSpinModel, plus one new FAST pipeline oracle: the order-4 series
of a 6-dimer ring against its ED at λ = 0.1 and 0.05 must shrink by 2⁵ (ratio inside (2⁴, 2⁶)); a 6-dimer ring is
exact for the infinite chain below order 6, so it runs in 0.3 s. `tests/test_kernels_dimer_series.py` (7, fast):
cluster parity at 1e-12 on the chain and Rb₂Cu₃SnF₁₂, with and without sectors; the whole Rb₂Cu₃SnF₁₂ hopping series
and ground-state energy at order 4 at 1e-12, with a spy confirming one kernel PT per topology; a raising kernel logged
once with NumPy's series returned bit-identically; `mode: entangled` passing `kernels` and `n_workers` through, the
dispersion independent of the kernel set to 1e-10 and resummed as
`series_resum` asks — which retires that key's entry in the key-coverage audit's allow-list. rsMagCalc `tests/test_dimer_series_parity.py` (19): the PT engine
at 1e-12 on quasi-degenerate, multi-order (eigenvalue_series' use), p = 1, scipy-sparse and excited-model-space
problems; `cluster_pt_batch` at 1e-12 on the chain's, Rb₂Cu₃SnF₁₂'s and random nloc = 3 clusters (complex Hermitian
pair operators, a link with pa > pb); order-6 rounding within 3× NumPy's relabelling spread; n_workers = 1 and the
full pool bit-identical; the degenerate-model-space refusal and malformed inputs raising ValueError.

**Timing** (M1 Pro, 10 cores, 2026-10-29 08:51–10:31, no pytest or other compute job running; `hopping_series` of
`examples/entangled/Rb2Cu3SnF12/series_dispersion.py`'s model, wall time, `<this session's scratch>/bench.py ORDER
serial|pool|rust`). "serial" patches `os.cpu_count` to 2, so the NumPy path takes no pool; "pool" is the NumPy
default (spawn pool of cores − 1 for cluster sizes ≥ 4); "before" is pyMagCalc `bbadb64`, "after" this change:

| order | NumPy serial, before | NumPy pool, before | NumPy pool, after | `kernels: rust` | rust vs pool | peak RSS (rust) |
|---|---|---|---|---|---|---|
| 5 | 30.3 s | 11.0 s | 8.2 s | **1.6 s** | 7× | 0.33 GB |
| 6 | 566 s (after) | 115 s | 111 s | **21 s** | 5.4× | 1.1–1.35 GB |
| 7 | — | 1945 s (32 min) | — | **385 s (6.4 min)** | 5.1× | 1.6–4.3 GB |

One order-6 six-link cluster (400 of the 8152 topologies): NumPy 66 ms, the kernel 13.8 ms on one thread and 1.9 ms
on all ten. The cost grows 13–18× per order under either kernel set. Order 8, the papers', was not run here: extrapolated it
is about two hours under `rust` against an overnight run with the pool, and memory is the thing to watch — the peak
RSS at order 7 already reached 4.3 GB in one run (it depends on how many of the largest clusters run at once), and
this machine has 16 GB, so cap `n_workers` there.

**Gate** (2026-10-29 11:51–12:05, workspace root, `MAGCALC_NO_GUI=1 python -m pytest -m "" -rs`, after merging:
pyMagCalc `ff5c014`, rsMagCalc `9061710` rebuilt and installed (`pip install -e . --no-build-isolation`, stroff
8-aligned), fMagCalc `4a5ff4b`, magpipe `e92b2c7`): **2435 passed, 3 failed, 2 skipped, 13 min 37 s**. The three
failures were magpipe's live Materials Project tests failing to resolve `api.materialsproject.org`, and all three
pass on rerun, so **2438 pass**: the 2405 of the Luttinger–Tisza gate plus the 33 added here (7 in
`test_dimer_series.py`, 7 in `test_kernels_dimer_series.py`, 19 in rsMagCalc `test_dimer_series_parity.py`). The
two skips are the structural pair. Before merging, the same suite ran from a mirror of the root with the worktrees
on `PYTHONPATH`; every added test passed there too.

### Gapless and degenerate S(Q,ω) under `backend: fortran`, 2026-10-28 — two Fortran defects, pinned to Sunny and a closed form

OPEN_WORK item 2 read: "fMagCalc zeroes the intensity of modes below 1e-3 meV and NumPy
does not, so a gapless model's powder average depends on `backend:`; decide which is
right, against an independent reference, before changing either." Two references
decided it, and they found two Fortran defects. The second was much larger.

**1. The 1e-3 meV cut was wrong on the exact-H path.** The ferromagnetic chain's
one-magnon intensity is (S/2)(1 + Q̂_z²) exactly, at every q however small its energy.

* NumPy reproduces it down to 4e-11 meV and gives 0 at the exact zero, where H(q) = 0.
* Fortran zeroed every mode from 4e-11 to 9.9e-4 meV.

The cut came from fMagCalc `30856dc`, which fought streaks on `run_sqw_model`, the path
that rebuilds H(q) from bonds. That rebuild differs from the exact H by ~1e-14, enough
to light up a Goldstone mode's ill-conditioned eigenvector. pyMagCalc calls `run_sqw`
with the exact H instead. There the pseudo-norm rule both backends share already zeroes
a genuinely zero mode: α = 0 when |N_ii| < 1e-12, where N_ii is the eigenvector's
para-unitary norm. So the right treatment is NumPy's. `sqw_intensity` now takes the cut
as an argument: 0 from `run_sqw`, 1e-3 meV from `run_sqw_model`, whose streak test
still passes. The convention is that an exactly-zero mode carries no inelastic
intensity, which is also a set of measure zero in any powder integral.

**2. Degenerate magnons lost all their intensity, at energies far from zero.** This came
out of checking item 1 on a Néel chain, where the two magnons are degenerate at every q.

* After the cut was gone, Fortran still returned exactly 0 at 172 of 1495 powder points
  above 0.2 meV, with energies of 0.06–0.36 meV. NumPy matched Sunny at all 1495.
* The cause was in `kkd_matrix`'s ±q eigenvector matching. Fortran paired each +q
  vector with the single −q column of best overlap. Inside a degenerate pair, the two
  diagonalizations mix the subspace by an arbitrary unitary, so no column need overlap
  enough. The mode then stayed unmatched, with α(−q) = 0 and no intensity.
* pyMagCalc fixed exactly this on 2026-07-13 (`5449196`): it projects the +q target onto
  the degenerate −q subspace. Its docstring says that without it, degenerate bands
  "silently lose all S(q,w) weight". The Fortran port predated that and was never
  updated. `match_reorder` now carries the same algorithm, line for line.

Every collinear antiferromagnet run under `backend: fortran` was affected, in its S(Q,ω)
maps as well as its powder averages, at the fraction of q points where the two LAPACK
bases differ. The original library shows the same zeroed points. With the fix, the
degenerate-pair totals match NumPy to 1.9e-8 at all 1600 points of 8 powder shells,
the near-zero ones included.

**Oracles** (`tests/test_gapless_backends.py`, both backends through the `backend`
fixture):

* the FM chain's closed form at six energies from 4e-11 meV up;
* the exact zero giving 0;
* the Néel chain's degenerate-pair totals against **Sunny 0.8.1** at 400 points on two
  powder shells (`tests/data_afm_chain_sunny.json`, from
  `tests/afm_chain_sunny_reference.jl`). Above 0.2 meV both backends agree with Sunny to
  4.7e-7 relative. The cut-off exists because Sunny's 1e-8 regularization moves
  energies near the Goldstone points;
* gapless powder averages, FM and Néel, agreeing between the backends to 1e-8.

Only a degenerate group's total is an observable. How a pair splits its weight differs
by up to 3.6 between the backends and depends on each LAPACK's basis, so every
comparison is of totals. On the original library the three Fortran cases fail. fMagCalc
gets its own pin, `tests/test_degenerate_and_gapless.py`, against pyMagCalc as the
oracle. Its other tests, the streak regression and the KFe3J parity at 1e-8 among them,
pass unchanged (14 pytest, 7/7 ctest).

### Bond builders: the (site, image) loop skipped, 2026-10-28 — SCGA's exchange cache 9× at 128 sites, λ bit-identical

OPEN_WORK item 1. SCGA's `_exchange_cache` called `sun.entangled._pair_matrix` for every
(site, image) pair: 442k pairs at 128 sites, for 768 bonds. Each call paid SymPy's
`is_zero_matrix` four times, mostly on the one shared zero matrix the model builder puts
in every unset DM and Kex entry. That cost 2.7–6.9 s, depending on load. The same loop
ran in `thermal_mc.build_supercell`, the entangled-unit builder and the dimer series.

`sun.entangled.iter_pair_matrices` replaces it in all four:

* candidate pairs come from the nonzero entries of Jex, via SymPy's sparse `todok`;
* DM and Kex entries pass the same test `_pair_matrix` applies, memoized per object,
  with the shared zero skipped by identity;
* `_pair_matrix` then decides each candidate as before, so terms that cancel are
  still dropped;
* output is row-major, the loop's order.

The bond lists are therefore identical, entry for entry. Checked on all 61 buildable
shipped example configs, 118 560 pairs and 3598 bonds, with exceptions compared too, and
pinned in `tests/test_pair_iteration.py` against the loop itself. That test covers exact
and float exchange, DM-only and Kex-only pairs, a list-valued Kex, `None`, the shared zero,
an exactly cancelling pair, a NumPy Jex, and KFe3J's DM end to end. Removing the DM/Kex
scan, or the sort, fails it.

On the 128-site pyrochlore (S07's cell, 2×2×2, NN J only, nq = 6, kT = 0.05), times are
in seconds:

| step | before | after |
|---|---|---|
| `_exchange_cache` | 2.71 | 0.29 |
| `solve_lambda`, `kernels: numpy` | 9.7 | 7.4 |
| `solve_lambda`, `kernels: rust` | 3.4 | 1.2 |

λ = 0.059040732407190946 under both kernels, before and after (`==`). What remains of the
cache is the model's own `spin_interactions`, about 0.25 s.

**Gate:** (2026-10-28, workspace root, `MAGCALC_NO_GUI=1 python -m pytest -m "" -rs`, pyMagCalc `9b70cf5` plus both changes, fMagCalc `4a5ff4b` (the same source, built and installed), rsMagCalc `1d2532d`, magpipe 283f694): **2345 passed, 2 skipped, 0 failed, 16 min 32 s**. The 14 added tests are `tests/test_pair_iteration.py` (4), `tests/test_gapless_backends.py` (8) and fMagCalc's `tests/test_degenerate_and_gapless.py` (2). The two skips are the structural pair.

### anneal on K1's dense model, 2026-10-28 — 19–185× per run, all starts in parallel

`annealing.anneal`, the recommended ground-state search, was a per-spin Python
Metropolis loop: 0.36 s per 2000-sweep run at 16 sites, 2.8 s at 128, times
`num_starts`. Under `calculation: {kernels: rust}` it now runs in rsMagCalc
(`rsmagcalc.anneal`), together with its steepest-descent polish
(`rsmagcalc.steepest_descent`). `anneal_runs` sends every start in ONE call that runs
them in parallel on every core. `_minimize_anneal` uses it, `minimize_energy` takes
`kernels=`, and the runner passes `calculation.kernels` through.
`minimization.n_workers` keeps its meaning for the multistart methods' process pool
and is not used here.

**It reuses K1's dense model, not K1's `sweep`.** The move energy, the accept-and-
update-gradient step, the random spins and the per-site lengths are K1's. The move loop
is the annealer's own, because it differs from `thermal_mc._sweep`:

* a site drawn with replacement n times per temperature step;
* Sunny's `LocalSampler` mix of 0.5 uniform, 0.1 flip and 0.4 delta moves, with the
  delta width adapted toward 50 % acceptance after every step;
* the geometric schedule from `_schedule`, now shared by both kernel sets;
* tracking of the best state seen, with its energy recomputed from scratch at the end;
* the keep-if-better polish.

Calling `sweep` once per temperature would have dropped the delta move, which is what
keeps acceptance up at the cold end, and the best-state tracking.

**The NumPy path is bit-identical to before.** It was compared against the committed
module for uniform and mixed spins, and `anneal_runs` returns exactly the per-seed
`anneal` calls. The Rust anneal draws from a different stream, so a seed gives a
different, equally valid sample, as for every K1 kernel. Run r owns the stream of
`seeds[r]`, so a batch equals the per-seed calls whatever the thread count.

**Oracles.** They now run under both kernel sets (the `kernels` fixture):

* `tests/test_annealing.py`: the Néel chain's exact energy, |m_i| = S, all four methods
  through `minimize_energy`, the guard check on the result, reproducibility across
  seeds, the S06 closed-form canted minimum, the polish never raising the energy, and
  steepest descent staying monotone with a large on-site term;
* `tests/test_mixed_spin_classical.py`: the closed-triangle energy and law-of-cosines
  angles of the (1, 1, ½) trimer, the per-site lengths, and a new 24-start
  `minimize_energy` run;
* the runner test, which now also asserts that `calculation.kernels` reaches the
  annealer.

New tests in rsMagCalc's `tests/test_classical_parity.py`:

* `steepest_descent` matches the NumPy loop to 1e-12, with mixed spins and a large
  random on-site block, and so does `max_iter`;
* `anneal` is deterministic, batch-independent and thread-independent;
* it honours `m0` (run 0 only), and with zero sweeps its polish equals the steepest
  descent;
* the mixed-spin trimer's closed form is reached on all 8 runs by the Metropolis loop
  ALONE, with the polish off;
* bad input is refused.

pyMagCalc pins the fall-back: a raising kernel gives exactly the NumPy result, and the
failure is logged.

**Mutation-checked**, rebuilding each time:

* an inverted acceptance rule fails the Metropolis-only trimer test;
* a schedule that heats instead of cooling fails a test;
* dropping the on-site term from the steepest-descent field fails 4.

Forcing the polish to be taken unconditionally fails nothing, and that is correct: the
polish starts from the annealed state and returns the best state it visits, so it can
never come back higher. The keep-if-better check is redundant in both implementations.

**Timing** (10 cores, 2026-10-28; about three cores busy with other desktop
processes). The cell is an AFM chain built by `thermal_mc.build_supercell`, the runs
are 2000 sweeps with the polish, and "before" is the committed NumPy annealer:

| cell | one run, before → after | 4 runs, before → after (parallel) |
|---|---|---|
| 16 sites | 0.363 s → **2.0 ms** (185×) | 1.33 s → **2.8 ms** (478×) |
| 128 sites | 2.84 s → **0.147 s** (19×) | 10.9 s → **0.184 s** (60×) |

End to end, `examples/sunny_tutorials/S07_dipole_dipole/config.yaml` (16 sites, 12
starts × 3000 sweeps) went from 51.7 s to 46.1 s wall, with the annealing step going
from about 6 s to under 1 s by the log's timestamps. The rest is the SymPy build and
the dispersion. Both runs reach E_min = −6.974504 meV, 7 of 12 starts under NumPy and
4 of 12 under Rust. **Their bands along the Γ–X–W–L–K path differ by up to 0.093 meV,
and that is a domain, not an error.** The dipole-selected pyrochlore state comes in
cubic domains, the path picks out fixed directions, and the two random streams settle
in different domains. Checked over all 48 operations g of O_h, using the runner's own
two calculators: E_NumPy(g·q) equals E_Rust(q) to 2.5e-7 meV, which is the ground state's
convergence, for g the three-fold rotation about [111]. 16 of the 48 operations agree to
under 4e-7, a whole coset, as a domain relation implies. The other 32, the identity
among them, all leave at least 9.3e-2.

**Gate:** (2026-10-28, workspace root, `MAGCALC_NO_GUI=1 python -m pytest -m "" -rs`, pyMagCalc `b35e804` plus this change, rsMagCalc `1d2532d` (built from the same source), fMagCalc `eae20e3`, magpipe 283f694; about three cores busy with other desktop processes): **2331 passed, 2 skipped, 0 failed, 14 min 33 s**. The 33 added tests are the `rust` parametrizations of the annealer oracles, the new pyMagCalc tests and rsMagCalc's 7. The two skips are the structural pair.

### 1/S corrections: the k-mesh batched, 2026-10-28 — 25–27×, and bounded memory

`corrections.compute_corrections` built H(q) for the WHOLE mesh in one stack, then
looped over it in Python. At each k it ran a general eigensolve, `scipy.linalg.eigvals`,
only to report |Im ω|, and then the Colpa transform: a Cholesky, a Hermitian eigensolve
and an explicit inverse. At 16 sites that was 704 µs per k, and on the 64³ mesh §5d
recommends for the moment reduction it meant about 185 s. The single stack
also peaked near 8.6 GB there, more than half of this machine's 16 GB.

Two changes, in `_chunk_sums`:

* **No general eigensolve where the Cholesky succeeds.** With H₂ = K†K, g·H₂ = g·K†K is
  similar to the Hermitian K g K† = W, so its eigenvalues are exactly real wherever the
  Cholesky succeeds. |Im ω| can be nonzero only at a k where it fails.
* **The Colpa transform runs batched, chunk by chunk, on threads.** The steps are a
  Cholesky, the eigensolve of W, and one solve for the β† columns of T = K⁻¹UE, done
  over chunks of about 8 MB of H(q). Chunks are spread over `n_threads` threads (the
  default is every core; config key `corrections.n_threads`), since LAPACK releases the
  GIL. Each H(q) chunk is built just before it is solved. The per-chunk sums are combined
  in mesh order, so the thread count does not change the answer.

Any chunk where that argument does not hold goes through the original loop, kept
verbatim as `_chunk_sums_per_k`. That covers a Cholesky failing anywhere in the chunk,
and an H₂ that is not Hermitian to 1e-10 relative (the old eigensolve ran on the raw H).
The failure count, |Im ω|, the jitter retry, the warning and the refusal are therefore
exactly what they were.

**fMagCalc and K3 were not used, though OPEN_WORK proposed them.** `fmagcalc.run_dispersion`
returns energies only, and the moment reduction needs the Bogoliubov eigenvectors. There
is also no SU(N) corrections path for K3's `SunModel` to serve: `compute_corrections`
takes the dipole `MagCalc`.

**Oracles.** The physics pins are unchanged and now run through the batched path: the
ferromagnet's exact zero, and the S = ½ square-lattice dE = −0.157947 J and dS → 0.1966.
New tests in `tests/test_corrections.py`:

* the batched sums equal the original loop's on an 8-site supercell (1e-12);
* complex one-mode blocks are checked against the closed form ω = √(A² − |B|²),
  ⟨a†a⟩ = (A/ω − 1)/2;
* a chunk holding one unstable k reports the same failure count and |Im ω| as the old
  loop, and so does a non-Hermitian chunk;
* the thread count and the chunk size don't change the result (1e-14);
* the runner passes `n_threads` through;
* (slow) KFe3J, a 9-site non-collinear state with DM and complex H(q): the batched sums
  match the loop to 1e-16 relative in the energy sum and 2e-13 in the densities.

**Mutation-checked:** taking the β columns instead of the β† ones, solving with L instead
of L†, or skipping the Hermiticity check each fails the new tests. The second also fails
the textbook pins.

**Timing** (10 cores, 2026-10-28, no pytest live; about three cores busy with desktop
apps). The cell is the 2-site square-lattice antiferromagnet on a magnetic supercell,
with the lambdify done before the clock starts:

| cell, mesh | before | after | peak RSS before → after |
|---|---|---|---|
| 16 sites, 32³ | 23.1 s (704 µs/k) | **0.94 s** (25×; 4.3 s, 5.4×, on one thread) | 1.15 → 0.74 GB |
| 16 sites, 64³ | ≈ 185 s, not run (≈ 8.6 GB) | **5.2 s** | → 0.73 GB |
| 64 sites, 12³ | 39.6 s (22.9 ms/k) | **1.44 s** (27×) | 1.05 → 0.79 GB |

The before and after results agree to 5e-15 in dE and dS.

**Gate:** (2026-10-28, ended 09:00, workspace root, `MAGCALC_NO_GUI=1 python -m pytest -m "" -rs`, pyMagCalc `b7421b7` plus this change, rsMagCalc `ff463f3`, fMagCalc `eae20e3`, magpipe 283f694): **2295 passed, 3 failed, 2 skipped, 17 min 45 s**. The three failures were magpipe's live Materials Project tests (`magpipe/tests/test_mp.py`), each a DNS failure resolving api.materialsproject.org during the run; all three pass on rerun. That makes **2298 passing**, the previous 2292 plus this change's 6 tests. The two skips are the structural pair.

### Ewald dipolar sum: q-independent terms computed once, 2026-10-27 — 63–84× per q, A(q) pinned to Sunny

`ewald.dipole_ewald_at_q` ran a Python double loop over every site pair at every q, and
it runs under `backend: fortran` as well, because `_build_h_stack` adds the dipolar term
in Python. So a dipolar model got no compiled speedup at all; at 64 sites a 200-q path
cost about 3 min, ±q stacks included. Most of that work did not depend on q. The real-
space part depends on q only through exp(i q·L) over the lattice images L. Which images
lie inside the cutoff, and the 3×3 tensor each contributes to each pair, are fixed.
`ewald.DipoleEwald` now builds them once per cell, as a sparse (n²·9 × N_images)
matrix, so each q is one sparse matrix-vector product. The reciprocal part does depend
on q (k = G + q), but its pair dependence factorizes, exp(−ik·(r_j − r_i)) = conj(P_ki)
P_kj with P = exp(−ik·r). It is therefore six dense (n × N_k)(N_k × n) products
instead of n² Python iterations. `exchange_from_A` lost its per-pair loop the same
way. `core._ewald_A` keeps one `DipoleEwald` per cell, rebuilt only if the lattice, the
sites or `demag` change. The arithmetic is the old loop's, term for term; only the
summation order differs. `dipole_ewald_at_q` keeps its signature as a one-off wrapper.

**Pinned to Sunny's tensor, not only to its bands.** Until now the Sunny test
(`test_ewald_matches_sunny`) compared band energies on one chain at `atol=1e-5`.
`tests/data_ewald_sunny.json` now holds Sunny 0.8.1's own
`precompute_dipole_ewald_at_wavevector`, from `tests/ewald_sunny_reference.jl`, for
four cells: monoclinic 3-site, triclinic 2-site with a non-trivial `demag`, the 1-site
chain, and an irregular 8-site cell. The q points are Γ, generic points, a zone-
boundary point, and points outside the first zone. `tests/test_ewald_sunny_tensor.py`
compares A_ij(q) element by element against a bar of 1e-8 of the largest element.
Measured: 1.2e-15, 2.0e-15, 8.8e-15 and 1.8e-15. The OLD loop met the same reference to
1.5e-15 before the change. **Mutation-checked:** swapping i and j in the pair index,
dropping the gauge factor, or conjugating the wrong phase matrix each fails it, at an
error of about 2.

**Timing** (M1 Pro, 10 cores, 2026-10-27, no pytest live; 1-min load 8–15 for the
"before" runs and 3.5–4 for the "after" runs, CPU ~75 % idle). Simple-cubic cells,
a = 4 Å; "before" is one q, "after" averages 50 random q:

| cell | before, per q | after: build once, then per q | 200-q path, before → after |
|---|---|---|---|
| 8 sites | 13.8 ms | 3.3 ms, then **0.22 ms** (63×) | 2.8 s → 0.05 s |
| 64 sites | 455 ms | 140 ms, then **5.4 ms** (84×) | 91 s → 1.2 s |

End to end, the shipped `examples/sunny_tutorials/S07_dipole_dipole/config.yaml` (16
sites, 496 q, under `/usr/bin/time -p magcalc run`) went from 72 s to 54 s wall. Under
cProfile, `_ewald_h_stack` went from 20.3 s to 3.4 s; the rest is the SymPy build
(13.9 s) and the 12-start anneal (7.6 s), which this change does not touch. Its 496 × 16
band energies are identical to before, to 3.4e-15 meV.

**Gate** (2026-10-27, ended 14:58, workspace root, `MAGCALC_NO_GUI=1 python -m pytest
-m "" -rs`, pyMagCalc `49aa5b4` plus this change, rsMagCalc `ff463f3`, fMagCalc
`eae20e3`, magpipe 283f694): **2292 passed, 2 skipped, 0 failed, 16 min 54 s**. The 17
added tests are this change's 9 and the wiki generator's 8; the two skips are the
structural pair.

### K6, 2026-10-27 — mixed-spin dipole S(Q,ω) on fMagCalc (no Rust kernel)

The optional K6 of `RUST_KERNELS_PLAN.md`. `backend: fortran` used to fall back to the NumPy pool for every
mixed-spin model, because fMagCalc applied one global √(S/2) where `linalg._spin_prefactor_vector` applies
√(S_i/2) per site. **It went into fMagCalc, not rsMagCalc**, after a profile and a test on paper (both in the
hand-over note `k6-mixed-spin-sqw-handover.md`):

* where the time goes, per q on a 20-site mixed YIG (SW21 with the 12 d sites at S = 0.8): `KKdMatrix` 81 % (half
  LAPACK `eig`, half the Python ±q matching `_match_block`), H(±q) 12 %, the contraction 7 %; on a 2-site chain the
  Pool spawn and per-worker lambdify (1.3–2 s) is the whole run. A compiled kernel's ceiling is the same whichever
  language, and fMagCalc already has a `KKdMatrix` pinned to these Python helpers;
* the prefactor is a per-site ROW scale of K and Kd, and the contraction already weights site i's rows by ff_i —
  so the UNMODIFIED Fortran kernel fed `ff_i·√(S_i/S)` already matched NumPy (1.6e-15 on the chain, 3.6e-11 on
  YIG, against 1.2 and 2.0 absolute errors with the global factor). A second compiled `KKdMatrix` in Rust, for
  mixed spin only, would have been the duplicate §0 of the plan forbids.

**The change.** fMagCalc: `kkd_matrix` takes `spins(n)` and scales row block 3i−2..3i by `sqrt(spins(i)/2)` after
the matmuls; `run_sqw` / `run_sqw_model` / the C API (`magcalc_sqw_c`, `magcalc_sqw_model_c`: the third argument is
now a pointer to n doubles, not a scalar by value) carry it; the Python wrappers take `spins=None` (default: n copies
of S) and `powder_average(..., spins=)`; the subprocess transport, whose file format holds one S, raises
`NotImplementedError` on a mixed vector. pyMagCalc: `_calculate_sqw_fortran` passes `spin_mags`, and the
mixed-spin fallback in `_calculate_sqw_single_domain` is gone. An fMagCalc that predates `spins=` (detected by
signature) still falls back for a mixed-spin model, with a warning saying to reinstall. The ground-state guards are
untouched; S(Q,ω) runs after them as before.

**Bit-identity, uniform spin.** With every S_i equal the row scale multiplies by the same double as the old
scalar product, so nothing that is not mixed moves. Checked, not assumed: 26 arrays (fixture `run_sqw` on KFe3J
and the 9-site GUI model, `run_sqw_model` on both, `powder_average`, and pyMagCalc `backend: fortran` end to end on
KFe3J, a KFe3J powder, a uniform S = 3/2 chain and SW21 YIG — energies, intensities, eigenvalues, info) recorded
from fMagCalc HEAD before the change and compared with `np.array_equal` after, on BOTH builds (the
`-march=native` CMake dev build and the portable pip wheel): **26/26 bit-identical** on each.

**Oracles.**
* New, independent: the alternating-spin AFM chain in closed form (`tests/test_mixed_spin_sqw.py`). With
  A = 2JS_B, B = 2JS_A, C = 2J√(S_AS_B)cos πh and tanh 2θ = 2C/(A+B): ω = E ± (A−B)/2, and
  I_α = ½(f_A√S_A cosh θ − f_B√S_B sinh θ)²(1 + q̂_z²), I_β with A↔B — per chemical cell, the transverse
  correlation of each mode under the perp projector. It weights the sublattices by √S_i separately, so a collapsed
  spin length fails it outright. NumPy and Fortran both match it to 1e-10, energies AND per-band intensities, with
  form factors and q off the chain axis. (Derived here 2026-10-27; it agreed with the NumPy path, which was
  already pinned to Sunny's absolute scale and to the decoupled-sublattice identity, to every printed digit on the
  first run.)
* Fortran vs NumPy on the mixed YIG (degenerate bands: per band where non-degenerate, per degenerate-group sum
  otherwise, 1e-10; slow) and on a mixed-spin powder average (gapped by an easy axis, see below).
* `test_intensity_layer.py`'s two mixed-spin identities now take the `backend` fixture (`tests/conftest.py`,
  `numpy` | `fortran`, the latter skipping by name when fMagCalc is absent or pre-K6).
* fMagCalc `tests/test_sqw_per_site_spins.py`: the uniform vector bit-identical to the scalar call (all three
  entry points), the vector equal to folding √(S_i/S) into ff (an exact row-scale identity, 1e-12), parity with
  pyMagCalc's NumPy path on the mixed chain (exact-H 1e-10; bond-model path at its 1e-8/1e-7 reconstruction floor),
  bad vectors, the subprocess refusal. Its existing parity tests are unchanged and pass.

**One difference the parity tests had to step around, pre-existing and not K6's** (resolved 2026-10-28, see
"Gapless and degenerate S(Q,ω) under `backend: fortran`"): fMagCalc zeroed the intensity
of any mode with |E| < 1e-3 (the Goldstone-streak fix, fMagCalc `30856dc`); NumPy does not. A gapless model's
powder average therefore differs by the Goldstone weight on every sphere (every sphere holds h = 0), so the
mixed-spin powder parity runs on an easy-axis-gapped chain.

**Timing** (`python time_k6.py <ferri|yig> <nq> 3` from `pyMagCalc/`: `calculate_sqw` on the same model and q list
under `backend: numpy` — exactly the path a mixed-spin `backend: fortran` took before — then `fortran`; 3 runs each,
the first Fortran run pays the one-off lambdify of the vectorized H stack. 2026-10-27 11:47, M1 Pro, 1-min load
7.9–8.4 at the start of each, CPU 62–69 % idle; desktop apps, OneDrive and the hourly rclone mirror running, no
other compute job):

| model | q | NumPy pool | Fortran, first | Fortran, warm |
|---|---|---|---|---|
| S = (½, 2) chain, 2 sites | 300 | 1.16 s | 0.10 s (12×) | 0.004 s |
| mixed YIG, 20 sites | 300 | 2.14 s | 0.39 s (5.5×) | 0.074 s (29×) |
| mixed YIG, 20 sites | 3000 | 4.41 s | 1.10 s (4.0×) | 0.75 s (5.9×) |

Energies agree with NumPy to 2.6e-13 in every run.

Gated 2026-10-27 11:56–12:12 from the workspace root (`MAGCALC_NO_GUI=1 python -m pytest -m "" -rs`): **2275 passed, 2 skipped, 0 failed, 16 min 01 s** — 15 more than the K5 gate, all K6's (6 in `test_mixed_spin_sqw.py`, the 2 `fortran` cases of `test_intensity_layer.py`, 2 in `test_ewald.py`, 5 in fMagCalc `test_sqw_per_site_spins.py`); the same two structural skips (`seekpath` absent, ALPS present), no crashed xdist worker. 1-min load 9.7 at the start (CPU 71 % idle, after waiting out another session's pytest), 159 at the end (the gate's own workers).

**Ewald under `backend: fortran` is carried, and now pinned.** Both Fortran routes take H(q) from
`MagCalc._build_h_stack`, which adds `_ewald_h_stack` (−q for the −q stack); pyMagCalc never uses fMagCalc's
bond-model `run_sqw_model`, the one route that would lack A(q). (The K6 STEP-1 note briefly claimed the opposite,
from reading that function only up to its lambdify — wrong, and corrected.) `tests/test_ewald.py::
test_fortran_backend_carries_the_ewald_term` runs a dipolar FM chain, uniform and mixed spin, moments along the
chain (the dipolar ground state, no imaginary mode), under both backends: dispersion and S(Q,ω) energies at 1e-10,
intensities per degenerate group at 1e-10, and Ewald moving the bands by > 1e-3 so a stack that dropped it fails.

### Rust kernels K5, 2026-10-27 — the SCGA loops behind `kernels: rust`

M5 of `RUST_KERNELS_PLAN.md`. rsMagCalc `1288d66` (`src/scga.rs`, `ScgaModel`, packed once from `scga._exchange_cache`
and cached on it) carries the three per-q loops of `magcalc/scga.py`, each with rayon over q: the J(q) eigenvalues over
the λ mesh (`solve_lambda`), the per-site traces Tr₃[(λ+J(q))⁻¹]_ii over the same mesh
(`sublattice_sum_rule_residuals`, the inequivalent-sublattice check), and the 3×3 correlation tensor C(q) of
`scga_intensities` before the cross-section contraction. Stays in NumPy: the brentq root solve, its bracketing, the
refusal and its message, the form factors (evaluated per |q| in Python and passed in, so the kernel cannot drift from
`magcalc.form_factors`) and `_contract`. `solve_lambda`, `sublattice_sum_rule_residuals`, `scga_intensities` and
`compute_scga` take `kernels` / `n_workers`; the runner's `scga` task passes the resolved `calculation.kernels` and
`scga.n_workers` (else `calculation.n_workers`); a failure at packing or at any call is logged by `kernels.fell_back`
and that loop runs in NumPy. The NumPy tensor body is now its own function, `scga.correlation_tensor` (same
arithmetic), so the parity tests compare against the shipped loop rather than a copy of it.

**Profiled before it was written** (cProfile, `solve_lambda` + `scga_intensities` on 400 q; `<this session's
scratch>/prof.py`). On the test models SCGA is cheap: square (N = 1, 20² mesh) 13 + 23 ms, kagome (N = 3, 25²) 96 + 52
ms, chain (2000 q) 40 + 22 ms — nothing a kernel can buy measurably there. On a 16-site conventional pyrochlore cell
(16³ = 4096 mesh q) `solve_lambda` is 4.25 s, of which 3.06 s is the Python bond loop of `fourier_exchange_matrix`,
0.55 s LAPACK `eigvalsh`, 0.29 s `inv` (+ 0.13 s trace calls): **~79 % Python-bound**; the q-map is ~90 % Python (the
N² loop over 3×3 blocks). So the kernel is not threading-only on cells of that size. It becomes LAPACK-bound as N
grows: at 128 sites (2×2×2 pyrochlore, 3N = 384) the mesh is 1.37 s `eigvalsh` against 0.21 s of J(q) build, and
there the eigenvalue gain IS only threads (below).

**Exact where it can be.** J(q) is summed in the cache's bond order and symmetrized as 0.5·(J + J^H), and the
eigenvalues are `zheevd` jobz = 'N' on the lower triangle of the column-major J(q) with the QUERIED workspace (the
blocking of the tridiagonal reduction depends on it) — the call `eigvalsh` makes. Result: **J(q) and its mesh
eigenvalues are bit-identical to NumPy's** on every fixture, so `solve_lambda` returns the same λ to the last bit
(asserted `==` in `test_kernels_scga.py` and through a config run). The sum rule is solved on λI + J(q), Hermitian
positive definite for λ above the mesh's λ_min, so the solve is a Cholesky (`zpotrf`; `zpotrs` for C(q) = kT W^H
(λ+J)⁻¹ W with W the 3N×3 stack f_j e^{iq·r_j} g_jᵀ — 3 right-hand sides instead of an explicit inverse; `ztrtri` for
the inverse's diagonal), not NumPy's LU: tensors agree to ≤ 2e-15 relative on the ≤ 16-site fixtures (3e-14 at 128
sites), residuals to 4e-16. Off the mesh, or below λ_min, λ + J(q) can be indefinite; then the kernel uses NumPy's own
LU (`zgesv`) on the same matrix, pinned at 1e-10.

**One test assertion changed, and why.** `test_the_refusal_reports_the_actual_violation` checked the message for
"+262"; the residual of that mixed-spin chain is 2.625 to 1e-15, exactly a %.3g half-way point, and NumPy's sum lands
one ulp below ("+262") while the kernel's lands one ulp above ("+263"). The value itself is still pinned at 1e-6, under
both kernels; the string check accepts either rounding.

**Pinned to.** rsMagCalc `tests/test_scga_parity.py` (75): on the chain, square, square + DM (complex J(q)), the
Sunny-compared kagome, a kagome with an ANISOTROPIC off-diagonal g and Cu²⁺ form factors, and the 16-site pyrochlore —
J(q) and the mesh eigenvalues at 1e-12 and bit-equal, λ / λ_min / the total residual from `solve_lambda(kernels=
"rust")` at 1e-12, the per-site residuals at λ and λ + 0.37 at 1e-12 (directly and through pyMagCalc), C(q) against
`correlation_tensor` at 1e-12 with and without `apply_g`, the indefinite LU branch, S(q) end to end in `perp`, `trace`,
`zz`, threads not changing a bit, empty batches, loud shape errors, and a guard that the g / form-factor fixture really
differs from g = 2, f = 1. Every SCGA oracle runs under both kernel sets through the `kernels` fixture:
`tests/test_scga.py` (the chain closed form at three kT, high-T flattening, Sunny SCGA on square and kagome, the
refusal and its reported residuals, the 1e-12 per-site rule on kagome) plus two exact identities added with K5 — **the
free-spin sum rule** (J = 0: λ = 3kT/S² and the bare S(q) = N S² at every q, to 1e-12) and **the high-T Curie limit**
(square lattice at kT = 200 J: S(q)/S² = 1 − S² J_q/(3kT) to 1e-4, with the correction itself > 1e-3 so it is seen);
and the SCGA branch of `tests/test_hamiltonian_terms.py`, which now also checks that a parameterised g-tensor gives the
numeric tensor's S(q) to 1e-12 and differs from g = 2's. `tests/test_kernels_scga.py` (11, fast) pins the wiring: every
entry point reaching the kernel (a call-counting spy) with λ `==` NumPy's; `lam=None` solving through the kernel; the
empty q batch under both; a failure injected in each of the three calls and in packing, each logged with NumPy
answering; the packed model reused and repacked when the bond list changes; the refusal unchanged under rust (and
raised before any S(q)); a config run with `tasks: {scga: true}`, `kernels: rust`, `scga.n_workers: 2` whose log names
the kernel and whose saved λ and S(q) equal the `numpy` run's.

**Timing** (M1 Pro, 10 cores, 2026-10-27 09:59–10:03, no pytest or `magcalc run` live; 1-min load 4–41 from desktop
apps, CPU 74–79 % idle, 10.1 of 11 GB swap in use). `<this session's scratch>/bench.py <model> [nq] [n_qmap]
[repeats]`, best of 3 (1 for 128 sites), two runs agreeing to a few %; q-map = 1681 q of an (h, k, 0.1) grid (400 at 128
sites); "rust all" = default pool:

| model (mesh) | | `solve_lambda` | q-map |
|---|---|---|---|
| square, N = 1 (20²) | numpy / rust 1 thr / rust all | 8.1 / 1.3 / 1.3 ms | 66 / 16 / 16 ms |
| kagome, N = 3 (25²) | numpy / rust 1 thr / rust all | 73 / 8.4 / 6.5 ms | 180 / 17.6 / 16.3 ms |
| pyrochlore, N = 16 (16³) | numpy / rust 1 thr / rust all | **3935 / 797 / 172 ms** | **2464 / 52 / 27 ms** |
| pyrochlore 2×2×2, N = 128 (4³) | numpy / rust 1 thr / rust all | 4896 / 4498 / 3043 ms | **28 914 / 772 / 183 ms** |

What each number is. At N = 16 the one-thread gain in `solve_lambda` (4.9×) is the Python J(q) build removed; what is
left on one thread is LAPACK (`eigvalsh` 558 ms, the traces 166 ms), and threads add 4.6×. At N = 128 `solve_lambda`
gains little because **2.8 s of it is `_exchange_cache`**, the pre-existing O(N·N_OUC) loop of `_pair_matrix` SymPy
zero-tests (442 368 calls), which K5 does not replace; the mesh loops themselves are 1.37 s + 0.31 s in NumPy against
0.21 s + 0.04 s in the kernel on all threads (eigenvalues: threads only, 6.5×). The q-map's N² block loop is the
Python-bound part at every size (166× at N = 128); on the small models what is left of it is Python — the
per-q `_contract` (14 ms for 1681 q) and form factors, shared with the NumPy path on purpose. λ is identical and S(q)
agrees to ≤ 3e-14 in every row.

**Gate** (2026-10-27 10:05–10:19, workspace root, `MAGCALC_NO_GUI=1 python -m pytest -m "" -rs`, pyMagCalc `9d5d943` plus the uncommitted K5 wiring, rsMagCalc `406345c` installed, magpipe 283f694): **2260 passed, 2 skipped, 0 failed, 13 min 35 s** — 100 more than the sparse-guard gate, all K5's (75 in rsMagCalc `test_scga_parity.py`, 11 in `test_kernels_scga.py`, 13 in `test_scga.py` (its 9 `rust` parametrizations + the two new identities under both), 1 in `test_hamiltonian_terms.py`); the same two structural skips (`seekpath` absent, ALPS present), no crashed xdist worker. 1-min load 5.7 at the start (CPU 71 % idle), 35 (5-min 156, the gate's own workers) at the end.

### Rust kernels: guard 3 on a SPARSE Cholesky, 2026-10-27 — same verdicts, 37 ms → 0.6 ms per q

K4 left guard 3 (H₂(q) + ε·I ≻ 0 at every q the KPM computes, ε = `h2_rel_tolerance` × the q-independent
`_reference_h2_scale()`) as 86 % of a KPM q: one DENSE `zpotrf` of the 2D × 2D matrix per q, ~1.9 G complex flops at
2D = 1800 on a matrix that is 0.7 % nonzero. rsMagCalc `src/sparse_guard.rs` now decides the same question with a
multifrontal sparse Cholesky, and `SunModel.stable` picks it per model. Nothing changes on the pyMagCalc side but a
docstring: `is_stable_at` / `unstable_indices` / `assert_stable` call the same `stable(qs, shift)`.

**Profiled before it was written** (S09 L = 30, H₂ at a generic q; symbolic fill counted in Python): natural order
56.6 % fill, 5.9e8 flops; RCM + banded `zpbtrf` bandwidth 178, 2.4e7 flops, measured 9.8 ms vs dense 72 ms on one
thread (7×); **minimum degree 90 115 nonzeros in L (5.6 % of a dense triangle), 7.6e6 flops — 250× fewer than dense.**
So the kernel uses minimum degree, not a band.

**Exact, not approximate — the argument, then the oracles.** (i) The matrix entries are BIT-IDENTICAL to the dense
path's: `SunModel::assemble` (the body `build_h` always had, now emitting `add(row, col, value)`) produces its
contributions in a q-independent order; the analysis records which slot each lands in, and the per-q build sums the
same values in the same order, then symmetrizes as `h2_colmajor` does (0.5·(H₂[r,c] + conj H₂[c,r]), shift added to
the real diagonal last). (ii) P^T A P is positive definite iff A is, and the multifrontal factorization computes
exactly the pivots of the Cholesky of P^T A P (each front's pivot block is a diagonal block of a Schur complement), so
the first front whose `zpotrf` fails IS a non-positive pivot, and if none fails the factor exists. Only the rounding of
the factorization differs, i.e. a verdict can move only when min eig(H₂ + εI) is itself at round-off (~n·eps·‖H₂‖) —
which the 1e-6 relative shift sits seven orders above. No Lanczos or Ritz bound anywhere (OPEN_WORK item 10: those err
toward "stable"). The unshifted round-off case (`shift = 0` at a Goldstone zero) is a coin toss under either ordering,
and the one test that compares it stays on the dense path.

**The algorithm.** Analysis, once per packed model (lazily, 36 ms at L = 30): the slot sequence; the symmetric
adjacency of H₂; indistinguishable variables merged into supervariables; a weighted minimum-degree elimination on the
explicit elimination graph (ties to the lowest index, so it is deterministic), whose cliques ARE the factor's column
structures; a variable merged into its child's front when that adds no fill; per front its index list, its slice of the
original entries and each child's relative row positions — every one LOOKED UP, so a structure the fronts cannot hold
is an analysis error (and the dense path answers) rather than a dropped entry. Per q: assemble, then front by front
scatter the entries, extend-add the children, `zpotrf` the pivot block (return at the first failure), `ztrsm` and
`zherk` the Schur update — Accelerate LAPACK/BLAS on each front, as the dense path uses. At L = 30: 673 fronts, the
largest 180.

**Dense fallback.** `stable(sparse=None)` uses the sparse path when 2D ≥ 64 and it needs at most a quarter of the dense
flops; otherwise (a small cell, a dense operator, an analysis error) the dense `zpotrf` as before. `sparse=True/False`
forces either, `guard_info()` reports the analysis and the choice. Crossover measured on the 120° cell, one thread,
µs per q dense/sparse: 2D = 18 5/5, 72 50/34, 162 257/94, 288 782/163, 882 10 590/809. A chain coupled at every distance
(every site touches every other, 2D = 80) keeps the dense path.

**Pinned to.** `rsMagCalc/tests/test_sparse_guard.py` (8): the MARGIN, which is the exactness test — with λ = min eig H₂
from `eigvalsh`, the verdict at shift = −λ ± 1e-9·scale must be (λ + shift > 0) on both paths, on disordered σ = 0.6
cells (2D = 72, 288) and the clean 288 one, at Γ, K and generic q; the Goldstone zeros on the 12×12 120° cell where
sparse is automatic (passed with ε, refused with −1e-9·scale on both paths, so the exact zero is really seen);
Sunny's σ = 1/3 on three relaxed 81-site realizations along the Γ–K–M path, q by q equal to `is_stable_at`, ≥ 2 of 3
refused; the dense operator keeping the dense path; the analysis of every K3 fixture; threads. In
`tests/test_sun_lswt_parity.py` every guard case now forces BOTH paths against the automatic one and `is_stable_at`.
pyMagCalc `tests/test_kpm_stability.py::test_the_rust_guard_s_sparse_and_dense_paths_give_identical_verdicts` runs
every fixture of that file (the stationary maximum, the ferromagnet through Γ, the frustrated chain at J₂ = 0.1, 0.24,
0.26, 0.5, the 9×9 120° state through Γ and K, σ = 1/3 on 6×6 and 9×9) through both, forced, against NumPy. The
`kernels`-parametrized oracles of that file and of `test_s09_disorder_kpm` already reach the sparse path under `rust`
(the 9×9 cell is above the switch). **Mutation-checked**: skipping one extend-add entry in seven fails 7 of those tests
(all three margin cases, the Goldstone, σ = 1/3, threads, and the parity file's Γ/K case).

**Timing** (M1 Pro, 10 cores, 2026-10-27 09:16–09:23, load average 13–23, no pytest live), S09 L = 30, σ = 0.1 seed 0
relaxed, 40 q of the Γ–K–M path; `<this session's scratch>/bench_guard.py 30 0.1 40` and K4's `bench_k4.py 30 3 40`:

| | dense `zpotrf` | sparse | |
|---|---|---|---|
| guard alone, default pool | 38.2 ms/q | **1.05 ms/q** | 36× |
| guard alone, one thread | 47.6 ms/q | **1.50 ms/q** | 32× |
| guard alone, σ = 1/3, default pool | 40.3 ms/q | 0.83 ms/q | verdicts identical |
| (c) `kernels: rust` KPM q, guard included | 43.3 ms/q (K4) | **7.6 ms/q** (guard 0.6, KPM 7.0) | 5.7× |
| (c) the same, one thread | 70–80 ms/q (K4) | **23.9 ms/q** (guard 1.5, KPM 22.4) | |

Against (a) `kernels: numpy` (1369 ms/q in the same run) a KPM q is now **180×**; the guard went from 86 % of it to 8 %,
and checking every q exactly is no longer the bottleneck. The maps still agree with (a) to 2e-15.

**Gate** (2026-10-27 09:23–09:39, workspace root, `MAGCALC_NO_GUI=1 python -m pytest -m "" -rs`, pyMagCalc `24bc2ae`
plus this change uncommitted, rsMagCalc `8f6ed78` plus this change uncommitted and installed, magpipe 283f694):
**2160 passed, 2 skipped, 0 failed, 15 min 25 s** — the 9 added tests are this change's (8 in
`test_sparse_guard.py`, 1 in `test_kpm_stability.py`); the same two structural skips (`seekpath` absent, ALPS
present). Not idle: 1-min load 14 at the start, 53 (5-min 244) at the end.

### Rust kernels K4, 2026-10-27 — the KPM Chebyshev recursion behind `kernels: rust`

M4 of `RUST_KERNELS_PLAN.md`. rsMagCalc `332d667` (`src/kpm.rs`) carries the moment loop of `sun/kpm.kpm_sqw`
(β_{m+1} = 2Aβ_m − β_{m−1} with A = conj(g·H₂)/γ on the three observable vectors, μ_m^{ab} = v_a†β_m^b) and
`spectral_bound`'s power iteration on D̂†D̂, three ways: `kpm_moments(HMat, v, M, γ)` and `spectral_bound(HMat, ...)` on
a matrix already built, and `kpm_batch` (`rsmagcalc.SunModel.kpm_moments`) over a q batch with rayon, H(q) built inside
by K3's `build_h`, so no 2D×2D matrix crosses into Python. The Chebyshev coefficients, the +ω cutoff and its smearing,
the energy-grid reconstruction and the cross-section contraction stay in NumPy. What carries over exactly: the
recursion on conj(D̂) and the transposed (b, a) weight order (the two past KPM bugs), the bound's start vector (NumPy's
own `default_rng(seed)` draw, generated on the Python side and passed in), the moment count from `tol` (the same integer:
round-half-even in Python's order of operations, and `_kpm_rust` re-derives every count and refuses a mismatch), and the
Goldstone point (γ ≤ 0 → 1, the RAW bound returned so the same warning is issued once per q).

**Where the speed comes from is sparsity, not LAPACK and not threads.** cProfile of the NumPy path at S09 L = 30
(2D = 1800, M = 176): 0.50 s H(q) build, 0.25 s `spectral_bound` (a dense D̂†D̂ matmul, then 40 matvecs), 0.85 s moment
loop (176 dense 1800² matmuls on 3 vectors), 0.09 s reconstruction. H(q) of that supercell is 0.7 % nonzero (23 400 of
3.24 M entries), so the kernel scans each H(q) once into CSR when it is at most a quarter full and runs both the
recursion and the power iteration sparse, O(nnz·M); a denser operator (a small entangled cell) goes through the
platform `zgemm`, as NumPy's does. The skipped entries are exact zeros of the dense product too, so the two differ only
in summation order. The reconstruction was also restructured in NumPy, with the same arithmetic: `_cheb_coefs`'s node
matrix does not depend on ω, so the whole energy grid goes through ONE call (151 calls → 1, 90 ms → 2 ms per q) and
through one `contract_cross_section`.

**The runner under `kernels: rust`.** Guard 3 is unchanged in what it decides: `SUNModel.unstable_indices` (the per-q
verdict list `assert_stable` now reports from) runs K3's shifted Cholesky over EVERY q as one batch, same criterion,
same `h2_rel_tolerance`, and `on_imaginary: error` raises through the same `assert_stable(q_cart[iq:iq+1], context=
"(KPM S(q,w), q-point i of n)")` call as before, naming the first unstable q — now before any moment is computed
rather than after the q before it; `warn` warns with the same count and computes. Only then `kpm_sqw_batch` runs the
K4 batch. The NumPy loop (one H(q) per q shared by the guard and `kpm_sqw(hmat=)`) is the `kernels: numpy` path,
unchanged. `kpm_sqw` / `kpm_sqw_batch` take `kernels` and `n_workers` (default: the model's own), so a script such as
`disorder_kpm.py` gets K4 from the model it builds; a kernel that raises is logged by `kernels.fell_back` and the NumPy
loop answers.

**Pinned to.** `rsMagCalc/tests/test_kpm_parity.py` (63 tests): moments against `kpm._moments_numpy` at the SAME γ to
1e-12 relative to the largest moment, batch and standalone, with the operator forced dense, forced CSR and left to
choose, on every K3 fixture — plain and biquadratic SU(3), mixed spin N = (2, 3), a mixed-spin 3-cell supercell, the
non-collinear 120° supercell at 9, 36 and 81 sites (600 moments on the last), the entangled dimer chain — and the
ferromagnet; the bound to 1e-12 against `spectral_bound`; the `tol`-derived moment count equal to NumPy's integer at
three (fwhm, tol) pairs; the `chiral` channel end to end (rust vs numpy to 1e-11, with and without `hmat`); the
Goldstone point (H(Γ) ≡ 0 on the s = ½ ferromagnet: bound 0, γ = 1, the warning under both, identical columns); the
empty batch; refused arguments. Every K4 oracle runs under both kernel sets through the `kernels` fixture:
`test_kpm` (convergence to the exact `structure_factor` on the square AFM and the NON-COLLINEAR 81-site supercell, the
integrated intensity, the chiral sign, and the bound `kpm_sqw` actually used enclosing the spectrum),
`test_kpm_stability` (K3 had parametrized it; its runner cases now take the K4 batch), `test_s09_disorder_kpm` (the
disordered relaxed 81-site state against exact diagonalization, broadening, weight above the exact bandwidth),
`test_bond_disorder`. `tests/test_kernels_kpm.py` (12, fast) pins the wiring: every entry point reaching the kernel (a
call-counting spy) and agreeing with NumPy to 1e-10 on a mixed-spin supercell; `unstable_indices` against
`is_stable_at` and the frustrated chain's closed form (unstable exactly for 0 < h < ¼); the empty batch; a failing
batch, a failing standalone call and a wrong moment count each logged with NumPy answering; the runner refusing a
non-minimum under rust with the q named and the K4 batch never called; `on_imaginary: warn` computing; the runner map
and the shipped S09 `config_supercell.yaml` KPM task equal under both kernels. There was no S09 KPM case in
`test_sunny_tutorials` to parametrize; the shipped-config case lives in `test_kernels_kpm` instead.

**Timing** (M1 Pro, 10 cores, 2026-10-27 08:41–08:43, no pytest or `magcalc run` live; desktop apps busy: 1-min load
average 12–21, CPU 33–62 % idle, 9.7 of 10 GB swap in use — absolute numbers are this machine's, not a clean
benchmark). S09 L = 30 (900 sites, 2D = 1800), the σ = 0.1 seed-0 disordered state RELAXED (`minimize_energy`,
stable at every q measured), the tutorial's grid (0–3 meV step 0.02, fwhm 0.2, tol 0.02 → M = 176), generic q of
Γ–K–M; per q including the guard, as the runner pays it. Measured with
`<this session's scratch>/bench_k4.py 30 3 40`, two runs:

| configuration | per q | what runs |
|---|---|---|
| (a) `kernels: numpy` | **1376, 1399 ms** | the runner loop: NumPy H(q), NumPy Cholesky guard, NumPy moment loop (3 q) |
| (b) K3 H(q), NumPy moment loop | **982, 977 ms** | the same loop with the model on `kernels: rust` — the runner as K3 left it |
| (c) `kernels: rust` | **43.3, 43.7 ms** (guard 37, KPM 6) | `unstable_indices` + `kpm_sqw_batch`, 40 q, default pool |
| (c) `kernels: rust`, `n_workers: 1` | 70, 80 ms (guard 48–51, KPM 23–29) | the same, one thread |
| `kpm_sqw` one q per call, rust | 23, 34 ms | what `disorder_kpm.py`'s `sqw()` does (no guard) |

Against the plan: S09 L = 30 "from 1.7 s/q to under 0.2 s/q" — 43 ms/q with the guard included (70–80 ms on one
thread), from 1.4 s/q measured here; the M4 row's "S09 disorder KPM ≥ 5×" — **32×** over (a) and **22×** over (b),
the K3-only state. The KPM part alone (moments + bound + reconstruction, guard excluded) went from ~0.9 s/q under (b)
to 23–29 ms on one thread, i.e. ~35× from sparsity alone; rayon adds ~4× on that part. The maps agree with (a) to
2e-15 relative. **The guard is now 86 % of a KPM q** (one dense 1800² `zpotrf` per q, LAPACK-bound, so threads buy
only 1.3× over one) where it was ~5 % before; that is the cost of checking every q exactly. (Since removed without changing a
verdict: the sparse guard, section above — 0.6 ms of a 7.6 ms q.)

**Gate** (2026-10-27 08:42–09:00, workspace root, `MAGCALC_NO_GUI=1 python -m pytest -m "" -rs`, pyMagCalc `e85d96c` plus the uncommitted K4 wiring, rsMagCalc `332d667` installed, magpipe 283f694): **2151 passed, 2 skipped, 0 failed, 18 min 00 s** — 87 more than the K3 gate, all K4's (the `rust` parametrizations of `test_kpm` and `test_s09_disorder_kpm`, `tests/test_kernels_kpm.py`, rsMagCalc's `tests/test_kpm_parity.py`), the same two structural skips (`seekpath` absent, ALPS present), no crashed xdist worker. Not an idle run: desktop apps held the 1-min load at ~19 when it started, and it read 126 (5-min 271) when it ended, a load the gate's own `-n auto` workers do not explain alone; another session's unrelated pytest started in its last minute. So 18 min against K3's 13 min 34 s is the machine, not the 87 tests.

### Rust kernels K3, 2026-10-26 — SU(N) LSWT behind `kernels: rust`

M3 of `RUST_KERNELS_PLAN.md`. rsMagCalc `063113c` (`src/sun_lswt.rs`, `rsmagcalc.SunModel`; `src/lapack.rs`) carries
the per-q hot path of `SUNModel`: the H(q) assembly (`hamiltonian` was a Python loop over bonds × n_ops² with an
`np.outer` per term), the general eigenvalues `dispersion` / `max_imaginary` / the adapters' `stability_report` reduce,
the Colpa diagonalization (`_bogoliubov`, with the explicit [D positive descending, D negative] order), the per-band
one-magnon tensors of `structure_factor` (before the cross-section contraction, which stays in NumPy so every engine
shares ONE `contract_cross_section`), and the shifted Cholesky of the H₂ ⪰ 0 guard (`is_stable_at` / `assert_stable`),
each over a q BATCH with rayon. The guard is reimplemented identically, not approximately: Python computes the shift
`h2_rel_tolerance × _reference_h2_scale()` (the q-INDEPENDENT scale, Gap 4 #16) and the kernel answers "is H₂(q) +
shift·I positive definite" with LAPACK `zpotrf` — the same routine `np.linalg.cholesky` calls. The reference-state
tables (`t`, `tb`, `Q`, `QA`, `s0`) are still built by `_prepare` in Python and packed once per reference state
(`SUNModel._rust`, dropped by every `_prepare` and keyed on the bond/on-site/operator lists the way `sun.dynamics._fast`
is), ragged per site, so mixed spin runs through it. The entangled engine IS a `SUNModel`, so it gets the kernel with
no code of its own. pyMagCalc selects it with `kernels.rust_for` + `fell_back` at every entry point; the runner hands its
resolved `kernels` to `SUNCalculator` / `EntangledCalculator`, which also read `calculation.n_workers`, and both now
call the batch methods (`dispersion_batch`, `structure_factor_batch`, `eigenvalues`). A NON-minimum is not a kernel
failure: the kernel raises `rsmagcalc.NotPositiveDefinite` and pyMagCalc re-raises the NumPy path's `LinAlgError`, with
no fallback logged. `kpm.py` is untouched; the runner's KPM loop gets the fast H(q) through `hamiltonian`.

**Why LAPACK and not faer.** The plan named faer first. The deciding fact was the oracle, not speed: NumPy here links
Accelerate, and inside a DEGENERATE eigenspace the per-band split of the one-magnon weight is a basis choice that only
the same eigensolver reproduces. Calling Accelerate's ILP64 `zheevd`/`zpotrf`/`zgeev` (the symbols NumPy 2 calls) keeps
every band-by-band oracle test meaningful under both kernel sets. One layout detail mattered: `zgeev` is handed H in
column-major order, as NumPy hands it; passing the row-major buffer (Hᵀ, same eigenvalues) cost ~sqrt(eps) = 5e-8 on
S09's degenerate non-normal g·H₂ at 2D = 1800, against ~1e-15 with the same layout.

**Pinned to.** `rsMagCalc/tests/test_sun_lswt_parity.py` (37 tests): H(q) elementwise to 1e-13 (measured 2e-15 on FeI2,
1.4e-16 at S09 L = 30), eigenvalues against `np.linalg.eigvals`, energies and per-band correlation tensors to 1e-10
(summed over degenerate groups), T para-unitary with T†H₂T = diag(E) and equal to NumPy's columns up to a phase, the
guard verdict by verdict against `is_stable_at` on the `test_kpm_stability` fixtures (the stationary maximum, the
frustrated chain on both sides of J₂ = |J₁|/4, the 120° supercell with its Goldstone zeros, a σ = 0.6 disordered
supercell against the exact eigvalsh criterion), on mixed spin, on the 12-wide biquadratic operator set and on the
entangled dimer chain (the constituent offsets of the moment operator). Every K3 oracle in the plan runs under both
kernel sets through the `kernels` fixture: `test_sun` (gates 1–3 vs dipole and Sunny, the FeI2 bands AND intensities vs
Sunny, the runner end to end with `sqw_map`, the wrong-reference-state trap), `test_mixed_spin_sun`, `test_powder_sun`,
`test_sun_zeeman`, `test_sun_missing_terms`, `test_pair_operator`, `test_entangled_units`, `test_kpm_stability` (the
runner's KPM refusals included), `test_bond_disorder` (bit-identical at σ = 0, so the packed copy is rebuilt after a
disorder draw), and a new S09 case in `test_sunny_tutorials`: the shipped `config_supercell.yaml` through
`SUNCalculator` lands on E/site = −0.375 exactly and on the closed-form folded bands to 1e-9, with the Goldstone zero at
K. (The other S0x cases there are dipole-mode or K2.) `tests/test_kernels_sun.py` (16, fast) pins the wiring: every
entry point on a mixed-spin supercell, the empty batch, a failing kernel logged with NumPy answering (per kernel
method, and when packing fails), a non-minimum raising `LinAlgError` with no fallback logged, the cache following a
relaxation / disorder draw / swapped bond list, deepcopy and pickle, and both config paths (`mode: SUN` with
`sqw_map` and `n_workers`, `mode: entangled`).

**End to end, FeI2** (`examples/materials/FeI2/config_fei2_sun.yaml`, plots off, 60 q, `tasks: {minimization,
dispersion, sqw_map}`), the same config with `calculation.kernels` switched: energies agree to 5.3e-15 and intensities
to 3.1e-15 (the saved `disp`/`sqw` npz files), and the run log names the kernel after the engine line,
`kernels: rust (rsmagcalc 0.1.0, git 063113cfacf6, release)`.

**Timing** (M1 Pro, 10 cores, re-measured 2026-10-27 00:27–00:32 with no other compute job running: the stale K1
process PID 27374 that loaded the first measurement had exited with its session, and no pytest or `magcalc run` was
live. The machine was still not quiet — CPU 50–69 % idle, the 1-min load average 96–112 and falling, OneDrive and
`contactsd` busy, 8.9 GB of swap in use — so treat these as a lower bound on the speedup and the absolute numbers as
this machine's, not a clean benchmark. The first, fully loaded measurement of 2026-10-26 is kept in brackets.)

| path | numpy | rust | how measured |
|---|---|---|---|
| FeI2 SU(3) `magcalc run`, wall (minimization + 60-q dispersion + S(Q,ω)) | **5.6 s, 6.1 s** [11.5 s] | **1.7 s, 1.7 s** [4.0 s] | `/usr/bin/time -p env MAGCALC_NO_GUI=1 magcalc run fei2_<k>.yaml`, two runs each |
| FeI2 per q: H(q) / dispersion / S(Q,ω) / guard (121 q) | 47 / 46 / 60 / 69 ms | 0.04 / 0.09 / 0.16 / 0.06 ms | in-process `perf_counter`, batch of 121 q (loaded run only, not re-measured) |
| S09 L = 30 (900 sites, 2D = 1800): one H(q) build | **425 ms** [870 ms] | **15.8 ms** (1 thread), 13.6 ms [17.1, 16.5] | `<scratch>/bench_s09.py 30`, 8 generic q |
| S09 L = 30: guard (shifted Cholesky) per q | 468 ms [789 ms] | 61.9 ms (1 thread), 49.5 ms [114, 53] | same |
| S09 L = 30: general eigenvalues (`zgeev`) per q | 7.1 s [17.5 s] | 7.0 s (1 thread), 3.0 s [11.4, 6.2] | same, 4 q |
| S09 L = 30: one-magnon `structure_factor` per q | 8.3 s | 6.7 s (1 thread), 3.7 s | same, 4 q (first run of this line) |

The H(q) build is the plan's headline: 620 ms/q in `OPEN_WORK.md`, 425 ms here, 14–16 ms in the kernel (27–31×) — the
build that dominated KPM's per-q cost at L = 30. The general eigenproblem and the Colpa step are LAPACK-bound under
both (single-thread Rust ≈ NumPy, both calling Accelerate), so the kernel buys only its thread parallelism there, about
2.3×. FeI2 end to end: 3.4–3.6×, and its outputs agreed again on this run (energies 5.3e-15, intensities 3.1e-15).

**The S09 `structure_factor` line reports a per-band intensity gap of 7.4e-2, and it is not a defect.** At q = (0.31,
0.19, 0) the 30×30 supercell folds 30 bands onto ω = 1.000000 exactly, and inside a degenerate eigenspace how the weight
splits between bands is a choice of basis, not an observable (the same point as the collinear chirality in §6 of
`CLAUDE.md`). Summed over every degenerate group the two kernels agree to 3.6e-16, the total intensity to 5.6e-16, and
every non-degenerate band to 1e-28; the other three q points agree per band to 1e-10 or better. The earlier claim that
calling the same LAPACK eigensolver reproduces NumPy's per-band split holds for 2-fold groups but not for this 30-fold
one, so `tests/test_sun_lswt_parity.py` and `test_kernels_sun.py` are right to compare group sums. Checked with
`<this session's scratch>/k3check/s09_groups.py`.

**Gate** (2026-10-27, workspace root, `MAGCALC_NO_GUI=1 python -m pytest -m "" -rs`, pyMagCalc `22e2d7a`, rsMagCalc
`2cda740` installed, magpipe 283f694): **2064 passed, 2 skipped, 0 failed, 13 min 34 s** — 151 more than the K1+K2 gate,
all from K3 (the `rust` parametrizations of its oracle files, `tests/test_kernels_sun.py`, the new S09 SU(N) case
and rsMagCalc's `tests/test_sun_lswt_parity.py`), and the same two structural skips (`seekpath` absent, ALPS present). No
crashed xdist worker.

### Rust kernels K2, 2026-10-26 — the CP^(N−1) sampler and dynamics behind `kernels: rust`

M2 of `RUST_KERNELS_PLAN.md`. rsMagCalc `e865402` (`src/cpn.rs`, `rsmagcalc.CpnModel`) carries the CP^(N−1)
Metropolis sweep with pyMagCalc's adaptive step size (`thermalize`: the tuning half and the recorded fixed-sigma half,
on an `rsmagcalc.Rng` stream), the RK4 integrator (`evolve`), the damped quench (`quench`) and the trajectory loop of
`sun.dynamics.sampled_correlations` (`sampled_trajectories`, rayon over trajectories, stream (seed, i) each). A
`SUNModel`'s per-site operator tables, on-site matrices and bond list are packed ONCE as ragged arrays, so a
mixed-spin cell runs through the kernel where the NumPy `_fast` path refuses it. pyMagCalc selects it per call
(`kernels.rust_for` + `fell_back`; `thermalize` by the type of its `rng`, as K1's `_sweep`), the runner passes
`kernels` and the new `sun_sampled_correlations.n_workers`, and `tests/test_kernels_cpn.py` pins the wiring, the
fallback-with-logged-exception, chunk independence and the config path. The NumPy path is untouched.

**The algorithmic fix.** `metropolis_sweep` evaluates the TOTAL energy before and after every proposal, O(L²) per
sweep (cProfile on the 144-site S09 model: 99 % of a 1.9 s sweep in `SUNModel._energy_of`, called 2L times). The
kernel's dE is the term-by-term change of the bond terms touching the site plus its on-site term — exact for any bond
list, self-bonds included, not a mean-field linearization — O(L z n_ops²) per sweep. The equilibration verdict
(drift, `equilibrated`, warn/error) is ONE Python function, `_thermalize_verdict`, fed by the per-sweep record the
kernel returns, so the diagnostics are produced by the kernel rather than bypassed and cannot diverge between kernels.

**Pinned to.** `rsMagCalc/tests/test_cpn_parity.py` (28 tests): dE = E(after) − E(before) to 1e-12 on plain SU(3),
the biquadratic 12-wide operator set, a mixed-spin cell, a mixed-spin supercell and a self-bonded one-site cell;
h_i|Z_i⟩ against the `local_field` definition; `evolve` against `sun.dynamics.evolve` to 1e-12 over 50 steps (all
four cells); `quench` and `damped_step` to 1e-12; the sampler against the closed-form partition function of a
decoupled site, determinism per seed, chunk and thread independence, |Z_i| = 1. Every K2 oracle in the plan runs under
both kernel sets through the `kernels` fixture: `test_sun_dynamics` (N = 2 reduces to Landau–Lifshitz; the low-T
peaks on the SU(N) LSWT band and the hardening on cooling), `test_sun_sampler_equilibration` (the closed-form
⟨E⟩(kT), sigma-start independence, the stuck-sampler number, the verdict firing on the stuck run and not the good
one), `test_sun_quench` (dE/dt = −2λ Var h, the LOWEST eigenvector, monotone descent), the SU(N) half of
`test_classical_absolute_normalization` (the 1e-14 equal-time sum rule) and `test_classical_window`, and the S06 quench
in `test_sunny_tutorials`.

**Timing, the plan's acceptance (S09 thermalization ≥ 10×).** The S09 triangular AFM of `disorder_kpm.py` at
L = 12 (144 sites, 864 directed bonds, N = 2) in its 120° state, kT = 0.1, sigma 0.1 adapted; this machine (M1 Pro,
10 cores; a stale `python -` process from the K1 session held one core, nothing else), 2026-10-26, `MAGCALC_NO_GUI=1
python <scratch>/bench_k2.py` (the calls below, timed with `perf_counter`):

| what | numpy | rust | ratio |
|---|---|---|---|
| `thermalize`, per sweep (40 / 4000 sweeps) | **1457 ms** | **0.087 ms** | 17 000× |
| `evolve`, per RK4 step (1000 steps) | 0.42 ms | 0.12 ms | 3.5× |
| `sampled_correlations`, 4 traj × 200 sweeps + 512 steps, 3 q | **870 s** | **8.2 s** | 106× |
| S06 quench, L = 40 (1600 sites, N = 3), 25 600 steps (`quench.py --kernels`) | 175 s (6.8 ms/step) | 44.5 s (1.7 ms/step) | 3.9× |

The S06 quench reproduces the tutorial's charges under BOTH kernels, +12 → −4 → −6 at τ = 4, 16, 256, from the same
random start (the kernels agree per step to 1e-12). The RK4 gains are modest because `_h_times_z` was already
vectorized (its einsums are the time at 1600 sites; the kernel's inner loop is scalar complex arithmetic over the
bond list and single-threaded per trajectory). Once the loops were in Rust the remaining cost of
`sampled_correlations` was the Python `moment_of` loop inside `dynamical_structure_factor` (0.9 M iterations for
4 × 256 steps at 144 sites), so it is now `moments_along`, one einsum over the trajectory including the
entangled-unit staggered phases — a NumPy change that both kernels share, pinned against `moment_of` on the dimer
chain and by the sum rule.

**What the second random stream exposed this time** (the K1 lesson applied: compare ENSEMBLES before touching a
tolerance). One K2 oracle failed under `rust` on the first full run, `test_mode_hardens_toward_the_lswt_band_on_cooling`:
its kT = 0.06 and kT = 0.02 peaks landed on the same 0.077 meV bin (1.918) with 3 trajectories. The sampler was not at
fault: thermalized ⟨E − E0⟩ after the test's own 500 adapted sweeps sits on equipartition 2 L kT (CP², four real degrees
of freedom per site) at all three temperatures under both kernels — kT = 0.15: 5.24 ± 0.10 (64 Rust seeds), 5.73 ± 0.28
(8 NumPy) vs 4.80; kT = 0.06: 2.013 ± 0.047, 1.90 ± 0.15 vs 1.92; kT = 0.02: 0.635 ± 0.013, 0.671 ± 0.034 vs 0.640 —
and 24 Rust trajectories put the peaks at 1.611 < 1.841 < 1.918, exactly the NumPy values the test's docstring records.
A 3-trajectory argmax at one-bin resolution is marginal; the `rust` parametrization now runs 24 trajectories (7 s),
the `numpy` one keeps 3 (~2 min of Python), and the asymmetry is written into the test.

**Gate on this tree** (pyMagCalc branch `feature/rust-kernels-k1` at `395da88` plus the uncommitted K2 wiring,
rsMagCalc `d9bfc99` installed, magpipe `283f694`), `MAGCALC_NO_GUI=1 python -m pytest -m "" -rs` from the workspace
root, 2026-10-26, machine otherwise idle (the one stale K1 process aside): **1913 passed, 2 skipped, 0 failed,
20 min 21 s** — the K1 gate's 1850 plus 28 rsMagCalc K2 parity tests, 30 `rust` parametrizations of the K2 oracles and
5 new wiring tests; the two skips are the structural pair (`seekpath` absent at `test_cell_utils.py:146`, ALPS present
at `magpipe/tests/test_alps.py:187`). The gate floor is unchanged (the `numpy` parametrizations of K1's two low-T
tests, ~3 min each); the K2 `numpy` oracles are 2–3 min of Python at most.

### Rust kernels K1, 2026-10-26 — the classical samplers behind `kernels: rust`

M1 of `RUST_KERNELS_PLAN.md`. rsMagCalc `1a8dbdd` (`src/classical.rs`) carries the Metropolis sweep (uniform and
`propose: flip`), the whole replica-exchange loop (rayon over replicas), the Wang–Landau move loop, the rk4 and
implicit-midpoint integrators, the Heun Langevin step and the trajectory loop of `sampled_correlations` (rayon over
trajectories; trajectory i owns the PCG64 stream (seed, i), so chunking and thread count change nothing). pyMagCalc
selects it per call (`kernels.rust_for` + `fell_back`: a kernel that raises is logged and the NumPy loop answers —
pinned by `tests/test_kernels_classical.py`, which also drives each of the four classical tasks from a config under
`kernels: rust`). The NumPy path is untouched and is the parity reference.

**Pinned to.** `evolve` against `classical_dynamics.evolve` from the same m0, 50 rk4 and 50 midpoint steps: 2e-16
(tolerance 1e-12); the single-move dE of the sweep against E(after) − E(before): 1e-12 (`rsMagCalc/tests/
test_classical_parity.py`). Every K1 oracle in the plan runs under both kernel sets through the `kernels` fixture:
`test_thermal_mc`, `test_classical_absolute_normalization`, `test_integrators`, `test_classical_window`,
`test_classical_to_quantum`, `test_static_correlations`, `test_wang_landau`, `test_disorder`, and S02/S05 in
`test_sunny_tutorials` (the flip proposal stays Ising from a polarized start under both).

**Timing, the plan's acceptance.** The 128-trajectory case of `test_low_T_ferromagnet_matches_the_lswt_absolute_intensity`
(gapped FM chain, L = 32, kT = 0.005, 3000 sweeps, 4096 rk4 steps, 5 q), the same `sampled_correlations` call under
each kernel set, this machine (M1 Pro, 10 cores; OneDrive and VS Code were active, no other pytest), 2026-10-26:

| kernels | 128 trajectories | command |
|---|---|---|
| numpy | **168.8 s** | `MAGCALC_NO_GUI=1 python -I <scratch>/time_k1.py numpy` (the test's call, `n_traj=128, therm_sweeps=3000`) |
| rust | **3.3 s, 3.4 s** (two runs) | same script with `rust`; rayon over all cores |

51×. (`CLAUDE.md` used to quote "~7 min" for this test; that figure was a contended-machine one — 183 s was measured
again inside the gate the same day.) The test itself under `rust` now runs 512 trajectories × 30 000 sweeps in 18 s.

**What the second random stream exposed, and the trap it is.** Two low-T tests failed under `rust` on the first full
run, and both were the tests, not the kernel — found by comparing the ENSEMBLES rather than the spectra: 3000 uniform
sweeps at kT = 0.005 leave ⟨E − E0⟩ = 0.1830 ± 0.0027 (NumPy, 128 states) and 0.1795 ± 0.0009 (Rust, 1024 states)
against equipartition N·kT = 0.160 — the same 12 % excess under both samplers, so the thermalization is incomplete
and identical. At that setting the converged classical/LSWT ratio is **1.077** (1024 Rust trajectories), and the
"1.015, per q 0.96–1.07" the test's docstring records for 128 NumPy trajectories at seed 1 was a 2σ draw (Rust seeds
1–4: means 1.083, 0.993, 1.049, 1.012). Thermalized — 30 000 sweeps bring ⟨E − E0⟩ to 0.160 — the ratio is **0.95,
per q 0.92–1.00**, inside the test's 6 % / 15 % with margin. Likewise the c2q q-independence test's 6-trajectory
spread is 0.29 at NumPy seed 1 but 0.92 at seed 2 (Rust: 0.90, 1.17; at 384 trajectories 0.22, 0.15). The `rust`
parametrizations now use the thermalized, high-statistics settings; the `numpy` ones keep the historical, marginal
settings because the thermalized run would be ~30 min of Python — and that asymmetry is written into each test.
Gate floor: the `numpy` parametrizations of this test (~3 min) and of S05 (~3.5 min) are now the slowest tests;
the plan's "below 1 min" needs the NumPy oracle dropped from the gate, which is not on offer.

**Gate on this tree** (pyMagCalc branch `feature/rust-kernels-k1`, rsMagCalc `f1c7923` installed, magpipe `283f694`),
`MAGCALC_NO_GUI=1 python -m pytest -m "" -rs` from the workspace root, 2026-10-26: **1850 passed, 2 skipped, 0 failed,
20 min 24 s** — the previous 1753 plus 61 `rust` parametrizations and 36 rsMagCalc parity tests; the two skips are
the structural `seekpath` and ALPS ones. Not an idle machine (OneDrive, VS Code and one stray Python process were
busy), so the time is an upper bound; no xdist worker crashed.

### Rule-value string parser, 2026-10-18

Found while making magpipe emit the allowed forms `magcalc symmetry` prints. A rule `value` written as one string
(`D1x,sqrt(3)*D1x/3,D1z`) was parsed by deleting every `[`, `]`, `(` and `)` and then splitting on every comma.
`_split_rule_value` now strips one outer pair and splits only at bracket depth 0.

| Defect | What hid it | Pinned by |
|--------|-------------|-----------|
| `sqrt(3)*D1x/3` became `sqrt3*D1x/3`, an unbound symbol. On KFe3J's kagome NN bond with in-plane moments every band came out NaN, no error; the decimal spelling `0.5773502691896258*D1x` gives finite bands, and now both give the same ones. With the symmetry-conflict check above (branch `claude/cool-gagarin-59fa30` before the merge) the same value, which is the allowed one, was refused as a conflict, by an error that tells the user to paste that form. Also: `(J1+J2)/2, 0, 0` lost its grouping, `atan2(Dy, Dx)` split in two, `0,,Dz` silently became (0, Dz, 0), and a 3×3 string flattened to a 9-vector and died on a shape error | Every shipped rule writes a string value with bare names (`0,Dy,Dz`), or uses a YAML list, which never reaches the parser; explicit `pair:` entries go through `parse_expr` and were always right | `tests/test_rule_value_parsing.py`. Premise from geometry, not the engine: the bond's little group is a mirror that reverses it, so the in-plane D is ⊥ the bond, D_y/D_x = −r_x/r_y = √3/3, and the CLI's allowed matrix agrees. Oracle: the exact, bracketed, decimal and YAML-list spellings give the identical 36-entry table; for `anisotropic_exchange` and `interaction_matrix` (the CLI's full allowed matrix pasted as one string) string = list. 8 of its tests fail on the old parser |

Key conventions worth knowing before touching the engine:
- pyMagCalc stores `HMat(q) = g · H2(q)` with H2 Hermitian, g = diag(I,−I); eigenvalues
  are the ±ω pairs. (The name `_build_TwogH2_matrix` is just a name.)
- Bonds are listed in BOTH directions with NO 1/2 on the hopping — this is how
  `H = (1/2) Σ_ordered` is encoded. The on-site (mean-field / local-field) term is the
  q=0 sum, NO phase (a phase there makes a ferromagnet's H(q) cancel to zero).
- The dipole ground state ≠ the SU(N) ground state when an anisotropy is present
  (a coherent state has ⟨Sz²⟩ ≠ (S n_z)²). Find the SU(N) GS in SU(N).
