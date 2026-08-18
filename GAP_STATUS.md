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

Test suite: **837 passed, 3 skipped** on the last full gate (`pytest -m ""` from the
workspace root, 2026-08-16, 46 min under load). 832 tests collect from inside
`pyMagCalc/`, of which 654 are the fast default suite. The merge gate is
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
| 9 | **SCGA (paramagnetic diffuse scattering)** | ✅ | `tasks: {scga: true}` + `scga: {temperature, mesh_density, cross_section}`. Self-consistent Gaussian approximation: classical spins, hard length constraint softened to a global Lagrange multiplier λ, static S(q) = kT·pref†(λ+J(q))⁻¹pref with λ from the spin sum rule. SAME `fourier_exchange_matrix` J(q) as the LT guard; single-λ (one symmetry class — Bravais + kagome/pyrochlore-type, refuses inequivalent sublattices). Above T_N, so the LSWT ground-state guard is auto-skipped for a pure-SCGA run. Validated: exact classical-chain closed form (λ=√(4J²+(3kT/S²)²), S(q)=3kT/(λ+2Jcosq)) to 1e-9; **Sunny 0.8.1 SCGA** on square-lattice AND kagome AFM — λ and S(q) to 6 digits (matches `ssf_perp`, apply_g, (2/3)Tr at q→0); sum rule + high-T flat limit. `magcalc/scga.py`, `tests/test_scga.py` |
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
| 19 | Static / energy-integrated correlations | ✅ | `tasks.static_sqw` (LSWT) and `tasks.static_correlations` (classical) | Sunny `intensities_static` to 6e-9; free-spin sum rule 2S²/3 (perp) and S² (trace) EXACTLY, at every q and T |
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

Key conventions worth knowing before touching the engine:
- pyMagCalc stores `HMat(q) = g · H2(q)` with H2 Hermitian, g = diag(I,−I); eigenvalues
  are the ±ω pairs. (The name `_build_TwogH2_matrix` is just a name.)
- Bonds are listed in BOTH directions with NO 1/2 on the hopping — this is how
  `H = (1/2) Σ_ordered` is encoded. The on-site (mean-field / local-field) term is the
  q=0 sum, NO phase (a phase there makes a ferromagnet's H(q) cancel to zero).
- The dipole ground state ≠ the SU(N) ground state when an anisotropy is present
  (a coherent state has ⟨Sz²⟩ ≠ (S n_z)²). Find the SU(N) GS in SU(N).
