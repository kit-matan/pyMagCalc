# MagCalc — a guide for AI models (and people) starting work on it

**Read this first.** It explains what MagCalc is, how the code is organized, the physics
conventions that are easy to get wrong, how correctness is established here, and how to
start working without breaking anything. It is written so that a model with no prior
context can make a correct first change. It is a map, not the whole territory: the
authoritative details live in the files it points to (§12), and when they disagree, the
code and the tests win.

*Last reviewed: 2026-10-31, against pyMagCalc `d557a7c` plus the dimer series' projective cluster-additive one-particle block (`_pca_one_particle`, OPEN_WORK item 11). This file is kept current with
every change that alters what it describes (§13).*

---

## 1. What MagCalc is

MagCalc computes **magnetic excitations of crystals** — the quantity measured by
inelastic neutron scattering (INS). Given a crystal structure, a spin Hamiltonian
(exchange, Dzyaloshinskii–Moriya, anisotropies, fields, …) and a magnetic ground state, it
returns:

- **spin-wave dispersions** ω(q): the energy of each magnon band along a path in
  reciprocal space;
- the **dynamical structure factor** S(Q,ω): the intensity a neutron spectrometer would
  measure, including form factors, polarization, temperature, twin domains, instrument
  resolution and powder averaging;
- **fits** of the Hamiltonian's parameters to measured dispersions or intensities.

The core method is **linear spin-wave theory (LSWT)**: expand the spins about a classical
energy minimum with Holstein–Primakoff bosons, keep the quadratic part, and diagonalize it
with a Bogoliubov transformation at each q. Around that core MagCalc adds methods for
regimes LSWT does not cover: SU(N) spin-wave theory for single-ion (multipolar) physics,
entangled clusters (dimers → triplons), a dimer series expansion, the self-consistent
Gaussian approximation (paramagnetic diffuse scattering), classical Monte Carlo and spin
dynamics at finite temperature, the kernel polynomial method for very large cells, and
1/S corrections.

It occupies the same niche as **SpinW** (MATLAB) and **Sunny.jl** (Julia), and every
feature is validated against one of them or against an exact analytic result (§9).

A calculation is described by **one declarative YAML file** and run with:

```bash
magcalc run config.yaml
```

---

## 2. The repositories

All live under the GitHub user **`kit-matan`**. pyMagCalc is the only required one.

| Repo | Language | Role | Required? |
|---|---|---|---|
| [pyMagCalc](https://github.com/kit-matan/pyMagCalc) | Python | The engine: config, model building, every physics method, CLI, GUI server, tests. **Nearly all work happens here.** | yes |
| [fMagCalc](https://github.com/kit-matan/fMagCalc) | Fortran, OpenMP, LAPACK | Compiled hot path for the **dipole** LSWT engine: per-q diagonalization, S(Q,ω), powder averaging, including mixed spin. Selected by `calculation: {backend: fortran}`. | no |
| [rsMagCalc](https://github.com/kit-matan/rsMagCalc) | Rust, PyO3 | Compiled kernels for the loops that are Python-bound per site or per q **outside** the dipole path: classical and CP^(N−1) samplers, the annealing ground-state search, SU(N) LSWT, KPM, SCGA, the sparse KPM stability guard, the dimer series' per-cluster perturbation theory. Selected by `calculation: {kernels: rust}`. | no |
| magpipe | Python | A materials-discovery pipeline built **on** MagCalc as a library: Materials Project screen → DFT energy mapping → ED/QMC thermodynamics → a MagCalc config and spectrum. Never reimplements physics MagCalc has. | no |

The maintainer's workspace also holds a vendored **Sunny.jl 0.8.1** checkout, used only as
a validation oracle, and an `archive/` of retired implementations that must never be built
on. Neither is part of the published repos.

Both compiled companions follow **one contract**: pyMagCalc never requires them; an
unavailable or failing companion logs a WARNING and the NumPy path runs instead; the NumPy
path is never removed, because it is the fallback **and** the reference the compiled code
is pinned to; and the run log names the build that ran. A full Rust **port** of MagCalc was
evaluated on 2026-10-25 and rejected. The engine is built on SymPy, the dipole hot path was
already compiled, and a rewrite would reopen every defect the validation history caught.
rsMagCalc is the scoped alternative.

---

## 3. How a calculation flows through the code

```
config.yaml
   │  schema.py            Pydantic v2 validation (MagCalcConfig)
   │  config_loader.py     load; mcif.py expands `from_mcif:` magnetic CIFs
   ▼
GenericSpinModel           generic_model.py + config_builder.py
   │  atoms: Wyckoff orbits expanded by spglib; magnetic supercell replicated
   │  bonds: `symmetry_rules` propagated by the space group (J' = R J Rᵀ; DM as axial vectors)
   ▼
symbolic.py                SymPy Holstein–Primakoff Hamiltonian, Fourier transform → H(q)
   ▼
MagCalc (core.py)          lambdify H(q); batched Colpa Bogoliubov work (numerical.py)
   │                       — or hand the hot loop to fMagCalc (backend: fortran)
   ▼
runner.py                  the declarative task pipeline:
                           minimization → ground-state guards → dispersion → S(Q,ω)
                           → beyond-LSWT tasks → powder → fit → plots and data files
```

`mode: SUN` and `mode: entangled` leave this path after the model is built.
`sun/adapter.py` converts the `GenericSpinModel` into a `SUNModel` (`sun/lswt.py`), which
builds H(q) numerically, with no SymPy. The runner drives it through the same task
pipeline.

---

## 4. The three LSWT engines (`calculation.mode`)

| Mode | Module | One site is… | Use it for |
|---|---|---|---|
| `dipole` (default) | `core.py`, `symbolic.py`, `numerical.py`, `linalg.py` | one classical spin, one boson | ordinary magnets; the SpinW-equivalent engine |
| `SUN` | `sun/lswt.py`, `sun/adapter.py` | an N = 2S+1 level coherent state, N−1 bosons | S ≥ 1 with single-ion anisotropy or crystal fields: the multipolar bands that dipole LSWT **cannot** produce (FeI₂) |
| `entangled` | `sun/entangled.py`, `sun/dimer_series.py` | a whole cluster (dimer, trimer) as one SU(N) site | singlet ground states with triplon excitations; `series_order: N` switches the dispersion to a linked-cluster series for strong coupling |

The SU(N) ground state differs from the dipole one whenever an anisotropy is present.
**Never seed one engine with the other's ground state.** SU(N) refuses a supplied
structure that its own energy audit can lower.

---

## 5. Beyond LSWT

| Task (`tasks:` key) | Module | What it computes | Pinned to |
|---|---|---|---|
| `scga` | `scga.py` | paramagnetic diffuse S(q) above T_N (one Lagrange multiplier; refuses inequivalent sublattices) | Sunny SCGA (square, kagome); exact classical-chain closed form |
| `thermal_mc` | `thermal_mc.py` | ⟨E⟩, C, M, χ versus T by parallel-tempering Metropolis | exact classical dimer; Langevin function; single-spin partition function |
| `wang_landau` | `thermal_mc.py` | the density of states g(E), so one run covers every T | the classical dimer's exactly flat g(E) |
| `sampled_correlations` | `classical_dynamics.py` | finite-T classical S(q,ω): thermalize, then Landau–Lifshitz dynamics | equal-time sum rule to 1e-14; low-T ferromagnet against LSWT |
| `sun_sampled_correlations` | `sun/dynamics.py` | the same for SU(N) coherent states (CP^(N−1)) | the sampler's own closed-form partition function |
| `static_correlations`, `static_sqw` | `thermal_mc.py`, `numerical.py` | energy-integrated correlations, classical and LSWT | Sunny `intensities_static`; free-spin sum rules |
| `kpm_sqw` | `sun/kpm.py` | Chebyshev S(q,ω) with no diagonalization, for very large SU(N) cells | convergence to the exact `structure_factor` |
| `corrections` | `corrections.py` | 1/S zero-point energy and ordered-moment reduction | Sunny; square-lattice Heisenberg AFM (dE = −0.157947 J) |
| `minimization` | `core.py`, `annealing.py`, `spiral_opt.py` | the classical ground state: `method: anneal` (preferred), `steep`, or scipy multistart; `optimize_k: true` for single-k spirals | analytic minima (SW20 in field, S06) |

The other task keys are `dispersion`, `sqw_map`, `powder_average`, `energy_cut`, `fit`,
`export_csv`, `plot_*`, and `satellites` for single-k spirals.

---

## 6. The config file

The config is the **single source of truth**. The web Studio (`gui/`) and the native
SwiftUI app (`native/MagCalcStudio/`) are editors of the same file; they never hold a
separate model. Top-level blocks:

| Block | Holds |
|---|---|
| `crystal_structure` | `lattice_parameters` + `space_group` + `wyckoff_atoms` (preferred), or `lattice_vectors` + `atoms_uc`; per-atom `spin_S`, `ion` (form factor), `g`; `magnetic_supercell` |
| `interactions` | `symmetry_rules` (preferred), explicit bond lists, `single_ion_anisotropy`, `sia_matrix`, `stevens`, `biquadratic`, `pair_operator`, `kitaev`, `dipole_dipole` |
| `parameters`, `parameter_order` | named values the interactions refer to (safe SymPy expressions allowed) |
| `magnetic_structure` | `pattern`, `single_k` (rotating-frame spiral), `multi_k`, `explicit`; or `from_mcif:` at top level |
| `tasks` | which steps to run (§5) |
| `q_path` | named points in **chemical** RLU plus `path` and `points_per_segment` |
| `calculation` | `mode`, `backend`, `kernels`, `n_workers`, `temperature`, `domains`, `cross_section`, `cache_mode`, the guard tolerances, `anisotropy_renormalization`, `classical_energy`, `series_order` |
| per-task blocks | `minimization`, `scga`, `thermal_mc`, `sampled_correlations`, `kpm`, `corrections`, `powder_average`, `energy_cut`, `fitting`, … |
| `output`, `plotting` | file names, `save_plot`, `show_plot`, resolution |

**`pyMagCalc/CLAUDE.md` holds the authoring rules. Read it before writing or editing
any config.** It covers the symmetry model, `ref_pair` rules, the hard errors, SU(N) and
entangled modes, the intensity layer, and the compact YAML style that generated configs
must use (`examples/spinw_tutorials/_compact_yaml.py`, never plain `yaml.safe_dump`).

The smallest complete model, a ferromagnetic S = 1 chain. It runs as written, and its band
is ω = 2|J|S(1 − cos 2πh) exactly:

```yaml
crystal_structure:
  lattice_vectors: [[3.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
  atoms_uc:
  - {label: Fe, pos: [0.0, 0.0, 0.0], spin_S: 1.0, ion: Fe2+}
interactions:
  symmetry_rules:
  - {type: heisenberg, distance: 3.0, value: J1}
parameters: {J1: -1.0}           # meV; negative = ferromagnetic
parameter_order: [J1]
magnetic_structure: {type: pattern, pattern_type: ferromagnetic, direction: [0, 0, 1]}
tasks: {dispersion: true, sqw_map: true}
q_path:
  G: [0, 0, 0]
  X: [0.5, 0, 0]
  path: [G, X]
  points_per_segment: 11
calculation: {cache_mode: none}
output: {save_data: true, disp_data_filename: disp.npz, sqw_data_filename: sqw.npz}
plotting: {save_plot: false, show_plot: false, plot_structure: false}
```

The saved `q_vectors` are **Cartesian, in Å⁻¹**, not RLU, even though `q_path` is given
in RLU.

---

## 7. Command line

```bash
magcalc run <config.yaml>          # run the tasks
magcalc validate <config.yaml>     # schema check only
magcalc symmetry <config.yaml> [--max-distance Å] [--json]   # bond orbits + allowed matrices
magcalc fit <config.yaml>          # lmfit against neutron data in the `fitting:` block
magcalc mcif <file.mcif> -o out.yaml       # magnetic CIF → config fragment
magcalc mcif-out <config.yaml> -o out.mcif # magnetic structure → mCIF
magcalc init <config.yaml>         # template
magcalc where                      # which engine copy is running, and whether rsMagCalc is available
magcalc guard [--install|--uninstall]      # interpreter-startup shadow guard
```

**Use `magcalc run` for anything you intend to believe.** Building a `MagCalc` object by
hand in Python skips the runner. That means it skips the `magnetic_structure` block and
**all the ground-state guards**, so you can get a confident spectrum about a state that is
not a minimum. Reserve the direct API for re-evaluating an already validated model (the
`DispersionEvaluator` fast path used by fitting).

---

## 8. Conventions that are easy to get wrong

Most are defined in `GAP_STATUS.md`. Each one has caused a real, plausible-looking wrong
result at least once.

- **H(q) = g·H₂ with g = diag(I, −I).** Bonds are listed **in both directions** with no ½
  on the hopping; the on-site mean-field term is the q = 0 sum and carries **no phase**.
  Putting a phase on it makes a ferromagnet's H(q) cancel to zero.
- **Rules expand to both bond directions automatically.** Never also list the reverse bond
  by hand.
- **Spin length is per site.** `spin_S` is set atom by atom, and mixed-spin models are
  supported. The S(Q,ω) prefactor is √(S_i/2) **per site**. Classically, `S_val` (binds
  the symbolic S) and the per-site lengths `S_vec` are separate quantities.
- **Absolute intensity equals Sunny's**, band by band. Sunny's `ssf_perp` applies the
  g-tensor by default, which makes it 4× ours at g = 2; compare with `apply_g=false`. A
  discrepancy that is a clean constant factor is a **bug** until proven otherwise.
- **Degenerate bands.** Inside an exactly degenerate group, how the weight splits between
  bands is a basis choice, not an observable. Compare group sums. This was seen as a
  7e-2 per-band gap with 4e-16 agreement on the group sum.
- **The dipole engine is Sunny's `:dipole_uncorrected`** (and SpinW's convention).
  `calculation.anisotropy_renormalization: rcs` switches to Sunny's corrected `:dipole`.
  SU(N) needs neither.
- **Zeeman:** `parameters: {H_mag: <Tesla>, H_dir: [...]}` reproduces the electron g = 2
  splitting 2·μ_B·H_mag.
- **`q_path` is in chemical RLU** even with a magnetic supercell (bands fold). S(Q,ω) is
  normalized per chemical cell.
- **Symmetry rules have hard errors:** a non-scalar rule without `ref_pair`, a rule that
  matches no bond, an ambiguous reference bond, and a value the bond's own site symmetry
  forbids (Moriya's rules) all raise. `magcalc symmetry <config>` prints the allowed form
  of every orbit.

---

## 9. How correctness is established here — read this before changing physics

The engine has repeatedly produced spectra that **looked right and were wrong**: a dropped
anisotropy, a sign flip chosen by floating-point rounding, an intensity off by 4/3, a
ground state that was actually a maximum. Every one was caught by an independent oracle or
an exact identity, never by looking at a plot. So:

1. **Every feature and fix gets a test pinned to an independent reference:** Sunny.jl,
   SpinW, or an exact analytic identity (a closed form, a sum rule, a partition function).
   **Never** a number the engine produced itself. Never trust a published reference number
   without re-converging it: Sunny's own FeI₂ example sits in a local minimum.
2. **Do not weaken the ground-state guards.** LSWT about a non-minimum returns a
   meaningless but plausible spectrum. That has historically been the #1 source of silent
   wrongness. The guards are:
   - an imaginary-mode check, which ANDs an absolute and a relative tolerance;
   - an energy audit, which catches a stationary *maximum* the first cannot see (on
     every dipole model, quartic energies included, since 2026-10-29; when it cannot
     run the runner says so at WARNING);
   - for KPM, which never diagonalizes, H₂(q) ⪰ 0 at every q.

   `on_imaginary: warn` is only for an instability that is understood, with a comment
   saying why.
3. **Reproducibility is not correctness.** A deterministic bug downstream of a stochastic
   search reproduces perfectly on every seed.
4. **A skipped test is not a passing test.** Run with `-rs` and account for every skip.
   The only accepted skips are named optional dependencies.

`GAP_STATUS.md` is the record of what is validated against what, and its closing section,
"How things were validated (and the recurring trap)", is required reading.

---

## 10. Tests

```bash
pytest                       # fast suite: 1249 of 1521 tests; "-m 'not slow'" is the default
pytest -m slow -k <name>     # one feature's deep validations
pytest -m ""                 # everything: REQUIRED before merging to master
pytest -n0 ...               # serial; needed for -s or pdb
```

- Both `pytest.ini` files run in parallel (`-n auto --dist worksteal`, pytest-xdist is a
  required dev dependency).
- From the maintainer's workspace root, plain `pytest` also collects fMagCalc's,
  rsMagCalc's and magpipe's tests. The last full gate there was **2737 passed, 2 skipped,
  2 xfailed** (2026-10-31). The two skips are structural: `seekpath` absent, ALPS
  present. The two xfails are strict pins of OPEN_WORK item 9.
- **Tests never open a GUI window.** `tests/conftest.py` forces the Agg backend and sets
  `MAGCALC_NO_GUI=1`. A bare `magcalc run` outside pytest is not guarded; prefix it with
  `MAGCALC_NO_GUI=1`.
- **Fixtures for the compiled paths:** `kernels` runs a test under `numpy` and `rust`;
  `backend` runs it under `numpy` and `fortran`. The compiled case skips, by name, when the
  companion is not installed.
- Keep at least one quick test of every feature **outside** the `slow` marker, so the fast
  suite touches every code path.
- Run the full gate on an otherwise idle machine. Crashed xdist workers under load show up
  as false failures.

---

## 11. Module map (`pyMagCalc/magcalc/`)

| Module | Role |
|---|---|
| `cli.py` | Typer CLI: `run`, `validate`, `symmetry`, `fit`, `mcif`, `mcif-out`, `init`, `where`, `guard` |
| `schema.py`, `config_loader.py`, `yaml_io.py` | config validation, loading, canonical YAML output |
| `config_builder.py`, `generic_model.py` | symmetry expansion of atoms and bonds; `GenericSpinModel`; magnetic structures and supercells |
| `cif_utils.py`, `cell_utils.py`, `mcif.py` | CIF reading; primitive/standard cells and BZ paths; magnetic CIF import and export |
| `symbolic.py` | SymPy Holstein–Primakoff Hamiltonian and its Fourier transform |
| `core.py` | `MagCalc`: lambdified H(q), dispersion, S(Q,ω) (`calculate_sqw` → `sqw_amplitudes`, batched Colpa; the per-q pool behind `sqw_method: perq`), `sqw_batch`, powder, minimization, the fMagCalc hand-off, `DispersionEvaluator` |
| `linalg.py`, `numerical.py` | Colpa (`colpa_batch`, `SqwAmplitudes`), the per-q Bogoliubov fallback (`KKdMatrix`) and workers, `contract_cross_section_batch` (every cross-section), domains, powder sampling |
| `runner.py` | the task pipeline and the ground-state guards |
| `sun/lswt.py`, `sun/operators.py`, `sun/adapter.py` | SU(N) LSWT (`SUNModel`), local operator algebra, runner adapter |
| `sun/entangled.py`, `sun/dimer_series.py` | entangled units; linked-cluster dimer series with Dlog-Padé |
| `sun/kpm.py`, `sun/dynamics.py` | KPM S(q,ω); CP^(N−1) classical dynamics, quench, topological charge |
| `scga.py`, `thermal_mc.py`, `classical_dynamics.py`, `annealing.py`, `spiral_opt.py`, `corrections.py` | the beyond-LSWT methods (§5) |
| `stevens.py`, `ewald.py`, `form_factors.py`, `constants.py` | Stevens operators, Ewald dipolar sums, magnetic form factors, the one definition of each physical constant |
| `diffraction.py`, `binning.py` | elastic magnetic intensities; histogram binning and NeXus input |
| `neutron_data.py` | reduced INS data on its own bins: `BinnedData` (counts, exposure, true energy edges, the bin → Q cell → sample quadrature), `QFrame`, `T0EnergyMap`, the MDNorm and count-cell readers, a versioned native `.npz` |
| `forward.py` | `ForwardModel`: the expected count μ per data bin (exact energy integral over each bin through the resolution's `fraction(lo, hi, centre)`, crystals with mosaic, Q smearing, Bose factor); `GaussianEnergyResolution`; `poisson_deviance` |
| `resolution.py` | an instrument's energy resolution from its moderator pulse, choppers and flight paths: `ChopperSpectrometer`, `DiskChopper`, pulse shapes (`IkedaCarpenter`, `read_sns_source_table`), `cncs(...)`; `ChopperEnergyResolution.fraction` for `ForwardModel` |
| `qe_resolution.py` | the Q resolution and Q–E correlation (Level 1): exact `kinematics`, its 4-D covariance, geometry providers (`RunGoniometer`, `RotationScan`), `QEResolution`, whose kernel enters `ForwardModel`'s sample quadrature |
| `fitting.py` | lmfit wrapper for dispersion, S(Q,ω) and powder data |
| `plotting.py` | figures; `show_plot_if_possible` honours `MAGCALC_NO_GUI` |
| `kernels.py` | selection of the Rust kernels: `resolve`, `rust_for`, `fell_back`, `make_rng` |
| `provenance.py`, `_shadow_guard*.py` | `magcalc where` and the startup guard against a second importable copy |

Other directories: `tests/` (104 test files), `examples/` (`materials/`, `entangled/`,
`spinw_tutorials/` with 30 SpinW ports, `sunny_tutorials/` with all 9 Sunny tutorials,
`fitting/`), `gui/` (FastAPI and React Studio), `native/MagCalcStudio/` (SwiftUI macOS and
iOS app).

---

## 12. Where to read next

| File | What it is for |
|---|---|
| `CLAUDE.md` | config-authoring rules and the engine's hard errors; **read before touching any config** |
| `GAP_STATUS.md` | what is done and what it is validated against; internal conventions; the validation history |
| `OPEN_WORK.md` | what is next, each item with its injection point and its oracle |
| `TUTORIAL.md` | the user-facing reference for every config block, backend and method; mirrored to the GitHub wiki |
| `README.md` | overview, installation, CLI |
| `examples/*/README.md` | per-example physics and status tables |
| fMagCalc `README.md`, rsMagCalc `README.md` | building and using the compiled companions |

---

## 13. Starting work: a checklist

1. `pip install -e .` inside `pyMagCalc/`, then **`magcalc where`**. An editable install
   does not automatically win: its finder sits *after* `sys.path`, so a second checkout,
   your working directory or `PYTHONPATH` can silently swap the engine. Every `magcalc run`
   logs the engine line first; read it when behaviour is inexplicable.
2. Run the fast suite once, so you know what green looks like on your machine.
3. Read `CLAUDE.md`, then the parts of `GAP_STATUS.md` and `OPEN_WORK.md` that touch your
   task.
4. Make the change with a test pinned to an oracle (§9). For compiled code, run it under
   both fixture values.
5. Run `pytest -m "" -rs` before merging to `master`, and account for every skip.
6. **Update the documentation in the same change.** These files describe the system and
   must not drift from it:
   - `README.md`
   - `TUTORIAL.md`, plus the wiki mirror via `python tools/sync_wiki.py`
   - this guide
   - `GAP_STATUS.md` and `OPEN_WORK.md`
   - the companion READMEs

   When a change alters a feature, a default, a config key, a command, a test count or a
   convention described in any of them, update it in the same commit.
7. Commit as `kit-matan <kittiwit.mat@mahidol.ac.th>` (each repo's local git config sets
   this), so GitHub credits the work to the right account.

---

## 14. Where things stand (2026-10-30)

- **`calculate_sqw` runs on Colpa's method** (2026-10-30). The per-q process pool used
  `eig` at ±q with basis matching. The default is now batched Cholesky plus `eigh`,
  in-process, about 70–100× faster on 16 sites. Every cross-section and the three
  single-k channels are supported, through one vectorized contraction. Non-positive-
  definite q (Goldstone points, unstable q) still take KKdMatrix. The pool stays for one
  release as `calculation: {sqw_method: perq}`. It is pinned by every existing Sunny
  S(Q,ω) test and by an exact single-k = supercell identity for every cross-section.
  Colpa is also 4–5× faster end to end than fMagCalc's S(Q,ω), whose retirement is
  recommended (`GAP_STATUS.md`, "calculate_sqw on Colpa's method"). Two older defects
  turned up: the k_case 2 polarized clamp (OPEN_WORK item 9), and a test whose "T = 0 fit
  is biased 17 %" claim was rounding noise.

- **Phase 1 of the neutron-data layer** (2026-10-30). `MagCalc.sqw_batch` computes the
  one-magnon S(Q,ω) by Colpa's method on stacks of q, about 27,000 q/s on a 16-site
  cell. It equals the per-q path to 1e-9 and Sunny directly, and applies per-site g
  tensors on request (Sunny `apply_g=true`, pinned on an off-diagonal, non-symmetric g).
  `magcalc/neutron_data.py` reads Mantid MDNorm 4-D exports and per-run count cells into
  one `BinnedData`, with frame and t0 energy maps and an exact exposure-weighted Q
  quadrature. `magcalc/forward.py` predicts μ in every bin. A ferromagnetic chain's
  closed form, integrated by SciPy quadrature, is reached to 8e-7. It is a Python API
  only: there is no config block yet, and no fitting
  (`GAP_STATUS.md`, "Neutron data and the forward model, phase 1").

- **Phase 2 of the neutron-data layer, the energy resolution** (2026-10-30).
  `magcalc/resolution.py` builds a chopper spectrometer's energy line shape from its
  moderator pulse, chopper openings and flight paths, exactly rather than as a Gaussian,
  and `ForwardModel` integrates any line shape over each bin through `fraction(lo, hi,
  centre)`. `GaussianEnergyResolution` is unchanged, bit for bit. Given PyChop's own time
  widths, the propagation reproduces PyChop's CNCS widths to 3e-14. CNCS is built in;
  AMATERAS waits for its chopper geometry. `magcalc/qe_resolution.py` adds the Q
  resolution and the Q–E correlation through the sample quadrature; its covariance
  matches an independent ray trace to 0.3% (`GAP_STATUS.md`, "phase 2, the energy
  resolution" and "the Q-E resolution"; `OPEN_WORK.md` item 8).

- **Every planned Rust kernel is in** (K1–K5, plus a sparse exact KPM guard). The sixth,
  mixed-spin dipole S(Q,ω), went into fMagCalc instead of Rust, because fMagCalc already
  owned that path and lacked only the per-site prefactor. Measured speedups are recorded in
  `GAP_STATUS.md`, "Rust kernels K1…K5" and "K6".
- **The Ewald dipolar sum** precomputes its q-independent part once per cell
  (`ewald.DipoleEwald`), 63–84× faster per q. Its tensor A_ij(q) is pinned to Sunny
  element by element (`tests/test_ewald_sunny_tensor.py`).
- **The 1/S corrections** solve the k-mesh in batched chunks on every core, 25–27×
  faster, and their memory no longer grows with the mesh: 64³ at 16 sites takes about
  5 s in under 1 GB (`GAP_STATUS.md`, "1/S corrections: the k-mesh batched").
- **The annealing ground-state search** runs in rsMagCalc under `kernels: rust`, all
  starts in parallel: 0.36 s → 2 ms per 2000-sweep run at 16 sites, 2.8 s → 0.15 s at
  128 (`GAP_STATUS.md`, "anneal on K1's dense model").
- **Gapless and degenerate models under `backend: fortran`** now agree with NumPy and
  with Sunny. fMagCalc dropped every mode below 1e-3 meV, and gave degenerate magnons
  (every q of a collinear antiferromagnet) zero intensity at about one powder point in
  eight. Both are fixed (`GAP_STATUS.md`, "Gapless and degenerate S(Q,ω)").
- **The real-space bond builders** (SCGA, thermal MC, entangled, dimer series) no longer
  test every (site, image) pair in SymPy: 9× on SCGA's exchange cache at 128 sites.
- **The Luttinger–Tisza q≠0 guard** builds J(k) from one block per cell translation
  and scans its grid in batches, 12–19× faster with the same k*: 47 s → 3.3 s on YIG,
  29 s → 2.5 s on α-Cu₂V₂O₇. It had been the largest Python cost of a plain dipole run
  (`GAP_STATUS.md`, "Luttinger–Tisza guard: J(k) from J(R)").
- **The classical energy** the minimizer and the energy-audit guard use is assembled
  from the bond arrays, with only the on-site, Zeeman and biquadratic terms in SymPy,
  and built once per run instead of twice. A warm FeI₂ run goes from 24 s to 6.8 s.
  `calculation: {classical_energy: symbolic}` keeps the old whole-SymPy build
  (`GAP_STATUS.md`, "Classical energy from the bond arrays").
- **The dimer series expansion** (K7, 2026-10-29) runs each cluster's perturbation
  theory in rsMagCalc under `kernels: rust`, threaded over each cluster size's
  topologies. On the six-dimer Rb₂Cu₃SnF₁₂ cell order 6 takes 21 s instead of 115 s
  with the NumPy process pool, and order 7 6.4 min instead of 32 min
  (`GAP_STATUS.md`, "Rust kernels K7"). An earlier plan had excluded it as
  "symbolic"; it is numerical throughout.
  How many clusters run at once is set by memory, not by `n_workers` alone. Each
  cluster is charged its own scratch against `$RSMAGCALC_SCRATCH_GB`, or else half the
  available memory, cgroup limits included. Until 2026-10-31 a fixed 3 GB was divided
  by a level's largest cluster, which left order 8 on 2 threads of a 128-core node
  (`GAP_STATUS.md`, "Dimer series under `kernels: rust`: memory admits each cluster").
  The subtraction after each kernel call runs in forked processes on Linux
  (`DimerSeriesModel.subtract_workers`, default `n_workers`). The additivity check is
  logged once per cluster size; it used to be one line per cluster, 82 342 lines at
  order 8. Those lines were not noise. With DM, the T0 (Sz = 0) band's des Cloizeaux
  effective Hamiltonian is not cluster additive, because the T0 states share their
  sector with the vacuum. That block now goes through Hörmann and Schmidt's projective
  cluster-additive transformation (OPEN_WORK item 11, done the same day).
- **Biquadratic exchange is right off the collinear axis** (2026-10-29). The
  symbolic build had dropped, with only a WARNING, a biquadratic's cross-site terms on
  a bond with no bilinear coupling, and on a non-collinear intercell bond the part that
  vanishes for collinear moments. That made the spectrum depend on where the unit cell
  started. Both are fixed and pinned to Sunny, a dropped term now raises, and
  `pair_operator`, which dipole mode had ignored without a word, now raises there
  (`GAP_STATUS.md`, "Biquadratic off the collinear axis").
- **The energy audit runs on quartic models** (2026-10-29). It used to skip any model
  with a biquadratic or a rank ≥ 4 Stevens term without a word, leaving a supplied
  stationary maximum unguarded. It now relaxes on the energy itself and its gradient,
  pinned to the 120° triangular AFM with a biquadratic and to a B₄⁰ spin's closed form
  (`GAP_STATUS.md`, "Energy audit on a non-quadratic classical energy").
- **`minimize_energy` on quartic mixed-spin models** (2026-10-30). Its SymPy fallback gave
  every site the reference spin length; it now uses each site's own, pinned to a (1, ½)
  Néel chain's exact −0.9875 meV and its exact canting angle (`GAP_STATUS.md`,
  "minimize_energy on a quartic mixed-spin energy"). Nothing is open in `OPEN_WORK.md`.
- **Not supported:**
  - mixed spins in the classical dipole samplers: `thermal_mc` and
    `sampled_correlations` refuse them, while the SU(N) sampler accepts them;
  - Stevens terms of rank ≥ 4 in the classical samplers (use `mode: SUN`).

  Domain averaging **does** work in SU(N) and entangled modes
  (`tests/test_domains_sun.py`).
