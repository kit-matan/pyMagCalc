# pyMagCalc Tutorial

Welcome to the `pyMagCalc` tutorial! Use this guide to perform Linear Spin-Wave Theory (LSWT) calculations using the modern command-line interface.

## 1. Installation

Install `pyMagCalc` in editable mode to get the `magcalc` command:

```bash
# From the project root
pip install -e .
```

Verify installation:
```bash
magcalc --help
```

Optionally, install the compiled **fMagCalc** backend for much faster
dispersion / S(Q,ω) / powder calculations on dense q-grids — see
**§4c. Faster Calculations with the Fortran Backend**.

---

## 2. Quick Start (modern workflow)

The fastest way to get results is using the **pyMagCalc Studio** GUI followed by the CLI.

### Step 0: Using pyMagCalc Studio
The Designer allows you to generate robust, symmetry-consistent configurations without manual scripting.

1.  **Start Services**:
    Run the one-click launcher from the project root:
    ```bash
    ./start_magcalc.sh
    ```
    This handles everything (ports, backend, frontend, browser).

    > **Note**: The system now uses robust symmetry analysis (Pymatgen/Spglib). CIF imports will automatically be reduced to their unique Wyckoff positions (asymmetric unit).

    > **Tip**: MagCalc Studio is also available as a **native macOS & iOS app** (SwiftUI) with Metal-backed 3D rendering, embedded backend management, and keyboard shortcuts. See `native/MagCalcStudio/README.md` for build instructions.

2.  **Design**:
    -   **Load CIF**: Import your crystal structure. The app will automatically detect the space group and populate only the unique basis atoms.
    -   **Define Rules**: Add Bonding Rules (e.g., "Heisenberg" or "DM"). The system automatically expands these based on the structure's space group checking against symmetry constraints.
    -   **Configure Tasks**: Enable "Dispersion", "S(Q,w) Map", or "Powder Average" in the **Tasks & Plotting** tab.
    -   **Safe Parameters**: Parameters can use mathematical expressions (e.g., `1.5 * sqrt(3)`). The system evaluates these safely using SymPy.
3.  **Save**: Click **"Save to Disk"**. This creates an expanded `config_designer.yaml` in your workspace root.

    The app supports a seamless **Design -> Save -> Run** workflow. After saving, you can immediately run the configuration.

### Step 1: Initialize a Project (Legacy / Manual)
If you prefer manual configuration, create a new template:

```bash
mkdir my_project
cd my_project
magcalc init config.yaml
```

This creates a `config.yaml` file with placeholders for your crystal structure and magnetic interactions.

### Step 2: Edit Configuration
Open `config.yaml` and define your physics.

**Key Sections:**
-   `crystal_structure`: Lattice parameters and atom positions.
-   `interactions`: Can be defined as explicit pairs (e.g., `heisenberg`) or as `symmetry_rules` for distance-based automatic expansion.
-   `minimization`: Settings for finding the ground state.
    -   `initial_configuration`: **Crucial** for complex systems to avoid local minima. define `theta` and `phi` for each atom.
    -   `n_workers`: Number of CPU cores for parallel minimization (default: 1).
    -   `method`: **`anneal`** (Monte-Carlo, recommended), `steep`, or a gradient method. See §4e2.
    -   `early_stopping`: (gradient multistart only) stop after finding the same ground state N times; defaults to `max(10, 2 x n_sites)`.
-   `plotting`: Control output behavior.
    -   `show_plot`: Set to `true` to see plots on screen, `false` to save only.
    -   `plot_structure`: Visualize the minimized magnetic state.
-   `parameters`: Values for variables used in interactions (e.g., `J1: 3.23`).
-   `tasks`: Toggle `minimization`, `dispersion`, `sqw_map`, and `export_csv`.
-   `output`: Define output filenames for data (e.g., `disp_csv_filename: "my_data.csv"`).

### Step 3: Validate
The CLI uses a robust **Pydantic schema** to check if your configuration is valid before running heavy calculations. This provides clear error messages for missing fields or type mismatches.

```bash
magcalc validate config.yaml
```

### Step 4: Run
Execute the calculations. The CLI will handle ground state minimization, dispersion, and plotting automatically.

```bash
magcalc run config.yaml
```

Results (plots and data) are saved to the folders specified in the `output` and `plotting` sections (default: `plots/` and `data/`).

The first line of the run log names the engine that produced it:

```
21:04:55 - magcalc engine: /Users/you/magcalc/pyMagCalc/magcalc (git fe91e646792c)
```

### Which `magcalc` am I running?

```bash
magcalc where
```

Prints the package directory actually imported, its git HEAD, and — the part a manual
`python -c "import magcalc; print(magcalc.__file__)"` cannot give you — a warning listing
any *other* importable copy, exiting non-zero if there is one.

Worth knowing, because the editable install does not automatically win: `pip install -e .`
appends its finder to `sys.meta_path`, *after* the normal path search, so any `magcalc/`
directory on `sys.path` shadows it. `sys.path[0]` is the working directory for
`python -c` / `python -m magcalc`, pytest's rootdir under `pytest`, and a script's own
directory for anything in `scripts/`. So working inside a second checkout — an old copy, a
cloud-synced backup — silently runs that copy's engine. The symptoms look like bugs in the
code you are editing: a fix "not applied", a documented key "unsupported", a pinned number
that moved. Run `magcalc where` first.

### Automatic protection

`magcalc where` and the run-log line both live *inside* the package, so neither can report
the worst case: an old copy winning outright, with no such code in it. For that, install
the interpreter-startup guard once per interpreter:

```bash
magcalc guard --install        # report status with no flag; --uninstall to remove
```

It adds a `.pth` to site-packages — outside every copy of `magcalc` — which warns whenever
more than one is importable, whichever wins:

```
!!! magcalc shadow warning: 2 importable copies of `magcalc` !!!
    WINS -> /somewhere/old-checkout/magcalc
    also -> /Users/you/magcalc/pyMagCalc/magcalc
```

It costs ~0.25 ms per interpreter startup, is silent when only one copy exists, and also
raises a `MagcalcShadowWarning` so it shows up in pytest's warnings summary rather than
being swallowed by output capture. Set `MAGCALC_SHADOW_GUARD=off` when a second checkout
is deliberate (a git worktree, comparing versions). A fresh virtualenv starts unprotected
— `magcalc where` tells you which state you are in.

---

## 3. Configuration Reference

The `config.yaml` file is the heart of your calculation.

### Crystal Structure
Define the unit cell and magnetic atoms.

```yaml
crystal_structure:
  lattice_parameters:
    a: 7.3
    b: 7.3
    c: 17.2
    gamma: 120
  atoms_uc:
    - label: "Fe1"
      pos: [0.5, 0.5, 0.0]
      spin_S: 2.5
      magmom_classical: [0, 0, 1] # Initial guess
```

### Interactions
`pyMagCalc` supports both **explicit pair interactions** and **distance-based symmetry rules**.

#### 1. Distance-Based (Symmetry Rules)
Best for large systems. Define a distance, and the system finds all symmetry-equivalent bonds.
```yaml
interactions:
  symmetry_rules:
    # A bare `distance` is valid ONLY for the scalar type.
    - {type: heisenberg, distance: 3.23, value: J1}
    # Every NON-scalar type (dm, interaction_matrix, anisotropic_exchange,
    # kitaev) needs a reference bond instead: the matrix is propagated as
    # J' = R J R^T, and DM parts as axial vectors.
    - {type: dm, ref_pair: [Fe0, Fe1], offset: [0, 0, 0], value: [0, Dy, Dz]}
```

> **Three hard errors to know about** (each used to be a silent failure — a
> warning plus a Hamiltonian quietly missing a term):
> **(a)** a bare `distance` on a non-scalar rule raises; **(b)** a rule that
> matches no bond raises, instead of expanding to nothing; **(c)** an ambiguous
> `ref_pair` — several images of the same length for a directional rule, or a
> `distance` window spanning two orbits — raises and names the candidates, so pin
> it with `offset: [u, v, w]`. That last one bit a real config: two screw-related
> bonds differed by one ULP, and which one won the `<` comparison flipped the sign
> of a DM component for the whole orbit.

Run `magcalc symmetry <config> [--max-distance Å]` to see the bond orbits and the
symmetry-**allowed** exchange matrix for each — the fast way to choose `ref_pair`
bonds. `CLAUDE.md` §2 is the full reference for the rule grammar.

#### 2. Explicit Pair Interactions

List bonds directly (`heisenberg`, `dm_interaction`, `kitaev`, …) when the
coupling genuinely breaks the detected crystal symmetry — order-dependent
couplings, deliberately sub-symmetric models, or a case where spglib finds a
*higher* group than the physical one because only the magnetic sublattice is
listed. Say why in a comment when you do.

#### 3. Explicit Full 3×3 Exchange Matrices
For bond-dependent interactions (e.g., Kitaev, anisotropic exchange with DM), specify the full 3×3 exchange matrix per bond:
```yaml
interactions:
  - type: interaction_matrix
    pair: ["Ir0", "Ir2"]
    rij_offset: [-1, -1, 0]
    value:
      - [0.51, 0.0, 0.0]
      - [0.0, 0.51, 0.0]
      - [0.0, 0.0, 0.51]
```
The symmetric part encodes Heisenberg + anisotropic exchange, and the antisymmetric part encodes DM. Both directions of each bond must be listed (the engine does not auto-symmetrize). See `examples/spinw_tutorials/SW16_Na2IrO3_Kitaev/` for a full Kitaev honeycomb example.

#### 4. Terms beyond bilinear exchange

All of these live under `interactions:` as well; `CLAUDE.md` §5b is the full
reference with the conventions for each.

```yaml
interactions:
  sia_matrix:   [{matrix: [[Axx,0,0],[0,Ayy,0],[0,0,Azz]], atoms: [Fe0]}]
  stevens:      [{B: {'2,0': B20, '4,0': B40}, atoms: [Yb0]}]   # crystal fields
  biquadratic:  [{pair: [A, B], rij_offset: [0,0,0], value: -0.037}]
  pair_operator: [{pair: [A, B], rij_offset: [0,0,0], poly: [0, 1.0, -0.4]}]
  dipole_dipole: {method: ewald}        # exact lattice sum; prefer over `truncated`
```

The per-site **g-tensor** goes on the atom, not here
(`g: 2.0` | `[gxx,gyy,gzz]` | a 3×3 | `{g_par, g_perp, axis}`), which makes the
Zeeman term `μ_B B·g_i·S_i`.

An on-site or bond term that matches **no** bonds, or an unsupported Stevens
order, raises — it is never silently dropped.

> **Anisotropy renormalization.** Dipole mode replaces an on-site operator by its
> classical (s → ∞) polynomial, which overestimates a rank-k term at finite s.
> That is SpinW's convention and Sunny's `:dipole_uncorrected`, and it is
> pyMagCalc's default. Sunny's default `:dipole` applies the RCS correction
> instead, and the difference is large (a `stevens` B₄⁰ gap of 13.13 meV vs
> 1.53 meV at s = 2). Opt in with `calculation: {anisotropy_renormalization: rcs}`,
> or use `mode: SUN`, which is exact and needs no factor.

### Data Export (CSV)
To export your results to a readable CSV format (compatible with Excel/Origin):
```yaml
output:
  export_csv: true
  disp_csv_filename: "disp_results.csv"
  sqw_csv_filename: "sqw_results.csv"
```
**Formats:**
*   **Dispersion**: One row per Q-point: `qx, qy, qz, en0, en1, ...`
*   **S(Q,w)**: Tidy format (one row per mode): `qx, qy, qz, mode, energy, intensity`

---

## 4. Best Practices & Troubleshooting

### Avoiding Imaginary Energies
Imaginary energy eigenvalues mean the magnetic structure is **not the classical
ground state** — the spin-wave expansion is about the wrong state and the spectrum
is meaningless. The engine now **fails the run** in this case rather than writing a
plausible-looking plot (`calculation.on_imaginary: error`, the default; see §4f).

**Solution**: use the Monte-Carlo annealer, which is built to escape local minima:

```yaml
tasks: {minimization: true}
minimization: {method: anneal, num_starts: 4, n_sweeps: 2000, seed: 0}
```

See **§4e2** for the method comparison. Guiding the solver by hand with
`initial_configuration` is still supported and can help, but it is no longer the
primary answer.
*   For a 120-degree structure (e.g., Kagome), initialize spins at 0, 120, and 240 degrees.
*   Example (from `KFe3J/config_kfe3j.yaml`):

```yaml
minimization:
  initial_configuration:
    - atom_index: 0
      theta: 1.57 # 90 deg
      phi: 0.0
    - atom_index: 1
      theta: 1.57
      phi: 2.09 # 120 deg
    - atom_index: 2
      theta: 1.57
      phi: 4.18 # 240 deg
```

### Performance & Caching
*   Use `cache_mode: 'none'` (default) to avoid disk I/O. This is recommended for small systems or when rapidly iterating on symmetry rules.
*   Use `cache_mode: 'auto'` to reuse symbolic calculations for very large units cells where matrix construction is slow.
*   Set `calculate_dispersion: false` if you only want to change plot aesthetics (titles, limits) without re-running the physics.

---

## 4a. Reusing a Minimized Magnetic Structure

Energy minimization can be slow, and you usually only need to do it once. After a
minimization run you can **save the resulting structure** and reuse it as a fixed
input for later dispersion / S(Q,ω) / fitting runs — no need to re-minimize.

**In the GUI:** run a calculation with minimization enabled, then in the
**Run & Analyze** tab click **"Use as Manual Structure"** on the interactive
magnetic-structure result. This copies the per-spin directions into the
**Mag. Structure** tab (as a `generic` direction list) and turns minimization
**off**, so every subsequent run reuses that exact ground state.

**In a config file:** set a `magnetic_structure` block with `pattern_type:
generic` and one unit vector per spin, and disable minimization:

```yaml
tasks:
  minimization: false      # reuse the structure below instead of re-minimizing
  dispersion: true

magnetic_structure:
  enabled: true
  type: pattern
  pattern_type: generic    # one direction per spin, applied in atom order
  directions:
    - [0.0, 0.0, -1.0]
    - [0.0, 0.0,  1.0]
```

When minimization is off and a `magnetic_structure` is given, the runner applies
it directly to the spin model before the LSWT calculation. This same fixed
structure is used by fitting, so a fit does not re-minimize the ground state on
every iteration.

---

## 4b. Fitting to Neutron Data

pyMagCalc can fit the spin Hamiltonian to inelastic-neutron-scattering data of
three kinds, using [lmfit](https://lmfit.github.io/) under the hood:

| `fitting.type` | What is fitted | Data columns (CSV, `#` comments) |
| -------------- | -------------- | -------------------------------- |
| `dispersion`   | magnon peak positions E(Q) (single crystal) | `h, k, l, E, sigma [, mode]` |
| `sqw`          | single-crystal intensity I(Q, ω) | `h, k, l, energy, intensity, error` |
| `powder`       | powder-averaged intensity I(\|Q\|, ω) | `\|Q\|, energy, intensity, error` |

The fit keeps **one** calculator alive across the whole minimization (the
symbolic Hamiltonian is built once), so it is far faster than re-instantiating
the model per evaluation. For `dispersion` fits it goes one step further:

> **Fast dispersion path (default).** A `dispersion` fit compiles the symbolic
> Hamiltonian **once** into a numerical function of `(q, S, parameters)` — a
> `DispersionEvaluator` — so every iteration is a pure array/eigenvalue
> evaluation (~ms per q-point) with **no per-call `subs` or re-`lambdify`**.
> For large magnetic cells (many spins) this is **orders of magnitude** faster
> than the legacy path that re-lambdified on each parameter update: e.g. a
> 6-sublattice kagome model over 176 q-points drops from ~25 s per iteration
> to ~70 ms (a full six-parameter fit finishes in seconds rather than hours).
> The results are numerically identical to `calculate_dispersion`. Disable it
> with `fitting.fast: false` to force the legacy per-iteration path.

At the end of a fit the shared calculator is left **synchronized with the
best-fit parameters**, so any subsequent `dispersion` / `sqw` / `powder` /
plotting task (via `magcalc run`) renders the optimized model directly against
the data.

### Running a fit (CLI)

```bash
magcalc fit examples/fitting/fit_dispersion.yaml
```

This writes `fit_report.txt` (lmfit report + uncertainties), `fit_params.yaml`
(best-fit values) and `fit_comparison.png` (data vs. best-fit model). You can
also set `tasks: { fit: true }` in any config and use `magcalc run`. Output
names are configurable via `output.fit_report_filename`,
`output.fit_params_filename` and `plotting.fit_plot_filename`.

### The `fitting:` block

```yaml
tasks:
  fit: true
  plot_fit: true

fitting:
  type: dispersion           # dispersion | sqw | powder
  data_file: disp_data.txt   # resolved relative to the config file
  method: leastsq            # any lmfit method (least_squares, differential_evolution, ...)
  vary: [J1, J2]             # subset of parameter names to optimize (scalars only)
  bounds:
    J1: [0.0, 5.0]
  expr:                      # optional lmfit constraint expressions between params
    J2: "0.1 * J1"
  match: nearest             # dispersion only: nearest | mode (band assignment)
  fast: true                 # dispersion only: compile-once fast path (default true)
  lineshape: lorentzian      # intensity fits: lorentzian | gaussian

  # Intensity fits (sqw / powder) add three nuisance parameters:
  scale:            { value: 1.0, vary: true }
  background:       { value: 0.0, vary: true }
  energy_broadening:{ value: 0.3, vary: false }   # FWHM in meV
```

* Parameters not listed in `vary` are held fixed at their `parameters:` values.
* `dispersion` assigns each data point to the closest model band by default; add
  a 1-based `mode` column and set `match: mode` to pin specific branches.
* `fast` (dispersion only, default `true`) toggles the compile-once evaluator
  described above; set it `false` only to cross-check against the legacy path.
* `bounds` (`{name: [min, max]}`) and `expr` (`{name: "expression"}`) map
  directly onto lmfit parameter hints, so you can constrain or tie parameters.
* Intensity fits broaden the modes with the same line-shape used by the S(Q,ω)
  plot (`lineshape`), scaled by `scale` plus a flat `background`.

### Programmatic use (Python API)

For custom fitting drivers or repeated dispersion evaluations at many
parameter sets, compile the evaluator yourself:

```python
import magcalc as mc

calc = mc.MagCalc(spin_model_module=model, spin_magnitude=0.5,
                  hamiltonian_params=p0)

evaluator = calc.compile_dispersion_evaluator()   # one-time lambdify
E = evaluator.energies(q_cart)                     # (N_q, n_modes), default params
E = evaluator.energies(q_cart, new_params)         # any params, no re-compile
```

`DispersionEvaluator.energies` returns mode energies (ascending per q-point)
identically to `MagCalc.calculate_dispersion`, but skips all symbolic work, so
it is ideal inside `scipy.optimize` / `lmfit` residual functions. It never
mutates the calculator's state. **Note:** the magnetic structure is baked into
the compiled Hamiltonian; if you install a new ground state (e.g. re-minimize
via `mpr`), recompile the evaluator.

### GUI

The **Data Fitting** tab lets you choose the data type, upload a data file,
pick which parameters vary (with optional bounds), select an lmfit method, and
run the fit. Results (report, best-fit parameters, comparison plot) appear in
the **Run & Analyze** tab.

### Limitations (v1)

* Only **scalar** parameters can be varied (vectors such as a field direction
  stay fixed).
* The magnetic ground state is held fixed during the fit (it is not
  re-minimized as the parameters change). When the exchange parameters move the
  ground state significantly, wrap the fit in a short **outer loop**:
  re-minimize the structure and recompile the evaluator at the new best-fit
  parameters, then refit — usually 2–4 iterations to converge. (See the
  `examples/fitting/` scripts for this pattern.)
* Intensity fits use a single global scale, flat background and a simple
  Gaussian/Lorentzian energy broadening — not a full instrument-resolution
  convolution.

---

## 4c. Faster Calculations with the Fortran Backend (fMagCalc)

[fMagCalc](https://github.com/kit-matan/fMagCalc) is a compiled companion
package that runs pyMagCalc's numerical hot path — the per-q diagonalization
of the Bogoliubov Hamiltonian, the S(Q,ω) intensity contraction, and powder
averaging — in an OpenMP-parallel Fortran core backed by LAPACK. The physics
and results are identical to the NumPy path (parity to ~1e-13); the win is
speed: the compute kernel is 18–100× faster than the multiprocessing NumPy
path, and on the KFe3J example with 8000 q-points the end-to-end S(Q,ω) run is
roughly an order of magnitude faster. It shines on dense q-grids and powder
averages; for a handful of q-points the default NumPy path is fine.

pyMagCalc never *requires* fMagCalc: if it is missing or broken, any
`backend: fortran` request logs a WARNING and falls back to NumPy, so runs
always complete.

### Installing fMagCalc

fMagCalc compiles during installation, so you need a Fortran compiler,
CMake ≥ 3.20, and LAPACK first:

```bash
# macOS
brew install gcc cmake          # gfortran + CMake; LAPACK comes from Accelerate

# Debian/Ubuntu Linux
sudo apt install gfortran cmake libopenblas-dev
```

Then install with pip, either directly from GitHub or from a local clone:

```bash
pip install git+https://github.com/kit-matan/fMagCalc
# or, from a local checkout:
pip install /path/to/fMagCalc
```

Verify the compiled library is found (must print `ctypes`, not `subprocess`):

```bash
python -c "import fmagcalc; print(fmagcalc.backend)"
```

### Using it from a config file (CLI)

Add `backend: fortran` to the `calculation` block — it applies to the
dispersion, S(Q,ω), and powder tasks of that run:

```yaml
calculation:
  cache_mode: auto
  backend: fortran        # numpy (default) | fortran
```

Then run as usual:

```bash
magcalc run config.yaml
```

The log line `Calculating dispersion... (backend=fortran)` confirms the
selection, and `Dispersion computed via fMagCalc Fortran backend` confirms it
was actually used (rather than falling back).

### Using it from the GUI

In **pyMagCalc Studio**, open **Tasks & Plotting → Calculation Settings** and
set **Compute Backend** to *Fortran (fMagCalc)*. The setting is saved into the
generated config as `calculation.backend`.

### Using it from the Python API

`calculate_dispersion`, `calculate_sqw`, and `calculate_powder_average` all
accept a `backend` keyword:

```python
E = calc.calculate_dispersion(q_list, backend="fortran")
sqw = calc.calculate_sqw(q_list, backend="fortran")
powder = calc.calculate_powder_average(q_mags, num_samples=200, backend="fortran")
```

### Troubleshooting

* **`backend='fortran' requested but the fMagCalc package could not be
  imported`** — fMagCalc is not installed in the active environment. Install
  it as above (check `pip show fmagcalc`), or point the `FMAGCALC_PATH` env
  var at a source checkout's `python/` directory (development fallback).
* **`fMagCalc's compiled library is not available`** — the package imported
  but `fmagcalc.backend` is `subprocess`, meaning `libfmagcalc` was not found.
  Reinstall with pip (which compiles it into the package), or in a source
  checkout run `cmake -S . -B build && cmake --build build`. The
  `FMAGCALC_LIB` env var can point at a specific library file.
* **No Fortran compiler / CMake at install time** — `pip install` fails with a
  build error; install the prerequisites above and retry. pyMagCalc itself is
  unaffected — just leave `backend` at `numpy` until fMagCalc installs.
* Fitting note: `calculation.backend` is honored by fits too — `sqw` and
  `powder` fits evaluate each iteration through the selected backend, so
  fMagCalc speeds them up directly. `dispersion` fits with `fast: true` (the
  default) instead use the compile-once `DispersionEvaluator` (§4b) and do not
  need fMagCalc.

---

## 4d. Spiral (Rotating-Frame) Structures

pyMagCalc supports incommensurate magnetic orders via a **rotating-frame**
formulation. Instead of building a (possibly very large) magnetic supercell,
you specify a propagation vector **k** and a rotation axis; the engine
constructs the LSWT Hamiltonian in the rotating frame and solves for the
central magnon branch (the phason + optical modes).

### Configuration

```yaml
magnetic_structure:
  enabled: true
  type: single_k           # `type: spiral` is a deprecated alias (same fields)
  k: [0.23, 0.0, 0.0]      # propagation vector in RLU of the unit cell
  axis: [0.0, 0.0, 1.0]    # rotation axis (spins lie ⊥ to this)

tasks:
  minimization: false       # structure is fixed by k and axis
  dispersion: true
```

The spins are automatically placed perpendicular to `axis` with the
inter-site phase determined by **k**. For multi-sublattice spirals (e.g.,
120° triangular order on kagome), add a `local_directions` pattern:

```yaml
magnetic_structure:
  enabled: true
  type: single_k
  k: [0.333333, 0.333333, 0.0]
  axis: [0.0, 0.0, 1.0]
  local_directions:           # per-sublattice starting directions (⊥ axis)
    - [1.0, 0.0, 0.0]
    - [-0.5, 0.866025, 0.0]   # 120°
    - [-0.5, -0.866025, 0.0]  # 240°
```

`local_directions` is the **rotating-frame** convention. Two alternatives take
the same slot: `S0` (lab-frame cell-0 directions, the SpinW/Sunny convention —
the engine back-rotates them per site by `R(−2πk·d_i, axis)`), and a `u`/`v`
basis. They are not interchangeable; picking the wrong one silently gives a
different structure.

### Optimizing k, and when the rotating frame is not enough

```yaml
magnetic_structure:
  type: single_k
  minimization: {enabled: true, optimize_k: true, lt_guess: true, k_grid: 24}
  # optional: optimize_axis: true
```

This is the Sunny `minimize_spiral_energy!` analogue: it optimizes (k, spin
directions) from a Luttinger–Tisza initial guess and writes
`optimized_structure.yaml`.

Two limits worth knowing:

* **The rotating frame assumes the Hamiltonian is rotationally invariant about
  `axis`.** A DM vector, an SIA axis or a field that is not parallel to it breaks
  that, and the engine warns
  (`magnetic_structure.enforce_rotational_symmetry: warn|error|off`).
* **`dipole_dipole: {method: ewald}` now works with `single_k`** (2026-08-13; it
  used to refuse). It is exact when `A(q)` is uniaxial about the spiral axis and
  when 2k is a reciprocal-lattice vector; otherwise it drops the ±2k umklapp —
  the same approximation Sunny makes, worth ~10–20 % of the dipolar shift — and
  the engine measures the dropped weight and warns. For an exact answer at
  commensurate k use `crystal_structure.magnetic_supercell` instead.

For a commensurate k you can always use the real-space route instead:
`crystal_structure: {magnetic_supercell: [n1, n2, n3]}` (or `'auto'`) replicates
the chemical cell and turns the `single_k` structure into the equivalent
real-space pattern. Prefer the rotating frame for true incommensurate spirals
(exact, no ghost bands); prefer the supercell for collinear k = ½-type and
multi-k states.

### Validated examples

| Tutorial | Description | Validation |
|----------|-------------|------------|
| SW03 | Frustrated J1–J2 chain | Exact analytic helix, error ~1e-12 |
| SW08 | √3×√3 kagome AFM | Band-by-band match with 9-site supercell (~1e-8) |
| SW15 | Ba₃NbFe₃Si₂O₁₄ (langasite) | Chirality-dependent k_z matches tutorial |
| SW18 | Distorted kagome | Classical E/site matches to 4 significant figures |

See `examples/spinw_tutorials/` for runnable configs.

> **Satellites are computed for you now.** This note used to say "the full
> neutron cross-section also has weight at ω(q±k); overlay the ±k-shifted
> branches yourself". The engine adds those branches directly —
> `satellites: true` (in `magnetic_structure` or `tasks`), **on by default for
> S(Q,ω), off by default for dispersion**. The result then carries `3·nspins`
> modes, channel-major `[q−k | q | q+k]`, and S(Q,ω) uses the Toth & Lake
> three-channel projection, so the satellite intensities are right rather than
> merely present. Validated against Sunny `SpinWaveTheorySpiral` and SpinW.

---

## 4e. Mixed-Spin Models

pyMagCalc supports models where different magnetic sites carry **different
spin magnitudes**. Each site's Holstein–Primakoff expansion is scaled by its
own `spin_S`, producing bands on distinct energy scales.

### Configuration

Simply assign different `spin_S` values to atoms in the crystal structure:

```yaml
crystal_structure:
  atoms_uc:
    - { label: Cu1, pos: [0.0, 0.0, 0.0], spin_S: 0.5, ion: Cu2+ }
    - { label: Cu2, pos: [0.5, 0.0, 0.0], spin_S: 0.5, ion: Cu2+ }
    - { label: Fe1, pos: [0.0, 0.5, 0.0], spin_S: 2.0, ion: Fe2+ }
    - { label: Fe2, pos: [0.5, 0.5, 0.0], spin_S: 2.0, ion: Fe2+ }
```

Interactions (Heisenberg, DM, interaction matrices) work as usual — the
engine automatically accounts for the different spin magnitudes in the LSWT
matrix elements.

See `examples/spinw_tutorials/SW19_different_ions/config.yaml` for a
complete example: Cu²⁺ (S=½) + Fe²⁺ (S=2) AFM chains with distinct energy
scales (~1.4 meV vs ~4 meV).

> **This limitation is GONE** (it used to read "the S(Q,ω) intensity prefactor
> currently uses the single reference `S`, so relative intensities are
> approximate"). The prefactor is **√(S_i/2) per site**, so mixed-spin
> intensities are now correct as well as the dispersions — the old global factor
> made every site whose S differed from the reference wrong by √(S_i/S_ref),
> a 60 % error on a Cu(½)+Fe(2) model. Two things follow from it:
>
> * the **Fortran backend still applies the global factor**, so a mixed-spin
>   S(Q,ω) falls back to NumPy automatically (dispersions are unaffected);
> * **SU(N) mode supports mixed spin too** (`sun/lswt.py` indexes through a
>   per-site offsets table), pinned by the exact decoupled-sublattice identity
>   for (½,1), (1,3/2) and (½,3/2). `model.M`/`model.N` are `None` for such a
>   cell — use `Ms`/`Ns`/`D` — so anything still assuming a uniform N fails
>   loudly instead of silently using site 0's value.

---

## 4e2. Finding the Ground State (`method: anneal`)

LSWT is an expansion about a classical energy **minimum**. Expand about anything
else and the spectrum is meaningless — so the ground-state search matters as much
as the Hamiltonian.

```yaml
tasks: {minimization: true}
minimization:
  method: anneal        # Monte-Carlo annealing -- the recommended choice
  num_starts: 4         # independent annealing runs
  n_sweeps: 2000        # temperature steps (each = one attempted move per spin)
  seed: 0
  # optional: T_start / T_end (meV; default from the coupling scale), polish: true
```

| method | what it is | when |
|---|---|---|
| `anneal` (`monte_carlo`) | Metropolis + geometric cooling (SpinW `anneal`, Sunny `LocalSampler`), then a polish that is **kept only if it lowers the energy** | **default choice** — crosses barriers, does not get trapped |
| `steep` (`optmagsteep`) | align each spin with its local field (SpinW `optmagsteep`) | fast polisher; **monotone**, so it cannot escape a local minimum |
| `L-BFGS-B`, `TNC`, … | legacy random multistart in (θ, φ) | compatibility; weaker than it looks (polar coordinate singularities) |

**Why this matters.** On SW20 in field (16 sites = 32 free angles, true minimum
**−9.662153 meV**), multistart L-BFGS with 24 starts returned −8.994590 — a local
minimum. Raising it to 200 starts does reach the true value, but only 3 of those
200 starts find it; annealing finds it in **1 run out of 1, in ~0.9 s**. (These
figures were re-measured after the 2026-07 Zeeman calibration fix; the older
−5.716074 / −5.338112 pair quoted here came from a half-strength field.)

The engine **fails the run** if the structure is not a minimum
(`calculation.on_imaginary: error`, the default) — see §4f. Every method reports
`hits` (how many runs reached the best energy) and warns when `hits == 1`.

> **Reproducibility is not correctness.** The advice here used to end at "accept a
> ground state only when the energy is reproducible across several `seed` values",
> and that is necessary but not sufficient: a *deterministic* bug downstream of a
> stochastic search reproduces perfectly. `anneal`'s polish used to be taken
> unconditionally, and on a model with a large single-ion term it walked away from
> the minimum Metropolis had already found — returning a local **maximum** on 4 of
> 4 seeds, reporting exactly the consensus that is supposed to certify a ground
> state. Fixed 2026-08-13 (the polish is kept only if it lowers the energy), and
> the ground-state guards below are the second line of defence — which is why
> running with `on_imaginary: warn|off` costs more than it looks.

---

## 4f. Modeling the Measurement (temperature, twins, resolution, cuts)

These options shape the computed **intensities** (never the mode energies) so
S(Q,ω), powder and constant-energy-cut outputs can be compared directly with
inelastic-neutron-scattering data.

### Sample environment (`calculation:`)

```yaml
calculation:
  # Finite temperature: every mode intensity is multiplied by the thermal
  # Bose prefactor |1/(1 - exp(-E/kT))| (energy loss: n+1; gain: n).
  temperature: 5.0            # Kelvin

  # Magnetic/structural domains (twins). Shorthand for an n-fold axis:
  domains: {axis: [0, 0, 1], n_fold: 3}
  # ... or an explicit, COMPLETE list (include the angle-0 domain):
  # domains:
  #   - {axis: [0, 0, 1], angle: 0,   weight: 1}
  #   - {axis: [0, 0, 1], angle: 120, weight: 1}
  #   - {axis: [0, 0, 1], angle: 240, weight: 1}

  # Cross-section: 'perp' (unpolarized default), 'trace', or a lab-frame
  # tensor component 'xx' | 'yy' | 'zz' | 'xy' | ... (signed, real part).
  cross_section: perp
```

**Polarized and Blume–Maleev cross-sections.** With the polarization along **q**
all magnetic scattering is spin-flip and the two beams differ by the chiral term,
so `chiral` / `sf+` / `sf-` are available as plain strings. For an arbitrary
polarization axis or a BM frame component, pass a mapping:

```yaml
calculation:
  cross_section: {polarization: [0, 0, 1], channel: sf}    # sf | sf+ | sf- | nsf
  # or a Blume-Maleev component (the Sunny `ssf_custom_bm` analogue):
  # cross_section: {bm: {u: [1, 0, 0], v: [0, 1, 0]}, component: '23'}
```

`u`/`v` (or `normal`) are **Cartesian** lab vectors, as `domains.axis` is; the BM
axes follow Sunny (`e1 = q̂`, `e3 = the scattering-plane normal`, `e2 = e3 × q̂`),
and a q outside the plane is a hard error checked up front rather than returning
an all-NaN map. `P ∥ q` reproduces the plain `sf±` strings bit-for-bit, and
`SF + NSF = perp` for any P.

> Careful with "chiral vanishes for a collinear structure": that holds **per band
> only when P ∥ q**. A collinear magnet's two magnons are degenerate and
> oppositely handed, so at general P the chirality is non-zero band by band and
> cancels only in the band *sum* — and how it splits between the degenerate pair
> is basis-dependent, i.e. not an observable at all.

**Absolute normalization.** pyMagCalc's S(Q,ω) **equals Sunny's**, pinned band by
band on a ferromagnet (S = ½, 1, 2), a Néel antiferromagnet and a non-collinear
helix. (This entry used to claim a 3/4 convention difference. It was wrong — the
factor lived in hardcoded reference numbers in a test that compared only a ratio.
A clean constant factor is a bug until proven otherwise.) The one real difference
is not an overall factor: Sunny's `ssf_perp` applies the g-tensor by default, so
it is 4× ours at g = 2 — compare against `ssf_perp(sys; apply_g=false)`.

Domain averaging samples the unrotated model at `R^T q` per domain — exact for
`perp`/`trace` (the polarization projector transforms covariantly) and
rejected for tensor components. Modes are concatenated domain-major, so a
3-domain S(Q,ω) has `3 × n_modes` columns per q. Powder averages skip domains
(a spherical average is rotation-invariant); dispersion and fitting remain
single-domain.

### Instrument resolution (`plotting.resolution`)

```yaml
plotting:
  resolution:
    # FWHM(E): scalar, or numpy.polyval coefficients (HIGHEST power first —
    # the SpinW sw_instrument 'dE' polyfit convention).
    de_fwhm: [-0.0125, 0.107143, -0.141071, 0.059286]
    shape: gaussian           # default gaussian when de_fwhm is given
    dq_fwhm: 0.05             # Gaussian smoothing along the q axis (1/A)
    ei: 25.0                  # direct geometry: mask E > Ei ...
    two_theta: [5, 130]       # ... and |Q| outside detector coverage
                              #     (powder maps only)
  energy_grid_step: 0.01      # energy grid of the map (default 0.05 meV)
```

See `examples/spinw_tutorials/SW37_.../config.yaml` for the tutorial's cubic
dE(E). For an arbitrary dE function (not a polynomial), call
`magcalc.plotting.broaden_spectrum` with per-mode `width=` from Python.

### Constant-energy cuts on a 2-D q grid (`tasks.energy_cut`)

```yaml
tasks: {energy_cut: true}
energy_cut:
  origin: [0.0, 0.0, 0.0]                  # RLU corner of the grid
  axis1: {vec: [4.0, 0.0, 0.0], points: 121}
  axis2: {vec: [0.0, 4.0, 0.0], points: 121}
  cuts:
    - {center: 3.75, fwhm: 0.25}           # Gaussian energy window
    - {band: [3.5, 4.01]}                  # hard integration window
```

Writes `energy_cut_data.npz` (grid, panels, labels) and a panel figure
(`plotting.energy_cut_plot_filename`). Intensities inherit `temperature`,
`domains` and `cross_section`. See SW10 for the worked example.

---

## 4g. SU(N) mode and entangled units (dimers / trimers)

Dipole LSWT expands each spin as a single Holstein–Primakoff boson about a
classical moment. Two engines go beyond that, both selected under `calculation:`.

### SU(N) mode — single-ion / multipolar excitations

`calculation: {mode: SUN}` gives each site a full N = 2S+1 level Hilbert space with
N−1 bosons (as in Sunny's `:SUN`). This captures single-ion (crystal-field /
quadrupolar) excitations that dipole LSWT structurally cannot — e.g. the FeI₂
quadrupolar band. For S=½ (N=2) it is identical to dipole LSWT.

```yaml
calculation: {mode: SUN}
tasks: {dispersion: true, sqw_map: true}
```

It reuses the same `crystal_structure` / `interactions` / `magnetic_structure`
blocks. Single-ion anisotropy and Stevens operators feed the on-site N×N term. The
runnable, Sunny-validated example is `examples/materials/FeI2/config_fei2_sun.yaml`
(`examples/sunny_tutorials/S03_FeI2_SUN/` is a README pointing at it, not a config).

### Entangled units — valence-bond solids (dimers, trimers)

When the ground state is a product of **singlet dimers** (a valence-bond solid), the
moment is zero and dipole/single-site LSWT see nothing. `calculation: {mode:
entangled}` groups spins into units and treats each unit as ONE effective SU(N) site:
the intra-unit coupling is diagonalized exactly (the reference is the unit ground
state — e.g. a dimer singlet) and the excitations are the **triplons**.

```yaml
calculation: {mode: entangled}
units: [[0, 1], [2, 3]]           # each unit = a list of site indices
# a dimer that straddles the cell boundary: [i, [j, [ox, oy, oz]]]
```

Validated on a dimer chain (triplon `ω(q)=√(J²−JJ′cos2πq)`), Cu₅SbO₆, and the
Rb₂Cu₃SnF₁₂ pinwheel VBS (`examples/entangled/`). The harmonic bond-operator level is
exact only for weak inter-dimer coupling; for strong coupling (J′≈J) add the
high-order **dimer series expansion**:

```yaml
calculation: {mode: entangled, series_order: 5, series_resum: dlog_pade}
```

`series_order: N` switches the dispersion to a linked-cluster expansion to order N in
all inter-dimer couplings, resummed per band with Dlog-Padé (the method of Matan
*et al.* / the Rb₂Cu₃SnF₁₂ analysis). See
`examples/entangled/Rb2Cu3SnF12/series_dispersion.py`.

## 4h. Beyond LSWT: diffuse, thermal, and dynamical methods

Tasks for regimes an expansion about an ordered state does not cover. The classical
ones are paramagnetic-friendly, so run **alone** they auto-skip the LSWT
ground-state guard (no ordered state required); combine them with an LSWT task and
the guard re-arms.

```yaml
# (a) SCGA — paramagnetic diffuse S(q) above T_N (self-consistent Gaussian)
tasks: {scga: true}
scga: {temperature: 1.5, mesh_density: 20, cross_section: perp}   # temperature = kT (meV)

# (b) Thermal Monte-Carlo — <E>, C, magnetization, susceptibility vs T (parallel tempering)
tasks: {thermal_mc: true}
thermal_mc: {temperatures: [0.2, 0.5, 1, 2, 4], supercell: [6, 6, 1],
             n_sweeps: 4000, n_equil: 1500}

# (c) SampledCorrelations — classical-dynamics S(q,ω) (full thermal lineshape)
tasks: {sampled_correlations: true}
sampled_correlations: {temperature: 0.5, supercell: [16, 1, 1], dt: 0.02,
                       n_steps: 2048, n_traj: 8, therm_sweeps: 2000,
                       window: cosine, subtract_elastic: true}   # lineshape: see 4h

# (d) KPM — Chebyshev S(q,ω) with no diagonalization (large SU(N)/entangled cells)
calculation: {mode: SUN}
tasks: {kpm_sqw: true}
kpm: {e_min: 0, e_max: 10, e_step: 0.05, fwhm: 0.1, tol: 0.02}    # or moments: N

# (e) Wang-Landau — the density of states g(E): ONE run, every temperature
tasks: {wang_landau: true}
wang_landau: {supercell: [4,4,1], temperatures: [0.25,0.5,1,2,4], n_bins: 100,
              f_final: 1.0e-6}

# (f) Static / energy-integrated correlations — LSWT band sum, and the classical
#     instantaneous <|S(q)|^2> with no dynamics at all (two different things)
tasks: {static_sqw: true, static_correlations: true}
```

**Site-level disorder** is available to every real-space classical sampler above
(`thermal_mc`, `sampled_correlations`, `static_correlations`, `wang_landau`). A
vacancy deletes the site's rows/columns from the classical energy, so it removes
every bond it took part in; `periodic: false` on an axis drops the bonds that wrap
it. LSWT does not support disorder — its front end is symbolic and per-cell — so
for disordered *spectra* the route is a supercell plus KPM
(`sun.lswt.apply_bond_disorder`, Sunny tutorial 09's recipe).

```yaml
thermal_mc: {supercell: [8, 8, 1], disorder: {vacancy_concentration: 0.1, seed: 0},
             periodic: [true, true, false]}
```

**Integrators for (c).** Thermalize by Metropolis or by the `langevin_step`
thermostat; measure with `integrator: 'rk4'` or `'midpoint'`. The implicit
midpoint rule is symplectic — energy drift 1e-12 against RK4's 8e-5 over a long
run, and |S| conserved exactly without renormalizing. `suggest_timestep` picks
`dt` from the largest local field.

> **Fixed 2026-08-15 — read this if you ran an anisotropic model through these
> samplers before that date.** The supercell builder they share
> (`thermal_mc.build_supercell`) assembled the classical energy from the bond list
> alone, so `single_ion_anisotropy` / `sia_matrix` / `stevens` did **not** reach
> `thermal_mc`, `wang_landau`, `static_correlations` or the classical
> `sampled_correlations` — measured as exactly absent on a model with D = 19. Those
> runs were exchange-only. Exchange-only models were unaffected, as was the annealer
> (`minimization`), which uses a different and correct builder. One new limit comes
> with the fix: a **Stevens term of rank k ≥ 4 now raises** here, because its
> classical polynomial is quartic/sextic and cannot live in `E = ½mᵀHm + bᵀm` at all
> — use `calculation: {mode: SUN}`, which carries the full operator.

KPM never diagonalizes, which is the point of it — and means it cannot notice by
itself that it is expanding about a non-minimum (there is no Cholesky to fail, so no
imaginary energy to report; it would return a smooth, plausible, wrong S(q,ω)). The
task therefore checks `min eig H₂(q) ≥ 0` at **every q it computes**, which the
ordinary ground-state guards cannot substitute for: they run once, on the reference
state within its own cell, and a state can be a genuine in-cell minimum while being
unstable to a modulation that cell cannot hold. It costs 1-5 % of the KPM work.
`calculation.on_imaginary` downgrades or disables it with the other two guards, and
`calculation.h2_rel_tolerance` (default 1e-6, relative to ‖H₂‖) sets the threshold.
A model you build or perturb in a **script** is not seen by the runner, so call
`model.assert_stable(qs_cart)` yourself — see
`examples/sunny_tutorials/S09_triangular_AFM/disorder_kpm.py`.

Each evaluates on the config's `q_path` (SCGA / KPM / SampledCorrelations) or the
temperature list (thermal MC) and writes an `.npz` (`scga.npz`, `thermal_mc.npz`,
`sampled_correlations.npz`, `kpm_sqw.npz`). What each is validated against:

| Task | Method | Independent oracle |
|---|---|---|
| `scga` | self-consistent Gaussian, `kT(λ+J(q))⁻¹` | Sunny SCGA (square + kagome) + exact chain |
| `thermal_mc` | parallel-tempering Metropolis on a PBC supercell | Langevin function + exact classical dimer |
| `sampled_correlations` | Landau–Lifshitz on thermal states (RK4 or symplectic midpoint), space-time FFT | Larmor freq., energy conservation, LSWT dispersion, and the equal-time sum rule for the absolute scale |
| `kpm_sqw` | para-unitary Chebyshev of the LSWT spectral function | the engine's own exact diagonalization, on a non-collinear cell and in the chiral channel |
| `wang_landau` | flat-histogram sampling of g(E) | the classical dimer, whose g(E) is EXACTLY flat in closed form |
| `static_correlations` | instantaneous ⟨\|S(q)\|²⟩ over Metropolis samples | free-spin sum rule (2S²/3 perp, S² trace) exactly, at every q and T |

The classical intensities are on the **same absolute scale as `calculate_sqw` and
Sunny** — per chemical cell, with the 1/2π of the time transform (fixed 2026-08-13;
they used to be 2π/dt ≈ 314× too large and a further `n_atoms` too small).

**Integrate over the feature, not the whole frequency axis.** By default no
time-domain window is applied, so leakage weighted by the linearly growing
classical→quantum factor adds ~16 % if you sum everything out to the Nyquist
frequency. Two knobs, added 2026-08-15, and **use them together**:

```yaml
sampled_correlations:     {window: cosine, subtract_elastic: true}
sun_sampled_correlations: {window: cosine, subtract_elastic: true}
```

`window: cosine` is Sunny's lag window. It is exactly a [¼, ½, ¼] smoothing of the
spectrum — one bin Δω = 2π/T of broadening, no more, and it preserves the two-sided
ω-integral exactly. It is **opt-in here and Sunny's default there** for a measured
reason: the same one-bin smear lands on the elastic delta of an ordered magnet, where
the classical→quantum factor is ω/kT one bin away — 31 at kT = 0.005 with
Δω = 0.153 meV, which put **9.10 into a single bin of a spectrum whose entire band sum
is 0.5**. `subtract_elastic: true` removes the delta first and the problem with it.

Setting one without the other is warned about rather than left to this page: the engine
computes the amplification c2q(Δω) from your own time grid and kT and names it, on the
Python API and through the runner alike. Silence it, or promote it to a hard error,
with `on_elastic_leakage: off | warn | error` in either block.

## 4i. 1/S corrections

Zero-point energy and ordered-moment reduction, the next order of the 1/S expansion:

```yaml
tasks: {corrections: true}
corrections: {k_mesh: [16, 16, 16]}
```

Logs `dE` (add to the classical energy) and `dS_i` (⟨S^z⟩ = S − dS), and saves
`corrections.npz`. Validated against Sunny and the textbook square-lattice Heisenberg
antiferromagnet (`dE=−0.157947 J/site`, `dS=0.1966`).

---

## 5. Default Python Library Usage (Advanced)

For complex workflows (e.g., scanning over parameters), you can use `pyMagCalc` as a Python library.

> **Read this before using the direct API for physics.** Building `MagCalc`
> yourself is *not* equivalent to `magcalc run <config>`: it consumes the
> interactions and the lattice, and it **skips the whole runner pipeline** — the
> `magnetic_structure` block is not applied, in-memory edits to the config dict
> after construction are not seen, and, most importantly, **none of the
> ground-state guards run**. What you get back is LSWT about whatever state the
> model happens to hold, which for an antiferromagnet supplied with the default
> all-parallel state is a stationary *maximum* — and that returns a real,
> positive, entirely plausible spectrum. Use `magcalc run` (or `magcalc.runner`)
> for anything you intend to believe, and keep the direct API for what it is good
> at: repeated evaluations of an *already validated* model, e.g. the
> `DispersionEvaluator` loop in §4b.

```python
import magcalc as mc
from magcalc.generic_model import GenericSpinModel
import yaml

# Load Config
with open("examples/materials/KFe3J/config_kfe3j.yaml") as f:
    config = yaml.safe_load(f)

# Initialize Model
model = GenericSpinModel(config)

# Initialize Calculator
calc = mc.MagCalc(spin_model_module=model, spin_magnitude=2.5, hamiltonian_params=[3.23])

# Calculate Dispersion
q_path = [[0,0,0], [0.5,0,0], [0.33,0.33,0]]
energies = calc.calculate_dispersion(q_path)
print(energies)
```

---

## 6. SpinW and Sunny Tutorial Ports

pyMagCalc includes **30 ported SpinW tutorials** (SW01–SW38) under
`examples/spinw_tutorials/` and **9 ported Sunny.jl tutorials** (S01–S09) under
`examples/sunny_tutorials/`. Most are a self-contained, runnable `config.yaml`
reproducing the physics of the corresponding
[SpinW](https://spinw.org/tutorials/) or [Sunny.jl](https://github.com/SunnySuite/Sunny.jl)
tutorial. Three Sunny rows are shaped differently, deliberately: **S03** is a README
pointing at `examples/materials/FeI2/config_fei2_sun.yaml`, and **S06** (a
non-equilibrium CP² quench) and **S09** (disorder + KPM) ship a companion script —
`quench.py`, `disorder_kpm.py` — because neither is an equilibrium calculation and
substituting one would produce a folder that looks like a port and is not one.

### Running a tutorial

```bash
magcalc run examples/spinw_tutorials/SW01_FM_chain/config.yaml
magcalc run examples/materials/FeI2/config_fei2_sun.yaml   # the S03 SU(N) model
```

### What is covered

| Range | Topics |
|-------|--------|
| SW01–SW03 | FM/AFM chains, frustrated J1–J2 chain with exact spiral |
| SW04–SW09 | Frustrated square lattice, kagome (FM, AFM, √3×√3, DM) |
| SW10–SW11 | Constant-energy cuts, La₂CuO₄ |
| SW12–SW15 | Triangular easy-plane, LiNiPO₄, YVO₃, langasite spiral |
| SW16–SW19 | Na₂IrO₃ Kitaev, symbolic LSWT, distorted kagome spiral, mixed-spin |
| SW20–SW38 | Yb₂Ti₂O₇ g-tensor, YIG, biquadratic, Stevens operators, dipole-dipole, resolution |
| S01–S09 (Sunny) | CoRh₂O₄, FeI₂ SU(N), finite-T dynamics, Ising MC, CP² skyrmions, dipole-dipole, momentum conventions, disorder + KPM |

### Key conventions

pyMagCalc and SpinW/Sunny differ in bookkeeping:

1.  **Ordered-pair counting**: pyMagCalc uses `H = (1/2) Σ_ij`, so every
    bond must appear in both directions. Distance-based `symmetry_rules`
    handle this automatically.
2.  **Explicit magnetic cells**: Incommensurate orders use either the
    rotating-frame spiral (§4d) or an explicit supercell, not a propagation
    vector on the chemical cell.
3.  **q-paths stay in CHEMICAL RLU** when the cell is enlarged through
    `crystal_structure.magnetic_supercell` — the engine keeps the chemical
    reciprocal basis and the bands simply fold, which is the SpinW/Sunny
    convention and is also how S(Q,ω) is normalized (per chemical cell). You
    only have to rescale RLU by `n` yourself in the old style, where the
    replicated cell is hand-written as the `crystal_structure`; prefer
    `magnetic_supercell` and avoid the question.

See `examples/spinw_tutorials/README.md` for the full status table,
validation details, and the engine bug fixes made during porting.
