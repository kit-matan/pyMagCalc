# pyMagCalc: Linear Spin-Wave Theory Calculator

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-development-orange.svg)]()

## Introduction

`pyMagCalc` is a Python package for modeling magnetic excitations. Its core is Linear Spin-Wave Theory (LSWT) — define a spin model (Hamiltonian, magnetic structure, lattice) and compute spin-wave dispersions and dynamic structure factors S(Q,ω) — with a second **SU(N)** engine for single-ion/multipolar excitations and **entangled-unit** (dimer/trimer) modes for valence-bond solids. Beyond LSWT it also provides paramagnetic diffuse scattering (**SCGA**), finite-temperature **classical Monte-Carlo** and **spin dynamics**, a **KPM** spectral solver for large cells, and a high-order **dimer series expansion** for strongly-coupled dimer magnets. Every feature is validated against an independent reference — Sunny.jl, SpinW, or an exact analytic result.

## Key Features

*   **pyMagCalc Studio:** Interactive modern web GUI for designing models from CIF files and symmetry-based bonding rules. Provides a seamless **Design -> Save -> Run** workflow with 3D visualization. Also available as a **native macOS & iOS app** (SwiftUI) with Metal-backed 3D rendering and embedded backend management.
*   **Symmetry-Aware Mechanics:** Automatically propagates Heisenberg ($J$), DM ($D$), Anisotropic Exchange ($T$), **Kitaev ($K$)**, and full 3×3 **Interaction Matrices** across the crystal using space-group symmetry operators (via `pymatgen` and `spglib`).
*   **Mixed-Spin Models:** Supports different spin magnitudes per site (e.g., Cu²⁺ S=½ + Fe²⁺ S=2), in **both** the dipole and SU(N) engines. Each site's Holstein–Primakoff expansion is scaled by its own `spin_S`, and the S(Q,ω) prefactor is √(S_i/2) **per site**, so mixed-spin intensities are correct as well as the energies.
*   **Spiral (Rotating-Frame) Structures:** Incommensurate magnetic orders are handled via a rotating-frame formulation (`type: single_k`) with an exact local-frame construction, validated against analytic helix dispersions and Sunny `SpinWaveTheorySpiral`.
*   **SU(N) Mode:** A second LSWT engine (`calculation.mode: SUN`, as in Sunny) where each site carries an N=2S+1 Hilbert space — capturing single-ion (multipolar) excitations that dipole LSWT structurally cannot. Validated against Sunny `:SUN` on FeI₂ (energy, all 8 bands, and intensities).
*   **Entangled Units & Dimer Series:** Treat a cluster (dimer/trimer) as ONE effective SU(N) site (`calculation.mode: entangled`) — the reference is the unit ground state (e.g. a singlet dimer, invisible to dipole/single-site LSWT) and the excitations are its triplons. For strongly-coupled dimer magnets (J′≈J), a high-order **linked-cluster dimer series expansion + Dlog-Padé** resummation gives the quantitative dispersion. Reproduces published results on Cu₅SbO₆ and the Rb₂Cu₃SnF₁₂ pinwheel VBS.
*   **Beyond LSWT — diffuse, thermal, dynamical:** **SCGA** paramagnetic diffuse S(q) above T_N (vs Sunny); **thermal Monte-Carlo** with parallel tempering for C(T)/M(T)/χ(T); **Wang–Landau** flat-histogram g(E), so one run covers every temperature; **SampledCorrelations** finite-T classical spin dynamics S(q,ω) (Langevin or Metropolis thermalization, RK4 or symplectic implicit-midpoint integration) whose intensities are on the **same absolute scale as LSWT and Sunny**; **site-level disorder** (vacancies, open boundaries, bond disorder) across all the real-space classical samplers; and a **KPM** Chebyshev spectral solver that computes S(q,ω) without diagonalization for large/disordered cells — with a per-q positive-definiteness guard, since KPM never diagonalizes and so cannot notice by itself that it is expanding about a non-minimum.
*   **1/S Corrections:** Zero-point energy and ordered-moment reduction (`tasks: {corrections: true}`), validated against Sunny and the textbook square-lattice Heisenberg antiferromagnet.
*   **Magnetic CIF (mCIF):** Import magnetic structures from mCIF / magnetic space groups (`from_mcif:`, `magcalc mcif`), validated against Sunny.
*   **Tutorial Ports:** 30 validated SpinW tutorials (SW01–SW38) and **all 9** Sunny.jl tutorials (S01–S09), nearly all of them a runnable `magcalc run` configuration — FM/AFM chains, kagome lattices, Kitaev honeycomb, spirals, mixed-spin, SU(N), finite-T Monte-Carlo, CP² skyrmions, disorder + KPM, and dipole-dipole models.
*   **Robust CIF Import:** Imports crystal structures from CIF files, automatically detecting symmetry and reducing to unique Wyckoff positions.
*   **Security & Safety:** Replaced insecure `eval()` with a SymPy-based safe evaluator for mathematical expressions in Hamiltonian parameters.
*   **Stable Runner Engine:** Standardized task architecture with concise keys and improved error handling to prevent runtime crashes.
*   **3D Visualization:** Visualizes the magnetic structure in 3D with scaled spins, DM vectors (arrows), and orientation guides. Includes zero-vector guarding and memory optimizations.
*   **Flexible Caching:** Supports disk caching (`auto`, `r`, `w`) for expensive symbolic matrices, or `none` for purely in-memory execution.
*   **Ground-State Search:** Monte-Carlo simulated annealing (`method: anneal`, SpinW `anneal` / Sunny `LocalSampler`) and local-field steepest descent (`method: steep`, SpinW `optmagsteep`), plus automatic **ground-state guards** that fail the run when the magnetic structure is not a classical minimum (LSWT about a non-minimum silently yields a meaningless spectrum). Three of them: an imaginary-mode check, an energy audit that catches a stationary *maximum* the first cannot see, and — for KPM, which never diagonalizes — a per-q `H₂(q) ⪰ 0` test.
*   **Hamiltonian Terms:** Beyond bilinear exchange — anisotropic **per-site g-tensors** (incl. uniaxial about a *local* axis, for rare-earth pyrochlores), full 3×3 single-ion anisotropy, **Stevens operators** O_k^q (k=2/4/6, crystal fields), genuine **biquadratic** exchange (valid for non-collinear structures), arbitrary two-site **pair operators** (Sunny `set_pair_coupling!`), long-range **dipole-dipole** coupling by exact Ewald summation — including with a rotating-frame `single_k` structure — and real-space **multi-k** structures. Optional **RCS anisotropy renormalization** (`anisotropy_renormalization: rcs`) switches the dipole engine from SpinW's convention (Sunny's `:dipole_uncorrected`, the default here) to Sunny's corrected `:dipole`.
*   **Measurement Modeling:** Finite-temperature (Bose) intensities, magnetic/structural domain (twin) averaging, cross-section selection (perp/trace/tensor components), **polarized and Blume–Maleev frames** (arbitrary polarization axis, SF/NSF/chiral channels, BM components — the Sunny `ssf_custom_bm` analogue), instrument resolution (energy-dependent FWHM polynomial, |Q| smoothing, direct-geometry Ei/two-theta kinematic masking), first-class 2-D constant-energy cuts (`tasks.energy_cut`), and **NeXus/histogram binning** (`magcalc/binning.py`) — all config-driven. The S(Q,ω) absolute scale **equals Sunny's**, pinned band by band.
*   **Symmetry Analyzer CLI:** `magcalc symmetry <config>` prints the space group, the symmetry-inequivalent bond orbits, and the symmetry-**allowed** exchange matrix per bond (the Sunny `print_symmetry_table` analogue) — the fast way to pick `ref_pair` bonds and see which matrix entries symmetry zeros or ties. `--cells` / `--species` / `--bz-path` add the crystal utilities (primitive cell, standardization, subcrystal, irreducible BZ paths).
*   **Knowing Which Engine Ran:** `magcalc where` reports the `magcalc` package actually imported, its git HEAD, and any *other* importable copy — because `pip install -e .` does **not** win by default (its finder is appended to `sys.meta_path`, so anything on `sys.path` shadows it). Every `magcalc run` logs the same line first. An interpreter-startup check that fires even when a stale copy wins outright — which the in-package detector structurally cannot do — is **installed automatically on your first `magcalc` command** (`magcalc guard --install` / `--uninstall` to manage it by hand, `MAGCALC_SHADOW_GUARD=off` to opt out of both).
*   **Data Fitting:** Fits the spin Hamiltonian to inelastic-neutron-scattering data — magnon dispersion `E(Q)`, single-crystal `I(Q, ω)`, or powder `I(|Q|, ω)` — via [lmfit](https://lmfit.github.io/) (bounds, tied/fixed parameters, uncertainties, choice of optimizer). Dispersion fits use a **compile-once fast evaluator** (`DispersionEvaluator`) that skips all per-iteration symbolic work, making fits of large magnetic cells orders of magnitude faster.
*   **Data Export (CSV):** Export results to `.csv` or `.npz` files for external analysis.
*   **Validated Configurations:** Supports declarative YAML configurations validated against a robust **Pydantic schema** for immediate error feedback.

## Directory Structure

*   `magcalc/`: Core Python package.
    *   `core.py`: Main `MagCalc` class, calculation logic, and the `DispersionEvaluator` fast dispersion engine.
    *   `generic_model.py`: `GenericSpinModel` for YAML-based model loading (mixed-spin, spiral, interaction matrices).
    *   `symbolic.py`: Symbolic Hamiltonian construction and Fourier transforms.
    *   `sun/`: SU(N) and entangled-unit engines — `lswt.py` (SU(N) LSWT), `entangled.py` (dimer/trimer units), `dimer_series.py` (linked-cluster series + Dlog-Padé), `kpm.py` (Chebyshev spectral S(q,ω)).
    *   `scga.py`: Self-consistent Gaussian approximation (paramagnetic diffuse S(q)).
    *   `annealing.py` / `thermal_mc.py` / `classical_dynamics.py`: ground-state annealing, finite-T Monte-Carlo (parallel tempering + Wang–Landau), and classical spin dynamics (SampledCorrelations); `sun/dynamics.py` is the CP^(N−1) counterpart, including the dissipative quench.
    *   `corrections.py`: 1/S zero-point energy and ordered-moment reduction.
    *   `spiral_opt.py`: Luttinger–Tisza ordering vector and spiral-energy optimization.
    *   `ewald.py`: Exact Ewald summation of the long-range dipolar interaction.
    *   `binning.py`: Histogram binning and NeXus (`.nxs`) reading for experiment data.
    *   `constants.py`: The shared physical constants (μ_B, the electron g, the dipolar prefactors) — deliberately one definition each, since six modules used to carry their own copy.
    *   `provenance.py` / `_shadow_guard.py`: which engine copy is running (`magcalc where`, `magcalc guard`).
    *   `mcif.py`: Magnetic CIF / magnetic space-group import.
    *   `fitting.py`: Data-fitting engine (`run_fit`, `FitProblem`) for dispersion / S(Q,ω) / powder data.
    *   `runner.py`: Declarative task runner (minimization → dispersion → S(Q,ω) → SCGA / thermal-MC / dynamics / corrections → powder → fit → plot).
    *   `linalg.py`: Matrix operations and Bogoliubov transformation utilities.
    *   `config_loader.py`: Utilities for loading and validating configurations.
    *   `schema.py`: Pydantic V2 models for robust configuration validation.
*   `gui/`: Web-based pyMagCalc Studio (FastAPI backend + React frontend).
*   `native/`: Native macOS & iOS SwiftUI app (`MagCalcStudio`).
*   `scripts/`: Executable scripts for running calculations and inspecting models (e.g., `run_magcalc.py`).
*   `examples/`: Sample data and scripts organized by category.
    *   `materials/`: Real material studies.
        *   `KFe3J/`: KFe₃(OH)₆(SO₄)₂ (Jarosite) — Kagome antiferromagnet.
        *   `aCVO/`: α-Cu₂V₂O₇ — Honeycomb-like antiferromagnet with DM interactions.
        *   `CCSF/`: Cs₂Cu₂SnF₁₂ — frustrated antiferromagnet.
        *   `ZnCVO/`: ZnCu₂V₂O₇.
        *   `FeI2/`: FeI₂ — triangular-lattice Ising-type antiferromagnet.
        *   `mcif/`: Magnetic CIF import examples.
    *   `entangled/`: Entangled-unit (dimer) models — `dimer_chain/`, `Cu5SbO6/` (dimer expansion), `Rb2Cu3SnF12/` (pinwheel VBS + dimer series).
    *   `spinw_tutorials/`: 30 ported SpinW tutorials (SW01–SW38), each a runnable `config.yaml`.
    *   `sunny_tutorials/`: all 9 Sunny.jl tutorials (S01–S09) — CoRh₂O₄, FeI₂ SU(N), finite-T, Ising MC, CP² skyrmions, dipole-dipole, momentum conventions, disorder + KPM. S06 and S09 ship a companion script (`quench.py`, `disorder_kpm.py`) because neither is an equilibrium calculation; S03 is a README pointing at `materials/FeI2/config_fei2_sun.yaml`.
    *   `fitting/`: Example fitting configuration and synthetic data.
    *   `plots/`: Centralized directory where all example scripts save their output plots.
*   `tests/`: Unit and integration tests ensuring package reliability.

## Dependencies

*   Python (3.8+)
*   NumPy (>=1.20)
*   SciPy (>=1.7.0)
*   SymPy (>=1.9)
*   Matplotlib (>=3.4.0)
*   tqdm (>=4.60.0)
*   PyYAML (>=5.4)
*   ASE (>=3.22.0, for CIF file reading)
*   Typer (>=0.9, CLI framework)
*   Pydantic (>=2.0, configuration validation)
*   lmfit (>=1.2, for data fitting)
*   pytest (>=7.0.0, for testing)
*   pymatgen (>=2023.0) + spglib (>=2.0) — optional, for the GUI's symmetry analysis
*   fMagCalc (optional compiled Fortran backend — see Installation below)

## Installation

1.  Clone the repository.
2.  Install the package in editable mode (recommended for development):
    ```bash
    pip install -e .
    ```
    This installs the `magcalc` command-line tool and all dependencies.

### Optional: compiled Fortran backend (fMagCalc)

[fMagCalc](https://github.com/kit-matan/fMagCalc) is a compiled OpenMP/LAPACK
engine for the numerical hot path (per-q diagonalization, S(Q,ω) intensities,
powder averaging). It is optional — pyMagCalc is fully functional without it —
but much faster at large q-counts.

**Prerequisites** (compiles on install): a Fortran compiler, CMake ≥ 3.20, and
LAPACK.

```bash
# macOS
brew install gcc cmake          # gfortran + CMake; LAPACK comes from Accelerate

# Debian/Ubuntu Linux
sudo apt install gfortran cmake libopenblas-dev
```

**Install** straight from GitHub (or from a local clone):

```bash
pip install git+https://github.com/kit-matan/fMagCalc
# or: pip install /path/to/fMagCalc
```

**Verify** — should print `ctypes`:

```bash
python -c "import fmagcalc; print(fmagcalc.backend)"
```

**Use it** by selecting the backend in any of the three interfaces (all fall
back to NumPy with a warning if fMagCalc is unavailable):

```yaml
# config.yaml — applies to dispersion, S(Q,w), and powder tasks
calculation:
  backend: fortran        # default: numpy
```

```python
# Python API
calc.calculate_dispersion(q_list, backend="fortran")
calc.calculate_sqw(q_list, backend="fortran")
```

In **pyMagCalc Studio**, set **Tasks & Plotting → Calculation Settings →
Compute Backend** to *Fortran (fMagCalc)*.

See **TUTORIAL.md §4c** for details and troubleshooting.

## CLI Usage

The command-line interface makes it easy to manage calculations.

### 1. Initialize a Project
Create a template configuration file:
```bash
magcalc init my_config.yaml
```

### 2. Validate Configuration
Check if your configuration file is valid:
```bash
magcalc validate my_config.yaml
```

### 3. Run Calculations
Run dispersion, S(Q,w), and plotting as defined in the config:
```bash
magcalc run my_config.yaml
```

### 4. Fit to Neutron Data
Fit the spin Hamiltonian to experimental data defined in the config's
`fitting:` block (or add `tasks: { fit: true }` and use `magcalc run`):
```bash
magcalc fit my_config.yaml
```
This writes an lmfit report (`fit_report.txt`), the best-fit parameters
(`fit_params.yaml`), and a data-vs-model comparison plot (`fit_comparison.png`).
See **TUTORIAL.md** (§4b) for the full `fitting:` block reference, the
compile-once fast dispersion path, and the `DispersionEvaluator` Python API.

### 5. Inspect Symmetry and Import Magnetic CIFs
Print the space group, symmetry-inequivalent bond orbits, and allowed exchange
matrices (handy for choosing `ref_pair` bonds):
```bash
magcalc symmetry my_config.yaml --max-distance 5.0
```
Import a magnetic structure from an mCIF (magnetic space group):
```bash
magcalc mcif structure.mcif -o config_from_mcif.yaml
```

### 6. Check Which Engine Is Running
If behaviour is inexplicable — a fix "not applied", a documented key
"unsupported", a pinned number that moved — read this before anything else:
```bash
magcalc where                  # the imported package, its git HEAD, and any other copy
magcalc guard                  # interpreter-startup check: status (armed automatically
                               #   on first use; --install / --uninstall to manage)
```
An editable install does not automatically win: `pip install -e .` appends its
finder to `sys.meta_path`, i.e. *after* the normal path search, so any `magcalc/`
directory on `sys.path` shadows it — and `sys.path[0]` is the working directory
for `python -c`, pytest's rootdir under `pytest`, and a script's own directory
for anything in `scripts/`.

## Graphical User Interface: pyMagCalc Studio

The **pyMagCalc Studio** is a modern web application designed to simplify the creation of complex spin models using a "pure" (symmetry-based) approach.

### Feature Highlights
- **CIF Integration**: Load crystal structures directly from CIF files.
- **Symmetry-Aware Bonding**: Define interaction rules (Heisenberg, DM, Anisotropic Exchange) that are automatically propagated by space group symmetry.
- **Real-time Feedback**: Export to YAML or save expanded configurations directly to your workspace.

### Starting the Designer
The easiest way to start the application is using the **One-Click Launcher**:

```bash
./start_magcalc.sh
```

This script will:
1.  Kill any existing processes on ports 8000/5173.
2.  Start the Python backend and React frontend.
3.  Automatically open your browser to [http://localhost:5173/](http://localhost:5173/).
4.  Stop all services cleanly when you press `Ctrl+C`.

> **Note**: Always check what a symmetry rule actually expanded to — `magcalc symmetry <config>` prints the bond orbits and the symmetry-**allowed** matrix per bond. The engine now raises rather than expanding silently to nothing (a rule matching no bond, a non-scalar rule without a `ref_pair`, or an ambiguous reference bond are all hard errors), but a rule that expands to the wrong *orbit* is still yours to catch.

The designer facilitates a seamless **Design -> Save -> Run** workflow, where the generated `config_designer.yaml` can be executed immediately using `magcalc run`.

### Native macOS & iOS Application (SwiftUI)

MagCalc Studio is also available as a **native SwiftUI application** for macOS and iOS/iPadOS, located in `native/MagCalcStudio/`. It provides full feature parity with the web app, plus native advantages:

*   **Embedded backend management** (macOS) — start/stop the Python server from Settings; no terminal needed.
*   **Metal-backed 3D rendering** — SceneKit crystal and spin-structure visualization.
*   **Project files** — save/open models as JSON via native file panels (iCloud Drive compatible).
*   **Keyboard shortcuts** — ⌘R runs, ⌘. stops a calculation.
*   **iOS/iPadOS** — connects to a Mac running the backend on your network.

**Build & Run** (requires Xcode 16+ and [XcodeGen](https://github.com/yonaskolb/XcodeGen)):
```bash
cd native/MagCalcStudio
xcodegen generate
open MagCalcStudio.xcodeproj
```

See `native/MagCalcStudio/README.md` for full build instructions and feature details.

#### Legacy Python-wrapped App

A lightweight Python wrapper (using `pywebview`) is also available:
```bash
./run_native.sh
# or: python gui/native_app.py
```
This requires building the frontend first (`cd gui && npm install && npm run build`).

## Basic Usage

### Running Example Scripts

The recommended way to run examples is via the CLI using the modern configuration files:

```bash
# Run the Jarosite (KFe3J) example
magcalc run examples/materials/KFe3J/config_kfe3j.yaml

# Run the CVO example
magcalc run examples/materials/aCVO/config_acvo.yaml

# Run a SpinW tutorial port
magcalc run examples/spinw_tutorials/SW01_FM_chain/config.yaml
```

Plots are automatically saved to `examples/plots/` (or next to the config for SpinW tutorials). You can toggle on-screen display using the `show_plot` option in the config.

### Scripting with Python (Advanced)

For custom workflows or parameter scans, you can use the library directly:

```python
import magcalc as mc
from magcalc.generic_model import GenericSpinModel
import yaml

# 1. Load Model from YAML
with open("examples/materials/KFe3J/config_kfe3j.yaml") as f:
    config = yaml.safe_load(f)
model = GenericSpinModel(config)

# 2. Initialize Calculator
calc = mc.MagCalc(spin_model_module=model, spin_magnitude=2.5, cache_mode='none')

# 3. Minimize Energy
# Use a smart initial guess to avoid local minima
x0 = ... 
min_res = calc.minimize_energy(x0=x0)
calc.sm.set_magnetic_structure(min_res.x)

# 4. Visualize
mc.plot_magnetic_structure(calc.sm.atom_pos(), min_res.x, show_plot=True)

# 5. Calculate Dispersion
# ...
```

> **This is not equivalent to `magcalc run <config>`.** A hand-built `MagCalc`
> takes the lattice and the interactions and **skips the runner pipeline**: the
> config's `magnetic_structure` block is not applied, edits made to the config
> dict after construction are not seen, and **none of the ground-state guards
> run**. You then get LSWT about whatever state the model happens to hold — and
> for an antiferromagnet left in the default all-parallel state that is a
> stationary *maximum*, which returns a real, positive, entirely plausible
> spectrum. Prefer `magcalc run` for anything you intend to believe; reserve the
> direct API for repeated evaluation of an already-validated model (see the
> `DispersionEvaluator` pattern in **TUTORIAL.md §4b**).

## Configuration

The core of `pyMagCalc` is the declarative YAML configuration (e.g., `config.yaml`). It defines:

*   **Structure**: Lattice vectors, atoms (with per-atom `spin_S` for mixed-spin models), and optional `ion` for form factors.
*   **Interactions**: Heisenberg ($J$), DM ($D$), Anisotropic Exchange ($K, \Gamma, \Gamma'$), Kitaev, full 3×3 Interaction Matrices, and Single-Ion Anisotropy (SIA) with arbitrary axes.
*   **Magnetic Structure**: Collinear patterns (`generic`, `afm`, `fm`), incommensurate spirals (`type: single_k` with `k`, `axis`), or `type: multi_k` with a magnetic supercell.
*   **Minimization**: Initial guess (`initial_configuration`) and method.
*   **Plotting**: Options like `show_plot`, `plot_structure`, and axis limits.

## Symbolic Parameter Evaluation

Hamiltonian parameters in `pyMagCalc` support safe mathematical expressions via `SymPy`. This allows for physically intuitive definitions directly in the YAML config:

```yaml
parameters:
  J_ex: "2.5 * exp(-1.2)"
  D_vec: "[0, 0, 1.25 * sqrt(2)]"
  K_kitaev: "12.5 / 3.0"
```

The server uses a dedicated `_safe_eval` helper ensuring calculation robustness while preventing arbitrary code execution.

## Examples

Examples are organized under `examples/` in three categories:

### Materials (`examples/materials/`)
*   **`KFe3J/`**: Kagome Antiferromagnet (Jarosite).
    *   Uses `config.yaml` for a fully declarative workflow.
    *   Demonstrates `initial_configuration` for handling complex ground states (120-degree structure).
*   **`aCVO/`**: 1D Chain / Honeycomb (α-Cu₂V₂O₇).
    *   Demonstrates handling of imaginary eigenvalues via correct ground state finding.
    *   Features complex Dzyaloshinskii-Moriya interactions.
*   **`CCSF/`**: Cs₂Cu₂SnF₁₂ — frustrated antiferromagnet with symmetry-based configuration.
*   **`ZnCVO/`**: ZnCu₂V₂O₇.
*   **`FeI2/`**: FeI₂ — triangular-lattice antiferromagnet with long-range interactions (J3 at `rij_offset: [2,0,0]`).

### Entangled Units (`examples/entangled/`)
Dimer/trimer valence-bond-solid models (`calculation.mode: entangled`):
*   **`dimer_chain/`**: a chain of S=½ dimers whose triplon `ω(q)=√(J²−JJ′cos2πq)` matches the exact bond-operator result.
*   **`Cu5SbO6/`**: reproduces the J1–J2–J4 dimer expansion of Piyakulworawat *et al.*, PRR **8**, 013247 (2026).
*   **`Rb2Cu3SnF12/`**: the pinwheel VBS (Matan *et al.*, Nat. Phys. **6**, 865 (2010)) — single-dimer building block plus the full 6-dimer pinwheel via `series_dispersion.py` (linked-cluster + Dlog-Padé).

### SpinW & Sunny Tutorial Ports
30 SpinW tutorials (`examples/spinw_tutorials/`, SW01–SW38) and all 9 Sunny.jl tutorials (`examples/sunny_tutorials/`, S01–S09), nearly all of them a runnable `config.yaml`. These cover FM/AFM chains, frustrated and kagome lattices, constant-energy cuts, real materials (La₂CuO₄, LiNiPO₄, YVO₃, CoRh₂O₄, FeI₂), spiral/incommensurate structures, Kitaev honeycomb, SU(N), finite-T Monte-Carlo, and dipole-dipole models. See each directory's `README.md` for the full status table and physics conventions.

### Fitting (`examples/fitting/`)
A worked **dispersion** fit: `fit_dispersion.yaml` (a rectangular-lattice Néel antiferromagnet) against synthetic data generated from the analytic dispersion by `make_synthetic_data.py`, so the fit is checked against something the engine did not produce. The `sqw` and `powder` fit types are supported by `magcalc fit` and covered by `tests/test_fitting.py`, but have no worked example here yet.

## Testing

```bash
pytest                      # the FAST suite (the default is `-m "not slow"`)
pytest -m slow -k <name>    # one feature's deep validations
pytest -m ""                # EVERYTHING -- the merge gate
```

`pytest` alone runs the fast suite (647 of 825 tests), which is the iteration
loop, **not** a merge gate: the `slow` marker holds the exact-diagonalization
oracles, convergence sweeps and integration runs. Before merging to `master`, run
`pytest -m ""` — from the workspace root that is plain `pytest`, since the root
`pytest.ini` deliberately does not inherit `-m "not slow"`, and it also picks up
the `fMagCalc/` tests. The last full gate was **830 passed, 3 skipped**
(2026-08-15, 30 min).

Every feature is pinned to an **independent** reference — Sunny.jl (checked out
in-repo), SpinW, or an exact analytic identity — never to a self-generated golden
number, and every feature keeps at least one fast test so the quick suite still
touches its code path. `GAP_STATUS.md` records what each was validated against,
and closes with the recurring trap this rule exists for: the engine has
repeatedly produced plausible-but-wrong spectra, and every one was caught by an
oracle or an exact identity, never by inspection.

## Acknowledgments

This codebase was developed with the assistance of **Google Gemini** and **Anthropic Claude**.

