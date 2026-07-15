# pyMagCalc: Linear Spin-Wave Theory Calculator

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-development-orange.svg)]()

## Introduction

`pyMagCalc` is a Python package for performing Linear Spin-Wave Theory (LSWT) calculations. It allows users to define a spin model (Hamiltonian, magnetic structure, lattice) and compute spin-wave dispersion relations and dynamic structure factors S(Q,ω).

## Key Features

*   **pyMagCalc Studio:** Interactive modern web GUI for designing models from CIF files and symmetry-based bonding rules. Provides a seamless **Design -> Save -> Run** workflow with 3D visualization. Also available as a **native macOS & iOS app** (SwiftUI) with Metal-backed 3D rendering and embedded backend management.
*   **Symmetry-Aware Mechanics:** Automatically propagates Heisenberg ($J$), DM ($D$), Anisotropic Exchange ($T$), **Kitaev ($K$)**, and full 3×3 **Interaction Matrices** across the crystal using space-group symmetry operators (via `pymatgen` and `spglib`).
*   **Mixed-Spin Models:** Supports different spin magnitudes per site (e.g., Cu²⁺ S=½ + Fe²⁺ S=2). Each site's Holstein–Primakoff expansion is scaled by its own `spin_S`.
*   **Spiral (Rotating-Frame) Structures:** Incommensurate magnetic orders are handled via a rotating-frame formulation (`type: spiral`) with an exact local-frame construction, validated against analytic helix dispersions.
*   **SpinW Tutorial Ports:** 19 validated SpinW tutorials (SW01–SW19) ported as runnable `magcalc run` configurations, covering FM/AFM chains, kagome lattices, Kitaev honeycomb, spirals, mixed-spin models, and more.
*   **Robust CIF Import:** Imports crystal structures from CIF files, automatically detecting symmetry and reducing to unique Wyckoff positions.
*   **Security & Safety:** Replaced insecure `eval()` with a SymPy-based safe evaluator for mathematical expressions in Hamiltonian parameters.
*   **Stable Runner Engine:** Standardized task architecture with concise keys and improved error handling to prevent runtime crashes.
*   **3D Visualization:** Visualizes the magnetic structure in 3D with scaled spins, DM vectors (arrows), and orientation guides. Includes zero-vector guarding and memory optimizations.
*   **Flexible Caching:** Supports disk caching (`auto`, `r`, `w`) for expensive symbolic matrices, or `none` for purely in-memory execution.
*   **Ground-State Search:** Monte-Carlo simulated annealing (`method: anneal`, SpinW `anneal` / Sunny `LocalSampler`) and local-field steepest descent (`method: steep`, SpinW `optmagsteep`), plus automatic **ground-state guards** that fail the run when the magnetic structure is not a classical minimum (LSWT about a non-minimum silently yields a meaningless spectrum).
*   **Hamiltonian Terms:** Beyond bilinear exchange — anisotropic **per-site g-tensors** (incl. uniaxial about a *local* axis, for rare-earth pyrochlores), full 3×3 single-ion anisotropy, **Stevens operators** O_k^q (k=2/4/6, crystal fields), genuine **biquadratic** exchange (valid for non-collinear structures), long-range **dipole-dipole** coupling (validated against Sunny), and real-space **multi-k** structures.
*   **Measurement Modeling:** Finite-temperature (Bose) intensities, magnetic/structural domain (twin) averaging, cross-section selection (perp/trace/tensor components), instrument resolution (energy-dependent FWHM polynomial, |Q| smoothing, direct-geometry Ei/two-theta kinematic masking), and first-class 2-D constant-energy cuts (`tasks.energy_cut`) — all config-driven.
*   **Data Fitting:** Fits the spin Hamiltonian to inelastic-neutron-scattering data — magnon dispersion `E(Q)`, single-crystal `I(Q, ω)`, or powder `I(|Q|, ω)` — via [lmfit](https://lmfit.github.io/) (bounds, tied/fixed parameters, uncertainties, choice of optimizer). Dispersion fits use a **compile-once fast evaluator** (`DispersionEvaluator`) that skips all per-iteration symbolic work, making fits of large magnetic cells orders of magnitude faster.
*   **Data Export (CSV):** Export results to `.csv` or `.npz` files for external analysis.
*   **Validated Configurations:** Supports declarative YAML configurations validated against a robust **Pydantic schema** for immediate error feedback.

## Directory Structure

*   `magcalc/`: Core Python package.
    *   `core.py`: Main `MagCalc` class, calculation logic, and the `DispersionEvaluator` fast dispersion engine.
    *   `generic_model.py`: `GenericSpinModel` for YAML-based model loading (mixed-spin, spiral, interaction matrices).
    *   `symbolic.py`: Symbolic Hamiltonian construction and Fourier transforms.
    *   `fitting.py`: Data-fitting engine (`run_fit`, `FitProblem`) for dispersion / S(Q,ω) / powder data.
    *   `runner.py`: Declarative task runner (minimization → dispersion → S(Q,ω) → powder → fit → plot).
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
    *   `spinw_tutorials/`: 19 ported SpinW tutorials (SW01–SW19), each a runnable `config.yaml`.
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

> **Note**: Symmetry-generated interactions are currently in active development and may not fully propagate as expected. Please double-check your expanded configuration.

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

## Configuration

The core of `pyMagCalc` is the declarative YAML configuration (e.g., `config.yaml`). It defines:

*   **Structure**: Lattice vectors, atoms (with per-atom `spin_S` for mixed-spin models), and optional `ion` for form factors.
*   **Interactions**: Heisenberg ($J$), DM ($D$), Anisotropic Exchange ($K, \Gamma, \Gamma'$), Kitaev, full 3×3 Interaction Matrices, and Single-Ion Anisotropy (SIA) with arbitrary axes.
*   **Magnetic Structure**: Collinear patterns (`generic`, `afm`, `fm`), or incommensurate spirals (`type: spiral` with `k`, `axis`).
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

### SpinW Tutorial Ports (`examples/spinw_tutorials/`)
19 SpinW tutorials (SW01–SW19) ported to pyMagCalc, each as a runnable `config.yaml`. These cover:
*   FM/AFM chains (SW01–SW03), frustrated lattices (SW04), kagome models (SW05–SW09)
*   Constant-energy cuts (SW10), real materials — La₂CuO₄, LiNiPO₄, YVO₃ (SW11, SW13, SW14)
*   Spiral/incommensurate structures (SW03, SW08, SW15, SW18)
*   Kitaev honeycomb with full 3×3 interaction matrices (SW16)
*   Symbolic LSWT verification (SW17), mixed-spin models (SW19)

See `examples/spinw_tutorials/README.md` for the full status table and physics conventions.

### Fitting (`examples/fitting/`)
Example fitting configurations and synthetic data for dispersion / S(Q,ω) / powder fits.

## Testing

Run the test suite using `pytest` from the project root:

```bash
pytest
```

## Acknowledgments

This codebase was developed with the assistance of **Google Gemini** and **Anthropic Claude**.

