# pyMagCalc: Linear Spin-Wave Theory Calculator

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-development-orange.svg)]()

## Introduction

`pyMagCalc` is a Python package for performing Linear Spin-Wave Theory (LSWT) calculations. It allows users to define a spin model (Hamiltonian, magnetic structure, lattice) and compute spin-wave dispersion relations and dynamic structure factors S(Q,ω).

## Key Features

*   **pyMagCalc Studio:** Interactive web GUI for designing models from CIF files and symmetry-based bonding rules.
*   **Symmetry-Aware Mechanics:** Automatically propagates Heisenberg ($J$), DM ($D$), Anisotropic Exchange ($T$), and **Kitaev ($K$)** rules across the crystal using space-group symmetry operators (via `pymatgen` and `spglib`).
*   **Robust CIF Import:** Imports crystal structures from CIF files, automatically detecting symmetry and reducing to unique Wyckoff positions for a clean workflow.
*   **Diffraction Physics:** Calculates spin-wave dispersion and dynamic structure factors $S(Q,\omega)$ with magnetic form factor and polarization factor corrections.
*   **Energy Minimization:** Numerically finds the classical magnetic ground state by minimizing the Hamiltonian energy, supporting **parallel multistart** and **early stopping** for robust convergence.
*   **3D Visualization:** Visualizes the magnetic structure in 3D with scaled spins, DM vectors (arrows), and orientation guides.
*   **Symbolic Engine:** Generates symbolic quadratic boson Hamiltonians using `SymPy` for arbitrary spin interactions.
*   **Numerical Engine:** Efficient numerical evaluation using `NumPy` and `multiprocessing` for parallel q-point calculations.
*   **Flexible Caching:** Supports disk caching (`auto`, `r`, `w`) for expensive symbolic matrices, or `none` for purely in-memory execution.
*   **Data Export (CSV):** Optionally export dispersion and $S(Q,\omega)$ results to `.csv` files for external analysis.
*   **Flexible Inputs:** Supports declarative YAML configurations (validated against schema) or Python-based model definitions.

## Directory Structure

*   `magcalc/`: Core Python package.
    *   `core.py`: Main `MagCalc` class and calculation logic.
    *   `generic_model.py`: `GenericSpinModel` for YAML-based model loading.
    *   `linalg.py`: Matrix operations and Bogoliubov transformation utilities.
    *   `config_loader.py`: Utilities for loading and validating configurations.
*   `scripts/`: Executable scripts for running calculations and inspecting models (e.g., `run_magcalc.py`).
*   `examples/`: Sample data and scripts for various materials.
    *   `KFe3J/`: KFe3(OH)6(SO4)2 (Jarosite) - Kagome antiferromagnet.
    *   `aCVO/`: alpha-Cu2V2O7 - Honeycomb-like antiferromagnet with Dzyaloshinskii-Moriya interactions.
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
*   pytest (>=7.0.0, for testing)

## Installation

1.  Clone the repository.
2.  Install the package in editable mode (recommended for development):
    ```bash
    pip install -e .
    ```
    This installs the `magcalc` command-line tool and all dependencies.

## CLI Usage (New)

The new command-line interface makes it easy to manage calculations.

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

## Basic Usage

### Running Example Scripts

The recommended way to run examples is via the CLI using the modern configuration files:

```bash
# Run the Jarosite (KFe3J) example
magcalc run examples/KFe3J/config_modern.yaml

# Run the CVO example
magcalc run examples/aCVO/config_modern.yaml
```

Plots are automatically saved to `examples/plots/`. You can toggle on-screen display using the `show_plot` option in the config.

### Scripting with Python (Advanced)

For custom workflows or parameter scans, you can use the library directly:

```python
import magcalc as mc
from magcalc.generic_model import GenericSpinModel
import yaml

# 1. Load Model from YAML
with open("examples/KFe3J/config_modern.yaml") as f:
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

The core of `pyMagCalc` is the declarative YAML configuration (e.g., `config_modern.yaml`). It defines:

*   **Structure**: Lattice vectors and atoms.
*   **Interactions**: Heisenberg ($J$), DM ($D$), Anisotropic Exchange ($K, \Gamma, \Gamma'$), Kitaev, and Single-Ion Anisotropy (SIA) with arbitrary axes.
*   **Minimization**: Initial guess (`initial_configuration`) and method.
*   **Plotting**: Options like `show_plot`, `plot_structure`, and axis limits.

## Examples

*   **`examples/KFe3J/`**: Kagome Antiferromagnet (Jarosite).
    *   Uses `config_modern.yaml` for a fully declarative workflow.
    *   Demonstrates `initial_configuration` for handling complex ground states (120-degree structure).
*   **`examples/aCVO/`**: 1D Chain / Honeycomb (Cu2V2O7).
    *   Demonstrates handling of imaginary eigenvalues via correct ground state finding.
    *   Features complex Dzyaloshinskii-Moriya interactions.

## Testing

Run the test suite using `pytest` from the project root:

```bash
pytest
```

## Acknowledgments

This codebase was developed with the assistance of **Google Gemini**, an advanced agentic AI coding assistant.

