# pyMagCalc: Linear Spin-Wave Theory Calculator

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-development-orange.svg)]()

## Introduction

`pyMagCalc` is a Python package for performing Linear Spin-Wave Theory (LSWT) calculations. It allows users to define a spin model (Hamiltonian, magnetic structure, lattice) and compute spin-wave dispersion relations and dynamic structure factors S(Q,ω).

## Key Features

*   **MagCalc Pure Designer:** Interactive web GUI for designing models from CIF files and symmetry-based bonding rules.
*   **Diffraction Physics:** Calculates spin-wave dispersion and neutron scattering intensity (S(Q,ω)) with magnetic form factor and polarization factor corrections.
*   **Energy Minimization:** Numerically finds the classical magnetic ground state by minimizing the Hamiltonian energy, supporting both simple and complex (e.g., canted) structures.
*   **3D Visualization:** Visualizes the magnetic structure in 3D with scaled spins and correct aspect ratios.
*   **Modular Architecture:** Separation of concerns with `MagCalc` core logic, linear algebra utilities, and model definitions.
*   **Symbolic Hamiltonian:** Generates symbolic quadratic boson Hamiltonians using `SymPy` for arbitrary spin interactions.
*   **Numerical Engine:** Efficient numerical evaluation using `NumPy` and `multiprocessing` for parallel q-point calculations.
*   **Caching System:** Caches computationally expensive symbolic Hamiltonian diagonalization to disk for faster re-runs.
*   **Centralized Outputs:** Automatically saves all generated plots and data to structured directories (`examples/plots/` and `cache/data/`).
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
*   NumPy
*   SciPy
*   SymPy
*   Matplotlib
*   tqdm
*   PyYAML
*   ASE (Atomistic Simulation Environment, used for CIF file reading)
*   pytest (for testing)

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

## Graphical User Interface: MagCalc Pure Designer

The **MagCalc Pure Designer** is a modern web application designed to simplify the creation of complex spin models using a "pure" (symmetry-based) approach.

### Feature Highlights
- **CIF Integration**: Load crystal structures directly from CIF files.
- **Symmetry-Aware Bonding**: Define interaction rules (Heisenberg, DM, Anisotropic Exchange) that are automatically propagated by space group symmetry.
- **Real-time Feedback**: Export to YAML or save expanded configurations directly to your workspace.

### Starting the Designer

1.  **Start the Backend**:
    ```bash
    python3 gui/server.py
    ```
2.  **Start the Frontend**:
    ```bash
    cd gui
    npm install  # (First time only)
    npm run dev
    ```
3.  **Access**: Open [http://localhost:5173/](http://localhost:5173/) in your browser.

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
calc = mc.MagCalc(spin_model_module=model, spin_magnitude=2.5, cache_mode='auto')

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
*   **Interactions**: Heisenberg ($J$), DM ($D$), Single-Ion Anisotropy, etc.
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
