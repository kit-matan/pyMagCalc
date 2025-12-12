# pyMagCalc: Linear Spin-Wave Theory Calculator

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-development-orange.svg)]()

## Introduction

`pyMagCalc` is a Python package for performing Linear Spin-Wave Theory (LSWT) calculations. It allows users to define a spin model (Hamiltonian, magnetic structure, lattice) and compute spin-wave dispersion relations and dynamic structure factors S(Q,ω).

## Key Features

*   **Diffraction Physics:** Calculates spin-wave dispersion and neutron scattering intensity (S(Q,ω)) with magnetic form factor and polarization factor corrections.
*   **Modular Architecture:** Separation of concerns with `MagCalc` core logic, linear algebra utilities, and model definitions.
*   **Symbolic Hamiltonian:** Generates symbolic quadratic boson Hamiltonians using `SymPy` for arbitrary spin interactions.
*   **Numerical Engine:** Efficient numerical evaluation using `NumPy` and `multiprocessing` for parallel q-point calculations.
*   **Caching System:** Caches computationally expensive symbolic Hamiltonian diagonalization to disk for faster re-runs.
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
    *   `ZnCVO/`: Zn-doped CVO examples.
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
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Add the project root to your `PYTHONPATH` or run scripts from the root directory to ensure `import magcalc` works.

## Basic Usage

### Running from Command Line

Use the `run_magcalc.py` script with a configuration file:

```bash
python scripts/run_magcalc.py examples/KFe3J/KFe3J_declarative.yaml
```

### Scripting with Python

```python
import numpy as np
import magcalc as mc
from magcalc.generic_model import GenericSpinModel

# Example: Using the Calculator programmatically
# calculator = mc.MagCalc(config_filepath="examples/KFe3J/KFe3J_declarative.yaml")
# dispersion = calculator.calculate_dispersion(q_points)
```

## Spin Model Definition

Define your physics in a `material_config.yaml` file (recommended) or a Python module. The YAML format allows specifying:
*   **Crystal Structure**: Lattice parameters and magnetic atoms.
*   **Interactions**: Heisenberg exchange ($J$), Dzyaloshinskii-Moriya ($D$), and Single-Ion Anisotropy.
*   **Parameters**: Numerical values for symbolic constants.

See `magcalc/material_config_schema.yaml` for layout details.

## Examples

The `examples/` directory contains fully functional examples:
*   **`examples/KFe3J/`**: Extensive example for Jarosite, showcasing:
    *   Declarative YAML configuration.
    *   Comparison with legacy Python-defined models.
    *   Scripts for plotting dispersion and S(Q,ω) cuts.
*   **`examples/aCVO/`**: Alpha-Cu2V2O7 example.
    *   Includes `sw_CVO.py` for comprehensive spin-wave calculations.
    *   demonstrates handling of complex magnetic structures and magnetic fields.
    *   **Note**: Scripts in this directory may require setting `sys.path` if run directly.t

## Testing

Run the test suite using `pytest` from the project root:

```bash
pytest
```
