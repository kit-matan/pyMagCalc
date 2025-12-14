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

---

## 2. Quick Start (CLI Workflow)

The fastest way to get results is using the `magcalc` CLI.

### Step 1: Initialize a Project
Create a new configuration template:

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
-   `interactions`: Heisenberg ($J$), Dzyaloshinskii-Moriya ($D$), and Single-Ion Anisotropy ($K$).
-   `parameters`: Values for variables used in interactions (e.g., `J1: 3.23`).
-   `tasks`: Toggle `run_dispersion`, `run_sqw_map`, and `plot_*` flags.

### Step 3: Validate
Check if your configuration is physically valid without running heavy calculations:

```bash
magcalc validate config.yaml
```

### Step 4: Run
Execute the calculations. The CLI will handle ground state minimization, dispersion, and plotting automatically.

```bash
magcalc run config.yaml
```

Results (plots and data) are saved to the folders specified in the `output` and `plotting` sections (default: `plots/` and `data/`).

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
Support for Heisenberg exchange and Dzyaloshinskii-Moriya (DM) interactions.

```yaml
interactions:
  heisenberg:
    - pair: ["Fe1", "Fe1"]
      J: "J1"
      rij_offset: [0.5, 0.0, 0.0]
```

---

## 4. Default Python Library Usage (Advanced)

For complex workflows (e.g., scanning over parameters), you can use `pyMagCalc` as a Python library.

```python
import magcalc as mc
from magcalc.generic_model import GenericSpinModel
import yaml

# Load Config
with open("config.yaml") as f:
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
