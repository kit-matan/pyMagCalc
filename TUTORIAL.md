# pyMagCalc Tutorial

Welcome to the `pyMagCalc` tutorial! this guide will walk you through the complete workflow:
1.  **Defining a Spin Model** (declarative YAML).
2.  **Finding the Ground State** (Energy Minimization).
3.  **Calculating Spin Dynamics** (Dispersion and S(Q,ω)).
4.  **Visualizing Results** (3D Structure and Plots).

---

## Prerequisites

Ensure `pyMagCalc` is installed and the project root is in your `PYTHONPATH`:

```bash
# From project root
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

---

## Part 1: Workflow Overview

The modern `magcalc` workflow involves three main steps:

1.  **Configure**: Define your crystal and magnetic interactions in a `.yaml` file.
2.  **Minimize**: Use the `minimize_energy()` method to find the classical angle configuration ($\theta, \phi$) for the ground state.
3.  **Calculate**: Use this optimized structure to compute Magnon Dispersion and Neutron Scattering Intensity.

### Example: Finding the Ground State

Before calculating spin waves, we often need to know the magnetic ground state. `pyMagCalc` can find this for you numerically.

```python
import magcalc as mc
import numpy as np

# 1. Initialize
calc = mc.MagCalc(config_filepath="examples/KFe3J/config.yaml")

# 2. Minimize Energy
# You can provide a guess (x0) to guide the solver, especially for complex chiral structures.
# x0 is an array of [theta0, phi0, theta1, phi1, ...]
res = calc.minimize_energy()
print(f"Minimized Energy: {res.fun:.4f} meV")

# 3. Visualize Structure
# Save a 3D plot of the spins to verify your ground state
mc.plot_magnetic_structure(
    calc.sm.atom_pos(), 
    res.x, 
    save_filename="examples/plots/structure_check.png"
)
```

---

## Part 2: Configuring a Model (YAML)

Declarative YAML is the recommended way to define models.

**`material_config.yaml` example:**

```yaml
crystal_structure:
  lattice_parameters:
    a: 7.3  # Angstroms
    b: 7.3
    c: 17.2
    alpha: 90
    beta: 90
    gamma: 120
  atoms_uc:
    - label: "Fe1"
      element: "Fe"
      pos: [0.5, 0.5, 0.0]
      spin_S: 2.5
      magmom_classical: [0, 0, 1] # Initial guess for minimization

interactions:
  heisenberg:
    - pair: ["Fe1", "Fe1"] # Self-interaction handled by connectivity or distinct sites
      J: "J1"
      rij_offset: [0.5, 0.0, 0.0] # Relative vector or neighbor definition

parameters:
  J1: 3.14 # meV
```

---

## Part 3: Running Calculations

Once you have your model and ground state, calculating dispersion is straightforward.

### Dispersion Relation

Calculates eigenenergies for a path of Q-vectors.

```python
# Define typical high-symmetry path
q_path = np.array([
    [0, 0, 0],   # Gamma
    [0.5, 0, 0], # M
    [0.33, 0.33, 0] # K
])

# Calculate
energies_list = calc.calculate_dispersion(q_path)
```

### Dynamic Structure Factor S(Q,ω)

Calculates the neutron scattering intensity, properly applying the magnetic form factor and polarization factor.

```python
# Calculate for the same path
sqw_result = calc.calculate_sqw(q_path)

# Access results
print("Energies:", sqw_result.energies)
print("Intensities:", sqw_result.intensities)
```

---

## Part 4: Visualization & Plots

`pyMagCalc` now centralizes plot outputs for neatness.
-   **Structure Plots**: Use `mc.plot_magnetic_structure` to see the real-space configuration.
-   **Data Plots**: Use `matplotlib` or built-in helpers to plot dispersion/S(Q,ω).

By default, example scripts save plots to `examples/plots/`.

---

## API Reference: MagCalc Class

### `mc.MagCalc`

The main controller class.

#### `__init__(self, config_filepath=None, ...)`
Initializes the calculator.
*   `config_filepath`: Path to the YAML model definition.

#### `minimize_energy(self, x0=None, method='L-BFGS-B', ...)`
Numerically minimizes the classical Hamiltonian energy.
*   **Returns**: `scipy.optimize.OptimizeResult`.
*   **`x0`**: Optional initial guess for angles `[th0, ph0, th1, ph1...]`. Crucial for finding the correct ground state in complex energy landscapes (e.g., chirality).
*   **`method`**: Optimization algorithm.

#### `calculate_dispersion(self, q_vectors)`
Computes spin-wave accumulation energies.
*   **`q_vectors`**: List or Array of Q-points `[[h,k,l], ...]`.
*   **Returns**: `DispersionResult` object containing `.energies` (array).

#### `calculate_sqw(self, q_vectors)`
Computes neutron scattering intensities.
*   **Returns**: `SqwResult` object containing `.energies` and `.intensities`.

### `mc.plot_magnetic_structure`

Standalone function to visualize spin configurations.

```python
magcalc.plot_magnetic_structure(
    atom_positions,
    spin_angles,
    save_filename=None,
    title="Magnetic Structure"
)
```
*   **`atom_positions`**: Nx3 array of Cartesian coordinates.
*   **`spin_angles`**: 1D array of angles `[th, ph, th, ph...]` (usually from `minimize_energy().x`).
*   **Features**:
    *   3D scatter plot of atoms.
    *   Arrows representing spin direction.
    *   **Equal aspect ratio** enforced for accurate geometry.
    *   **Dynamic scaling**: Arrow size scales with nearest-neighbor distance.

---

## Advanced: Caching
`pyMagCalc` uses a dual caching system:
1.  **Symbolic Cache**: Stores diagonalization of the symbolic Hamiltonian. (Slow generation, fast reuse).
2.  **Numerical Cache**: Stores results of `calculate_dispersion` for specific parameter sets.

This ensures that re-running scripts is extremely fast once the heavy lifting is done.
