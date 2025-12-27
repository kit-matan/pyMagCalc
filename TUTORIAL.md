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
2.  **Design**:
    -   **Load CIF**: Import your crystal structure. The app will automatically detect the space group and populate only the unique basis atoms.
    -   **Define Rules**: Add Bonding Rules (e.g., "Heisenberg" or "DM"). The system automatically expands these based on the structure's space group checking against symmetry constraints.
    -   **Configure Tasks**: Enable "Run Dispersion" or "Run S(Q,w)" in the Tasks tab.
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
-   `interactions`: Heisenberg ($J$), Dzyaloshinskii-Moriya ($D$), Anisotropic Exchange, Kitaev, and Single-Ion Anisotropy ($K$).
-   `minimization`: Settings for finding the ground state.
    -   `initial_configuration`: **Crucial** for complex systems to avoid local minima. define `theta` and `phi` for each atom.
    -   `n_workers`: Number of CPU cores for parallel minimization (default: 1).
    -   `early_stopping`: Stop after finding the same ground state N times (default: 0/disabled).
-   `plotting`: Control output behavior.
    -   `show_plot`: Set to `true` to see plots on screen, `false` to save only.
    -   `plot_structure`: Visualize the minimized magnetic state.
-   `parameters`: Values for variables used in interactions (e.g., `J1: 3.23`).
-   `tasks`: Toggle `minimization`, `dispersion`, `sqw_map`, and `export_csv`.
-   `output`: Define output filenames for data (e.g., `disp_csv_filename: "my_data.csv"`).

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
Support for Heisenberg, DM, Kitaev, and Single-Ion Anisotropy (SIA) interactions.

```yaml
interactions:
  heisenberg:
    - pair: ["Fe1", "Fe1"]
      J: "J1"
      rij_offset: [0.5, 0.0, 0.0]
  kitaev:
    - pair: ["Fe1", "Fe1"]
      K: "K_kit"
      rij_offset: [0.5, 0.5, 0.0]
  single_ion_anisotropy:
    - atom_label: "Fe1"
      value: "D_sia"
      axis: [0, 0, 1]
    - atom_label: "Fe1"
      value: "E_sia"
      axis: [1, 1, 0]
```

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
```

---

---

## 4. Best Practices & Troubleshooting

### Avoiding Imaginary Energies
If you see **imaginary energy eigenvalues** (warnings in the log), your system is likely in a saddle point or local minimum, not the true ground state.

**Solution**: Use `initial_configuration` in the `minimization` section to guide the solver.
*   For a 120-degree structure (e.g., Kagome), initialize spins at 0, 120, and 240 degrees.
*   Example (from `KFe3J/config_modern.yaml`):

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
*   Set `calculate_dispersion_new: false` if you only want to change plot aesthetics (titles, limits) without re-running the physics.

---

## 5. Default Python Library Usage (Advanced)

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
