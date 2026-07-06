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
    -   `early_stopping`: Stop after finding the same ground state N times (default: 3).
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
    - type: "heisenberg"
      distance: 3.23
      value: "J1"
    - type: "dm_interaction"
      distance: 3.23
      value: [0, "Dy", "Dz"]
```

#### 2. Explicit Pairs
Define specific bonds between atom labels. Use `rij_offset` for neighbors in other cells.
```yaml
interactions:
  heisenberg:
    - pair: ["Fe1", "Fe1"]
      J: "J1"
      rij_offset: [1, 0, 0]
  single_ion_anisotropy:
    - atom_label: "Fe1"
      value: "D_sia"
      axis: [0, 0, 1]
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

## 4. Best Practices & Troubleshooting

### Avoiding Imaginary Energies
If you see **imaginary energy eigenvalues** (warnings in the log), your system is likely in a saddle point or local minimum, not the true ground state.

**Solution**: Use `initial_configuration` in the `minimization` section to guide the solver.
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
