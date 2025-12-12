# pyMagCalc Tutorial

Welcome to the `pyMagCalc` tutorial! This guide will walk you through setting up a simple spin model, calculating its dispersion relation, and exploring more complex examples.

## Prerequisites

Before starting, ensure you have installed the package dependencies:
```bash
pip install -r requirements.txt
```
And add the package to your python path:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

---

## Step 1: Defining a Spin Model (YAML)

`pyMagCalc` uses a declarative YAML format to define materials. Let's create a simple **1D Ferromagnetic Chain** as a "Hello World" example.

Create a file named `simple_chain.yaml`:

```yaml
crystal_structure:
  lattice_parameters:
    a: 3.0
    b: 10.0 # Large spacing to isolate chains
    c: 10.0
    alpha: 90
    beta: 90
    gamma: 90

  atoms_uc:
    - label: "spin1"
      element: "Cu"
      pos: [0.0, 0.0, 0.0]
      spin_S: 0.5
      magmom_classical: [0.0, 0.0, 1.0] # Ferromagnetic alignment along c-axis (z)

interactions:
  heisenberg:
    # Nearest neighbor exchange along the chain (a-axis)
    # J < 0 for Ferromagnetic in this convention H = 0.5 * sum J * S.S
    # (Note: Check convention in your specific version, frequent convention is H = sum J S.S)
    - pair: ["spin1", "spin1"]
      J: "J1"
      rij_offset: [1, 0, 0] # Interaction with neighbor in next cell along a

parameters:
  J1: -1.0  # Ferromagnetic exchange (meV)

calculation_settings:
  neighbor_shells: [1, 0, 0] # Only look for neighbors along a-axis
```

---

## Step 2: Running the Calculation

You can use the `scripts/run_magcalc.py` script to run this model.

```bash
python scripts/run_magcalc.py simple_chain.yaml
```

*Note: The script might expect specific output configuration sections in the YAML. If `run_magcalc.py` requires more setup, we can use the Python API directly.*

---

## Step 3: Using the Python API

For more control, use Python scripts. Create `run_chain.py`:

```python
import numpy as np
import matplotlib.pyplot as plt
import magcalc as mc

# 1. Initialize the Calculator
# You can load from the YAML file we just created
calc = mc.MagCalc(config_filepath="simple_chain.yaml")

# 2. Define the path in Reciprocal Space
# For a 1D chain along 'a', we want to scan q = (h, 0, 0)
h_vals = np.linspace(0, 2, 100)
q_vectors = []
for h in h_vals:
    # Convert hkl to absolute q-vector if needed, or pass hkl if supported by your version.
    # Typically generically Reciprocal Lattice Units (r.l.u) are used.
    q_vectors.append([h, 0, 0])
q_vectors = np.array(q_vectors)

# 3. Calculate Dispersion
# Returns energies for each band at each q-point
energies = calc.calculate_dispersion(q_vectors) 

# 4. Plot
plt.figure(figsize=(8, 5))
plt.plot(h_vals, energies, 'b-')
plt.xlabel("q (r.l.u) [h, 0, 0]")
plt.ylabel("Energy (meV)")
plt.title("1D Ferromagnetic Chain Dispersion")
plt.grid(True)
plt.savefig("chain_dispersion.png")
plt.show()
```

Run it:
```bash
python run_chain.py
```

You should see a quadratic-like dispersion near q=0 for a ferromagnet.

---

## Step 4: Advanced Example (Jarosite)

For a full-scale example, check `examples/KFe3J/`.

1.  **Navigate to the folder**:
    ```bash
    cd examples/KFe3J
    ```
2.  **Inspect the parameters**:
    Open `KFe3J_declarative.yaml`. You will see:
    *   3 Fe atoms in a Kagome geometry.
    *   Next-Nearest Neighbor interactions ($J_1$, $J_2$).
    *   Dzyaloshinskii-Moriya (DM) interactions ($D_z$, $D_p$).
3.  **Run the plotting script**:
    ```bash
    python disp_KFe3J.py
    ```
    *(Note: Ensure you are in the project root or have set PYTHONPATH)*

---

## Step 5: Troubleshooting

*   **`ModuleNotFoundError: No module named 'magcalc'`**:
    Make sure you are running python from the parent folder of `magcalc` or have added it to `sys.path`.
*   **SymPy Slowness**:
    The first time you run a model, `pyMagCalc` uses SymPy to derive the Hamiltonian symbolically. This can take time. It caches the result, so the second run will be essentially instantaneous.

---

## Support

If you encounter issues, please open an issue on the GitHub repository.
