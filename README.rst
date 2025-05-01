===========================================
pyMagCalc: Linear Spin-Wave Theory Calculator
===========================================

.. image:: https://img.shields.io/badge/python-3.8%2B-blue.svg
   :target: https://www.python.org/
.. image:: https://img.shields.io/badge/status-development-orange.svg

Introduction
------------

``pyMagCalc`` is a Python package for performing Linear Spin-Wave Theory (LSWT) calculations. It allows users to define a spin model (Hamiltonian, magnetic structure, lattice) and compute spin-wave dispersion relations and dynamic structure factors S(Q,ω).

This version features a significant refactoring from the original functional approach into an object-oriented design centered around the ``MagCalc`` class, offering improved usability, maintainability, and performance.

Key Features
------------

*   **Object-Oriented Design:** Core calculations managed by the ``MagCalc`` class.
*   **Symbolic Hamiltonian:** Generates the symbolic quadratic boson Hamiltonian (``gen_HM``) using SymPy.
*   **Numerical Calculations:**
    *   Calculates spin-wave dispersion relations.
    *   Calculates the dynamic structure factor S(Q,ω), including the polarization factor.
*   **Performance:** Utilizes multiprocessing (``Pool``) and ``tqdm`` progress bars for potentially long calculations (symbolic substitution, dispersion, S(Q,ω)).
*   **Caching:** Saves/loads computationally expensive symbolic matrices (``HMat_sym``, ``Ud_sym``) to/from pickle files (``.pck``) to speed up subsequent runs.
*   **Flexibility:** Allows dynamic updating of spin magnitude (``S``) and Hamiltonian parameters after initialization.
*   **Output:** Provides a method to save calculation results (q-vectors, energies, intensities) to compressed NumPy files (``.npz``).
*   **Code Quality:** Includes type hinting and basic logging for better maintainability and traceability.
*   **Utilities:** Includes a basic CIF file reader (`read_cif.py`) and an example of using it to define a spin model (`KFe3J/spin_model_cif.py`).
*   **Ground State Example:** Includes an example script (`KFe3J/magStruct.py`) for finding the classical ground state using optimization.
*   **Testing:** Includes unit and integration tests using ``pytest``.

Dependencies
------------

*   Python (3.8+ recommended)
*   NumPy
*   SciPy
*   SymPy
*   Matplotlib (for plotting examples)
*   tqdm (for progress bars)
*   PyYAML (for config file in some examples)
*   ASE (Atomistic Simulation Environment, for CIF reading utility)
*   pytest (for running tests)

You can typically install these using pip:

.. code-block:: bash

   pip install numpy scipy sympy matplotlib tqdm pyyaml ase pytest

Installation
------------

Currently, ``pyMagCalc`` can be used directly from the source directory. Ensure the main directory (`pyMagCalc`) is in your Python path or run your scripts from within this directory.

Basic Usage (using MagCalc class)
---------------------------------

1.  **Define your Spin Model:** Create a Python module (e.g., ``my_spin_model.py``) containing the required functions (see "Spin Model Definition" below).
2.  **Write your Calculation Script:**

.. code-block:: python

   import numpy as np
   import magcalc as mc
   import my_spin_model  # Import your spin model module
   import logging

   # Configure logging (optional but recommended)
   logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

   # --- User Inputs ---
   spin_S = 2.5
   hamiltonian_params = [3.23, 0.11, 0.218, -0.195, 0.0] # Example: J1, J2, Dy, Dz, H
   cache_base = "my_model_run" # Base name for cache files in pckFiles/
   cache_mode = 'w' # 'w' to generate cache first time, 'r' to read afterwards
   output_npz_file = "my_results.npz"

   # Define q-points for calculation (example path)
   q_points = np.linspace([0,0,0], [np.pi, 0, 0], 51)

   # --- Calculation ---
   try:
       # 1. Instantiate the calculator
       calculator = mc.MagCalc(
           spin_magnitude=spin_S,
           hamiltonian_params=hamiltonian_params,
           cache_file_base=cache_base,
           spin_model_module=my_spin_model, # Pass the imported module
           cache_mode=cache_mode
       )

       # 2. Calculate Dispersion
       energies = calculator.calculate_dispersion(q_points)

       # 3. Calculate S(q,w)
       q_out, sqw_energies, intensities = calculator.calculate_sqw(q_points)

       # 4. Save results (example)
       if energies is not None and q_out is not None:
           results_data = {
               'q_disp': q_points,
               'E_disp': energies,
               'q_sqw': q_out,
               'E_sqw': sqw_energies,
               'I_sqw': intensities
           }
           calculator.save_results(output_npz_file, results_data)

       # 5. Optionally update parameters and recalculate
       # calculator.update_hamiltonian_params([3.3, 0.1, 0.2, -0.2, 0.0])
       # energies_updated = calculator.calculate_dispersion(q_points)

   except Exception as e:
       logging.exception(f"An error occurred: {e}")

Spin Model Definition
---------------------

Users must provide a Python module (e.g., ``spin_model.py``) that defines the magnetic system. This module must contain the following functions:

*   ``atom_pos()``: Returns a NumPy array of atomic positions within the magnetic unit cell. Shape: `(N, 3)`, where N is the number of spins in the unit cell.
*   ``atom_pos_ouc()``: Returns a NumPy array of atomic positions including those in neighboring unit cells relevant for interactions. Shape: `(M, 3)`, where M >= N.
*   ``mpr(params)``: Returns a list of SymPy rotation matrices (one for each spin in the unit cell) to transform local spin operators (quantization axis along local z) to the global coordinate system. Can depend on Hamiltonian parameters `params`.
*   ``spin_interactions(params)``: Returns a tuple `(Jex, DM)` where `Jex` is a SymPy matrix of exchange constants `J_ij` between spin `i` (in unit cell) and spin `j` (in OUC), and `DM` is a SymPy matrix containing DM vectors `D_ij`.
*   ``Hamiltonian(Sxyz, params)``: Returns a SymPy expression for the total spin Hamiltonian, defined using the provided list of global spin operators `Sxyz` (list of 3x1 SymPy matrices) and Hamiltonian parameters `params`.

Refer to ``/Users/kmatan/Library/CloudStorage/OneDrive-MahidolUniversity/research/magcalc/pyMagCalc/KFe3J/spin_model.py`` and ``/Users/kmatan/Library/CloudStorage/OneDrive-MahidolUniversity/research/magcalc/pyMagCalc/spin_model_fm.py`` for examples.

Caching
-------

The symbolic matrices ``HMat_sym`` (representing 2gH) and ``Ud_sym`` can be computationally expensive to generate. ``MagCalc`` uses caching to avoid regenerating them on every run.

*   **`cache_mode='w'` (Write):** Generates the symbolic matrices using ``gen_HM`` and saves them to ``pckFiles/<cache_file_base>_HM.pck`` and ``pckFiles/<cache_file_base>_Ud.pck``. Use this mode the first time you run a calculation for a specific model or if the model definition changes.
*   **`cache_mode='r'` (Read):** Loads the symbolic matrices from the cache files. This is much faster. Use this mode for subsequent runs with the same model but potentially different numerical parameters or q-points.
*   **`cache_file_base`:** A string identifier used to name the cache files. Choose a unique name for each distinct spin model.

The ``pckFiles`` directory will be created if it doesn't exist.

Examples
--------

The ``/Users/kmatan/Library/CloudStorage/OneDrive-MahidolUniversity/research/magcalc/pyMagCalc/KFe3J/`` directory contains example scripts for KFe3 Jarosite:

*   ``spin_model.py``: Defines the KFe3J spin model.
*   ``config.yaml``: Configuration file for parameters used by ``disp_KFe3J.py``.
*   ``disp_KFe3J.py``: Calculates and plots dispersion using ``MagCalc`` and reads parameters from ``config.yaml``.
*   ``EQmap_KFe3J.py``: Calculates and plots the S(Q,ω) intensity map using ``MagCalc``.
*   ``calculate_and_save_dispersion.py``: Example showing calculation and saving results using ``MagCalc``.
*   ``magStruct.py``: Example script to find the classical magnetic ground state.
*   ``spin_model_cif.py``: Example defining the spin model based on a CIF file using ``read_cif.py``.
*   ``test_magcalc.py``: Pytest tests for the ``magcalc`` module.

*Note:* ``HKmap_KFe3J.py`` and ``lmfit_KFe3J.py`` currently still use the older functional interface (``magcalc2.py``) and have not been updated to use the ``MagCalc`` class.

Testing
-------

Unit and integration tests are written using ``pytest``. Run tests from the ``pyMagCalc`` directory:

.. code-block:: bash

   pytest

TODO / Future Work
------------------

*   Update ``HKmap_KFe3J.py`` and ``lmfit_KFe3J.py`` to use the ``MagCalc`` class.
*   Implement more sophisticated classical ground state finding methods.
*   Explore alternative LSWT formalisms (e.g., Colpa diagonalization).
*   Improve error handling and logging.
*   Consider packaging for easier distribution (e.g., via pip).

Authors
-------

*   Kit Matan
*   Pharit Piyawongwatthana
*   AI Assistant (Refactoring)