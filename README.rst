===========================================
pyMagCalc: Linear Spin-Wave Theory Calculator
===========================================

.. image:: https://img.shields.io/badge/python-3.8%2B-blue.svg
   :target: https://www.python.org/
.. image:: https://img.shields.io/badge/status-development-orange.svg

Introduction
------------

``pyMagCalc`` is a Python package for performing Linear Spin-Wave Theory (LSWT) calculations. It allows users to define a spin model (Hamiltonian, magnetic structure, lattice) and compute spin-wave dispersion relations and dynamic structure factors S(Q,ω).

Key Features
------------

*   **Object-Oriented Design:** Core calculations managed by the ``MagCalc`` class within the ``magcalc`` package.
*   **Symbolic Hamiltonian:** Generates symbolic quadratic boson Hamiltonians using SymPy.
*   **Numerical Calculations:** Computes dispersion relations and S(Q,ω) with polarization factors.
*   **Performance:** Parallelized calculations using ``multiprocessing``.
*   **Caching:** Caches expensive symbolic computations to speed up subsequent runs.
*   **Flexible Configuration:** Supports both Python-module defined models and YAML-based declarative configurations.

Directory Structure
-------------------

*   ``magcalc/``: Core python package containing ``MagCalc``, ``GenericSpinModel``, and utilities.
*   ``scripts/``: Top-level executable scripts (e.g., ``run_magcalc.py``, ``inspect_hm.py``).
*   ``examples/``: Sample data, configurations, and scripts (e.g., ``KFe3J/``, ``ZnCVO/``).
*   ``tests/``: Unit and integration tests.
*   ``archive/``: Deprecated or unused files.

Dependencies
------------

*   Python (3.8+)
*   NumPy
*   SciPy
*   SymPy
*   Matplotlib
*   tqdm
*   PyYAML
*   ASE (Atomistic Simulation Environment, used for CIF file reading)
*   pytest (for testing)

Installation
------------

1.  Clone the repository.
2.  Install dependencies:

    .. code-block:: bash

       pip install -r requirements.txt

3.  Ensure the project root is in your ``PYTHONPATH`` or run scripts from the project root directory.

Basic Usage
-----------

Running the Example Script
~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to run ``pyMagCalc`` is using the provided ``run_magcalc.py`` script with a configuration file.

.. code-block:: bash

   python scripts/run_magcalc.py examples/KFe3J/KFe3J_declarative.yaml

Using as a Library
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import magcalc as mc 
   # ... define model or load config ...

   # Example with configuration file:
   # calculator = mc.MagCalc(config_filepath="examples/KFe3J/KFe3J_declarative.yaml")
   # energies = calculator.calculate_dispersion(q_points)

Spin Model Definition
---------------------

You can define a spin model in two ways:

1.  **Declarative YAML (Recommended):** Define crystal structure, interactions, and parameters in a YAML file. See ``examples/KFe3J/KFe3J_declarative.yaml`` for a complete example.
2.  **Python Module (Legacy/Advanced):** Create a module implementing ``atom_pos``, ``spin_interactions``, ``Hamiltonian``, etc. See ``examples/KFe3J/spin_model.py`` for an example.

Examples
--------

The ``examples/`` directory contains structured examples:

*   ``examples/KFe3J/``: KFe3(OH)6(SO4)2 (Jarosite) example.
    *   ``KFe3J_declarative.yaml``: Configuration for the ``GenericSpinModel``.
    *   ``spin_model.py``: Legacy Python-defined model.
*   ``examples/ZnCVO/`` & ``examples/aCVO/``: Additional material examples.

Testing
-------

Run the test suite using ``pytest`` from the project root:

.. code-block:: bash

   pytest