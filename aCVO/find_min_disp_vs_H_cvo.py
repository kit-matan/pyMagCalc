#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finds the minimum of the spin-wave dispersion for the aCVO model
as a function of the applied magnetic field H.

This script iterates through a range of magnetic field values,
calculates the spin-wave dispersion for each field, finds the
minimum energy in that dispersion, and then plots the minimum
energy as a function of H.
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp  # For sp.Symbol('H')
import timeit

# --- Add pyMagCalc directory to sys.path ---
# This allows importing MagCalc from the parent directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYMAGCALC_DIR = os.path.dirname(SCRIPT_DIR)
if PYMAGCALC_DIR not in sys.path:
    sys.path.insert(0, PYMAGCALC_DIR)

try:
    from magcalc import MagCalc
except ImportError as e:
    print(f"Error importing MagCalc: {e}")
    print(f"Ensure magcalc.py is in {PYMAGCALC_DIR} or in your PYTHONPATH.")
    sys.exit(1)

# --- Import the local spin model for aCVO ---
try:
    import spin_model as cvo_model_module
except ImportError as e:
    print(f"Error importing local aCVO spin_model.py: {e}")
    sys.exit(1)


def find_min_dispersion_vs_H():
    """
    Main function to calculate and plot minimum dispersion energy vs. H.
    """
    start_time_script = timeit.default_timer()

    # --- Configuration ---
    config_filename = "config.yaml"
    config_filepath = os.path.join(SCRIPT_DIR, config_filename)

    if not os.path.exists(config_filepath):
        print(f"Error: Configuration file not found at {config_filepath}")
        return

    with open(config_filepath, "r") as f:
        config = yaml.safe_load(f)

    cvo_config = config.get("cvo_model", {})
    if not cvo_config:
        print(f"Error: 'cvo_model' section not found in {config_filepath}")
        return

    calc_settings = cvo_config.get("calculation", {})
    model_params_config = cvo_config.get("model_params", {})

    if not model_params_config:
        print(f"Error: 'model_params' section not found in {config_filepath}")
        return

    # Extract spin magnitude (S) and other Hamiltonian parameters
    # The order of parameters must match how aCVO/spin_model.py expects them in its Hamiltonian function
    # For aCVO/spin_model.py, spin_interactions(p) expects p = [J1, J2, J3, G1, Dx, H]
    param_order_for_magcalc = ["J1", "J2", "J3", "G1", "Dx", "H"]

    try:
        spin_S_val = float(
            model_params_config.pop("S")
        )  # Remove S as it's a direct MagCalc arg
        initial_hamiltonian_params_list = [
            float(model_params_config[p_name]) for p_name in param_order_for_magcalc
        ]
    except KeyError as e:
        print(
            f"Error: Missing parameter {e} in 'model_params' section of {config_filepath}. Required: 'S' and {param_order_for_magcalc}"
        )
        return

    # --- Parameters for the Scan ---
    # Magnetic field scan range (Tesla)
    H_values = np.linspace(0, 40, 41)  # 0 to 40 T, 1 T steps (41 points)
    print(
        f"Scanning H from {H_values[0]} T to {H_values[-1]} T in {len(H_values)} steps."
    )

    # Q-points for dispersion calculation (along (0, qy, 0) direction)
    # Based on the hint that minimum is around qy ~ 2.5
    # We scan qy from 2.3 to 2.7
    qy_min = 2.3
    qy_max = 2.7
    num_qy_points = 101  # Sufficiently dense sampling in the focused region
    qy_values = np.linspace(qy_min, qy_max, num_qy_points)
    q_points_for_scan = np.array([[0, qy_val, 0] for qy_val in qy_values])
    print(
        f"Calculating dispersion over {len(q_points_for_scan)} q-points along (0,qy,0) with qy in [{qy_min}, {qy_max}]."
    )

    # Cache settings for MagCalc
    # The symbolic cache (HMat_sym, Ud_sym) is generated once with H as a symbol.
    # So, the cache_file_base for MagCalc itself does not change with numerical H.
    magcalc_cache_file_base = calc_settings.get(
        "cache_file_base_prefix", "CVO_model_cache_min_disp_scan"
    )
    magcalc_cache_mode = calc_settings.get(
        "cache_mode", "r"
    )  # Default to 'r' after first 'w' run

    # --- Initialize MagCalc ---
    # We are now passing the spin_model_module directly, not config_filepath for model definition
    print(f"Initializing MagCalc with aCVO spin_model.py module.")
    print(
        f"MagCalc cache_file_base: {magcalc_cache_file_base}, cache_mode: '{magcalc_cache_mode}'"
    )

    try:
        calculator = MagCalc(
            spin_model_module=cvo_model_module,
            spin_magnitude=spin_S_val,
            hamiltonian_params=initial_hamiltonian_params_list,
            cache_file_base=magcalc_cache_file_base,
            cache_mode=magcalc_cache_mode,
        )
    except Exception as e:
        print(f"Error initializing MagCalc: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return

    # --- Iterate Over Magnetic Field Values ---
    min_energies_vs_H = []

    # Find the index of 'H' in our defined script parameter order.
    # MagCalc will have params_sym as p0, p1, ... corresponding to this order.
    try:
        H_script_param_index = param_order_for_magcalc.index("H")
    except ValueError:
        # This should not happen if 'H' is in param_order_for_magcalc
        print(
            f"Error: Magnetic field parameter 'H' not found in defined script parameter order: {param_order_for_magcalc}"
        )
        return

    print(f"MagCalc symbolic parameters: {calculator.params_sym}")
    print(f"Starting scan over {len(H_values)} magnetic field values...")
    for i, current_H_val in enumerate(H_values):
        loop_start_time = timeit.default_timer()

        # Update the 'H' parameter in MagCalc
        current_numerical_params = list(
            calculator.hamiltonian_params
        )  # Get a mutable copy
        current_numerical_params[H_script_param_index] = float(current_H_val)
        calculator.update_hamiltonian_params(current_numerical_params)

        # Calculate dispersion for the current H
        dispersion_data_list = calculator.calculate_dispersion(q_points_for_scan)

        # Find the minimum energy for the current H
        current_min_energy = np.inf
        if dispersion_data_list:
            for energies_at_q in dispersion_data_list:
                if energies_at_q is not None and energies_at_q.size > 0:
                    current_min_energy = min(
                        current_min_energy, np.nanmin(energies_at_q)
                    )

        min_energies_vs_H.append((current_H_val, current_min_energy))
        loop_duration = timeit.default_timer() - loop_start_time
        print(
            f"H = {current_H_val:.2f} T, Min Energy = {current_min_energy:.4f} meV (step {i+1}/{len(H_values)}, took {loop_duration:.2f}s)"
        )

    # --- Process and Plot Results ---
    H_array = np.array([item[0] for item in min_energies_vs_H])
    min_E_array = np.array([item[1] for item in min_energies_vs_H])

    plt.figure(figsize=(10, 6))
    plt.plot(H_array, min_E_array, marker="o", linestyle="-")
    plt.xlabel("Magnetic Field H (T)")
    plt.ylabel("Minimum Spin-Wave Energy (meV)")
    plt.title(
        f"aCVO: Minimum Dispersion Energy vs. Magnetic Field (qy in [{qy_min}, {qy_max}])"
    )
    plt.grid(True)
    plt.tight_layout()

    plot_filename = os.path.join(SCRIPT_DIR, "aCVO_min_energy_vs_H_plot.png")
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()

    # --- Save Raw Data ---
    data_filename = os.path.join(SCRIPT_DIR, "aCVO_min_energy_vs_H_data.npz")
    np.savez_compressed(data_filename, H_values=H_array, min_energies=min_E_array)
    print(f"Data saved to {data_filename}")

    total_script_time = timeit.default_timer() - start_time_script
    print(f"Total script execution time: {total_script_time:.2f} seconds.")


if __name__ == "__main__":
    find_min_dispersion_vs_H()
