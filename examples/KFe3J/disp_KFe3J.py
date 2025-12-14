# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:18:18 2018

@author: Kit Matan

Calculate and plot the spin-wave dispersion for KFe3(OH)6(SO4)2
(Updated to use MagCalc class)
"""
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
import sys
import os

# Adjust sys.path to correctly locate the magcalc package
# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (parent of parent of script if in examples/KFe3J)
# Assuming layout: root/examples/KFe3J/disp_KFe3J.py
project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

import magcalc as mc  # Import the refactored magcalc module
import spin_model as kfe3j_model  # Import the specific spin model
import yaml  # Import YAML library

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def calculate_and_save_dispersion(
    p, S, wr, cache_base, output_filename
):  # Added cache_base
    """Calculate spin-wave dispersion for KFe3(OH)6(SO4)2 and save results.
    Inputs:
        p: list of parameters
        S: spin value
        cache_base: base name for cache files
        wr: 'w' for write cache, 'r' for read cache
        output_filename: path to save the calculated data (.npz)
    Returns:
        bool: True if calculation and saving were successful, False otherwise."""
    # --- q-point generation (exactly as in the old version) ---
    intv = 0.05
    qsx = np.arange(0 - intv / 2, 2 * np.pi / np.sqrt(3) + intv / 2, intv)
    qsy = np.arange(0 - intv / 2, 2 * np.pi + intv / 2, intv)
    q_list = []
    for i in range(len(qsx)):
        q1 = np.array([qsx[i], 0, 0])
        q_list.append(q1)
    for i in range(len(qsy)):
        q1 = np.array([0, qsy[i], 0])
        q_list.append(q1)
    q_vectors_array = np.array(q_list)  # Convert to array for calculator
    # --- End q-point generation ---

    cache_base = "KFe3J"  # Cache file base name specific to this model
    En = None  # Initialize En to None

    try:
        # --- NEW WAY: Instantiate MagCalc and call method ---
        # First pass: Energy Minimization
        logger.info(f"Initializing MagCalc for Energy Minimization (cache base: {cache_base}_min)...")
        calculator_min = mc.MagCalc(
            spin_magnitude=S,
            hamiltonian_params=p,
            cache_file_base=cache_base + "_min",
            cache_mode=wr,
            spin_model_module=kfe3j_model,
        )
        
        logger.info("Minimizing energy to determine canting angle...")
        # Initial guess: 120-degree structure (Correct Chirality)
        x0 = np.array([
            np.pi/2, np.deg2rad(300),
            np.pi/2, np.deg2rad(180),
            np.pi/2, np.deg2rad(60)
        ])
        min_res = calculator_min.minimize_energy(x0=x0, method="L-BFGS-B")
        
        if min_res.success:
            logger.info(f"Energy minimization successful. Energy: {min_res.fun:.4f} meV")
            angles = min_res.x
            nspins = calculator_min.nspins
            thetas = [angles[2*i] for i in range(nspins)]
            avg_theta = np.mean(thetas)
            ca_minimized = avg_theta - np.pi/2.0
            logger.info(f"Minimized Average Theta: {avg_theta:.4f} rad, Calculated ca: {ca_minimized:.4f} rad")
            p.append(ca_minimized)
        else:
             logger.warning("Energy minimization failed. Using default analytical canting.")

        # Second pass: LSWT with updated parameters
        logger.info(f"Initializing MagCalc for LSWT ({cache_base})...")
        calculator = mc.MagCalc(
            spin_magnitude=S,
            hamiltonian_params=p,
            cache_file_base=cache_base,
            cache_mode=wr,
            spin_model_module=kfe3j_model,
        )

        logger.info("Calculating dispersion...")
        # Call the method on the instance, passing only the q-vectors
        res = calculator.calculate_dispersion(q_vectors_array)
        En = res.energies if res else None
        # --- End NEW WAY ---

        # --- Save Results ---
        if En is not None:
            logger.info(f"Saving calculation results to {output_filename}...")
            try:
                # Ensure En is suitable for saving (list of arrays or similar)
                results_to_save = {"q_vectors": q_vectors_array, "energies": En}
                calculator.save_results(output_filename, results_to_save)
                logger.info("Results saved successfully.")
                return True  # Indicate success
            except Exception as e:
                logger.error(
                    f"Failed to save results to {output_filename}: {e}", exc_info=True
                )
                return False  # Indicate failure
        else:
            logger.error("Calculation returned None, cannot save results.")
            return False  # Indicate failure

    except (FileNotFoundError, AttributeError, RuntimeError, ValueError) as e:
        logger.error(f"Error during MagCalc setup or calculation: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    return False  # Indicate failure if try block failed before saving


def plot_dispersion_from_file(filename):
    """Loads dispersion data from a .npz file and plots it."""
    # --- Plotting (only if calculation succeeded) ---
    logger.info(f"Loading data from {filename} for plotting...")
    try:
        data = np.load(
            filename, allow_pickle=True
        )  # allow_pickle needed if En was saved as list of arrays
        q_vectors_array = data["q_vectors"]
        En = data["energies"]
        logger.info("Data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Data file not found: {filename}. Cannot plot.")
        return
    except KeyError as e:
        logger.error(
            f"Missing expected key '{e}' in data file {filename}. Cannot plot."
        )
        return
    except Exception as e:
        logger.error(f"Error loading data from {filename}: {e}", exc_info=True)
        return

    # Need to recalculate qsx, qsy lengths based on loaded data
    # This assumes the q-path structure is consistent with how it was saved
    # Find the split point between qx and qy paths
    split_index = -1
    for i in range(1, len(q_vectors_array)):
        # Check if y-component becomes non-zero (start of qy path)
        # or if x-component becomes zero after being non-zero (end of qx path)
        if (
            abs(q_vectors_array[i, 1]) > 1e-9 and abs(q_vectors_array[i - 1, 1]) < 1e-9
        ) or (
            abs(q_vectors_array[i, 0]) < 1e-9 and abs(q_vectors_array[i - 1, 0]) > 1e-9
        ):
            # A more robust check might be needed depending on q-path generation
            # This simple check assumes qx path comes first, then qy path
            if (
                abs(q_vectors_array[i - 1, 1]) < 1e-9
            ):  # Check if previous was on qx path
                split_index = i
                break

    if split_index == -1:
        logger.warning(
            "Could not reliably determine split between qx and qy paths from loaded data. Assuming first half is qx."
        )
        split_index = len(q_vectors_array) // 2  # Fallback, might be incorrect

    qsx = q_vectors_array[:split_index, 0]  # Extract x-component for the first part
    qsy = q_vectors_array[split_index:, 1]  # Extract y-component for the second part
    len_qsx = len(qsx)
    len_qsy = len(qsy)

    if len(En) != len(q_vectors_array):
        logger.error(
            f"Mismatch between number of q-points ({len(q_vectors_array)}) and energy results ({len(En)}). Cannot plot reliably."
        )
        return

    logger.info("Processing results for plotting...")
    # --- Energy extraction (exactly as in the old version, with added safety) ---
    # Use len_qsx and len_qsy derived from loaded data
    try:
        Ekx1 = [
            En[i][0] if En[i] is not None and len(En[i]) > 0 else np.nan
            for i in range(len_qsx)
        ]
        Ekx2 = [
            En[i][1] if En[i] is not None and len(En[i]) > 1 else np.nan
            for i in range(len_qsx)
        ]
        Ekx3 = [
            En[i][2] if En[i] is not None and len(En[i]) > 2 else np.nan
            for i in range(len_qsx)
        ]
        Eky1 = [
            (
                En[len_qsx + i][0]
                if En[len_qsx + i] is not None and len(En[len_qsx + i]) > 0
                else np.nan
            )
            for i in range(len_qsy)
        ]
        Eky2 = [
            (
                En[len_qsx + i][1]
                if En[len_qsx + i] is not None and len(En[len_qsx + i]) > 1
                else np.nan
            )
            for i in range(len_qsy)
        ]
        Eky3 = [
            (
                En[len_qsx + i][2]
                if En[len_qsx + i] is not None and len(En[len_qsx + i]) > 2
                else np.nan
            )
            for i in range(len_qsy)
        ]
    except (IndexError, TypeError) as e:
        logger.error(
            f"Error extracting energies from results: {e}. Check number of modes and result structure."
        )
        return
    # --- End energy extraction ---

    # --- Plotting commands (exactly as in the old version) ---
    logger.info("Generating plot...")
    plt.figure()  # Ensure a new figure is created

    # Use qsx and qsy derived from loaded data
    qsyn = qsx[-1] + (
        qsy[-1] - qsy
    )  # More robust transformation based on actual ranges  # Keep original x-axis transformation
    plt.plot(
        qsx,
        Ekx1,
        "r-",
        qsx,
        Ekx2,
        "g-",
        qsx,
        Ekx3,
        "b-",
        qsyn,
        Eky1,
        "r-",
        qsyn,
        Eky2,
        "g-",
        qsyn,
        Eky3,
        "b-",
        marker=".",
        markersize=2,  # Optional: add markers to see points
    )
    # Vertical lines
    plt.plot([qsx[-1], qsx[-1]], [-1, 25], "k:")  # Use end of loaded qsx
    plt.plot(
        [
            qsyn[len(qsyn) // 2],  # Approximate K point location on transformed axis
            qsyn[len(qsyn) // 2],
        ],
        [-1, 25],
        "k:",
    )
    plt.xlim([0, 2 * np.pi / np.sqrt(3) + 2 * np.pi])
    plt.ylim([0, 20])
    plt.xticks([])  # Remove numerical ticks
    # Text labels
    plt.text(0, -1, r"$\Gamma$", fontsize=12)
    plt.text(2 * np.pi / np.sqrt(3) - 0.1, -1, "M", fontsize=12)  # Added fontsize
    plt.text(
        2 * np.pi / np.sqrt(3) + 2 * np.pi - 4 * np.pi / 3 - 0.1, -1, "K", fontsize=12
    )
    plt.text(2 * np.pi / np.sqrt(3) + 2 * np.pi - 0.1, -1, r"$\Gamma$", fontsize=12)
    plt.ylabel(r"$\hbar\omega$ (meV)", fontsize=12)
    plt.yticks(np.arange(0, 21, 5.0))
    plt.title("Spin-waves for KFe$_3$(OH)$_6$(SO$_4$)$_2$")
    # --- End original plotting commands ---

    plt.grid(axis="y", linestyle="--", alpha=0.7)  # Keep added grid
    plt.tight_layout()  # Keep added layout adjustment
    plt.show()


if __name__ == "__main__":
    st_main = default_timer()  # Start timer at the beginning

    # --- Load Configuration ---
    # --- Load Configuration ---
    config_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    config = {}
    try:
        with open(config_filename, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_filename}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_filename}. Exiting.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(
            f"Error parsing configuration file {config_filename}: {e}. Exiting."
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}. Exiting.")
        sys.exit(1)

    # Extract parameters from config
    model_p = config.get("model_params", {})
    calc_p = config.get("calculation", {})
    output_p = config.get("output", {})

    S_val = model_p.get("S", 2.5)
    params_val = [model_p.get(k, 0.0) for k in ["J1", "J2", "Dy", "Dz", "H"]]
    write_read_mode = calc_p.get("cache_mode", "r")
    cache_base_name = calc_p.get("cache_file_base", "KFe3J_cache")
    # Updated filename extraction logic
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_filename = output_p.get("disp_data_filename") or output_p.get("data_filename", "KFe3J_disp_data.npz")
    
    if not os.path.isabs(raw_filename):
        data_filename = os.path.join(script_dir, raw_filename)
    else:
        data_filename = raw_filename
    # --- End Configuration Loading ---

    calculation_successful = calculate_and_save_dispersion(
        params_val, S_val, write_read_mode, cache_base_name, data_filename
    )  # Pass cache_base_name
    if calculation_successful:
        logger.info("Calculation successful, proceeding to plot...")
        plot_dispersion_from_file(data_filename)

    et_main = default_timer()
    logger.info(f"Total run-time: {np.round((et_main - st_main) / 60, 2)} min.")

# %%
