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
import magcalc as mc  # Import the refactored magcalc module
import sys

# It's often good practice to configure logging in the main script too
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def plot_disp(p, S, wr):
    """Plot spin-wave dispersion for KFe3(OH)6(SO4)2
    Inputs:
        p: list of parameters
        S: spin value
        wr: 'w' for write to file, 'r' for read from file"""

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
        logger.info(f"Initializing MagCalc for {cache_base}...")
        calculator = mc.MagCalc(
            spin_magnitude=S,
            hamiltonian_params=p,
            cache_file_base=cache_base,
            cache_mode=wr,
        )

        logger.info("Calculating dispersion...")
        # Call the method on the instance, passing only the q-vectors
        En = calculator.calculate_dispersion(q_vectors_array)
        # --- End NEW WAY ---

    except (FileNotFoundError, AttributeError, RuntimeError, ValueError) as e:
        logger.error(f"Error during MagCalc setup or calculation: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

    # --- Plotting (only if calculation succeeded) ---
    if En is None:
        logger.error("Dispersion calculation failed or returned None. Cannot plot.")
        return  # Exit the function if En is None

    if len(En) != len(q_vectors_array):
        logger.error(
            f"Mismatch between number of q-points ({len(q_vectors_array)}) and energy results ({len(En)}). Cannot plot reliably."
        )
        return

    logger.info("Processing results for plotting...")
    # --- Energy extraction (exactly as in the old version, with added safety) ---
    try:
        Ekx1 = [En[i][0] if En[i] is not None else np.nan for i in range(len(qsx))]
        Ekx2 = [En[i][1] if En[i] is not None else np.nan for i in range(len(qsx))]
        Ekx3 = [En[i][2] if En[i] is not None else np.nan for i in range(len(qsx))]
        Eky1 = [
            En[len(qsx) + i][0] if En[len(qsx) + i] is not None else np.nan
            for i in range(len(qsy))
        ]
        Eky2 = [
            En[len(qsx) + i][1] if En[len(qsx) + i] is not None else np.nan
            for i in range(len(qsy))
        ]
        Eky3 = [
            En[len(qsx) + i][2] if En[len(qsx) + i] is not None else np.nan
            for i in range(len(qsy))
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

    qsyn = (
        2 * np.pi + 2 * np.pi / np.sqrt(3) - qsy
    )  # Keep original x-axis transformation
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
    )
    # Vertical lines
    plt.plot([2 * np.pi / np.sqrt(3), 2 * np.pi / np.sqrt(3)], [-1, 25], "k:")
    plt.plot(
        [
            2 * np.pi / np.sqrt(3) + 2 * np.pi - 4 * np.pi / 3,
            2 * np.pi / np.sqrt(3) + 2 * np.pi - 4 * np.pi / 3,
        ],
        [-1, 25],
        "k:",
    )
    # Limits and labels
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
    st_main = default_timer()
    # KFe3Jarosite
    S_val = 5.0 / 2.0  # spin value
    params = [3.23, 0.11, 0.218, -0.195, 0]

    # Set cache mode: 'w' for first run, 'r' afterwards
    write_read_mode = "r"  # Default to read mode

    logger.info(f"Starting KFe3J dispersion plot with mode='{write_read_mode}'...")
    plot_disp(params, S_val, write_read_mode)

    et_main = default_timer()
    logger.info(f"Total run-time: {np.round((et_main - st_main) / 60, 2)} min.")

# %%
