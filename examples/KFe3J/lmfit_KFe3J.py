# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:18:18 2018
Modified to use MagCalc class and lmfit.

@author: Kit Matan
@contributor: AI Assistant

Fit spin-wave model parameters for KFe3(OH)6(SO4)2 to experimental data
using lmfit and the MagCalc class.
"""
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
import sys
import os

# Adjust sys.path to correctly locate the magcalc package
# Get the directory of the current script (examples/KFe3J)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory
project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

import magcalc as mc  # Import the refactored magcalc module
import spin_model as kfe3j_model  # Import the specific spin model
import yaml  # Import YAML library
import sys
import os
import lmfit  # Import lmfit

# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global variable to hold the MagCalc instance (for efficiency within lmfit)
# We initialize it once before the fit starts.
calculator_instance = None


def load_experimental_data(filename="KFe3J_exp.dat"):
    """Loads experimental dispersion data from a file."""
    logger.info(f"Loading experimental data from '{filename}'...")
    try:
        # Original format seems to be: q_val, dq, energy, error, branch_idx, path_idx
        # where path_idx=2 means q_val is qx, path_idx=1 means q_val is qy
        data = np.loadtxt(filename, delimiter=",")  # Added delimiter=','
        q_val = data[:, 0]
        # dq = data[:, 1] # Not used directly in fitting objective
        e_exp = data[:, 2]
        err_exp = data[:, 3]
        branch_idx = data[:, 4].astype(int)  # Which calculated mode this corresponds to
        path_idx = data[:, 5].astype(int)  # 2 for qx path, 1 for qy path

        # Construct the q-vectors based on path_idx
        q_exp = np.zeros((len(q_val), 3))
        is_qx = path_idx == 2
        is_qy = path_idx == 1

        q_exp[is_qx, 0] = q_val[is_qx] * 2.0  # Scale only qx component
        q_exp[is_qy, 1] = q_val[is_qy]  # Use original q_val for qy component
        # qz is assumed 0

        logger.info(f"Loaded {len(e_exp)} experimental data points.")
        # Return q_exp, e_exp, err_exp, and also original q_val, path_idx, branch_idx for plotting/processing
        return (
            q_exp,
            e_exp,
            err_exp,
            q_val,
            path_idx,
            branch_idx,
        )  # Return original q_val

    except FileNotFoundError:
        logger.error(f"Experimental data file not found: {filename}")
        return None, None, None, None, None, None
    except Exception as e:
        logger.error(f"Error loading experimental data from {filename}: {e}")
        return None, None, None, None, None, None


def objective_function(
    params, q_exp, e_exp, err_exp, branch_idx_exp
):  # Added branch_idx_exp
    """
    Objective function for lmfit. Calculates residuals between
    calculated dispersion and experimental data.
    Assumes branch_idx_exp corresponds element-wise to q_exp, e_exp, err_exp.
    """
    global calculator_instance  # Use the global calculator instance

    # Extract numerical values from lmfit Parameters object
    # Expect: [J1, J2, Dy, Dz, H]
    J1_val = params["J1"].value
    J2_val = params["J2"].value
    Dy_val = params["Dy"].value
    Dz_val = params["Dz"].value
    H_mag_val = params["H"].value
    
    S_val = params["S"].value 

    # Construct correct parameter list for spin_model [J1, J2, Dy, Dz, H_dir, H_mag]
    # We enforce H || c for this analysis as per script context
    H_dir_val = [0.0, 0.0, 1.0]

    p_values_for_model = [J1_val, J2_val, Dy_val, Dz_val, H_dir_val, H_mag_val]

    # Update parameters in the existing calculator instance
    try:
        calculator_instance.update_spin_magnitude(S_val)
        calculator_instance.update_hamiltonian_params(p_values_for_model) 
        # Minimize energy to find new ground state for these parameters
        calculator_instance.minimize_energy(method="L-BFGS-B") 
    except Exception as e:
        logger.error(f"Error updating calculator parameters: {e}")
        # Return large residuals if update fails
        return np.full_like(e_exp, 1e6)

    # Calculate dispersion using the updated calculator
    # Use serial=True to avoid multiprocessing overhead in fitting loop
    try:
        res = calculator_instance.calculate_dispersion(q_exp, serial=True)
        energies_calc_list = res.energies if res else None
    except Exception as e:
        logger.warning(f"Error during dispersion calculation (likely invalid parameters): {e}")
        return np.full_like(e_exp, 1e6)  # Return large residuals on failure

    if energies_calc_list is None:
        logger.warning("Dispersion calculation returned None.")
        return np.full_like(e_exp, 1e6)

    # --- Process calculated energies ---
    # This part needs to match the experimental data structure.
    # Select the calculated mode corresponding to the experimental branch index.
    e_calc = []
    for i, en_arr in enumerate(energies_calc_list):
        branch = branch_idx_exp[i]  # Get the required branch for this q-point
        if en_arr is not None and len(en_arr) > branch and not np.isnan(en_arr[branch]):
            # Check if calculation succeeded, has enough modes, and the specific mode is not NaN
            e_calc.append(en_arr[branch])  # Select the correct mode
        elif (
            en_arr is not None
            and branch == 0
            and len(en_arr) > 0
            and not np.isnan(en_arr[0])
        ):
            # Fallback for safety? Or should this be NaN? Let's try NaN first.
            # logger.warning(f"Branch {branch} not available at q_exp[{i}], using branch 0 as fallback.")
            # e_calc.append(en_arr[0])
            e_calc.append(
                np.nan
            )  # Append NaN if the required branch is invalid/missing
        else:
            e_calc.append(np.nan)  # Append NaN if calculation failed for this q

    e_calc = np.array(e_calc)

    # Check for calculation failures (NaNs)
    valid_indices = ~np.isnan(e_calc)
    if not np.all(valid_indices):
        logger.warning(
            f"NaNs encountered in calculated energies for {np.sum(~valid_indices)} points."
        )
        # Only calculate residuals for valid points, return large residuals for others?
        # Or maybe stop the fit? For now, let's use only valid points.
        # This might bias the fit if failures are systematic.
        residuals = np.full_like(e_exp, 1e6)  # Default large residual
        residuals[valid_indices] = (
            e_calc[valid_indices] - e_exp[valid_indices]
        ) / err_exp[valid_indices]
    else:
        # Calculate residuals (difference divided by error)
        residuals = (e_calc - e_exp) / err_exp

    # You might want to add constraints or penalties here if needed

    # Print progress (optional, can be slow)
    # logger.debug(f"Params: {p_values}, Chi-sq: {np.sum(residuals**2):.4f}")

    return residuals


# --- New Plotting Function ---
def plot_fit_results(
    result, q_exp, e_exp, err_exp, q_val_orig, path_idx, branch_idx_exp, calculator
):
    """Plots the experimental data and the best-fit dispersion curves."""
    logger.info("Plotting fit results...")

    # --- Calculate dispersion curves using best-fit parameters ---
    best_fit_params = [
        result.params["J1"].value,
        result.params["J2"].value,
        result.params["Dy"].value,
        result.params["Dz"].value,
        [0.0, 0.0, 1.0],  # H_dir
        result.params["H"].value,  # H_mag
    ]
    best_fit_S = result.params["S"].value

    try:
        calculator.update_spin_magnitude(best_fit_S)
        calculator.update_hamiltonian_params(best_fit_params)
        # Minimize energy for the best fit parameters before plotting
        calculator.minimize_energy(method="L-BFGS-B")
    except Exception as e:
        logger.error(f"Error updating calculator with best-fit parameters: {e}")
        return

    # Define q-path for plotting curves (similar to disp_KFe3J.py)
    intv = 0.05
    qsx_plot = np.arange(0 - intv / 2, 2 * np.pi / np.sqrt(3) + intv / 2, intv)
    qsy_plot = np.arange(0 - intv / 2, 2 * np.pi + intv / 2, intv)
    q_plot_list = []
    for qx_val in qsx_plot:
        q_plot_list.append(np.array([qx_val, 0, 0]))
    for qy_val in qsy_plot:
        q_plot_list.append(np.array([0, qy_val, 0]))  # Path along axes

    try:
        res_fit = calculator.calculate_dispersion(q_plot_list)
        En_fit_list = res_fit.energies if res_fit else None
    except Exception as e:
        logger.error(f"Error calculating dispersion for plotting: {e}")
        return

    if En_fit_list is None:
        logger.error("Dispersion calculation for plotting returned None.")
        return

    # Extract modes for plotting
    len_qsx = len(qsx_plot)
    len_qsy = len(qsy_plot)
    try:
        Ekx1 = [
            (
                En_fit_list[i][0]
                if En_fit_list[i] is not None and len(En_fit_list[i]) > 0
                else np.nan
            )
            for i in range(len_qsx)
        ]
        Ekx2 = [
            (
                En_fit_list[i][1]
                if En_fit_list[i] is not None and len(En_fit_list[i]) > 1
                else np.nan
            )
            for i in range(len_qsx)
        ]
        Ekx3 = [
            (
                En_fit_list[i][2]
                if En_fit_list[i] is not None and len(En_fit_list[i]) > 2
                else np.nan
            )
            for i in range(len_qsx)
        ]
        Eky1 = [
            (
                En_fit_list[len_qsx + i][0]
                if En_fit_list[len_qsx + i] is not None
                and len(En_fit_list[len_qsx + i]) > 0
                else np.nan
            )
            for i in range(len_qsy)
        ]
        Eky2 = [
            (
                En_fit_list[len_qsx + i][1]
                if En_fit_list[len_qsx + i] is not None
                and len(En_fit_list[len_qsx + i]) > 1
                else np.nan
            )
            for i in range(len_qsy)
        ]
        Eky3 = [
            (
                En_fit_list[len_qsx + i][2]
                if En_fit_list[len_qsx + i] is not None
                and len(En_fit_list[len_qsx + i]) > 2
                else np.nan
            )
            for i in range(len_qsy)
        ]
    except (IndexError, TypeError) as e:
        logger.error(f"Error extracting fitted energies for plotting: {e}")
        return

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    # Plot calculated dispersion curves
    # Original transformation from lmfit_KFe3J.py: qsyn = 2 * np.pi + 2 * np.pi / np.sqrt(3) - qsy
    # Let's use the one from disp_KFe3J.py for consistency: qsyn = qsx[-1] + (qsy[-1] - qsy)
    qsyn_plot = qsx_plot[-1] + (qsy_plot[-1] - qsy_plot)
    plt.plot(qsx_plot, Ekx1, "r-", label="Mode 1 (Fit)")
    plt.plot(qsx_plot, Ekx2, "g-", label="Mode 2 (Fit)")
    plt.plot(qsx_plot, Ekx3, "b-", label="Mode 3 (Fit)")
    plt.plot(qsyn_plot, Eky1, "r-")
    plt.plot(qsyn_plot, Eky2, "g-")
    plt.plot(qsyn_plot, Eky3, "b-")

    # Plot experimental data
    is_qx = path_idx == 2
    is_qy = path_idx == 1
    # Use the original 1D q_val and apply scaling/transformation for plotting
    kx_exp_plot = q_val_orig[is_qx] * 2.0  # Scale qx points
    ky_exp_plot = q_val_orig[is_qy]  # Original qy points
    # Apply same transformation to experimental qy points
    kyn_exp_plot = qsx_plot[-1] + (qsy_plot[-1] - ky_exp_plot)
    # Assuming error bars are symmetric (err_exp)
    plt.errorbar(
        kx_exp_plot,  # Use 1D scaled qx values
        e_exp[is_qx],
        yerr=err_exp[is_qx],
        fmt="ko",
        label="Data (qx path)",
        capsize=3,
    )
    plt.errorbar(
        kyn_exp_plot,  # Use 1D transformed qy values
        e_exp[is_qy],
        yerr=err_exp[is_qy],
        fmt="ks",
        label="Data (qy path)",
        capsize=3,
    )

    # Plot formatting (similar to original lmfit script and disp_KFe3J.py)
    plt.plot([qsx_plot[-1], qsx_plot[-1]], [-1, 25], "k:")  # M point line
    # Approximate K point location on transformed axis
    k_point_approx_x = qsyn_plot[len(qsyn_plot) // 2]  # Midpoint of transformed qy path
    plt.plot([k_point_approx_x, k_point_approx_x], [-1, 25], "k:")  # K point line
    plt.xlim([qsx_plot[0], qsyn_plot[0]])  # Full range of transformed axis
    plt.ylim([0, 20])
    plt.xticks([])
    plt.text(qsx_plot[0], -1, r"$\Gamma$", fontsize=12, ha="center")
    plt.text(qsx_plot[-1], -1, "M", fontsize=12, ha="center")
    plt.text(
        k_point_approx_x, -1, "K", fontsize=12, ha="center"
    )  # Approx K point label
    plt.text(
        qsyn_plot[0], -1, r"$\Gamma$", fontsize=12, ha="center"
    )  # End of transformed axis
    plt.ylabel(r"$\hbar\omega$ (meV)", fontsize=12)
    plt.yticks(np.arange(0, 21, 5.0))
    plt.title("Spin-waves for KFe$_3$(OH)$_6$(SO$_4$)$_2$ - Fit Result")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    st_main = default_timer()

    # --- Load Configuration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_filename = os.path.join(script_dir, "config.yaml")
    config = {}
    try:
        with open(config_filename, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_filename}")
    except FileNotFoundError:
        logger.warning(
            f"Configuration file {config_filename} not found. Using default parameters."
        )
    except yaml.YAMLError as e:
        logger.error(
            f"Error parsing configuration file {config_filename}: {e}. Using default parameters."
        )
    except Exception as e:
        logger.error(
            f"Unexpected error loading configuration: {e}. Using default parameters."
        )

    # Extract parameters from config
    # prioritize 'hamiltonian_params' list as per new config format
    ham_params_list = config.get("hamiltonian_params", None)
    
    if ham_params_list and len(ham_params_list) >= 6:
        logger.info(f"Using initial parameters from config: {ham_params_list}")
        initial_params = ham_params_list
    else:
        logger.warning("hamiltonian_params not found or invalid in config. Using defaults.")
        # Fallback defaults
        initial_params = [
            0.0, # J1
            0.0, # J2
            0.0, # Dy
            0.0, # Dz
            [0.0, 0.0, 1.0], # H_dir
            0.0, # H_mag
        ]

    # Model params section (S)
    model_p_dict = config.get("parameters", {}) # Look in 'parameters' dict for S
    initial_S = model_p_dict.get("S", config.get("model_params", {}).get("S", 2.5))
    calc_p = config.get("calculation", {})
    fit_p = config.get("fitting", {})
    cache_base_name = calc_p.get("cache_file_base", "KFe3J_cache")
    exp_data_file = fit_p.get("exp_data_file", "../data/sw_KFe3J.txt")  # Updated default path to examples/data
    # --- End Configuration Loading ---

    # --- Load Experimental Data ---
    q_exp, e_exp, err_exp, q_val_orig, path_idx, branch_idx_exp = (
        load_experimental_data(  # Renamed q_val_scaled -> q_val_orig
            os.path.join(script_dir, exp_data_file)
        )
    )
    if q_exp is None:
        sys.exit(1)  # Exit if data loading failed

    # --- Initialize MagCalc Instance (Crucial: Use 'r' mode for fitting) ---
    logger.info("Initializing MagCalc instance for fitting (using cache mode 'r')...")
    try:
        calculator_instance = mc.MagCalc(
            spin_magnitude=initial_S,
            hamiltonian_params=initial_params,
            cache_file_base=cache_base_name,
            cache_mode="w",  # Force regenerate to ensure consistency, init happens once
            spin_model_module=kfe3j_model,
        )
    except Exception as e:
        logger.error(
            f"Failed to initialize MagCalc: {e}. Ensure cache files exist (run once with 'w' mode)."
        )
        sys.exit(1)

    # --- Setup lmfit Parameters ---
    params = lmfit.Parameters()
    # Add parameters with initial values, bounds, vary=True/False
    # Example: params.add('name', value=initial_value, min=lower_bound, max=upper_bound, vary=True)
    params.add("J1", value=initial_params[0], min=0.0, vary=True)  # J1 likely positive
    params.add("J2", value=initial_params[1], vary=True)
    params.add("Dy", value=initial_params[2], vary=True)
    params.add("Dz", value=initial_params[3], vary=True)
    params.add("H", value=initial_params[5], vary=False)  # H_mag is at index 5
    params.add("S", value=initial_S, min=0.1, vary=False)  # Keep S fixed for example

    # --- Perform the Fit ---
    logger.info("Starting lmfit minimization...")
    # Pass branch_idx_exp to the objective function via fcn_args
    minimizer = lmfit.Minimizer(
        objective_function, params, fcn_args=(q_exp, e_exp, err_exp, branch_idx_exp)
    )
    # Choose a method, e.g., 'leastsq' (Levenberg-Marquardt), 'nelder', 'powell'
    result = minimizer.minimize(method="leastsq")

    # --- Report Results ---
    logger.info("\n" + "=" * 20 + " Fitting Complete " + "=" * 20)
    lmfit.report_fit(result)

    # You can also plot the best fit against the data here if desired
    # --- Plot Fit Results ---
    if calculator_instance is not None:
        plot_fit_results(
            result,
            q_exp,
            e_exp,
            err_exp,
            q_val_orig,  # Pass original q_val
            path_idx,  # Pass path index
            branch_idx_exp,  # Correct variable name
            calculator_instance,
        )

    et_main = default_timer()
    logger.info(f"Total run-time: {np.round((et_main - st_main) / 60, 2)} min.")

# %%
