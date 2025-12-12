#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for KFe3(OH)6(SO4)2 using the configuration-driven MagCalc.
Compares results from the new config-driven approach with the original
hardcoded spin_model.py.
"""
import numpy as np
import os
import sys
import logging
from timeit import default_timer

# Adjust sys.path to correctly locate the pyMagCalc package
# Get the directory of the current script (tests)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (pyMagCalc)
project_root_dir = os.path.dirname(current_script_dir)

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

import magcalc as mc

from sympy import Matrix as sp_Matrix  # For type hinting
# Helper to import KFe3J.spin_model from examples
sys.path.insert(0, os.path.join(project_root_dir, 'examples'))
import KFe3J.spin_model as kfe3j_hardcoded_spin_model  # Original model

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_q_vectors_for_test():
    """Generates a simple path of q-vectors for testing."""
    q_list = []
    # Gamma to M path (along kx up to 2*pi/sqrt(3))
    for qx_val in np.linspace(0, 2 * np.pi / np.sqrt(3), 5):
        q_list.append(np.array([qx_val, 0, 0]))
    # M to K path (example, not a straight line in original plot)
    # For simplicity, just a few points along ky from M
    # M is at (2pi/sqrt(3), 0, 0)
    # K is approx (2pi/sqrt(3), 2pi/3, 0)
    # For this test, let's just do a simple path to keep it short
    for qy_val in np.linspace(0, np.pi, 3, endpoint=False)[1:]:  # avoid M again
        q_list.append(np.array([2 * np.pi / np.sqrt(3), qy_val, 0]))
    return np.array(q_list)


if __name__ == "__main__":
    st_main = default_timer()
    logger.info("--- Starting KFe3J Configuration Test ---")

    # --- Parameters (must match kfe3j_config.yaml and original model expectations) ---
    S_val = 2.5
    # Order for original model: J1, J2, Dy, Dz, H
    params_val_original_model = [3.23, 0.11, 0.218, -0.195, 0.0]

    # Use declarative config as it works with GenericSpinModel
    config_file_path = os.path.join(project_root_dir, "examples", "KFe3J", "KFe3J_declarative.yaml")

    q_vectors_test = generate_q_vectors_for_test()
    logger.info(f"Testing with {len(q_vectors_test)} q-points.")

    energies_config_driven = None
    energies_hardcoded_model = None
    matrices_config_driven = None  # For intermediate results
    matrices_hardcoded_model = None  # For intermediate results

    calculator_config: mc.MagCalc = None
    calculator_hardcoded: mc.MagCalc = None

    HMat_sym_config: sp_Matrix = None
    Ud_sym_config: sp_Matrix = None
    HMat_sym_hardcoded: sp_Matrix = None
    Ud_sym_hardcoded: sp_Matrix = None

    # --- 1. Run with new configuration-driven model ---
    logger.info("Calculating dispersion using kfe3j_config.yaml...")
    try:
        calculator_config = mc.MagCalc(  # type: ignore
            config_filepath=config_file_path,
            # spin_magnitude, hamiltonian_params are taken from config
            cache_file_base="kfe3j_test_cache_config",
            cache_mode="w",  # Force regeneration for test
        )
        energies_config_driven = calculator_config.calculate_dispersion(q_vectors_test)
        # Attempt to get intermediate matrices (assuming attribute name)
        matrices_config_driven = getattr(
            calculator_config, "_intermediate_numerical_H_matrices_disp", None
        )
        if matrices_config_driven is not None:
            logger.info(
                "Intermediate matrices retrieved from config-driven calculator."
            )
        if energies_config_driven:
            logger.info("Dispersion calculation with config SUCCEEDED.")
        else:
            logger.error("Dispersion calculation with config FAILED (returned None).")

        HMat_sym_config = calculator_config.HMat_sym
        Ud_sym_config = calculator_config.Ud_sym
    except Exception as e:
        logger.error(
            f"Error initializing or running MagCalc with config: {e}", exc_info=True
        )

    # --- 2. Run with original hardcoded spin_model.py ---
    logger.info("Calculating dispersion using original KFe3J.spin_model.py...")
    try:
        calculator_hardcoded = mc.MagCalc(  # type: ignore
            spin_magnitude=S_val,
            hamiltonian_params=params_val_original_model,
            spin_model_module=kfe3j_hardcoded_spin_model,
            cache_file_base="kfe3j_test_cache_hardcoded",
            cache_mode="w",  # Force regeneration for test
        )
        energies_hardcoded_model = calculator_hardcoded.calculate_dispersion(
            q_vectors_test
        )
        # Attempt to get intermediate matrices (assuming attribute name)
        matrices_hardcoded_model = getattr(
            calculator_hardcoded, "_intermediate_numerical_H_matrices_disp", None
        )
        if matrices_hardcoded_model is not None:
            logger.info(
                "Intermediate matrices retrieved from hardcoded model calculator."
            )
        if energies_hardcoded_model:
            logger.info("Dispersion calculation with hardcoded model SUCCEEDED.")
        else:
            logger.error(
                "Dispersion calculation with hardcoded model FAILED (returned None)."
            )
        HMat_sym_hardcoded = calculator_hardcoded.HMat_sym
        Ud_sym_hardcoded = calculator_hardcoded.Ud_sym
    except Exception as e:
        logger.error(
            f"Error initializing or running MagCalc with hardcoded model: {e}",
            exc_info=True,
        )

    # --- 3. Compare Symbolic Matrices (Ud_sym and HMat_sym) ---
    logger.info("Comparing symbolic matrices (Ud_sym and HMat_sym)...")
    symbolic_matrices_match = True
    if (
        Ud_sym_config is not None
        and Ud_sym_hardcoded is not None
        and HMat_sym_config is not None
        and HMat_sym_hardcoded is not None
    ):

        # --- 3a. Compare Ud_sym ---
        logger.info("Comparing Ud_sym...")
        try:
            # Create symbol mapping: hardcoded (p0,p1,..) to config (J1,J2,..)
            # This assumes the order in kfe3j_config.yaml 'parameters' matches
            # the order of params_val_original_model.
            params_sym_hardcoded = calculator_hardcoded.params_sym
            params_sym_config = calculator_config.params_sym

            if len(params_sym_hardcoded) == len(params_sym_config):
                symbol_map_hard_to_config = {
                    ps_hard: ps_conf
                    for ps_hard, ps_conf in zip(params_sym_hardcoded, params_sym_config)
                }
                # Also map S_sym if they are different objects (though usually named 'S')
                if calculator_hardcoded.S_sym != calculator_config.S_sym:
                    symbol_map_hard_to_config[calculator_hardcoded.S_sym] = (
                        calculator_config.S_sym
                    )

                Ud_sym_hardcoded_mapped = Ud_sym_hardcoded.subs(
                    symbol_map_hard_to_config
                )

                if Ud_sym_hardcoded_mapped == Ud_sym_config:
                    logger.info(
                        "SUCCESS: Symbolic Ud_sym matrices are identical after symbol mapping."
                    )
                else:
                    logger.warning(
                        "Symbolic Ud_sym matrices differ even after symbol mapping."
                    )
                    # For debugging, one might print them, but they can be large.
                    # logger.debug(f"Ud_sym_config:\n{Ud_sym_config}")
                    # logger.debug(f"Ud_sym_hardcoded_mapped:\n{Ud_sym_hardcoded_mapped}")
                    symbolic_matrices_match = False
            else:
                logger.error(
                    "Cannot map Ud_sym symbols due to different number of parameters."
                )
                symbolic_matrices_match = False

            # Numerical check for Ud_sym (using their respective numerical params, which should be identical for this test)
            # calculator_config.hamiltonian_params are from the YAML
            # params_val_original_model are for the hardcoded
            # For this test, these two lists of values *must* be the same.
            if not np.allclose(
                np.array(calculator_config.hamiltonian_params),
                np.array(params_val_original_model),
            ):
                logger.warning(
                    "Numerical parameters for config and hardcoded model differ. Ud_sym numerical check might be misleading."
                )

            Ud_num_conf = np.array(
                Ud_sym_config.subs(
                    [(calculator_config.S_sym, calculator_config.spin_magnitude)]
                    + list(
                        zip(
                            calculator_config.params_sym,
                            calculator_config.hamiltonian_params,
                        )
                    )
                ).evalf(),
                dtype=np.complex128,
            )

            Ud_num_hard = np.array(
                Ud_sym_hardcoded.subs(
                    [(calculator_hardcoded.S_sym, S_val)]
                    + list(
                        zip(calculator_hardcoded.params_sym, params_val_original_model)
                    )
                ).evalf(),
                dtype=np.complex128,
            )

            if np.allclose(Ud_num_conf, Ud_num_hard, atol=1e-7):
                logger.info("SUCCESS: Numerically evaluated Ud_sym matrices match.")
            else:
                logger.error(
                    "FAILURE: Numerically evaluated Ud_sym matrices DO NOT match."
                )
                logger.error(
                    f"Max diff Ud_sym: {np.max(np.abs(Ud_num_conf - Ud_num_hard))}"
                )
                symbolic_matrices_match = False
        except Exception as e:
            logger.error(f"Error during Ud_sym comparison: {e}", exc_info=True)
            symbolic_matrices_match = False

        # --- 3b. Compare HMat_sym (numerically at test q-points) ---
        logger.info("Comparing HMat_sym numerically at test q-points...")
        all_hmat_num_match = True
        try:
            for q_idx, q_pt_val in enumerate(q_vectors_test):
                subs_conf = (
                    [(calculator_config.S_sym, calculator_config.spin_magnitude)]
                    + list(zip(calculator_config.k_sym, q_pt_val))
                    + list(
                        zip(
                            calculator_config.params_sym,
                            calculator_config.hamiltonian_params,
                        )
                    )
                )
                # Convert subs_conf list of tuples to a dictionary for evalf's subs argument
                subs_conf_dict = dict(subs_conf)
                HMat_evalf = HMat_sym_config.evalf(subs=subs_conf_dict)

                # DEBUG: Check for remaining free symbols
                remaining_symbols = HMat_evalf.free_symbols
                if remaining_symbols:
                    logger.error(
                        f"HMat_sym_config still has free symbols after evalf(subs=...) at q={q_pt_val}: {remaining_symbols}"
                    )
                    # Optionally, log the specific elements with remaining symbols
                    # for r_idx in range(HMat_evalf.rows):
                    #     for c_idx in range(HMat_evalf.cols):
                    #         if HMat_evalf[r_idx, c_idx].free_symbols:
                    #             logger.error(f"  Element ({r_idx},{c_idx}) '{HMat_evalf[r_idx, c_idx]}' has symbols: {HMat_evalf[r_idx, c_idx].free_symbols}")

                HMat_num_conf = np.array(HMat_evalf, dtype=np.complex128)

                subs_hard = (
                    [(calculator_hardcoded.S_sym, S_val)]
                    + list(zip(calculator_hardcoded.k_sym, q_pt_val))
                    + list(
                        zip(calculator_hardcoded.params_sym, params_val_original_model)
                    )
                )
                HMat_num_hard = np.array(
                    HMat_sym_hardcoded.subs(subs_hard).evalf(), dtype=np.complex128
                )

                if not np.allclose(HMat_num_conf, HMat_num_hard, atol=1e-7):
                    logger.warning(
                        f"Numerically evaluated HMat_sym mismatch at q = {q_pt_val}:"
                    )
                    logger.warning(
                        f"  Max absolute difference: {np.max(np.abs(HMat_num_conf - HMat_num_hard))}"
                    )
                    all_hmat_num_match = False
                    symbolic_matrices_match = False
                    break  # Stop after first mismatch for HMat_sym
            if all_hmat_num_match:
                logger.info(
                    "SUCCESS: Numerically evaluated HMat_sym matrices match for all test q-points."
                )
        except Exception as e:
            logger.error(
                f"Error during HMat_sym numerical comparison: {e}", exc_info=True
            )
            symbolic_matrices_match = False
            all_hmat_num_match = False  # Ensure this flag is false on error

    # --- 4. Compare final dispersion results ---
    if energies_config_driven is not None and energies_hardcoded_model is not None:
        logger.info("Comparing dispersion results...")
        if len(energies_config_driven) != len(energies_hardcoded_model):
            logger.error(
                f"Mismatch in number of q-point results: Config ({len(energies_config_driven)}) vs Hardcoded ({len(energies_hardcoded_model)})"
            )
        else:
            all_match = True
            for i in range(len(q_vectors_test)):
                q_pt = q_vectors_test[i]
                e_conf = (
                    np.array(energies_config_driven[i])
                    if energies_config_driven[i] is not None
                    else np.array([np.nan])
                )
                e_hard = (
                    np.array(energies_hardcoded_model[i])
                    if energies_hardcoded_model[i] is not None
                    else np.array([np.nan])
                )

                if not np.allclose(
                    e_conf, e_hard, atol=1e-5, equal_nan=True
                ):  # Increased tolerance slightly
                    logger.warning(f"Mismatch at q = {q_pt}:")
                    logger.warning(f"  Config:    {e_conf}")
                    logger.warning(f"  Hardcoded: {e_hard}")
                    all_match = False

            if all_match:
                logger.info(
                    "SUCCESS: Dispersion results match between config-driven and hardcoded models."
                )
            else:
                logger.error(
                    "FAILURE: Dispersion results DO NOT match. Check warnings above."
                )
    else:
        logger.error("Cannot compare results as one or both calculations failed.")

    # --- 5. Compare Intermediate Numerical Hamiltonian Matrices (from dispersion calculation) ---
    logger.info("Comparing intermediate matrices (e.g., Spin Wave Hamiltonians)...")
    if matrices_config_driven is not None and matrices_hardcoded_model is not None:
        if len(matrices_config_driven) != len(matrices_hardcoded_model):
            logger.error(
                f"Mismatch in number of intermediate matrices: Config ({len(matrices_config_driven)}) vs Hardcoded ({len(matrices_hardcoded_model)})"
            )
        elif len(matrices_config_driven) != len(q_vectors_test):
            logger.error(
                f"Mismatch in number of intermediate matrices ({len(matrices_config_driven)}) and q-points ({len(q_vectors_test)})"
            )
        else:
            all_matrices_match = True
            for i in range(len(q_vectors_test)):
                q_pt = q_vectors_test[i]
                m_conf = matrices_config_driven[i]
                m_hard = matrices_hardcoded_model[i]

                if m_conf is None or m_hard is None:
                    if m_conf is None and m_hard is None:
                        logger.info(
                            f"Both intermediate matrices are None at q = {q_pt}, skipping comparison for this q-point."
                        )
                        continue
                    else:
                        logger.warning(
                            f"One intermediate matrix is None at q = {q_pt}:"
                        )
                        logger.warning(f"  Config Matrix is None: {m_conf is None}")
                        logger.warning(f"  Hardcoded Matrix is None: {m_hard is None}")
                        all_matrices_match = False
                        continue

                if not np.allclose(m_conf, m_hard, atol=1e-5, equal_nan=True):
                    logger.warning(f"Intermediate matrices mismatch at q = {q_pt}:")
                    # Optionally log the matrices, but they can be large.
                    # logger.warning(f"  Config Matrix:\n{m_conf}")
                    # logger.warning(f"  Hardcoded Matrix:\n{m_hard}")
                    logger.warning(
                        f"  Shape Config: {m_conf.shape}, Shape Hardcoded: {m_hard.shape}"
                    )
                    logger.warning(
                        f"  Max absolute difference: {np.max(np.abs(m_conf - m_hard))}"
                    )
                    all_matrices_match = False
            if all_matrices_match:
                logger.info(
                    "SUCCESS: Intermediate matrices match between config-driven and hardcoded models."
                )
            else:
                logger.error(
                    "FAILURE: Intermediate matrices DO NOT match. Check warnings above."
                )
    elif (
        energies_config_driven is not None and energies_hardcoded_model is not None
    ):  # Only log if calculations themselves succeeded
        logger.warning(
            "Could not compare intermediate matrices: one or both sets of matrices were not retrieved."
        )
        logger.warning(
            f"  Matrices from config-driven calculator available: {matrices_config_driven is not None}"
        )
        logger.warning(
            f"  Matrices from hardcoded calculator available: {matrices_hardcoded_model is not None}"
        )
        logger.warning(
            "  (This might indicate the attribute '_intermediate_numerical_H_matrices_disp' is not set by MagCalc instances, or the dispersion calculation failed before this point)."
        )

    et_main = default_timer()
    logger.info(f"Total test run-time: {np.round((et_main - st_main), 2)} sec.")
    logger.info("--- KFe3J Configuration Test Finished ---")
