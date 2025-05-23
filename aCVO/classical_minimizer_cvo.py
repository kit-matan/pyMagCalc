#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical energy minimization for alpha-Cu2V2O7 spin model.

This script defines the classical energy function for the CVO spin model
and uses scipy.optimize to find the spin orientations (angles) that minimize
this energy for a given set of Hamiltonian parameters and applied field.

The minimized angles represent the classical ground state configuration,
which is required for a valid Linear Spin Wave Theory calculation.

@author: Kit Matan
@contributor: AI Assistant
"""
import numpy as np
from scipy.optimize import minimize
import logging
import sympy as sp  # Added for sp.Matrix check if needed, though DM should be numerical

# --- Basic Logging Setup ---
# Configure logging early, before any custom log messages are emitted.
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG to see more details during minimization
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)  # Get the logger for this module
# --- End Logging Setup ---


# Import the spin model module to get structure and interaction info
# We need functions like atom_pos, atom_pos_ouc, spin_interactions, and the al list
cvo_model = None  # Initialize cvo_model
AL_SPIN_PREFERENCE = None  # Initialize AL_SPIN_PREFERENCE
NSPINS_MODEL = 16  # Default for CVO model
try:
    import spin_model_hc as cvo_model  # Use the active spin_model_hc

    if hasattr(cvo_model, "AL_SPIN_PREFERENCE"):
        AL_SPIN_PREFERENCE = cvo_model.AL_SPIN_PREFERENCE
        NSPINS_MODEL = len(AL_SPIN_PREFERENCE)
    else:
        logging.error(
            "Failed to access 'AL_SPIN_PREFERENCE' list from 'spin_model_hc.py'."
        )
except ImportError:
    logging.error(
        "Failed to import 'spin_model_hc.py'. "
        "Please ensure this file exists and is in the Python path."
    )


def classical_energy(theta_angles, params, S, spin_model_module):
    """
    Calculates the classical energy of the spin system for given spin orientations.

    Assumes spins are primarily in the global x-z plane with azimuthal angle
    fixed at 0 or pi based on the spin's +/- a-axis preference (defined by AL_SPIN_PREFERENCE).
    Minimization is performed over the polar angles (theta).

    Args:
        theta_angles (np.ndarray): 1D array of 16 polar angles (in radians).
        params (list): List of numerical Hamiltonian parameters [J1, J2, J3, G1, Dx, H].
        S (float): Numerical spin magnitude.
        spin_model_module: The imported spin model module (e.g., spin_model_cvo_Hc).

    Returns:
        float: The total classical energy in meV. Returns np.inf if inputs are invalid.
    """
    if spin_model_module is None or AL_SPIN_PREFERENCE is None:
        logger.error(
            "Spin model or AL_SPIN_PREFERENCE not loaded. Cannot calculate classical energy."
        )
        return np.inf

    if len(theta_angles) != NSPINS_MODEL:
        logger.error(f"Expected {NSPINS_MODEL} theta angles, got {len(theta_angles)}")
        return np.inf

    try:
        # Get structure and interaction info from the spin model
        apos = spin_model_module.atom_pos()
        nspin = len(apos)  # Should be 16
        apos_ouc = spin_model_module.atom_pos_ouc()
        nspin_ouc = len(apos_ouc)

        # spin_interactions in spin_model_hc.py expects symbolic params to return symbolic DM.
        # If passed numerical params, Dx becomes numerical, and DM vectors become numerical.
        # This is suitable for classical energy calculation.
        # params = [J1, J2, J3, G1, Dx_num, H_field_num]
        Jex, Gex, DM_matrix_of_matrices, H_field = spin_model_module.spin_interactions(
            params
        )
        # H_field from spin_interactions is the symbolic H if symbolic p was passed.
        # Here, params is numerical, so H_field is numerical H_field_num.
        H_field_num_from_params = params[
            5
        ]  # Explicitly use H from input params for Zeeman

        # Convert the SymPy Matrix of SymPy 1x3 Matrices for DM into a NumPy array of 3D vectors
        DM_numerical_vectors = np.zeros((nspin, nspin_ouc, 3), dtype=float)
        for i in range(nspin):
            for j in range(nspin_ouc):
                dm_vec_sym_matrix = DM_matrix_of_matrices[
                    i, j
                ]  # This is a sympy.Matrix(1,3)
                if not isinstance(
                    dm_vec_sym_matrix, sp.MatrixBase
                ):  # Check if it's a SymPy Matrix
                    logger.error(
                        f"DM element ({i},{j}) is not a SymPy Matrix: {type(dm_vec_sym_matrix)}. Value: {dm_vec_sym_matrix}"
                    )
                    # Assuming it might be a numerical array or list if not SymPy Matrix
                    DM_numerical_vectors[i, j, :] = np.array(
                        dm_vec_sym_matrix, dtype=float
                    ).flatten()
                    continue
                try:
                    # Convert each component of the 1x3 SymPy Matrix to float
                    dm_vec_num = np.array(
                        [float(comp.evalf()) for comp in dm_vec_sym_matrix]
                    )
                    DM_numerical_vectors[i, j, :] = dm_vec_num
                except Exception as e:
                    logger.error(
                        f"Failed to numerically evaluate DM vector for pair ({i},{j}): {dm_vec_sym_matrix}. Error: {e}"
                    )
                    return np.inf  # Cannot proceed if DM evaluation fails

        # --- Construct classical spin vectors ---
        classical_spin_vectors = np.zeros((nspin, 3), dtype=float)
        for i in range(nspin):
            theta_i = theta_angles[i]
            phi_i = 0.0 if AL_SPIN_PREFERENCE[i] == 1 else np.pi

            classical_spin_vectors[i, 0] = S * np.sin(theta_i) * np.cos(phi_i)  # Sx
            classical_spin_vectors[i, 1] = (
                S * np.sin(theta_i) * np.sin(phi_i)
            )  # Sy (should be 0)
            classical_spin_vectors[i, 2] = S * np.cos(theta_i)  # Sz

        # --- Calculate total classical energy ---
        total_energy = 0.0

        for i in range(nspin):  # Spin i in the unit cell
            S_vec_i = classical_spin_vectors[i, :]

            for j in range(nspin_ouc):  # Spin j in the OUC
                j_uc = j % nspin  # Equivalent spin in UC for orientation
                S_vec_j = classical_spin_vectors[j_uc, :]

                J_ij = Jex[i, j]
                G_ij = Gex[i, j]
                DM_ij = DM_numerical_vectors[i, j, :]

                if (
                    J_ij != 0 or G_ij != 0 or np.any(DM_ij != 0)
                ):  # Process only if there's an interaction
                    # Heisenberg
                    total_energy += 0.5 * J_ij * np.dot(S_vec_i, S_vec_j)

                    # DM interaction
                    cross_prod_ij = np.cross(S_vec_i, S_vec_j)
                    total_energy += 0.5 * np.dot(DM_ij, cross_prod_ij)

                    # Anisotropic exchange
                    total_energy += (
                        0.5
                        * G_ij
                        * (
                            S_vec_i[0] * S_vec_j[0]
                            - S_vec_i[1] * S_vec_j[1]
                            - S_vec_i[2] * S_vec_j[2]
                        )
                    )

        # Zeeman term
        # Field is applied along the global x-axis.
        gamma = 2.0
        mu = 5.7883818066e-2
        # H_field_strength = params[5] # This is H_field_num_from_params

        zeeman_energy_contribution = 0.0
        for i in range(nspin):
            S_vec_i = classical_spin_vectors[i, :]
            # Zeeman term: - S . H. If H is along +x, then - Sx * H_magnitude
            zeeman_energy_contribution -= (
                gamma * mu * S_vec_i[2] * H_field_num_from_params
            )

        if abs(H_field_num_from_params) > 1e-6:
            logger.debug(
                f"H_field={H_field_num_from_params}, Zeeman E: {zeeman_energy_contribution:.4f}, Sum Sx/S: {np.sum(classical_spin_vectors[:,0])/S:.4f}"
            )
        total_energy += zeeman_energy_contribution
        return total_energy

    except Exception as e:
        logger.error(f"Error calculating classical energy: {e}", exc_info=True)
        return np.inf


def find_ground_state_orientations(params, S, spin_model_module):
    """
    Finds the classical ground state spin orientations (theta angles)
    by minimizing the classical energy.

    Args:
        params (list): List of numerical Hamiltonian parameters [J1, J2, J3, G1, Dx, H].
        S (float): Numerical spin magnitude.
        spin_model_module: The imported spin model module (e.g., spin_model_cvo_Hc).

    Returns:
        np.ndarray: 1D array of 16 optimal polar angles (theta) in radians.
                    Returns None if minimization fails.
    """
    if spin_model_module is None or AL_SPIN_PREFERENCE is None:
        logger.error(
            "Spin model or AL_SPIN_PREFERENCE not loaded. Cannot find ground state."
        )
        return None

    H_field_val = params[5]
    logger.info(f"Starting classical energy minimization for H={H_field_val}...")

    initial_theta_guess = np.full(
        NSPINS_MODEL, np.pi / 2.0 - (0.05 if H_field_val != 0 else 0.0)
    )  # Perturb if H is non-zero
    bounds = [(1e-9, np.pi - 1e-9)] * NSPINS_MODEL

    method = "L-BFGS-B"

    try:
        result = minimize(
            classical_energy,
            initial_theta_guess,
            args=(params, S, spin_model_module),
            method=method,
            bounds=bounds,
            tol=1e-6,
            options={  # Tighter tolerances might be needed
                "maxiter": 5000,  # Increased maxiter
                "ftol": 1e-9,  # Tighter function tolerance
                "gtol": 1e-7,  # Tighter gradient tolerance
            },  # More options for robustness
        )

        if result.success:
            logger.info(
                f"Classical minimization successful. Final energy: {result.fun:.6f} meV. Iterations: {result.nit}."
            )
            optimal_theta_angles = np.clip(result.x, 0, np.pi)

            # --- Log Canting Angles ---
            optimal_theta_degrees = np.degrees(optimal_theta_angles)
            canting_angles_degrees = 90.0 - optimal_theta_degrees
            logger.info(f"Optimal theta angles (degrees): {optimal_theta_degrees}")
            logger.info(
                f"Canting angles (90 - theta, degrees): {canting_angles_degrees}"
            )
            logger.info(
                f"Mean canting angle (degrees): {np.mean(canting_angles_degrees):.4f}"
            )
            # --- End Log Canting Angles ---
            return optimal_theta_angles
        else:
            logger.error(f"Classical minimization failed: {result.message}")
            logger.error(f"Final function value: {result.fun}")
            logger.error(f"Number of iterations: {result.nit}")
            logger.error(
                f"Jacobian norm: {np.linalg.norm(result.jac) if hasattr(result, 'jac') else 'N/A'}"
            )
            return None

    except Exception as e:
        logger.error(
            f"An error occurred during classical minimization: {e}", exc_info=True
        )
        return None


# Example usage (optional, for testing the minimizer standalone)
if __name__ == "__main__":
    if cvo_model is None or AL_SPIN_PREFERENCE is None:
        logger.error("Cannot run example: CVO model or AL_SPIN_PREFERENCE not loaded.")
    else:
        logger.info("Running standalone test for classical_minimizer_cvo.py...")
        # Define example parameters (match your config_cvo.yaml)
        example_params_h0 = [2.49, 1.12 * 2.49, 2.03 * 2.49, 0.28, 2.67, 0.0]  # H=0
        example_S_val = 0.5

        logger.info(f"\n--- Test Case: H=0 ---")
        optimal_angles_h0 = find_ground_state_orientations(
            example_params_h0, example_S_val, cvo_model
        )
        if optimal_angles_h0 is not None:
            logger.info("Optimal theta angles at H=0 (degrees):")
            logger.info(np.degrees(optimal_angles_h0))
            # For H=0, if the ground state is spins in a-b plane, angles should be ~90 deg.
            logger.info(
                f"Mean canting angle at H=0 (degrees): {np.mean(90.0 - np.degrees(optimal_angles_h0)):.6f}"
            )

        # Define example parameters with H field
        example_params_h14 = [
            2.49,
            1.12 * 2.49,
            2.03 * 2.49,
            0.28,
            2.67,
            14.0,
        ]  # H=14.0
        logger.info(f"\n--- Test Case: H={example_params_h14[5]} ---")
        optimal_angles_h14 = find_ground_state_orientations(
            example_params_h14, example_S_val, cvo_model
        )
        if optimal_angles_h14 is not None:
            logger.info(f"Optimal theta angles at H={example_params_h14[5]} (degrees):")
            logger.info(np.degrees(optimal_angles_h14))
            canting_degrees_h14 = 90.0 - np.degrees(optimal_angles_h14)
            logger.info(f"Canting towards +z (degrees, 90 - theta):")
            logger.info(canting_degrees_h14)
            logger.info(f"Mean canting (degrees): {np.mean(canting_degrees_h14):.6f}")
