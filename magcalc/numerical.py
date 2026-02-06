import sys
import logging
import numpy as np
import scipy.linalg as la
import sympy as sp
from sympy import lambdify
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy.typing as npt

# Internal imports
try:
    from .linalg import (
        KKdMatrix,
        # Import any other linalg functions if strictly needed by numerical.py logic 
        # (none observed in process_calc_disp/sqw bodies)
    )
except ImportError:
    # Fallback if relative import fails (e.g. running script directly)
    from linalg import KKdMatrix

from .form_factors import get_form_factor

logger = logging.getLogger(__name__)

# --- Numerical Constants ---
ENERGY_IMAG_PART_THRESHOLD: float = 1e-5
SQW_IMAG_PART_THRESHOLD: float = 1e-4
Q_ZERO_THRESHOLD: float = 1e-10
PROJECTION_CHECK_TOLERANCE: float = 1e-5

@dataclass
class DispersionResult:
    """Result of a spin-wave dispersion calculation."""
    q_vectors: npt.NDArray[np.float64]
    energies: npt.NDArray[np.float64]

@dataclass
class SqwResult:
    """Result of a dynamic structure factor calculation."""
    q_vectors: npt.NDArray[np.float64]
    energies: npt.NDArray[np.float64]
    intensities: npt.NDArray[np.float64]

# --- Global variable for worker processes ---
_worker_HMat_func = None

def _init_worker(HMat_sym, full_symbol_list):
    """
    Initializer function for multiprocessing worker.
    Lambdifies the symbolic Hamiltonian once per process.
    """
    global _worker_HMat_func
    
    try:
        # We need to import numpy inside the worker if used in lambdify modules
        import numpy as np 
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        _worker_HMat_func = lambdify(full_symbol_list, HMat_sym, modules=["numpy"], cse=True)
    except Exception as e:
        sys.stderr.write(f"Error in worker initialization: {e}\n")
        raise e

def substitute_expr(
    args: Tuple[sp.Expr, Union[Dict, List[Tuple[sp.Expr, sp.Expr]]]],
) -> sp.Expr:
    """
    Perform symbolic substitution on a SymPy expression.
    """
    expr, subs_dict = args
    result: sp.Expr = expr.subs(subs_dict, simultaneous=True)
    return result

def process_calc_disp(
    args: Tuple[
        npt.NDArray[np.float64],
        int,
        float,
        Union[List[float], npt.NDArray[np.float64]],
    ],
) -> Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.complex128]]]:
    """
    Worker function for parallel dispersion calculation at a single q-point.
    Uses pre-initialized _worker_HMat_func.
    """
    (
        _,  # Ud_numeric (unused by HMat_func as it's baked into params/sym)
        q_vector,
        nspins,
        spin_magnitude_num,
        hamiltonian_params_num,
    ) = args
    
    global _worker_HMat_func
    if _worker_HMat_func is None:
        raise RuntimeError("Worker not initialized with HMat_func")
        
    q_label = f"q={q_vector}"
    nan_energies: npt.NDArray[np.float64] = np.full((nspins,), np.nan)
    HMat_numeric: Optional[npt.NDArray[np.complex128]] = None
    
    try:
        numerical_args = (
            list(q_vector) + [spin_magnitude_num] + list(hamiltonian_params_num)
        )
        HMat_numeric = np.array(
            _worker_HMat_func(*numerical_args), dtype=np.complex128
        )
    except Exception:
        logger.exception(f"Error evaluating HMat function at {q_label}.")
        return nan_energies, None
    
    try:
        eigenvalues = la.eigvals(HMat_numeric)
    except np.linalg.LinAlgError:
        logger.error(f"Eigenvalue calculation failed for {q_label}.")
        return (nan_energies, HMat_numeric)
    except Exception:
        logger.exception(f"Unexpected error during eigenvalue calculation for {q_label}.")
        return (nan_energies, HMat_numeric)
        
    try:
        imag_part_mags = np.abs(np.imag(eigenvalues))
        if np.any(imag_part_mags > ENERGY_IMAG_PART_THRESHOLD):
            logger.warning(
                f"Significant imaginary part in eigenvalues for {q_label}. Max imag: {np.max(imag_part_mags)}"
            )
        eigenvalues_sorted_real = np.real(np.sort(eigenvalues))
        energies = eigenvalues_sorted_real[nspins:]
        
        if len(energies) != nspins:
            logger.warning(
                f"Unexpected number of positive energies ({len(energies)}) found for {q_label}. Expected {nspins}."
            )
            if len(energies) > nspins:
                energies = energies[:nspins]
            else:
                energies = np.pad(
                    energies, (0, nspins - len(energies)), constant_values=np.nan
                )
        return energies, HMat_numeric
    except Exception:
        logger.exception(f"Error during eigenvalue sorting/selection for {q_label}.")
        return nan_energies, HMat_numeric

def process_calc_Sqw(
    args: Tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.float64],
        int,
        float,
        Union[List[float], npt.NDArray[np.float64]],
        Optional[List[str]], # ion_list
    ],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Worker function for parallel S(q,w) calculation at a single q-point.
    Uses pre-initialized _worker_HMat_func.
    """
    (
        Ud_numeric,
        q_vector,
        nspins,
        spin_magnitude_num,
        hamiltonian_params_num,
        ion_list,
    ) = args
    
    global _worker_HMat_func
    if _worker_HMat_func is None:
        raise RuntimeError("Worker not initialized with HMat_func")
        
    q_label = f"q={q_vector}"
    nan_energies = np.full((nspins,), np.nan)
    nan_intensities = np.full((nspins,), np.nan)
    nan_result = (q_vector, nan_energies, nan_intensities)
    
    try:
        numerical_args_base = [spin_magnitude_num] + list(hamiltonian_params_num)
        numerical_args_plus_q = list(q_vector) + numerical_args_base
        numerical_args_minus_q = list(-q_vector) + numerical_args_base
        
        Hmat_plus_q = np.array(
            _worker_HMat_func(*numerical_args_plus_q), dtype=np.complex128
        )
        Hmat_minus_q = np.array(
            _worker_HMat_func(*numerical_args_minus_q), dtype=np.complex128
        )
    except Exception:
        logger.exception(f"Error evaluating HMat function at {q_label}.")
        return nan_result
        
    try:
        K_matrix, Kd_matrix, eigenvalues = KKdMatrix(
            spin_magnitude_num, Hmat_plus_q, Hmat_minus_q, Ud_numeric, q_vector, nspins
        )
        if (
            np.isnan(K_matrix).any()
            or np.isnan(Kd_matrix).any()
            or np.isnan(eigenvalues).any()
        ):
            logger.error(f"NaN encountered in KKdMatrix result for {q_label}.")
            return nan_result
    except Exception:
        logger.exception(f"Unexpected error during KKdMatrix execution for {q_label}.")
        return nan_result
        
    try:
        imag_energy_mag = np.abs(np.imag(eigenvalues[0:nspins]))
        if np.any(imag_energy_mag > ENERGY_IMAG_PART_THRESHOLD):
            logger.warning(
                f"Significant imaginary part in energy eigenvalues for {q_label}. Max imag: {np.max(imag_energy_mag)}"
            )
        energies = np.real(eigenvalues[0:nspins])
        sqw_complex_accumulator = np.zeros(nspins, dtype=complex)
        
        q_norm_sq = np.dot(q_vector, q_vector)
        q_mag = np.sqrt(q_norm_sq)
        
        # Pre-calculate form factors for all spins at this Q
        ff_values = np.ones(nspins)
        if ion_list:
            for i in range(nspins):
                ff_values[i] = get_form_factor(ion_list[i], q_mag)

        for mode_index in range(nspins):
            spin_correlation_matrix = np.zeros((3, 3), dtype=complex)
            intensity_one_mode = 0.0 + 0.0j
            for alpha in range(3):
                for beta in range(3):
                    correlation_sum = 0.0 + 0.0j
                    for spin_i in range(nspins):
                        for spin_j in range(nspins):
                            idx_K = 3 * spin_i + alpha
                            idx_Kd = 3 * spin_j + beta
                            correlation_sum += (
                                ff_values[spin_i] * ff_values[spin_j] *
                                K_matrix[idx_K, mode_index]
                                * Kd_matrix[idx_Kd, mode_index + nspins]
                            )
                    spin_correlation_matrix[alpha, beta] = correlation_sum
            
            if q_norm_sq < Q_ZERO_THRESHOLD:
                for alpha in range(3):
                    intensity_one_mode += spin_correlation_matrix[alpha, alpha]
            else:
                q_normalized = q_vector / np.sqrt(q_norm_sq)
                for alpha in range(3):
                    for beta in range(3):
                        delta_ab = 1.0 if alpha == beta else 0.0
                        polarization_factor = (
                            delta_ab - q_normalized[alpha] * q_normalized[beta]
                        )
                        intensity_one_mode += (
                            polarization_factor * spin_correlation_matrix[alpha, beta]
                        )
            
            if np.abs(np.imag(intensity_one_mode)) > SQW_IMAG_PART_THRESHOLD:
                logger.warning(
                    f"Significant imaginary part in Sqw for {q_label}, mode {mode_index}: {np.imag(intensity_one_mode)}"
                )
            sqw_complex_accumulator[mode_index] = intensity_one_mode
            
        intensities = np.real(sqw_complex_accumulator)
        intensities[intensities < 0] = 0
        return q_vector, energies, intensities
    except Exception:
        logger.exception(f"Error during intensity calculation for {q_label}.")
        return nan_result
