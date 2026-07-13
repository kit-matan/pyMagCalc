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
    """Result of a spin-wave dispersion calculation.

    For a single-k structure with satellites enabled, energies has 3*nspins
    columns ordered channel-major: [omega(q-k) | omega(q) | omega(q+k)].
    """
    q_vectors: npt.NDArray[np.float64]
    energies: npt.NDArray[np.float64]
    branch_labels: Optional[List[str]] = None

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

        # Vectorized spin correlation matrix.
        # K is (3*N, 2*N); reshape to (N, 3, 2*N) so axis 0 indexes spins and
        # axis 1 indexes Cartesian components. Multiply by form factors and
        # restrict mode axis to magnon modes (mode_index in [0, N) for K and
        # [N, 2N) for Kd, matching the original loop).
        K_modes = K_matrix[:, :nspins].reshape(nspins, 3, nspins)
        Kd_modes = Kd_matrix[:, nspins:].reshape(nspins, 3, nspins)
        K_w = K_modes * ff_values[:, None, None]
        Kd_w = Kd_modes * ff_values[:, None, None]
        # spin_corr[alpha, beta, mode] = sum_{i,j} K_w[i, alpha, mode] * Kd_w[j, beta, mode]
        spin_corr = np.einsum("iam,jbm->abm", K_w, Kd_w)

        if q_norm_sq < Q_ZERO_THRESHOLD:
            intensity_per_mode = np.einsum("aam->m", spin_corr)
        else:
            q_normalized = q_vector / np.sqrt(q_norm_sq)
            polarization = np.eye(3) - np.outer(q_normalized, q_normalized)
            intensity_per_mode = np.einsum("ab,abm->m", polarization, spin_corr)

        max_imag = float(np.max(np.abs(np.imag(intensity_per_mode)))) if nspins > 0 else 0.0
        if max_imag > SQW_IMAG_PART_THRESHOLD:
            worst_mode = int(np.argmax(np.abs(np.imag(intensity_per_mode))))
            logger.warning(
                f"Significant imaginary part in Sqw for {q_label}, mode {worst_mode}: {np.imag(intensity_per_mode[worst_mode])}"
            )

        intensities = np.real(intensity_per_mode)
        intensities[intensities < 0] = 0
        return q_vector, energies, intensities
    except Exception:
        logger.exception(f"Error during intensity calculation for {q_label}.")
        return nan_result


def spiral_channel_tensors(
    n_axis: npt.NDArray[np.float64], k_case: int
) -> List[Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]]:
    """Left/right rotating-frame projection tensors for the three spiral channels.

    Convention pinned to Sunny SpinWaveTheorySpiral.jl (intensities_bands) /
    SpinW spinwave.m (Toth & Lake 2015): with n the rotation axis,
    R1 = (I - i*[n]x - n n^T)/2 and R2 = n n^T, the lab-frame correlation of
    channel c is L_c . S'_c . R_c with (channels ordered q-k, q, q+k):
      k_case 3 (generic):        (R1, R1), (R2, R2), (conj(R1), conj(R1))
      k_case 2 (2k integer):     satellites coincide; cross terms give
                                 ((I - n n^T), R1), (R2, R2), ((I - n n^T), conj(R1))
    k_case 1 is handled by the commensurate path upstream.
    """
    n = np.asarray(n_axis, dtype=float)
    n = n / np.linalg.norm(n)
    nx = np.array([
        [0.0, -n[2], n[1]],
        [n[2], 0.0, -n[0]],
        [-n[1], n[0], 0.0],
    ])
    nnT = np.outer(n, n)
    R1 = 0.5 * (np.eye(3) - 1j * nx - nnT).astype(np.complex128)
    R2 = nnT.astype(np.complex128)
    if k_case == 2:
        perp = (np.eye(3) - nnT).astype(np.complex128)
        return [(perp, R1), (R2, R2), (perp, np.conj(R1))]
    return [(R1, R1), (R2, R2), (np.conj(R1), np.conj(R1))]


def process_calc_Sqw_single_k(
    args: Tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.float64],
        int,
        float,
        Union[List[float], npt.NDArray[np.float64]],
        Optional[List[str]],  # ion_list
        npt.NDArray[np.float64],  # k_cart
        npt.NDArray[np.float64],  # n_axis
        int,  # k_case
    ],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Worker for S(q,w) of a single-k (spiral) structure at one q-point.

    Evaluates the rotating-frame Hamiltonian at the three channels q-k, q, q+k
    and rotates each channel's correlation tensor back to the lab frame with
    the spiral projection tensors. The neutron polarization factor and the
    magnetic form factor always use the PHYSICAL q. Returns 3*nspins modes,
    channel-major: [q-k | q | q+k]. A failing channel yields NaNs for its own
    modes only.
    """
    (
        Ud_numeric,
        q_vector,
        nspins,
        spin_magnitude_num,
        hamiltonian_params_num,
        ion_list,
        k_cart,
        n_axis,
        k_case,
    ) = args

    global _worker_HMat_func
    if _worker_HMat_func is None:
        raise RuntimeError("Worker not initialized with HMat_func")

    k_cart = np.asarray(k_cart, dtype=float)
    tensors = spiral_channel_tensors(n_axis, k_case)
    shifts = [-k_cart, np.zeros(3), +k_cart]

    energies_all = np.full(3 * nspins, np.nan)
    intensities_all = np.full(3 * nspins, np.nan)

    # Physical-q quantities (shared by all channels)
    q_norm_sq = float(np.dot(q_vector, q_vector))
    q_mag = np.sqrt(q_norm_sq)
    ff_values = np.ones(nspins)
    if ion_list:
        for i in range(nspins):
            ff_values[i] = get_form_factor(ion_list[i], q_mag)
    if q_norm_sq < Q_ZERO_THRESHOLD:
        polarization = None  # use trace
    else:
        q_normalized = q_vector / q_mag
        polarization = np.eye(3) - np.outer(q_normalized, q_normalized)

    numerical_args_base = [spin_magnitude_num] + list(hamiltonian_params_num)

    for c, (shift, (T_left, T_right)) in enumerate(zip(shifts, tensors)):
        q_c = q_vector + shift
        q_label = f"q={q_vector} (channel {['q-k', 'q', 'q+k'][c]})"
        sl = slice(c * nspins, (c + 1) * nspins)
        try:
            Hmat_plus_q = np.array(
                _worker_HMat_func(*(list(q_c) + numerical_args_base)),
                dtype=np.complex128,
            )
            Hmat_minus_q = np.array(
                _worker_HMat_func(*(list(-q_c) + numerical_args_base)),
                dtype=np.complex128,
            )
            K_matrix, Kd_matrix, eigenvalues = KKdMatrix(
                spin_magnitude_num, Hmat_plus_q, Hmat_minus_q, Ud_numeric, q_c, nspins
            )
            if (
                np.isnan(K_matrix).any()
                or np.isnan(Kd_matrix).any()
                or np.isnan(eigenvalues).any()
            ):
                logger.error(f"NaN encountered in KKdMatrix result for {q_label}.")
                continue

            imag_energy_mag = np.abs(np.imag(eigenvalues[0:nspins]))
            if np.any(imag_energy_mag > ENERGY_IMAG_PART_THRESHOLD):
                logger.warning(
                    f"Significant imaginary part in energy eigenvalues for {q_label}. "
                    f"Max imag: {np.max(imag_energy_mag)}"
                )
            energies_all[sl] = np.real(eigenvalues[0:nspins])

            # Rotating-frame spin correlation tensor (same einsum as the
            # commensurate worker), with form factors at the physical |q|.
            K_modes = K_matrix[:, :nspins].reshape(nspins, 3, nspins)
            Kd_modes = Kd_matrix[:, nspins:].reshape(nspins, 3, nspins)
            K_w = K_modes * ff_values[:, None, None]
            Kd_w = Kd_modes * ff_values[:, None, None]
            spin_corr = np.einsum("iam,jbm->abm", K_w, Kd_w)

            # Rotate back to the lab frame: L . S' . R per mode.
            spin_corr_lab = np.einsum("ax,xym,yb->abm", T_left, spin_corr, T_right)

            if polarization is None:
                intensity_per_mode = np.einsum("aam->m", spin_corr_lab)
            else:
                intensity_per_mode = np.einsum("ab,abm->m", polarization, spin_corr_lab)

            max_imag = (
                float(np.max(np.abs(np.imag(intensity_per_mode)))) if nspins > 0 else 0.0
            )
            if max_imag > SQW_IMAG_PART_THRESHOLD:
                worst_mode = int(np.argmax(np.abs(np.imag(intensity_per_mode))))
                logger.warning(
                    f"Significant imaginary part in Sqw for {q_label}, mode "
                    f"{worst_mode}: {np.imag(intensity_per_mode[worst_mode])}"
                )

            channel_int = np.real(intensity_per_mode)
            channel_int[channel_int < 0] = 0
            intensities_all[sl] = channel_int
        except Exception:
            logger.exception(f"Error during single-k S(q,w) calculation for {q_label}.")
            continue

    return q_vector, energies_all, intensities_all
