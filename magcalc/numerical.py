import sys
import logging
import numpy as np
import scipy.linalg as la
from sympy import lambdify
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
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
KB_MEV_PER_K: float = 0.08617333262  # Boltzmann constant in meV/K


def thermal_bose_prefactor(
    energies: npt.NDArray[np.float64],
    temperature_K: float,
    e_floor: float = 1e-6,
) -> npt.NDArray[np.float64]:
    """Thermal (Bose) occupation prefactor |1/(1 - exp(-E/kT))| per mode.

    Multiplying the T=0 LSWT intensities by this factor gives the
    finite-temperature cross-section: (n(E)+1) for energy loss (E > 0) and
    n(|E|) for energy gain (E < 0) — the same detailed-balance form as Sunny's
    thermal_prefactor and SpinW's sw_egrid 'T' option. Energies in meV,
    temperature in Kelvin.

    The prefactor diverges like kT/E as E -> 0 (Goldstone modes); |E| is
    floored at e_floor meV so the result stays finite.
    """
    E = np.asarray(energies, dtype=float)
    kT = KB_MEV_PER_K * float(temperature_K)
    if kT <= 0.0:
        return np.ones_like(E)
    E_safe = np.where(np.abs(E) < e_floor, np.where(E < 0.0, -e_floor, e_floor), E)
    with np.errstate(divide="ignore", over="ignore"):
        return np.abs(1.0 / np.expm1(-E_safe / kT))


_XYZ_INDEX = {"x": 0, "y": 1, "z": 2}


def contract_cross_section(
    spin_corr: npt.NDArray[np.complex128],
    q_vector: npt.NDArray[np.float64],
    cross_section: str = "perp",
) -> Tuple[npt.NDArray[np.complex128], bool]:
    """Contract the 3x3 spin-correlation tensor to a per-mode cross-section.

    spin_corr is S^{ab}(q, mode) with shape (3, 3, n_modes). Supported
    cross_section values:
      - "perp"  (default): unpolarized neutron factor sum_ab (d_ab - q^_a q^_b)
        S^{ab}; falls back to the trace at |q| ~ 0 where q^ is undefined.
      - "trace": sum_a S^{aa} (Sunny ssf_trace).
      - a two-letter component like "xx", "zz", "xy": the single lab-frame
        tensor component S^{ab} (Sunny ssf_custom analogue). Off-diagonal
        components are generally complex; the caller reports the real part
        and must not clamp it (it is legitimately signed).

    Returns (per_mode_values, clamp) where clamp says whether negatives are
    numerical noise that should be clipped to zero (true for perp/trace and
    diagonal components).
    """
    cs = (cross_section or "perp").lower()
    if cs == "perp":
        q_norm_sq = float(np.dot(q_vector, q_vector))
        if q_norm_sq < Q_ZERO_THRESHOLD:
            return np.einsum("aam->m", spin_corr), True
        q_hat = np.asarray(q_vector, dtype=float) / np.sqrt(q_norm_sq)
        polarization = np.eye(3) - np.outer(q_hat, q_hat)
        return np.einsum("ab,abm->m", polarization, spin_corr), True
    if cs == "trace":
        return np.einsum("aam->m", spin_corr), True
    if cs in ("chiral", "sf+", "sf-", "sf_plus", "sf_minus"):
        # Chiral (polarization-dependent) term. With the neutron polarization along
        # q (the usual longitudinal SF/NSF setup) ALL magnetic scattering is
        # spin-flip, and the two beam polarizations differ by the chiral term
        #
        #     M_ch(q, w) = i * qhat . [ sum_ab eps_abc S^ab(q, w) ]
        #     sigma_SF^(+/-) = S_perp -/+ M_ch
        #
        # M_ch is the antisymmetric (imaginary) part of the correlation tensor, so
        # it is nonzero only for a chiral structure (a spiral): it vanishes
        # identically for any collinear or coplanar-with-a-mirror state. The SIGN
        # convention here is pinned to Sunny (tests/test_polarized.py).
        q_norm_sq = float(np.dot(q_vector, q_vector))
        if q_norm_sq < Q_ZERO_THRESHOLD:
            n_modes = spin_corr.shape[2]
            zero = np.zeros(n_modes, dtype=np.complex128)
            if cs == "chiral":
                return zero, False
            return np.einsum("aam->m", spin_corr), True
        q_hat = np.asarray(q_vector, dtype=float) / np.sqrt(q_norm_sq)
        eps = np.zeros((3, 3, 3))
        eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
        eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0
        chiral = 1j * np.einsum("abc,c,abm->m", eps, q_hat, spin_corr)
        if cs == "chiral":
            return chiral, False          # signed: do NOT clamp negatives
        polarization = np.eye(3) - np.outer(q_hat, q_hat)
        perp = np.einsum("ab,abm->m", polarization, spin_corr)
        sign = -1.0 if cs in ("sf+", "sf_plus") else +1.0
        return perp + sign * chiral, True
    if len(cs) == 2 and cs[0] in _XYZ_INDEX and cs[1] in _XYZ_INDEX:
        a, b = _XYZ_INDEX[cs[0]], _XYZ_INDEX[cs[1]]
        return spin_corr[a, b, :], a == b
    raise ValueError(
        f"Unknown cross_section '{cross_section}'. Use 'perp', 'trace', or a "
        f"component like 'xx', 'yy', 'zz', 'xy'."
    )

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

def fibonacci_sphere_points(q_mag: float, num_samples: int) -> npt.NDArray[np.float64]:
    """Uniform directions on the |q| sphere -- the SAME Fibonacci construction the
    dipole engine's powder average uses (core.calculate_powder_average)."""
    if q_mag < Q_ZERO_THRESHOLD:
        return np.zeros((1, 3))
    indices = np.arange(0, num_samples, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_samples)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    return np.column_stack((q_mag * np.sin(phi) * np.cos(theta),
                            q_mag * np.sin(phi) * np.sin(theta),
                            q_mag * np.cos(phi)))


def powder_sample_modes(calculator, q_magnitudes, num_samples: int = 50,
                        backend: str = "numpy", temperature=None,
                        cross_section: str = "perp"):
    """Sample-resolved powder modes: the CORRECT input for powder lineshapes.

    For each |q| shell, evaluates calculate_sqw at `num_samples` Fibonacci-sphere
    directions and returns every mode of every direction:

        energies   (n_shells, num_samples * n_modes)
        intensities(n_shells, num_samples * n_modes)   [each direction / num_samples]

    Broadening these (plot_sqw_map / broaden_spectrum) reproduces the true powder
    average I(|Q|, w) -- the SpinW `powspec` convention: each sampled direction
    deposits its intensity at its OWN mode energy, building the full band shape,
    van Hove peaks and all. Averaging the mode energies over the sphere FIRST (the
    legacy per-mode representation) collapses a dispersive band to its center --
    e.g. Cu5SbO6's 10 meV-wide triplon band became a ~1 meV blob at J1, in
    contradiction with the published powder spectrum (PRR 8, 013247, Fig. 5).

    Works with any calculator exposing calculate_sqw (dipole MagCalc, SU(N),
    entangled). The legacy sphere-averaged representation can be derived from the
    result by reshaping to (n_shells, num_samples, n_modes) and averaging axis 1.
    """
    q_mags = np.asarray(q_magnitudes, dtype=float).ravel()
    all_q = []
    for q_mag in q_mags:
        # Always num_samples points (a Q~0 shell is direction-independent anyway),
        # keeping the output rectangular.
        pts = fibonacci_sphere_points(max(float(q_mag), Q_ZERO_THRESHOLD),
                                      num_samples)
        if len(pts) == 1:
            pts = np.repeat(pts, num_samples, axis=0)
        all_q.extend(pts)
    res = calculator.calculate_sqw(np.asarray(all_q), backend=backend,
                                   temperature=temperature,
                                   cross_section=cross_section)
    n_modes = res.energies.shape[1]
    E = np.asarray(res.energies).reshape(len(q_mags), num_samples * n_modes)
    I = np.asarray(res.intensities).reshape(len(q_mags),
                                            num_samples * n_modes) / num_samples
    return E, I


def powder_average_from_sqw(calculator, q_magnitudes, num_samples: int = 50,
                            backend: str = "numpy", temperature=None,
                            cross_section: str = "perp") -> SqwResult:
    """Powder-average S(|q|, w) for ANY calculator exposing calculate_sqw.

    Identical conventions to the dipole engine (core.calculate_powder_average):
    Fibonacci-sphere sampling per |q| shell, one batched calculate_sqw call, then
    the per-shell nanmean of the mode-resolved (energies, intensities). Used by the
    SU(N) and entangled calculators; the averaging is an exact identity over their
    own (independently validated) calculate_sqw, pinned by tests/test_powder_sun.py.
    """
    q_mags = np.asarray(q_magnitudes, dtype=float).ravel()
    all_q, seg = [], []
    for q_mag in q_mags:
        pts = fibonacci_sphere_points(float(q_mag), num_samples)
        all_q.extend(pts)
        seg.append(len(pts))
    res = calculator.calculate_sqw(np.asarray(all_q), backend=backend,
                                   temperature=temperature,
                                   cross_section=cross_section)
    E, I, idx = [], [], 0
    for count in seg:
        E.append(np.nanmean(res.energies[idx:idx + count], axis=0))
        I.append(np.nanmean(res.intensities[idx:idx + count], axis=0))
        idx += count
    q_out = np.column_stack((q_mags, np.zeros_like(q_mags), np.zeros_like(q_mags)))
    return SqwResult(q_vectors=q_out, energies=np.array(E), intensities=np.array(I))


# --- Global variable for worker processes ---
_worker_HMat_func = None

def _init_worker(HMat_sym, full_symbol_list):
    """
    Initializer function for multiprocessing worker.
    Lambdifies the symbolic Hamiltonian once per process.
    """
    global _worker_HMat_func

    try:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        _worker_HMat_func = lambdify(full_symbol_list, HMat_sym, modules=["numpy"], cse=True)
    except Exception as e:
        sys.stderr.write(f"Error in worker initialization: {e}\n")
        raise e

def process_calc_disp(
    args: Tuple[
        npt.NDArray[np.float64],
        int,
        float,
        Union[List[float], npt.NDArray[np.float64]],
        Optional[npt.NDArray[np.complex128]],   # h_dip(+q): Ewald, precomputed
    ],
) -> Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.complex128]]]:
    """
    Worker function for parallel dispersion calculation at a single q-point.
    Uses pre-initialized _worker_HMat_func.

    `h_dip` is the long-range dipolar (Ewald) contribution to H(q), already in the
    host's g*H2 convention. It is precomputed in the parent process because A(q) is an
    infinite lattice sum: it cannot be expressed as bonds and so cannot come through
    the symbolic Hamiltonian.
    """
    (
        _,  # Ud_numeric (unused by HMat_func as it's baked into params/sym)
        q_vector,
        nspins,
        spin_magnitude_num,
        hamiltonian_params_num,
        h_dip,
    ) = args
    
    if _worker_HMat_func is None:   # set per process by _init_worker
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
        if h_dip is not None:
            HMat_numeric = HMat_numeric + h_dip
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
        str,  # cross_section
        Optional[List[float]],  # spin_magnitudes (per site; mixed spin)
        Optional[Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]],  # h_dip(+q), h_dip(-q)
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
        cross_section,
        spin_magnitudes,
        h_dip_pair,
    ) = args
    
    if _worker_HMat_func is None:   # set per process by _init_worker
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
        if h_dip_pair is not None:
            Hmat_plus_q = Hmat_plus_q + h_dip_pair[0]
            Hmat_minus_q = Hmat_minus_q + h_dip_pair[1]
    except Exception:
        logger.exception(f"Error evaluating HMat function at {q_label}.")
        return nan_result
        
    try:
        K_matrix, Kd_matrix, eigenvalues = KKdMatrix(
            spin_magnitude_num, Hmat_plus_q, Hmat_minus_q, Ud_numeric, q_vector, nspins,
            spin_magnitudes=spin_magnitudes
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

        intensity_per_mode, clamp = contract_cross_section(
            spin_corr, q_vector, cross_section
        )

        if clamp:
            # Imaginary parts are numerical noise for perp/trace/diagonal
            # contractions; off-diagonal components are legitimately complex
            # (the caller gets the real part) so no warning there.
            max_imag = float(np.max(np.abs(np.imag(intensity_per_mode)))) if nspins > 0 else 0.0
            if max_imag > SQW_IMAG_PART_THRESHOLD:
                worst_mode = int(np.argmax(np.abs(np.imag(intensity_per_mode))))
                logger.warning(
                    f"Significant imaginary part in Sqw for {q_label}, mode {worst_mode}: {np.imag(intensity_per_mode[worst_mode])}"
                )

        intensities = np.real(intensity_per_mode)
        if clamp:
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
        str,  # cross_section
        Optional[List[float]],  # spin_magnitudes (per site; mixed spin)
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
        cross_section,
        spin_magnitudes,
    ) = args

    if _worker_HMat_func is None:   # set per process by _init_worker
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
                spin_magnitude_num, Hmat_plus_q, Hmat_minus_q, Ud_numeric, q_c, nspins,
                spin_magnitudes=spin_magnitudes
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

            # Contract at the PHYSICAL q (not the channel q).
            intensity_per_mode, clamp = contract_cross_section(
                spin_corr_lab, q_vector, cross_section
            )

            if clamp:
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
            if clamp:
                channel_int[channel_int < 0] = 0
            intensities_all[sl] = channel_int
        except Exception:
            logger.exception(f"Error during single-k S(q,w) calculation for {q_label}.")
            continue

    return q_vector, energies_all, intensities_all
