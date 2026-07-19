#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Spin-Wave Theory (LSWT) Calculator Module.

This module provides the `MagCalc` class and supporting functions to perform
LSWT calculations for magnetic materials. It takes a user-defined spin model
(Hamiltonian, structure, interactions) and computes:

1.  Spin-wave dispersion relations (energy vs. momentum).
2.  Dynamic structure factor S(q, ω) (neutron scattering intensity).

Key features include symbolic manipulation using SymPy, numerical diagonalization
using SciPy/NumPy, parallel processing for performance, and caching of
computationally expensive symbolic results.

@author: Kit Matan and Pharit Piyawongwatthana
@contributor: AI Assistant (Refactoring, Docstrings)
Refactored by AI Assistant
"""
# import spin_model as sm # REMOVED: User-defined spin model will be passed explicitly
from .generic_model import GenericSpinModel
import sympy as sp
from sympy import lambdify
import numpy as np
from scipy import linalg as la
from scipy.optimize import minimize # Added for classical energy minimization
import timeit
import sys
import pickle
from multiprocessing import Pool
from tqdm import tqdm # Added for progress indication

import hashlib  # For numerical cache key generation
import logging
import os  # Added for cpu_count
import json  # For metadata

# Type Hinting Imports
import types  # Added for ModuleType hint
import numpy.typing as npt
from typing import List, Tuple, Dict, Any, Optional, Union

# --- Modularized Imports ---
from .symbolic import gen_HM
from .numerical import (
    process_calc_disp,
    process_calc_Sqw,
    process_calc_Sqw_single_k,
    _init_worker,
    DispersionResult,
    SqwResult,
    ENERGY_IMAG_PART_THRESHOLD,
    fibonacci_sphere_points,
    thermal_bose_prefactor,
)
from .linalg import rotation_matrix
# --- End Modularized Imports ---


def reciprocal_b_matrix(unit_cell: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Reciprocal-lattice B-matrix (rows b1, b2, b3, with 2*pi) from real-space
    lattice vectors given as rows a1, a2, a3."""
    uc = np.asarray(unit_cell, dtype=float)
    a1, a2, a3 = uc[0], uc[1], uc[2]
    V = np.dot(a1, np.cross(a2, a3))
    return np.array([
        2 * np.pi * np.cross(a2, a3) / V,
        2 * np.pi * np.cross(a3, a1) / V,
        2 * np.pi * np.cross(a1, a2) / V,
    ])


# --- Imports for Configuration-Driven Model (with fallback for direct script execution) ---
try:
    # This works when magcalc.py is imported as part of the pyMagCalc package
    from .config_loader import load_spin_model_config
except ImportError:
    # This block executes if the relative import fails.
    # This can happen when run as a script or by multiprocessing spawns.
    # Add the script's directory to sys.path to allow direct imports.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    try:
        from config_loader import load_spin_model_config
    except ImportError as e:
        # If direct import also fails, then there's a more fundamental issue.
        raise ImportError(
            f"Failed to import local modules config_loader. Original error: {e}"
        ) from e
import matplotlib.pyplot as plt  # Added for plotting

# --- Basic Logging Setup ---
# Library should not configure basicConfig. Use NullHandler to silence by default.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
# --- End Logging Setup ---


# --- Analytical classical energy (quadratic form in spin components) ---
# The classical energy of any bilinear spin Hamiltonian (Heisenberg, DM,
# anisotropic exchange, single-ion, Zeeman) is a quadratic form in the Cartesian
# spin components:  E(m) = 0.5 m^T H m + b^T m + c.  Extracting (H, b, c) once
# (see MagCalc._extract_classical_quadratic) lets the minimizer evaluate energy
# and gradient as pure NumPy array ops — no SymPy in the hot loop — which is
# faster per call, scales with N (not bond count), and is trivially picklable so
# parallel workers need no per-task lambdify. These are module-level (not
# closures) so they pickle cleanly to multiprocessing workers.

def _parse_domains(domains) -> Optional[List[Tuple[npt.NDArray[np.float64], float]]]:
    """Normalize a domain/twin specification into [(R, weight), ...].

    Two accepted forms (angles in degrees, axes in Cartesian lab coordinates):
      - shorthand dict {'axis': [0,0,1], 'n_fold': 3}: n equal-weight domains
        rotated by k*360/n (k = 0..n-1), identity included;
      - explicit list [{'axis': ..., 'angle': ..., 'weight': ...}, ...]. The
        list is the COMPLETE set of domains — include {'angle': 0} for the
        original orientation. 'weight' defaults to 1; weights are normalized
        to sum to 1.

    Returns None when no averaging is requested (empty/absent spec, or a
    single identity domain).
    """
    if not domains:
        return None
    if isinstance(domains, dict):
        n = int(domains.get('n_fold', 0))
        if n < 1:
            raise ValueError(
                f"domains dict shorthand needs 'n_fold' >= 1, got {domains}.")
        axis = domains.get('axis', [0, 0, 1])
        entries = [{'axis': axis, 'angle': 360.0 * k / n, 'weight': 1.0}
                   for k in range(n)]
    else:
        entries = list(domains)
    if len(entries) == 0:
        return None
    out = []
    for e in entries:
        angle = float(e.get('angle', e.get('angle_deg', 0.0)))
        R = rotation_matrix(e.get('axis', [0, 0, 1]), angle)
        out.append((R, float(e.get('weight', 1.0))))
    w_sum = sum(w for _, w in out)
    if w_sum <= 0:
        raise ValueError("Domain weights must sum to a positive number.")
    out = [(R, w / w_sum) for R, w in out]
    if len(out) == 1 and np.allclose(out[0][0], np.eye(3)):
        return None
    return out


def _classical_spin_components(x, S, n):
    """Cartesian spin components m (3N,) from angle vector x=[th0,ph0,th1,...].
    Returns m and the trig factors (reused by the gradient)."""
    th = x[0::2]
    ph = x[1::2]
    st, ct = np.sin(th), np.cos(th)
    sp_, cp = np.sin(ph), np.cos(ph)
    m = np.empty(3 * n)
    m[0::3] = S * st * cp
    m[1::3] = S * st * sp_
    m[2::3] = S * ct
    return m, st, ct, sp_, cp


def _classical_energy_np(x, H, b, c, S, n):
    m, *_ = _classical_spin_components(x, S, n)
    return float(0.5 * (m @ (H @ m)) + b @ m + c)


def _classical_grad_np(x, H, b, c, S, n):
    m, st, ct, sp_, cp = _classical_spin_components(x, S, n)
    gm = H @ m + b  # dE/dm (H is the symmetric Hessian)
    dth = np.empty(3 * n)
    dph = np.empty(3 * n)
    dth[0::3] = S * ct * cp
    dth[1::3] = S * ct * sp_
    dth[2::3] = -S * st
    dph[0::3] = -S * st * sp_
    dph[1::3] = S * st * cp
    dph[2::3] = 0.0
    gm3 = gm.reshape(n, 3)
    g = np.empty(2 * n)
    g[0::2] = np.einsum("ij,ij->i", gm3, dth.reshape(n, 3))  # dE/dtheta_i
    g[1::2] = np.einsum("ij,ij->i", gm3, dph.reshape(n, 3))  # dE/dphi_i
    return g


def _minimize_worker(args):
    """
    Worker function for parallel minimization.
    Args: (payload, x0, method, bounds, constraints, kwargs) where payload is
        ('np', H, b, c, S, n)                       -> analytical NumPy energy/grad
        ('sym', E_sym_num, Grad_sym_num, opt_vars)  -> symbolic fallback (lambdified here)
    The analytical payload carries only NumPy arrays, so there is no per-task
    SymPy lambdify (faster worker startup) and the energy/gradient evaluate as
    plain array ops.
    """
    from functools import partial
    payload, x0, method, bounds, constraints, kwargs = args
    from scipy.optimize import minimize, basinhopping, differential_evolution

    if payload[0] == "np":
        _, H, b, c, S, n = payload
        wrapper = partial(_classical_energy_np, H=H, b=b, c=c, S=S, n=n)
        jac_wrapper = partial(_classical_grad_np, H=H, b=b, c=c, S=S, n=n)
    else:
        import sympy as sp
        _, E_sym_num, Grad_sym_num, opt_vars = payload
        # cse=True factors shared trig subexpressions across the energy and every
        # gradient component (~10x faster per call), which dominates runtime.
        E_func = sp.lambdify(opt_vars, E_sym_num, modules="numpy", cse=True)
        def wrapper(x):
            return E_func(*x)
        jac_wrapper = None
        if Grad_sym_num is not None:
            Grad_func = sp.lambdify(opt_vars, Grad_sym_num, modules="numpy", cse=True)
            def jac_wrapper(x):
                return Grad_func(*x)

    if method == 'basinhopping':
        niter = kwargs.pop('niter', 100)
        min_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds, 'constraints': constraints}
        if jac_wrapper:
             min_kwargs['jac'] = jac_wrapper
        return basinhopping(wrapper, x0, niter=niter, minimizer_kwargs=min_kwargs, **kwargs)
        
    elif method == 'differential_evolution':
        constraints_diff = constraints if constraints is not None else ()
        # differential_evolution doesn't use gradient for the main search, but can use it for polishing
        # implicitly via minimizer_kwargs? SciPy's DE polish uses L-BFGS-B but doesn't easily expose passing jac to it unless we use custom minimizer.
        # For now, let's just pass it to 'minimizer_kwargs' just in case the version supports it or for consistency, usually it won't hurt.
        # Actually checking docs: DE polish uses L-BFGS-B, but there is no direct 'jac' arg to DE.
        return differential_evolution(wrapper, bounds, constraints=constraints_diff, **kwargs)
        
    else:
        # Standard minimize
        return minimize(
            wrapper,
            x0,
            method=method,
            bounds=bounds,
            constraints=constraints,
            jac=jac_wrapper,
            **kwargs,
        )


# --- Fast dispersion evaluator (for fitting / repeated parameter changes) ---
class DispersionEvaluator:
    """
    Fast, reusable dispersion evaluator for repeated parameter changes.

    Wraps a single lambdified copy of the symbolic Bogoliubov Hamiltonian
    ``H(q, S, params)`` so mode energies can be evaluated at arbitrary
    Hamiltonian parameters with **no per-call symbolic work** (no ``subs`` and
    no re-``lambdify``). This is the hot path for data fitting, where the
    standard :meth:`MagCalc.calculate_dispersion` pays the lambdify cost on
    every parameter update.

    The energies replicate :func:`magcalc.numerical.process_calc_disp`
    exactly: eigenvalues of the numerical Hamiltonian, sorted, upper
    ``nspins`` branch (padded with NaN when fewer are found).

    Note:
        The magnetic structure enters through the local-frame rotations baked
        into the symbolic Hamiltonian at generation time. If the model's
        structure is changed outside the symbolic parameters (e.g. a new
        minimized configuration installed via ``mpr``), the evaluator must be
        recompiled with :meth:`MagCalc.compile_dispersion_evaluator`.
    """

    def __init__(self, hmat_func, n_args: int, nspins: int,
                 spin_magnitude: float, default_params: List[Any]):
        self._f = hmat_func
        self._n_args = n_args
        self.nspins = nspins
        self.spin_magnitude = float(spin_magnitude)
        self.default_params = list(default_params)

    @staticmethod
    def _flatten_params(params: List[Any]) -> List[float]:
        """Flatten vector parameters to match the flat symbol list."""
        flat: List[float] = []
        for p in params:
            if isinstance(p, (list, tuple, np.ndarray)):
                flat.extend(float(v) for v in p)
            else:
                flat.append(float(p))
        return flat

    def energies(
        self,
        q_vectors: Union[List[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        params: Optional[List[Any]] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Mode energies at each q for the given Hamiltonian parameters.

        Args:
            q_vectors: (N, 3) array or list of Cartesian momentum vectors.
            params: Hamiltonian parameter list (vectors allowed; flattened
                internally). Defaults to the parameters the evaluator was
                compiled with.

        Returns:
            (N, nspins) array of mode energies, ascending per q-point.
        """
        p = self.default_params if params is None else list(params)
        base_args = [self.spin_magnitude] + self._flatten_params(p)
        if 3 + len(base_args) != self._n_args:
            raise ValueError(
                f"Parameter list yields {3 + len(base_args)} lambdify arguments; "
                f"the compiled Hamiltonian expects {self._n_args}."
            )
        qs = np.atleast_2d(np.asarray(q_vectors, dtype=float))
        n = self.nspins
        out = np.empty((qs.shape[0], n), dtype=float)
        # Track the largest imaginary eigenvalue part across all q-points. A
        # significant value signals an unstable (non-ground-state) structure.
        # This mirrors the warning emitted by the standard per-q path
        # (numerical.py), so the fast path does not silently hide instabilities.
        # It is aggregated into a single warning to stay cheap in fitting loops.
        max_imag = 0.0
        max_imag_q = None
        for i, q in enumerate(qs):
            H = np.array(self._f(*(list(q) + base_args)), dtype=np.complex128)
            ev = np.linalg.eigvals(H)
            imag_mag = float(np.max(np.abs(np.imag(ev)))) if ev.size else 0.0
            if imag_mag > max_imag:
                max_imag = imag_mag
                max_imag_q = q
            e = np.real(np.sort(ev))[n:]
            if len(e) != n:
                e = e[:n] if len(e) > n else np.pad(
                    e, (0, n - len(e)), constant_values=np.nan
                )
            out[i] = e
        if max_imag > ENERGY_IMAG_PART_THRESHOLD:
            logger.warning(
                "DispersionEvaluator: significant imaginary eigenvalue part "
                f"(max {max_imag:.3g} at q={np.round(max_imag_q, 4)}); the "
                "magnetic structure may be unstable/not a ground state."
            )
        return out


# --- MagCalc Class ---
class MagCalc:
    """
    Performs Linear Spin Wave Theory (LSWT) calculations.

    This class handles the setup of symbolic Hamiltonian matrices based on a
    user-provided spin model and calculates spin-wave dispersion relations
    and dynamic structure factors S(q,w) using multiprocessing.

    Attributes:
        spin_magnitude (float): Numerical value of the spin magnitude S.
        hamiltonian_params (List[float]): Numerical Hamiltonian parameters.
        cache_file_base (str): Base name for cache files.
        cache_mode (str): Cache mode ('r', 'w', 'auto', 'none').
        sm: The imported spin model module or object.
        nspins (int): Number of spins in the magnetic unit cell.
        k_sym (List[sp.Symbol]): List of momentum symbols [kx, ky, kz].
        S_sym (sp.Symbol): Symbolic spin magnitude 'S'.
        params_sym (Tuple[sp.Symbol, ...]): Tuple of symbolic parameters (p0, p1,...).
        full_symbol_list (List[sp.Symbol]): Combined list of all symbols for HMat lambdification.
        HMat_sym (Optional[sp.Matrix]): Symbolic Hamiltonian matrix (2gH).
        Ud_sym (Optional[sp.Matrix]): Symbolic rotation matrix Ud.
        Ud_numeric (Optional[npt.NDArray[np.complex128]]): Numerical rotation matrix Ud.

    Methods:
        calculate_dispersion(q_list): Computes spin-wave energies for a list of Q-points.
        calculate_sqw(q_list, omega_grids, ...): Computes S(Q,w) with convolution.
        minimize_energy(x0, ...): Numerically finds the classical magnetic ground state.
    """

    def __init__(
        self,
        spin_magnitude: Optional[float] = None,
        hamiltonian_params: Optional[Union[List[float], npt.NDArray[np.float64]]] = None,
        cache_file_base: Optional[
            str
        ] = None,  # Made optional, will be derived from config if possible
        spin_model_module: Optional[types.ModuleType] = None,
        cache_mode: str = "r",
        Ud_numeric_override: Optional[npt.NDArray[np.complex128]] = None,
        config_filepath: Optional[str] = None,  # For configuration-driven model
        initialize: bool = True,
    ):
        """
        Initializes the MagCalc LSWT calculator.

        Loads or generates the necessary symbolic matrices (Hamiltonian HMat=2gH,
        rotation Ud) based on the provided spin model and parameters. Pre-calculates
        the numerical rotation matrix Ud_numeric.

        Args:
            spin_magnitude (Optional[float]): The numerical value of the spin magnitude S.
                Must be positive. Required if config_filepath is None.
            hamiltonian_params (Optional[List[float]]): Numerical parameters for the Hamiltonian.
                Required if config_filepath is None.
            cache_file_base (Optional[str]): Base name for cache files.
                If None, defaults to "magcalc_cache" or derived from config.
            spin_model_module (Optional[types.ModuleType]): Python module defining the spin model.
                Must contain `atom_pos`, `atom_pos_ouc`, `Hamiltonian`, `unit_cell`.
            cache_mode (str): Cache mode: 'r' (read-only), 'w' (force write/regenerate).
                Defaults to "r".
            Ud_numeric_override (Optional[npt.NDArray]): Manually provided numerical rotation matrix Ud.
                Advanced usage.
            config_filepath (Optional[str]): Path to a YAML configuration file.
                If provided, parameters will be loaded from this file.
            initialize (bool): If True (default), proceeds to generate/load symbolic Hamiltonian.
                If False, initialization is skipped (useful for minimization-only instances).
        """
        logger.info(f"Initializing MagCalc (cache_mode='{cache_mode}')...")

        if cache_mode not in ["r", "w", "auto", "none"]:
            raise ValueError(
                f"Invalid cache_mode '{cache_mode}'. Use 'r', 'w', 'auto', or 'none'."
            )
        self.cache_mode = cache_mode

        # --- Define Cache Directories ---
        # Root cache directory, one level above the pyMagCalc package
        self.cache_root_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "cache")
        )
        self.symbolic_cache_dir = os.path.join(self.cache_root_dir, "symbolic_matrices")
        self.numerical_cache_dir = os.path.join(
            self.cache_root_dir, "numerical_results"
        )

        os.makedirs(self.cache_root_dir, exist_ok=True)
        os.makedirs(self.symbolic_cache_dir, exist_ok=True)
        os.makedirs(self.numerical_cache_dir, exist_ok=True)
        # --- End Cache Directory Definitions ---

        self.raw_config_data: Optional[Dict[str, Any]] = (
            None  # Stores the full loaded config
        )
        self.model_config_data: Optional[Dict[str, Any]] = (
            None  # Stores the relevant model section
        )
        self.config_filepath: Optional[str] = config_filepath  # Store config_filepath
        self.config_data: Optional[Dict[str, Any]] = None  # Ensure attribute exists
        self.p_numerical: Optional[Dict[str, float]] = (
            None  # Stores numerical params by name  # Stores numerical params by name
        )

        if config_filepath:
            logger.info(f"Using configuration file: {config_filepath}")
            self.config_data = load_spin_model_config(config_filepath)
            self.raw_config_data = (
                self.config_data
            )  # Keep a reference to the raw loaded data

            # Determine the actual model configuration section
            current_config_root = self.raw_config_data
            if (
                current_config_root is not None
                and "crystal_structure" not in current_config_root
            ):
                if len(current_config_root) == 1:
                    first_key = list(current_config_root.keys())[0]
                    potential_model_section = current_config_root[first_key]
                    if (
                        isinstance(potential_model_section, dict)
                        and "crystal_structure" in potential_model_section
                    ):
                        logger.info(
                            f"Model configuration found under top-level key: '{first_key}'"
                        )
                        self.model_config_data = potential_model_section
                    else:
                        self.model_config_data = current_config_root  # Will likely fail later, but consistently
                else:
                    self.model_config_data = (
                        current_config_root  # Will likely fail later
                    )
            else:
                self.model_config_data = (
                    current_config_root  # crystal_structure is at the root
                )

            if (
                self.model_config_data is None
            ):  # Should not happen if raw_config_data was not None
                raise ValueError(
                    f"Failed to determine model configuration section from: {config_filepath}"
                )
            # Extract calculation settings from config
            calculation_section = self.raw_config_data.get("calculation", {})
            
            # Update cache_mode from config if valid
            config_cache_mode = calculation_section.get("cache_mode")
            if config_cache_mode in ["r", "w", "auto"]:
                if cache_mode == "r": # Only override default 'r' if config specifies something
                     self.cache_mode = config_cache_mode
                     logger.info(f"Using cache_mode='{self.cache_mode}' from configuration.")
            
            # Update cache_file_base from config if not provided in args
            config_cache_base = calculation_section.get("cache_file_base")
            if cache_file_base is None:
                if config_cache_base:
                    self.cache_file_base = config_cache_base
                else:
                    base_name_from_config = os.path.splitext(
                        os.path.basename(config_filepath)
                    )[0]
                    self.cache_file_base = base_name_from_config + "_cache"
            else:
                self.cache_file_base = cache_file_base

            # Instantiate GenericSpinModel with config
            base_path = os.path.dirname(os.path.abspath(config_filepath))
            self.sm = GenericSpinModel(self.model_config_data, base_path=base_path)

            # Extract spin_magnitude from config
            crystal_structure_data = self.model_config_data.get("crystal_structure")
            if crystal_structure_data is None:
                raise ValueError(
                    f"Missing required section 'crystal_structure' in model configuration from: {config_filepath}"
                )

            atoms_uc_config = crystal_structure_data.get("atoms_uc", [])
            if not atoms_uc_config:
                raise ValueError(
                    "Configuration error: 'atoms_uc' is missing or empty in 'crystal_structure'."
                )
            if "spin_S" not in atoms_uc_config[0]:
                raise ValueError(
                    "Configuration error: 'spin_S' is missing for the first atom in 'atoms_uc'."
                )
            self.spin_magnitude = float(atoms_uc_config[0]["spin_S"])
            if self.spin_magnitude <= 0:
                raise ValueError(
                    f"Spin magnitude from config ({self.spin_magnitude}) must be positive."
                )

            # Extract numerical Hamiltonian parameters from config
            # Try "model_params" first (legacy python-model configs), then "parameters".
            self.p_numerical = self.model_config_data.get("model_params")
            if self.p_numerical is None:
                self.p_numerical = self.model_config_data.get("parameters")

            if self.p_numerical is None or not isinstance(self.p_numerical, dict):
                raise ValueError(
                    "Configuration error: Neither 'model_params' nor 'parameters' section found in model configuration, or it's not a dictionary."
                )
            if not all(isinstance(v, (int, float, list, tuple, np.ndarray)) for v in self.p_numerical.values()):
                raise TypeError(
                    "All values in the parameter section ('model_params' or 'parameters') of config must be numbers or lists/arrays of numbers."
                )

            # Ordered list of numerical parameter values
            _parameter_names_ordered = list(self.p_numerical.keys())
            self._params_sym_raw = []
            self._params_val_raw = []
            for name in _parameter_names_ordered:
                val = self.p_numerical[name]
                if isinstance(val, (list, tuple, np.ndarray)):
                    val_list = [float(v) for v in val]
                    sym_list = [sp.Symbol(f"{name}_{i}", real=True) for i in range(len(val_list))]
                    self._params_sym_raw.append(sym_list)
                    self._params_val_raw.append(val_list)
                else:
                    self._params_sym_raw.append(sp.Symbol(name, real=True))
                    self._params_val_raw.append(float(val))

            self.params_sym = self._params_sym_raw 

            # Determine nspins from config
            self.nspins = len(atoms_uc_config)
            if self.nspins == 0:  # Should have been caught by atoms_uc_config check
                raise ValueError(
                    "Configuration error: 'atoms_uc' in config led to zero spins."
                )
            logger.info(
                f"Model loaded from config: {self.nspins} spins, S={self.spin_magnitude}, {len(self.hamiltonian_params) if hasattr(self, 'hamiltonian_params') else 'unknown'} parameters."
            )

        else:  # Traditional spin_model_module approach
            logger.info("Using spin_model_module approach.")
            if spin_model_module is None:
                raise ValueError(
                    "spin_model_module must be provided if config_filepath is not set."
                )
            if cache_file_base is None:
                raise ValueError(
                    "cache_file_base must be provided if config_filepath is not set."
                )
            # Relaxed check: Allow class instances (like GenericSpinModel) that have necessary methods
            if not hasattr(spin_model_module, "Hamiltonian"):
                raise TypeError(
                    "spin_model_module validation failed: missing 'Hamiltonian' method/attribute."
                )
            self.sm = spin_model_module

            # Validate and set spin_magnitude
            assert (
                spin_magnitude is not None
            ), "spin_magnitude cannot be None when config_filepath is not used"
            if not isinstance(spin_magnitude, (int, float)):
                raise TypeError("spin_magnitude must be a number.")
            if spin_magnitude <= 0:
                raise ValueError("spin_magnitude must be positive.")
            self.spin_magnitude = float(spin_magnitude)

            # Validate and set hamiltonian_params (list of floats or nests)
            if hamiltonian_params is None:
                raise ValueError("hamiltonian_params cannot be None.")
            
            if isinstance(hamiltonian_params, np.ndarray):
                hamiltonian_params_list = hamiltonian_params.tolist()
            else:
                hamiltonian_params_list = list(hamiltonian_params)
                
            self._params_sym_raw = []
            self._params_val_raw = []
            _sym_counter = 0
            for p in hamiltonian_params_list:
                if isinstance(p, (list, tuple, np.ndarray)):
                    try:
                        val_list = [float(v) for v in p]
                    except (ValueError, TypeError):
                        raise TypeError("All elements in hamiltonian_params must be numbers.")
                    sym_list = [sp.Symbol(f"p{_sym_counter}_{i}", real=True) for i in range(len(val_list))]
                    self._params_sym_raw.append(sym_list)
                    self._params_val_raw.append(val_list)
                    _sym_counter += 1
                else:
                    self._params_sym_raw.append(sp.Symbol(f"p{_sym_counter}", real=True))
                    try:
                        self._params_val_raw.append(float(p))
                    except (ValueError, TypeError):
                        raise TypeError("All elements in hamiltonian_params must be numbers.")
                    _sym_counter += 1

            self.params_sym = self._params_sym_raw

            self.cache_file_base = cache_file_base
            # Check spin_model_module validity and get nspins
            self._validate_spin_model_module()
            try:
                self.nspins = len(self.sm.atom_pos())  # type: ignore
                if self.nspins == 0:
                    raise ValueError("spin_model.atom_pos() returned an empty list.")
            except Exception as e:
                logger.exception("Error getting nspins from spin_model.atom_pos()")
                raise RuntimeError("Failed to determine nspins from spin model.") from e

        # --- Magnetic supercell normalization ---
        # Sunny/SpinW normalize S(q,w) per CHEMICAL cell (Avec Avec'/Ncells);
        # with a magnetic_supercell of N cells the raw intensities are N times
        # larger, so divide them back for size-independent output.
        self.supercell_ncells: int = int(np.prod(
            getattr(self.sm, 'supercell_dims', [1, 1, 1])))

        # --- Single-k (propagation vector) structure attributes ---
        # Populated when the spin model carries a rotating-frame single-k
        # magnetic structure (GenericSpinModel 'single_k' type). Used for
        # satellite (q +/- k) dispersion branches and the three-channel
        # incommensurate S(q,w).
        self.k_rlu: Optional[npt.NDArray[np.float64]] = None
        self.k_cart: Optional[npt.NDArray[np.float64]] = None
        self.spiral_axis: Optional[npt.NDArray[np.float64]] = None
        self.k_case: Optional[int] = None
        try:
            if getattr(self.sm, 'use_rotating_frame', False):
                k_rlu = getattr(self.sm, 'k_rlu', None)
                if k_rlu is not None:
                    self.k_rlu = np.asarray(k_rlu, dtype=float)
                    B = reciprocal_b_matrix(np.array(self.sm.unit_cell(), dtype=float))
                    self.k_cart = self.k_rlu @ B
                    self.spiral_axis = np.asarray(self.sm.spiral_axis, dtype=float)
                    self.k_case = int(self.sm.k_case)
                    logger.info(
                        f"Single-k structure active: k={self.k_rlu.tolist()} (RLU), "
                        f"axis={np.round(self.spiral_axis, 6).tolist()}, k_case={self.k_case}."
                    )
        except Exception:
            logger.exception("Failed to extract single-k structure attributes; "
                             "satellite branches disabled.")
            self.k_rlu = self.k_cart = self.spiral_axis = self.k_case = None

        # --- Consolidation ---
        def _flatten(l):
            flat = []
            for item in l:
                if isinstance(item, list): flat.extend(item)
                else: flat.append(item)
            return flat

        temp_val_flat = _flatten(self._params_val_raw)
        temp_sym_flat = _flatten(self._params_sym_raw)
        
        self.hamiltonian_params = []
        self.params_sym_flat = []
        
        # Avoid duplicate symbols by filtering out special symbols handled separately
        special_names = {"kx", "ky", "kz", "S"}
        for val, sym in zip(temp_val_flat, temp_sym_flat):
            if sym.name not in special_names:
                self.hamiltonian_params.append(val)
                self.params_sym_flat.append(sym)
            else:
                logger.debug(f"Filtering out special parameter '{sym.name}' from generic parameter list to avoid duplicates.")

        self.kx, self.ky, self.kz = sp.symbols("kx ky kz", real=True)
        self.k_sym: List[sp.Symbol] = [self.kx, self.ky, self.kz]
        self.S_sym: sp.Symbol = sp.Symbol("S", real=True)

        # self.params_sym is already set in the if/else block above
        # For the traditional path, it's p0, p1...
        # For the config path, it's symbols derived from parameter names.
        if not hasattr(self, "params_sym") or self.params_sym is None:
            # This should not happen if the logic above is correct
            raise RuntimeError("self.params_sym was not set during initialization.")

        self.full_symbol_list: List[sp.Symbol] = (
            self.k_sym + [self.S_sym] + self.params_sym_flat
        )

        # --- Load or Generate Symbolic Matrices ---
        self.HMat_sym: Optional[sp.Matrix] = None
        self.Ud_sym: Optional[sp.Matrix] = None
        self.Ud_numeric: Optional[npt.NDArray[np.complex128]] = None

        if initialize:
            # _load_or_generate_matrices raises exceptions on failure
            self._load_or_generate_matrices()
    
            # --- Pre-calculate numerical Ud ---
            if Ud_numeric_override is not None:
                logger.info("Using externally provided Ud_numeric_override.")
                self.set_external_Ud_numeric(Ud_numeric_override)  # Use the existing setter
            else:
                if self.Ud_sym is not None:
                    # _calculate_numerical_ud raises exceptions on failure
                    self._calculate_numerical_ud()
                else:
                    # This case should ideally be caught by _load_or_generate_matrices
                    raise RuntimeError(
                        "Ud_sym is None after matrix loading/generation and no Ud_numeric_override was provided."
                    )
    
            if self.Ud_numeric is None:  # Final check
                raise RuntimeError("Ud_numeric was not set during initialization.")
            logger.info("MagCalc initialization complete.")
        else:
            logger.info("MagCalc initialized in lightweight mode (Hamiltonian generation skipped).")

        # Initialize attribute for storing intermediate Hamiltonian matrices from dispersion calculation
        self._intermediate_numerical_H_matrices_disp: List[
            Optional[npt.NDArray[np.complex128]]
        ] = []

    def _validate_spin_model_module(self):
        """Checks if the provided spin_model_module has required functions."""
        # if self.config_data:  # Skip if using config file (Legacy check, now we validate self.sm regardless)
        #     return

        if self.sm is None: # Should not happen if logic is correct
            raise RuntimeError(
                "Spin model module (self.sm) is not properly set for validation."
            )

        required_funcs = [
            "atom_pos",
            "atom_pos_ouc",
            "mpr",
            "Hamiltonian",
            "spin_interactions",
        ]
        missing_funcs = [
            f
            for f in required_funcs
            if not hasattr(self.sm, f) or not callable(getattr(self.sm, f))
        ]
        if missing_funcs:
            raise AttributeError(
                f"Required function(s) {missing_funcs} not found or not callable in spin_model_module '{self.sm.__name__}'."
            )

    def _read_symbolic_cache_metadata(
        self, meta_filepath: str
    ) -> Optional[Dict[str, Any]]:
        """Reads symbolic cache metadata from a JSON file."""
        try:
            with open(meta_filepath, "r") as f:
                metadata = json.load(f)
            logger.debug(f"Successfully read metadata from {meta_filepath}")
            return metadata
        except FileNotFoundError:
            logger.info(f"Metadata file not found: {meta_filepath}")
            return None
        except json.JSONDecodeError:
            logger.warning(
                f"Error decoding JSON from metadata file {meta_filepath}. File might be corrupted."
            )
            return None
        except Exception as e:
            logger.warning(f"Failed to read metadata file {meta_filepath}: {e}")
            return None

    def _rotation_signature(self) -> Optional[str]:
        """Hash of the ground-state rotation matrices that gen_HM bakes into
        HMat_sym (via ``sm.mpr``). The symbolic Hamiltonian depends on these
        float rotations, so the cache MUST be invalidated when the magnetic
        structure changes — otherwise 'auto' would serve a matrix built for a
        different ground state (wrong physics). Returns None if the rotations
        can't be obtained, in which case the caller treats the cache as
        rotation-agnostic (legacy behavior) rather than crashing.

        Rounded to 1e-6 so reproducible-but-noisy minimization still hits the
        cache, while a genuinely different ground state misses it.
        """
        try:
            mats = self.sm.mpr(list(self.params_sym))  # type: ignore
            hasher = hashlib.sha256()
            for m in mats:
                arr = np.array(sp.Matrix(m).tolist(), dtype=float)
                hasher.update(np.round(arr, 6).tobytes())
            return hasher.hexdigest()
        except Exception as e:
            logger.debug(f"Could not compute rotation signature for cache: {e}")
            return None

    def _model_structure_hash(self) -> Optional[str]:
        """Short hash of the model's STRUCTURE — number of spins, parameter
        symbols, atom positions, and the bond/interaction topology — used to
        namespace the symbolic cache file so different models sharing the same
        ``cache_file_base`` (e.g. every model the GUI runs writes
        ``.config_gui_run_cache``) get distinct cache files instead of clobbering
        each other.

        Deliberately EXCLUDES the ground-state rotations: those are handled by
        the rotation_signature in the validity check, so a model whose ground
        state changes reuses one cache slot (regenerated safely) rather than
        spawning an unbounded set of files during a fitting sweep. This hash is
        only a namespacing convenience — correctness is still guaranteed by
        _check_parameter_consistency_with_cache, so a hash collision at worst
        triggers a safe regeneration.
        """
        try:
            parts = [
                str(self.nspins),
                repr([str(s) for s in self.params_sym]),
            ]
            try:
                parts.append(repr(self.sm.atom_pos_ouc()))  # type: ignore
            except Exception:
                parts.append(repr(self.sm.atom_pos()))  # type: ignore
            try:
                parts.append(repr(self.sm.spin_interactions(list(self.params_sym))))  # type: ignore
            except Exception:
                pass
            hasher = hashlib.sha256("||".join(parts).encode("utf-8"))
            return hasher.hexdigest()[:12]
        except Exception as e:
            logger.debug(f"Could not compute model structure hash for cache: {e}")
            return None

    def _write_symbolic_cache_metadata(self, meta_filepath: str):
        """Writes current model parameters to a symbolic cache metadata JSON file."""
        model_source_type = "config" if self.config_data else "module"
        model_identifier = self.config_filepath if self.config_data else self.sm.__name__  # type: ignore

        metadata = {
            "spin_magnitude": self.spin_magnitude,
            "hamiltonian_params": self.hamiltonian_params,
            "model_source_type": model_source_type,
            "model_identifier": model_identifier,
            "rotation_signature": self._rotation_signature(),
            # "pyMagCalc_version": __version__ # TODO: Add versioning if MagCalc gets one
        }
        try:
            with open(meta_filepath, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Symbolic cache metadata saved to {meta_filepath}")
        except Exception as e:
            logger.error(f"Failed to write metadata file {meta_filepath}: {e}")

    def _check_parameter_consistency_with_cache(
        self, cached_meta: Dict[str, Any]
    ) -> bool:
        """Compares current parameters with cached metadata."""
        if cached_meta is None:
            return False

        current_model_identifier = self.config_filepath if self.config_data else self.sm.__name__  # type: ignore

        s_match = np.isclose(cached_meta.get("spin_magnitude"), self.spin_magnitude)
    
        try:
            p_match = np.allclose(
                cached_meta.get("hamiltonian_params", []),
                self.hamiltonian_params,
                equal_nan=True,
            )
        except (ValueError, TypeError):
            # ValueError can happen if shapes don't match (e.g. different number of params)
            # TypeError can happen if types are incompatible
            p_match = False

        id_match = cached_meta.get("model_identifier") == current_model_identifier

        if not s_match:
            logger.info(
                f"Metadata check: spin_magnitude mismatch (cache: {cached_meta.get('spin_magnitude')}, current: {self.spin_magnitude})."
            )
            return False

        if not id_match:
            logger.info(
                f"Metadata check: model_identifier mismatch (cache: {cached_meta.get('model_identifier')}, current: {current_model_identifier})."
            )
            return False

        # Ground-state rotations are baked into HMat_sym, so a different magnetic
        # structure must invalidate the symbolic cache (otherwise 'auto' returns
        # a matrix built for the wrong ground state). If the current model has a
        # rotation signature, the cache must carry a matching one — a cache that
        # predates rotation tracking (cached_rot is None) can't prove it was
        # built for this ground state, so regenerate rather than risk it.
        current_rot = self._rotation_signature()
        if current_rot is not None and cached_meta.get("rotation_signature") != current_rot:
            logger.info(
                "Metadata check: ground-state rotation signature mismatch or "
                "absent (magnetic structure differs, or cache predates rotation "
                "tracking). Regenerating symbolic cache."
            )
            return False

        # Symbol-compatibility check: the loaded HMat must only contain symbols
        # the current model knows how to bind at lambdify time. A cache from a
        # different model/version under the same cache_file_base could carry an
        # extra symbol, which would survive substitution and blow up as "Cannot
        # convert expression to float". Reject such caches outright.
        try:
            if self.HMat_sym is not None and self.full_symbol_list is not None:
                allowed = {str(s) for s in self.full_symbol_list}
                extra = {str(s) for s in self.HMat_sym.free_symbols} - allowed
                if extra:
                    logger.info(
                        f"Metadata check: cached HMat contains symbols {sorted(extra)} "
                        f"not in the current model's symbol list. Regenerating."
                    )
                    return False
        except Exception as e:
            logger.debug(f"Symbol-compatibility cache check skipped: {e}")

        # Structural check: number of parameters
        cached_params = cached_meta.get("hamiltonian_params", [])
        if len(cached_params) != len(self.hamiltonian_params):
            logger.info(
                f"Metadata check: structural mismatch (parameter count changed from {len(cached_params)} to {len(self.hamiltonian_params)}). Regenerating symbolic cache."
            )
            return False

        # Numerical check
        if not p_match:
            logger.debug(
                f"Metadata check: numerical parameter values changed. Symbolic cache is still valid, but will update numerical Ud."
            )
            # We return True here because for SYMBOLIC matrices, numerical value changes don't require regeneration.
            # Ud_numeric will be re-calculated during MagCalc initialization.
            return True

        return True

    def _generate_and_save_symbolic_matrices(
        self, hm_cache_file: str, ud_cache_file: str, meta_cache_file: str
    ):
        """Generates symbolic matrices and saves them along with metadata."""
        logger.info(
            f"Generating symbolic matrices (HMat, Ud) for {self.cache_file_base}..."
        )
        try:
            self.HMat_sym, self.Ud_sym = gen_HM(
                self.sm,  # type: ignore
                self.k_sym,
                self.S_sym,
                list(self.params_sym),
            )
        except Exception as e:
            logger.exception("Failed to generate symbolic matrices in gen_HM.")
            raise RuntimeError("Symbolic matrix generation failed.") from e

        if not isinstance(self.HMat_sym, sp.Matrix) or not isinstance(
            self.Ud_sym, sp.Matrix
        ):
            raise RuntimeError(
                "Symbolic matrix generation did not return valid SymPy Matrices."
            )

        logger.info(f"Writing HMat to {hm_cache_file}")
        try:
            with open(hm_cache_file, "wb") as outHM:
                pickle.dump(self.HMat_sym, outHM)
        except (IOError, pickle.PicklingError) as e:
            logger.error(f"Error writing HMat cache file: {e}")
            raise
        logger.info(f"Writing Ud to {ud_cache_file}")
        try:
            with open(ud_cache_file, "wb") as outUd:
                pickle.dump(self.Ud_sym, outUd)
        except (IOError, pickle.PicklingError) as e:
            logger.error(f"Error writing Ud cache file: {e}")
            raise

        self._write_symbolic_cache_metadata(meta_cache_file)

    def _load_or_generate_matrices(self):
        """
        Load symbolic HMat (2gH) and Ud matrices from cache or generate them.
        Handles reading from `.pck` files if `cache_mode='r'` or 'auto' (and valid),
        or calling `gen_HM` and writing the files if `cache_mode='w'` or 'auto' (and regeneration needed).
        """
        # Namespace the cache file by model structure so different models that
        # share a cache_file_base (notably every GUI run, which uses
        # ".config_gui_run_cache") don't collide. Falls back to the bare base if
        # the hash can't be computed, preserving legacy behavior.
        struct_hash = self._model_structure_hash()
        effective_base = (
            f"{self.cache_file_base}_{struct_hash}" if struct_hash else self.cache_file_base
        )
        hm_cache_file: str = os.path.join(
            self.symbolic_cache_dir, effective_base + "_HM.pck"
        )
        ud_cache_file: str = os.path.join(  # type: ignore
            self.symbolic_cache_dir, effective_base + "_Ud.pck"
        )
        meta_cache_file: str = os.path.join(
            self.symbolic_cache_dir, effective_base + "_meta.json"
        )

        if self.cache_mode == "auto":
            perform_generation = True
            if (
                os.path.exists(hm_cache_file)
                and os.path.exists(ud_cache_file)
                and os.path.exists(meta_cache_file)
            ):
                try:
                    logger.info(
                        f"Auto mode: Found existing symbolic cache and metadata for {self.cache_file_base}."
                    )
                    with open(hm_cache_file, "rb") as f_hm:
                        self.HMat_sym = pickle.load(f_hm)
                    with open(ud_cache_file, "rb") as f_ud:
                        self.Ud_sym = pickle.load(f_ud)

                    cached_meta = self._read_symbolic_cache_metadata(meta_cache_file)
                    if cached_meta and self._check_parameter_consistency_with_cache(
                        cached_meta
                    ):
                        logger.info(
                            f"Auto mode: Symbolic cache is valid for {self.cache_file_base}. Using cached matrices."
                        )
                        perform_generation = False
                    else:
                        logger.info(
                            f"Auto mode: Symbolic cache parameters mismatch or metadata invalid for {self.cache_file_base}. Regenerating."
                        )
                except Exception as e:
                    logger.warning(
                        f"Auto mode: Error loading or validating symbolic cache for {self.cache_file_base}. Regenerating. Error: {e}"
                    )
                    self.HMat_sym = None
                    self.Ud_sym = None
            else:
                logger.info(
                    f"Auto mode: Symbolic cache or metadata not found for {self.cache_file_base}. Generating."
                )

            if perform_generation:
                self._generate_and_save_symbolic_matrices(
                    hm_cache_file, ud_cache_file, meta_cache_file
                )

        elif self.cache_mode == "none":
            logger.info("Cache mode is 'none'. Generating symbolic matrices in memory without saving to disk.")
            self.HMat_sym, self.Ud_sym = gen_HM(
                self.sm,
                self.k_sym,
                self.S_sym,
                list(self.params_sym),
            )
        elif self.cache_mode == "r":
            logger.info(
                f"Importing symbolic matrices from cache files ({hm_cache_file}, {ud_cache_file})..."
            )
            try:
                with open(hm_cache_file, "rb") as inHM:
                    self.HMat_sym = pickle.load(inHM)
                with open(ud_cache_file, "rb") as inUd:
                    self.Ud_sym = pickle.load(inUd)
            except FileNotFoundError as e:
                logger.error(
                    f"Cache file not found: {e}. Run with cache_mode='w' or 'auto' first."
                )
                raise
            except (pickle.UnpicklingError, EOFError, ImportError, AttributeError) as e:
                logger.error(
                    f"Error loading cache files (may be corrupted or incompatible): {e}"
                )
                raise pickle.PickleError("Failed to load cache file.") from e
            except Exception as e:  # Catch any other unexpected error during loading
                logger.exception("An unexpected error occurred loading cache files.")
                raise RuntimeError("Cache file loading failed.") from e

        elif self.cache_mode == "w":
            self._generate_and_save_symbolic_matrices(
                hm_cache_file, ud_cache_file, meta_cache_file
            )
        else:  # Should have been caught by __init__
            raise ValueError(
                f"Internal error: Unhandled cache_mode '{self.cache_mode}' in _load_or_generate_matrices."
            )
        # Final check after load/generate
        if self.HMat_sym is None or self.Ud_sym is None:
            raise RuntimeError(
                f"Symbolic matrices HMat_sym or Ud_sym are None after loading/generation."
            )
        if not isinstance(self.HMat_sym, sp.Matrix) or not isinstance(
            self.Ud_sym, sp.Matrix  # type: ignore
        ):
            raise TypeError("Loaded cache files do not contain valid SymPy Matrices.")

    def _calculate_numerical_ud(self):
        """
        Calculate the numerical Ud matrix by substituting parameters into Ud_sym.

        Uses the current `self.spin_magnitude` and `self.hamiltonian_params`
        to substitute values into the symbolic `self.Ud_sym` matrix and stores
        the result in `self.Ud_numeric`.
        """
        # Ud_sym existence checked in __init__ before calling this
        logger.info("Calculating numerical Ud matrix...")
        # Ensure correct number of symbols/params match
        if len(self.params_sym_flat) != len(self.hamiltonian_params):
            raise ValueError(
                f"Mismatch between number of symbolic params ({len(self.params_sym_flat)}) and numerical params ({len(self.hamiltonian_params)})."
            )

        # All symbols in params_sym_flat are sp.Symbol by construction
        param_substitutions_ud: List[Tuple[sp.Symbol, float]] = [
            (self.S_sym, self.spin_magnitude)
        ] + list(zip(self.params_sym_flat, self.hamiltonian_params))

        try:
            # Use evalf(subs=...) for potentially better performance/stability
            Ud_num_sym = self.Ud_sym.evalf(subs=dict(param_substitutions_ud))  # type: ignore
            self.Ud_numeric = np.array(Ud_num_sym, dtype=np.complex128)  # type: ignore
        except Exception as e:
            logger.exception(
                "Error during substitution/evaluation into symbolic Ud matrix."
            )
            raise RuntimeError("Failed to calculate numerical Ud matrix.") from e

        if self.Ud_numeric is None:  # Should not happen if evalf succeeds
            raise RuntimeError("Ud_numeric calculation resulted in None.")

    # --- NEW METHODS ---
    def update_spin_magnitude(self, new_spin_magnitude: float):
        """
        Update the spin magnitude S and recalculate dependent numerical matrices.

        Args:
            new_spin_magnitude (float): The new numerical value for S. Must be positive.

        Raises:
            TypeError: If new_spin_magnitude is not a number.
            ValueError: If new_spin_magnitude is not positive.
            RuntimeError: If recalculation of Ud_numeric fails.
        """
        logger.info(f"Updating spin magnitude to {new_spin_magnitude}...")
        if not isinstance(new_spin_magnitude, (int, float)):
            raise TypeError("new_spin_magnitude must be a number.")
        if new_spin_magnitude <= 0:
            raise ValueError("new_spin_magnitude must be positive.")

        self.spin_magnitude = float(new_spin_magnitude)
        # Recalculate Ud_numeric as it depends on S
        self._calculate_numerical_ud()
        logger.info("Spin magnitude updated and Ud_numeric recalculated.")

    def update_hamiltonian_params(
        self, new_hamiltonian_params: Union[List[Any], npt.NDArray[np.float64]]
    ):
        """
        Update the Hamiltonian parameters and recalculate dependent numerical matrices.

        Args:
            new_hamiltonian_params (Union[List[Any], npt.NDArray[np.float64]]):
                The new list or array of numerical Hamiltonian parameters. Must
                have the same length as the original parameters. Elements can be
                numbers or sequences (vectors).

        Raises:
            TypeError: If new_hamiltonian_params structure is invalid.
            ValueError: If the number of new parameters does not match the expected number.
            RuntimeError: If recalculation of Ud_numeric fails.
        """
        logger.info("Updating Hamiltonian parameters...")
        expected_len = len(self.params_sym)
        expected_len_flat = len(self.params_sym_flat)

        if isinstance(new_hamiltonian_params, np.ndarray):
            new_hamiltonian_params = (
                new_hamiltonian_params.tolist()
            )  # Convert numpy array
        if not isinstance(new_hamiltonian_params, list):
            raise TypeError("new_hamiltonian_params must be a list or NumPy array.")
        # Accept BOTH shapes: the nested form (vector parameters as sublists,
        # matching params_sym) and the flat scalar form (matching
        # params_sym_flat and self.hamiltonian_params itself). Without the flat
        # form, round-tripping the object's own flat `hamiltonian_params`
        # failed for any model with a vector parameter (e.g. H_dir).
        n_given = len(new_hamiltonian_params)
        all_scalars = all(isinstance(x, (int, float)) for x in new_hamiltonian_params)
        if n_given != expected_len and not (all_scalars and n_given == expected_len_flat):
            raise ValueError(
                f"Incorrect number of parameters provided. Expected {expected_len} "
                f"(nested; vectors as sublists) or {expected_len_flat} (flat scalars), "
                f"got {n_given}."
            )

        # Validate elements similarly to __init__
        if not all(isinstance(x, (int, float, list, tuple, np.ndarray)) for x in new_hamiltonian_params):
             raise TypeError("All elements in new_hamiltonian_params must be numbers or sequences (vectors).")

        # Store as a FLAT list of scalars so it matches `params_sym_flat`
        # (vectors such as a field direction are expanded component-wise),
        # exactly as __init__ does. Keeping vectors as nested sublists here
        # would desynchronize the length from params_sym_flat and break
        # _calculate_numerical_ud for any model with a vector parameter.
        self.hamiltonian_params = []
        for p in new_hamiltonian_params:
            if isinstance(p, (list, tuple, np.ndarray)):
                self.hamiltonian_params.extend(float(v) for v in p)
            else:
                self.hamiltonian_params.append(float(p))

        # Recalculate Ud_numeric as it depends on parameters
        self._calculate_numerical_ud()
        logger.info("Hamiltonian parameters updated and Ud_numeric recalculated.")

    def set_external_Ud_numeric(self, Ud_matrix_numerical: npt.NDArray[np.complex128]):
        """
        Allows setting the Ud_numeric matrix externally.
        This is useful if Ud_numeric is derived from a field-dependent classical ground state
        that is determined outside the initial symbolic generation of Ud_sym.

        Args:
            Ud_matrix_numerical (npt.NDArray[np.complex128]): The externally calculated
                numerical Ud matrix (3N x 3N).
        """
        if (
            not isinstance(Ud_matrix_numerical, np.ndarray)
            or Ud_matrix_numerical.ndim != 2
        ):
            raise TypeError("Ud_matrix_numerical must be a 2D NumPy array.")

        expected_dim = 3 * self.nspins
        if Ud_matrix_numerical.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Ud_matrix_numerical has incorrect shape {Ud_matrix_numerical.shape}. Expected ({expected_dim}, {expected_dim})."
            )

        self.Ud_numeric = Ud_matrix_numerical.astype(np.complex128)
        logger.info(
            f"External Ud_numeric matrix has been set. Shape: {self.Ud_numeric.shape}"
        )

    # Removed _generate_matrices_from_config

    def _generate_numerical_cache_key(
        self, q_vectors_list: List[npt.NDArray[np.float64]], calculation_type: str
    ) -> str:
        """
        Generates a unique cache key for numerical results.
        Args:
            q_vectors_list: List of 1D NumPy arrays representing q-vectors.
            calculation_type: String identifier like "dispersion" or "sqw".
        Returns:
            A hex digest string representing the cache key.
        """
        hasher = hashlib.md5()

        # 0. Algorithm version: bump whenever a change in linalg.py /
        # numerical.py affects the numerical output so that stale cache
        # entries are invalidated automatically.
        #   v2 - fix vertical streaks in S(Q,w) by replacing the custom
        #        -q basis selection in KKdMatrix with standard Gram–Schmidt
        hasher.update(b"algo_v2")

        # 1. Symbolic model identifier
        hasher.update(str(self.cache_file_base).encode("utf-8"))
        # 1b. Hash of the actual symbolic Hamiltonian. Without this, editing
        # the model (interactions, transformations, etc.) without changing
        # cache_file_base or numerical parameters would silently reuse stale
        # numerical results.
        try:
            if self.HMat_sym is not None:
                hasher.update(sp.srepr(self.HMat_sym).encode("utf-8"))
        except Exception:
            # Hashing should never break the calculation; fall back silently.
            hasher.update(b"HMat_sym_unhashable")
        # 1c. Long-range dipolar (Ewald) spec. The Ewald term is added to H(q)
        # NUMERICALLY (it never enters HMat_sym), so toggling or editing the
        # dipole_dipole block would otherwise serve a stale cached result.
        try:
            hasher.update(repr(self._ewald_spec()).encode("utf-8"))
        except Exception:
            hasher.update(b"ewald_spec_unhashable")
        # 2. Spin magnitude
        hasher.update(str(self.spin_magnitude).encode("utf-8"))
        # 3. Hamiltonian parameters
        hasher.update(str(self.hamiltonian_params).encode("utf-8"))
        # 4. Ud_numeric matrix (critical for spin configuration)
        if self.Ud_numeric is not None:
            hasher.update(self.Ud_numeric.tobytes())
        else:
            # This case should ideally not be reached if __init__ ensures Ud_numeric is set
            hasher.update(b"Ud_numeric_None")
            logger.warning("_generate_numerical_cache_key: self.Ud_numeric is None.")
        # 5. Calculation type
        hasher.update(calculation_type.encode("utf-8"))
        # 6. Q-vectors content and order
        for q_vec in q_vectors_list:
            hasher.update(q_vec.tobytes())

        return calculation_type + "_" + hasher.hexdigest()

    # --- END NEW METHODS ---

    def calculate_dispersion(
        self,
        q_vectors_list: Union[List[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        processes: Optional[int] = None,
        serial: bool = False,
        backend: str = "numpy",
        satellites: Optional[bool] = None,
    ) -> Optional[DispersionResult]:
        """
        Calculate the spin-wave dispersion relation over a list of q-points.

        Args:
            q_vectors_list (Union[List[npt.NDArray[np.float64]], npt.NDArray[np.float64]]):
                A list or NumPy array of momentum vectors q = [qx, qy, qz].
                Each vector should be a 1D array/list of 3 numbers.
            processes (Optional[int]): Number of worker processes (default: cpu_count).
            serial (bool): If True, run in the main process without multiprocessing (avoid spawn/fork overhead).
            backend (str): "numpy" (default) uses the in-process multiprocessing
                path. "fortran" uses the external fMagCalc backend if available,
                transparently falling back to NumPy otherwise.
            satellites (Optional[bool]): For a single-k structure, also compute
                the satellite branches omega(q -/+ k) by evaluating the same
                rotating-frame Hamiltonian at the shifted momenta (SpinW/Sunny
                three-branch convention). The result then has 3*nspins energy
                columns, channel-major [q-k | q | q+k]. Default False
                (central branch only, matching the previous behavior). Ignored
                when no single-k structure is active.

        Returns:
            Optional[DispersionResult]: A result object containing energies and q-vectors.
            Returns None if the calculation fails to start.
        """
        if satellites and self.k_cart is not None:
            # Expand each q into its three channels and reuse the plain path;
            # the numerical cache key follows the expanded q-list automatically.
            if isinstance(q_vectors_list, np.ndarray) and q_vectors_list.ndim == 1:
                q_vectors_list = [q_vectors_list]
            q_orig = [np.asarray(q, dtype=float) for q in q_vectors_list]
            q_expanded: List[npt.NDArray[np.float64]] = []
            for q in q_orig:
                q_expanded.extend([q - self.k_cart, q, q + self.k_cart])
            res = self.calculate_dispersion(
                q_expanded, processes=processes, serial=serial, backend=backend,
                satellites=False,
            )
            if res is None:
                return None
            nq = len(q_orig)
            E = np.asarray(res.energies).reshape(nq, 3, self.nspins)
            energies = np.concatenate([E[:, 0, :], E[:, 1, :], E[:, 2, :]], axis=1)
            labels = (["q-k"] * self.nspins + ["q"] * self.nspins
                      + ["q+k"] * self.nspins)
            return DispersionResult(
                q_vectors=np.array(q_orig), energies=energies, branch_labels=labels
            )

        start_time = timeit.default_timer()
        if self.HMat_sym is None:  # Should be caught by __init__
            logger.error("Symbolic matrix HMat_sym not available.")
            return None

        logger.info("Running dispersion calculation...")

        if self.HMat_sym is None or self.Ud_numeric is None:
            logger.error("HMat or Ud_numeric is missing. Cannot calculate dispersion.")
            return None

        # Convert to list of lists/arrays
        if isinstance(q_vectors_list, np.ndarray):
            if q_vectors_list.ndim == 1 and q_vectors_list.shape == (3,):  # Single vector case
                q_vectors_list = [q_vectors_list]  # Wrap in a list
            elif q_vectors_list.ndim != 2 or q_vectors_list.shape[1] != 3:
                raise ValueError("q_vectors_list NumPy array must be 2D with shape (N, 3).")
            # Convert rows to separate arrays for pool.imap if needed, or keep as list of arrays
            q_vectors_list = [q_vec for q_vec in q_vectors_list]
        elif isinstance(q_vectors_list, list):
            if not q_vectors_list:
                raise ValueError("q_vectors_list list cannot be empty.")
            if not all(
                isinstance(q, (list, np.ndarray)) and len(q) == 3 for q in q_vectors_list
            ):
                raise ValueError(
                    "Each element in q_vectors_list list must be a list/array of length 3."
                )
            # Ensure elements are numpy arrays
            q_vectors_list = [np.array(q, dtype=float) for q in q_vectors_list]
        else:
            raise TypeError("q_vectors_list must be a list or NumPy array.")
        # --- End q_vector validation ---

        # Opt-in Fortran backend (falls back to NumPy on any unavailability).
        if backend == "fortran":
            fortran_result = self._calculate_dispersion_fortran(q_vectors_list)
            if fortran_result is not None:
                return fortran_result

        num_processes = processes if processes is not None else os.cpu_count()

        # Prepare numeric params list corresponding to self.params_sym_flat symbols
        filtered_h_params = list(self.hamiltonian_params)

        # Long-range dipolar (Ewald) term: A(q) is an infinite lattice sum, so it
        # cannot be a bond and cannot come through the symbolic Hamiltonian. Compute
        # it here (parent process) and hand each worker its H_dip(q).
        dip_stack = self._ewald_h_stack(np.array(q_vectors_list, dtype=float))
        task_args = [
            (
                self.Ud_numeric,  # Pass numerical Ud
                np.array(q, dtype=np.float64),
                self.nspins,
                self.spin_magnitude,
                filtered_h_params,
                None if dip_stack is None else dip_stack[iq],
            )
            for iq, q in enumerate(q_vectors_list)
        ]

        results = []

        # --- Numerical Cache Check ---
        cache_key = self._generate_numerical_cache_key(q_vectors_list, "dispersion")
        cache_filepath = os.path.join(self.numerical_cache_dir, cache_key + ".pkl")

        if self.cache_mode not in ['w', 'none'] and os.path.exists(cache_filepath):
            try:
                with open(cache_filepath, "rb") as f:
                    cached_result = pickle.load(f)
                logger.info(
                    f"Loaded dispersion results from numerical cache: {cache_filepath}"
                )
                # Note: _intermediate_numerical_H_matrices_disp will not be populated from cache
                self._intermediate_numerical_H_matrices_disp = [None] * len(
                    q_vectors_list
                )
                if isinstance(cached_result, DispersionResult):
                    return cached_result
                elif isinstance(cached_result, list):
                    # Legacy cache support
                    logger.warning("Loaded legacy cache format (list). Converting to DispersionResult.")
                    return DispersionResult(
                        q_vectors=np.array(q_vectors_list),
                        energies=np.array(cached_result)
                    )
                else:
                    logger.warning("Unknown cache format. Recalculating.")
                    # Fall through to recalculation
            except Exception as e:
                logger.warning(
                    f"Failed to load from numerical cache {cache_filepath}: {e}. Recalculating."
                )
        # Clear previous intermediate matrices if any
        self._intermediate_numerical_H_matrices_disp = []

        # --- Serial Execution Path ---
        if serial or (num_processes == 1):
             logger.info("Running dispersion calculation in SERIAL mode.")
             # Use global worker func? No, simpler to just re-use the worker logic but we need the lambda.
             # We can't easily use 'process_calc_disp' without _worker_HMat_func being set in THIS process.
             # So we must initialize it here.

             # Filter symbols for lambdify
             lambdify_symbols = [s for s in self.full_symbol_list if isinstance(s, sp.Symbol)]

             # Initialize worker in main process for serial execution
             # This sets numerical._worker_HMat_func used by process_calc_disp
             _init_worker(self.HMat_sym, lambdify_symbols)

             # Run loop
             for args in tqdm(task_args, total=len(task_args), desc="Dispersion (Serial)", unit="q-point"):
                 results.append(process_calc_disp(args))

        else:
            # --- Parallel Execution Path ---
            try:
                # We must pickle the symbolic matrix only once.
                lambdify_symbols = [s for s in self.full_symbol_list if isinstance(s, sp.Symbol)]

                with Pool(
                    initializer=_init_worker,
                    initargs=(self.HMat_sym, lambdify_symbols),
                    processes=num_processes
                ) as pool:
                    # Use imap to report progress with tqdm
                    results = list(
                        tqdm(
                            pool.imap(process_calc_disp, task_args),
                            total=len(task_args),
                            desc="Dispersion",
                            unit="q-point",
                        )
                    )
            except Exception as e:
                logger.exception(f"Parallel processing failed during dispersion calculation: {e}")
                return None

        # Unpack results
        # results is list of (energies, HMat_numeric)
        energies_list = []
        self._intermediate_numerical_H_matrices_disp = [] # Clear or store
        
        for res_energies, res_H_matrix in results:
            if res_energies is None: # Should be handled by worker returning NaNs usually
                 energies_list.append(np.full(self.nspins, np.nan))
            else:
                energies_list.append(res_energies)
            self._intermediate_numerical_H_matrices_disp.append(res_H_matrix)

        num_failures = sum(
            1 for en in energies_list if np.isnan(en).any()
        )
        if num_failures > 0:
            logger.warning(
                f"Dispersion calculation failed for {num_failures} out of {len(q_vectors_list)} q-points. Check logs for details."
            )

        end_time: float = timeit.default_timer()
        logger.info(
            f"Run-time for dispersion calculation: {np.round((end_time - start_time) / 60, 2)} min."
        )

        dispersion_result = DispersionResult(
            q_vectors=np.array(q_vectors_list),
            energies=np.array(energies_list) 
        )
        # --- Save to Numerical Cache ---
        if self.cache_mode != 'none':
            try:
                with open(cache_filepath, "wb") as f:
                    pickle.dump(dispersion_result, f)
                logger.info(
                    f"Saved dispersion results to numerical cache: {cache_filepath}"
                )
            except Exception as e:
                logger.warning(f"Failed to save to numerical cache {cache_filepath}: {e}")

        return dispersion_result

    def compile_dispersion_evaluator(self) -> DispersionEvaluator:
        """
        Compile a fast, parameter-symbolic dispersion evaluator.

        The symbolic Hamiltonian is lambdified **once** over
        ``(q, S, parameters)``; the returned :class:`DispersionEvaluator`
        then computes mode energies at any parameter values as a pure
        numerical operation. Use this in fitting loops instead of calling
        :meth:`update_hamiltonian_params` + :meth:`calculate_dispersion`
        per iteration.

        Returns:
            DispersionEvaluator: reusable evaluator bound to the current
            symbolic Hamiltonian (and hence the current magnetic structure).

        Raises:
            RuntimeError: If the symbolic Hamiltonian is not available.
        """
        if self.HMat_sym is None:
            raise RuntimeError(
                "Symbolic Hamiltonian not available; cannot compile evaluator."
            )
        if self._ewald_spec() is not None:
            # The Ewald dipolar term is added to H(q) numerically; the compiled
            # evaluator only sees the symbolic HMat and would silently drop it,
            # fitting a Hamiltonian missing the dipole-dipole interaction.
            raise RuntimeError(
                "Fast dispersion evaluator does not support dipole_dipole "
                "(method: ewald): the dipolar A(q) is added to H(q) numerically "
                "and would be silently omitted. Use the standard "
                "calculate_dispersion path (fitting.fast: false falls back to it)."
            )
        lambdify_symbols = [
            s for s in self.full_symbol_list if isinstance(s, sp.Symbol)
        ]
        t0 = timeit.default_timer()
        hmat_func = lambdify(
            lambdify_symbols, self.HMat_sym, modules=["numpy"], cse=True
        )
        logger.info(
            f"Compiled dispersion evaluator ({len(lambdify_symbols)} symbols) "
            f"in {timeit.default_timer() - t0:.2f} s."
        )
        return DispersionEvaluator(
            hmat_func,
            n_args=len(lambdify_symbols),
            nspins=self.nspins,
            spin_magnitude=self.spin_magnitude,
            default_params=list(self.hamiltonian_params),
        )

    @staticmethod
    def _locate_fmagcalc() -> Optional[str]:
        """Best-effort: make the `fmagcalc` package importable. Honors the
        FMAGCALC_PATH env var, then falls back to a sibling checkout
        (../fMagCalc/python next to the pyMagCalc repo). Returns the path added
        to sys.path, or None."""
        candidates = []
        env_path = os.environ.get("FMAGCALC_PATH")
        if env_path:
            candidates.append(env_path)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.append(os.path.join(os.path.dirname(repo_root), "fMagCalc", "python"))
        for c in candidates:
            if c and os.path.isdir(os.path.join(c, "fmagcalc")) and c not in sys.path:
                sys.path.insert(0, c)
                return c
        return None

    def _minimize_anneal(
        self, H_q, b_q, c_q, S_val, nspins, method="anneal", num_starts=1,
        x0=None, seed=0, n_sweeps=2000, T_start=None, T_end=None,
        polish=True, **_ignored,
    ):
        """Ground state by simulated annealing (or pure steepest descent).

        method:
          'anneal' / 'monte_carlo' -- Metropolis with a geometric cooling schedule,
              restarted `num_starts` times (SpinW `anneal`; Sunny `LocalSampler`
              proposal mix). Crosses barriers, so it does not get stuck the way
              multistart L-BFGS does.
          'steep' / 'optmagsteep' -- iterative alignment with the local field only
              (SpinW `optmagsteep`). Monotone: fast, but cannot escape a local
              minimum. Included for parity / polishing.

        Every run is finished with an L-BFGS polish so the answer is a true
        stationary point, not a low-temperature snapshot. Returns the usual
        OptimizeResult, with `hits` = how many restarts reached the best energy
        (the same convergence evidence the multistart path reports).
        """
        from scipy.optimize import OptimizeResult

        from .annealing import (
            anneal, cartesian_to_angles, random_spins, steepest_descent,
        )

        m0 = None
        if x0 is not None:
            m0, *_ = _classical_spin_components(np.asarray(x0, float), S_val, nspins)

        n_runs = max(int(num_starts), 1)
        best_x, best_e, hits = None, np.inf, 0
        steep_only = method in ("steep", "optmagsteep")

        logger.info(
            f"Ground state via {'steepest descent' if steep_only else 'simulated annealing'}"
            f" ({n_runs} run(s)"
            f"{'' if steep_only else f', {n_sweeps} sweeps x {nspins} moves'})..."
        )

        for run in range(n_runs):
            rng = np.random.default_rng((seed or 0) + run)
            if steep_only:
                start = m0 if (m0 is not None and run == 0) \
                    else random_spins(nspins, S_val, rng)
                m, e = steepest_descent(start, H_q, b_q, c_q, S_val, nspins)
            else:
                m, e = anneal(H_q, b_q, c_q, S_val, nspins, n_sweeps=n_sweeps,
                              T_start=T_start, T_end=T_end,
                              seed=(seed or 0) + run,
                              m0=m0 if run == 0 else None)

            x = cartesian_to_angles(m, nspins)
            if polish:
                res = minimize(_classical_energy_np, x,
                               args=(H_q, b_q, c_q, S_val, nspins),
                               jac=_classical_grad_np, method="L-BFGS-B",
                               options={"maxiter": 1000})
                x, e = res.x, float(res.fun)

            if np.isclose(e, best_e, atol=1e-6):
                hits += 1
                if e < best_e:      # same minimum, marginally better polish
                    best_e, best_x = e, x
            elif e < best_e:
                best_e, best_x, hits = e, x, 1
                logger.info(f"New best energy found: {best_e:.6f} meV (run {run + 1})")

        result = OptimizeResult(
            x=best_x, fun=best_e, success=best_x is not None, nit=n_runs,
            message=f"{method}: best of {n_runs} run(s)",
        )
        result.global_message = f"{method} ({n_runs} runs, {hits} hit the minimum)"
        result.hits = hits
        result.num_starts = n_runs
        if hits <= 1 and n_runs > 1:
            logger.warning(
                f"{method}: the best energy ({best_e:.6f} meV) was reached by only "
                f"{hits} of {n_runs} runs. Raise `num_starts` or `n_sweeps` and check "
                f"the energy is reproducible across `seed`s."
            )
        logger.info(f"{method} complete: E_min = {best_e:.6f} meV ({hits}/{n_runs} runs).")
        return result

    def relax_from_current(
        self, params: Optional[List[float]] = None, tol: float = 1e-6
    ) -> Optional[Tuple[float, float]]:
        """Energy of the CURRENT magnetic structure vs. the energy after a local
        downhill relaxation from it. Returns (E_now, E_relaxed), or None if the
        classical energy is not available.

        This is the second, independent ground-state guard. It catches the case the
        imaginary-energy check CANNOT see: a stationary point that is a maximum or
        saddle -- e.g. a ferromagnetic structure supplied for an antiferromagnet.
        There the Bogoliubov problem stays diagonal, `process_calc_disp` sorts the
        +/- omega pairs and returns the upper half, so the spectrum comes back real
        and positive (it is |omega|) and nothing looks wrong. Energy is the honest
        test: if a downhill step lowers it, the structure is not the ground state.
        """
        if params is None:
            params = self.hamiltonian_params
        if params is None:
            return None
        nspins = len(self.sm.atom_pos())
        nspins_ouc = len(self.sm.atom_pos_ouc())
        S_val = float(self.spin_magnitude)
        quad = self._extract_classical_quadratic(params, nspins, nspins_ouc, S_val)
        if quad is None:
            return None
        H_q, b_q, c_q = quad

        # Current spin directions, as angles. mpr() may hand back SYMBOLIC rotation
        # matrices (models with a `transformations` block build them from the
        # parameter symbols), so bind the numeric parameters before converting. A
        # diagnostic must never break an otherwise-working run: if the rotations
        # still cannot be made numeric, skip this guard rather than raise.
        rot = self.sm.mpr(list(self.params_sym))
        subs = dict(zip(self.params_sym_flat, params))
        x0 = np.zeros(2 * nspins)
        for i in range(nspins):
            R_i = rot[i]
            try:
                if hasattr(R_i, "subs"):
                    R_i = R_i.subs(subs)
                R = np.array(R_i, dtype=float)
            except (TypeError, ValueError):
                logger.debug(
                    "relax_from_current: rotation matrices are not numeric "
                    "(symbolic model); skipping the classical-energy ground-state guard."
                )
                return None
            z = R @ np.array([0.0, 0.0, 1.0])          # local z -> global spin dir
            x0[2 * i] = np.arccos(np.clip(z[2], -1.0, 1.0))
            x0[2 * i + 1] = np.arctan2(z[1], z[0])

        e_now = _classical_energy_np(x0, H_q, b_q, c_q, S_val, nspins)

        # Relax from SLIGHTLY PERTURBED copies, not from x0 itself. A maximum or
        # saddle (e.g. a ferromagnetic state with antiferromagnetic exchange) is a
        # stationary point: the gradient there is zero, so a descent started exactly
        # at it never moves and the energy looks stable. An infinitesimal nudge is
        # what makes it roll away. The perturbation is deliberately SMALL (0.05 rad)
        # so this stays a LOCAL-minimum test -- it must not hop to some distant
        # basin and start reporting legitimate metastable states as errors.
        rng = np.random.default_rng(0)
        e_best = e_now
        for _ in range(5):
            xp = x0 + rng.normal(0.0, 0.05, size=x0.shape)
            res = minimize(_classical_energy_np, xp,
                           args=(H_q, b_q, c_q, S_val, nspins),
                           jac=_classical_grad_np, method="L-BFGS-B",
                           options={"maxiter": 500, "ftol": tol})
            e_best = min(e_best, float(res.fun))
        return float(e_now), float(e_best)

    # ---------------- long-range dipole-dipole (Ewald) -----------------------
    def _ewald_spec(self):
        """The `interactions.dipole_dipole` block, when it asks for Ewald."""
        cfg = getattr(self.sm, "config", None)
        if not isinstance(cfg, dict):
            return None
        spec = cfg.get("dipole_dipole")
        if spec is None and isinstance(cfg.get("interactions"), dict):
            spec = cfg["interactions"].get("dipole_dipole")
        if not isinstance(spec, dict):
            return None
        if str(spec.get("method", "truncated")).lower() != "ewald":
            return None
        return spec

    def _ewald_A(self, q_rlu):
        """Ewald A(q) (n,n,3,3), cached per q (bounded cache; the guard,
        dispersion, and both S(q,w) branches would otherwise each redo the full
        lattice sum for overlapping q sets)."""
        from .ewald import dipole_ewald_at_q

        key = tuple(np.round(np.asarray(q_rlu, dtype=float), 10))
        cache = getattr(self, "_ewald_A_cache", None)
        if cache is None:
            cache = self._ewald_A_cache = {}
        hit = cache.get(key)
        if hit is not None:
            return hit

        spec = self._ewald_spec()
        cs = self.sm.config["crystal_structure"]
        lat = np.asarray(cs["lattice_vectors"], dtype=float)
        atoms = cs["atoms_uc"]
        frac = np.asarray([a["pos"] for a in atoms], dtype=float)
        demag = spec.get("demag")
        demag = np.asarray(demag, dtype=float) if demag is not None else None
        A = dipole_ewald_at_q(lat, frac, q_rlu, demag=demag)
        if len(cache) > 4096:   # keep memory bounded on very long q-paths
            cache.clear()
        cache[key] = A
        return A

    def _ewald_g(self):
        atoms = self.sm.config["crystal_structure"]["atoms_uc"]
        spec = self._ewald_spec() or {}
        override = spec.get("g")
        out = []
        for a in atoms:
            g = override if override is not None else a.get("g", 2.0)
            if not isinstance(g, (int, float)):
                raise ValueError(
                    f"dipole_dipole (ewald) needs a scalar g per site; site "
                    f"{a.get('label')} has g={g!r}. Set `g:` in the dipole_dipole block.")
            out.append(float(g))
        return np.asarray(out, dtype=float)

    def _ewald_nambu(self, q_cart):
        """Dipolar contribution to H(q), in the host's g*H2 convention.

        A(q) is an infinite lattice sum, so it cannot be a bond list and cannot come
        through the symbolic Hamiltonian -- it is added to H(q) numerically here. The
        block construction is the standard Nambu one (verified to reproduce
        pyMagCalc's own H(q) from a bond list, element for element).
        """
        from .ewald import exchange_from_A

        n = int(self.nspins)
        lat = np.asarray(self.sm.config["crystal_structure"]["lattice_vectors"], float)
        # q_cart -> rlu
        q_rlu = np.asarray(q_cart, dtype=float) @ np.linalg.inv(
            2.0 * np.pi * np.linalg.inv(lat).T)

        Jq = exchange_from_A(self._ewald_A(q_rlu), self._ewald_g())
        J0 = exchange_from_A(self._ewald_A(np.zeros(3)), self._ewald_g())

        rots = [np.array(R, dtype=float) for R in self.sm.mpr(list(self.params_sym))]
        mags = self.sm.spin_magnitudes()
        S = [float(m) for m in mags] if mags and len(mags) == n             else [float(self.spin_magnitude)] * n

        H11 = np.zeros((n, n), complex); H22 = np.zeros((n, n), complex)
        H12 = np.zeros((n, n), complex); H21 = np.zeros((n, n), complex)
        for i in range(n):
            for j in range(n):
                Ji = np.sqrt(S[i] * S[j]) * (rots[i].T @ Jq[i, j] @ rots[j])
                Qm = 0.5 * (Ji[0, 0] + Ji[1, 1] - 1j * (Ji[0, 1] - Ji[1, 0]))
                Qp = 0.5 * (Ji[0, 0] + Ji[1, 1] + 1j * (Ji[0, 1] - Ji[1, 0]))
                Pm = 0.5 * (Ji[0, 0] - Ji[1, 1] - 1j * (Ji[0, 1] + Ji[1, 0]))
                Pp = 0.5 * (Ji[0, 0] - Ji[1, 1] + 1j * (Ji[0, 1] + Ji[1, 0]))
                H11[i, j] += Qm; H22[i, j] += Qp
                H12[i, j] += Pp; H21[i, j] += Pm
                # on-site classical local field: the q=0 sum, NO phase
                Ji0 = rots[i].T @ J0[i, j] @ rots[j]
                H11[i, i] -= S[j] * np.real(Ji0[2, 2])
                H22[i, i] -= S[j] * np.real(Ji0[2, 2])

        g_metric = np.diag([1.0] * n + [-1.0] * n)
        return g_metric @ np.block([[H11, H12], [H21, H22]])

    def _ewald_h_stack(self, q_grid):
        """(Nq, 2N, 2N) dipolar contribution, or None when Ewald is not requested."""
        if self._ewald_spec() is None:
            return None
        return np.array([self._ewald_nambu(q) for q in np.asarray(q_grid, float)])

    def max_imaginary_energy(
        self,
        n_q: int = 16,
        seed: int = 0,
        q_cart: Optional[npt.NDArray[np.float64]] = None,
    ) -> float:
        """Largest |Im(omega)| over a sample of q -- a direct test that the current
        magnetic structure really is a classical energy MINIMUM.

        LSWT expanded about a non-minimum (a saddle, or a local minimum that is not
        the ground state) has imaginary magnon energies. The engine's per-q warnings
        for this were easy to miss and the imaginary part was then discarded, so a
        wrong ground state produced a plausible-looking spectrum. This returns the
        number so callers can act on it.

        `q_cart`: extra q-points to test, in addition to the random sample. Callers
        SHOULD pass the q-path they are about to plot. A broad instability shows up
        anywhere, but an instability confined to particular q -- e.g. SW23, where the
        acoustic branch collapses only at the magnetic satellites -- is easily missed
        by a purely random sample, and that is exactly the q a user is looking at.

        Cheap: it diagonalizes the SAME H(q) the dispersion uses.
        """
        return self.stability_report(n_q=n_q, seed=seed, q_cart=q_cart)["max_imag"]

    def stability_report(
        self,
        n_q: int = 16,
        seed: int = 0,
        q_cart: Optional[npt.NDArray[np.float64]] = None,
    ) -> Dict[str, float]:
        """Imaginary-mode diagnostics: {'max_imag', 'band_scale', 'relative'}.

        `relative` = max|Im(omega)| / bandwidth is the number to threshold on, NOT the
        raw meV. An absolute cutoff cannot separate a real instability from numerical
        noise across models whose energy scales differ by orders of magnitude: SW07's
        120-degree kagome carries ~1e-3 meV of noise on a 2.4 meV band (the Goldstone
        modes sit at omega ~ 0, where the Bogoliubov problem is singular and the
        minimizer's residual error leaks into the imaginary part), while SW23's genuine
        instability is 1.5 meV on a 59 meV band. In meV those are only 3 orders apart;
        relative to the bandwidth they are 5e-4 vs 2.5e-2 -- cleanly separated.
        """
        if self.HMat_sym is None:
            return {"max_imag": 0.0, "band_scale": 0.0, "relative": 0.0}
        B = reciprocal_b_matrix(np.asarray(self.sm.unit_cell(), dtype=float))
        rng = np.random.default_rng(seed)
        q_sample = rng.uniform(-0.5, 0.5, size=(int(n_q), 3)) @ B
        if q_cart is not None and len(q_cart) > 0:
            extra = np.asarray(q_cart, dtype=float).reshape(-1, 3)
            if len(extra) > 256:
                idx = np.linspace(0, len(extra) - 1, 256).astype(int)
                extra = extra[idx]
            q_sample = np.vstack([q_sample, extra])

        h = self._build_h_stack(q_sample, float(self.spin_magnitude))
        worst, band = 0.0, 0.0
        for m in h:
            try:
                ev = la.eigvals(m)
            except Exception:
                continue
            worst = max(worst, float(np.max(np.abs(np.imag(ev)))))
            band = max(band, float(np.max(np.abs(np.real(ev)))))
        return {
            "max_imag": worst,
            "band_scale": band,
            "relative": worst / band if band > 1e-12 else 0.0,
        }

    def _build_h_stack(
        self,
        q_grid: npt.NDArray[np.float64],
        S: float,
        negate: bool = False,
    ) -> npt.NDArray[np.complex128]:
        """Evaluate the dynamical matrix H(q) over the whole q-grid at once.

        Vectorized replacement for the per-q ``lambdify`` loop. The exact-H
        Fortran path needs a dense ``(Nq, 2N, 2N)`` stack of Hamiltonians; the
        old code built it with a Python ``for`` loop calling a lambdified matrix
        function once per q-point, which becomes the dominant cost for large
        unit cells (the Fortran kernel is far faster than this Python build).

        Here we lambdify the *flattened* list of the 4N^2 matrix entries and
        call it once with the q-components as arrays, so SymPy/NumPy evaluates
        the q-dependence across all Nq points with vectorized ufuncs. The only
        Python-level loop left is over the matrix entries (4N^2, independent of
        Nq) to broadcast constant entries up to the grid length — assembling a
        ragged mix of scalars and ``(Nq,)`` arrays into one dense block.

        ``list(self.HMat_sym)`` iterates row-major, matching the ``[[...]]``
        layout lambdify would otherwise emit, so the result is bit-identical to
        the per-q path.
        """
        # Lambdify once per symbolic matrix: this method is called by the
        # stability guard on every run, by compute_corrections, and per Fortran
        # call -- re-running CSE + lambdify each time dominated the cost for
        # large cells. The cache key is the HMat_sym object itself (replaced on
        # regeneration); parameter VALUES are passed as arguments, so they may
        # change freely without invalidating the compiled function.
        cached = getattr(self, "_h_stack_cache", None)
        if cached is None or cached[0] is not self.HMat_sym:
            lam = [s for s in self.full_symbol_list if isinstance(s, sp.Symbol)]
            entry_func = lambdify(lam, list(self.HMat_sym), modules=["numpy"], cse=True)
            self._h_stack_cache = (self.HMat_sym, entry_func)
        entry_func = self._h_stack_cache[1]
        base = [S] + list(self.hamiltonian_params)
        two_n = 2 * int(self.nspins)
        nq = len(q_grid)
        qx, qy, qz = q_grid[:, 0], q_grid[:, 1], q_grid[:, 2]
        if negate:
            qx, qy, qz = -qx, -qy, -qz
        entries = entry_func(qx, qy, qz, *base)
        flat = np.empty((two_n * two_n, nq), dtype=np.complex128)
        for k, col in enumerate(entries):
            flat[k] = col  # scalar entries broadcast to (Nq,); array entries copy
        stack = np.ascontiguousarray(flat.T.reshape(nq, two_n, two_n))
        dip = self._ewald_h_stack(q_grid if not negate else -np.asarray(q_grid, float))
        if dip is not None:
            stack = stack + dip
        return stack

    def _calculate_dispersion_fortran(
        self,
        q_vectors_list: List[npt.NDArray[np.float64]],
    ) -> Optional[DispersionResult]:
        """Opt-in dispersion via the external fMagCalc Fortran backend.

        Builds the exact lambdified H(q) stack (the same H the NumPy path
        diagonalizes) and runs the Fortran zgeev/OpenMP loop. Returns a
        DispersionResult, or None to fall back to NumPy. Every fallback is
        logged at WARNING level since the Fortran backend was explicitly
        requested.
        """
        try:
            import fmagcalc
        except Exception:
            located = self._locate_fmagcalc()
            try:
                import fmagcalc
            except Exception:
                logger.warning(
                    "backend='fortran' requested but the fMagCalc package could not be "
                    "imported; using NumPy instead. Install it with `pip install "
                    "/path/to/fMagCalc` or set FMAGCALC_PATH. (searched: %s)",
                    located or "PYTHONPATH only",
                )
                return None
        if getattr(fmagcalc, "backend", None) != "ctypes":
            logger.warning(
                "backend='fortran' requested but fMagCalc's compiled library is not "
                "available (reinstall with `pip install /path/to/fMagCalc`, or run "
                "`cmake -S . -B build && cmake --build build` in a source checkout); "
                "using NumPy instead."
            )
            return None
        try:
            S = float(self.spin_magnitude)
            q_grid = np.array([np.asarray(q, dtype=float) for q in q_vectors_list])

            h_plus = self._build_h_stack(q_grid, S)

            energies, info = fmagcalc.run_dispersion(h_plus)
            logger.info("Dispersion computed via fMagCalc Fortran backend (exact-H path).")
            return DispersionResult(q_vectors=q_grid, energies=energies)
        except Exception:
            logger.exception("fMagCalc dispersion backend failed; falling back to NumPy.")
            return None

    def _calculate_sqw_fortran(
        self,
        q_vectors_list: List[npt.NDArray[np.float64]],
        ion_list: Optional[List[str]],
    ) -> Optional[SqwResult]:
        """Opt-in S(q,w) via the external fMagCalc Fortran backend.

        Returns a SqwResult, or None if the backend (the `fmagcalc` package and
        its compiled ctypes library) is unavailable or errors — the caller then
        falls back to the NumPy path. This keeps pyMagCalc fully functional
        without fMagCalc installed. Because the user explicitly asked for the
        Fortran backend, every fallback is logged at WARNING level.
        """
        try:
            import fmagcalc
        except Exception:
            located = self._locate_fmagcalc()  # try env var / sibling checkout
            try:
                import fmagcalc
            except Exception:
                logger.warning(
                    "backend='fortran' requested but the fMagCalc package could not be "
                    "imported; using NumPy instead. Install it with `pip install "
                    "/path/to/fMagCalc` or set FMAGCALC_PATH. (searched: %s)",
                    located or "PYTHONPATH only",
                )
                return None
        if getattr(fmagcalc, "backend", None) != "ctypes":
            logger.warning(
                "backend='fortran' requested but fMagCalc's compiled library is not "
                "available (reinstall with `pip install /path/to/fMagCalc`, or run "
                "`cmake -S . -B build && cmake --build build` in a source checkout); "
                "using NumPy instead."
            )
            return None
        try:
            from .form_factors import get_form_factor

            n = int(self.nspins)
            S = float(self.spin_magnitude)
            Ud = np.asarray(self.Ud_numeric, dtype=np.complex128)
            q_grid = np.array([np.asarray(q, dtype=float) for q in q_vectors_list])

            # Feed the EXACT lambdified Hamiltonian (what the NumPy path
            # diagonalizes), not the bond-reconstructed one. The reconstruction
            # differs from the exact H only at the ~1e-14 level, but at
            # degenerate eigenspaces KKdMatrix is ill-conditioned and that tiny
            # perturbation flips it onto a different eigenbasis, producing
            # spurious vertical streaks in S(Q,w). Using the exact H keeps the
            # Fortran result aligned with NumPy. (run_sqw_model — building H(q)
            # in Fortran from bonds — remains available for max speed on systems
            # without degeneracies.)
            h_plus = self._build_h_stack(q_grid, S)
            h_minus = self._build_h_stack(q_grid, S, negate=True)

            ff = np.ones((len(q_grid), n))
            if ion_list:
                for iq, q in enumerate(q_grid):
                    qmag = float(np.linalg.norm(q))
                    ff[iq] = [get_form_factor(ion_list[i], qmag) for i in range(n)]

            res = fmagcalc.run_sqw(h_plus, h_minus, Ud, ff, S, q_grid)
            logger.info("S(q,w) computed via fMagCalc Fortran backend (exact-H path).")
            return SqwResult(
                q_vectors=q_grid,
                energies=res["energies"],
                intensities=res["intensities"],
            )
        except Exception:
            logger.exception("fMagCalc backend failed; falling back to NumPy.")
            return None

    def calculate_sqw(
        self,
        q_vectors: Union[List[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        backend: str = "numpy",
        satellites: Optional[bool] = None,
        temperature: Optional[float] = None,
        domains=None,
        cross_section: str = "perp",
    ) -> Optional[SqwResult]:
        """
        Calculate the dynamical structure factor S(q,w) over a list of q-points.

        Args:
            q_vectors (Union[List[npt.NDArray[np.float64]], npt.NDArray[np.float64]]):
                A list or NumPy array of momentum vectors q = [qx, qy, qz].
                Each vector should be a 1D array/list of 3 numbers.
            backend (str): "numpy" (default) uses the in-process multiprocessing
                path. "fortran" uses the external fMagCalc backend if available,
                transparently falling back to NumPy otherwise.
            satellites (Optional[bool]): For a single-k structure, compute the
                full incommensurate cross-section: the three channels at
                q-k, q, q+k with the rotating-frame projection tensors (Toth &
                Lake 2015; SpinW/Sunny convention). The result then has
                3*nspins modes per q, channel-major [q-k | q | q+k]. Defaults
                to True when a single-k structure is active (that IS the
                physical cross-section), False otherwise.
            temperature (Optional[float]): Sample temperature in Kelvin. When
                set (> 0), every mode intensity is multiplied by the thermal
                Bose prefactor |1/(1 - exp(-E/kT))| (Sunny's kT option /
                SpinW sw_egrid 'T'). Default None = T -> 0 (bare LSWT).
            domains: Magnetic/structural domain (twin) averaging spec — see
                _parse_domains. The result is the weighted average of the
                cross-sections of each rotated crystal, computed by sampling
                the unrotated model at R^T q (exact for 'perp'/'trace'). Modes
                are concatenated per domain: energies/intensities get
                n_domains * n_modes columns. Not supported together with a
                lab-frame component cross_section.
            cross_section (str): 'perp' (default, unpolarized (delta_ab -
                q^_a q^_b) contraction), 'trace', or a lab-frame tensor
                component like 'xx', 'zz', 'xy' (real part; off-diagonal
                components are signed).

        Returns:
            Optional[SqwResult]: Object containing q_vectors, energies, and intensities.
        """
        # Validate the cross-section spec HERE: the pool workers swallow
        # exceptions into NaN results, so a typo would otherwise produce an
        # all-NaN map instead of an error.
        cs_norm = (cross_section or "perp").lower()
        _CS_NAMED = ("perp", "trace", "chiral", "sf+", "sf-", "sf_plus", "sf_minus")
        if cs_norm not in _CS_NAMED and not (
                len(cs_norm) == 2 and set(cs_norm) <= set("xyz")):
            raise ValueError(
                f"Unknown cross_section '{cross_section}'. Use 'perp', 'trace', "
                f"'chiral', 'sf+'/'sf-' (polarized, P || q), or a component like "
                f"'xx', 'yy', 'zz', 'xy'.")

        # Convert q_vectors to list of arrays if it's a 2D array
        if isinstance(q_vectors, np.ndarray) and q_vectors.ndim == 2:
            q_vectors_list = [q_vectors[i, :] for i in range(q_vectors.shape[0])]
        elif isinstance(q_vectors, list):
            q_vectors_list = [np.array(q) for q in q_vectors]
        else:
             logger.error("Invalid input for q_vectors. Must be list of arrays or 2D array.")
             return None

        domain_list = _parse_domains(domains)
        if domain_list is not None:
            cs = (cross_section or "perp").lower()
            if cs not in ("perp", "trace", "chiral", "sf+", "sf-", "sf_plus", "sf_minus"):
                # A lab-frame component of a rotated crystal is not the same
                # component of the unrotated one; rotating q alone would be
                # silently wrong.
                raise ValueError(
                    "domains averaging supports only cross_section 'perp' or "
                    f"'trace', not '{cross_section}'.")
            logger.info(f"Averaging over {len(domain_list)} domains...")
            energies_parts = []
            intensities_parts = []
            for R, weight in domain_list:
                q_rot = [R.T @ np.asarray(q, dtype=float) for q in q_vectors_list]
                res_d = self._calculate_sqw_single_domain(
                    q_rot, backend, satellites, cross_section)
                if res_d is None:
                    logger.error("Domain S(q,w) calculation failed; aborting average.")
                    return None
                energies_parts.append(res_d.energies)
                intensities_parts.append(res_d.intensities * weight)
            result = SqwResult(
                q_vectors=np.array([np.asarray(q, dtype=float) for q in q_vectors_list]),
                energies=np.concatenate(energies_parts, axis=1),
                intensities=np.concatenate(intensities_parts, axis=1),
            )
        else:
            result = self._calculate_sqw_single_domain(
                q_vectors_list, backend, satellites, cross_section)

        if result is not None and temperature is not None and temperature > 0:
            result.intensities = result.intensities * thermal_bose_prefactor(
                result.energies, temperature)
        return result

    def _calculate_sqw_single_domain(
        self,
        q_vectors_list: List[npt.NDArray[np.float64]],
        backend: str,
        satellites: Optional[bool],
        cross_section: str = "perp",
    ) -> Optional[SqwResult]:
        """Single-orientation S(q,w) engine behind calculate_sqw (which layers
        domain averaging and the thermal factor on top)."""
        logger.info("Starting S(q,w) calculation...")
        start_t: float = timeit.default_timer()

        # Input Validation
        if self.HMat_sym is None or self.Ud_numeric is None:
            logger.error("Symbolic matrices not initialized. Cannot calculate S(q,w).")
            return None

        # Resolve the single-k (incommensurate) mode. k_case 1 (k integer)
        # means the three channels coincide; use the commensurate path.
        single_k_active = self.k_cart is not None and self.k_case in (2, 3)
        if satellites is None:
            satellites = single_k_active
        use_single_k = bool(satellites) and single_k_active

        ion_list = None
        if hasattr(self.sm, 'ion_list'):
            ion_list = self.sm.ion_list()
        # Per-site spin magnitudes: the S(q,w) prefactor is sqrt(S_i/2) PER SITE, so a
        # mixed-spin model needs the list, not the single reference S. None (or a
        # uniform list) reproduces the old scalar path exactly.
        spin_mags = None
        try:
            mags = self.sm.spin_magnitudes()
            if mags and len(mags) == self.nspins:
                spin_mags = [float(m) for m in mags]
        except Exception:
            spin_mags = None

        # Opt-in Fortran backend (falls back to NumPy on any unavailability).
        if backend == "fortran":
            if use_single_k:
                logger.info("Single-k S(q,w) not available in the Fortran backend; using NumPy.")
            elif (cross_section or "perp").lower() != "perp":
                logger.info(
                    f"cross_section='{cross_section}' not available in the "
                    "Fortran backend (it computes the perp contraction); using NumPy.")
            elif spin_mags is not None and len(set(np.round(spin_mags, 12))) > 1:
                # The Fortran kernel applies a single global sqrt(S/2); with mixed
                # spins that makes every relative intensity wrong by sqrt(S_i/S_ref).
                logger.warning(
                    "Mixed spin magnitudes detected; the Fortran backend applies a "
                    "single global sqrt(S/2) prefactor and would give wrong relative "
                    "intensities. Using NumPy for S(q,w).")
            else:
                fortran_result = self._calculate_sqw_fortran(q_vectors_list, ion_list)
                if fortran_result is not None:
                    if self.supercell_ncells > 1:
                        fortran_result.intensities = (
                            fortran_result.intensities / self.supercell_ncells)
                    return fortran_result

        # Dipolar (Ewald) contribution at +q and -q, precomputed here (see dispersion).
        q_arr = np.array(q_vectors_list, dtype=float)
        dip_p = self._ewald_h_stack(q_arr)
        dip_m = self._ewald_h_stack(-q_arr)
        dip_pairs = None if dip_p is None else [
            (dip_p[i], dip_m[i]) for i in range(len(q_vectors_list))]
        if dip_pairs is not None and use_single_k:
            raise NotImplementedError(
                "Ewald dipole-dipole is not yet supported together with a single-k "
                "(rotating-frame) structure: the three q +/- k channels each need "
                "their own A(q). Use a magnetic_supercell instead, or "
                "dipole_dipole.method: truncated.")

        if use_single_k:
            logger.info(
                "Computing incommensurate S(q,w): three channels q-k, q, q+k "
                f"(k={self.k_rlu.tolist()} RLU, k_case={self.k_case})."
            )
            worker_func = process_calc_Sqw_single_k
            task_args = [
                (
                    self.Ud_numeric,
                    q,
                    self.nspins,
                    self.spin_magnitude,
                    list(self.hamiltonian_params),
                    ion_list,
                    self.k_cart,
                    self.spiral_axis,
                    self.k_case,
                    cross_section,
                    spin_mags,
                )
                for q in q_vectors_list
            ]
        else:
            worker_func = process_calc_Sqw
            task_args = [
                (
                    self.Ud_numeric,
                    q,
                    self.nspins,
                    self.spin_magnitude,
                    list(self.hamiltonian_params),
                    ion_list,
                    cross_section,
                    spin_mags,
                    None if dip_pairs is None else dip_pairs[iq],
                )
                for iq, q in enumerate(q_vectors_list)
            ]

        try:
             with Pool(
                processes=os.cpu_count(),
                initializer=_init_worker,
                initargs=(self.HMat_sym, [s for s in self.full_symbol_list if isinstance(s, sp.Symbol)]),
            ) as pool:
                # Use imap to report progress with tqdm. imap preserves order.
                results = list(
                    tqdm(
                        pool.imap(worker_func, task_args),
                        total=len(task_args),
                        desc="S(q,w)",
                        unit="q-point",
                    )
                )
                
             q_vectors_out, energies_out, intensities_out = zip(*results)
             
        except Exception:
            logger.exception("Error unpacking results from parallel processing.")
            return None

        num_failures = sum(np.isnan(en).any() for en in energies_out)
        if num_failures > 0:
            logger.warning(
                f"S(q,w) calculation failed for {num_failures} out of {len(q_vectors_list)} q-points. Check logs for details."
            )

        end_time: float = timeit.default_timer()
        logger.info(
            f"Run-time for S(q,w) calculation: {np.round((end_time - start_t) / 60, 2)} min."
        )

        intensities_arr = np.array(intensities_out)
        if self.supercell_ncells > 1:
            # Per-chemical-cell normalization (Sunny/SpinW convention).
            intensities_arr = intensities_arr / self.supercell_ncells

        return SqwResult(
            q_vectors=np.array(q_vectors_out),
            energies=np.array(energies_out),
            intensities=intensities_arr
        )

    def calculate_powder_average(
        self,
        q_magnitudes: Union[List[float], npt.NDArray[np.float64]],
        num_samples: int = 100,
        backend: str = "numpy",
        temperature: Optional[float] = None,
        cross_section: str = "perp",
    ) -> Optional[SqwResult]:
        """
        Calculate the powder-averaged dynamic structure factor S(|q|, w).

        For each momentum magnitude |q|, it averages S(q, w) over a sphere of radius |q|.
        Optimized to use batch processing for maximum parallel efficiency.

        Domain (twin) averaging is deliberately NOT applied here: the full
        spherical average is invariant under any rigid rotation of the
        crystal, so domains would multiply the cost without changing the
        result.

        Args:
            q_magnitudes (Union[List[float], npt.NDArray[np.float64]]):
                A list or NumPy array of momentum magnitudes |q|.
            num_samples (int): Number of points to sample on the sphere for each |q|.
            temperature (Optional[float]): Sample temperature in Kelvin; applies
                the Bose prefactor per (q, mode) BEFORE the spherical average
                (see calculate_sqw).
            cross_section (str): Contraction of the spin-correlation tensor
                (see calculate_sqw).

        Returns:
            Optional[SqwResult]: Object containing q_magnitudes (as vectors [|q|, 0, 0]),
                                 energies, and averaged intensities.
        """
        if isinstance(q_magnitudes, (list, tuple)):
            q_mags = np.array(q_magnitudes)
        else:
            q_mags = q_magnitudes

        logger.info(f"Preparing batch powder average for {len(q_mags)} magnitudes (samples/mag={num_samples})...")
        start_t = timeit.default_timer()

        all_q_vectors = []
        segment_sizes = []

        # 1. Generate all Q-vectors upfront (shared Fibonacci-sphere construction;
        # a |Q| ~ 0 shell collapses to the single point [0, 0, 0]).
        for q_mag in q_mags:
            vectors = fibonacci_sphere_points(float(q_mag), num_samples)
            all_q_vectors.extend(vectors)
            segment_sizes.append(len(vectors))
        
        total_points = len(all_q_vectors)
        logger.info(f"Starting batch S(q,w) calculation for {total_points} total q-points...")

        # 2. Run single batch calculation
        # This maximizes CPU usage by keeping the worker pool alive for the entire duration
        res = self.calculate_sqw(
            all_q_vectors, backend=backend,
            temperature=temperature, cross_section=cross_section)
        
        if res is None:
            logger.error("Batch calculation failed.")
            return None

        # 3. Aggregate results
        all_avg_intensities = []
        all_energies = []
        
        current_idx = 0
        for count in segment_sizes:
            # Slice the results for this magnitude
            chunk_intensities = res.intensities[current_idx : current_idx + count]
            chunk_energies = res.energies[current_idx : current_idx + count]
            
            # Average
            # Note: For energies, LSWT usually gives same energies for all directions in isotropic cases,
            # but for anisotropic systems they vary. Conventionally for powder, we might just report
            # the average, or if users plot dispersion, they might expect specific modes.
            # Here we follow the previous logic: mean of energies (and intensities).
            
            # Warning: averaging energies of different modes mixing might be physically ambiguous 
            # if modes cross, but standard practice without specific tracking.
            if chunk_intensities.size > 0:
                avg_i = np.nanmean(chunk_intensities, axis=0)
                avg_e = np.nanmean(chunk_energies, axis=0)
                all_avg_intensities.append(avg_i)
                all_energies.append(avg_e)
            else:
                all_avg_intensities.append(np.full(self.nspins, np.nan))
                all_energies.append(np.full(self.nspins, np.nan))
                
            current_idx += count

        end_time = timeit.default_timer()
        logger.info(
            f"Run-time for powder average calculation: {np.round((end_time - start_t) / 60, 2)} min."
        )

        # Return result with q_vectors as [q_mag, 0, 0] for plotting consistency
        q_vectors_out = np.column_stack((q_mags, np.zeros_like(q_mags), np.zeros_like(q_mags)))

        return SqwResult(
            q_vectors=q_vectors_out,
            energies=np.array(all_energies),
            intensities=np.array(all_avg_intensities)
        )


    def save_results(self, filename: str, results_dict: Dict[str, Any]):
        """
        Save calculation results to a compressed NumPy (.npz) file.

        Args:
            filename (str): The name of the file to save the results to.
                            '.npz' extension is recommended.
            results_dict (Dict[str, Any]): A dictionary where keys are string
                names (e.g., 'q_vectors', 'energies', 'intensities') and
                values are the corresponding data (e.g., NumPy array, list/tuple
                of NumPy arrays). Sequences of arrays will be saved directly.

        Raises:
            TypeError: If results_dict is not a dictionary.
            ValueError: If filename is empty.
            IOError: If there is an error writing the file.
            Exception: For other potential errors during saving.
        """
        if not isinstance(results_dict, dict):
            raise TypeError("results_dict must be a dictionary.")
        if not filename:
            raise ValueError("filename cannot be empty.")

        logger.info(f"Saving results to '{filename}'...")
        try:
            # Pass the dictionary directly to savez_compressed
            # It handles saving sequences of arrays appropriately.
            np.savez_compressed(filename, **results_dict)
            logger.info(f"Results successfully saved to '{filename}'.")
        except (IOError, OSError) as e:
            logger.error(f"Failed to save results to '{filename}': {e}")
            raise IOError(f"File saving failed: {e}") from e
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred while saving results to '{filename}'."
            )
            raise

    def _extract_classical_quadratic(self, params, nspins, nspins_ouc, S_val):
        """Extract the classical energy as a quadratic form in the Cartesian spin
        components: E(m) = 0.5 m^T H m + b^T m + c, where m stacks the 3N
        components of the N unit-cell spins. Valid for any bilinear/single-ion
        spin Hamiltonian (Heisenberg, DM, anisotropic exchange, single-ion,
        Zeeman) — i.e. quadratic in the spin components.

        Returns (H (3N,3N), b (3N,), c (scalar)) as NumPy arrays, or None if the
        energy is NOT quadratic in the components (verified numerically), in
        which case the caller falls back to the symbolic lambdify path. The
        OUC->UC component sharing (each over-unit-cell spin is a copy of its
        unit-cell parent) is built into H, so the Hessian/gradient accumulate
        all bond contributions onto the unit-cell variables.
        """
        try:
            d = 3 * nspins
            comps = sp.symbols(f"_mc0:{d}", real=True)
            Svec = [
                sp.Matrix(comps[3 * (i % nspins): 3 * (i % nspins) + 3])
                for i in range(nspins_ouc)
            ]
            Ec = self.sm.Hamiltonian(Svec, self.params_sym)  # type: ignore
            Ec = Ec.subs(self.S_sym, S_val).subs(list(zip(self.params_sym_flat, params)))
            Ec = Ec.as_real_imag()[0]
            Ef = lambdify(comps, Ec, modules="numpy", cse=True)

            # Extract (H, b, c) by NUMERIC probing of Ef rather than a symbolic
            # Hessian (which is the bulk of the setup cost). For an exact
            # quadratic these identities hold with step h=1 (no truncation error):
            #   c     = E(0)
            #   b_i   = (E(e_i) - E(-e_i)) / 2
            #   H_ii  = E(e_i) + E(-e_i) - 2c
            #   H_ij  = E(e_i+e_j) - E(e_i) - E(e_j) + c     (i<j)
            eye = np.eye(d)
            c = float(Ef(*np.zeros(d)))
            Ep = np.array([float(Ef(*eye[i])) for i in range(d)])
            Em = np.array([float(Ef(*(-eye[i]))) for i in range(d)])
            b = (Ep - Em) / 2.0
            H = np.zeros((d, d))
            H[np.diag_indices(d)] = Ep + Em - 2.0 * c
            for i in range(d):
                for j in range(i + 1, d):
                    Hij = float(Ef(*(eye[i] + eye[j]))) - Ep[i] - Ep[j] + c
                    H[i, j] = H[j, i] = Hij

            # Verify the energy really is quadratic: the reconstruction must match
            # a direct eval at random points. Mismatch => higher-order terms exist
            # (not a standard bilinear spin model) => signal symbolic fallback.
            rng = np.random.default_rng(0)
            for _ in range(4):
                mv = rng.uniform(-S_val, S_val, size=d)
                analytic = 0.5 * (mv @ (H @ mv)) + b @ mv + c
                if not np.isclose(analytic, float(Ef(*mv)), rtol=1e-7, atol=1e-7):
                    logger.info(
                        "Classical energy is not quadratic in spin components; "
                        "using symbolic path."
                    )
                    return None
            # Long-range dipolar (Ewald) term. The minimizer MUST optimize the same
            # Hamiltonian that LSWT diagonalizes -- otherwise it finds a ground state
            # that is not a minimum of the actual model, and the spectrum comes out
            # imaginary (exactly the g-tensor bug fixed earlier). A(q=0) is the full
            # lattice sum between sites of the cell, so it adds straight into H.
            if self._ewald_spec() is not None:
                from .ewald import exchange_from_A
                J0 = exchange_from_A(self._ewald_A(np.zeros(3)), self._ewald_g())
                for i in range(nspins):
                    for j in range(nspins):
                        H[3 * i:3 * i + 3, 3 * j:3 * j + 3] += np.real(J0[i, j])

            return H, b, c
        except Exception as e:
            logger.info(f"Quadratic energy extraction failed ({e}); using symbolic path.")
            return None

    def _minimize_batched(self, H, b, c, S, n, num_starts, seed,
                          max_iter=2000, tol=1e-8, topk=10):
        """Vectorized multistart minimization. Optimizes ALL `num_starts`
        simultaneously with projected-gradient descent on the unit spheres (one
        set of NumPy array ops for the whole batch — no per-start Python loop),
        then polishes the few lowest-energy candidates with scipy for precision.

        Requires the analytical quadratic energy E(m)=0.5 m^T H m + b^T m + c.
        Returns a scipy OptimizeResult (best found), or None to signal fallback.

        The batch is seeded, so the result is reproducible; momentum GD with the
        1/L step (L = S^2 * spectral radius of H) is scale-free. Polishing the
        top-k distinct candidates guards against imperfect batch convergence —
        the final energy/structure is whatever scipy refines them to.
        """
        try:
            from scipy.optimize import minimize as _sp_min
            rng = np.random.default_rng(seed)
            B, d = int(num_starts), 3 * n
            v = rng.normal(size=(B, n, 3))
            v /= np.linalg.norm(v, axis=2, keepdims=True)
            lam = float(np.max(np.abs(np.linalg.eigvalsh(H))))
            lr = 1.0 / (S * S * lam + 1e-12)
            beta = 0.9
            mom = np.zeros_like(v)
            for _ in range(max_iter):
                m = (S * v).reshape(B, d)
                gn = (S * (m @ H + b)).reshape(B, n, 3)        # dE/dn
                gtan = gn - np.sum(gn * v, axis=2, keepdims=True) * v  # tangent
                if np.max(np.abs(gtan)) < tol:
                    break
                mom = beta * mom + gtan
                v = v - lr * mom
                v /= np.linalg.norm(v, axis=2, keepdims=True)
            m = (S * v).reshape(B, d)
            E = 0.5 * np.sum(m * (m @ H), axis=1) + m @ b + c
            order = np.argsort(E)[:min(topk, B)]
            best = None
            for k in order:
                vi = v[k]
                th = np.arccos(np.clip(vi[:, 2], -1.0, 1.0))
                ph = np.arctan2(vi[:, 1], vi[:, 0])
                x0 = np.empty(2 * n)
                x0[0::2] = th
                x0[1::2] = ph
                r = _sp_min(_classical_energy_np, x0, jac=_classical_grad_np,
                            args=(H, b, c, S, n), method="L-BFGS-B")
                if best is None or r.fun < best.fun:
                    best = r
            return best
        except Exception as e:
            logger.info(f"Batched minimization failed ({e}); falling back to multistart.")
            return None

    def minimize_energy(
        self,
        params: Optional[List[float]] = None,
        x0: Optional[npt.NDArray[np.float64]] = None,
        method: str = "TNC",
        bounds: Optional[List[Tuple[float, float]]] = None,
        constraints: Optional[Dict] = None,
        num_starts: int = 1,
        n_workers: int = 1,
        early_stopping: int = 10,
        seed: Optional[int] = 0,
        batched: bool = False,
        **kwargs,
    ) -> Any:
        logger.info(
            f"Starting minimization (method={method}, num_starts={num_starts}, n_workers={n_workers}, early_stopping={early_stopping}, seed={seed})"
        )
        """
        Find the classical ground state magnetic structure by minimizing the energy.

        This method constructs the classical energy function from the symbolic
        Hamiltonian (defined by the spin_model_module) and minimizes it with
        respect to the spin orientation angles (theta, phi) for each spin in the unit cell.

        Args:
            params (Optional[List[float]]): Numerical Hamiltonian parameters. If None, uses self.hamiltonian_params.
            x0 (Optional[np.ndarray]): Initial guess for angles [theta_0, phi_0, theta_1, phi_1, ...].
                                       Length should be 2 * nspins.
            method (str): Minimization method (default: 'L-BFGS-B').
            bounds (Optional[List[Tuple]]): Bounds for variables.
            constraints (Optional[Dict]): Constraints for optimization.
            num_starts (int): Number of independent minimizations from random starts (default: 1).
                              If num_starts > 1 and x0 is provided, x0 is used as the first start.
            n_workers (int): Number of parallel workers for multistart (default: 1).
            early_stopping (int): Stop after N hits of the same minimum energy (default: 10).

        Returns:
            OptimizeResult: The result of the optimization (best result found).
        """
        if params is None:
            if self.hamiltonian_params is None:
                raise ValueError("Hamiltonian parameters must be provided either to __init__ or minimize_energy.")
            params = self.hamiltonian_params

        if early_stopping < 10:
            logger.warning(
                f"early_stopping is set to {early_stopping} (less than 10). "
                "This could lead to a wrong magnetic structure."
            )

        # 1. Model sizes
        nspins = len(self.sm.atom_pos())
        nspins_ouc = len(self.sm.atom_pos_ouc())
        S_val = self.spin_magnitude

        # 2. Energy + gradient for the optimizer. Prefer the analytical NumPy
        # form: the classical energy is a quadratic form in the spin components,
        # so (H,b,c) extracted once give energy/gradient as pure array ops (no
        # SymPy in the optimizer's hot loop — the dominant cost). `payload` is
        # what each multistart task carries to _minimize_worker; with the
        # analytical form it is just NumPy arrays (no per-task lambdify, and
        # picklable to parallel workers). The angle-based symbolic energy is only
        # built on the fallback path, so the common case skips it entirely.
        quad = self._extract_classical_quadratic(params, nspins, nspins_ouc, S_val)

        # Monte-Carlo / annealing ground-state search (SpinW `anneal` /
        # `optmagsteep`, Sunny `LocalSampler`). Random-multistart L-BFGS reliably
        # gets trapped on frustrated landscapes; annealing crosses barriers.
        if method in ("anneal", "monte_carlo", "steep", "optmagsteep"):
            if quad is None:
                raise ValueError(
                    f"method='{method}' needs the classical energy as a quadratic form, "
                    f"which could not be extracted for this model. Use a gradient "
                    f"method (e.g. L-BFGS-B) instead."
                )
            return self._minimize_anneal(
                *quad, S_val, nspins, method=method, num_starts=num_starts,
                x0=x0, seed=seed, **kwargs)

        if quad is not None:
            H_q, b_q, c_q = quad
            # Fast path: batched multistart (all starts vectorized on the spheres,
            # then scipy-polished). Only for plain gradient methods — basinhopping
            # / differential_evolution keep their dedicated scipy drivers.
            if batched and method not in ("basinhopping", "differential_evolution"):
                logger.info(
                    f"Minimizing via batched projected-gradient ({num_starts} starts, vectorized)..."
                )
                res = self._minimize_batched(H_q, b_q, c_q, S_val, nspins, num_starts, seed)
                if res is not None:
                    res.global_message = "batched projected-gradient multistart"
                    logger.info(f"Batched minimization complete: E_min={res.fun:.6f}")
                    return res
                logger.info("Batched minimization unavailable; using scipy multistart.")
            payload = ("np", H_q, b_q, c_q, S_val, nspins)
            logger.info("Using analytical NumPy classical energy (quadratic-form extraction).")
        else:
            # Fallback: symbolic angle-based energy + gradient (lambdified per
            # worker with cse=True). Build trig spin vectors in the unit-cell
            # angles, then differentiate.
            logger.info("Energy not quadratic; building symbolic angle-based energy/gradient...")
            theta_sym = sp.symbols(f"theta0:{nspins}", real=True)
            phi_sym = sp.symbols(f"phi0:{nspins}", real=True)
            S_vectors_ouc = []
            for i_ouc in range(nspins_ouc):
                th = theta_sym[i_ouc % nspins]
                ph = phi_sym[i_ouc % nspins]
                S_vectors_ouc.append(sp.Matrix([
                    self.S_sym * sp.sin(th) * sp.cos(ph),
                    self.S_sym * sp.sin(th) * sp.sin(ph),
                    self.S_sym * sp.cos(th),
                ]))
            opt_vars = []
            for i in range(nspins):
                opt_vars.extend([theta_sym[i], phi_sym[i]])
            E_sym_num = self.sm.Hamiltonian(S_vectors_ouc, self.params_sym).subs(self.S_sym, S_val)
            E_sym_num = E_sym_num.subs(list(zip(self.params_sym_flat, params))).as_real_imag()[0]
            Grad_sym_num = [sp.diff(E_sym_num, v).doit() for v in opt_vars]
            payload = ("sym", E_sym_num, Grad_sym_num, opt_vars)

        if bounds is None:
            # Bounds: theta [0, pi], phi [0, 2pi]
            bounds = []
            for _ in range(nspins):
                bounds.append((0, np.pi))     # Theta
                bounds.append((0, 2 * np.pi)) # Phi

        best_res = None
        best_energy = np.inf

        # Seed a LOCAL rng so the random multistart is reproducible: the ground
        # state (hence the rotations baked into HMat_sym) is otherwise different
        # every run, which both makes results non-reproducible AND defeats the
        # symbolic cache (its rotation signature never matches). A fixed default
        # seed gives the same set of starts each run; pass seed=None to opt back
        # into nondeterministic exploration. Using a local Generator avoids
        # perturbing global numpy random state elsewhere.
        rng = np.random.default_rng(seed)

        tasks = []
        for start_idx in range(num_starts):
            # 5. Determine initial guess for this start
            if start_idx == 0 and x0 is not None:
                current_x0 = x0
            else:
                # Default guess: Random to break symmetry
                current_x0 = rng.random(2 * nspins)
                # Scale theta [0, pi], phi [0, 2pi]
                for i in range(nspins):
                    current_x0[2 * i] *= np.pi
                    current_x0[2 * i + 1] *= 2 * np.pi
            
            # Unified task tuple for both sequential and parallel paths; both run
            # through _minimize_worker. dict(kwargs) gives each start its own copy
            # (some methods, e.g. basinhopping, pop keys from it).
            tasks.append((payload, current_x0, method, bounds, constraints, dict(kwargs)))

        if n_workers > 1:
            from multiprocessing import Pool

            best_res = None
            best_energy = np.inf
            hits = 0
            global_message = f"Maximum number of starts ({num_starts}) reached"

            with Pool(processes=n_workers) as pool:
                try:
                    # Ordered imap (not imap_unordered): with seeded starts this
                    # makes which degenerate ground state wins deterministic, so
                    # the baked rotations — and the symbolic cache — are stable
                    # run-to-run. (Completion-order selection would otherwise vary.)
                    for res in tqdm(pool.imap(_minimize_worker, tasks), total=len(tasks), desc="Minimization Starts"):
                        if res.success:
                            if np.isclose(res.fun, best_energy, atol=1e-6):
                                hits += 1
                                if res.fun < best_energy:
                                    best_energy = res.fun
                                    best_res = res
                            elif res.fun < best_energy:
                                best_energy = res.fun
                                best_res = res
                                hits = 1
                                logger.info(
                                    f"New best energy found: {best_energy:.6f} meV"
                                )

                            if early_stopping > 0 and hits >= early_stopping:
                                global_message = f"Stopped early because it found the same answer {early_stopping} times"
                                logger.info(global_message)
                                pool.terminate()
                                break
                except KeyboardInterrupt:
                    pool.terminate()
                    raise

        else:
            # Sequential execution (original logic with early stopping)
            hits = 0
            global_message = f"Maximum number of starts ({num_starts}) reached"
            for start_idx, task_args in enumerate(tqdm(tasks, desc="Minimization Starts")):
                # Same worker as the parallel path (unified energy/grad handling).
                res = _minimize_worker(task_args)

                if res.success:
                    if np.isclose(res.fun, best_energy, atol=1e-6):
                        hits += 1
                        if res.fun < best_energy:
                            best_energy = res.fun
                            best_res = res
                    elif res.fun < best_energy:
                        best_energy = res.fun
                        best_res = res
                        hits = 1
                        logger.info(
                            "New best energy found at start {}: {:.6f} meV".format(
                                start_idx + 1, best_energy
                            )
                        )
                    
                    if early_stopping > 0 and hits >= early_stopping:
                        global_message = f"Stopped early because it found the same answer {early_stopping} times"
                        logger.info(global_message)
                        break
                else:
                    logger.debug(
                        "Start {} failed to converge: {}".format(
                            start_idx + 1, res.message
                        )
                    )

        if best_res is None:
            logger.warning("All minimization starts failed.")
            # If all failed, we might want to return the last res if it exists,
            # but usually it's better to return something indicating failure.
            # However, for consistency with original behavior:
        if best_res is not None:
             best_res.global_message = global_message
             # Expose the multistart convergence evidence. `hits` is how many
             # independent starts reached the best energy; hits == 1 means the
             # minimum was seen exactly once, which is weak evidence that it is
             # the GLOBAL minimum -- the failure mode that silently produces
             # imaginary magnon energies downstream.
             best_res.hits = hits
             best_res.num_starts = num_starts
             if hits <= 1 and num_starts > 1:
                 logger.warning(
                     f"Minimization: the best energy ({best_energy:.6f} meV) was reached by "
                     f"only {hits} of {num_starts} starts. That is weak evidence it is the "
                     f"global minimum -- a local minimum here yields IMAGINARY magnon "
                     f"energies. Raise `num_starts` and `early_stopping`, and check the "
                     f"energy is reproducible across `seed`s."
                 )

        return best_res


# --- Plotting Helper Functions ---
def plot_dispersion_from_data(
    q_values: np.ndarray,
    energies_list: List[Optional[npt.NDArray[np.float64]]],
    title: str = "Spin Wave Dispersion",
    q_labels: Optional[List[str]] = None,
    q_ticks_positions: Optional[List[float]] = None,
    save_filename: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Plots spin wave dispersion from loaded data.

    Args:
        q_values (np.ndarray): Array of q-values for the x-axis (e.g., path length).
        energies_list (List[Optional[npt.NDArray[np.float64]]]): List of energy arrays,
            one for each q-point. Each array contains energies for different modes.
        title (str): Title of the plot.
        q_labels (Optional[List[str]]): Labels for specific q-points on the x-axis.
        q_ticks_positions (Optional[List[float]]): Positions for q_labels.
        save_filename (Optional[str]): If provided, saves the plot to this file.
        show_plot (bool): If True, displays the plot.
    """
    logger.info(f"Plotting dispersion: {title}")
    plt.figure(figsize=(8, 6))

    num_modes = 0
    if energies_list and energies_list[0] is not None:
        num_modes = energies_list[0].shape[0]

    for mode_idx in range(num_modes):
        mode_energies = []
        valid_q_values_for_mode = []
        for i, q_energy_array in enumerate(energies_list):
            if q_energy_array is not None and mode_idx < len(q_energy_array):
                mode_energies.append(q_energy_array[mode_idx])
                valid_q_values_for_mode.append(q_values[i])
        if mode_energies:
            plt.plot(valid_q_values_for_mode, mode_energies, marker=".", linestyle="-")

    plt.xlabel("q (path length or index)")
    plt.ylabel("Energy (meV)")
    plt.title(title)
    if q_labels and q_ticks_positions:
        plt.xticks(q_ticks_positions, q_labels)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename)
        logger.info(f"Dispersion plot saved to {save_filename}")
    if show_plot:
        plt.show()
    plt.close()


def plot_sqw_from_data(
    q_values: np.ndarray,
    energies_list: List[Optional[npt.NDArray[np.float64]]],
    intensities_list: List[Optional[npt.NDArray[np.float64]]],
    title: str = "S(q,w) Intensity Map",
    energy_max: Optional[float] = None,
    q_labels: Optional[List[str]] = None,
    q_ticks_positions: Optional[List[float]] = None,
    save_filename: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Plots S(q,w) intensity map from loaded data using a scatter plot.
    Size and color of points represent intensity.
    """
    logger.info(f"Plotting S(q,w) map: {title}")
    plt.figure(figsize=(10, 6))

    all_q = []
    all_e = []
    all_i = []

    for i, (q_val, e_arr, i_arr) in enumerate(
        zip(q_values, energies_list, intensities_list)
    ):
        if e_arr is not None and i_arr is not None:
            for energy, intensity in zip(e_arr, i_arr):
                if (
                    not np.isnan(energy)
                    and not np.isnan(intensity)
                    and intensity > 1e-3
                ):  # Threshold intensity
                    all_q.append(q_val)
                    all_e.append(energy)
                    all_i.append(intensity)

    if not all_q:
        logger.warning("No data to plot for S(q,w).")
        return

    scatter = plt.scatter(
        all_q, all_e, c=all_i, s=np.sqrt(all_i) * 20, cmap="viridis", alpha=0.7
    )  # Scale size for visibility
    plt.colorbar(scatter, label="Intensity (arb. units)")
    plt.xlabel("q (path length or index)")
    plt.ylabel("Energy (meV)")
    plt.title(title)
    if energy_max:
        plt.ylim(0, energy_max)
    if q_labels and q_ticks_positions:
        plt.xticks(q_ticks_positions, q_labels)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename)
        logger.info(f"S(q,w) map saved to {save_filename}")
    if show_plot:
        plt.show()
    plt.close()


def plot_magnetic_structure(
    atom_positions: np.ndarray,
    spin_angles: np.ndarray,
    title: str = "Magnetic Structure",
    save_filename: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Plots the magnetic structure in 3D.

    Args:
        atom_positions (np.ndarray): Nx3 array of atom coordinates.
        spin_angles (np.ndarray): 1D array of angles [theta_0, phi_0, theta_1, phi_1, ...].
        title (str): Plot title.
        save_filename (Optional[str]): Filename to save the plot.
        show_plot (bool): Whether to show the plot.
    """
    logger.info(f"Plotting magnetic structure: {title}")
    
    # Check dimensions
    nspins = len(atom_positions)
    if len(spin_angles) != 2 * nspins:
        logger.error(f"Mismatch: {nspins} atoms but {len(spin_angles)} angles (expected {2*nspins}).")
        return

    fig = plt.figure(figsize=(6, 6))
    # Ensure 3D projection is available
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    xs = atom_positions[:, 0]
    ys = atom_positions[:, 1]
    zs = atom_positions[:, 2]
    
    # Calculate vector components
    us = []
    vs = []
    ws = []
    
    for i in range(nspins):
        th = spin_angles[2*i]
        ph = spin_angles[2*i+1]
        
        sx = np.sin(th) * np.cos(ph)
        sy = np.sin(th) * np.sin(ph)
        sz = np.cos(th)
        
        us.append(sx)
        vs.append(sy)
        ws.append(sz)
        
    # Calculate distances to determine scale
    if nspins > 1:
        # Simple N^2 distance check
        min_dist = np.inf
        for i in range(nspins):
            for j in range(i + 1, nspins):
                d = np.linalg.norm(atom_positions[i] - atom_positions[j])
                if d < min_dist and d > 1e-6:
                    min_dist = d
        if np.isinf(min_dist):
            min_dist = 1.0 # Fallback
    else:
        min_dist = 1.0
        
    arrow_length = 0.4 * min_dist
    
    # Plot atoms
    ax.scatter(xs, ys, zs, c='k', s=50, label='Atoms')
    
    # Plot spins as quivers
    # length kwarg in quiver uses the data coordinates if correct normalization isn't applied, 
    # but matplotlib 3d quiver length is tricky.
    # Usually length is a multiplier of the vector magnitude.
    # Since we normalized inputs (us, vs, ws are unit vectors), 
    # the arrows will have length 'length'.
    ax.quiver(xs, ys, zs, us, vs, ws, length=arrow_length, normalize=True, color='r', linewidth=2, label='Spins')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Enforce Equal Aspect Ratio via cubic bounding box
    ranges = np.array([np.ptp(xs), np.ptp(ys), np.ptp(zs)])
    max_range = ranges.max()
    if max_range < 1e-6:
        max_range = 1.0
        
    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range * 0.6, mid_x + max_range * 0.6)
    ax.set_ylim(mid_y - max_range * 0.6, mid_y + max_range * 0.6)
    ax.set_zlim(mid_z - max_range * 0.6, mid_z + max_range * 0.6)
    
    # Optional: set box aspect to 1,1,1 so the axis box appears cubic
    try:
        ax.set_box_aspect([1,1,1])
    except:
        pass # Older matplotlib versions
        
    if save_filename:
        plt.savefig(save_filename)
        logger.info(f"Magnetic structure plot saved to {save_filename}")
        
        # Export JSON for the 3D GUI visualizer
        json_filename = save_filename.rsplit('.', 1)[0] + '.json'
        try:
            import json
            data = {
                "atoms": atom_positions.tolist(),
                "vectors": [[us[i], vs[i], ws[i]] for i in range(nspins)]
            }
            with open(json_filename, 'w') as f:
                json.dump(data, f)
            logger.info(f"Magnetic structure JSON saved to {json_filename}")
        except Exception as e:
            logger.error(f"Failed to save magnetic structure JSON: {e}")
        
    if show_plot:
        plt.show()
    plt.close()
