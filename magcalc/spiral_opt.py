"""Generalized spiral (single-k) ground-state optimization.

Finds the propagation vector k, the spin directions, and optionally the
rotation axis that minimize the classical spiral energy — the analogue of
Sunny's minimize_spiral_energy! / SpinW's optmagstr for single-k structures.

Conventions (matching Sunny SpiralEnergy.jl):
- Spins are parameterized as LAB-frame unit vectors of the cell-0 sites.
- The pair energy uses cell-offset phases: E_ij = S^2/2 * s_i . J_ij R(2*pi
  k.m_ij, axis) s_j, where m_ij is the integer cell offset of site j's image.
  With integer offsets the energy is exactly periodic in k -> k + integers,
  so k can always be reported wrapped to [0, 1).
- The optimized state is committed to the model as a 'single_k' structure
  with rotating-frame local_directions obtained by the standard back-rotation
  u_i = R(-2*pi*k.d_i, axis) s_i (d_i = fractional atom position), matching
  pyMagCalc's full-position phase convention (CLAUDE.md section 3).

The Luttinger-Tisza initial guess minimizes the smallest eigenvalue of the
Fourier exchange matrix J(k) (Sunny LuttingerTisza.jl), with an eta-smoothed
minimum for gradient friendliness.
"""

import logging
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
from numpy import linalg as la
from scipy.optimize import minimize

from .generic_model import (
    GenericSpinModel,
    interactions_to_numpy,
    rotation_about_axis,
    spiral_propagation_case,
)

logger = logging.getLogger(__name__)

MU_B = 5.788e-2  # meV/T, matching generic_model conventions
GAMMA = 2.0      # electron g-factor convention of _classical_energy_func


@dataclass
class SpiralOptResult:
    """Result of a spiral (single-k) energy minimization."""
    k_rlu: np.ndarray
    axis: np.ndarray
    thetas: List[float]          # rotating-frame angles committed to the model
    phis: List[float]
    spins_lab: np.ndarray        # (N, 3) lab-frame cell-0 spin directions
    energy: float                # total classical energy of one cell
    energy_per_site: float
    converged: bool
    message: str = ""
    scipy_result: Any = field(default=None, repr=False)


def _skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def _build_bond_arrays(model: GenericSpinModel, p_num):
    """Precompute lab-frame bond matrices and integer cell offsets.

    Returns (J_full (N, N_ouc, 3, 3), offsets (N_ouc, 3) integer cell offsets
    of each OUC site relative to its home cell, ouc_map (N_ouc,) site indices).
    J_full uses generic_model's conventions: J*I + skew'(D) + Kex with
    skew'(D) = [[0, Dz, -Dy], [-Dz, 0, Dx], [Dy, -Dx, 0]] so that
    s_i . skew'(D) . s_j = D . (s_i x s_j).
    """
    Jex_sym, DM_sym, Kex_sym = model._spin_interactions_lab(p_num)
    Jex, DM, Kex = interactions_to_numpy(Jex_sym, DM_sym, Kex_sym)
    N, N_ouc = Jex.shape

    J_full = np.zeros((N, N_ouc, 3, 3))
    eye = np.eye(3)
    for i in range(N):
        for j in range(N_ouc):
            d = DM[i, j]
            D_skew = np.array([
                [0.0, d[2], -d[1]],
                [-d[2], 0.0, d[0]],
                [d[1], -d[0], 0.0],
            ])
            J_full[i, j] = Jex[i, j] * eye + D_skew + Kex[i, j]

    apos = np.asarray(model.atom_pos(), dtype=float)
    apos_ouc = np.asarray(model.atom_pos_ouc(), dtype=float)
    inv_uc = la.inv(np.asarray(model.unit_cell(), dtype=float))
    ouc_map = np.arange(len(apos_ouc)) % N
    offsets_f = (apos_ouc - apos[ouc_map]) @ inv_uc
    offsets = np.round(offsets_f)
    if np.max(np.abs(offsets_f - offsets)) > 1e-6:
        logger.warning(
            "OUC positions are not integer cell translations of the unit cell "
            "sites (max deviation %.2e); spiral energy may be inaccurate.",
            float(np.max(np.abs(offsets_f - offsets))),
        )
    return J_full, offsets, ouc_map


def _parse_sia(model: GenericSpinModel, p_num):
    """Numeric single-ion anisotropy entries: list of (site_indices, D, axis)."""
    param_map = model._resolve_param_map(p_num)
    try:
        atom_labels = [a.get('label') for a in
                       model.config.get('crystal_structure', {}).get('atoms_uc', [])]
    except Exception:
        atom_labels = []
    label_to_idx = {lbl: i for i, lbl in enumerate(atom_labels) if lbl is not None}
    N = len(model.atom_pos())

    entries = []
    for interaction in model.interactions_config:
        if interaction.get('type') != 'sia':
            continue
        val = interaction.get('value')
        if isinstance(val, str):
            val = param_map.get(val)
        if val is None:
            continue
        try:
            D_sia = float(val)
        except (TypeError, ValueError):
            continue
        target_labels = interaction.get('atoms') or interaction.get('atom_labels')
        if target_labels:
            idx = [label_to_idx[l] for l in target_labels if l in label_to_idx]
        else:
            idx = list(range(N))
        axis = np.asarray(interaction.get('axis', [0, 0, 1]), dtype=float)
        nrm = la.norm(axis)
        if nrm > 1e-12:
            axis = axis / nrm
        entries.append((np.asarray(idx, dtype=int), D_sia, axis))
    return entries


def _resolve_field(model: GenericSpinModel, p_num) -> Optional[np.ndarray]:
    """External field vector (Tesla), following minimize_energy's conventions."""
    params_dict = model._resolve_param_map(p_num)
    if any(k in params_dict for k in ('Hx', 'Hy', 'Hz')):
        return np.array([float(params_dict.get('Hx', 0.0)),
                         float(params_dict.get('Hy', 0.0)),
                         float(params_dict.get('Hz', 0.0))])
    for name in ('H', 'H_mag', 'H_field'):
        if name in params_dict:
            h_val = params_dict[name]
            h_dir = params_dict.get('H_dir')
            if isinstance(h_val, (list, tuple, np.ndarray)):
                return np.asarray(h_val, dtype=float)
            if h_dir is not None and isinstance(h_dir, (list, tuple, np.ndarray)):
                return np.asarray(h_dir, dtype=float) * float(h_val)
            return np.array([0.0, 0.0, float(h_val)])
    return None


def _spins_from_angles(angles):
    theta = angles[0::2]
    phi = angles[1::2]
    st, ct = np.sin(theta), np.cos(theta)
    sph, cph = np.sin(phi), np.cos(phi)
    return np.stack([st * cph, st * sph, ct], axis=1)


def _reg(v):
    """Sunny's regularizer: blows up as spins align with the spiral axis."""
    return 1.0 / (1.0 - v**2 + 0.1)


class _SpiralEnergy:
    """Callable classical spiral energy over x = [k(3), (axis angles), spin angles]."""

    def __init__(self, J_full, offsets, ouc_map, sia_entries, H_vec, S_val,
                 axis, optimize_axis):
        self.J_full = J_full            # (N, N_ouc, 3, 3)
        self.offsets = offsets          # (N_ouc, 3)
        self.ouc_map = ouc_map          # (N_ouc,)
        self.sia = sia_entries
        self.H_vec = H_vec
        self.S = float(S_val)
        self.axis0 = axis
        self.optimize_axis = bool(optimize_axis)
        self.N = J_full.shape[0]
        self.lam = 0.0                  # regularization weight (stage 1)
        # index bookkeeping
        self.n_axis_pars = 2 if self.optimize_axis else 0

    def unpack(self, x):
        k = np.asarray(x[:3], dtype=float)
        if self.optimize_axis:
            tn, pn = x[3], x[4]
            axis = np.array([np.sin(tn) * np.cos(pn),
                             np.sin(tn) * np.sin(pn),
                             np.cos(tn)])
        else:
            axis = self.axis0
        spins = _spins_from_angles(np.asarray(x[3 + self.n_axis_pars:]))
        return k, axis, spins

    def x0(self, k, spins, axis=None):
        parts = [np.asarray(k, dtype=float)]
        if self.optimize_axis:
            a = self.axis0 if axis is None else np.asarray(axis, dtype=float)
            a = a / la.norm(a)
            parts.append(np.array([np.arccos(np.clip(a[2], -1, 1)),
                                   np.arctan2(a[1], a[0])]))
        angles = np.empty(2 * self.N)
        for i, s in enumerate(np.asarray(spins, dtype=float)):
            s = s / la.norm(s)
            angles[2 * i] = np.arccos(np.clip(s[2], -1, 1))
            angles[2 * i + 1] = np.arctan2(s[1], s[0])
        parts.append(angles)
        return np.concatenate(parts)

    def __call__(self, x):
        k, axis, spins = self.unpack(x)
        n = axis / (la.norm(axis) or 1.0)
        K = _skew(n)
        K2 = K @ K

        # R(2*pi k.m) per OUC site (phases depend only on the cell offset).
        phases = 2.0 * np.pi * (self.offsets @ k)      # (N_ouc,)
        s_ph, c_ph = np.sin(phases), np.cos(phases)
        # R[j] = I + sin*K + (1-cos)*K^2, shape (N_ouc, 3, 3)
        R = (np.eye(3)[None, :, :]
             + s_ph[:, None, None] * K[None, :, :]
             + (1.0 - c_ph)[:, None, None] * K2[None, :, :])

        s_ouc = spins[self.ouc_map]                    # (N_ouc, 3)
        Rs = np.einsum('jab,jb->ja', R, s_ouc)         # rotated neighbor spins
        E = 0.5 * self.S**2 * np.einsum('ia,ijab,jb->', spins, self.J_full, Rs)

        for idx, D_sia, ax in self.sia:
            proj = spins[idx] @ ax
            E += self.S**2 * float(np.sum(D_sia * proj**2))

        if self.H_vec is not None:
            E += GAMMA * MU_B * self.S * float(np.sum(spins @ self.H_vec))

        if self.lam != 0.0:
            E += self.lam * float(np.sum(_reg(spins @ n)))
        return E


def luttinger_tisza_guess(J_full, offsets, ouc_map, k_grid=16, refine=True):
    """Coarse-grid + local minimization of eigmin(J(k)) (Sunny LuttingerTisza.jl).

    Returns (k_guess, axis_guess or None). J(k) is the 3N x 3N Fourier exchange
    matrix with cell-offset phases; its minimal eigenvector suggests the
    polarization plane (axis = Re x Im of the site eigenvector).
    """
    N = J_full.shape[0]

    # Only scan k components along directions where interacting bonds extend.
    bonded = np.any(np.abs(J_full) > 1e-12, axis=(0, 2, 3))   # (N_ouc,)
    bond_offsets = offsets[bonded] if np.any(bonded) else offsets
    active = [d for d in range(3) if np.any(np.abs(bond_offsets[:, d]) > 0.5)]

    def J_of_k(k):
        ph = np.exp(2j * np.pi * (offsets @ k))        # (N_ouc,)
        Jk = np.zeros((3 * N, 3 * N), dtype=complex)
        for i in range(N):
            blocks = J_full[i] * ph[:, None, None]     # (N_ouc, 3, 3)
            for j in range(N):
                sel = (ouc_map == j)
                if np.any(sel):
                    Jk[3 * i:3 * i + 3, 3 * j:3 * j + 3] += blocks[sel].sum(axis=0)
        return 0.5 * (Jk + Jk.conj().T)

    def lt_energy(k, eta=0.0):
        w, V = la.eigh(J_of_k(k))
        if eta <= 0:
            return w[0], V[:, 0]
        ws = np.exp(-(w - w[0]) / eta)
        return float(np.sum(ws * w) / np.sum(ws)), V[:, 0]

    best_k, best_e, best_v = np.zeros(3), np.inf, None
    grid = np.arange(k_grid) / float(k_grid)
    axes_vals = [grid if d in active else np.array([0.0]) for d in range(3)]
    for kx, ky, kz in product(*axes_vals):
        k = np.array([kx, ky, kz])
        e, v = lt_energy(k)
        if e < best_e - 1e-12:
            best_e, best_k, best_v = e, k, v

    if refine and active:
        res = minimize(lambda k: lt_energy(k, eta=1e-8)[0], best_k, method='Nelder-Mead',
                       options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2000})
        if res.fun < best_e:
            best_k = res.x % 1.0
            _, best_v = lt_energy(best_k)

    axis_guess = None
    if best_v is not None:
        v0 = best_v[:3]
        ax = np.cross(np.real(v0), np.imag(v0))
        nrm = la.norm(ax)
        if nrm > 1e-6:
            axis_guess = ax / nrm
    return best_k % 1.0, axis_guess


def nice_k_string(k, max_den=12, tol=1e-4):
    """Human-readable k with near-rational components annotated."""
    parts = []
    for comp in np.asarray(k, dtype=float):
        frac = Fraction(comp).limit_denominator(max_den)
        if abs(float(frac) - comp) < tol and frac.denominator > 1:
            parts.append(f"{comp:.8f} (~{frac.numerator}/{frac.denominator})")
        else:
            parts.append(f"{comp:.8f}")
    return "[" + ", ".join(parts) + "]"


def optimize_spiral(model: GenericSpinModel, p_num, min_cfg: Dict[str, Any],
                    S_val: float = 1.0) -> SpiralOptResult:
    """Minimize the classical spiral energy over (k, spin directions[, axis]).

    Follows Sunny's minimize_spiral_energy!: unit spins + k optimized jointly,
    two-stage regularization pushing spins away from the rotation axis. The
    optimized structure is committed to the model as a 'single_k' rotating-frame
    magnetic structure (mag_struct_cfg + set_magnetic_structure), so a MagCalc
    instance built afterwards bakes in the optimized k.
    """
    ms_cfg = model.mag_struct_cfg if isinstance(getattr(model, 'mag_struct_cfg', None), dict) else {}
    axis = np.asarray(
        min_cfg.get('axis', ms_cfg.get('axis', [0, 0, 1])), dtype=float)
    axis = axis / (la.norm(axis) or 1.0)
    optimize_axis = bool(min_cfg.get('optimize_axis', False))

    J_full, offsets, ouc_map = _build_bond_arrays(model, p_num)
    sia_entries = _parse_sia(model, p_num)
    H_vec = _resolve_field(model, p_num)
    N = J_full.shape[0]

    energy = _SpiralEnergy(J_full, offsets, ouc_map, sia_entries, H_vec, S_val,
                           axis, optimize_axis)

    # --- Initial k: config k_init > lt_guess > structure k > random ---
    k_init = min_cfg.get('k_init')
    axis_guess = None
    if k_init is None and min_cfg.get('lt_guess', True):
        try:
            k_init, axis_guess = luttinger_tisza_guess(
                J_full, offsets, ouc_map, k_grid=int(min_cfg.get('k_grid', 16)))
            logger.info(f"Luttinger-Tisza k guess: {np.round(k_init, 6).tolist()}"
                        + (f", axis guess {np.round(axis_guess, 4).tolist()}"
                           if axis_guess is not None else ""))
        except Exception:
            logger.exception("Luttinger-Tisza guess failed; falling back.")
            k_init = None
    if k_init is None:
        k_init = ms_cfg.get('k', None)
    rng = np.random.default_rng(seed=int(min_cfg.get('seed', 42)))
    if k_init is None:
        k_init = rng.uniform(0, 1, 3)
    k_init = np.asarray(k_init, dtype=float)
    if optimize_axis and axis_guess is not None and 'axis' not in min_cfg:
        axis = axis_guess
        energy.axis0 = axis_guess

    num_starts = int(min_cfg.get('num_starts', 8))
    maxiter = int(min_cfg.get('maxiter', 5000))
    ftol = float(min_cfg.get('ftol', 1e-12))

    # angle bounds: theta in [0, pi], phi free; k and axis angles free
    bounds = [(None, None)] * 3
    if optimize_axis:
        bounds += [(None, None)] * 2
    for _ in range(N):
        bounds += [(0.0, np.pi), (None, None)]

    def _one_start(x0):
        # Stage 1: regularized (push spins off the axis), from Sunny.
        e0 = abs(energy(x0)) / max(N, 1)
        energy.lam = max(e0, 1e-3)
        res1 = minimize(energy, x0, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': maxiter, 'ftol': ftol})
        # Stage 2: true energy.
        energy.lam = 0.0
        res2 = minimize(energy, res1.x, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': maxiter, 'ftol': ftol})
        return res2

    best = None
    for start in range(max(num_starts, 1)):
        # in-plane-ish random spins, jittered k
        spins0 = []
        u_seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u0 = u_seed - np.dot(u_seed, axis) * axis
        u0 /= la.norm(u0)
        v0 = np.cross(axis, u0)
        for _ in range(N):
            ang = rng.uniform(0, 2 * np.pi)
            tilt = rng.normal(0, 0.2)
            s = np.cos(ang) * u0 + np.sin(ang) * v0 + tilt * axis
            spins0.append(s / la.norm(s))
        k0 = k_init if start == 0 else (k_init + rng.normal(0, 0.05, 3))
        x0 = energy.x0(k0, spins0)
        try:
            res = _one_start(x0)
        except Exception:
            logger.exception(f"Spiral optimization start {start} failed.")
            continue
        if best is None or res.fun < best.fun - 1e-12:
            best = res

    if best is None:
        raise RuntimeError("All spiral optimization starts failed.")

    k_opt, axis_opt, spins_opt = energy.unpack(best.x)
    axis_opt = axis_opt / la.norm(axis_opt)
    # Cell-offset phases make the energy exactly periodic in k, so wrapping to
    # [0, 1) is free. Components along directions with no *interacting* bonds
    # do not enter the energy; zero them for a clean report.
    bonded = np.any(np.abs(J_full) > 1e-12, axis=(2, 3))   # (N, N_ouc)
    bond_offsets = offsets[np.any(bonded, axis=0)]
    for d in range(3):
        if not np.any(np.abs(bond_offsets[:, d]) > 0.5):
            k_opt[d] = 0.0
    k_opt = k_opt % 1.0
    E_opt = float(best.fun)  # stage 2 runs with lam = 0 (true energy)

    # --- Commit to the model: rotating-frame local directions via the
    # full-position back-rotation u_i = R(-2*pi k.d_i, axis) s_i ---
    apos = np.asarray(model.atom_pos(), dtype=float)
    inv_uc = la.inv(np.asarray(model.unit_cell(), dtype=float))
    d_frac = apos @ inv_uc
    thetas, phis, local_dirs = [], [], []
    for i in range(N):
        u = rotation_about_axis(-2.0 * np.pi * float(d_frac[i] @ k_opt), axis_opt) @ spins_opt[i]
        u = u / la.norm(u)
        local_dirs.append([float(c) for c in u])
        thetas.append(float(np.arccos(np.clip(u[2], -1.0, 1.0))))
        phis.append(float(np.arctan2(u[1], u[0])))

    new_ms = dict(ms_cfg) if ms_cfg else {}
    new_ms.update({
        'type': 'single_k',
        'enabled': True,
        'k': [float(c) for c in k_opt],
        'axis': [float(c) for c in axis_opt],
        'local_directions': local_dirs,
        'k_case': spiral_propagation_case(k_opt),
    })
    for stale in ('S0', 'u', 'v', 'n', 'real_space'):
        new_ms.pop(stale, None)
    model.mag_struct_cfg = new_ms
    model.config['magnetic_structure'] = dict(new_ms)
    model.set_magnetic_structure(thetas, phis)

    msg = (f"Spiral optimum: k = {nice_k_string(k_opt)}, "
           f"axis = {np.round(axis_opt, 6).tolist()}, "
           f"E/site = {E_opt / N:.8f}")
    logger.info(msg)

    return SpiralOptResult(
        k_rlu=k_opt, axis=axis_opt, thetas=thetas, phis=phis,
        spins_lab=spins_opt, energy=E_opt, energy_per_site=E_opt / N,
        converged=bool(best.success), message=msg, scipy_result=best,
    )
