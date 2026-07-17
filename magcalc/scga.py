"""Self-consistent Gaussian approximation (SCGA) — paramagnetic diffuse scattering.

Above the ordering temperature a frustrated magnet has no long-range order but a
structured, temperature-dependent diffuse S(q) (spin-liquid correlations, pinch
points, ...). LSWT — an expansion about an ordered state — says nothing there. The
SCGA is the standard classical-Gaussian tool for it: treat the spins as classical
vectors with a Boltzmann weight, and replace the hard local length constraint
|S_i| = S_i by a WEAKER global one enforced with a uniform Lagrange multiplier λ.

The Gaussian integral then gives the static structure factor in closed form:

    <S^α_i(-q) S^β_j(q)> = kT [ (λ 1 + J(q))^{-1} ]_{iα,jβ},

where J(q)_{iα,jβ} = Σ_R M^{αβ}(i, j+R) e^{i q·R} is the Fourier exchange matrix
(the SAME 3N×3N Hermitian matrix the Luttinger–Tisza guard uses, R the integer cell
offset), and λ is fixed by the spin sum rule

    (1/N_q) Σ_q Tr (λ 1 + J(q))^{-1} = β Σ_i S_i²         (β = 1/kT).

This is exactly Sunny 0.8.1's SCGA (`src/SCGA/SCGA.jl`): same J(q) convention
(`fourier_exchange_matrix`), same single-λ sum rule, same intensity
`kT · pref† (λ+J(q))^{-1} pref` with the neutron polarization/form-factor prefactor.
Validated against (a) the exact closed form for the classical Heisenberg chain,
(b) Sunny's own SCGA on the square-lattice and kagome antiferromagnets, and (c) the
sum rule and high-T (flat S(q) → S²) limits. See tests/test_scga.py.

Only the single-λ theory (all sites one symmetry class) is implemented — it covers
Bravais lattices and the symmetric frustrated lattices (kagome, pyrochlore, ...) SCGA
is used for; a genuinely inequivalent-sublattice model raises rather than silently
using one λ.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy import optimize

from .form_factors import get_form_factor
from .numerical import contract_cross_section
from .sun.entangled import _pair_matrix

logger = logging.getLogger(__name__)


@dataclass
class SCGAResult:
    q_vectors: np.ndarray          # (Nq, 3) cartesian
    intensities: np.ndarray        # (Nq,) S(q)
    lam: float                     # Lagrange multiplier λ (meV)
    temperature: float             # kT (meV)
    sum_rule_residual: float       # |(1/Nq)Σ Tr(λ+J)^-1 - βΣS²| — should be ~0
    lam_min: float                 # -min eigenvalue of J(q) over the mesh


def fourier_exchange_matrix(model, params, q_cart, _cache=None):
    """The 3N×3N Hermitian Fourier exchange matrix J(q) (Sunny's convention).

    J(q)_{iα,jβ} = Σ_{OUC images j' of j} M^{αβ}(i, j') e^{i q·R(j')},  R = integer
    cell offset of the image (NOT the intra-cell position — that phase is carried by
    the observable prefactor, matching Sunny). H = (1/2) Σ_q S(-q)† J(q) S(q).
    """
    if _cache is None:
        _cache = _exchange_cache(model, params)
    N = _cache["N"]
    A = np.zeros((3 * N, 3 * N), dtype=complex)
    q = np.asarray(q_cart, float)
    for (i, jc, R_cart, M) in _cache["bonds"]:
        ph = np.exp(1j * float(q @ R_cart))
        A[3 * i:3 * i + 3, 3 * jc:3 * jc + 3] += ph * M
    A = 0.5 * (A + A.conj().T)
    return A


def _exchange_cache(model, params):
    """Precompute the directed bond list (i, jc, R_cartesian, 3×3 M) once."""
    Jex, DM, Kex = model.spin_interactions(list(params or []))
    apos = np.asarray(model.atom_pos(), float)
    aouc = np.asarray(model.atom_pos_ouc(), float)
    lat = np.asarray(model.unit_cell(), float)
    inv_lat = np.linalg.inv(lat)
    N = len(apos)
    bonds = []
    for i in range(N):
        for j in range(len(aouc)):
            M = _pair_matrix(Jex, DM, Kex, i, j)
            if not np.any(M):
                continue
            jc = j % N
            R = np.round((aouc[j] - apos[jc]) @ inv_lat).astype(int)
            bonds.append((i, jc, R.astype(float) @ lat, M))
    spins = np.asarray(model.spin_magnitudes(), float)
    return {"N": N, "bonds": bonds, "pos": apos, "spins": spins,
            "ions": _ion_list(model, N), "g": _g_tensors(model, N)}


def _g_tensors(model, N):
    """Per-site 3×3 g-tensors (default 2·I), for the magnetic-moment prefactor."""
    try:
        gt = model._resolve_g_tensors()
    except Exception:
        gt = None
    if gt is None:
        return [2.0 * np.eye(3) for _ in range(N)]
    return [np.array([[float(g[a, b]) for b in range(3)] for a in range(3)]) for g in gt]


def _ion_list(model, N):
    try:
        ions = model.ion_list()
        if ions and len(ions) == N:
            return list(ions)
    except Exception:
        pass
    return [None] * N


def _q_mesh(lat, nq, wraps):
    """Regular BZ mesh (cartesian) over the reciprocal cell; `wraps[d]` False → the
    system is finite/uncoupled along d and that axis collapses to q=0 (as Sunny does)."""
    B = 2 * np.pi * np.linalg.inv(lat).T
    axes = [np.arange(nq) / nq - 0.5 if w else np.array([0.0]) for w in wraps]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid @ B


def _wraps(cache):
    w = np.zeros(3, bool)
    for (_i, _j, R_cart, _M) in cache["bonds"]:
        w |= np.abs(R_cart) > 1e-9
    return w if w.any() else np.array([True, True, True])


def solve_lambda(model, params, kT, nq=12, mesh=None):
    """Find the single Lagrange multiplier λ from the spin sum rule.

    Returns (lam, lam_min, residual, Js) with Js the cached J(q) eigenvalues.
    The sum rule Σ_q Tr(λ+J(q))^{-1} = β Σ_i S_i² N_q is monotically decreasing in
    λ on (−min eig, ∞), so a scalar root find is robust.
    """
    cache = _exchange_cache(model, params)
    lat = np.asarray(model.unit_cell(), float)
    if mesh is None:
        mesh = _q_mesh(lat, nq, _wraps(cache))
    beta = 1.0 / kT
    evals = []
    for q in mesh:
        A = fourier_exchange_matrix(model, params, q, cache)
        evals.append(np.linalg.eigvalsh(A))
    evals = np.concatenate(evals)                       # 3N * Nq eigenvalues
    Nq = len(mesh)
    s2 = float(np.sum(cache["spins"] ** 2))             # per cell
    target = beta * s2 * Nq                             # = Σ_{q,ev} 1/(λ+ev)
    lam_min = -float(evals.min())

    def g(lam):
        return float(np.sum(1.0 / (lam + evals))) - target

    # bracket: g(lam_min+) = +inf, g(inf) = 0- ; decreasing
    lo = lam_min + 1e-9 * max(abs(lam_min), 1.0)
    hi = lam_min + max(1.0, kT)
    while g(hi) > 0:
        hi = lam_min + 2 * (hi - lam_min)
        if hi - lam_min > 1e12:
            raise RuntimeError(
                "SCGA sum rule has no solution in a sane range -- check kT and the "
                "spin magnitudes (target lambda would exceed lambda_min + 1e12).")
    lam = optimize.brentq(g, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=200)
    residual = abs(g(lam)) / target
    return lam, lam_min, residual, cache, mesh


def scga_intensities(model, params, q_vectors, kT, nq=12, cross_section="perp",
                     lam=None, mesh=None, apply_g=True):
    """Static SCGA structure factor S(q) at the given cartesian q-vectors.

    S(q) = kT · contract[ Σ_{ij} f_i f_j e^{-iq·r_i} g_i (λ+J(q))^{-1}_{ij} g_j^T e^{+iq·r_j} ].
    With `apply_g` the neutron sees the magnetic moment g·S (Sunny's default); set it
    False for the bare spin structure factor.
    """
    cache = _exchange_cache(model, params)
    if lam is None:
        lam, lam_min, residual, cache, mesh = solve_lambda(model, params, kT, nq, mesh)
    N = cache["N"]
    pos = cache["pos"]
    ions = cache["ions"]
    g = cache["g"] if apply_g else [np.eye(3)] * N
    qs = np.asarray(q_vectors, float).reshape(-1, 3)
    out = np.zeros(len(qs))
    for iq, q in enumerate(qs):
        A = fourier_exchange_matrix(model, params, q, cache)
        Cinv = np.linalg.inv(lam * np.eye(3 * N) + A)
        qmag = float(np.linalg.norm(q))
        ff = np.array([get_form_factor(ions[i], qmag) if ions[i] else 1.0
                       for i in range(N)])
        phase = np.array([np.exp(1j * float(q @ pos[i])) for i in range(N)])
        # 3×3 magnetic-moment correlation tensor C^{αβ}(q)
        C = np.zeros((3, 3), dtype=complex)
        for i in range(N):
            for j in range(N):
                blk = Cinv[3 * i:3 * i + 3, 3 * j:3 * j + 3]
                C += (ff[i] * ff[j] * np.conj(phase[i]) * phase[j]
                      * (g[i] @ blk @ g[j].T))
        C *= kT
        out[iq] = _contract(C, q, cross_section)
    return out


def _contract(C, q, cross_section):
    """Neutron cross-section contraction of a 3×3 correlation tensor at q."""
    cs = (cross_section or "perp").lower()
    if cs == "trace":
        return float(np.real(np.trace(C)))
    if cs in ("xx", "yy", "zz"):
        a = {"xx": 0, "yy": 1, "zz": 2}[cs]
        return float(np.real(C[a, a]))
    # perp: (δ - q̂q̂) : C. At q → 0 the rotational average (2/3) Tr (Sunny's
    # convention; Sunny.jl PR #131), i.e. spherical average of uncorrelated data.
    qn = np.linalg.norm(q)
    if qn < 1e-12:
        return float((2.0 / 3.0) * np.real(np.trace(C)))
    qh = q / qn
    P = np.eye(3) - np.outer(qh, qh)
    return float(np.real(np.sum(P * C)))


def compute_scga(model, params, q_vectors, kT, nq=12, cross_section="perp"):
    """Top-level: solve λ then evaluate S(q). Returns an SCGAResult."""
    lam, lam_min, residual, cache, mesh = solve_lambda(model, params, kT, nq)
    qs = np.asarray(q_vectors, float).reshape(-1, 3)
    I = scga_intensities(model, params, qs, kT, nq, cross_section, lam=lam, mesh=mesh)
    logger.info("SCGA: kT=%.4g, λ=%.6g (λ_min=%.6g), %d q-mesh, sum-rule resid %.1e",
                kT, lam, lam_min, len(mesh), residual)
    return SCGAResult(q_vectors=qs, intensities=I, lam=lam, temperature=kT,
                      sum_rule_residual=residual, lam_min=lam_min)
