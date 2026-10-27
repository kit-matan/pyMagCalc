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

HOW THE REFUSAL IS DECIDED. Not from the symmetry labelling but from the sum rule
itself: with one λ only the TOTAL constraint (1/N_q) Σ_q Tr (λ+J(q))⁻¹ = β Σ_i S_i²
is imposed, while the theory Sunny solves imposes it per sublattice,

    (1/N_q) Σ_q Tr₃ [(λ_i + J(q))⁻¹]_{ii} = β S_i²   for every site i.

Sites related by symmetry satisfy each of those automatically once the total does
(measured deviation ~1e-16 on the kagome AFM), so checking the per-site residuals
tests the physics rather than the labelling, and it costs one extra pass over the
λ-mesh. The check exists because it was not there: a mixed-spin chain (S = 1 and
S = 5/2) returned an S(q) whose two sublattice sum rules were off by +262% and
−42% with a total residual of 3e-16 -- no warning, no clue in the output.
`find_lagrange_multiplier_multi`, Sunny's multi-λ branch, is what would actually
compute this case, and is not implemented here.

KERNELS. `calculation: {kernels: rust}` (K5 of RUST_KERNELS_PLAN.md) moves the three
per-q loops into rsMagCalc's `ScgaModel`: the J(q) eigenvalues over the λ mesh, the
per-site traces of (λ+J(q))⁻¹ behind the sublattice check, and the correlation tensor
C(q) of `scga_intensities`. J(q) and its eigenvalues are bit-identical to the loops
below, so `solve_lambda` returns the same λ; the root solve, its bracketing, the
refusal and `_contract` stay here. Any kernel failure logs and runs the NumPy loop,
which is also the parity reference (tests/test_kernels_scga.py, and rsMagCalc's
tests/test_scga_parity.py).
"""
import logging
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from . import kernels as _kernels
from .form_factors import get_form_factor
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


def _rust_call(cache, kernels, what, call):
    """`call(rsmagcalc.ScgaModel)` under kernels "rust", else None.

    The packed model is cached on `cache` (keyed on the bond list it was packed
    from). A failure -- packing or the call -- is logged by `kernels.fell_back`
    and returns None, so the caller runs its NumPy loop."""
    rs = _kernels.rust_for(kernels)
    if rs is None:
        return None
    try:
        key = (id(cache["bonds"]), len(cache["bonds"]))
        packed = cache.get("_rust")
        if packed is None or packed[0] != key:
            packed = (key, rs.ScgaModel.from_cache(cache))
            cache["_rust"] = packed
        return call(packed[1])
    except Exception:
        _kernels.fell_back(what)
        return None


def _g_tensors(model, N):
    """Per-site 3×3 g-tensors (default 2·I), for the magnetic-moment prefactor.

    Only a model without the resolver (a legacy python_model_file) falls back to 2·I;
    a declared g-tensor that cannot be resolved must raise. The old bare `except
    Exception` turned every resolver failure -- including parameterised g entries,
    broken until 2026-10-13 -- into a silent g = 2 (tests: test_scga_g_tensors_*)."""
    resolver = getattr(model, "_resolve_g_tensors", None)
    gt = resolver() if resolver is not None else None
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


#: Per-sublattice sum-rule residual above which the single-λ theory is refused.
#: Symmetry-equivalent sites land at ~1e-16 (kagome AFM), so this is loose by ten
#: orders of magnitude and still catches an inequivalent sublattice, which is off by
#: order unity. See the module docstring.
SUBLATTICE_SUM_RULE_TOL: float = 1e-6


def sublattice_sum_rule_residuals(model, params, kT, lam, cache, mesh, kernels=None,
                                  n_workers=None):
    """Relative per-site sum-rule residuals of the single-λ solution.

        r_i = [ (1/N_q) Σ_q Tr₃ ((λ+J(q))⁻¹)_{ii} ] / (β S_i²) − 1

    Zero for every site iff the one λ enforces every sublattice constraint, which is
    the condition under which the single-λ theory is the theory. Returns an (N,) array.
    """
    r = _rust_call(cache, kernels, "SCGA sum-rule residuals",
                   lambda rm: rm.sum_rule_residuals(mesh, lam, kT, n_workers))
    if r is not None:
        return np.asarray(r, float)
    N = cache["N"]
    eye = np.eye(3 * N)
    acc = np.zeros(N)
    for q in mesh:
        A = fourier_exchange_matrix(model, params, q, cache)
        Cinv = np.linalg.inv(lam * eye + A)
        for i in range(N):
            acc[i] += float(np.real(np.trace(Cinv[3 * i:3 * i + 3, 3 * i:3 * i + 3])))
    acc /= len(mesh)
    target = np.asarray(cache["spins"], float) ** 2 / float(kT)
    return acc / target - 1.0


def solve_lambda(model, params, kT, nq=12, mesh=None,
                 sublattice_tol=SUBLATTICE_SUM_RULE_TOL, kernels=None, n_workers=None):
    """Find the single Lagrange multiplier λ from the spin sum rule.

    Returns (lam, lam_min, residual, Js) with Js the cached J(q) eigenvalues.
    The sum rule Σ_q Tr(λ+J(q))^{-1} = β Σ_i S_i² N_q is monotically decreasing in
    λ on (−min eig, ∞), so a scalar root find is robust.

    Raises when the resulting λ does not satisfy the PER-SUBLATTICE sum rules to
    `sublattice_tol` -- i.e. when the model needs the multi-λ theory that is not
    implemented here. Pass `sublattice_tol=float('inf')` to compute the single-λ
    answer anyway, knowing it violates the constraint it is derived from.

    `kernels="rust"` computes the mesh eigenvalues and the sublattice residuals in
    rsMagCalc (`n_workers` threads); the root solve is the same either way.
    """
    cache = _exchange_cache(model, params)
    lat = np.asarray(model.unit_cell(), float)
    if mesh is None:
        mesh = _q_mesh(lat, nq, _wraps(cache))
    beta = 1.0 / kT
    evals = _rust_call(cache, kernels, "SCGA mesh eigenvalues",
                       lambda rm: rm.eigenvalues(mesh, n_workers).ravel())
    if evals is None:
        evals = []
        for q in mesh:
            A = fourier_exchange_matrix(model, params, q, cache)
            evals.append(np.linalg.eigvalsh(A))
        evals = np.concatenate(evals)                   # 3N * Nq eigenvalues
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
    if cache["N"] > 1 and np.isfinite(sublattice_tol):
        r = sublattice_sum_rule_residuals(model, params, kT, lam, cache, mesh,
                                          kernels=kernels, n_workers=n_workers)
        if np.max(np.abs(r)) > sublattice_tol:
            worst = int(np.argmax(np.abs(r)))
            per_site = ", ".join(f"site {i} (S={s_i:g}): {100 * r_i:+.3g}%"
                                 for i, (s_i, r_i)
                                 in enumerate(zip(cache["spins"], r)))
            raise ValueError(
                f"SCGA: this model needs a Lagrange multiplier PER SUBLATTICE, and "
                f"only the single-lambda theory is implemented. One lambda = "
                f"{lam:.6g} meV satisfies the total spin sum rule (residual "
                f"{residual:.1e}) but violates the per-site constraints by up to "
                f"{100 * abs(r[worst]):.3g}% -- {per_site}. The sites are not one "
                f"symmetry class (unequal spin magnitudes will do it, as will "
                f"inequivalent local environments), so a single lambda over- and "
                f"under-constrains different sublattices and the S(q) it gives is "
                f"not the SCGA of this model. Sunny 0.8.1's multi-lambda branch "
                f"(find_lagrange_multiplier_multi) is the theory for it, and is not "
                f"implemented here -- so for THIS model there is no SCGA answer to "
                f"get from magcalc, and the thing to do is either reduce the model "
                f"to one symmetry class (equal spins, equivalent environments) or "
                f"use Sunny. (From the Python API only, and knowing it violates the "
                f"constraint it is derived from, solve_lambda(..., "
                f"sublattice_tol=float('inf')) returns the single-lambda number "
                f"anyway; there is deliberately no config key for it.)")
    return lam, lam_min, residual, cache, mesh


def scga_intensities(model, params, q_vectors, kT, nq=12, cross_section="perp",
                     lam=None, mesh=None, apply_g=True, cache=None, kernels=None,
                     n_workers=None):
    """Static SCGA structure factor S(q) at the given cartesian q-vectors.

    S(q) = kT · contract[ Σ_{ij} f_i f_j e^{-iq·r_i} g_i (λ+J(q))^{-1}_{ij} g_j^T e^{+iq·r_j} ].
    With `apply_g` the neutron sees the magnetic moment g·S (Sunny's default); set it
    False for the bare spin structure factor. `cache` may carry a precomputed
    _exchange_cache (compute_scga passes solve_lambda's, avoiding a rebuild).
    `kernels="rust"` computes the tensors C(q) in rsMagCalc (`n_workers` threads);
    the form factors are evaluated here and passed in, and `_contract` is shared.
    """
    if cache is None:
        cache = _exchange_cache(model, params)
    if lam is None:
        lam, lam_min, residual, cache, mesh = solve_lambda(
            model, params, kT, nq, mesh, kernels=kernels, n_workers=n_workers)
    N = cache["N"]
    ions = cache["ions"]
    qs = np.asarray(q_vectors, float).reshape(-1, 3)
    out = np.zeros(len(qs))
    if _kernels.rust_for(kernels) is not None:
        qmag = np.array([float(np.linalg.norm(q)) for q in qs])
        ff = np.ones((len(qs), N))
        for i in range(N):
            if ions[i]:
                ff[:, i] = get_form_factor(ions[i], qmag)
        Cs = _rust_call(cache, kernels, "SCGA correlations",
                        lambda rm: rm.correlations(qs, lam, kT, ff, apply_g, n_workers))
        if Cs is not None:
            for iq, q in enumerate(qs):
                out[iq] = _contract(Cs[iq], q, cross_section)
            return out
    for iq, q in enumerate(qs):
        C = correlation_tensor(model, params, q, kT, lam, cache, apply_g)
        out[iq] = _contract(C, q, cross_section)
    return out


def correlation_tensor(model, params, q, kT, lam, cache, apply_g=True):
    """The 3×3 magnetic-moment correlation tensor C^{αβ}(q) of `scga_intensities`
    at one cartesian q, before the cross-section contraction (the NumPy loop, and
    the parity reference of the `kernels: rust` path)."""
    N = cache["N"]
    pos = cache["pos"]
    ions = cache["ions"]
    g = cache["g"] if apply_g else [np.eye(3)] * N
    q = np.asarray(q, float)
    A = fourier_exchange_matrix(model, params, q, cache)
    Cinv = np.linalg.inv(lam * np.eye(3 * N) + A)
    qmag = float(np.linalg.norm(q))
    ff = np.array([get_form_factor(ions[i], qmag) if ions[i] else 1.0
                   for i in range(N)])
    phase = np.array([np.exp(1j * float(q @ pos[i])) for i in range(N)])
    C = np.zeros((3, 3), dtype=complex)
    for i in range(N):
        for j in range(N):
            blk = Cinv[3 * i:3 * i + 3, 3 * j:3 * j + 3]
            C += (ff[i] * ff[j] * np.conj(phase[i]) * phase[j]
                  * (g[i] @ blk @ g[j].T))
    C *= kT
    return C


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


def compute_scga(model, params, q_vectors, kT, nq=12, cross_section="perp",
                 kernels=None, n_workers=None):
    """Top-level: solve λ then evaluate S(q). Returns an SCGAResult."""
    lam, lam_min, residual, cache, mesh = solve_lambda(model, params, kT, nq,
                                                       kernels=kernels,
                                                       n_workers=n_workers)
    qs = np.asarray(q_vectors, float).reshape(-1, 3)
    I = scga_intensities(model, params, qs, kT, nq, cross_section, lam=lam,
                         mesh=mesh, cache=cache, kernels=kernels, n_workers=n_workers)
    logger.info("SCGA: kT=%.4g, λ=%.6g (λ_min=%.6g), %d q-mesh, sum-rule resid %.1e",
                kT, lam, lam_min, len(mesh), residual)
    return SCGAResult(q_vectors=qs, intensities=I, lam=lam, temperature=kT,
                      sum_rule_residual=residual, lam_min=lam_min)
