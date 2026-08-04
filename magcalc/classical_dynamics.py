"""Finite-temperature classical spin dynamics and SampledCorrelations S(q,ω).

LSWT gives S(q,ω) as an expansion about the ORDERED ground state; above T_N, or when
you want the full thermal lineshape (damping, multi-magnon continuum, paramagnons),
the classical route is real-time spin dynamics on a thermally sampled ensemble — the
`SampledCorrelations` idea (Sunny): thermalize, evolve the deterministic
Landau–Lifshitz equation, and Fourier transform the trajectory in space and time.

Landau–Lifshitz (undamped, microcanonical measurement):

    dS_i/dt = − S_i × B_i,    B_i = ∂E/∂S_i = (H S)_i + b_i,

with the SAME classical energy E = ½ SᵀH S + bᵀS as `thermal_mc`/`annealing`. This
conserves energy and |S_i|, and its small-amplitude normal modes ARE the spin-wave
frequencies (for a ferromagnet, exactly the LSWT dispersion). Thermal states are
drawn by Metropolis (`thermal_mc`), then evolved; averaging |Σ_r e^{-iq·r} S_r(t)|²
over trajectories gives

    S(q,ω) = ⟨ |S(q,ω)|² ⟩ / (N_t N),   S(q,ω) = Σ_t e^{iωt} Σ_r e^{-iq·r} S_r(t).

Validated (tests/test_classical_dynamics.py) against exact/independent results:
  * a single spin in a field precesses at the Larmor frequency ω = gμ_B B (the S(0,ω)
    peak) — pins the integrator, the time axis and the frequency convention;
  * the undamped integrator conserves energy to O(dt⁴) (RK4);
  * for a Heisenberg ferromagnet the low-T S(q,ω) peak positions fall on the exact
    LSWT magnon dispersion the pyMagCalc engine computes — tying the dynamics to the
    validated spin-wave engine.
"""
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def local_field(H, b, m):
    """B_i = (H m)_i + b_i, shape (N, 3)."""
    return (H @ m.ravel() + b).reshape(-1, 3)


def _deriv(H, b, m):
    return -np.cross(m, local_field(H, b, m))


def llg_rk4_step(H, b, m, dt, S):
    """One RK4 step of dS/dt = −S×B, renormalizing |S_i|=S (removes O(dt⁵) drift)."""
    k1 = _deriv(H, b, m)
    k2 = _deriv(H, b, m + 0.5 * dt * k1)
    k3 = _deriv(H, b, m + 0.5 * dt * k2)
    k4 = _deriv(H, b, m + dt * k3)
    m = m + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    m *= S / np.linalg.norm(m, axis=1, keepdims=True)
    return m



def implicit_midpoint_step(H, b, m, dt, S, tol=1e-12, max_iter=50):
    """One step of dS/dt = -S x B by the IMPLICIT MIDPOINT rule (Sunny's
    `ImplicitMidpoint`).

        S_{n+1} = S_n + dt * f((S_n + S_{n+1}) / 2)

    solved by fixed-point iteration. Unlike RK4 this is symplectic: it conserves the
    energy and every |S_i| to the solver tolerance rather than to O(dt^4), so the
    error stays BOUNDED over a long trajectory instead of drifting secularly. That
    matters for S(q,w), where a slow energy drift smears the very peaks being
    measured.

    Note no renormalization is applied afterwards: the midpoint rule preserves |S_i|
    exactly (the update is orthogonal to the midpoint spin), and rescaling would mask
    a genuine solver failure.
    """
    m_new = m.copy()
    for _ in range(max_iter):
        mid = 0.5 * (m + m_new)
        cand = m + dt * _deriv(H, b, mid)
        delta = float(np.abs(cand - m_new).max())
        m_new = cand
        if delta < tol:
            break
    else:
        logger.warning(
            f"implicit midpoint did not converge in {max_iter} iterations "
            f"(last step {delta:.2e}); reduce dt.")
    return m_new


def suggest_timestep(H, b, m, S, target=0.033):
    """A time step resolving the fastest precession (Sunny `suggest_timestep`).

    The fastest frequency in the system is set by the largest local field,
    omega_max ~ |B_i|_max, so dt = target / omega_max with `target` a fraction of a
    radian per step (Sunny's default accuracy knob is comparable). Returns a float in
    the same time units the rest of this module uses.
    """
    B = local_field(H, b, m)
    w_max = float(np.abs(B).max())
    if w_max <= 0:
        raise ValueError(
            "suggest_timestep: the local field vanishes everywhere, so there is no "
            "timescale to resolve (is the Hamiltonian empty?).")
    return float(target) / w_max


def langevin_step(H, b, m, dt, S, kT, damping, rng):
    """One step of the stochastic Landau-Lifshitz (Langevin) equation.

        dS/dt = -S x G + (lambda/S) S x (S x G) + noise,    G = dE/dS

    MIND THE SIGN. `local_field` returns the energy GRADIENT G = dE/dS, not the
    physical field B = -G, and the module's precession term is written -S x G to
    match. The damping must then be +(lambda/S) S x (S x G): using -(...) relaxes
    spins AWAY from the energy minimum, which shows up as a magnetization of the
    right magnitude and the WRONG SIGN -- plausible enough to miss without an exact
    reference. Check: for E = -h S^z (so G = -h zhat) and S along x,
    S x (S x G) = +S^2 h zhat, i.e. toward +z, which is the minimum.

    integrated by Heun's method. The noise amplitude is fixed by the
    fluctuation-dissipation theorem, so the stationary distribution is Boltzmann at
    `kT` -- which is the whole point, and what the tests check rather than assume.

    This is a THERMOSTAT (Sunny's `Langevin`), an alternative to Metropolis for
    preparing thermal states. Measurement trajectories should still be run
    undamped/deterministic; damping distorts the lineshape.
    """
    lam = float(damping)
    beta = 1.0 / float(kT)
    # <xi_a(t) xi_b(t')> = 2 lambda kT delta_ab delta(t-t') / S  (FDT)
    sigma = np.sqrt(2.0 * lam * float(kT) * dt / float(S))

    def drift(x):
        B = local_field(H, b, x)
        return -np.cross(x, B) + (lam / float(S)) * np.cross(x, np.cross(x, B))

    noise = sigma * rng.standard_normal(m.shape)
    f0 = drift(m)
    pred = m + dt * f0 - np.cross(m, noise)
    f1 = drift(pred)
    out = m + 0.5 * dt * (f0 + f1) - np.cross(0.5 * (m + pred), noise)
    # |S| is conserved by the continuum equation but not by a finite Heun step
    out *= float(S) / np.linalg.norm(out, axis=1, keepdims=True)
    return out


def langevin_thermalize(H, b, S, m, kT, dt, n_steps, damping=0.1, seed=0):
    """Run the Langevin thermostat for `n_steps`, returning the final configuration."""
    rng = np.random.default_rng(seed)
    for _ in range(int(n_steps)):
        m = langevin_step(H, b, m, dt, S, kT, damping, rng)
    return m


def evolve(H, b, S, m0, dt, n_steps, record_every=1, integrator="rk4"):
    """Deterministic LL trajectory. Returns (n_rec, N, 3) recorded configurations.

    `integrator`: "rk4" (default, historical) or "midpoint" (implicit midpoint --
    symplectic, bounded energy error over long runs).
    """
    step_fn = {"rk4": llg_rk4_step,
               "midpoint": implicit_midpoint_step}.get(str(integrator).lower())
    if step_fn is None:
        raise ValueError(
            f"integrator must be 'rk4' or 'midpoint', got {integrator!r}.")
    m = m0.copy()
    rec = []
    for step in range(n_steps):
        if step % record_every == 0:
            rec.append(m.copy())
        m = step_fn(H, b, m, dt, S)
    return np.array(rec)


def energy(H, b, m):
    mr = m.ravel()
    return 0.5 * float(mr @ (H @ mr)) + float(b @ mr)


@dataclass
class DynamicsResult:
    q_vectors: np.ndarray        # (Nq, 3) cartesian
    energies: np.ndarray         # (Nω,) meV
    sqw: np.ndarray              # (Nω, Nq) S(q,ω)
    temperature: float
    classical_to_quantum: bool = False   # was the c2q factor already applied?


def dynamical_structure_factor(traj, pos, q_cart, dt, cross_section="perp"):
    """S(q,ω) from one trajectory `traj` (n_t, N, 3) at cartesian q-vectors.

    Returns (energies (n_t//2,), sqw (n_ω, n_q)). Time→ω by FFT; positive ω only.
    """
    n_t, N, _ = traj.shape
    qs = np.asarray(q_cart, float).reshape(-1, 3)
    phase = np.exp(-1j * (qs @ pos.T))                  # (n_q, N)
    # S^a(q, t) = Σ_r e^{-iq·r} S^a_r(t)  -> (n_t, n_q, 3)
    Sqt = np.einsum("qr,tra->tqa", phase, traj)
    Sqw = np.fft.fft(Sqt, axis=0)                       # (n_t, n_q, 3)
    n_w = n_t // 2
    energies = 2 * np.pi * np.fft.fftfreq(n_t, d=dt)[:n_w]
    out = np.zeros((n_w, len(qs)))
    for iq, q in enumerate(qs):
        tensor = np.einsum("wa,wb->wab", Sqw[:n_w, iq, :].conj(), Sqw[:n_w, iq, :])
        out[:, iq] = np.real(_contract(tensor, q, cross_section)) / (n_t * N)
    return energies, out


def _contract(tensor, q, cross_section):
    """(n_ω,) neutron contraction of a (n_ω,3,3) correlation tensor at q."""
    cs = (cross_section or "perp").lower()
    if cs == "trace":
        return np.einsum("waa->w", tensor)
    if cs in ("xx", "yy", "zz"):
        a = {"xx": 0, "yy": 1, "zz": 2}[cs]
        return tensor[:, a, a]
    qn = np.linalg.norm(q)
    P = np.eye(3) if qn < 1e-12 else np.eye(3) - np.outer(q / qn, q / qn)
    return np.einsum("ab,wab->w", P, tensor)



def classical_to_quantum_factor(energies, kT):
    """|w/kT| / (1 - exp(-w/kT)) -- the classical-to-quantum correspondence factor.

    Classical spin dynamics equipartitions: every mode carries kT, so the classical
    S(q,w) is NOT on the quantum intensity scale. The factor below maps it onto one,
    and is what Sunny applies in `intensities(sc, ...; kT)`
    (SampledCorrelations/DataRetrieval.jl). It is equivalent to
    `abs(w/kT) * thermal_prefactor(w; kT)`, tends to 1 as w -> 0 (where classical
    statistics are already right) and to |w|/kT in the quantum limit w >> kT, which
    is exactly the Bose suppression a classical calculation is missing.

    `kT` in meV, matching `energies`. Returns 1 everywhere for kT = None.
    """
    w = np.asarray(energies, float)
    if kT is None:
        return np.ones_like(w)
    if kT <= 0:
        raise ValueError(
            f"classical_to_quantum needs a positive kT in meV, got {kT}. Pass "
            f"`classical_to_quantum: false` to leave the classical scale alone.")
    x = w / kT
    out = np.ones_like(x)
    nz = x != 0.0
    # -expm1(-x), not 1 - exp(-x): the latter cancels catastrophically for small x
    # (at x ~ 3e-9 it is already wrong in the 8th digit), which matters because the
    # w -> 0 end of the grid is where the factor is supposed to be exactly 1.
    out[nz] = np.abs(x[nz]) / -np.expm1(-x[nz])
    return out


def sampled_correlations(model, params, q_cart, kT, supercell=(6, 1, 1),
                         dt=0.02, n_steps=2048, n_traj=8, therm_sweeps=2000,
                         record_every=1, cross_section="perp", seed=0,
                         classical_to_quantum=True, disorder=None,
                         periodic=(True, True, True)):
    """Thermalize by Metropolis then evolve LL dynamics; average S(q,ω) over `n_traj`
    independent thermal starts. Returns a DynamicsResult.

    `classical_to_quantum` (default on) rescales the result onto the quantum
    intensity scale -- see `classical_to_quantum_factor`. Without it the output is
    the raw classical S(q,ω), which equipartitions and therefore carries far too
    much weight at ħω >> kT; set it False only to inspect that raw quantity."""
    from .thermal_mc import build_supercell, _sweep

    H, b, N, S, pos = build_supercell(model, params, supercell, disorder=disorder,
                                      periodic=periodic)
    rng = np.random.default_rng(seed)
    beta = 1.0 / kT

    energies = None
    acc = None
    for it in range(n_traj):
        m = rng.standard_normal((N, 3))
        m *= S / np.linalg.norm(m, axis=1, keepdims=True)
        g = H @ m.ravel() + b
        for _ in range(therm_sweeps):
            _sweep(m, g, H, b, beta, S, rng)
        traj = evolve(H, b, S, m, dt, n_steps, record_every)
        e, sqw = dynamical_structure_factor(traj, pos, q_cart, dt * record_every,
                                            cross_section)
        acc = sqw if acc is None else acc + sqw
        energies = e
    sqw = acc / n_traj
    if classical_to_quantum:
        sqw = sqw * classical_to_quantum_factor(energies, kT)[:, None]
    return DynamicsResult(q_vectors=np.asarray(q_cart, float).reshape(-1, 3),
                          energies=energies, sqw=sqw, temperature=kT,
                          classical_to_quantum=bool(classical_to_quantum))
