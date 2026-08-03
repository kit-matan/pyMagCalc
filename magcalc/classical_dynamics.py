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


def evolve(H, b, S, m0, dt, n_steps, record_every=1):
    """Deterministic LL trajectory. Returns (n_rec, N, 3) recorded configurations."""
    m = m0.copy()
    rec = []
    for step in range(n_steps):
        if step % record_every == 0:
            rec.append(m.copy())
        m = llg_rk4_step(H, b, m, dt, S)
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
                         classical_to_quantum=True):
    """Thermalize by Metropolis then evolve LL dynamics; average S(q,ω) over `n_traj`
    independent thermal starts. Returns a DynamicsResult.

    `classical_to_quantum` (default on) rescales the result onto the quantum
    intensity scale -- see `classical_to_quantum_factor`. Without it the output is
    the raw classical S(q,ω), which equipartitions and therefore carries far too
    much weight at ħω >> kT; set it False only to inspect that raw quantity."""
    from .thermal_mc import build_supercell, _sweep

    H, b, N, S, pos = build_supercell(model, params, supercell)
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
