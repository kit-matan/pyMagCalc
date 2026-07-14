"""Monte-Carlo / annealing ground-state search for the classical spin Hamiltonian.

Closes the gap with SpinW's `anneal` + `optmagsteep` and Sunny's `LocalSampler`.
Random-multistart L-BFGS (the previous only option) reliably lands in local minima
on frustrated systems, and LSWT expanded about one of those is meaningless -- see
`tests/test_ground_state_guard.py`.

Everything here works on the classical energy in CARTESIAN spin components,

    E(m) = 1/2 m^T H m + b . m + c,      m = [m_0x, m_0y, m_0z, m_1x, ...],  |m_i| = S

which `MagCalc._extract_classical_quadratic` already provides. That form is what
makes a real Metropolis sampler cheap: a single-site move touches only 3 of the 3N
components, so with the gradient g = H m + b carried along,

    dE = delta . g_i + 1/2 delta^T H_ii delta                     (O(1))
    g += H[:, block_i] @ delta        after an accepted move       (O(3N))

i.e. no full energy re-evaluation per proposal.

Two drivers:

* `anneal`  -- Metropolis with a geometric temperature schedule (SpinW `anneal`).
               Proposals are Sunny's `LocalSampler` mix: uniform resampling of a
               spin on the sphere, a spin flip, and a small adaptive delta move.
* `steepest_descent` -- iteratively align each spin with its local field
               (SpinW `optmagsteep`). Very fast, but it only ever goes downhill, so
               it is used to polish an annealed state, not to explore.

Both are followed by an L-BFGS polish in `MagCalc.minimize_energy`, so the returned
state is a true stationary point, not merely a low-temperature snapshot.
"""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def energy(m: np.ndarray, H: np.ndarray, b: np.ndarray, c: float) -> float:
    return float(0.5 * m @ (H @ m) + b @ m + c)


def random_spins(n: int, S: float, rng: np.random.Generator) -> np.ndarray:
    """n random spins of magnitude S, uniform on the sphere, flattened to 3n."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return (S * v).ravel()


def steepest_descent(
    m: np.ndarray,
    H: np.ndarray,
    b: np.ndarray,
    c: float,
    S: float,
    n: int,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, float]:
    """SpinW `optmagsteep`: repeatedly align each spin with its local field.

    The energy as a function of one spin is
        E(m_i) = 1/2 m_i^T H_ii m_i + m_i . h_i + const,
        h_i = (H m)_i + b_i - H_ii m_i          (field from everything except itself)
    so the downhill choice on the sphere |m_i| = S is m_i = -S * h_i / |h_i|. With an
    on-site anisotropy (H_ii != 0) that is a good step but not exactly optimal; the
    caller's L-BFGS polish cleans that up.

    Monotone by construction -- it cannot escape a local minimum. Use it to polish.
    """
    m = m.copy()
    e_prev = energy(m, H, b, c)
    for _ in range(max_iter):
        g = H @ m + b
        for i in range(n):
            sl = slice(3 * i, 3 * i + 3)
            Hii = H[sl, sl]
            h_ext = g[sl] - Hii @ m[sl]
            norm = np.linalg.norm(h_ext)
            if norm < 1e-12:
                continue
            new = -S * h_ext / norm
            delta = new - m[sl]
            m[sl] = new
            g += H[:, sl] @ delta          # keep the gradient in sync
        e = energy(m, H, b, c)
        if abs(e_prev - e) < tol:
            break
        e_prev = e
    return m, energy(m, H, b, c)


def anneal(
    H: np.ndarray,
    b: np.ndarray,
    c: float,
    S: float,
    n: int,
    n_sweeps: int = 2000,
    T_start: Optional[float] = None,
    T_end: Optional[float] = None,
    seed: int = 0,
    m0: Optional[np.ndarray] = None,
    polish_steep: bool = True,
) -> Tuple[np.ndarray, float]:
    """Simulated annealing (Metropolis) for the classical ground state.

    n_sweeps: temperature steps; each sweep attempts `n` single-site moves.
    T_start / T_end: temperatures in meV. Default T_start from the energy scale of
        the couplings (so the schedule adapts to the model rather than to a number
        the user has to guess), T_end small enough to freeze.

    Returns the BEST state seen (not the final one -- Metropolis can wander uphill
    at the end of a finite schedule).
    """
    rng = np.random.default_rng(seed)
    m = random_spins(n, S, rng) if m0 is None else m0.copy()

    # Energy scale: the typical local field a spin feels. This sets a temperature
    # that is hot enough to melt the structure but not so hot the schedule wastes
    # its whole budget in a paramagnet.
    scale = float(np.abs(H).sum(axis=1).max() * S + np.abs(b).max())
    scale = max(scale, 1e-6)
    T0 = float(T_start) if T_start is not None else scale
    T1 = float(T_end) if T_end is not None else scale * 1e-6
    T1 = max(T1, 1e-12)
    cooling = (T1 / T0) ** (1.0 / max(n_sweeps - 1, 1))

    g = H @ m + b
    e = energy(m, H, b, c)
    m_best, e_best = m.copy(), e

    sigma = 0.6            # width of the small "delta" move, adapted on the fly
    accepted = 0
    attempted = 0
    T = T0

    for sweep in range(n_sweeps):
        for _ in range(n):
            i = int(rng.integers(n))
            sl = slice(3 * i, 3 * i + 3)
            old = m[sl]

            # Sunny LocalSampler's proposal mix.
            u = rng.random()
            if u < 0.5:                                    # propose_uniform
                v = rng.normal(size=3)
                new = S * v / np.linalg.norm(v)
            elif u < 0.6:                                  # propose_flip
                new = -old
            else:                                          # propose_delta
                v = old + sigma * S * rng.normal(size=3)
                nv = np.linalg.norm(v)
                if nv < 1e-12:
                    continue
                new = S * v / nv

            delta = new - old
            dE = float(delta @ g[sl] + 0.5 * delta @ (H[sl, sl] @ delta))

            attempted += 1
            if dE <= 0.0 or rng.random() < np.exp(-dE / T):
                m[sl] = new
                g += H[:, sl] @ delta
                e += dE
                accepted += 1
                if e < e_best:
                    e_best = e
                    m_best = m.copy()

        # Keep the delta move in a useful regime (~50% acceptance).
        if attempted > 0:
            rate = accepted / attempted
            sigma = float(np.clip(sigma * (1.2 if rate > 0.5 else 0.8), 1e-3, 2.0))
            accepted = attempted = 0
        T *= cooling

    # Recompute from scratch: `e` was accumulated incrementally over ~n_sweeps*n
    # moves and will have drifted.
    e_best = energy(m_best, H, b, c)

    if polish_steep:
        m_best, e_best = steepest_descent(m_best, H, b, c, S, n)

    return m_best, e_best


def cartesian_to_angles(m: np.ndarray, n: int) -> np.ndarray:
    """Cartesian spins (3n,) -> the angle vector [th0, ph0, th1, ph1, ...]."""
    x = np.zeros(2 * n)
    for i in range(n):
        v = m[3 * i:3 * i + 3]
        r = np.linalg.norm(v)
        if r < 1e-12:
            continue
        x[2 * i] = np.arccos(np.clip(v[2] / r, -1.0, 1.0))
        x[2 * i + 1] = np.arctan2(v[1], v[0])
    return x
