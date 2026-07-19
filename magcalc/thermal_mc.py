"""Thermal (finite-temperature) classical Monte-Carlo with parallel tempering.

`annealing.py` does the T→0 ground-state search; this does finite-T thermodynamics:
the temperature dependence of the energy, specific heat, magnetization, and
susceptibility of the classical spin Hamiltonian, on an explicit periodic supercell.

The classical energy is the same quadratic form the minimizer uses,
E(m) = ½ mᵀH m + bᵀm + c with m the 3N Cartesian components of N unit vectors of
length S (annealing.energy). Here H and b are assembled for an L₁×L₂×L₃ PERIODIC
SUPERCELL from the model's bonds (`spin_interactions`) and field
(`spiral_opt._resolve_field`), so genuine spatial fluctuations and correlations are
captured — not just a uniform single cell.

Sampling uses **parallel tempering** (replica exchange): one Metropolis replica per
temperature on a ladder, with periodic swaps between adjacent temperatures that keep
each replica Boltzmann-distributed while letting configurations tunnel through
barriers — essential for frustrated magnets whose single-T Metropolis freezes.

Validated (tests/test_thermal_mc.py) against EXACT results:
  * N non-interacting spins in a field: <m·B̂>/S = −L(βgμ_B|B|S), the Langevin
    function, at every temperature;
  * the classical Heisenberg dimer: <E>(T) from the exact 1-D partition-function
    integral;
  * the fluctuation identity C = Var(E)/(N kT²) equals the finite-difference d<E>/dT;
  * parallel tempering reproduces independent single-temperature Metropolis.
"""
import logging
from dataclasses import dataclass

import numpy as np

from .sun.entangled import _pair_matrix

logger = logging.getLogger(__name__)

MU_B = 5.788e-2      # meV / T
GAMMA = 2.0          # electron g (matches the LSWT/entangled Zeeman convention)


def build_supercell(model, params, supercell=(4, 4, 1)):
    """Assemble (H, b, N, S, pos) for a periodic L₁×L₂×L₃ supercell.

    H (3N×3N) is the exchange/anisotropy Hessian; b (3N,) the Zeeman field term, so
    E = ½ mᵀH m + bᵀm. Bonds come from `spin_interactions` (both directions, giving a
    symmetric H — the ½ convention), the field from `_resolve_field` with the
    per-site g-tensor (default g=2), matching the LSWT engine.
    """
    from .spiral_opt import _resolve_field
    from .scga import _g_tensors

    Jex, DM, Kex = model.spin_interactions(list(params or []))
    apos = np.asarray(model.atom_pos(), float)
    aouc = np.asarray(model.atom_pos_ouc(), float)
    lat = np.asarray(model.unit_cell(), float)
    inv = np.linalg.inv(lat)
    n = len(apos)
    L1, L2, L3 = (int(x) for x in supercell)
    ncell = L1 * L2 * L3
    N = n * ncell
    mags = np.asarray(model.spin_magnitudes(), float)
    if mags.size and float(np.ptp(mags)) > 1e-12:
        # A single |m_i| = S is assumed throughout the sampler; silently using
        # site 0's spin for every site would give wrong thermodynamics.
        raise NotImplementedError(
            f"thermal_mc / sampled_correlations support a single spin magnitude; "
            f"got mixed spins {sorted(set(mags.tolist()))}.")
    S = float(mags[0])

    cells = [(a, b_, c_) for a in range(L1) for b_ in range(L2) for c_ in range(L3)]
    cell_id = {cc: k for k, cc in enumerate(cells)}

    def site(cc, i):
        return cell_id[(cc[0] % L1, cc[1] % L2, cc[2] % L3)] * n + i

    H = np.zeros((3 * N, 3 * N))
    for i in range(n):
        for j in range(len(aouc)):
            M = _pair_matrix(Jex, DM, Kex, i, j)
            if not np.any(M):
                continue
            jc = j % n
            off = np.round((aouc[j] - apos[jc]) @ inv).astype(int)
            for cc in cells:
                a = site(cc, i)
                bnb = site((cc[0] + off[0], cc[1] + off[1], cc[2] + off[2]), jc)
                H[3 * a:3 * a + 3, 3 * bnb:3 * bnb + 3] += M

    b = np.zeros(3 * N)
    Hvec = _resolve_field(model, params)
    if Hvec is not None and np.linalg.norm(Hvec) > 0:
        g = _g_tensors(model, n)
        for i in range(n):
            bi = MU_B * (g[i].T @ np.asarray(Hvec, float))
            for cc in cells:
                a = site(cc, i)
                b[3 * a:3 * a + 3] = bi

    pos = np.zeros((N, 3))
    for cc in cells:
        for i in range(n):
            pos[site(cc, i)] = apos[i] + np.array(cc, float) @ lat
    return 0.5 * (H + H.T), b, N, S, pos


@dataclass
class ThermalResult:
    temperatures: np.ndarray     # kT (meV)
    energy: np.ndarray           # <E>/N per spin
    heat_capacity: np.ndarray    # C/N = Var(E)/(N kT²)
    magnetization: np.ndarray    # |<M>| per spin (= |Σ m|/(N S))
    susceptibility: np.ndarray   # (<M²>-<M>²) N /(kT), per spin
    mag_vector: np.ndarray       # <M>/(N S), (nT, 3)
    n_spins: int
    accept_rate: float


def _sweep(m, g, H, b, beta, S, rng):
    """One Metropolis sweep (N single-spin updates, random point on the sphere)."""
    N = m.shape[0]
    acc = 0
    order = rng.permutation(N)
    for a in order:
        sl = slice(3 * a, 3 * a + 3)
        v = rng.standard_normal(3)
        v *= S / np.linalg.norm(v)
        d = v - m[a]
        Haa = H[sl, sl]
        dE = float(g[sl] @ d + 0.5 * d @ (Haa @ d))
        if dE <= 0 or rng.random() < np.exp(-beta * dE):
            m[a] = v
            g += H[:, sl] @ d
            acc += 1
    return acc / N


def parallel_tempering(H, b, N, S, temperatures, n_sweeps=4000, n_equil=1500,
                       swap_every=1, seed=0, measure_every=1):
    """Replica-exchange Metropolis over the temperature ladder. Returns a
    ThermalResult with per-temperature thermodynamic averages."""
    temps = np.asarray(sorted(temperatures), float)
    R = len(temps)
    betas = 1.0 / temps
    rng = np.random.default_rng(seed)

    m = np.zeros((R, N, 3))
    g = np.zeros((R, 3 * N))
    for r in range(R):
        v = rng.standard_normal((N, 3))
        v *= S / np.linalg.norm(v, axis=1, keepdims=True)
        m[r] = v
        g[r] = H @ v.ravel() + b

    def energy(r):
        mr = m[r].ravel()
        return 0.5 * float(mr @ (H @ mr)) + float(b @ mr)

    sumE = np.zeros(R)
    sumE2 = np.zeros(R)
    sumM = np.zeros((R, 3))
    sumM2 = np.zeros(R)
    nmeas = 0
    acc_tot = 0.0
    acc_cnt = 0

    for sweep in range(n_sweeps):
        for r in range(R):
            acc_tot += _sweep(m[r], g[r], H, b, betas[r], S, rng)
            acc_cnt += 1
        # replica swaps on adjacent temperatures
        if swap_every and sweep % swap_every == 0:
            Es = np.array([energy(r) for r in range(R)])
            for r in range(R - 1):
                delta = (betas[r] - betas[r + 1]) * (Es[r] - Es[r + 1])
                if delta >= 0 or rng.random() < np.exp(delta):
                    m[[r, r + 1]] = m[[r + 1, r]]
                    g[[r, r + 1]] = g[[r + 1, r]]
                    Es[[r, r + 1]] = Es[[r + 1, r]]
        if sweep >= n_equil and (sweep - n_equil) % measure_every == 0:
            for r in range(R):
                E = energy(r)
                Mvec = m[r].sum(axis=0)
                sumE[r] += E
                sumE2[r] += E * E
                sumM[r] += Mvec
                sumM2[r] += Mvec @ Mvec
            nmeas += 1

    E_avg = sumE / nmeas
    E2_avg = sumE2 / nmeas
    C = (E2_avg - E_avg**2) / (N * temps**2)
    Mvec_avg = sumM / nmeas
    M2_avg = sumM2 / nmeas
    Mmag = np.linalg.norm(Mvec_avg, axis=1)
    chi = (M2_avg - Mmag**2) / (N * temps)
    return ThermalResult(
        temperatures=temps, energy=E_avg / N, heat_capacity=C,
        magnetization=Mmag / (N * S), susceptibility=chi,
        mag_vector=Mvec_avg / (N * S), n_spins=N,
        accept_rate=acc_tot / max(acc_cnt, 1))


def run_thermal_mc(model, params, temperatures, supercell=(4, 4, 1),
                   n_sweeps=4000, n_equil=1500, seed=0):
    """Convenience: build the supercell and run parallel tempering."""
    H, b, N, S, _pos = build_supercell(model, params, supercell)
    return parallel_tempering(H, b, N, S, temperatures, n_sweeps=n_sweeps,
                              n_equil=n_equil, seed=seed)
