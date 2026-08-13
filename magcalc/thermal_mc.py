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

from .annealing import energy as _cell_energy
from .constants import GAMMA_ELECTRON as GAMMA, MU_B
from .sun.entangled import _pair_matrix

logger = logging.getLogger(__name__)


def n_chemical_cells(model, supercell):
    """Number of CHEMICAL unit cells in a `build_supercell` box.

    The normalizing volume for every S(q) / S(q,ω) this module and
    `classical_dynamics` produce, matching Sunny and pyMagCalc's own LSWT
    (`MagCalc.supercell_ncells`, `SUNModel.n_cells`). Two factors multiply:
    the sampler's own L₁×L₂×L₃ replication, and any `magnetic_supercell` the model
    already carries -- `model.atom_pos()` is then the MAGNETIC cell, so its sites
    span `prod(supercell_dims)` chemical cells before the sampler replicates it.

    Note this counts CELLS, not surviving spins, so it does not change when
    `disorder` removes sites: diluting a magnet lowers its scattering, and a
    per-site normalization would divide that physics back out.
    """
    ncell = int(np.prod([int(x) for x in supercell]))
    dims = getattr(model, 'supercell_dims', None) or [1, 1, 1]
    return ncell * int(np.prod([int(x) for x in dims]))


def build_supercell(model, params, supercell=(4, 4, 1), disorder=None,
                    periodic=(True, True, True)):
    """Assemble (H, b, N, S, pos) for a periodic L₁×L₂×L₃ supercell.

    H (3N×3N) is the exchange/anisotropy Hessian; b (3N,) the Zeeman field term, so
    E = ½ mᵀH m + bᵀm. Bonds come from `spin_interactions` (both directions, giving a
    symmetric H — the ½ convention), the field from `_resolve_field` with the
    per-site g-tensor (default g=2), matching the LSWT engine.

    SITE-LEVEL INHOMOGENEITY (Gap 4 #16, classical half). Two options:

    `disorder`: {'vacancy_concentration': x, 'seed': n} to remove a random fraction
        x of sites, or {'vacancies': [flat site indices]} for named ones. A vacancy is
        implemented by DELETING its rows/columns from H and its entries from b, which
        removes every bond it took part in — no bookkeeping to get wrong, and E is
        exactly the restriction of the clean energy to the surviving sites.
        Sunny's `set_vacancy_at!`.
    `periodic`: per-axis booleans. False drops the bonds that wrap around that axis,
        i.e. an open boundary (Sunny's `remove_periodicity!`).

    Both act on the explicit real-space supercell, which is why they are available
    here and not in the symbolic LSWT front end.
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

    per = tuple(bool(x) for x in periodic)
    dims = (L1, L2, L3)

    def wraps(cc, off):
        """Does this bond cross a boundary that has been opened?"""
        for ax in range(3):
            if per[ax]:
                continue
            t = cc[ax] + off[ax]
            if t < 0 or t >= dims[ax]:
                return True
        return False

    H = np.zeros((3 * N, 3 * N))
    for i in range(n):
        for j in range(len(aouc)):
            M = _pair_matrix(Jex, DM, Kex, i, j)
            if not np.any(M):
                continue
            jc = j % n
            off = np.round((aouc[j] - apos[jc]) @ inv).astype(int)
            for cc in cells:
                if wraps(cc, off):
                    continue
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

    H = 0.5 * (H + H.T)
    keep = _surviving_sites(N, disorder)
    if keep is not None:
        if len(keep) == 0:
            raise ValueError(
                "disorder removed every site; lower `vacancy_concentration`.")
        idx = np.concatenate([[3 * a, 3 * a + 1, 3 * a + 2] for a in keep])
        H = H[np.ix_(idx, idx)]
        b = b[idx]
        pos = pos[keep]
        N = len(keep)
    return H, b, N, S, pos


def _surviving_sites(N, disorder):
    """Flat indices of the sites left after `disorder`, or None if there are none
    removed. Raises on a spec that cannot mean what it says."""
    if not disorder:
        return None
    spec = dict(disorder)
    vac = spec.get("vacancies")
    x = spec.get("vacancy_concentration", spec.get("concentration"))
    if vac is not None and x is not None:
        raise ValueError(
            "give `vacancies` (explicit indices) or `vacancy_concentration`, "
            "not both.")
    if vac is not None:
        vac = sorted({int(v) for v in vac})
        bad = [v for v in vac if v < 0 or v >= N]
        if bad:
            raise ValueError(
                f"vacancy indices {bad} are outside the supercell (0..{N - 1}).")
    elif x is not None:
        x = float(x)
        if not 0.0 <= x < 1.0:
            raise ValueError(
                f"vacancy_concentration must be in [0, 1), got {x}.")
        if x == 0.0:
            return None
        rng = np.random.default_rng(int(spec.get("seed", 0)))
        n_vac = int(round(x * N))
        vac = sorted(rng.choice(N, size=n_vac, replace=False).tolist())
    else:
        raise ValueError(
            f"disorder needs `vacancies` or `vacancy_concentration`, got "
            f"{sorted(spec)}.")
    if not vac:
        return None
    return np.array([a for a in range(N) if a not in set(vac)], dtype=int)


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


def _sweep(m, g, H, b, beta, S, rng, propose="uniform"):
    """One Metropolis sweep (N single-spin updates).

    `propose`:
      "uniform" (default) -- a random point on the sphere, the general Heisenberg move.
      "flip"              -- S -> -S, Sunny's `propose_flip`. Starting from a polarized
                             state this keeps every spin on +/-n forever, i.e. it makes
                             the sampler ISING. That is the only way to reproduce
                             Sunny's 2-D Ising tutorial with a continuous-spin engine.
    """
    N = m.shape[0]
    acc = 0
    order = rng.permutation(N)
    flip = str(propose).lower() == "flip"
    for a in order:
        sl = slice(3 * a, 3 * a + 3)
        if flip:
            v = -m[a]
        else:
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
                       swap_every=1, seed=0, measure_every=1,
                       propose="uniform", init=None):
    """Replica-exchange Metropolis over the temperature ladder. Returns a
    ThermalResult with per-temperature thermodynamic averages."""
    temps = np.asarray(sorted(temperatures), float)
    R = len(temps)
    betas = 1.0 / temps
    rng = np.random.default_rng(seed)

    m = np.zeros((R, N, 3))
    g = np.zeros((R, 3 * N))
    for r in range(R):
        if init is not None:
            # A polarized start; with propose="flip" this is what makes the sampler
            # Ising (Sunny's `polarize_spins!` + `propose_flip`).
            v = np.tile(np.asarray(init, float).reshape(1, 3), (N, 1))
            v *= S / np.linalg.norm(v, axis=1, keepdims=True)
        else:
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
            acc_tot += _sweep(m[r], g[r], H, b, betas[r], S, rng, propose)
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
                   disorder=None, periodic=(True, True, True),
                   propose="uniform", init=None, swap_every=1,
                   n_sweeps=4000, n_equil=1500, seed=0):
    """Convenience: build the supercell and run parallel tempering."""
    H, b, N, S, _pos = build_supercell(model, params, supercell,
                                       disorder=disorder, periodic=periodic)
    return parallel_tempering(H, b, N, S, temperatures, propose=propose, init=init,
                              swap_every=swap_every, n_sweeps=n_sweeps,
                              n_equil=n_equil, seed=seed)

@dataclass
class StaticResult:
    """Instantaneous (energy-integrated) classical structure factor."""
    q_vectors: np.ndarray        # (Nq, 3) cartesian
    sq: np.ndarray               # (Nq,) S(q), per site
    temperature: float
    n_spins: int
    n_samples: int


def static_correlations(model, params, q_cart, kT, supercell=(6, 6, 1),
                        n_samples=200, n_equil=2000, sample_every=10,
                        cross_section="perp", seed=0, disorder=None,
                        periodic=(True, True, True)):
    """Classical INSTANTANEOUS S(q) from Metropolis samples (Sunny's
    `SampledCorrelationsStatic`).

    No dynamics: thermalize, then average the equal-time correlation

        S(q) = (1/n_cells) sum_ab P_ab(q) < S^a(q) S^b(q)* >,
        S^a(q) = sum_r e^{-i q.r} S^a_r

    over decorrelated configurations, with P the neutron projector selected by
    `cross_section`. The 1/n_cells makes it PER CHEMICAL CELL -- the normalization
    LSWT and Sunny use -- so free isotropic spins give exactly n_atoms * 2S^2/3 in
    `perp` and n_atoms * S^2 in `trace`, at every q and every temperature, which is
    the identity this is pinned to. It used to divide by the SITE count instead;
    that agrees for a one-site cell (every model this file was tested on) and is
    n_atoms too small otherwise.

    This is the classical, all-temperature counterpart of the LSWT band sum
    (`numerical.static_intensities`): it needs no ordered state and so is valid above
    T_N, where LSWT has nothing to say. It is also exactly the frequency integral of
    `classical_dynamics.sampled_correlations` (two-sided, uncorrected) -- the sum
    rule that fixes both absolute scales at once.
    """
    from .classical_dynamics import _contract

    H, b, N, S, pos = build_supercell(model, params, supercell, disorder=disorder,
                                      periodic=periodic)
    qs = np.asarray(q_cart, float).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    beta = 1.0 / float(kT)

    m = rng.standard_normal((N, 3))
    m *= S / np.linalg.norm(m, axis=1, keepdims=True)
    g = H @ m.ravel() + b
    for _ in range(int(n_equil)):
        _sweep(m, g, H, b, beta, S, rng)
        g = H @ m.ravel() + b

    phase = np.exp(-1j * (qs @ pos.T))            # (n_q, N)
    acc = np.zeros((len(qs), 3, 3), dtype=complex)
    for _ in range(int(n_samples)):
        for _ in range(int(sample_every)):
            _sweep(m, g, H, b, beta, S, rng)
            g = H @ m.ravel() + b
        Sq = phase @ m                            # (n_q, 3)
        acc += np.einsum("qa,qb->qab", Sq.conj(), Sq)
    acc /= (int(n_samples) * n_chemical_cells(model, supercell))

    sq = np.array([np.real(_contract(acc[i][None, :, :], qs[i], cross_section))[0]
                   for i in range(len(qs))])
    return StaticResult(q_vectors=qs, sq=sq, temperature=float(kT),
                        n_spins=N, n_samples=int(n_samples))

@dataclass
class WangLandauResult:
    """Density of states g(E) and the thermodynamics reconstructed from it."""
    energies: np.ndarray         # bin centres (total energy of the cell)
    log_g: np.ndarray            # ln g(E), normalized so min(log_g[occupied]) = 0
    histogram: np.ndarray        # final visit histogram
    n_spins: int
    f_final: float               # modification factor reached
    n_refinements: int

    def thermodynamics(self, temperatures):
        """<E>/N, C/N from g(E) at any temperature -- one run, every T.

        That is the point of Wang-Landau over Metropolis: the density of states is
        temperature-independent, so the T sweep is a post-processing step rather than
        a separate simulation each.
        """
        temps = np.atleast_1d(np.asarray(temperatures, float))
        occ = self.histogram > 0
        E, lg = self.energies[occ], self.log_g[occ]
        e_out, c_out = [], []
        for T in temps:
            w = lg - E / T
            w -= w.max()                       # stabilize before exponentiating
            p = np.exp(w)
            p /= p.sum()
            e1 = float(np.sum(p * E))
            e2 = float(np.sum(p * E * E))
            e_out.append(e1 / self.n_spins)
            c_out.append((e2 - e1 * e1) / (self.n_spins * T * T))
        return np.array(e_out), np.array(c_out)


def wang_landau(H, b, N, S, e_min, e_max, n_bins=100, f_init=1.0, f_final=1e-6,
                flatness=0.8, sweeps_per_check=200, max_sweeps=4_000_000, seed=0):
    """Density of states g(E) by flat-histogram (Wang-Landau) sampling.

    A random walk in CONFIGURATION space that accepts a move with probability
    min(1, g(E_old)/g(E_new)), so it spends equal time in every energy bin; each visit
    multiplies g(E) by f and increments a histogram. When the histogram is flat to
    `flatness`, f -> sqrt(f) and the histogram resets. Converged g(E) then gives the
    thermodynamics at EVERY temperature from one run.

    Energies outside [e_min, e_max] are rejected, so the window must actually contain
    the states of interest -- `wang_landau_window` estimates it.

    NB these are CONTINUOUS classical spins, so g(E) is a density and the bin width
    enters as a constant offset in ln g; it cancels in every thermodynamic average.
    """
    rng = np.random.default_rng(seed)
    e_min, e_max = float(e_min), float(e_max)
    if not e_max > e_min:
        raise ValueError(f"need e_max > e_min, got [{e_min}, {e_max}].")
    edges = np.linspace(e_min, e_max, int(n_bins) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    log_g = np.zeros(int(n_bins))
    hist = np.zeros(int(n_bins), dtype=np.int64)     # reset at every refinement
    visits = np.zeros(int(n_bins), dtype=np.int64)   # never reset: occupancy record

    def bin_of(E):
        if E < e_min or E >= e_max:
            return -1
        return min(int((E - e_min) / width), int(n_bins) - 1)

    # start from a configuration inside the window
    m = rng.standard_normal((N, 3))
    m *= S / np.linalg.norm(m, axis=1, keepdims=True)
    E = _cell_energy(m.ravel(), H, b, 0.0)
    for _ in range(20000):
        if bin_of(E) >= 0:
            break
        a = rng.integers(N)
        v = rng.standard_normal(3)
        v *= S / np.linalg.norm(v)
        old = m[a].copy()
        m[a] = v
        E = _cell_energy(m.ravel(), H, b, 0.0)
        if bin_of(E) < 0:
            m[a] = old
            E = _cell_energy(m.ravel(), H, b, 0.0)
    if bin_of(E) < 0:
        raise RuntimeError(
            f"could not find a configuration with energy in [{e_min}, {e_max}]; "
            f"widen the window (see wang_landau_window).")

    f = float(f_init)
    n_ref, swept = 0, 0
    g = H @ m.ravel() + b
    while f > float(f_final) and swept < int(max_sweeps):
        for _ in range(int(sweeps_per_check)):
            swept += 1
            for _ in range(N):
                a = int(rng.integers(N))
                sl = slice(3 * a, 3 * a + 3)
                v = rng.standard_normal(3)
                v *= S / np.linalg.norm(v)
                d = v - m[a]
                dE = float(g[sl] @ d + 0.5 * d @ (H[sl, sl] @ d))
                E_new = E + dE
                nb, ob = bin_of(E_new), bin_of(E)
                if nb >= 0 and (log_g[ob] - log_g[nb] >= 0
                                or rng.random() < np.exp(log_g[ob] - log_g[nb])):
                    m[a] = v
                    E = E_new
                    g = H @ m.ravel() + b
                    ob = nb
                log_g[ob] += f
                hist[ob] += 1
                visits[ob] += 1
        occ = hist > 0
        if occ.sum() > 1 and hist[occ].min() >= flatness * hist[occ].mean():
            f *= 0.5           # ln f -> ln f / 2, i.e. f -> sqrt(f) on g itself
            hist[:] = 0
            n_ref += 1

    # `hist` is zeroed at each refinement, so the occupancy mask must come from the
    # cumulative record -- otherwise a run that ends just after a refinement reports
    # an empty density of states.
    occ = visits > 0
    if occ.any():
        log_g = log_g - log_g[occ].min()
    return WangLandauResult(energies=centres, log_g=log_g, histogram=visits,
                            n_spins=N, f_final=f, n_refinements=n_ref)


def wang_landau_window(H, b, N, S, n_samples=4000, pad=0.02, seed=0):
    """A safe [e_min, e_max] for `wang_landau`, from a quench and random sampling.

    Random configurations bracket the high-energy end and a steepest-descent quench
    the low-energy end; the window is padded outward because a state outside it is
    silently unreachable, which would bias g(E) rather than fail.
    """
    rng = np.random.default_rng(seed)
    lo, hi = np.inf, -np.inf
    for _ in range(int(n_samples)):
        m = rng.standard_normal((N, 3))
        m *= S / np.linalg.norm(m, axis=1, keepdims=True)
        e = _cell_energy(m.ravel(), H, b, 0.0)
        lo, hi = min(lo, e), max(hi, e)
    m = rng.standard_normal((N, 3))
    m *= S / np.linalg.norm(m, axis=1, keepdims=True)
    for _ in range(500):                     # align each spin with its local field
        fld = -(H @ m.ravel() + b).reshape(N, 3)
        nrm = np.linalg.norm(fld, axis=1, keepdims=True)
        ok = nrm[:, 0] > 1e-12
        m[ok] = S * fld[ok] / nrm[ok]
    lo = min(lo, _cell_energy(m.ravel(), H, b, 0.0))
    span = max(hi - lo, 1e-9)
    return lo - pad * span, hi + pad * span

