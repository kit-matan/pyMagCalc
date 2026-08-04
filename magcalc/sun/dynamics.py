"""Finite-temperature classical dynamics of SU(N) coherent states (Gap 4 #26).

Sunny's generalized spin dynamics (`04_GSD_FeI2.jl`) and `EntangledSampledCorrelations`.
`classical_dynamics.py` evolves DIPOLES under Landau-Lifshitz; that cannot represent
a site whose state is a general N-level coherent state, which is the whole point of
SU(N) mode and of entangled units.

THE EQUATIONS OF MOTION. A classical SU(N) spin is a normalized complex vector
Z_i in CP^(N-1), and the energy E({Z}) is the coherent-state expectation the LSWT
engine already builds. Hamilton's equations in that phase space are just the
time-dependent Schroedinger equation with a MEAN FIELD:

    i dZ_i/dt = (dE / dZ_i*) = h_i Z_i,     h_i = SUNModel.local_field(i, <O>)

so `h_i` is exactly the local N x N Hamiltonian the CP^(N-1) ground-state search
already computes -- no new physics input, only a different use of it. Because every
h_i is Hermitian this conserves |Z_i| = 1 and the total energy automatically.

For N = 2 it must reduce EXACTLY to Landau-Lifshitz, and that is the load-bearing
test (`tests/test_sun_dynamics.py`), the dynamical analogue of the "S=1/2 SU(N) ==
dipole" gate that anchors `tests/test_sun.py`.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def expectations(model, Z):
    """<Z_i|O^p|Z_i> for every site and operator, shape (L, n_ops)."""
    return np.array([[Z[i].conj() @ model.ops[i][p] @ Z[i]
                      for p in range(model.n_ops)] for i in range(model.L)],
                    dtype=complex)


def local_hamiltonians(model, Z):
    """The mean-field h_i for the current configuration."""
    s0 = expectations(model, Z)
    return [model.local_field(i, s0) for i in range(model.L)]


def energy(model, Z):
    """Classical energy of a coherent-state configuration (total, not per site)."""
    return model._energy_of([np.asarray(z, complex) for z in Z])


def _deriv(model, Z):
    """dZ_i/dt = -i h_i Z_i."""
    hs = local_hamiltonians(model, Z)
    return [-1j * (hs[i] @ Z[i]) for i in range(model.L)]


def _renormalize(Z):
    return [z / np.linalg.norm(z) for z in Z]


def rk4_step(model, Z, dt):
    """One RK4 step of the CP^(N-1) equations, renormalized (removes O(dt^5) drift).

    The overall phase of each Z_i is unphysical -- every observable is a sesquilinear
    form <Z|O|Z> -- so no phase fixing is needed or wanted.
    """
    k1 = _deriv(model, Z)
    Z2 = [Z[i] + 0.5 * dt * k1[i] for i in range(model.L)]
    k2 = _deriv(model, Z2)
    Z3 = [Z[i] + 0.5 * dt * k2[i] for i in range(model.L)]
    k3 = _deriv(model, Z3)
    Z4 = [Z[i] + dt * k3[i] for i in range(model.L)]
    k4 = _deriv(model, Z4)
    out = [Z[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
           for i in range(model.L)]
    return _renormalize(out)


def evolve(model, Z0, dt, n_steps, record_every=1):
    """Deterministic CP^(N-1) trajectory. Returns a list of configurations."""
    Z = [np.asarray(z, complex).copy() for z in Z0]
    rec = []
    for step in range(int(n_steps)):
        if step % record_every == 0:
            rec.append([z.copy() for z in Z])
        Z = rk4_step(model, Z, dt)
    return rec


def metropolis_sweep(model, Z, beta, rng, sigma=0.5):
    """One Metropolis sweep over CP^(N-1): perturb a site's state and accept.

    The proposal mixes a random complex vector into Z_i and renormalizes, which is
    ergodic on CP^(N-1) and reduces to the usual sphere move at N = 2. `sigma` sets
    the step size; it does not affect the stationary distribution, only the
    acceptance rate.
    """
    acc = 0
    s0 = expectations(model, Z)
    for i in rng.permutation(model.L):
        i = int(i)
        n = model.Ns[i]
        v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        cand = Z[i] + sigma * v
        cand = cand / np.linalg.norm(cand)
        old = Z[i]
        e_old = energy(model, Z)
        Z[i] = cand
        e_new = energy(model, Z)
        dE = e_new - e_old
        if dE <= 0 or rng.random() < np.exp(-beta * dE):
            acc += 1
            s0[i] = [cand.conj() @ model.ops[i][p] @ cand
                     for p in range(model.n_ops)]
        else:
            Z[i] = old
    return acc / max(model.L, 1)


def thermalize(model, Z, kT, n_sweeps, rng, sigma=0.5):
    beta = 1.0 / float(kT)
    for _ in range(int(n_sweeps)):
        metropolis_sweep(model, Z, beta, rng, sigma)
    return Z


def moment_of(model, Z, q_cart):
    """The neutron magnetic-moment amplitude M^a(q) for one configuration.

    Uses `model.moment_terms`, so an ENTANGLED unit contributes its q-dependent
    staggered combination sum_k e^{i q.d_k} S_k rather than its total spin -- the
    same operator the LSWT structure factor uses, which is what makes a dimer's
    singlet-triplet excitation visible at all.
    """
    q = np.asarray(q_cart, float)
    out = np.zeros(3, dtype=complex)
    pos = model.pos if model.pos is not None else np.zeros((model.L, 3))
    for i in range(model.L):
        ph_site = np.exp(-1j * float(np.dot(q, pos[i])))
        for (d_k, idx) in model.moment_terms[i]:
            ph = ph_site * np.exp(1j * float(np.dot(q, d_k)))
            for a in range(3):
                out[a] += ph * (Z[i].conj() @ model.ops[i][idx[a]] @ Z[i])
    return out


def sampled_correlations(model, q_cart, kT, dt=0.02, n_steps=512, n_traj=4,
                         therm_sweeps=200, record_every=1, cross_section="perp",
                         seed=0, sigma=0.5, classical_to_quantum=True,
                         subtract_elastic=False, random_start=False):
    """S(q,w) from CP^(N-1) trajectories on thermally sampled coherent states.

    The SU(N) counterpart of `classical_dynamics.sampled_correlations`: thermalize by
    Metropolis on CP^(N-1), evolve deterministically, Fourier transform the moment
    M^a(q, t) in time. `classical_to_quantum` applies the same correspondence factor
    as the dipole path (Gap 4 #17).

    THE MODEL MUST BE BIG ENOUGH FOR THE q YOU ASK FOR. This is real-space dynamics:
    a chemical cell of two sites cannot represent a spin wave at q = 0.3, and asking
    for one returns that two-site system's own normal mode instead -- a plausible
    number that is not the thing you wanted (it looks like a factor-of-2 error).
    Build the model on a supercell first:

        SUNModel.from_generic_model(m, supercell=[[16,0,0],[0,1,0],[0,0,1]])

    and use q commensurate with it. LSWT needs none of this because it works in
    q-space; this does.

    Trajectories START FROM `model.Z` -- the model's reference state, i.e. the ground
    state after `minimize_energy` -- and are then thermalized. Starting from RANDOM
    coherent states instead is wrong at low temperature: Metropolis cannot walk from
    a random point of CP^(N-1) down to the ordered state in a feasible number of
    sweeps, so the "low-T" ensemble is really a stuck high-energy one and its
    spectrum has nothing to do with magnons. Pass `random_start=True` only well above
    the ordering temperature. This mirrors Sunny's workflow: minimize, then thermalize.

    `subtract_elastic` removes the time-average of M(q, t), i.e. the ELASTIC line.
    Off by default, matching Sunny, whose S(q,w) contains it. Turn it on to isolate
    the inelastic response: for an ordered state at low T the static moment dominates,
    and on a coarse energy grid its delta smears over several bins and can outweigh
    the magnon peak entirely.
    """
    from ..classical_dynamics import _contract, classical_to_quantum_factor

    qs = np.asarray(q_cart, float).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    n_rec = int(np.ceil(n_steps / record_every))
    energies = 2 * np.pi * np.fft.fftfreq(n_rec, d=dt * record_every)[:n_rec // 2]
    acc = np.zeros((n_rec // 2, len(qs)))

    for _ in range(int(n_traj)):
        if random_start:
            Z = []
            for i in range(model.L):
                n = model.Ns[i]
                v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
                Z.append(v / np.linalg.norm(v))
        else:
            Z = [np.asarray(z, complex).copy() for z in model.Z]
        thermalize(model, Z, kT, therm_sweeps, rng, sigma)
        traj = evolve(model, Z, dt, n_steps, record_every)
        # M^a(q, t) -> M^a(q, w)
        Mt = np.array([[moment_of(model, cfg, q) for q in qs] for cfg in traj])
        if subtract_elastic:
            Mt = Mt - Mt.mean(axis=0, keepdims=True)
        Mw = np.fft.fft(Mt, axis=0)[:n_rec // 2]          # (n_w, n_q, 3)
        for iq, q in enumerate(qs):
            tensor = np.einsum("wa,wb->wab", Mw[:, iq, :].conj(), Mw[:, iq, :])
            acc[:, iq] += np.real(_contract(tensor, q, cross_section)) / (
                n_rec * max(model.L, 1))

    sqw = acc / int(n_traj)
    if classical_to_quantum:
        sqw = sqw * classical_to_quantum_factor(energies, kT)[:, None]
    return energies, sqw
