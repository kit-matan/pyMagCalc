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


# --------------------------------------------------------------------------
# The vectorized inner loop.
#
# WHY THIS EXISTS. The obvious implementation -- ask `SUNModel.local_field(i, .)`
# for each site -- scans the ENTIRE bond list once per site, so it costs
# O(sites x bonds) = O(sites^2). That is invisible in LSWT, where the local field
# is built a handful of times during the ground-state search, and it decides
# whether a quench is possible at all: it put Sunny tutorial 06's own system size
# (L = 40, 1600 sites) at ~16 s/step, i.e. 55 hours for the tutorial's 25600 steps.
#
# The rewrite is a change of loop order, not of physics: build every site's field
# from the bond list ONCE (bonds are grouped by source with `np.bincount`), and
# never form the N x N matrices h_i at all -- only h_i|Z_i>, which is what both
# flows actually use. `tests/test_sun_quench.py::test_the_fast_derivative_equals
# _the_local_field_definition` pins it against the `local_field` definition,
# including the biquadratic case where the operator set is 12 wide rather than 3.
def _fast(model):
    """Stacked arrays for the vectorized derivative, cached on the model.

    Invalidated by (id, len) of the three lists it reads, so replacing or growing
    `model.bonds` (as a test isolating a single site does) rebuilds rather than
    silently propagating a stale Hamiltonian.
    """
    key = tuple((id(x), len(x)) for x in (model.bonds, model.onsite, model.ops))
    cache = getattr(model, "_dyn_fast", None)
    if cache is not None and cache["key"] == key:
        return cache

    Ns = list(model.Ns)
    if len(set(Ns)) != 1:
        # Mixed spin: the per-site Hilbert spaces have different dimensions, so
        # they do not stack. Fall back to the per-site loop rather than padding.
        cache = {"key": key, "ok": False}
    else:
        L, nop, N = model.L, model.n_ops, Ns[0]
        A = np.zeros((L, N, N), dtype=complex)
        for (k, Amat) in model.onsite:
            A[k] += Amat
        cache = {
            "key": key, "ok": True, "L": L, "nop": nop, "N": N, "A": A,
            "ops": np.array([[np.asarray(model.ops[i][p], dtype=complex)
                              for p in range(nop)] for i in range(L)]),
            "bi": np.array([b[0] for b in model.bonds], dtype=int),
            "bj": np.array([b[1] for b in model.bonds], dtype=int),
            "J": np.array([np.asarray(b[3], dtype=complex) for b in model.bonds]),
        }
    model._dyn_fast = cache
    return cache


def _h_times_z(model, Zs):
    """h_i |Z_i> for every site at once, shape (L, N).

    The mean field is the same one `SUNModel.local_field` documents,

        h_i = sum_{bonds from i} sum_ab J_ab <O_j>^b O_i^a  +  A_i,

    so h_i|Z_i> = A_i|Z_i> + sum_a c_ia (O^a_i |Z_i>) with c the bond sum. Only the
    N-vectors O^a|Z> are ever formed; the matrices h_i are not.
    """
    f = _fast(model)
    OZ = np.einsum("ipab,ib->ipa", f["ops"], Zs)             # O^p |Z_i>
    s0 = np.einsum("ia,ipa->ip", Zs.conj(), OZ)              # <Z_i|O^p|Z_i>
    c = np.zeros((f["L"], f["nop"]), dtype=complex)
    if len(f["bi"]):
        contrib = np.einsum("kab,kb->ka", f["J"], s0[f["bj"]])
        for a in range(f["nop"]):                            # group bonds by source
            c[:, a] = (np.bincount(f["bi"], contrib[:, a].real, minlength=f["L"])
                       + 1j * np.bincount(f["bi"], contrib[:, a].imag,
                                          minlength=f["L"]))
    return np.einsum("iab,ib->ia", f["A"], Zs) + np.einsum("ip,ipa->ia", c, OZ)


def _as_array(Z):
    return np.array([np.asarray(z, dtype=complex) for z in Z])


def _deriv_arr(model, Zs):
    return -1j * _h_times_z(model, Zs)


def _damped_deriv_arr(model, Zs, damping):
    hZ = _h_times_z(model, Zs)
    exp_h = np.real(np.einsum("ia,ia->i", Zs.conj(), hZ))
    return -1j * hZ - float(damping) * (hZ - exp_h[:, None] * Zs)


def _deriv(model, Z):
    """dZ_i/dt = -i h_i Z_i."""
    if _fast(model)["ok"]:
        return list(_deriv_arr(model, _as_array(Z)))
    hs = local_hamiltonians(model, Z)
    return [-1j * (hs[i] @ Z[i]) for i in range(model.L)]


def _renormalize(Z):
    if isinstance(Z, np.ndarray):
        return Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return [z / np.linalg.norm(z) for z in Z]


def _rk4(f, Z, dt):
    """One RK4 step of dZ/dt = f(Z), renormalized (removes O(dt^5) drift).

    The overall phase of each Z_i is unphysical -- every observable is a
    sesquilinear form <Z|O|Z> -- so no phase fixing is needed or wanted.
    """
    k1 = f(Z)
    k2 = f(Z + 0.5 * dt * k1)
    k3 = f(Z + 0.5 * dt * k2)
    k4 = f(Z + dt * k3)
    return _renormalize(Z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


def rk4_step(model, Z, dt):
    """One RK4 step of the CP^(N-1) equations."""
    if _fast(model)["ok"]:
        return list(_rk4(lambda s: _deriv_arr(model, s), _as_array(Z), dt))
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
    if _fast(model)["ok"]:
        Zs = _as_array(Z0)
        rec = []
        for step in range(int(n_steps)):
            if step % record_every == 0:
                rec.append(list(Zs.copy()))
            Zs = _rk4(lambda s: _deriv_arr(model, s), Zs, dt)
        return rec
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
    as the dipole path (Gap 4 #17), and the result carries the same ABSOLUTE
    normalization as `SUNModel.structure_factor` -- per chemical cell, with the 1/2pi
    of the time transform -- so it can be compared with the LSWT spectrum directly.

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

    The per-trajectory transform is `dynamical_structure_factor` below; call it
    directly for the whole frequency axis (`two_sided`) or to hand it a trajectory
    you built yourself.

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
    dt_rec = dt * record_every
    energies = None
    acc = None

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
        e, sqw = dynamical_structure_factor(
            model, traj, qs, dt_rec, cross_section=cross_section,
            subtract_elastic=subtract_elastic)
        acc = sqw if acc is None else acc + sqw
        energies = e

    sqw = acc / int(n_traj)
    if classical_to_quantum:
        sqw = sqw * classical_to_quantum_factor(energies, kT)[:, None]
    return energies, sqw


def dynamical_structure_factor(model, traj, q_cart, dt, cross_section="perp",
                               subtract_elastic=False, two_sided=False):
    """S(q,w) from ONE CP^(N-1) trajectory, on the absolute LSWT/Sunny scale.

        S^ab(q,w) = (1/2pi) int dt e^{-iwt} <M^a(q,0)* M^b(q,t)> / n_cells

    with M(q) the neutron moment amplitude `moment_of` builds (the staggered
    combination for entangled units). Split out of `sampled_correlations` so the
    normalization has a per-trajectory oracle: its defining property is the
    equal-time sum rule int dw S = <|M(q)|^2>/n_cells, checked at machine precision
    in `tests/test_classical_absolute_normalization.py`.

    Both factors below were missing until 2026-08-13, exactly as in the dipole path:
    the transform kept the bare `np.fft.fft` sum (2pi/dt too large, and dependent on
    the time step), and the divisor was the SITE count where `SUNModel.structure_factor`
    -- the spectrum this is meant to be compared with -- divides by `n_cells`.
    """
    from ..classical_dynamics import _contract

    qs = np.asarray(q_cart, float).reshape(-1, 3)
    Mt = np.array([[moment_of(model, cfg, q) for q in qs] for cfg in traj])
    if subtract_elastic:
        Mt = Mt - Mt.mean(axis=0, keepdims=True)
    n_rec = len(traj)
    energies = 2 * np.pi * np.fft.fftfreq(n_rec, d=dt)
    Mw = np.fft.fft(Mt, axis=0)                              # (n_rec, n_q, 3)
    norm = dt / (2 * np.pi * n_rec * max(int(getattr(model, "n_cells", 1)), 1))
    sl = slice(None) if two_sided else slice(0, n_rec // 2)
    out = np.zeros((n_rec if two_sided else n_rec // 2, len(qs)))
    for iq, q in enumerate(qs):
        tensor = np.einsum("wa,wb->wab", Mw[sl, iq, :].conj(), Mw[sl, iq, :])
        out[:, iq] = np.real(_contract(tensor, q, cross_section)) * norm
    if two_sided:
        order = np.argsort(energies)
        return energies[order], out[order]
    return energies[sl], out

def damped_deriv(model, Z, damping):
    """dZ_i/dt = -i h_i Z_i - lambda (h_i - <h_i>) Z_i -- dissipative CP^(N-1) flow.

    The SU(N) analogue of Landau-Lifshitz-Gilbert, and what a QUENCH needs: Sunny's
    `Langevin(; damping, kT=0)`. Two exact properties fix the form and its sign, and
    both are asserted in `tests/test_sun_dynamics.py` rather than assumed -- the
    dipole damping sign was wrong on first writing (Gap 4 #18) and produced a
    right-magnitude, wrong-sign answer:

      * NORM. d|Z|^2/dt = 2 Re[-i<h> - lambda(<h> - <h>)] = 0, since <h> is real for
        Hermitian h. So |Z_i| = 1 is preserved by the flow itself.
      * ENERGY. dE/dt = 2 Re<dZ|h|Z> = -2 lambda (<h^2> - <h>^2) = -2 lambda Var(h),
        which is <= 0 and vanishes exactly when Z is an EIGENVECTOR of h -- the
        self-consistent mean-field condition. So the flow relaxes to a stationary
        state of the same equations the ground-state search solves.

    Sign check, single site: with eigenvalues eps_0 < eps_1 and Z = a|0> + b|1>, the
    |1> component obeys db/dt = -lambda(eps_1 - <h>) b and eps_1 - <h> = |a|^2
    (eps_1 - eps_0) > 0, so |b| decays: population flows toward the ground state.
    """
    if _fast(model)["ok"]:
        return list(_damped_deriv_arr(model, _as_array(Z), damping))
    lam = float(damping)
    hs = local_hamiltonians(model, Z)
    out = []
    for i in range(model.L):
        hZ = hs[i] @ Z[i]
        exp_h = np.real(Z[i].conj() @ hZ)
        out.append(-1j * hZ - lam * (hZ - exp_h * Z[i]))
    return out


def damped_step(model, Z, dt, damping):
    """One RK4 step of the damped flow."""
    if _fast(model)["ok"]:
        return list(_rk4(lambda s: _damped_deriv_arr(model, s, damping),
                         _as_array(Z), dt))

    def f(state):
        return damped_deriv(model, state, damping)
    k1 = f(Z)
    k2 = f([Z[i] + 0.5 * dt * k1[i] for i in range(model.L)])
    k3 = f([Z[i] + 0.5 * dt * k2[i] for i in range(model.L)])
    k4 = f([Z[i] + dt * k3[i] for i in range(model.L)])
    out = [Z[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
           for i in range(model.L)]
    return _renormalize(out)


def quench(model, Z0, dt, n_steps, damping=0.05, record_at=(), callback=None):
    """Damped relaxation from `Z0` (Sunny's randomize + Langevin at kT = 0).

    Returns (final state, {step: snapshot}). A quench is NOT the same as finding the
    ground state: it lands in whatever metastable texture the initial condition flows
    to, which for the right parameters is a skyrmion lattice. Metropolis or
    `minimize_energy` would destroy exactly the object of interest.

    `callback(step, Z)` is invoked at every recorded step, for progress reporting on
    a run long enough to need it.
    """
    fast = _fast(model)["ok"]
    Z = _as_array(Z0) if fast else [np.asarray(z, complex).copy() for z in Z0]
    want = set(int(t) for t in record_at)
    snaps = {}
    for step in range(int(n_steps) + 1):
        if step in want:
            snaps[step] = list(Z.copy()) if fast else [z.copy() for z in Z]
            if callback is not None:
                callback(step, snaps[step])
        if step < int(n_steps):
            Z = (_rk4(lambda s: _damped_deriv_arr(model, s, damping), Z, dt)
                 if fast else damped_step(model, Z, dt, damping))
    return (list(Z) if fast else Z), snaps


def dipole_field(model, Z):
    """<S^a_i> for every site, shape (L, 3) -- the texture a skyrmion plot shows."""
    return np.array([[np.real(Z[i].conj() @ model.ops[i][a] @ Z[i]) for a in range(3)]
                     for i in range(model.L)])


def topological_charge(spins, neighbours, min_length=1e-3):
    """Berg-Luescher skyrmion number of a 2-D DIPOLE texture.

    `spins` is (N, 3); `neighbours` a list of oriented triangles (i, j, k). The solid
    angle of each spherical triangle is summed and divided by 4 pi. That is an
    INTEGER for any closed texture -- exactly, not approximately: it is the degree of
    the discrete map, so it can be asserted rather than eyeballed off a plot.

    RAISES ON A COLLAPSED DIPOLE, which is not a technicality for SU(N) models. The
    formula normalizes every spin to the unit sphere, so a site whose dipole moment
    has (nearly) vanished contributes an arbitrary direction and the "charge" becomes
    noise that still comes back quantized-looking. An S = 1 site in a strong
    easy-plane field sits in exactly that state -- the quadrupolar |m=0> level, the
    white background of Sunny tutorial 06 -- so use `sun_topological_charge`, which
    is defined on the coherent states themselves and needs no dipole at all.
    """
    s = np.asarray(spins, float)
    lengths = np.linalg.norm(s, axis=1)
    if float(lengths.min()) < min_length:
        raise ValueError(
            f"topological_charge: {int((lengths < min_length).sum())} of {len(s)} "
            f"sites have |<S>| < {min_length} (smallest {lengths.min():.2e}). The "
            f"Berg-Luescher solid angle normalizes each spin, so a collapsed dipole "
            f"contributes an arbitrary direction and the result is noise. For an "
            f"SU(N) texture with quadrupolar sites use sun_topological_charge().")
    s = s / lengths[:, None]
    total = 0.0
    for (i, j, k) in neighbours:
        a, b, c = s[i], s[j], s[k]
        num = float(np.dot(a, np.cross(b, c)))
        den = 1.0 + float(np.dot(a, b)) + float(np.dot(b, c)) + float(np.dot(c, a))
        total += 2.0 * np.arctan2(num, den)
    return total / (4.0 * np.pi)


def berry_curvature(Z, neighbours):
    """SU(N) Berry phase of each oriented plaquette: arg(<z_i|z_j><z_j|z_k><z_k|z_i>).

    Sunny tutorial 06's `sun_berry_curvature`, and the quantity its figure plots.
    This is the CP^(N-1) skyrmion density: it is built from the coherent states
    directly, so it stays well defined where the DIPOLE vanishes -- which is most of
    the area in that tutorial, whose background is a quadrupolar paramagnet.
    """
    z = np.asarray(Z, dtype=complex)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    out = np.empty(len(neighbours))
    for t, (i, j, k) in enumerate(neighbours):
        out[t] = np.angle((z[i].conj() @ z[j]) * (z[j].conj() @ z[k])
                          * (z[k].conj() @ z[i]))
    return out


def sun_topological_charge(Z, neighbours):
    """CP^(N-1) skyrmion number: the summed Berry phase over 2 pi.

    At N = 2 this EQUALS the dipole `topological_charge`, to 7e-16: the Berry phase
    of a spin-1/2 spherical triangle is half its solid angle, and the two
    normalizations (2 pi here, 4 pi there) absorb the factor. That is an exact
    identity and it is the oracle for this function -- it ties the SU(N) charge to
    the already-pinned dipole one with no new reference number
    (`tests/test_sun_quench.py::test_the_SUN_charge_reduces_to_the_dipole_one_at_N2`).
    """
    return float(np.sum(berry_curvature(Z, neighbours)) / (2.0 * np.pi))


def triangulate_lattice(pos, a1, a2, n1, n2):
    """Oriented triangles of an n1 x n2 lattice laid out cell-major (2 per cell).

    `pos` is only used for its length; the connectivity comes from the periodic
    index arithmetic, which is what makes the charge well defined on a torus.
    """
    def idx(u, v):
        return (u % n1) * n2 + (v % n2)
    tris = []
    for u in range(n1):
        for v in range(n2):
            tris.append((idx(u, v), idx(u + 1, v), idx(u, v + 1)))
            tris.append((idx(u + 1, v), idx(u + 1, v + 1), idx(u, v + 1)))
    return tris

