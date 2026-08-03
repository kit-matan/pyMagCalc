"""Langevin thermostat, implicit-midpoint integrator, suggested timestep (Gap 4 #18).

`classical_dynamics` thermalized by Metropolis and evolved with undamped RK4. Added:

  * `langevin_step` / `langevin_thermalize` -- the stochastic Landau-Lifshitz
    thermostat (Sunny `Langevin`), an alternative way to prepare thermal states;
  * `implicit_midpoint_step` and `evolve(..., integrator="midpoint")` -- symplectic,
    so the energy error stays BOUNDED instead of drifting (Sunny `ImplicitMidpoint`);
  * `suggest_timestep` from the largest local field (Sunny `suggest_timestep`).

The oracles were already in this repo, which is what made #18 cheap: a thermostat
that does not sample Boltzmann fails the exact free-spin Langevin function and the
exact classical-dimer <E>(T) that `test_thermal_mc.py` pins the Metropolis sampler to.
Reusing them is deliberate -- two independent samplers must agree with the same
closed-form answer.

(GAP4_PLAN suggested Beale's exact 2-D Ising g(E) as an oracle. It does not apply
here: these are CONTINUOUS classical Heisenberg spins, not Ising. The Langevin
function and the dimer partition function are the exact results that do.)
"""
import numpy as np
import pytest

from magcalc.classical_dynamics import (energy, evolve, implicit_midpoint_step,
                                        langevin_step, langevin_thermalize,
                                        local_field, suggest_timestep)


def _L(x):
    """Langevin function coth(x) - 1/x."""
    return 1.0 / np.tanh(x) - 1.0 / x


def _free_spins(N, h):
    """E = -h * sum_i S^z_i: N non-interacting spins in a field along +z."""
    H = np.zeros((3 * N, 3 * N))
    b = np.zeros(3 * N)
    b[2::3] = -h
    return H, b


def _random_state(N, S, seed):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((N, 3))
    m *= S / np.linalg.norm(m, axis=1, keepdims=True)
    return m


# --------------------------------------------------------------------------
# Langevin thermostat
# --------------------------------------------------------------------------
def test_damping_relaxes_toward_the_energy_minimum():
    """REGRESSION, and the reason this test exists first.

    `local_field` returns the energy GRADIENT G = dE/dS, not the physical field
    B = -G. Writing the damping term with the textbook field's sign gives
    -(lam/S) S x (S x G), which relaxes spins AWAY from the minimum: the
    magnetization comes out with the right magnitude and the WRONG SIGN. That is
    exactly the kind of plausible answer this project keeps getting caught by, and
    only an exact reference distinguishes it.

    Deterministic check (kT = 0, no noise): the magnetization must grow along +z.
    """
    N, S, h = 50, 1.0, 0.8
    H, b = _free_spins(N, h)
    m = _random_state(N, S, seed=5)
    rng = np.random.default_rng(0)
    start = m[:, 2].mean()
    # relaxation rate ~ lambda * h = 0.4, so a few 1/e times is a few tens of time units
    for _ in range(3000):
        m = langevin_step(H, b, m, 0.02, S, kT=1e-12, damping=0.5, rng=rng)
    assert m[:, 2].mean() > start
    assert m[:, 2].mean() == pytest.approx(S, abs=1e-3), "should fully polarize"


@pytest.mark.parametrize("kT", [0.4, 1.0])
def test_free_spins_sample_the_langevin_function(kT):
    """The thermostat's stationary distribution must be Boltzmann: <S^z>/S = L(hS/kT)
    exactly, for the same model `test_thermal_mc.py` pins the Metropolis sampler to.
    Two independent samplers, one closed-form answer."""
    N, S, h = 400, 1.0, 0.8
    H, b = _free_spins(N, h)
    m = _random_state(N, S, seed=1)
    rng = np.random.default_rng(2)
    dt = 0.02
    for _ in range(3000):
        m = langevin_step(H, b, m, dt, S, kT, 0.5, rng)
    acc = []
    for _ in range(3000):
        m = langevin_step(H, b, m, dt, S, kT, 0.5, rng)
        acc.append(m[:, 2].mean())
    assert np.mean(acc) / S == pytest.approx(_L(h * S / kT), abs=0.01)


def test_thermalize_is_independent_of_the_starting_state():
    """Two very different starts must reach the same equilibrium magnetization --
    otherwise the run is measuring the initial condition, not the temperature."""
    N, S, h, kT = 300, 1.0, 0.8, 0.7
    H, b = _free_spins(N, h)
    cold = np.tile(np.array([0.0, 0.0, S]), (N, 1))
    hot = _random_state(N, S, seed=9)

    def equilibrium(start, seed):
        """TIME-average after equilibrating: a single snapshot of N = 300 spins has a
        statistical spread of ~0.05, which is larger than the effect being tested."""
        m = langevin_thermalize(H, b, S, start, kT, 0.02, 3000, damping=0.5, seed=seed)
        rng = np.random.default_rng(seed + 100)
        acc = []
        for _ in range(3000):
            m = langevin_step(H, b, m, 0.02, S, kT, 0.5, rng)
            acc.append(m[:, 2].mean())
        return float(np.mean(acc))

    a, c = equilibrium(cold, 3), equilibrium(hot, 4)
    assert a == pytest.approx(c, abs=0.02)
    assert a / S == pytest.approx(_L(h * S / kT), abs=0.02)


@pytest.mark.slow
@pytest.mark.parametrize("damping", [0.1, 0.5, 2.0])
def test_equilibrium_does_not_depend_on_the_damping(damping):
    """Damping sets how FAST equilibrium is reached, never where it is. A
    damping-dependent answer means the fluctuation-dissipation balance is wrong --
    which is the one way to get a thermostat subtly, consistently wrong."""
    N, S, h, kT = 400, 1.0, 0.8, 1.0
    H, b = _free_spins(N, h)
    m = _random_state(N, S, seed=7)
    rng = np.random.default_rng(11)
    for _ in range(6000):
        m = langevin_step(H, b, m, 0.01, S, kT, damping, rng)
    acc = []
    for _ in range(6000):
        m = langevin_step(H, b, m, 0.01, S, kT, damping, rng)
        acc.append(m[:, 2].mean())
    assert np.mean(acc) / S == pytest.approx(_L(h * S / kT), abs=0.015)


# --------------------------------------------------------------------------
# Implicit midpoint
# --------------------------------------------------------------------------
def _random_hamiltonian(N, seed, scale=0.3):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3 * N, 3 * N))
    return scale * 0.5 * (A + A.T), np.zeros(3 * N)


def test_midpoint_conserves_energy_far_better_than_rk4():
    """Symplectic vs not: over a long trajectory RK4's energy error accumulates while
    the midpoint rule's stays bounded. Measured here: ~8e-5 vs ~1e-12."""
    N, S = 8, 1.0
    H, b = _random_hamiltonian(N, seed=3)
    m0 = _random_state(N, S, seed=4)
    dt = 3 * suggest_timestep(H, b, m0, S)

    def drift(integrator):
        traj = evolve(H, b, S, m0, dt, 4000, record_every=100, integrator=integrator)
        E = np.array([energy(H, b, x) for x in traj])
        return float(np.abs(E - E[0]).max())

    d_rk4, d_mid = drift("rk4"), drift("midpoint")
    assert d_mid < 1e-9
    assert d_mid < 1e-4 * d_rk4


def test_midpoint_conserves_spin_length_without_renormalizing():
    """The midpoint update is orthogonal to the midpoint spin, so |S_i| is conserved
    by the SCHEME. RK4 needs an explicit rescale; masking a solver failure with one
    here would hide exactly the divergence worth seeing."""
    N, S = 6, 1.5
    H, b = _random_hamiltonian(N, seed=8)
    m = _random_state(N, S, seed=2)
    dt = suggest_timestep(H, b, m, S)
    for _ in range(500):
        m = implicit_midpoint_step(H, b, m, dt, S)
    assert np.linalg.norm(m, axis=1) == pytest.approx(np.full(N, S), abs=1e-9)


def test_midpoint_and_rk4_agree_at_short_time():
    """Both integrate the same equation, so over a few steps they must agree to the
    order of the scheme -- a check that 'conserves energy' is not being achieved by
    integrating the wrong dynamics."""
    N, S = 5, 1.0
    H, b = _random_hamiltonian(N, seed=12)
    m0 = _random_state(N, S, seed=13)
    dt = 0.1 * suggest_timestep(H, b, m0, S)
    a = evolve(H, b, S, m0, dt, 20, integrator="rk4")[-1]
    c = evolve(H, b, S, m0, dt, 20, integrator="midpoint")[-1]
    assert c == pytest.approx(a, abs=1e-6)


def test_unknown_integrator_raises():
    N, S = 3, 1.0
    H, b = _random_hamiltonian(N, seed=1)
    with pytest.raises(ValueError, match="rk4.*midpoint"):
        evolve(H, b, S, _random_state(N, S, 1), 0.01, 2, integrator="verlet")


# --------------------------------------------------------------------------
# suggest_timestep
# --------------------------------------------------------------------------
def test_suggest_timestep_scales_inversely_with_the_local_field():
    """dt ~ 1/omega_max: doubling every coupling must halve the suggested step."""
    N, S = 6, 1.0
    H, b = _random_hamiltonian(N, seed=21)
    m = _random_state(N, S, seed=22)
    dt1 = suggest_timestep(H, b, m, S)
    dt2 = suggest_timestep(2 * H, 2 * b, m, S)
    assert dt2 == pytest.approx(0.5 * dt1, rel=1e-12)


def test_suggested_timestep_actually_conserves_energy():
    """The point of the suggestion is that a run at that step is stable."""
    N, S = 8, 1.0
    H, b = _random_hamiltonian(N, seed=31)
    m0 = _random_state(N, S, seed=32)
    dt = suggest_timestep(H, b, m0, S)
    traj = evolve(H, b, S, m0, dt, 2000, record_every=200, integrator="rk4")
    E = np.array([energy(H, b, x) for x in traj])
    assert np.abs(E - E[0]).max() < 1e-6 * max(1.0, abs(E[0]))


def test_suggest_timestep_refuses_an_empty_hamiltonian():
    N, S = 4, 1.0
    with pytest.raises(ValueError, match="local field vanishes"):
        suggest_timestep(np.zeros((3 * N, 3 * N)), np.zeros(3 * N),
                         _random_state(N, S, 1), S)
