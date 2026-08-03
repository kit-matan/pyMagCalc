"""Wang-Landau density of states (Gap 4 #22) -- the last Tier-2 remnant.

Flat-histogram sampling of g(E), from which the thermodynamics at EVERY temperature
follows in one run (the density of states does not depend on T). Metropolis and
parallel tempering need a separate simulation per temperature.

ORACLE. GAP4_PLAN proposed Beale's exact 2-D Ising g(E). That does not apply here:
these are CONTINUOUS classical Heisenberg spins, not Ising. The exact result that
does is better, because it pins g(E) itself rather than only its consequences:

    For ONE classical dimer, E = J S_1.S_2 = J S^2 cos(theta), and cos(theta) is
    UNIFORMLY distributed for two random unit vectors. So g(E) is exactly CONSTANT
    on [-J S^2, +J S^2] -- a closed form for the whole density of states.

The thermodynamics that follows from it, <E>(T) = -J S^2 L(beta J S^2), is the same
closed form `test_thermal_mc.py` pins the Metropolis sampler to, so the two samplers
must agree with one another through it.
"""
import numpy as np
import pytest

from magcalc.thermal_mc import (parallel_tempering, wang_landau,
                                wang_landau_window)

S, J = 1.0, 1.0


def _L(x):
    return 1.0 / np.tanh(x) - 1.0 / x


def _dimer():
    """One classical Heisenberg dimer: E = 1/2 m^T H m = J S_1.S_2."""
    H = np.zeros((6, 6))
    for a in range(3):
        H[a, 3 + a] = J
        H[3 + a, a] = J
    return H, np.zeros(6), 2


def _run(f_final=1e-5, seed=1, n_bins=40):
    H, b, N = _dimer()
    a = J * S * S
    return wang_landau(H, b, N, S, -a * 1.001, a * 1.001, n_bins=n_bins,
                       f_final=f_final, sweeps_per_check=400, seed=seed)


def test_window_estimate_brackets_the_exact_energy_range():
    """The window must contain every reachable state: anything outside it is silently
    unreachable, which biases g(E) rather than failing."""
    H, b, N = _dimer()
    lo, hi = wang_landau_window(H, b, N, S, seed=0)
    a = J * S * S
    assert lo <= -a and hi >= a
    assert lo == pytest.approx(-a, abs=0.15) and hi == pytest.approx(a, abs=0.15)


def test_density_of_states_of_a_dimer_is_flat():
    """THE closed-form check: g(E) is constant for a classical dimer, so ln g must be
    flat. A wrong acceptance rule or a mis-binned energy tilts it immediately."""
    res = _run()
    occ = res.histogram > 0
    lg = res.log_g[occ]
    assert occ.sum() > 30, "most bins should be visited"
    assert lg.max() - lg.min() < 0.35, f"ln g spread {lg.max() - lg.min():.3f}"
    # and it must be flat, not merely bounded: no systematic slope across the window
    E = res.energies[occ]
    slope = np.polyfit(E, lg, 1)[0]
    assert abs(slope) < 0.25, f"ln g has a systematic slope {slope:.3f}"


@pytest.mark.parametrize("kT", [0.4, 0.8, 1.5, 3.0])
def test_thermodynamics_from_g_matches_the_exact_dimer(kT):
    """One g(E), every temperature: <E>(T) = -J S^2 L(beta J S^2) exactly -- the same
    closed form the Metropolis sampler is pinned to in test_thermal_mc.py."""
    res = _run()
    E, _C = res.thermodynamics([kT])
    a = J * S * S
    assert E[0] == pytest.approx(-a * _L(a / kT) / 2.0, abs=0.01)


def test_one_run_covers_every_temperature():
    """The whole point of Wang-Landau. Reconstructing a T sweep must need no further
    sampling, and must agree with the exact curve across it."""
    res = _run()
    temps = np.array([0.3, 0.5, 0.9, 1.4, 2.2, 4.0])
    E, C = res.thermodynamics(temps)
    a = J * S * S
    exact = np.array([-a * _L(a / T) / 2.0 for T in temps])
    assert E == pytest.approx(exact, abs=0.012)
    assert np.all(C > 0), "heat capacity must be positive"


def test_refinement_actually_ran():
    """f must have been reduced many times; a run that never flattens would return a
    meaningless g(E) without complaining."""
    res = _run()
    assert res.n_refinements >= 8
    assert res.f_final <= 1e-5


def test_bad_window_is_refused():
    H, b, N = _dimer()
    with pytest.raises(ValueError, match="e_max > e_min"):
        wang_landau(H, b, N, S, 1.0, -1.0)
    with pytest.raises(RuntimeError, match="widen the window"):
        wang_landau(H, b, N, S, 50.0, 60.0, n_bins=10, f_final=0.5)


@pytest.mark.slow
def test_agrees_with_parallel_tempering_on_the_same_model():
    """Two independent samplers, one model. Wang-Landau walks in energy space with a
    g(E)-dependent acceptance; parallel tempering walks in configuration space at
    fixed temperatures. They must land on the same <E>(T)."""
    H, b, N = _dimer()
    temps = np.array([0.5, 1.0, 2.0])
    pt = parallel_tempering(H, b, N, S, temps, n_sweeps=20000, n_equil=5000, seed=5)
    wl_E, _ = _run(f_final=1e-6, seed=7).thermodynamics(temps)
    assert wl_E == pytest.approx(pt.energy, abs=0.02)


@pytest.mark.slow
def test_finer_convergence_flattens_g_further():
    """A smaller final modification factor must reduce the residual error in ln g --
    if it does not, the run is converging to something other than the true g(E)."""
    coarse = _run(f_final=1e-3, seed=3)
    fine = _run(f_final=1e-6, seed=3)

    def spread(res):
        lg = res.log_g[res.histogram > 0]
        return float(lg.max() - lg.min())

    assert spread(fine) < spread(coarse)
