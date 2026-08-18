"""
The CLASSICAL ground-state search with more than one spin length per cell.

Everything downstream of the ground state already handled mixed spins: `gen_HM`
writes site i's Holstein-Primakoff expansion as `ratio_i * S_sym`
(`tests/test_mixed_spin_sun.py`), and the S(Q,w) prefactor is per-site
(`tests/test_intensity_layer.py`). The classical minimiser did not. It moved on
the constraint surface |m_i| = S_ref for EVERY site, `S_ref` being the first
atom's `spin_S`, because that is the scalar `MagCalc.spin_magnitude` carries.

That is not a rescaling. The classical energy is extracted as a quadratic form in
free Cartesian components -- correct for mixed spins already -- and the spin
lengths enter ONLY through the constraint the minimiser imposes on top of it. So
a uniform length changes which state minimises, whenever the minimising
directions depend on the lengths.

THE ORACLE, exact and analytic. For three moments with equal antiferromagnetic
exchange on all three bonds,

    E = J (m0.m1 + m1.m2 + m2.m0) = (J/2) (|sum m|^2 - sum |m_i|^2)

so the minimum is the CLOSED TRIANGLE `sum m_i = 0` whenever the three lengths
satisfy the triangle inequality -- and its angles are fixed entirely by the
LENGTHS, by the law of cosines:

    cos(theta_ij) = (|m_k|^2 - |m_i|^2 - |m_j|^2) / (2 |m_i| |m_j|)

For S = (1, 1, 1/2):  E = -1.125 J, theta_01 = 151.045 deg, theta_02 = 104.478 deg.
For S = (1, 1, 1):    E = -1.5   J, every angle 120 deg -- the uniform answer.

The old code returned the 120 deg state and E = -1.5 for BOTH, because -1.5 really
is the minimum of the energy it was minimising. It is not a state of the requested
Hamiltonian: evaluated with the true lengths that structure has E = -1.0, a
12.5% error, and LSWT (which does use the true per-site S) then finds it is not a
minimum and the ground-state guard kills the run -- pointing the user at
`num_starts`, which can never fix it.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from magcalc.annealing import (anneal, random_spins, spin_lengths,
                               steepest_descent)


def trimer_quadratic(J: float = 1.0):
    """E = J sum_<ij> m_i.m_j for a triangle, as (H, b, c) with E = 1/2 m^T H m."""
    H = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            if i != j:
                H[3 * i:3 * i + 3, 3 * j:3 * j + 3] = J * np.eye(3)
    return H, np.zeros(9), 0.0


def closed_triangle_angle(Si, Sj, Sk) -> float:
    """Law of cosines: the angle between m_i and m_j when the triangle closes."""
    return math.degrees(math.acos((Sk ** 2 - Si ** 2 - Sj ** 2) / (2 * Si * Sj)))


def best_anneal(H, b, c, S, n, runs=8, sweeps=4000):
    return min((anneal(H, b, c, S, n, n_sweeps=sweeps, seed=k) for k in range(runs)),
               key=lambda r: r[1])


# ---------------------------------------------------------------------------
# the helper that decides what "the spin length" means
# ---------------------------------------------------------------------------
def test_spin_lengths_broadcasts_a_scalar_and_checks_a_list():
    assert list(spin_lengths(1.5, 3)) == [1.5, 1.5, 1.5]
    assert list(spin_lengths([1.0, 1.0, 0.5], 3)) == [1.0, 1.0, 0.5]
    with pytest.raises(ValueError, match="for 3 sites"):
        spin_lengths([1.0, 0.5], 3)


def test_random_spins_gives_each_site_its_own_radius():
    S = np.array([2.5, 1.0, 0.5])
    m = random_spins(3, S, np.random.default_rng(0)).reshape(3, 3)
    assert np.allclose(np.linalg.norm(m, axis=1), S)


# ---------------------------------------------------------------------------
# the trimer, against the closed-triangle identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("S,expected_E", [
    ([1.0, 1.0, 1.0], -1.5),        # uniform: the 120 degree state, unchanged
    ([1.0, 1.0, 0.5], -1.125),
    ([2.5, 2.5, 2.0], -0.5 * (2.5 ** 2 + 2.5 ** 2 + 2.0 ** 2)),
    ([1.0, 0.8, 0.6], -0.5 * (1.0 + 0.64 + 0.36)),
])
def test_mixed_spin_trimer_finds_the_closed_triangle(S, expected_E):
    """
    E = (J/2)(|sum m|^2 - sum|m_i|^2), minimised by sum m_i = 0. So the energy is
    -(J/2) sum |m_i|^2 exactly -- a closed form in the LENGTHS alone, with no
    reference to the search that has to find it.
    """
    H, b, c = trimer_quadratic()
    m, E = best_anneal(H, b, c, S, 3)
    assert E == pytest.approx(expected_E, abs=1e-6)
    assert np.linalg.norm(m.reshape(3, 3).sum(0)) == pytest.approx(0.0, abs=1e-4)


def test_the_minimiser_honours_each_sites_own_spin_length():
    """The constraint itself: |m_i| = S_i, not S_ref for everyone."""
    S = [1.0, 1.0, 0.5]
    m, _ = best_anneal(*trimer_quadratic(), S, 3)
    assert np.linalg.norm(m.reshape(3, 3), axis=1) == pytest.approx(S, abs=1e-6)


def test_the_mixed_spin_angles_are_the_law_of_cosines_not_120_degrees():
    """
    The sharp discriminator. A uniform-length search returns 120 degrees for any
    spins; the true answer for (1, 1, 1/2) is 151.045 / 104.478, which is what
    the moment triangle has to be to close.
    """
    S = np.array([1.0, 1.0, 0.5])
    m = best_anneal(*trimer_quadratic(), S, 3)[0].reshape(3, 3)

    def angle(i, j):
        return math.degrees(math.acos(
            np.clip(m[i] @ m[j] / (S[i] * S[j]), -1.0, 1.0)))

    assert angle(0, 1) == pytest.approx(closed_triangle_angle(S[0], S[1], S[2]), abs=1e-3)
    assert angle(0, 2) == pytest.approx(closed_triangle_angle(S[0], S[2], S[1]), abs=1e-3)
    assert angle(1, 2) == pytest.approx(closed_triangle_angle(S[1], S[2], S[0]), abs=1e-3)
    assert angle(0, 1) == pytest.approx(151.045, abs=1e-2)
    assert abs(angle(0, 1) - 120.0) > 25.0          # nowhere near the uniform answer


def test_the_uniform_length_answer_is_not_a_state_of_the_mixed_hamiltonian():
    """
    What the old code returned, scored honestly. The 120 degree state with the
    TRUE lengths has E = -1.0, above the true minimum of -1.125 -- so LSWT about
    it is an expansion about a non-minimum, which is this project's #1 source of
    silently wrong spectra.
    """
    S = np.array([1.0, 1.0, 0.5])
    th = np.radians([0.0, 120.0, 240.0])
    m_120 = np.stack([S * np.cos(th), S * np.sin(th), np.zeros(3)], axis=1)
    E_120 = sum(m_120[i] @ m_120[j] for i, j in ((0, 1), (1, 2), (2, 0)))
    assert E_120 == pytest.approx(-1.0)
    assert E_120 > -1.125                        # strictly worse than the true GS


def test_steepest_descent_also_honours_the_per_site_lengths():
    """The `steep` / optmagsteep path shares the constraint and had the same bug."""
    S = np.array([1.0, 1.0, 0.5])
    H, b, c = trimer_quadratic()
    start = random_spins(3, S, np.random.default_rng(3))
    m, E = steepest_descent(start, H, b, c, S, 3)
    assert np.linalg.norm(m.reshape(3, 3), axis=1) == pytest.approx(S, abs=1e-9)
    assert E <= -1.0 + 1e-9                      # no worse than the 120 degree state


# ---------------------------------------------------------------------------
# uniform models must not move
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("S", [0.5, 1.0, 2.5])
def test_a_uniform_model_is_identical_whether_S_is_scalar_or_a_list(S):
    """
    The per-site path must reduce EXACTLY to the old scalar one, seed for seed --
    otherwise every validated ground state in the suite is up for re-checking.
    """
    H, b, c = trimer_quadratic()
    m_scalar, e_scalar = anneal(H, b, c, S, 3, n_sweeps=500, seed=7)
    m_list, e_list = anneal(H, b, c, [S] * 3, 3, n_sweeps=500, seed=7)
    assert e_list == pytest.approx(e_scalar, abs=0.0)
    assert np.array_equal(m_list, m_scalar)

    r_scalar = random_spins(3, S, np.random.default_rng(1))
    r_list = random_spins(3, [S] * 3, np.random.default_rng(1))
    assert np.array_equal(r_scalar, r_list)


# ---------------------------------------------------------------------------
# through MagCalc: the search and the energy-audit guard
# ---------------------------------------------------------------------------
TRIMER = {
    "crystal_structure": {
        "lattice_vectors": [[20.0, 0, 0], [0, 20.0, 0], [0, 0, 20.0]],
        "atom_mode": "explicit",
        "atoms_uc": [
            {"label": "A1", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0},
            {"label": "A2", "pos": [0.15, 0.0, 0.0], "spin_S": 1.0},
            {"label": "B1", "pos": [0.075, 0.129903811, 0.0], "spin_S": 0.5}],
    },
    "interactions": {"symmetry_rules": [
        {"type": "heisenberg", "distance": 3.0, "value": "J"}]},
    "parameters": {"J": 1.0}, "parameter_order": ["J"],
    "tasks": {"minimization": False, "dispersion": False},
    "plotting": {"save_plot": False, "show_plot": False, "plot_structure": False},
    "output": {"save_data": False},
    "calculation": {"cache_mode": "none"},
}

TRUE_E = -1.125          # (J/2)(0 - (1 + 1 + 0.25)), the closed triangle
UNIFORM_E = -1.5         # what a single-length search returns instead


def _trimer_calc(directions=None):
    import copy

    import magcalc as mc
    from magcalc.generic_model import GenericSpinModel

    cfg = copy.deepcopy(TRIMER)
    cfg["magnetic_structure"] = {
        "type": "pattern", "pattern_type": "generic",
        "directions": directions or [[0, 0, 1]] * 3}
    m = GenericSpinModel(cfg)
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    return mc.MagCalc(spin_model_module=m, spin_magnitude=1.0, cache_mode="none",
                      cache_file_base="mixedtrimer", hamiltonian_params=[1.0])


def test_MagCalc_reports_one_spin_length_per_site():
    calc = _trimer_calc()
    assert list(calc._classical_spin_lengths(3)) == [1.0, 1.0, 0.5]
    # and `spin_magnitude` stays the REFERENCE, which is what binds S_sym
    assert calc.spin_magnitude == 1.0


@pytest.mark.parametrize("method", ["anneal", "L-BFGS-B"])
def test_minimize_energy_finds_the_mixed_spin_ground_state(method):
    """
    Both search paths -- the annealer and the batched projected-gradient
    multistart -- constrain |m_i| = S_i, so both must land on -1.125 rather than
    the -1.5 a uniform-length search reports.
    """
    kw = {"n_sweeps": 4000} if method == "anneal" else {}
    res = _trimer_calc().minimize_energy(
        method=method, num_starts=24, seed=0, **kw)
    assert res.fun == pytest.approx(TRUE_E, abs=1e-5)
    assert res.fun > UNIFORM_E                  # -1.5 is not a state of this model


def test_the_energy_audit_guard_accepts_the_true_mixed_spin_ground_state():
    """
    The other half, and the one that made the bug fatal rather than merely wrong:
    `relax_from_current` scored the CORRECT structure at the uniform-length energy
    and "relaxed" it downhill to the 120 degree state, so supplying the right
    answer by hand was rejected too. There was no way to run this model at all.
    """
    S = np.array([1.0, 1.0, 0.5])
    t12, t13 = math.acos(-0.875), math.acos(-0.25)
    dirs = [[1.0, 0.0, 0.0],
            [math.cos(t12), math.sin(t12), 0.0],
            [math.cos(-t13), math.sin(-t13), 0.0]]
    e_now, e_relaxed = _trimer_calc(dirs).relax_from_current()
    assert e_now == pytest.approx(TRUE_E, abs=1e-6)
    assert e_relaxed >= e_now - 1e-6            # already a minimum: nowhere to fall


def test_the_energy_audit_still_rejects_a_genuine_non_minimum():
    """The guard must not have been loosened -- an all-parallel state still fails."""
    e_now, e_relaxed = _trimer_calc([[0, 0, 1]] * 3).relax_from_current()
    assert e_relaxed < e_now - 1e-6
    assert e_relaxed == pytest.approx(TRUE_E, abs=1e-5)
