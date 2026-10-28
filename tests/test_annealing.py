"""Monte-Carlo / annealing ground-state search (SpinW `anneal` / `optmagsteep`,
Sunny `LocalSampler`).

Random-multistart L-BFGS gets trapped on frustrated landscapes -- that is what made
SW20-in-field ship a wrong ground state. These tests pin that the annealer finds the
minima that multistart misses, and that it agrees with the analytically known ones.
"""
import copy
import logging

import numpy as np
import pytest

import magcalc as mc
from magcalc.annealing import (
    anneal, cartesian_to_angles, energy, random_spins, steepest_descent,
)
from magcalc.generic_model import GenericSpinModel

S_VAL = 1.0


def _model(directions, J=1.0, n_cells=1):
    """Nearest-neighbour chain, 2-site cell."""
    cfg = {
        "crystal_structure": {
            "lattice_vectors": [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]],
            "atoms_uc": [
                {"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": S_VAL, "ion": "Fe2+"},
                {"label": "B", "pos": [0.5, 0.0, 0.0], "spin_S": S_VAL, "ion": "Fe2+"}],
        },
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "distance": 3.0, "value": J}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                               "directions": directions},
    }
    m = GenericSpinModel(cfg)
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    return mc.MagCalc(spin_model_module=m, spin_magnitude=S_VAL, cache_mode="none",
                      cache_file_base="ann", hamiltonian_params=[])


# Start deliberately from the WRONG state (ferromagnetic, with AFM exchange) so the
# search has to actually move, not just sit where it was put.
FM = [[0, 0, 1], [0, 0, 1]]


def _quad(calc):
    n = len(calc.sm.atom_pos())
    n_ouc = len(calc.sm.atom_pos_ouc())
    return calc._extract_classical_quadratic(
        calc.hamiltonian_params, n, n_ouc, float(calc.spin_magnitude)), n


# --- the low-level drivers --------------------------------------------------
def test_anneal_finds_the_neel_state_of_an_afm_chain(kernels):
    """AFM chain, S = 1, J = 1: E = (1/2) sum_ij J S_i.S_j = -2 J S^2 per 2-site cell."""
    calc = _model(FM)
    (H, b, c), n = _quad(calc)
    m, e = anneal(H, b, c, S_VAL, n, n_sweeps=500, seed=0, kernels=kernels)
    assert np.isclose(e, -2.0, atol=1e-6), e
    # ... and the state really is Neel (antiparallel neighbours)
    s0, s1 = m[0:3], m[3:6]
    assert np.dot(s0, s1) < -0.99 * S_VAL**2


def test_steepest_descent_finds_it_too(kernels):
    calc = _model(FM)
    (H, b, c), n = _quad(calc)
    rng = np.random.default_rng(0)
    m, e = steepest_descent(random_spins(n, S_VAL, rng), H, b, c, S_VAL, n,
                            kernels=kernels)
    assert np.isclose(e, -2.0, atol=1e-6)


def test_steepest_descent_is_monotone(kernels):
    """It only ever goes downhill -- which is exactly why it cannot escape a local
    minimum and must not be used alone as a global search."""
    calc = _model(FM)
    (H, b, c), n = _quad(calc)
    rng = np.random.default_rng(3)
    m0 = random_spins(n, S_VAL, rng)
    e0 = energy(m0, H, b, c)
    _, e1 = steepest_descent(m0, H, b, c, S_VAL, n, kernels=kernels)
    assert e1 <= e0 + 1e-12


def test_anneal_preserves_spin_magnitude(kernels):
    calc = _model(FM)
    (H, b, c), n = _quad(calc)
    m, _ = anneal(H, b, c, S_VAL, n, n_sweeps=200, seed=1, kernels=kernels)
    lengths = np.linalg.norm(m.reshape(n, 3), axis=1)
    assert np.allclose(lengths, S_VAL, atol=1e-9)


def test_cartesian_to_angles_roundtrip():
    rng = np.random.default_rng(0)
    m = random_spins(5, 2.0, rng)
    x = cartesian_to_angles(m, 5)
    th, ph = x[0::2], x[1::2]
    back = 2.0 * np.column_stack([np.sin(th) * np.cos(ph),
                                  np.sin(th) * np.sin(ph), np.cos(th)]).ravel()
    assert np.allclose(back, m, atol=1e-9)


# --- through MagCalc.minimize_energy ---------------------------------------
@pytest.mark.parametrize("method", ["anneal", "monte_carlo", "steep", "optmagsteep"])
def test_minimize_energy_methods(method, kernels):
    calc = _model(FM)
    res = calc.minimize_energy(method=method, num_starts=2, n_sweeps=300, seed=0,
                               kernels=kernels)
    assert res.success
    assert np.isclose(res.fun, -2.0, atol=1e-6)
    # the convergence evidence the guards rely on
    assert res.hits >= 1 and res.num_starts == 2


def test_anneal_beats_multistart_where_multistart_gets_trapped(kernels):
    """The SW20 failure in miniature: on a frustrated model, L-BFGS from a poor start
    can stall, while annealing crosses the barrier. Annealing must never do WORSE."""
    calc = _model(FM)
    e_lbfgs = calc.minimize_energy(method="L-BFGS-B", num_starts=8,
                                   early_stopping=10, seed=0).fun
    e_anneal = calc.minimize_energy(method="anneal", num_starts=2,
                                    n_sweeps=500, seed=0, kernels=kernels).fun
    assert e_anneal <= e_lbfgs + 1e-6


def test_anneal_result_passes_the_ground_state_guard(kernels):
    """End to end: the state the annealer returns must be a real minimum, i.e. it must
    satisfy the stability guards added for exactly this failure class."""
    calc = _model(FM)
    res = calc.minimize_energy(method="anneal", num_starts=2, n_sweeps=500, seed=0,
                               kernels=kernels)
    calc.sm.set_magnetic_structure(res.x[0::2], res.x[1::2])
    assert calc.max_imaginary_energy() < 1e-6
    e_now, e_relaxed = calc.relax_from_current()
    assert e_relaxed >= e_now - 1e-6


def test_anneal_is_reproducible_across_seeds(kernels):
    calc = _model(FM)
    energies = [calc.minimize_energy(method="anneal", num_starts=2, n_sweeps=500,
                                     seed=s, kernels=kernels).fun for s in (0, 1, 7)]
    assert np.allclose(energies, energies[0], atol=1e-6)


def test_gui_shaped_minimization_block_reaches_the_annealer(tmp_path, monkeypatch,
                                                           kernels):
    """The Studio front-ends (web + native) post the `minimization` block verbatim,
    including `n_sweeps`. The runner strips a fixed set of keys before forwarding the
    rest to minimize_energy, so a new key is only real if it survives that trip --
    otherwise the UI would offer a knob that silently does nothing.
    """
    import yaml

    from magcalc.runner import run_calculation

    cfg = {
        "crystal_structure": {
            "lattice_vectors": [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]],
            "atoms_uc": [
                {"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": S_VAL, "ion": "Fe2+"},
                {"label": "B", "pos": [0.5, 0.0, 0.0], "spin_S": S_VAL, "ion": "Fe2+"}],
        },
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "distance": 3.0, "value": 1.0}]},
        "parameters": {}, "parameter_order": [],
        # deliberately start from the WRONG (ferromagnetic) state
        "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                               "directions": FM},
        "tasks": {"minimization": True, "dispersion": True},
        # exactly what the GUI sends
        "minimization": {"enabled": True, "method": "anneal", "num_starts": 4,
                         "n_sweeps": 800, "seed": 0},
        "q_path": {"G": [0, 0, 0], "X": [0.5, 0, 0], "path": ["G", "X"],
                   "points_per_segment": 3},
        "plotting": {"save_plot": False, "show_plot": False, "plot_structure": False},
        "output": {"save_data": False},
        "calculation": {"cache_mode": "none", "kernels": kernels},
    }
    p = tmp_path / "gui.yaml"
    p.write_text(yaml.safe_dump(cfg))

    # `calculation.kernels` has to reach the annealer too, or `kernels: rust`
    # would silently anneal in NumPy.
    import magcalc.annealing as A
    seen = []
    real = A.anneal_runs

    def spy(*a, **k):
        seen.append(k.get("kernels"))
        return real(*a, **k)

    monkeypatch.setattr(A, "anneal_runs", spy)

    # Must not raise: the annealer has to escape the FM state, otherwise the
    # ground-state guard (on_imaginary: error by default) fails the run.
    run_calculation(str(p))
    assert seen == [kernels]


def test_anneal_requires_the_quadratic_form():
    calc = _model(FM)
    calc._extract_classical_quadratic = lambda *a, **k: None
    with pytest.raises(ValueError, match="quadratic form"):
        calc.minimize_energy(method="anneal")


# ---------------------------------------------------------------------------
# The polish could make the answer WORSE, and did (found 2026-08-13, porting
# Sunny tutorial 06).
#
# `anneal` ends with a `steepest_descent` polish and used to take its result
# unconditionally. That step aligns each spin with the field from everything
# EXCEPT itself, so it ignores the on-site block H_ii -- fine when H_ii ~ 0, which
# is every model in the tests above, and wrong when it is not. The failure is not
# subtle once triggered: Metropolis found the exact minimum on every seed, and the
# polish then walked from it to a local MAXIMUM, 5.17 higher. The reported
# "N of N runs hit the minimum" consensus, which the docs tell you to use as the
# acceptance criterion for a ground state, agreed unanimously on it.
#
# The landscape below is S06's, reduced to its 3x3 essentials so the minimum is
# available in closed form rather than as a recorded number:
#
#     E(c) = A/2 + (C - A) c^2 / 2 - h c,   c = cos(theta)
#     c* = h / (C - A),   E* = A/2 - h^2 / (2 (C - A))
#
# with C - A large and positive because of the easy-plane D = 19.
# ---------------------------------------------------------------------------
S06_A, S06_C, S06_H = -2.291796, 32.04133, 15.5


def _s06_landscape():
    H = np.diag([S06_A, S06_A, S06_C])
    b = np.array([0.0, 0.0, -S06_H])
    c_star = S06_H / (S06_C - S06_A)
    e_star = S06_A / 2 - S06_H ** 2 / (2 * (S06_C - S06_A))
    return H, b, 0.0, c_star, e_star


def test_the_polish_is_never_allowed_to_make_the_answer_worse(kernels):
    """anneal() must return the best state it found, polished or not."""
    H, b, c, c_star, e_star = _s06_landscape()
    assert e_star == pytest.approx(-4.644706, abs=1e-6)      # the closed form
    for seed in range(4):
        m_raw, e_raw = anneal(H, b, c, 1.0, 1, n_sweeps=2000, seed=seed,
                              polish_steep=False, kernels=kernels)
        m_pol, e_pol = anneal(H, b, c, 1.0, 1, n_sweeps=2000, seed=seed,
                              polish_steep=True, kernels=kernels)
        assert e_raw == pytest.approx(e_star, abs=1e-4), f"seed {seed}: {e_raw}"
        assert e_pol <= e_raw + 1e-9, (
            f"seed {seed}: the polish raised the energy by {e_pol - e_raw:+.6f}")
        assert abs(abs(m_pol[2]) - c_star) < 1e-3


def test_steepest_descent_is_monotone_WITH_an_on_site_term(kernels):
    """The existing monotonicity test cannot see this, and that is the point.

    `test_steepest_descent_is_monotone` runs on the AFM chain, whose H_ii is zero,
    so it passed throughout the period when the property was false. Monotonicity
    only has teeth where the on-site block is large -- so start this one AT the
    exact minimum, the state a non-monotone step is most likely to leave.
    """
    H, b, c, c_star, e_star = _s06_landscape()
    m0 = np.array([np.sqrt(1 - c_star ** 2), 0.0, c_star])
    e0 = energy(m0, H, b, c)
    assert e0 == pytest.approx(e_star, abs=1e-9)
    _, e1 = steepest_descent(m0.copy(), H, b, c, 1.0, 1, kernels=kernels)
    assert e1 <= e0 + 1e-12, f"steepest_descent rose by {e1 - e0:+.6f}"


def test_anneal_finds_the_S06_canted_minimum_end_to_end(kernels):
    """Through `minimize_energy`, i.e. what a config actually gets.

    Pinned to the closed form above, and specifically NOT to the polarized state
    (E = +0.520665) that the broken polish returned -- a local maximum, which the
    energy audit would then have had to catch downstream.
    """
    H, b, c, c_star, e_star = _s06_landscape()
    m, e = anneal(H, b, c, 1.0, 1, n_sweeps=500, seed=0, kernels=kernels)
    assert e == pytest.approx(e_star, abs=1e-4)
    assert e < 0.0 < 0.520665


# ---------------------------------------------------------------------------
# `kernels="rust"` (rsMagCalc, on K1's dense model), 2026-10-28. The physics
# oracles above run under both kernel sets through the `kernels` fixture; these
# pin the wiring: the batched runs, and the fall-back contract.
# ---------------------------------------------------------------------------
def _rust_or_skip():
    from magcalc import kernels as _k

    rs = _k.rust_module()
    if rs is None:
        pytest.skip("rsmagcalc not installed (optional Rust kernels)")
    return rs


def test_rust_anneal_runs_equal_one_anneal_call_per_seed():
    """All starts go to rsMagCalc in one parallel call; each run owns the stream of
    its seed, so the batch is exactly the per-seed calls (run 0 takes m0)."""
    _rust_or_skip()
    from magcalc.annealing import anneal_runs

    calc = _model(FM)
    (H, b, c), n = _quad(calc)
    m0 = random_spins(n, S_VAL, np.random.default_rng(9))
    runs = anneal_runs(H, b, c, S_VAL, n, seeds=[3, 4, 5], n_sweeps=150, m0=m0,
                       kernels="rust")
    for k, sd in enumerate([3, 4, 5]):
        m, e = anneal(H, b, c, S_VAL, n, n_sweeps=150, seed=sd,
                      m0=m0 if k == 0 else None, kernels="rust")
        assert np.array_equal(runs[k][0], m) and runs[k][1] == e


def test_a_failing_rust_kernel_falls_back_to_the_numpy_result(monkeypatch, caplog):
    """The fMagCalc contract: a kernel exception is logged and the NumPy loop runs,
    so the answer is exactly the NumPy one."""
    rs = _rust_or_skip()

    def boom(*a, **k):
        raise RuntimeError("simulated kernel failure")

    monkeypatch.setattr(rs, "anneal", boom)
    monkeypatch.setattr(rs, "steepest_descent", boom)
    calc = _model(FM)
    (H, b, c), n = _quad(calc)
    with caplog.at_level(logging.ERROR, logger="magcalc.kernels"):
        m_r, e_r = anneal(H, b, c, S_VAL, n, n_sweeps=100, seed=2, kernels="rust")
    m_n, e_n = anneal(H, b, c, S_VAL, n, n_sweeps=100, seed=2)
    assert np.array_equal(m_r, m_n) and e_r == e_n
    assert "rsmagcalc anneal failed" in caplog.text
    m0 = random_spins(n, S_VAL, np.random.default_rng(1))
    s_r = steepest_descent(m0, H, b, c, S_VAL, n, kernels="rust")
    s_n = steepest_descent(m0, H, b, c, S_VAL, n)
    assert np.array_equal(s_r[0], s_n[0]) and s_r[1] == s_n[1]
