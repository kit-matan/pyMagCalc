"""The ground-state guard for the one path that cannot grow one for free: KPM.

Every other spectrum path here refuses to expand about a non-minimum without being
asked -- `_bogoliubov` Choleskys H2, that fails when H2 is not positive definite, and
`on_imaginary` turns the failure into a hard error. KPM never diagonalizes, so it has
neither a Cholesky to fail nor an imaginary energy to report: about a saddle or a
maximum it returns a smooth, plausible, meaningless S(q,w). `SUNModel.is_stable_at` /
`assert_stable` are the check it has to make explicitly, and this file pins them.

THE ORACLES, in the order they are used below -- no golden numbers anywhere:

  * a CLOSED FORM for the curvature. Put the FERROMAGNETIC state on an
    ANTIFERROMAGNETIC Hamiltonian and the classical energy is a stationary MAXIMUM,
    whose exact H2 is minus the magnon energy of the same model with J -> -J:

        min eig H2(q) = -S J (z - sum_d cos q.d)          (S = 1/2, one boson)

    The sign flip is the whole point -- flip the exchange, flip the curvature -- and
    the right-hand side is the engine's OWN validated ferromagnet dispersion (pinned
    separately against Sunny), so this is an identity between two things the engine
    computes by different routes, not a number anyone typed in.

  * an EXACT EIGENSOLVE for the cheap machinery. The guard is a shifted Cholesky and
    the reporting number a bisection over that shift, both chosen because `eigvalsh`
    is 45x more expensive at 2D = 1800; every verdict and every number they produce is
    checked here against `eigvalsh` on the same matrix.

  * the two EXISTING guards, run on the same model, as the control that says this
    check is not redundant: the frustrated ferromagnetic chain below is a genuine
    in-cell minimum with max|Im w| = 0 exactly, so the energy audit and the
    imaginary-mode check both PASS it while it is unstable to a spiral that no cell
    of that size can hold. A run of it with `tasks: {dispersion: true}` succeeds and
    returns a plausible spectrum. That is the failure this guard exists to catch.
"""
import copy
import os

import numpy as np
import pytest
import yaml

from magcalc import runner
from magcalc.generic_model import GenericSpinModel
from magcalc.sun.kpm import kpm_sqw
from magcalc.sun.lswt import H2_REL_TOL, SUNModel, apply_bond_disorder
from magcalc.sun.operators import coherent_from_direction

SQ_LAT = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 10.0]]
SQ_B = 2 * np.pi * np.linalg.inv(np.array(SQ_LAT, float)).T
SQ_OFFS = ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0])

TRI = np.array([[3.0, 0, 0], [-1.5, 3 * np.sqrt(3) / 2, 0], [0, 0, 10.0]])
TRI_B = 2 * np.pi * np.linalg.inv(TRI).T
TRI_OFFS = ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [1, 1, 0], [-1, -1, 0])
K120 = np.array([1 / 3, 1 / 3, 0.0])

CHAIN_LAT = [[1.0, 0, 0], [0, 6.0, 0], [0, 0, 6.0]]
CHAIN_B = 2 * np.pi * np.linalg.inv(np.array(CHAIN_LAT, float)).T


def _square(J, S=0.5):
    """One site per cell, `z` = 4 neighbours, in the FERROMAGNETIC state. With J > 0
    (antiferromagnetic) that state is a stationary MAXIMUM."""
    cfg = {"crystal_structure": {"lattice_vectors": SQ_LAT,
                                 "atoms_uc": [{"label": "A", "pos": [0, 0, 0],
                                               "spin_S": S}]},
           "interactions": {"heisenberg": [{"pair": ["A", "A"], "rij_offset": list(o),
                                            "value": J} for o in SQ_OFFS]},
           "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "parameters": {}, "parameter_order": []}
    return SUNModel.from_generic_model(GenericSpinModel(cfg), [])


def _chain_cfg(J1=-1.0, J2=0.5, S=0.5, **calculation):
    """A FERROMAGNETIC chain (J1 < 0) frustrated by an antiferromagnetic J2.

    For J2 > |J1| / 4 the classical ground state is an incommensurate SPIRAL, so the
    ferromagnetic state is a saddle -- but a saddle only with respect to modulations
    the one-site cell cannot represent, which is exactly why the two existing guards
    are blind to it (and why the dipole engine's Luttinger-Tisza guard, which would
    catch it, does not run in SU(N) mode).
    """
    return {"crystal_structure": {"lattice_vectors": CHAIN_LAT,
                                  "atoms_uc": [{"label": "A", "pos": [0, 0, 0],
                                                "spin_S": S}]},
            "interactions": {"heisenberg":
                             [{"pair": ["A", "A"], "rij_offset": list(o), "value": J1}
                              for o in ([1, 0, 0], [-1, 0, 0])] +
                             [{"pair": ["A", "A"], "rij_offset": list(o), "value": J2}
                              for o in ([2, 0, 0], [-2, 0, 0])]},
            "magnetic_structure": {"type": "pattern",
                                   "pattern_type": "ferromagnetic",
                                   "direction": [0, 0, 1]},
            "parameters": {}, "parameter_order": [],
            "calculation": dict(dict(mode="SUN", cache_mode="none"), **calculation),
            "q_path": {"Gamma": [0.0, 0, 0], "X": [0.5, 0, 0],
                       "path": ["Gamma", "X"], "points_per_segment": 9},
            "kpm": {"e_min": 0.0, "e_max": 3.0, "e_step": 0.25, "fwhm": 0.3,
                    "tol": 0.1},
            "tasks": {"kpm_sqw": True},
            "output": {"save_data": False},
            "plotting": {"enabled": False, "save_plot": False, "show_plot": False}}


def _chain_model(J1=-1.0, J2=0.5, S=0.5):
    return SUNModel.from_generic_model(GenericSpinModel(_chain_cfg(J1, J2, S)), [])


def _triangular_120(n):
    """The exact 120-degree ground state on an n x n cell -- NON-COLLINEAR, with a
    Goldstone zero at Gamma and at K."""
    cfg = {"crystal_structure": {"lattice_vectors": TRI.tolist(),
                                 "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0],
                                               "spin_S": 0.5}]},
           "interactions": {"heisenberg": [{"pair": ["A", "A"], "rij_offset": list(o),
                                            "value": 1.0} for o in TRI_OFFS]},
           "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "parameters": {}, "parameter_order": []}
    m = SUNModel.from_generic_model(GenericSpinModel(cfg), [],
                                    supercell=[[n, 0, 0], [0, n, 0], [0, 0, 1]])
    ang = 2 * np.pi * ((np.asarray(m.pos, float) @ np.linalg.inv(TRI)) @ K120)
    m.Z = [coherent_from_direction(s, d) for s, d in
           zip(m.S, np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1))]
    m._prepare()
    assert abs(m.energy_per_site() + 0.375) < 1e-10, "not the 120-degree ground state"
    return m


def _exact_min_eig(model, q_cart):
    return float(np.linalg.eigvalsh(model.h2_matrix(q_cart)).min())


# ---------------------------------------------------------------------------
# 1. The closed form, and the case the imaginary-mode check provably cannot see
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("J", [1.0, 2.5])
@pytest.mark.parametrize("q_rlu", [[0.3, 0.2, 0], [0.5, 0.5, 0], [0.1, 0, 0],
                                   [0.25, 0, 0]])
def test_min_h2_eigenvalue_matches_the_closed_form_at_a_maximum(J, q_rlu):
    """min eig H2 = -S J (z - sum_d cos q.d) for the ferromagnetic state of an
    ANTIFERROMAGNET -- minus the engine's own ferromagnet dispersion, which is the
    statement that flipping the exchange flips the curvature."""
    m, q = _square(J), np.array(q_rlu, float) @ SQ_B
    want = -0.5 * J * (4 - 2 * np.cos(q[0]) - 2 * np.cos(q[1]))
    assert m.min_h2_eigenvalue(q, method="exact") == pytest.approx(want, abs=1e-10)
    assert m.min_h2_eigenvalue(q, method="bisect") == pytest.approx(want, abs=1e-8)


def test_the_sign_flip_against_the_engines_own_ferromagnet_dispersion():
    """The other half of the same identity: the AFM's H2 curvature at the FM state is
    exactly MINUS the magnon energy of the model with J -> -J, band by band, computed
    by a different route (`dispersion`, i.e. eigenvalues of g H2)."""
    afm, fm = _square(1.0), _square(-1.0)
    for q_rlu in ([0.3, 0.2, 0], [0.25, 0, 0], [0.5, 0.5, 0], [0.13, 0.41, 0]):
        q = np.array(q_rlu, float) @ SQ_B
        assert _exact_min_eig(afm, q) == pytest.approx(-float(fm.dispersion(q)[0]),
                                                       abs=1e-12)


@pytest.mark.parametrize("q_rlu", [[0.3, 0.2, 0], [0.5, 0.5, 0], [0.25, 0, 0]])
def test_a_stationary_maximum_is_INVISIBLE_to_the_imaginary_mode_check(q_rlu):
    """Why H2 >= 0 and not max|Im w|. At a stationary maximum H2 stays diagonal, so
    g H2 has an entirely real spectrum and the imaginary-mode guard -- the one the
    runner already runs -- reads exactly 0.0 while the state is as wrong as a state
    can be. The H2 check is strictly sharper, and this is where they part."""
    m, q = _square(1.0), np.array(q_rlu, float) @ SQ_B
    assert m.max_imaginary(q) < 1e-12, "the weaker check is supposed to be blind here"
    assert _exact_min_eig(m, q) < -0.1, "test model must actually be a maximum"
    assert not m.is_stable_at(q)


# ---------------------------------------------------------------------------
# 2. No false alarms: true ground states, including at their Goldstone q
# ---------------------------------------------------------------------------

def test_a_true_ground_state_is_stable_everywhere_including_gamma():
    """A ferromagnet's H2 is EXACTLY ZERO at Gamma (and a Goldstone mode puts an exact
    zero eigenvalue at the ordering wavevector of any gapless magnet). An unshifted
    positive-definiteness test refuses those, and Gamma is in every path ever plotted,
    so this is the false-alarm case the shift and the q-independent scale exist for."""
    m = _square(-1.0)
    for q_rlu in ([0, 0, 0], [0.3, 0.2, 0], [0.5, 0.5, 0], [0.001, 0, 0]):
        q = np.array(q_rlu, float) @ SQ_B
        assert m.is_stable_at(q), f"false alarm at q = {q_rlu}"
    assert m.assert_stable(np.array([[0, 0, 0]], float) @ SQ_B)["stable"]


def test_a_non_collinear_ground_state_is_stable_at_gamma_and_at_K():
    """The 120-degree state on a 9x9 cell: non-collinear, with the Goldstone zero
    folded onto Gamma_super. Its energy is the exact -0.375, so any refusal here is
    the guard's fault and not the state's."""
    m = _triangular_120(9)
    qs = np.array([[0, 0, 0], K120, [0.5, 0.0, 0.0], [0.2, 0.1, 0.0],
                   [0.37, -0.11, 0.0]], float) @ TRI_B
    rep = m.assert_stable(qs)
    assert rep["stable"] and rep["n_unstable"] == 0 and rep["n_checked"] == len(qs)


def test_the_frustrated_chain_is_stable_below_its_transition_and_not_above():
    """J2 = |J1|/4 is the exact classical boundary at which the ferromagnet gives way
    to a spiral, so the guard's verdict is pinned to an analytic phase boundary rather
    than to a threshold anyone chose: J2 = 0.1 stable at every q, J2 = 0.5 not."""
    qs = np.array([[h, 0, 0] for h in np.linspace(0, 0.5, 21)], float) @ CHAIN_B
    assert _chain_model(J2=0.1).assert_stable(qs, on_failure="off")["stable"]
    assert not _chain_model(J2=0.5).assert_stable(qs, on_failure="off")["stable"]


# ---------------------------------------------------------------------------
# 3. The cheap machinery against the exact eigensolve
# ---------------------------------------------------------------------------

def test_the_verdict_agrees_with_the_exact_criterion_point_by_point():
    """`is_stable_at` is a shifted Cholesky; it must decide exactly what
    `eigvalsh(H2).min() > -eps` decides, at every q, on both sides of the boundary --
    including the q where the frustrated chain's band touches zero."""
    for J2 in (0.1, 0.5):
        m = _chain_model(J2=J2)
        eps = H2_REL_TOL * m._reference_h2_scale()
        for h in np.linspace(0, 0.5, 26):
            q = np.array([h, 0, 0], float) @ CHAIN_B
            assert m.is_stable_at(q) == (_exact_min_eig(m, q) > -eps), \
                f"J2={J2} h={h}: min eig {_exact_min_eig(m, q):+.3e}, eps {eps:.3e}"


def test_bisection_reproduces_the_eigensolve_on_a_disordered_supercell():
    """The reporting number. Bisection over the Cholesky shift is what runs instead of
    `eigvalsh` on a large cell, so it is checked against `eigvalsh` on a cell small
    enough to afford both -- on a DISORDERED, non-collinear state, where the smallest
    eigenvalue is not protected by any symmetry."""
    m = _triangular_120(6)
    apply_bond_disorder(m, 1 / 3, seed=0)
    m.minimize_energy(n_restarts=1, max_iter=2000, tol=1e-15)
    scale = m._reference_h2_scale()
    for q_rlu in ([0.2, 0.1, 0], [0.37, -0.11, 0], [0, 0, 0], K120, [0.44, 0.07, 0]):
        q = np.asarray(q_rlu, float) @ TRI_B
        assert m.min_h2_eigenvalue(q, method="bisect") == pytest.approx(
            _exact_min_eig(m, q), abs=1e-6 * scale)


def test_auto_picks_the_exact_route_only_while_it_is_affordable():
    """`method='auto'` is a cost switch, not a physics one, so it must agree with both
    named routes wherever they are both available."""
    m = _chain_model(J2=0.5)                      # 2D = 2, far below the switch
    q = np.array([0.15, 0, 0], float) @ CHAIN_B
    assert m.min_h2_eigenvalue(q) == m.min_h2_eigenvalue(q, method="exact")
    assert m.min_h2_eigenvalue(q, method="bisect") == pytest.approx(
        m.min_h2_eigenvalue(q, method="exact"), abs=1e-8)
    with pytest.raises(ValueError, match="method must be"):
        m.min_h2_eigenvalue(q, method="lanczos")


def test_the_shared_hamiltonian_shortcut_changes_nothing():
    """The guard and the spectrum share ONE build of g H2 per q -- that is what keeps
    the check at a few percent of the KPM cost. A shared build must give bit-identical
    answers to an independent one, on both sides."""
    m = _triangular_120(3)
    q = np.array([0.2, 0.1, 0], float) @ TRI_B
    H = np.asarray(m.hamiltonian(q), complex)
    egrid = np.arange(0.0, 3.0001, 0.1)
    a = kpm_sqw(m, q, egrid, 0.2, n_moments=60).intensities
    b = kpm_sqw(m, q, egrid, 0.2, n_moments=60, hmat=H).intensities
    assert np.array_equal(a, b)
    assert np.allclose(m.h2_matrix(q), m.h2_matrix(q, hmat=H), atol=0, rtol=0)
    assert m.is_stable_at(q, hmat=H) == m.is_stable_at(q)


def test_kpm_at_a_goldstone_point_is_finite_and_says_it_is_undefined(caplog):
    """Found while pointing a guarded KPM run at a path that starts at Gamma: for a
    ferromagnet H(Gamma) is IDENTICALLY zero, so the spectral bound gamma is zero and
    Â = D̂/gamma divided 0 by 0 -- the whole q-column came back NaN, silently, into the
    saved map, the plot and the logged intensity range. It is now finite, and warns:
    the value is 0 rather than the q -> 0 limit (0.587 here), because with every mode
    at omega = 0 the +/-omega poles coincide and cancel."""
    m = _square(-1.0)
    egrid = np.linspace(0, 4, 41)
    caplog.clear()
    with caplog.at_level("WARNING"):
        at_gamma = kpm_sqw(m, np.zeros(3), egrid, 0.4, n_moments=200).intensities
    assert np.all(np.isfinite(at_gamma)), "NaN column at Gamma"
    assert "identically zero" in caplog.text
    near = kpm_sqw(m, np.array([1e-4, 0, 0]) @ SQ_B, egrid, 0.4,
                   n_moments=200).intensities
    assert near.max() > 0.5, "the neighbouring q must carry the weight"


# ---------------------------------------------------------------------------
# 4. `assert_stable`'s three modes
# ---------------------------------------------------------------------------

def test_assert_stable_raises_warns_or_stays_quiet(caplog):
    """The three spellings are `calculation.on_imaginary`'s, so a config that has
    knowingly downgraded the other two guards downgrades this one with them."""
    m = _chain_model(J2=0.5)
    qs = np.array([[h, 0, 0] for h in (0.1, 0.15, 0.2, 0.4)], float) @ CHAIN_B
    with pytest.raises(ValueError, match="NOT a classical minimum"):
        m.assert_stable(qs)
    caplog.clear()
    with caplog.at_level("WARNING"):
        rep = m.assert_stable(qs, on_failure="warn")
    assert "NOT a classical minimum" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        rep_off = m.assert_stable(qs, on_failure="off")
    assert caplog.text == ""
    for r in (rep, rep_off):
        assert r["stable"] is False and r["n_checked"] == 4 and r["n_unstable"] == 3
        assert r["min_eig"] < -1e-3 and r["q_worst"] is not None
    with pytest.raises(ValueError, match="on_failure must be"):
        m.assert_stable(qs, on_failure="raise")


# ---------------------------------------------------------------------------
# 5. Through the runner, which is where a config meets it
# ---------------------------------------------------------------------------

def _run(tmp_path, cfg, name="c.yaml"):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        runner.run_calculation(path)
        return None
    except Exception as e:                       # noqa: BLE001 - message is the assert
        return str(e)
    finally:
        os.chdir(cwd)


def test_the_two_EXISTING_guards_pass_this_model(tmp_path):
    """The control, and the reason this item was not closed by the guards already
    there. The frustrated ferromagnetic chain is a genuine minimum within its cell and
    has an entirely real spectrum, so the energy audit and the imaginary-mode check
    both pass it -- a `dispersion` run of exactly this config succeeds and plots a
    perfectly plausible band. Only the per-q H2 check knows better."""
    cfg = _chain_cfg()
    cfg["tasks"] = {"dispersion": True}
    assert _run(tmp_path, cfg) is None


def test_the_runner_refuses_a_KPM_spectrum_about_a_non_minimum(tmp_path):
    msg = _run(tmp_path, _chain_cfg())
    assert msg is not None, "kpm_sqw ran about a non-minimum"
    assert "NOT a classical minimum" in msg and "KPM" in msg


def test_on_imaginary_off_lets_the_KPM_run_through(tmp_path):
    """The escape hatch has to work, or the guard gets removed instead of downgraded."""
    assert _run(tmp_path, _chain_cfg(on_imaginary="off")) is None


def test_a_stable_model_runs_the_KPM_task_untouched(tmp_path):
    """The guard must not cost a correct config anything -- the same config below its
    frustration boundary runs to completion."""
    assert _run(tmp_path, _chain_cfg(J2=0.1)) is None


def test_kpm_is_treated_as_an_LSWT_task_by_the_up_front_guards(tmp_path):
    """`kpm_sqw` was missing from the runner's `_lswt_tasks` list, so pairing it with
    any classical task -- which is a natural thing to do, thermodynamics plus the
    spectrum of the same model -- silenced the up-front ground-state guards for the
    whole run. The structure here is a ferromagnet on an antiferromagnet, i.e. the
    energy audit's own case, so it must be refused even in that combination."""
    cfg = _chain_cfg(J1=1.0, J2=0.0)             # AFM chain, ferromagnetic structure
    cfg["tasks"] = {"kpm_sqw": True, "thermal_mc": True}
    cfg["thermal_mc"] = {"temperatures": [1.0], "supercell": [4, 1, 1],
                         "n_sweeps": 10, "n_equil": 5}
    msg = _run(tmp_path, copy.deepcopy(cfg))
    assert msg is not None and "NOT a classical" in msg
