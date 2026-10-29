"""The energy audit (ground-state guard 2) on a classical energy that is NOT quadratic.

`relax_from_current` relaxed on the quadratic form (H, b, c). A `biquadratic`, or a
`stevens` term of rank >= 4, has no such form, so until 2026-10-29 it returned None and
the runner skipped the audit without a log line. The imaginary-mode check cannot see a
stationary MAXIMUM (the +/- omega pairs stay real and are sorted), so such a model had no
protection against the failure this guard exists for. It now relaxes on E(m) and its
gradient (`MagCalc._classical_energy_general`).

The oracles are exact identities, none of them the engine:

* the 120-degree triangular AFM with J = 1, B = 0.06, s = 1 in a 3x3 supercell, the
  state Sunny 0.8.1 relaxes to and cannot lower
  (`tests/data_biquadratic_noncollinear_sunny.json`): E/site = 3(J s^2 cos 120 +
  B s^4 cos^2 120) = -1.455 meV, and the ferromagnet, E/site = 3(J + B) s^4-wise, is a
  stationary maximum the runner must refuse;
* one spin under B_4^0 O_4^0, whose classical polynomial is
  s^4 (35 c^4 - 30 c^2 + 3) with c = cos(theta): along z it is a stationary maximum at
  8 B s^4, and the minimum is the cone c^2 = 3/7 at -24/7 B s^4;
* the gradient against central differences of the same energy.
"""
import copy
import json
import logging
import os
import sys

import numpy as np
import pytest
import yaml

import magcalc as mc
from magcalc.generic_model import GenericSpinModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_biquadratic_noncollinear import SUNNY, _triangular_cfg  # noqa: E402

N_TRI = 9   # sites in the 3x3 supercell


def _calc(cfg, S, tag):
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    return mc.MagCalc(spin_model_module=m, spin_magnitude=S, cache_mode="none",
                      cache_file_base=tag, hamiltonian_params=[])


def _run(cfg, tmp_path, name):
    from magcalc.runner import run_calculation
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(json.loads(json.dumps(cfg, default=float))))
    return run_calculation(str(p))


# --------------------------------------------------------------------------
# One spin, B_4^0 O_4^0: a cheap closed form, kept OUT of `slow`
# --------------------------------------------------------------------------
RUN_BLOCKS = {
    "q_path": {"G": [0.0, 0.0, 0.0], "X": [0.5, 0.0, 0.0],
               "path": ["G", "X"], "points_per_segment": 4},
    "plotting": {"save_plot": False, "show_plot": False, "plot_structure": False},
    "output": {"save_data": False},
}
S4, B40, JFM = 2.0, 0.01, -1.0
C_CONE = np.sqrt(3.0 / 7.0)


def _stevens_cfg(direction, mode="numeric"):
    """A ferromagnetic chain (its bond energy is the same for any uniform state, so the
    single-ion term alone decides) with a rank-4 Stevens term on the one site."""
    return {"crystal_structure": {
                "lattice_vectors": [[4.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
                "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0], "spin_S": S4}]},
            "interactions": {
                "heisenberg": [{"pair": ["A", "A"], "rij_offset": o, "value": JFM}
                               for o in ([1, 0, 0], [-1, 0, 0])],
                "stevens": [{"B": {"4,0": B40}, "atoms": ["A"]}]},
            "parameters": {}, "parameter_order": [],
            "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                   "directions": [list(direction)]},
            "calculation": {"on_imaginary": "error", "classical_energy": mode,
                            "cache_mode": "none"},
            "tasks": {"minimization": False, "dispersion": True}, **RUN_BLOCKS}


def _f4(c):
    return S4 ** 4 * (35 * c ** 4 - 30 * c ** 2 + 3)


E_BOND = JFM * S4 ** 2   # per site: two directed bonds, H = 1/2 sum_ordered


@pytest.mark.parametrize("mode", ["numeric", "symbolic"])
def test_stevens_rank4_maximum_is_caught_and_relaxes_to_the_cone(mode):
    calc = _calc(_stevens_cfg([0, 0, 1], mode), S4, f"st4max_{mode}")
    assert calc._extract_classical_quadratic([], 1, len(calc.sm.atom_pos_ouc()),
                                             S4) is None   # the case that used to skip
    e_now, e_relaxed = calc.relax_from_current()
    assert e_now == pytest.approx(E_BOND + B40 * _f4(1.0), abs=1e-10)
    assert e_relaxed == pytest.approx(E_BOND + B40 * _f4(C_CONE), abs=1e-7)


def test_stevens_rank4_cone_is_a_minimum():
    d = [float(np.sqrt(1 - C_CONE ** 2)), 0.0, float(C_CONE)]
    e_now, e_relaxed = _calc(_stevens_cfg(d), S4, "st4cone").relax_from_current()
    assert e_now == pytest.approx(E_BOND - 24 / 7 * B40 * S4 ** 4, abs=1e-10)
    assert e_relaxed >= e_now - 1e-9


def test_stevens_rank4_maximum_is_refused_by_the_runner(tmp_path):
    with pytest.raises(ValueError, match="NOT a classical energy minimum"):
        _run(_stevens_cfg([0, 0, 1]), tmp_path, "st4max")


def test_stevens_rank4_cone_runs(tmp_path):
    d = [float(np.sqrt(1 - C_CONE ** 2)), 0.0, float(C_CONE)]
    _run(_stevens_cfg(d), tmp_path, "st4cone")


def test_general_gradient_matches_central_differences():
    from magcalc.core import _general_energy_np, _general_grad_np
    calc = _calc(_stevens_cfg([0, 0, 1]), S4, "st4grad")
    fns = calc._classical_energy_general([], 1, len(calc.sm.atom_pos_ouc()), S4)
    S_vec = calc._classical_spin_lengths(1)
    rng = np.random.default_rng(3)
    for _ in range(3):
        x = rng.uniform(0.2, 2.9, size=2)
        g = _general_grad_np(x, *fns, S_vec, 1)
        h = 1e-6
        fd = [(_general_energy_np(x + h * e, *fns, S_vec, 1)
               - _general_energy_np(x - h * e, *fns, S_vec, 1)) / (2 * h) for e in np.eye(2)]
        assert g == pytest.approx(fd, abs=1e-5)


def test_runner_warns_when_the_audit_cannot_run(tmp_path, monkeypatch, caplog):
    """The minimum the task asked for: if the energy cannot be built, say so."""
    monkeypatch.setattr(mc.MagCalc, "_classical_energy_general", lambda *a, **k: None)
    d = [float(np.sqrt(1 - C_CONE ** 2)), 0.0, float(C_CONE)]
    with caplog.at_level(logging.WARNING):
        _run(_stevens_cfg(d), tmp_path, "st4warn")
    assert any("guard 2) did NOT run" in r.getMessage() for r in caplog.records)


def test_quadratic_models_keep_the_quadratic_audit(monkeypatch):
    """A quadratic model must not be routed through the general energy: its verdicts
    are pinned elsewhere (test_guard_tolerances, test_ground_state_guard) on (H, b, c)."""
    cfg = _stevens_cfg([1, 0, 0])
    cfg["interactions"].pop("stevens")
    cfg["interactions"]["single_ion_anisotropy"] = [
        {"value": -0.1, "axis": [0, 0, 1], "atoms": ["A"]}]
    calc = _calc(cfg, S4, "quad_sia")

    def boom(*a, **k):
        raise AssertionError("quadratic model reached the general energy")
    monkeypatch.setattr(mc.MagCalc, "_classical_energy_general", boom)
    e_now, e_relaxed = calc.relax_from_current()
    assert e_relaxed < e_now - 1e-3      # x is the hard axis of an easy-z D < 0


def test_audit_moves_on_per_site_lengths():
    """A (1, 1/2) Neel chain with a biquadratic on the intracell bond: per cell
    E = -J s_A s_B (two bonds) + B (s_A s_B)^2 = -0.9875 exactly. One length for both
    sites (the reference S = 1) would score it at -1.95, which is what the minimizer's
    symbolic fallback reports for it (OPEN_WORK item 6)."""
    J, B, sA, sB = 1.0, 0.05, 1.0, 0.5
    bonds = lambda t, v, pairs: [{"pair": p, "rij_offset": o, "value": v} for p, o in pairs]
    intra = ((["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]))
    inter = ((["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0]))
    cfg = {"crystal_structure": {
               "lattice_vectors": [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]],
               "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0], "spin_S": sA},
                            {"label": "B", "pos": [0.5, 0, 0], "spin_S": sB}]},
           "interactions": {"heisenberg": bonds("heisenberg", J, intra + inter),
                            "biquadratic": bonds("biquadratic", B, intra)},
           "parameters": {}, "parameter_order": [],
           "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                  "directions": [[0, 0, 1], [0, 0, -1]]},
           "calculation": {"on_imaginary": "off"}, "tasks": {}}
    e_now, e_relaxed = _calc(cfg, sA, "audit_mixed").relax_from_current()
    assert e_now == pytest.approx(-2 * J * sA * sB + B * (sA * sB) ** 2, abs=1e-12)
    assert e_relaxed >= e_now - 1e-9


# --------------------------------------------------------------------------
# The oracle the item names: Sunny's 120-degree state with a biquadratic
# --------------------------------------------------------------------------
def _ferro_tri():
    cfg = _triangular_cfg()
    cfg["magnetic_structure"] = {"type": "pattern", "pattern_type": "ferromagnetic",
                                 "direction": [1, 0, 0]}
    return cfg


@pytest.mark.slow
def test_biquadratic_120_degree_state_passes_and_the_ferromagnet_does_not():
    """One build (the 9-site symbolic LSWT is minutes on a busy machine): the audit
    reads the structure from the spin model, so the ferromagnet is swapped in after."""
    J, B, s = SUNNY["J"], SUNNY["B"], SUNNY["s"]
    c120 = np.cos(2 * np.pi / 3)
    exact = 3 * (J * s ** 2 * c120 + B * s ** 4 * c120 ** 2)
    assert exact == pytest.approx(SUNNY["energy_per_site"], abs=1e-12)
    calc = _calc(_triangular_cfg(), s, "audit_tri")
    e_now, e_relaxed = calc.relax_from_current()
    assert e_now / N_TRI == pytest.approx(exact, abs=1e-10)
    assert e_relaxed >= e_now - 1e-9          # Sunny's relaxation lowers nothing

    fm = GenericSpinModel(copy.deepcopy(_ferro_tri()))
    calc.sm.set_magnetic_structure(*fm.generate_magnetic_structure())
    e_now, e_relaxed = calc.relax_from_current()
    assert e_now / N_TRI == pytest.approx(3 * (J * s ** 2 + B * s ** 4), abs=1e-10)
    assert e_relaxed < e_now - 1.0


@pytest.mark.slow
def test_biquadratic_ferromagnet_is_refused_by_the_runner(tmp_path):
    cfg = _ferro_tri()
    cfg["calculation"].update(on_imaginary="error", cache_mode="none")
    cfg["tasks"] = {"minimization": False, "dispersion": True}
    cfg.update(copy.deepcopy(RUN_BLOCKS))
    with pytest.raises(ValueError, match="NOT a classical energy minimum"):
        _run(cfg, tmp_path, "tri_fm")
