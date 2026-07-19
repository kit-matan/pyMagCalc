"""Zeeman calibration, pinned to Sunny and exact analytics.

Regression tests for a silent factor-2 drift (first bad commit b9d1c62): the
legacy S^0 parameter filter used to DOUBLE-count the H_mag*H_dir Zeeman term;
gamma=1.0 had been set to compensate. When the boson-degree truncation removed
the double count, every in-field dipole result was silently halved -- SW29's
verified 0.10622/1.72668 became 0.511/1.322 -- and no test caught it because
nothing pinned an in-field spectrum. These do, against:

  * Sunny 0.8.1 (FM chain, S=1/2, J=-1 meV, B=2 T, g=2):
      omega(q=0) = 0.231535 meV (= g mu_B B), omega(ZB) = 2.231535 meV;
  * the exact canted-AFM analytic sin(alpha) = g mu_B B / (4 J S)
      (Sunny minimize: 0.115768 at J=1, S=1/2, B=2 T);
  * SW29's documented q=0 energies 0.10622 / 1.72668 meV (w0 -/+ 2 mu_B B).

The engine's convention: H_mag / Hx,Hy,Hz are B in TESLA and couple with the
electron g = 2 (gamma=2 * mu_B, or mu_B * B.g.S with per-site g-tensors).
"""
import copy
import os

import numpy as np
import pytest
import yaml

import magcalc as mc
from magcalc.generic_model import GenericSpinModel

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))

MU_B = 5.788e-2
B_T = 2.0
J = 1.0


def _fm_cfg(S, field_style):
    cfg = {
        "crystal_structure": {
            "lattice_vectors": [[3.0, 0, 0], [0, 20.0, 0], [0, 0, 20.0]],
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S}],
        },
        "interactions": {"heisenberg": [
            {"pair": ["A", "A"], "rij_offset": [1, 0, 0], "value": "J1"},
            {"pair": ["A", "A"], "rij_offset": [-1, 0, 0], "value": "J1"},
        ]},
        "magnetic_structure": {"enabled": True, "type": "pattern",
                               "pattern_type": "ferromagnetic",
                               "direction": [0, 0, -1]},
    }
    if field_style == "H_mag":
        cfg["parameters"] = {"J1": -J, "H_mag": B_T, "H_dir": [0, 0, 1]}
        cfg["parameter_order"] = ["J1", "H_mag", "H_dir"]
        params = [-J, B_T, [0, 0, 1]]
    else:  # "Hz"
        cfg["parameters"] = {"J1": -J, "Hz": B_T}
        cfg["parameter_order"] = ["J1", "Hz"]
        params = [-J, B_T]
    return cfg, params


def _calc(cfg, params, S, tag):
    gm = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = gm.generate_magnetic_structure()
    gm.set_magnetic_structure(th, ph)
    return mc.MagCalc(spin_model_module=gm, spin_magnitude=S,
                      hamiltonian_params=params,
                      cache_file_base=f"zeeman_cal_{tag}", cache_mode="none")


@pytest.mark.parametrize("S", [0.5, 1.0])
@pytest.mark.parametrize("field_style", ["H_mag", "Hz"])
def test_fm_zeeman_gap_is_g2_and_spec_independent(S, field_style):
    """FM-chain magnon gap = g mu_B B = 2 mu_B B (Sunny: 0.231535 meV at 2 T),
    independent of S and of HOW the field is written in the config (the old
    engine gave 2x for H_mag+H_dir but 1x for Hz)."""
    cfg, params = _fm_cfg(S, field_style)
    calc = _calc(cfg, params, S, f"{field_style}_{S}")
    qs = np.array([[0.0, 0, 0], [np.pi / 3.0, 0, 0]])   # q=0 and zone boundary (a=3)
    d = calc.calculate_dispersion(qs, serial=True)
    e = np.asarray(d.energies)
    gap = float(e[0, 0])
    zb = float(e[1, 0])
    assert gap == pytest.approx(2 * MU_B * B_T, abs=1e-8), \
        f"gap {gap} != 2 mu_B B (Sunny: 0.231535)"
    # exchange part unchanged: omega(ZB) = 4|J|S + gap
    assert zb == pytest.approx(4 * J * S + gap, abs=1e-8)


def _afm_cfg():
    return {
        "crystal_structure": {
            "lattice_vectors": [[3.0, 0, 0], [0, 20.0, 0], [0, 0, 20.0]],
            "atoms_uc": [
                {"label": "A", "pos": [0.0, 0, 0], "spin_S": 0.5},
                {"label": "B", "pos": [0.5, 0, 0], "spin_S": 0.5},
            ],
        },
        "parameters": {"J1": J, "H_mag": B_T, "H_dir": [0, 0, 1]},
        "parameter_order": ["J1", "H_mag", "H_dir"],
        "interactions": {"heisenberg": [
            {"pair": ["A", "B"], "rij_offset": [0, 0, 0], "value": "J1"},
            {"pair": ["B", "A"], "rij_offset": [0, 0, 0], "value": "J1"},
            {"pair": ["A", "B"], "rij_offset": [-1, 0, 0], "value": "J1"},
            {"pair": ["B", "A"], "rij_offset": [1, 0, 0], "value": "J1"},
        ]},
        "magnetic_structure": {"enabled": True, "type": "pattern",
                               "pattern_type": "generic",
                               "directions": [[1, 0, 0], [-1, 0, 0]]},
    }


def test_afm_canting_matches_g2_analytic_and_spiral_opt():
    """Canted AFM chain in transverse field: sin(alpha) = g mu_B B / (4 J S)
    (Sunny minimize_energy!: 0.115768). Both the LSWT-side minimizer and
    spiral_opt must find the SAME state -- the A1 cross-engine consistency."""
    sin_expected = 2 * MU_B * B_T / (4 * J * 0.5)

    cfg = _afm_cfg()
    calc = _calc(cfg, [J, B_T, [0, 0, 1]], 0.5, "afm_cant")
    res = calc.minimize_energy(method="anneal", num_starts=4, n_sweeps=2000, seed=0)
    sz = np.abs(np.cos(res.x[0::2]))          # |z-component| of each unit spin
    assert np.allclose(sz, sin_expected, atol=2e-3), \
        f"minimize_energy canting {sz} != {sin_expected}"

    from magcalc import spiral_opt
    gm = GenericSpinModel(copy.deepcopy(_afm_cfg()))
    sp_res = spiral_opt.optimize_spiral(
        gm, [J, B_T, [0, 0, 1]], {"num_starts": 4, "lt_guess": True, "seed": 1},
        S_val=0.5)
    sz_sp = np.abs(np.asarray(sp_res.spins_lab)[:, 2])
    assert np.allclose(sz_sp, sin_expected, atol=2e-3), \
        f"spiral_opt canting {sz_sp} != {sin_expected}"


def test_sw29_field_split_branches_restored():
    """SW29 (S=1 easy-axis AFM chain, B = 7 T || z): q=0 branches at
    w0 -/+ 2 mu_B B = 0.10622 / 1.72668 meV -- the values the tutorial port
    verified against the exact LSWT result, silently halved by b9d1c62."""
    cfg = yaml.safe_load(open(os.path.join(
        ROOT, "examples", "spinw_tutorials", "SW29_AFM_chain_field", "config.yaml")))
    gm = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = gm.generate_magnetic_structure()
    gm.set_magnetic_structure(th, ph)
    pv = [cfg["parameters"][k] for k in cfg["parameter_order"]]
    calc = mc.MagCalc(spin_model_module=gm, spin_magnitude=1.0,
                      hamiltonian_params=pv,
                      cache_file_base="zeeman_cal_sw29", cache_mode="none")
    d = calc.calculate_dispersion(np.array([[0.0, 0, 0]]), serial=True)
    e = np.sort(np.asarray(d.energies).ravel())
    assert e[0] == pytest.approx(0.10622, abs=2e-3)
    assert e[-1] == pytest.approx(1.72668, abs=2e-3)
