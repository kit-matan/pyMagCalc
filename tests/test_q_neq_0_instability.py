"""Tier 3 #13: the ground-state guard SEES a q!=0 (spiral) instability.

LSWT expands about a classical MINIMUM. The in-cell minimizer (anneal / energy
audit) only varies spin ANGLES within the given magnetic cell, so it cannot
discover that the true classical ground state is a longer-period MODULATION at
some q!=0. Both pre-existing guards are blind to this class:

  * the energy audit relaxes only within the cell, so it stays put;
  * the magnon spectrum of such a k=0 state comes back REAL and positive
    (process_calc_disp sorts the +-omega pairs and returns the upper half).

The fix (magcalc/spiral_opt.ordering_wavevector + a third runner guard) uses
Luttinger-Tisza to find the ordering wavevector k* from the exchange and flags
the case when k* is non-zero and lowers the classical energy below the k=0 state.

Reference: the J1-J2 chain has the EXACT analytic spiral pitch
    cos(2*pi*k) = -J1 / (4 J2),
so with J1 = -1, J2 = +2 the ground state is a spiral at k = arccos(1/8)/2pi =
0.230053... (SW03; also validated there against SpinW/Sunny). A ferromagnet is
the classic wrong-but-innocent-looking guess for this model.
"""
import copy
import io
import logging
import os

import numpy as np
import pytest
import yaml

from magcalc.generic_model import GenericSpinModel
from magcalc import spiral_opt
from magcalc.runner import run_calculation

HERE = os.path.dirname(__file__)
SW03 = os.path.join(HERE, "..", "examples", "spinw_tutorials",
                    "SW03_frustrated_chain", "config_spiral.yaml")
K_ANALYTIC = float(np.arccos(0.125) / (2 * np.pi))     # cos(2 pi k) = -J1/4J2 = 1/8


def _chain_model():
    cfg = yaml.safe_load(open(SW03))
    m = GenericSpinModel(cfg)
    pv = [cfg["parameters"][k] for k in cfg["parameter_order"]]
    return cfg, m, pv


def test_ordering_wavevector_matches_analytic_j1j2_spiral():
    """LT ordering vector reproduces the exact analytic pitch to ~1e-6."""
    _, m, pv = _chain_model()
    info = spiral_opt.ordering_wavevector(m, pv)
    assert abs(info["k"][0] - K_ANALYTIC) < 1e-5, f"{info['k'][0]} vs {K_ANALYTIC}"
    # inactive axes (chain b/c) snapped clean, and the spiral clearly beats k=0
    assert np.allclose(info["k"][1:], 0.0, atol=1e-6)
    assert info["gain"] > 0.5


def test_ferromagnet_on_frustrated_chain_is_flagged_as_q_neq_0():
    """The whole point: a k=0 FM supplied for the J1-J2 chain must be caught, even
    though its magnon spectrum is real and positive (both older guards are blind)."""
    cfg, _, _ = _chain_model()
    cfg["magnetic_structure"] = {"enabled": True, "type": "pattern",
                                 "pattern_type": "ferromagnetic", "direction": [0, 0, 1]}
    cfg["tasks"] = {"minimization": False, "dispersion": True}
    cfg["plotting"] = {"save_plot": False, "show_plot": False}
    cfg["calculation"] = {"cache_mode": "none", "on_imaginary": "error"}
    cfg["q_path"] = {"G": [0, 0, 0], "H": [1, 0, 0], "path": ["G", "H"],
                     "points_per_segment": 8}
    cfg["output"] = {"save_data": False}
    p = os.path.join(os.path.dirname(SW03), ".test_fm.yaml")
    yaml.safe_dump(cfg, open(p, "w"))
    try:
        with pytest.raises(ValueError) as exc:
            run_calculation(p)
    finally:
        if os.path.exists(p):
            os.remove(p)
    msg = str(exc.value)
    assert "SPIRAL" in msg and "Luttinger" in msg
    assert f"{K_ANALYTIC:.6f}"[:5] in msg          # names k* ~ 0.2300...


def test_true_spiral_config_is_not_flagged():
    """The rotating-frame single_k spiral IS the ground state, so the guard must stay
    silent (no false positive) -- it also confirms LT-flagged structures are only the
    wrong ones."""
    cfg = yaml.safe_load(open(SW03))          # type: single_k, the correct spiral
    cfg["tasks"] = {"minimization": False, "dispersion": True}
    cfg["plotting"] = {"save_plot": False, "show_plot": False}
    cfg["calculation"] = {"cache_mode": "none", "on_imaginary": "error"}
    cfg["q_path"] = {"G": [0, 0, 0], "H": [1, 0, 0], "path": ["G", "H"],
                     "points_per_segment": 8}
    cfg["output"] = {"save_data": False}
    p = os.path.join(os.path.dirname(SW03), ".test_spiral.yaml")
    yaml.safe_dump(cfg, open(p, "w"))
    try:
        run_calculation(p)                    # must not raise
    finally:
        if os.path.exists(p):
            os.remove(p)
