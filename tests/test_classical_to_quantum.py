"""Classical-to-quantum correspondence factor for `sampled_correlations` (Gap 4 #17).

Classical spin dynamics equipartitions -- every mode carries kT -- so the raw
classical S(q,w) is not on the quantum intensity scale: it holds too much weight
where hw << kT and too little where hw >> kT, by a factor kT/hw. Sunny multiplies by

    c2q(w) = |w/kT| / (1 - exp(-w/kT))

(`SampledCorrelations/DataRetrieval.jl:128`). pyMagCalc did not, so its classical
S(q,w) could not be compared with an LSWT calculation or with data on any absolute
footing. `sampled_correlations` now applies it by default;
`sampled_correlations: {classical_to_quantum: false}` returns the raw classical
quantity.

Validated three ways: the factor against Sunny's formula, its two analytic limits,
and -- the one that would catch applying it upside-down -- a physical prediction it
makes about a ferromagnet that the uncorrected spectrum badly violates.
"""
import copy

import numpy as np
import pytest

from magcalc import classical_dynamics as cd
from magcalc.classical_dynamics import classical_to_quantum_factor
from magcalc.generic_model import GenericSpinModel

# Sunny 0.8.1, verbatim formula:
#   c2q(ws, kT) = [iszero(w) ? 1.0 : abs((w/kT)/(1 - exp(-w/kT))) for w in ws]
WS = [0.0, 0.05, 0.25, 1.0, 2.5, 5.0, 12.0]
SUNNY_C2Q = {
    0.1: [1.0000000000, 1.2707470413, 2.7235637246, 10.0004540199, 25.0000000003,
          50.0000000000, 120.0000000000],
    0.5: [1.0000000000, 1.0508331945, 1.2707470413, 2.3130352855, 5.0339182745,
          10.0004540199, 24.0000000009],
    2.0: [1.0000000000, 1.0125520828, 1.0638017444, 1.2707470413, 1.7519388981,
          2.7235637246, 6.0149094699],
    20.0: [1.0000000000, 1.0012505208, 1.0062630208, 1.0252083247, 1.0638017444,
           1.1302029160, 1.3298215291],
}


@pytest.mark.parametrize("kT", sorted(SUNNY_C2Q))
def test_factor_matches_sunny(kT):
    got = classical_to_quantum_factor(np.array(WS), kT)
    assert got == pytest.approx(np.array(SUNNY_C2Q[kT]), rel=1e-9)


def test_analytic_limits():
    """w -> 0 gives 1 (classical statistics are already right there); w >> kT gives
    w/kT, which is exactly the Bose suppression a classical calculation lacks."""
    kT = 0.3
    assert classical_to_quantum_factor(np.array([0.0]), kT)[0] == 1.0
    tiny = classical_to_quantum_factor(np.array([1e-9]), kT)[0]
    assert tiny == pytest.approx(1.0, abs=1e-8)
    big = np.array([50.0 * kT, 100.0 * kT])
    assert classical_to_quantum_factor(big, kT) == pytest.approx(big / kT, rel=1e-15)


def test_restores_detailed_balance():
    """The classical S(q,w) is symmetric in w; the quantum one obeys detailed
    balance S(-w) = e^{-w/kT} S(w). The factor is exactly what supplies that:
    |c2q(w) / c2q(-w)| = e^{w/kT}. A sign slip fails here."""
    kT = 0.7
    w = np.array([0.2, 1.0, 3.0])
    ratio = classical_to_quantum_factor(w, kT) / classical_to_quantum_factor(-w, kT)
    assert np.abs(ratio) == pytest.approx(np.exp(w / kT), rel=1e-12)


def test_disabled_and_bad_kT():
    assert classical_to_quantum_factor(np.array(WS), None) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="positive kT"):
        classical_to_quantum_factor(np.array(WS), 0.0)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------
LAT = [[4.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]


def _fm_model():
    cfg = {"crystal_structure": {"lattice_vectors": LAT,
                                 "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0],
                                               "spin_S": 1.0}]},
           "interactions": {"heisenberg": [
               {"pair": ["A", "A"], "rij_offset": [1, 0, 0], "value": -1.0},
               {"pair": ["A", "A"], "rij_offset": [-1, 0, 0], "value": -1.0}]},
           "magnetic_structure": {"type": "pattern",
                                  "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "parameters": {}, "parameter_order": [],
           "calculation": {"on_imaginary": "off"}, "tasks": {}}
    B = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
    return GenericSpinModel(copy.deepcopy(cfg)), B


def _run(kT, c2q, **kw):
    m, B = _fm_model()
    q = np.array([[h, 0, 0] for h in (0.10, 0.20, 0.30, 0.40)]) @ B
    opts = dict(supercell=(20, 1, 1), dt=0.02, n_steps=1024, n_traj=6,
                therm_sweeps=1500, seed=1)
    opts.update(kw)
    return cd.sampled_correlations(m, [], q, kT, classical_to_quantum=c2q, **opts)


def test_flag_is_plumbed_and_is_exactly_the_factor():
    """Cheap plumbing check: the corrected result must be the raw one times the
    factor, element for element -- nothing else may change."""
    kT = 0.4
    raw = _run(kT, False, n_steps=256, n_traj=2, therm_sweeps=300)
    cor = _run(kT, True, n_steps=256, n_traj=2, therm_sweeps=300)
    assert raw.classical_to_quantum is False and cor.classical_to_quantum is True
    f = classical_to_quantum_factor(raw.energies, kT)[:, None]
    assert cor.sqw == pytest.approx(raw.sqw * f, rel=1e-12)


def test_high_temperature_is_a_no_op_where_hw_is_small():
    """kT >> hw is the classical limit, where the correction must do essentially
    nothing. Stated over the part of the grid where that actually holds: the FFT
    energy axis runs to pi/dt = 157 meV here, so "hw << kT" is a statement about the
    low-energy slice, not about the whole spectrum (at the zone top w/kT = 0.39 and
    the factor is a perfectly legitimate 1.22)."""
    kT = 400.0
    raw = _run(kT, False, n_steps=256, n_traj=2, therm_sweeps=300)
    cor = _run(kT, True, n_steps=256, n_traj=2, therm_sweeps=300)
    lo = raw.energies < 0.02 * kT
    assert lo.sum() > 5
    assert cor.sqw[lo] == pytest.approx(raw.sqw[lo], rel=0.011)


@pytest.mark.slow
def test_low_T_ferromagnet_weight_becomes_q_independent():
    """The physical check, and the one that would catch the factor applied upside
    down.

    For a ferromagnet, LSWT gives an energy-integrated intensity per q that is
    EXACTLY S, independent of q (pinned in test_absolute_normalization.py: 0.5, 1.0,
    2.0 for S = 1/2, 1, 2). A classical calculation instead equipartitions, giving
    weight ~ kT/w(q) -- strongly q-dependent, diverging as q -> 0. The correction
    must remove precisely that q-dependence.

    Measured here: the raw spectrum's weight varies by ~210% of its mean across the
    path (25.2 down to 3.6, tracking 1/w(q)); corrected, it is flat to ~29%, the
    residue being finite-T and finite-sampling. The two are a factor ~7 apart, so
    the tolerances below are nowhere near each other.

    NB this pins the SHAPE, not the absolute scale: the classical S(q,w) carries a
    normalization convention that has not been reconciled with the LSWT one (see
    GAP_STATUS Gap 4 #17). Do not read an absolute intensity off this path yet.
    """
    kT = 0.05

    def spread(res):
        dE = res.energies[1] - res.energies[0]
        tot = res.sqw.sum(axis=0) * dE
        return (tot.max() - tot.min()) / tot.mean()

    raw_spread = spread(_run(kT, False))
    cor_spread = spread(_run(kT, True))
    assert raw_spread > 1.5, f"raw classical weight should track 1/w(q): {raw_spread}"
    assert cor_spread < 0.5, f"corrected weight should be ~flat in q: {cor_spread}"
    assert cor_spread < 0.3 * raw_spread
