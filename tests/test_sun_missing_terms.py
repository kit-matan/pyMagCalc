"""Terms the SU(N) / entangled engines used to DROP without a word.

`SUNModel.from_generic_model` builds its bond list from (Jex, DM, Kex) and its
on-site list from sia / sia_matrix / stevens. Anything else in `interactions:`
simply never reached the Hamiltonian:

  * `interactions.biquadratic` -- the spectrum came back bit-identical to the
    model without it (measured: max|dE| = 0.0 against a 0.9 meV control from an
    SIA), no warning anywhere;
  * `interactions.dipole_dipole: {method: ewald}` -- likewise dropped, and worse,
    `generic_model` LOGGED "Ewald summation (no real-space bonds generated)"
    while nothing downstream consumed it.

Biquadratic is now supported exactly in SU(N); Ewald (and biquadratic in
entangled mode) raise. Both halves are pinned here, because "it raises" is a
claim that rots as quietly as "it computes the term".
"""
import copy

import numpy as np
import pytest

from magcalc.generic_model import GenericSpinModel
from magcalc.sun.entangled import build_entangled_model
from magcalc.sun.lswt import SUNModel

LAT = [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]
NN = [(["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
      (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0])]
HS = (0.13, 0.31, 0.5)


def _cfg(S=1.0, J=1.0, B=None, extra_inter=None, mode="SUN"):
    """Neel chain of 2 sites, NN Heisenberg J, optional biquadratic B."""
    atoms = [{"label": "A", "pos": [0.0, 0, 0], "spin_S": S},
             {"label": "B", "pos": [0.5, 0, 0], "spin_S": S}]
    inter = {"heisenberg": [{"pair": p, "rij_offset": o, "value": J} for p, o in NN]}
    if B:
        inter["biquadratic"] = [{"pair": p, "rij_offset": o, "value": B}
                                for p, o in NN]
    if extra_inter:
        inter.update(copy.deepcopy(extra_inter))
    return {"crystal_structure": {"lattice_vectors": LAT, "atoms_uc": atoms},
            "interactions": inter, "parameters": {}, "parameter_order": [],
            "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                   "directions": [[0, 0, 1], [0, 0, -1]]},
            "calculation": {"mode": mode}, "tasks": {}}


def _qs():
    B = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
    return [np.array([h, 0, 0]) @ B for h in HS]


def _model(cfg):
    return SUNModel.from_generic_model(GenericSpinModel(copy.deepcopy(cfg)), params=[])


def _bands(cfg):
    mdl = _model(cfg)
    return np.sort(np.array([np.sort(np.real(mdl.dispersion(q)))
                             for q in _qs()]), axis=1)


# ---------------------------------------------------------------------------
# Biquadratic in SU(N), against Sunny 0.8.1 :SUN
#
#   julia> sys = System(cryst, [1 => Moment(s=s, g=2), 2 => ...], :SUN)
#          set_exchange!(sys, 1.0, Bond(1,2,[0,0,0]); biquad=B)
#          set_exchange!(sys, 1.0, Bond(2,1,[1,0,0]); biquad=B)
#          set_dipole!(sys, (0,0,1), (1,1,1,1)); set_dipole!(sys, (0,0,-1), (1,1,1,2))
#          energy_per_site(sys); dispersion(SpinWaveTheory(sys; measure=nothing), qs)
# ---------------------------------------------------------------------------
SUNNY_BIQUAD = {
    # (2S, B) -> (E/site, bands at h = 0.13, 0.31, 0.5, sorted ascending)
    (2, 0.0): (-1.0, [[0.7942958, 0.7942958, 4.0, 4.0],
                      [1.6541612, 1.6541612, 4.0, 4.0],
                      [2.0, 2.0, 4.0, 4.0]]),
    (2, -0.4): (-1.8, [[1.1120141, 1.1120141, 4.7435161, 4.7435161],
                       [2.3158256, 2.3158256, 4.7788911, 4.7788911],
                       [2.8, 2.8, 4.8, 4.8]]),
    (2, 0.25): (-0.5, [[0.5957219, 0.5957219, 3.4697884, 3.4697884],
                       [1.2406209, 1.2406209, 3.4886983, 3.4886983],
                       [1.5, 1.5, 3.5, 3.5]]),
    (3, -0.4): (-5.175, [[2.3828874, 2.3828874, 9.343902, 9.343902, 10.8, 10.8],
                         [4.9624835, 4.9624835, 9.5047461, 9.5047461, 10.8, 10.8],
                         [6.0, 6.0, 9.6, 9.6, 10.8, 10.8]]),
    (3, 0.25): (-0.421875,
                [[0.4467914, 0.4467914, 3.4881778, 3.4881778, 7.875, 7.875],
                 [0.9304657, 0.9304657, 3.6539896, 3.6539896, 7.875, 7.875],
                 [1.125, 1.125, 3.75, 3.75, 7.875, 7.875]]),
}


@pytest.mark.parametrize("key", sorted(SUNNY_BIQUAD))
def test_sun_biquadratic_matches_sunny(key):
    """(S_i.S_j)^2 expanded as sum_ab (S_i^a S_i^b)(S_j^a S_j^b): every band AND
    the classical energy, for S = 1 and S = 3/2, at both signs of B."""
    two_s, B = key
    E_ref, bands_ref = SUNNY_BIQUAD[key]
    cfg = _cfg(S=two_s / 2.0, B=B)
    mdl = _model(cfg)
    assert mdl.energy_per_site() == pytest.approx(E_ref, abs=1e-9)
    got = np.sort(np.array([np.sort(np.real(mdl.dispersion(q)))
                            for q in _qs()]), axis=1)
    assert got == pytest.approx(np.array(bands_ref), abs=1e-6)


def test_biquadratic_is_exactly_a_heisenberg_shift_at_spin_half():
    """Operator identity: for S = 1/2, (S_i.S_j)^2 = 3/16 - (1/2) S_i.S_j, so a
    biquadratic B is indistinguishable from J -> J - B/2. Fails loudly on any
    factor, sign or normalization slip, and needs no oracle at all."""
    with_biq = _bands(_cfg(S=0.5, J=1.0, B=0.6))
    equivalent = _bands(_cfg(S=0.5, J=1.0 - 0.3, B=None))
    assert with_biq == pytest.approx(equivalent, abs=1e-12)


def test_biquadratic_actually_changes_the_spectrum():
    """The original bug, stated directly: adding the term must not be a no-op.
    (It was: max|dE| was exactly 0.0.)"""
    plain = _bands(_cfg(S=1.0))
    biq = _bands(_cfg(S=1.0, B=-0.4))
    assert np.abs(biq - plain).max() > 0.1


def test_plain_sun_model_keeps_three_operators():
    """No biquadratic -> the operator basis stays the 3 dipoles, so ordinary
    SU(N) runs pay none of the 12-operator cost."""
    assert _model(_cfg(S=1.0)).n_ops == 3
    assert _model(_cfg(S=1.0, B=-0.4)).n_ops == 12


def test_sun_biquadratic_intensities_match_sunny():
    """Not just the energies: the one-magnon weights too, against Sunny `:SUN`
    with `ssf_perp(sys; apply_g=false)` at q = 0.13, 0.31, 0.45. Compared as the
    per-q TOTAL, because both magnons are degenerate here and the two codes split
    a degenerate pair between the bands differently (Sunny 0.000759 + 0.206331,
    ours 0.103545 x 2 -- same 0.20709)."""
    qs = [np.array([h, 0, 0]) @ (2 * np.pi * np.linalg.inv(np.array(LAT, float)).T)
          for h in (0.13, 0.31, 0.45)]
    sunny_total = np.array([0.2070900, 0.5294730, 0.8540800])
    for B in (None, -0.4):
        mdl = _model(_cfg(S=1.0, B=B))
        got = np.array([np.real(mdl.structure_factor(q, cross_section="perp")[1]).sum()
                        for q in qs])
        assert got == pytest.approx(sunny_total, abs=1e-5), f"B={B}: {got}"


def test_biquadratic_ground_state_search_sees_the_term():
    """The CP^(N-1) search must minimize the SAME Hamiltonian that hamiltonian()
    diagonalizes -- its mean field used to run over the 3 dipoles only, which
    would have silently ignored the biquadratic part."""
    mdl = _model(_cfg(S=1.0, B=-0.4))
    E_supplied = mdl.energy_per_site()
    E_relaxed = mdl.minimize_energy(n_restarts=8, seed=3) / mdl.L
    assert E_relaxed <= E_supplied + 1e-9
    assert E_relaxed == pytest.approx(-1.8, abs=1e-6)   # Sunny's converged value


# ---------------------------------------------------------------------------
# What must REFUSE rather than drop
# ---------------------------------------------------------------------------
EWALD = {"dipole_dipole": {"method": "ewald"}}


def test_sun_refuses_ewald_dipole_dipole():
    with pytest.raises(NotImplementedError, match="ewald"):
        _model(_cfg(S=1.0, extra_inter=EWALD))


def test_sun_accepts_truncated_dipole_dipole():
    """`truncated` expands into ordinary bond matrices upstream, which this engine
    DOES read -- so it must not be swept up by the guard, and it must matter."""
    cfg = _cfg(S=1.0, extra_inter={"dipole_dipole": {"method": "truncated",
                                                     "cutoff": 12.0}})
    for a in cfg["crystal_structure"]["atoms_uc"]:
        a["g"] = 2.0
    assert np.abs(_bands(cfg) - _bands(_cfg(S=1.0))).max() > 1e-6


def _entangled_cfg(**kw):
    cfg = _cfg(mode="entangled", **kw)
    cfg["units"] = [["A", "B"]]
    cfg["magnetic_structure"] = {"type": "pattern",
                                 "pattern_type": "ferromagnetic",
                                 "direction": [0, 0, 1]}
    return cfg


def test_entangled_refuses_ewald_and_biquadratic():
    for cfg, pattern in ((_entangled_cfg(S=0.5, extra_inter=EWALD), "ewald"),
                         (_entangled_cfg(S=0.5, B=-0.4), "biquadratic")):
        m = GenericSpinModel(copy.deepcopy(cfg))
        with pytest.raises(NotImplementedError, match=pattern):
            build_entangled_model(m, params=[], units=[[0, 1]])
