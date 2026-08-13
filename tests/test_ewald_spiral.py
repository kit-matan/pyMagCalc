"""Long-range dipolar (Ewald) coupling with a single-k rotating-frame structure.

Gap #24b. Two independent oracles, in the order they have to be established:

1. **The magnetic_supercell identity.** At a COMMENSURATE k the rotating-frame
   description and the explicit real-space supercell are two cell choices for
   one infinite lattice and one spin configuration, so their magnon spectra must
   agree exactly -- the rotating frame's three channels [q-k | q | q+k] against
   the supercell's folded bands, at the same CARTESIAN q. No external reference
   needed. It is only a real check once it passes with the dipolar term switched
   OFF (`test_control_no_ewald_*`), which is where an earlier attempt at this
   item went wrong: it compared a 120-degree state that was not the ground state
   of the chain it was imposed on, and blamed the 2.8 meV disagreement on the
   band correspondence. The correspondence is fine; the model was not.

2. **Sunny 0.8.1** (`SpinWaveTheorySpiral` + `enable_dipole_dipole!`) at an
   INCOMMENSURATE k, where no supercell exists.

WHY THE FIRST ORACLE NEEDS A UNIAXIAL A(q). The three-term (`k_case` 3) form
keeps only the part of A(q) that commutes with rotations about the spiral axis;
the dropped `R1 A R1*` pieces transfer momentum by -/+2k and leave the channel
set entirely. So rotating-frame == supercell is an identity only when A(q) is
uniaxial about the axis, and the models below put k and q along a 4-fold axis to
arrange exactly that. `test_umklapp_is_dropped_when_A_is_not_uniaxial` pins the
other side of that statement -- the disagreement is real, is the size of the
dropped term, and is what `_check_ewald_spiral_validity` warns about.

`k_case` 2 (2k a reciprocal-lattice vector) is different: the umklapp folds back
into the same channel, `_ewald_J_rot` keeps it, and the supercell identity is
then exact even for a non-uniaxial A -- which is the only way to test the cross
terms at all.

THE TRAP THIS CAUGHT. pyMagCalc phases A(q) over the FULL bond vector where
Sunny phases it over lattice translations only, so the projector combination is
the MIRROR of Sunny's: R1 pairs with q+k, not q-k. Transcribing Sunny's
expression gives an H that is wrong only through the intra-cell offsets -- it is
exactly right for a one-site cell, and for any cell whenever A is uniaxial and
the sites carry no relative spiral phase. It passed every test written before
this file existed.
"""
import copy

import numpy as np
import pytest

import magcalc as mc
from magcalc.generic_model import GenericSpinModel

# --------------------------------------------------------------------------
# Model A -- tetragonal, two sites ON the 4-fold axis (z = 0 and z = 0.3).
# k and q along z, so A(q, q+/-k) is uniaxial about the spiral axis: the
# rotating-frame method is EXACT and the supercell identity applies. The 0.3
# offset breaks inversion, which is what makes the R1 <-> R1* assignment
# observable; a Bravais cell cannot see it.
# --------------------------------------------------------------------------
LAT_AX = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 6.0]]
B_AX = 2 * np.pi * np.linalg.inv(np.array(LAT_AX, float)).T
K_INCOMM = 0.2300534561          # J1 = -1, J2 = 2 spiral minimum (the SW03 value)
LS = [0.07, 0.19, 0.34]


def _chain_bonds(labels, axis_index, j1, j2):
    """Both bond directions, as pyMagCalc's H = (1/2) sum_ordered requires."""
    out = []
    for lbl in labels:
        for step, val in ((1, j1), (2, j2)):
            if val is None:
                continue
            off = [0, 0, 0]
            off[axis_index] = step
            out.append({"pair": [lbl, lbl], "rij_offset": list(off), "value": val})
            out.append({"pair": [lbl, lbl], "rij_offset": [-o for o in off], "value": val})
    return out


def _axial_config(k, j1, j2, S0, supercell=None, ewald=True):
    cfg = {
        "crystal_structure": {
            "lattice_vectors": copy.deepcopy(LAT_AX),
            "atoms_uc": [
                {"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0, "g": 2.0},
                {"label": "B", "pos": [0.0, 0.0, 0.3], "spin_S": 1.0, "g": 2.0},
            ],
        },
        "interactions": {"heisenberg": _chain_bonds(("A", "B"), 2, j1, j2)},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "single_k", "k": [0.0, 0.0, k],
                               "axis": [0.0, 0.0, 1.0], "S0": [list(s) for s in S0]},
    }
    if ewald:
        cfg["interactions"]["dipole_dipole"] = {"method": "ewald"}
    if supercell:
        cfg["crystal_structure"]["magnetic_supercell"] = supercell
    return cfg


def _build(cfg, tag):
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    return mc.MagCalc(spin_model_module=m, spin_magnitude=1.0, cache_mode="none",
                      cache_file_base=tag, hamiltonian_params=[])


def _q_axial(ls):
    return [np.array([0.0, 0.0, l]) @ B_AX for l in ls]


def _bands(calc, qs, satellites, channels=None):
    e = calc.calculate_dispersion(qs, serial=True, satellites=satellites).energies
    e = np.asarray(e)
    assert np.max(np.abs(np.imag(e))) < 1e-8, "structure is not a classical minimum"
    if channels is not None:
        e = e[:, channels]
    return np.sort(np.real(e), axis=1)


def _rot_vs_supercell(k, j1, j2, S0, supercell, ewald, qs, channels=None, tag="ews"):
    rot = _build(_axial_config(k, j1, j2, S0, ewald=ewald), tag + "_r")
    sup = _build(_axial_config(k, j1, j2, S0, supercell=supercell, ewald=ewald), tag + "_s")
    return (_bands(rot, qs, True, channels), _bands(sup, qs, False))


# --------------------------------------------------------------------------
# 1. The control: the identity must hold with NO dipolar term at all.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("S0", [
    ([1, 0, 0], [1, 0, 0]),                                   # Sunny/SpinW S0 convention
    ([1, 0, 0], [np.cos(0.2 * np.pi), np.sin(0.2 * np.pi), 0]),  # pure position spiral
])
def test_control_no_ewald_rotating_frame_equals_supercell(S0):
    """Establish the harness BEFORE trusting it about the dipolar term."""
    got, want = _rot_vs_supercell(1 / 3, 1.0, 0.5, S0, [1, 1, 3], False,
                                  _q_axial(LS), tag="ctl")
    assert got.shape == want.shape == (len(LS), 6)
    np.testing.assert_allclose(got, want, atol=1e-10)


# --------------------------------------------------------------------------
# 2. The same identity WITH Ewald -- the item itself.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("S0", [
    ([1, 0, 0], [1, 0, 0]),
    ([1, 0, 0], [np.cos(0.2 * np.pi), np.sin(0.2 * np.pi), 0]),
])
def test_ewald_rotating_frame_equals_supercell(S0):
    """Both spin-direction conventions: the S0 one is what catches a gauge error
    in the satellite terms, the position-spiral one cannot (its relative
    intra-cell spiral phase is zero in the rotating frame)."""
    got, want = _rot_vs_supercell(1 / 3, 1.0, 0.5, S0, [1, 1, 3], True,
                                  _q_axial(LS), tag="ew3")
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_ewald_actually_moves_the_bands():
    """A check a no-op passes is not a check: the dipolar term must matter by far
    more than the tolerance the identity is asserted at."""
    S0 = ([1, 0, 0], [1, 0, 0])
    qs = _q_axial(LS)
    with_ew = _bands(_build(_axial_config(1 / 3, 1.0, 0.5, S0), "mv_y"), qs, True)
    without = _bands(_build(_axial_config(1 / 3, 1.0, 0.5, S0, ewald=False), "mv_n"),
                     qs, True)
    assert np.max(np.abs(with_ew - without)) > 1e-2


def test_mirrored_projector_assignment_is_the_wrong_one():
    """Sunny's assignment transcribed literally (R1 with q-k) is WRONG here,
    because pyMagCalc's Fourier sign is the opposite one. Pin that it fails --
    otherwise the test above cannot distinguish the two."""
    S0 = ([1, 0, 0], [1, 0, 0])
    qs = _q_axial(LS)
    rot = _build(_axial_config(1 / 3, 1.0, 0.5, S0), "mir_r")
    sup = _build(_axial_config(1 / 3, 1.0, 0.5, S0, supercell=[1, 1, 3]), "mir_s")
    want = _bands(sup, qs, False)

    def sunny_order(self, q_cart):
        R1, R2 = self._spiral_projectors()
        R1c = np.conj(R1)
        k = np.asarray(self.k_cart, float)
        q = np.asarray(q_cart, float)
        sw = lambda L, J, R: np.einsum("ab,ijbc,cd->ijad", L, J, R)
        return (sw(R2, self._ewald_J_lab(q), R2)
                + sw(R1c, self._ewald_J_lab(q + k), R1c)
                + sw(R1, self._ewald_J_lab(q - k), R1))

    original = mc.MagCalc._ewald_J_rot
    try:
        mc.MagCalc._ewald_J_rot = sunny_order
        wrong = _bands(_build(_axial_config(1 / 3, 1.0, 0.5, S0), "mir_w"), qs, True)
    finally:
        mc.MagCalc._ewald_J_rot = original
    assert np.max(np.abs(wrong - want)) > 1e-2


# --------------------------------------------------------------------------
# 3. k_case 2 -- the five-term branch, where the cross terms are essential.
# --------------------------------------------------------------------------
LAT_CH = [[6.0, 0, 0], [0, 20.0, 0], [0, 0, 20.0]]     # chain along x
B_CH = 2 * np.pi * np.linalg.inv(np.array(LAT_CH, float)).T


def _chain_config(k, S0, supercell=None, ewald=True, j1=1.0, j2=None):
    cfg = {
        "crystal_structure": {
            "lattice_vectors": copy.deepcopy(LAT_CH),
            "atoms_uc": [
                {"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0, "g": 2.0},
                {"label": "B", "pos": [0.3, 0.0, 0.0], "spin_S": 1.0, "g": 2.0},
            ],
        },
        "interactions": {"heisenberg": _chain_bonds(("A", "B"), 0, j1, j2)},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "single_k", "k": [k, 0.0, 0.0],
                               "axis": [0.0, 0.0, 1.0], "S0": [list(s) for s in S0]},
    }
    if ewald:
        cfg["interactions"]["dipole_dipole"] = {"method": "ewald"}
    if supercell:
        cfg["crystal_structure"]["magnetic_supercell"] = supercell
    return cfg


def test_case2_cross_terms_are_needed():
    """A chain along x with the spiral axis along z: A(q) is NOT uniaxial about
    the axis, so the umklapp term is large. At k = 1/2 it folds back into the
    same channel, so the supercell identity is exact and it is the ONLY setting
    in which the cross terms (and their per-row gauge phase) can be tested --
    with a uniaxial A they contribute exactly zero.

    The state is non-collinear (the two sublattices are 90 degrees apart), so
    this is not the k = 1/2 collinear double-counting case.
    """
    S0 = ([1, 0, 0], [0, 1, 0])
    qs = [np.array([h, 0.0, 0.0]) @ B_CH for h in (0.07, 0.19, 0.31)]
    rot = _build(_chain_config(0.5, S0), "c2_r")
    sup = _build(_chain_config(0.5, S0, supercell=[2, 1, 1]), "c2_s")
    assert rot.k_case == 2
    want = _bands(sup, qs, False)
    assert want.shape == (3, 4)
    # k_case 2: q+k and q-k are the same channel, so compare [q | q+k].
    got = _bands(rot, qs, True, channels=slice(2, 6))
    np.testing.assert_allclose(got, want, atol=1e-10)

    def three_term(self, q_cart):
        R1, R2 = self._spiral_projectors()
        k = np.asarray(self.k_cart, float)
        q = np.asarray(q_cart, float)
        sw = lambda L, J, R: np.einsum("ab,ijbc,cd->ijad", L, J, R)
        return (sw(R2, self._ewald_J_lab(q), R2)
                + sw(R1, self._ewald_J_lab(q + k), R1)
                + sw(np.conj(R1), self._ewald_J_lab(q - k), np.conj(R1)))

    original = mc.MagCalc._ewald_J_rot
    try:
        mc.MagCalc._ewald_J_rot = three_term
        wrong = _bands(_build(_chain_config(0.5, S0), "c2_w"), qs, True,
                       channels=slice(2, 6))
    finally:
        mc.MagCalc._ewald_J_rot = original
    assert np.max(np.abs(wrong - want)) > 1e-3


# --------------------------------------------------------------------------
# 4. The regime that stays APPROXIMATE, and the warning that says so.
# --------------------------------------------------------------------------

def test_umklapp_is_dropped_when_A_is_not_uniaxial(caplog):
    """Same chain, k = 1/3 (k_case 3): the +/-2k terms now leave the channel set
    and are dropped, so the rotating frame and the supercell genuinely differ --
    by the size of the dropped term, not by round-off. The engine must SAY so.

    This is the same approximation Sunny makes; Sunny's own
    check_rotational_symmetry cannot see it, because the dipolar term is not in
    `interactions_union`.
    """
    S0 = ([1, 0, 0], [1, 0, 0])
    qs = [np.array([h, 0.0, 0.0]) @ B_CH for h in (0.07, 0.19)]
    rot = _build(_chain_config(1 / 3, S0, j2=0.5), "um_r")
    assert rot.k_case == 3
    with caplog.at_level("WARNING"):
        got = _bands(rot, qs, True)
    assert any("uniaxial about the spiral axis" in r.getMessage()
               for r in caplog.records), caplog.text

    sup_cfg = _chain_config(1 / 3, S0, supercell=[3, 1, 1], j2=0.5)
    want = _bands(_build(sup_cfg, "um_s"), qs, False)
    plain = _bands(_build(_chain_config(1 / 3, S0, ewald=False, j2=0.5), "um_n"),
                   qs, True)
    err = np.max(np.abs(got - want))
    dipolar_shift = np.max(np.abs(got - plain))
    assert err > 1e-4, "the dropped umklapp term should be visible"
    assert err < 0.5 * dipolar_shift, "it should still be a correction, not the whole term"


def test_uniaxial_case_does_not_warn(caplog):
    """The complement: with A uniaxial about the axis nothing is dropped, so the
    warning must stay quiet -- a guard that always fires teaches nothing."""
    with caplog.at_level("WARNING"):
        _bands(_build(_axial_config(1 / 3, 1.0, 0.5, ([1, 0, 0], [1, 0, 0])), "qt"),
               _q_axial(LS), True)
    assert not any("uniaxial about the spiral axis" in r.getMessage()
                   for r in caplog.records), caplog.text


# --------------------------------------------------------------------------
# 5. Sunny, at an incommensurate k (no supercell exists there).
# --------------------------------------------------------------------------
# Sunny 0.8.1, Model A with J1 = -1, J2 = 2 (spiral minimum at K_INCOMM),
# s = 1, g = 2, :dipole_uncorrected, enable_dipole_dipole!(sys,
# units.vacuum_permeability), SpinWaveTheorySpiral(k=[0,0,K_INCOMM],
# axis=[0,0,1]); dispersion at [0, 0, l], all six branches sorted ascending.
SUNNY_INCOMM = {
    0.07: [2.592906708, 2.616907631, 2.684623475, 2.798596852, 3.249647794, 3.371764520],
    0.19: [1.564773713, 1.656789740, 1.660269257, 1.893566516, 4.218451202, 4.245121354],
    0.34: [3.233679812, 3.263747246, 3.942255412, 3.980220174, 4.498167940, 4.553937209],
}
# Same model, dipole_dipole switched off -- so the comparison above cannot pass
# on the exchange part alone.
SUNNY_INCOMM_NO_EWALD = {
    0.07: [2.636484702, 2.636484702, 2.758499190, 2.758499190, 3.327310462, 3.327310462],
    0.19: [1.667385990, 1.667385990, 1.800792881, 1.800792881, 4.263566211, 4.263566211],
    0.34: [3.269017671, 3.269017671, 3.996616545, 3.996616545, 4.543433489, 4.543433489],
}


@pytest.mark.parametrize("ewald, ref", [(False, SUNNY_INCOMM_NO_EWALD),
                                        (True, SUNNY_INCOMM)])
def test_incommensurate_spiral_matches_sunny(ewald, ref):
    ls = sorted(ref)
    calc = _build(_axial_config(K_INCOMM, -1.0, 2.0, ([1, 0, 0], [1, 0, 0]),
                                ewald=ewald), f"sun_{int(ewald)}")
    got = _bands(calc, _q_axial(ls), True)
    for i, l in enumerate(ls):
        np.testing.assert_allclose(got[i], ref[l], atol=1e-6)


def test_sqw_energies_match_the_dispersion():
    """The S(Q,w) worker takes its own per-channel dipolar blocks across the
    multiprocessing boundary; regression against them silently going missing
    (the path this item's refusal used to sit on)."""
    calc = _build(_axial_config(K_INCOMM, -1.0, 2.0, ([1, 0, 0], [1, 0, 0])), "sqw")
    qs = _q_axial([0.07, 0.19])
    e_disp = np.sort(np.real(calc.calculate_dispersion(
        qs, serial=True, satellites=True).energies), axis=1)
    e_sqw = np.sort(np.real(calc.calculate_sqw(qs, satellites=True).energies), axis=1)
    np.testing.assert_allclose(e_sqw, e_disp, atol=1e-8)
    for i, l in enumerate([0.07, 0.19]):
        np.testing.assert_allclose(e_sqw[i], SUNNY_INCOMM[l], atol=1e-6)
