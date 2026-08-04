"""Vacancies and open boundaries in the classical samplers (Gap 4 #16, step 1).

Sunny's `set_vacancy_at!` and `remove_periodicity!`. GAP4_PLAN splits #16 in two, and
this is the first half: the classical modules already build an EXPLICIT real-space
supercell (`thermal_mc.build_supercell` returns (H, b, N, S, pos)), so a vacancy is
just the restriction of that quadratic form to the surviving sites and an open
boundary is just dropping the bonds that wrap. Both are available to `thermal_mc`,
`sampled_correlations`, `static_correlations` and `wang_landau`:

    thermal_mc:
      supercell: [8, 8, 1]
      disorder: {vacancy_concentration: 0.1, seed: 0}
      periodic: [true, true, false]

The LSWT half (disorder needs a large supercell and no eigensolve, i.e. the existing
KPM engine -- Sunny's example 09 recipe) is NOT done; see GAP_STATUS.md.

Every check below is an exact identity or an analytic count. No oracle is needed for
"a vacancy removes exactly this site's bonds".
"""
import copy

import numpy as np
import pytest

from magcalc.generic_model import GenericSpinModel
from magcalc.thermal_mc import build_supercell, parallel_tempering

LAT = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 9.0]]


def _square_lattice():
    """S = 1 square-lattice AFM, one site per cell, NN bonds in the ab plane."""
    cfg = {"crystal_structure": {"lattice_vectors": LAT,
                                 "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0],
                                               "spin_S": 1.0}]},
           "interactions": {"heisenberg": [
               {"pair": ["A", "A"], "rij_offset": o, "value": 1.0}
               for o in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0])]},
           "magnetic_structure": {"type": "pattern",
                                  "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "parameters": {}, "parameter_order": [],
           "calculation": {"on_imaginary": "off"}, "tasks": {}}
    return GenericSpinModel(copy.deepcopy(cfg))


def _n_bonds(H, N):
    """Distinct site pairs carrying a non-zero coupling block."""
    c = 0
    for a in range(N):
        for b in range(a + 1, N):
            if np.any(H[3 * a:3 * a + 3, 3 * b:3 * b + 3]):
                c += 1
    return c


def test_zero_concentration_is_bit_identical_to_the_clean_system():
    """The x -> 0 limit, exactly: no disorder key and x = 0 must produce the same
    matrices, not merely similar ones."""
    m = _square_lattice()
    a = build_supercell(m, [], (4, 4, 1))
    b = build_supercell(m, [], (4, 4, 1), disorder={"vacancy_concentration": 0.0})
    assert a[2] == b[2]
    assert np.abs(a[0] - b[0]).max() == 0.0
    assert np.abs(a[1] - b[1]).max() == 0.0


def test_a_vacancy_is_exactly_the_restriction_to_the_other_sites():
    """The defining property. Deleting a site's rows and columns removes every bond
    it took part in, so the diluted energy is the clean energy restricted to the
    survivors -- there is no separate bond bookkeeping that could disagree."""
    m = _square_lattice()
    H0, b0, N0, _, pos0 = build_supercell(m, [], (4, 4, 1))
    Hv, bv, Nv, _, posv = build_supercell(m, [], (4, 4, 1),
                                          disorder={"vacancies": [5]})
    keep = [a for a in range(N0) if a != 5]
    idx = np.concatenate([[3 * a, 3 * a + 1, 3 * a + 2] for a in keep])
    assert Nv == N0 - 1
    assert np.abs(Hv - H0[np.ix_(idx, idx)]).max() == 0.0
    assert np.abs(bv - b0[idx]).max() == 0.0
    assert np.abs(posv - pos0[keep]).max() == 0.0


def test_vacancy_count_matches_the_requested_concentration():
    m = _square_lattice()
    for x, n_left in ((0.1, 90), (0.25, 75), (0.5, 50)):
        _, _, N, _, _ = build_supercell(m, [], (10, 10, 1),
                                        disorder={"vacancy_concentration": x,
                                                  "seed": 0})
        assert N == n_left


def test_open_boundaries_drop_exactly_the_wrapping_bonds():
    """ANALYTIC COUNT, no reference needed. A 4x4 periodic square lattice has 2N = 32
    NN bonds; opening both in-plane axes leaves a 4x4 open grid, which has
    2 * 4 * 3 = 24."""
    m = _square_lattice()
    Hp, _, Np, _, _ = build_supercell(m, [], (4, 4, 1))
    assert _n_bonds(Hp, Np) == 32
    Ho, _, No, _, _ = build_supercell(m, [], (4, 4, 1), periodic=(False, False, True))
    assert _n_bonds(Ho, No) == 24
    # opening one axis only: 4 rows of 3 bonds along x, plus 16 periodic along y
    Hx, _, Nx, _, _ = build_supercell(m, [], (4, 4, 1), periodic=(False, True, True))
    assert _n_bonds(Hx, Nx) == 4 * 3 + 16


def test_open_boundary_raises_the_energy_per_spin():
    """Fewer bonds per spin means less binding: an open system must sit HIGHER in
    energy per spin than the periodic one at the same temperature."""
    m = _square_lattice()
    out = []
    for per in ((True, True, True), (False, False, True)):
        H, b, N, S, _ = build_supercell(m, [], (6, 6, 1), periodic=per)
        r = parallel_tempering(H, b, N, S, np.array([0.3]), n_sweeps=3000,
                               n_equil=1000, seed=2)
        out.append(r.energy[0])
    assert out[1] > out[0]


def test_dilution_raises_the_energy_per_spin_monotonically():
    """Physics, not plumbing: removing spins removes bonds from the survivors, so
    |E|/spin must fall monotonically with concentration."""
    m = _square_lattice()
    prev = None
    for x in (0.0, 0.1, 0.25):
        d = None if x == 0 else {"vacancy_concentration": x, "seed": 1}
        H, b, N, S, _ = build_supercell(m, [], (6, 6, 1), disorder=d)
        r = parallel_tempering(H, b, N, S, np.array([0.3]), n_sweeps=3000,
                               n_equil=1000, seed=2)
        if prev is not None:
            assert r.energy[0] > prev
        prev = r.energy[0]


@pytest.mark.slow
def test_results_are_self_averaging_across_disorder_seeds():
    """A single realization is not an answer. Different seeds at the same
    concentration must agree within the sample-to-sample spread -- if they do not, the
    supercell is too small and the number quoted is one realization, not the
    disorder-averaged quantity."""
    m = _square_lattice()
    vals = []
    for seed in range(5):
        H, b, N, S, _ = build_supercell(
            m, [], (10, 10, 1),
            disorder={"vacancy_concentration": 0.15, "seed": seed})
        r = parallel_tempering(H, b, N, S, np.array([0.3]), n_sweeps=4000,
                               n_equil=1500, seed=3)
        vals.append(r.energy[0])
    vals = np.array(vals)
    assert vals.std() / abs(vals.mean()) < 0.05, f"seed spread {vals.std():.4f}"


def test_bad_disorder_specs_raise():
    m = _square_lattice()
    with pytest.raises(ValueError, match="not both"):
        build_supercell(m, [], (3, 3, 1),
                        disorder={"vacancies": [0], "vacancy_concentration": 0.1})
    with pytest.raises(ValueError, match="outside the supercell"):
        build_supercell(m, [], (3, 3, 1), disorder={"vacancies": [999]})
    with pytest.raises(ValueError, match="must be in"):
        build_supercell(m, [], (3, 3, 1), disorder={"vacancy_concentration": 1.5})
    with pytest.raises(ValueError, match="needs `vacancies`"):
        build_supercell(m, [], (3, 3, 1), disorder={"seed": 3})
    with pytest.raises(ValueError, match="removed every site"):
        build_supercell(m, [], (2, 2, 1),
                        disorder={"vacancies": [0, 1, 2, 3]})
