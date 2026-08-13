"""Absolute normalization of the CLASSICAL S(q,w) (OPEN_WORK item 4).

Gap 4 #17 put the classical spectrum's SHAPE on the quantum footing (the
classical-to-quantum factor, `tests/test_classical_to_quantum.py`) and left its
absolute SCALE unreconciled -- "measured ~220 where LSWT gives S = 1", with a
standing instruction not to read an absolute intensity off that path. Treating that
clean constant as a bug rather than a convention, as GAP_STATUS says to, it was two
independent factors, and both were invisible to every test that existed:

  * **the time transform was never normalized.** S(q,w) is (1/2pi) int dt e^{-iwt}
    C(t); the code kept the bare `np.fft.fft` sum, i.e. `2pi/dt` too large. That is
    314x at the default dt = 0.02 -- and PROPORTIONAL TO 1/dt, so the answer moved
    when you refined the time step. No test compared two grids.
  * **the spatial sum was divided by the SITE count, not the CELL count.** LSWT
    S(q,w) is per chemical cell in pyMagCalc and in Sunny (`MagCalc.supercell_ncells`,
    `SUNModel.n_cells`, Sunny's `1/sqrt(prod(sys.dims))`), so a cell with two magnetic
    atoms scatters twice as much. Every classical model ever tested here had ONE site
    per cell, where the two divisors coincide exactly.

The oracle is an exact identity, not a reference number: the defining property of
S(q,w) is the equal-time sum rule

    int dw S^ab(q,w) = <S^a(q)* S^b(q)> / n_cells,

which the FIRST test below checks against the same trajectory, at machine precision,
on a one-site AND a two-site cell -- the second is what separates the two divisors.
The remaining tests close the loop the item actually asked about: the corrected
classical spectrum, on a low-T ferromagnet, reproduces the LSWT intensity that
`tests/test_absolute_normalization.py` pins to Sunny band by band.

WHY THE FERROMAGNET IS INTEGRATED OVER A WINDOW AND NOT OVER THE WHOLE AXIS.
Integrating the c2q-corrected spectrum over every frequency the FFT produces
overshoots LSWT by ~16 %, and that residue is NOT a normalization: it is spectral
leakage. pyMagCalc applies no time-domain window, so a rectangular one is implied,
whose sidelobes fall only as 1/(w-w0)^2 -- and c2q grows LINEARLY in w, out to the
Nyquist frequency pi/dt = 157 meV on a 4 meV band. Measured, with a +-1 meV window
around each LSWT band: ratio 1.015 (mean over q), 0.96-1.07 per q. Without a window:
1.16. Sunny multiplies its real-time correlations by a cosine window for exactly this
reason; pyMagCalc does not yet (see OPEN_WORK).
"""
import copy

import numpy as np
import pytest

import magcalc as mc
from magcalc import classical_dynamics as cd
from magcalc.generic_model import GenericSpinModel
from magcalc.numerical import static_intensities
from magcalc.thermal_mc import build_supercell, _sweep, static_correlations

LAT = [[4.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]
B_MAT = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T


def _model(n_atoms=1, J=-1.0, S=1.0, field=None):
    """FM (J < 0) chain along a, moments along +z, with `n_atoms` per chemical cell.

    `field` (Tesla) is applied along -z: pyMagCalc's Zeeman is +mu_B B.g.S, so the
    moments settle ANTIparallel to B. Getting that backwards puts the structure at a
    stationary MAXIMUM, where the guard is off and the bands come back as
    |gmu_B B - w_q| -- plausible numbers for the wrong state.
    """
    if n_atoms == 1:
        atoms = [{"label": "A", "pos": [0.0, 0, 0], "spin_S": S}]
        bonds = [{"pair": ["A", "A"], "rij_offset": o, "value": J}
                 for o in ([1, 0, 0], [-1, 0, 0])]
        dirs = [[0, 0, 1]]
    else:
        atoms = [{"label": "A", "pos": [0.0, 0, 0], "spin_S": S},
                 {"label": "B", "pos": [0.5, 0, 0], "spin_S": S}]
        bonds = [{"pair": ["A", "B"], "rij_offset": [0, 0, 0], "value": J},
                 {"pair": ["B", "A"], "rij_offset": [0, 0, 0], "value": J},
                 {"pair": ["B", "A"], "rij_offset": [1, 0, 0], "value": J},
                 {"pair": ["A", "B"], "rij_offset": [-1, 0, 0], "value": J}]
        dirs = [[0, 0, 1], [0, 0, 1]]
    params, order = {}, []
    if field is not None:
        params, order = {"H_mag": field, "H_dir": [0, 0, -1]}, ["H_mag", "H_dir"]
    cfg = {"crystal_structure": {"lattice_vectors": LAT, "atoms_uc": atoms},
           "interactions": {"heisenberg": bonds},
           "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                  "directions": dirs},
           "parameters": params, "parameter_order": order,
           "calculation": {"on_imaginary": "error"}, "tasks": {}}
    return GenericSpinModel(copy.deepcopy(cfg))


def _thermalized(model, L, kT, seed=0, n_sweeps=2000):
    H, b, N, S, pos = build_supercell(model, [], (L, 1, 1))
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((N, 3))
    m *= S / np.linalg.norm(m, axis=1, keepdims=True)
    g = H @ m.ravel() + b
    for _ in range(n_sweeps):
        _sweep(m, g, H, b, 1.0 / kT, S, rng)
        g = H @ m.ravel() + b
    return H, b, S, pos, m


def _equal_time(traj, pos, qs, cross_section, n_cells):
    """<S^a(q)* S^b(q)> / n_cells, contracted, averaged over the SAME trajectory."""
    ph = np.exp(-1j * (qs @ pos.T))
    Sq = np.einsum("qr,tra->tqa", ph, traj)
    tens = np.einsum("tqa,tqb->qab", Sq.conj(), Sq) / len(traj)
    return np.array([np.real(cd._contract(tens[i][None], qs[i], cross_section))[0]
                     for i in range(len(qs))]) / n_cells


# --------------------------------------------------------------------------
# 1. The identity that fixes both factors at once
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_atoms", [1, 2])
@pytest.mark.parametrize("cross_section", ["trace", "perp"])
def test_two_sided_integral_is_the_equal_time_correlator(n_atoms, cross_section):
    """int dw S(q,w) == <S(q)* S(q)>/n_cells, at every q, to machine precision.

    This is the DEFINITION of S(q,w), so it needs no oracle and no fitted constant,
    and it is sensitive to both missing factors at once: the 2pi/dt scales the left
    side only, and the site-vs-cell divisor scales it by n_atoms. On the old code the
    n_atoms = 2 case failed by 157 and the n_atoms = 1 case by 314.
    """
    L, dt, n_t = 10, 0.02, 512
    model = _model(n_atoms)
    H, b, S, pos, m = _thermalized(model, L, kT=0.6)
    qs = np.array([[h, 0, 0] for h in (0.1, 0.3, 0.5)]) @ B_MAT
    traj = cd.evolve(H, b, S, m, dt, n_t)

    e, sqw = cd.dynamical_structure_factor(traj, pos, qs, dt, cross_section,
                                           n_cells=L, two_sided=True)
    integral = sqw.sum(axis=0) * (e[1] - e[0])
    assert integral == pytest.approx(_equal_time(traj, pos, qs, cross_section, L),
                                     rel=1e-12)


def test_free_spins_give_the_per_cell_sum_rule():
    """N uncorrelated spins of length S on a TWO-site cell: <S^a(q) S^b(q)*> =
    N (S^2/3) delta_ab at every q, so the per-CELL S(q) is n_atoms * 2S^2/3 in
    `perp` and n_atoms * S^2 in `trace` -- twice the per-site value the estimator
    used to return. Exact, q-independent and temperature-independent.

    `tests/test_static_correlations.py` pins the same identity on a ONE-site cell,
    where the two normalizations agree exactly; that is why this factor survived.
    """
    S = 1.0
    model = _model(n_atoms=2, J=0.0, S=S)
    q = np.array([[0.13, 0.2, 0], [0.31, 0, 0], [0.5, 0.5, 0]]) @ B_MAT
    for cs, per_site in (("perp", 2 * S ** 2 / 3), ("trace", S ** 2)):
        r = static_correlations(model, [], q, kT=1.0, supercell=(6, 6, 1),
                                n_samples=400, n_equil=300, sample_every=2,
                                cross_section=cs, seed=3)
        assert r.sq == pytest.approx(2 * per_site, rel=0.12)
        assert (r.sq.max() - r.sq.min()) / r.sq.mean() < 0.15   # flat in q, not just right on average


def test_scale_is_independent_of_the_fft_grid():
    """Refining the time step must not move the intensity. It used to move it
    PROPORTIONALLY: the un-normalized FFT made S(q,w) go as 1/dt, so 0.02 vs 0.005
    differed by exactly 4. Same physical trajectory length, two samplings of it.
    """
    L, model = 12, _model(1)
    H, b, S, pos, m = _thermalized(model, L, kT=0.6)
    qs = np.array([[h, 0, 0] for h in (0.15, 0.35)]) @ B_MAT
    out = []
    for dt, n_t in ((0.02, 1024), (0.005, 4096)):        # both span t = 20.48
        traj = cd.evolve(H, b, S, m.copy(), dt, n_t)
        e, sqw = cd.dynamical_structure_factor(traj, pos, qs, dt, "trace",
                                               n_cells=L, two_sided=True)
        out.append(sqw.sum(axis=0) * (e[1] - e[0]))
    assert out[0] == pytest.approx(out[1], rel=0.02)


def test_n_cells_has_no_default():
    """The divisor is keyword-only and required on purpose: a default of "one cell
    per site" is right for every one-site model and silently wrong for the rest."""
    L, model = 4, _model(1)
    H, b, S, pos, m = _thermalized(model, L, kT=1.0, n_sweeps=20)
    traj = cd.evolve(H, b, S, m, 0.02, 32)
    q = np.array([[0.25, 0, 0]]) @ B_MAT
    with pytest.raises(TypeError):
        cd.dynamical_structure_factor(traj, pos, q, 0.02, "trace")
    with pytest.raises(ValueError, match="positive cell count"):
        cd.dynamical_structure_factor(traj, pos, q, 0.02, "trace", n_cells=0)


def test_the_two_classical_estimators_agree():
    """The instantaneous estimator and the frequency integral of the dynamical one
    are the same quantity computed two ways -- different sampler, different code
    path, same normalization. Reconciling the scale means these agree; before the
    fix they were 314 apart."""
    L, kT = 12, 0.8
    model = _model(1)
    qs = np.array([[h, 0, 0] for h in (0.2, 0.4)]) @ B_MAT
    stat = static_correlations(model, [], qs, kT=kT, supercell=(L, 1, 1),
                               n_samples=600, n_equil=2000, sample_every=4,
                               cross_section="trace", seed=5).sq

    acc = None
    for seed in range(8):
        H, b, S, pos, m = _thermalized(model, L, kT, seed=seed)
        traj = cd.evolve(H, b, S, m, 0.02, 1024)
        e, sqw = cd.dynamical_structure_factor(traj, pos, qs, 0.02, "trace",
                                               n_cells=L, two_sided=True)
        tot = sqw.sum(axis=0) * (e[1] - e[0])
        acc = tot if acc is None else acc + tot
    assert acc / 8 == pytest.approx(stat, rel=0.10)


# --------------------------------------------------------------------------
# 2. The reconciliation with LSWT / Sunny
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_low_T_ferromagnet_matches_the_lswt_absolute_intensity():
    """The item's actual question: is the corrected classical S(q,w) on the SAME
    absolute scale as LSWT? Yes, to ~2 %.

    A GAPPED ferromagnet, not the bare Heisenberg chain. In 1D with no gap the
    Goldstone mode makes sum_q 1/w_q diverge, the order-parameter direction wanders
    over the trajectory, and the measured transverse weight sits ~45 % above the
    harmonic value at kT = 0.02 on L = 20 -- a real finite-size, finite-T effect that
    converges only as kT -> 0 AND L -> infinity, and which reads exactly like a
    normalization error. A 20 T field along -z gaps the mode (gmu_B B = 2.315 meV)
    and the comparison becomes clean.

    The LSWT side is `static_intensities`, i.e. the band sum, which
    `tests/test_absolute_normalization.py` pins to Sunny 0.8.1 band by band -- so
    "equals LSWT" chains back to Sunny.
    """
    kT, L = 0.005, 32
    model = _model(1, field=20.0)
    params = [20.0, [0, 0, -1]]
    hs = [0.1, 0.2, 0.3, 0.4, 0.5]
    qs = np.array([[h, 0, 0] for h in hs]) @ B_MAT

    th, ph = model.generate_magnetic_structure()
    model.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=model, spin_magnitude=1.0, cache_mode="none",
                      cache_file_base="cabs_fm", hamiltonian_params=params)
    res_l = calc.calculate_sqw(qs, cross_section="perp")
    bands = np.real(res_l.energies).reshape(len(hs), -1)      # (n_q, n_modes)
    lswt = np.real(res_l.intensities).sum(axis=1)
    assert lswt == pytest.approx(0.5, abs=1e-9)     # S/2: perp keeps one transverse channel

    res = cd.sampled_correlations(model, params, qs, kT, supercell=(L, 1, 1),
                                  dt=0.02, n_steps=4096, n_traj=128,
                                  therm_sweeps=3000, seed=1)
    e, dE = res.energies, res.energies[1] - res.energies[0]
    got = np.empty(len(hs))
    for i in range(len(hs)):
        on_a_band = np.any(np.abs(e[:, None] - bands[i][None, :]) < 1.0, axis=1)
        got[i] = res.sqw[on_a_band, i].sum() * dE
    ratio = got / lswt
    assert ratio == pytest.approx(1.0, rel=0.15), f"per q: {ratio}"
    assert ratio.mean() == pytest.approx(1.0, rel=0.06), f"mean {ratio.mean()}"


# --------------------------------------------------------------------------
# 3. The SU(N) classical path carried the SAME two defects
# --------------------------------------------------------------------------
SUN_LAT = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 9.0]]
SUN_NN = ((["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
          (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0]))


def _sun_model(ncells, S=1.0, sia=-0.4):
    """AFM chain of a two-site cell, easy-axis, on an `ncells` x 1 x 1 supercell."""
    from magcalc.sun.lswt import SUNModel
    cfg = {"crystal_structure": {"lattice_vectors": SUN_LAT,
                                 "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0],
                                               "spin_S": S},
                                              {"label": "B", "pos": [0.5, 0, 0],
                                               "spin_S": S}]},
           "interactions": {"heisenberg": [{"pair": p, "rij_offset": o, "value": 1.0}
                                           for p, o in SUN_NN],
                            "single_ion_anisotropy": [{"value": sia, "axis": [0, 0, 1],
                                                       "atoms": ["A", "B"]}]},
           "parameters": {}, "parameter_order": [],
           "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                  "directions": [[0, 0, 1], [0, 0, -1]]},
           "calculation": {"mode": "SUN", "on_imaginary": "off"}, "tasks": {}}
    return SUNModel.from_generic_model(
        GenericSpinModel(copy.deepcopy(cfg)), params=[],
        supercell=[[ncells, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_sun_two_sided_integral_is_the_equal_time_correlator():
    """The same exact identity for the CP^(N-1) path, whose transform had the same
    two defects. `n_cells` is 6 here and `model.L` is 12, so the site-vs-cell divisor
    is a factor of 2 -- and `SUNModel.structure_factor`, the spectrum this one is
    meant to be compared with, divides by `n_cells`."""
    from magcalc.sun import dynamics as sd

    ncells = 6
    model = _sun_model(ncells)
    model.minimize_energy(n_restarts=4, seed=1)
    assert model.n_cells == ncells and model.L == 2 * ncells
    rng = np.random.default_rng(2)
    Z = [np.asarray(z, complex).copy() for z in model.Z]
    sd.thermalize(model, Z, kT=0.4, n_sweeps=60, rng=rng, sigma=0.1)
    traj = sd.evolve(model, Z, 0.02, 256)

    B = 2 * np.pi * np.linalg.inv(np.array(SUN_LAT, float)).T
    qs = np.array([[h, 0, 0] for h in (1 / 6, 0.5)]) @ B
    for cs in ("trace", "perp"):
        e, sqw = sd.dynamical_structure_factor(model, traj, qs, 0.02,
                                               cross_section=cs, two_sided=True)
        integral = sqw.sum(axis=0) * (e[1] - e[0])
        Mt = np.array([[sd.moment_of(model, cfg, q) for q in qs] for cfg in traj])
        tens = np.einsum("tqa,tqb->qab", Mt.conj(), Mt) / len(traj)
        eq = np.array([np.real(cd._contract(tens[i][None], qs[i], cs))[0]
                       for i in range(len(qs))]) / ncells
        assert integral == pytest.approx(eq, rel=1e-12), cs


def test_sun_scale_is_independent_of_the_fft_grid():
    """As for the dipole path: the un-normalized FFT made the SU(N) classical
    S(q,w) go as 1/dt. Measured agreement between dt = 0.02 and dt = 0.005 over the
    same trajectory length: 4e-4."""
    from magcalc.sun import dynamics as sd

    model = _sun_model(4)
    model.minimize_energy(n_restarts=4, seed=1)
    B = 2 * np.pi * np.linalg.inv(np.array(SUN_LAT, float)).T
    qs = np.array([[0.25, 0, 0], [0.5, 0, 0]]) @ B
    out = []
    for dt, n_steps in ((0.02, 1024), (0.005, 4096)):
        w, sqw = sd.sampled_correlations(model, qs, kT=0.4, dt=dt, n_steps=n_steps,
                                         n_traj=2, therm_sweeps=100, seed=3,
                                         classical_to_quantum=False)
        out.append(sqw.sum(axis=0) * (w[1] - w[0]))
    assert out[0] == pytest.approx(out[1], rel=0.01)
