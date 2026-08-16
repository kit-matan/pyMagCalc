"""Time-domain window on the classical S(q,w) (OPEN_WORK item 12).

Item 4 fixed the classical spectrum's absolute SCALE and left its LINESHAPE
implying a RECTANGULAR window: nothing tapered the time correlation before the
frequency transform, so the truncation at trajectory length T convolved every
feature with a Dirichlet kernel whose sidelobes decay only as 1/(w-w0)^2. That is
usually harmless and is not harmless here, because `classical_to_quantum_factor`
grows LINEARLY in w out to the Nyquist frequency pi/dt = 157 meV at the default
dt = 0.02 -- so leakage off a 4 meV band is amplified across a 40x wider axis.
Measured in item 4: integrating the whole frequency axis overshot the LSWT band sum
by 16 %, against 1.5 % over +-1 meV about the band.

`lag_window` now offers Sunny's cosine window -- OPT-IN, unlike Sunny, and section 4
below is the measurement that decided that: the same one-bin smear lands on the
elastic delta, which c2q then multiplies by ~31, so `cosine` has to be paired with
`subtract_elastic` (also added to the dipole path here; the SU(N) one already had
it).

WHY THE SUM-RULE TESTS CANNOT BE THIS ITEM'S ORACLE, and what is instead. A lag
window rescales C(dt); it leaves C(dt = 0) alone, so it leaves the w-integral --
i.e. every sum rule in `tests/test_classical_absolute_normalization.py` -- exactly
invariant. Those tests pass identically either way, by construction. The oracle here
is an EXACT ALGEBRAIC IDENTITY instead:

    cos^2(x) = 1/2 + (1/4) e^{2ix} + (1/4) e^{-2ix}

so multiplying the correlation by cos^2(pi |dt| / n_t) is the SAME operation as
convolving the spectrum with the 3-point kernel [1/4, 1/2, 1/4]. Everything this
window can and cannot do follows from that one line, and each consequence is checked
below: the kernel is non-negative (so a windowed spectrum cannot go negative where
the raw one was positive), it sums to 1 (so the integral is preserved to machine
precision), and it is three bins wide (so the cost is exactly one bin dw = 2pi/T of
broadening -- not "some" broadening that has to be taken on trust).
"""
import os
import sys

import numpy as np
import pytest

from magcalc import classical_dynamics as cd

sys.path.insert(0, os.path.dirname(__file__))
# The models and the thermalizer are item 4's, deliberately: this item is a change
# to the SAME estimator those tests pin, so it has to be measured on the same
# system, not on a fresh one chosen to suit it.
from test_classical_absolute_normalization import (B_MAT, SUN_LAT,  # noqa: E402
                                                   _model, _sun_model,
                                                   _thermalized)

DT, N_T, L = 0.02, 512, 10


def _traj(kT=0.6, seed=0):
    model = _model(1)
    H, b, S, pos, m = _thermalized(model, L, kT=kT, seed=seed)
    return cd.evolve(H, b, S, m, DT, N_T), pos


def _sqw(traj, pos, qs, window, two_sided=True, cross_section="trace"):
    return cd.dynamical_structure_factor(traj, pos, qs, DT, cross_section,
                                         n_cells=L, two_sided=two_sided,
                                         window=window)


QS = np.array([[h, 0, 0] for h in (0.1, 0.3, 0.5)]) @ B_MAT


# --------------------------------------------------------------------------
# 1. The identity the whole change reduces to
# --------------------------------------------------------------------------
def test_cosine_window_is_exactly_a_quarter_half_quarter_smoothing():
    """The load-bearing test: windowing == convolving with [1/4, 1/2, 1/4].

    Not an approximation and not a fitted constant -- it is cos^2 written in
    exponentials, so it must hold at machine precision, and it is what makes every
    other property below a consequence rather than an observation. Measured: 1.1e-16
    on a spectrum of scale ~1.9.
    """
    traj, pos = _traj()
    _e, raw = _sqw(traj, pos, QS, "rectangular")
    e, win = _sqw(traj, pos, QS, "cosine")
    hann = 0.25 * np.roll(raw, 1, axis=0) + 0.5 * raw + 0.25 * np.roll(raw, -1, axis=0)
    assert np.abs(win - hann).max() < 1e-12 * max(raw.max(), 1.0)
    assert np.allclose(_e, e)


def test_the_window_is_symmetric_in_the_lag_and_unity_at_zero_lag():
    """w(dt) = cos^2(pi|dt|/n_t): 1 at dt = 0, 0 at |dt| = n_t/2, symmetric under
    dt -> -dt in the fftfreq indexing. The symmetry is what keeps the windowed
    spectrum REAL, and w(0) = 1 is what keeps the integral exact."""
    w = cd.lag_window(64, "cosine")
    assert w[0] == pytest.approx(1.0)
    assert w[32] == pytest.approx(0.0, abs=1e-15)
    assert w[1:] == pytest.approx(w[:0:-1])          # w[k] == w[n-k]
    assert cd.lag_window(64, "rectangular") is None
    with pytest.raises(ValueError, match="cosine.*rectangular"):
        cd.lag_window(64, "hamming")


# --------------------------------------------------------------------------
# 2. What the identity guarantees
# --------------------------------------------------------------------------
def test_the_window_preserves_the_frequency_integral_exactly():
    """[1/4, 1/2, 1/4] sums to 1, so the w-integral -- hence every sum rule item 4
    pins -- is unchanged to machine precision.

    This is stated as a test rather than left implicit because it is also the
    explanation of why `test_classical_absolute_normalization.py` is blind to this
    change and had to be given a separate oracle.
    """
    traj, pos = _traj()
    e, raw = _sqw(traj, pos, QS, "rectangular")
    _e, win = _sqw(traj, pos, QS, "cosine")
    dE = e[1] - e[0]
    assert win.sum(axis=0) * dE == pytest.approx(raw.sum(axis=0) * dE, rel=1e-12)


def test_the_window_cannot_manufacture_negative_intensity():
    """A non-negative smoothing kernel maps a non-negative spectrum to a non-negative
    one. Worth pinning: a lag window in general (Blackman-Tukey) CAN produce negative
    estimates, and an S(q,w) that dips below zero would be a real defect downstream
    (it is fed to plots and to c2q). cos^2 is the safe choice, and this says so.
    """
    traj, pos = _traj()
    _e, raw = _sqw(traj, pos, QS, "rectangular")
    _e2, win = _sqw(traj, pos, QS, "cosine")
    assert raw.min() >= 0.0
    assert win.min() >= 0.0


def test_rectangular_reproduces_the_unwindowed_transform():
    """`window: rectangular` must be a no-op, so the pre-2026-08-15 behaviour stays
    reachable exactly -- computed here from the raw FFT rather than from a recorded
    number."""
    traj, pos = _traj()
    e, got = _sqw(traj, pos, QS, "rectangular", cross_section="trace")
    phase = np.exp(-1j * (QS @ pos.T))
    Sqt = np.einsum("qr,tra->tqa", phase, traj)
    Sqw = np.fft.fft(Sqt, axis=0)
    energies = 2 * np.pi * np.fft.fftfreq(N_T, d=DT)
    order = np.argsort(energies)
    expect = (np.abs(Sqw)**2).sum(axis=2)[order] * (DT / (2 * np.pi * N_T * L))
    assert np.allclose(got, expect, rtol=1e-12, atol=1e-15)
    assert np.allclose(e, energies[order])


def test_windowing_is_applied_before_the_one_sided_slice():
    """The lag domain is periodic in n_t, so the window has to act on the WHOLE
    frequency axis; windowing a half-axis would wrap onto the wrong period. The
    one-sided output must therefore be exactly the top half of the two-sided one.
    """
    traj, pos = _traj()
    e2, two = _sqw(traj, pos, QS, "cosine", two_sided=True)
    e1, one = _sqw(traj, pos, QS, "cosine", two_sided=False)
    # the one-sided branch keeps fft order 0..n/2, i.e. w >= 0 ascending
    pos_half = two[np.searchsorted(e2, -1e-12):][:len(e1)]
    assert np.allclose(one, pos_half, rtol=1e-12, atol=1e-18)


# --------------------------------------------------------------------------
# 3. The SU(N) path shares the implementation, so it must share the identity
# --------------------------------------------------------------------------
def test_the_sun_path_uses_the_same_window():
    """`sun/dynamics.dynamical_structure_factor` imports the same `lag_window` /
    `_window_correlation`; the identity must hold there too, or the two classical
    paths have quietly diverged again (which is exactly how the missing 1/2pi and the
    site-vs-cell divisor survived)."""
    from magcalc.sun import dynamics as sd

    model = _sun_model(4)
    model.minimize_energy(n_restarts=4, seed=1)
    rng = np.random.default_rng(2)
    Z = [np.asarray(z, complex).copy() for z in model.Z]
    sd.thermalize(model, Z, kT=0.4, n_sweeps=40, rng=rng, sigma=0.1,
                  on_unequilibrated="off")
    traj = sd.evolve(model, Z, 0.02, 256)
    B = 2 * np.pi * np.linalg.inv(np.array(SUN_LAT, float)).T
    qs = np.array([[0.25, 0, 0], [0.5, 0, 0]]) @ B

    e, raw = sd.dynamical_structure_factor(model, traj, qs, 0.02, two_sided=True,
                                           window="rectangular")
    _e, win = sd.dynamical_structure_factor(model, traj, qs, 0.02, two_sided=True,
                                            window="cosine")
    hann = 0.25 * np.roll(raw, 1, axis=0) + 0.5 * raw + 0.25 * np.roll(raw, -1, axis=0)
    assert np.abs(win - hann).max() < 1e-12 * max(raw.max(), 1.0)
    dE = e[1] - e[0]
    assert win.sum(axis=0) * dE == pytest.approx(raw.sum(axis=0) * dE, rel=1e-12)


# --------------------------------------------------------------------------
# 4. WHY THE WINDOW IS OPT-IN: the elastic line, and what to pair it with
# --------------------------------------------------------------------------
#
# Sunny's default is `:cosine`; pyMagCalc's is `rectangular`, and this is the
# measurement behind that divergence rather than a matter of taste. The one-bin Hann
# smear lands on the ELASTIC delta of an ordered magnet, and
# `classical_to_quantum_factor` is 1 at w = 0 and |w|/kT one bin away -- 31 at
# kT = 0.005 with dw = 0.153 meV. Measured on the gapped ferromagnetic chain
# (L = 24, n_traj = 16, q = 0.15; LSWT `perp` band sum 0.5):
#
#   window       subtract_elastic   whole-axis/LSWT   first inelastic bin
#   rectangular  False              1.55              0.00006
#   rectangular  True               1.40              0.00006
#   cosine       False              2.60              9.10        <- 18x the band sum
#   cosine       True               1.40              0.00005
#
# So the two windows agree once the delta is removed -- it WAS the whole difference --
# and `cosine` alone would trade a leakage error for a catastrophic low-energy bin on
# exactly the ordered magnets this engine is usually pointed at.

def test_the_shipped_defaults_are_the_unwindowed_ones():
    """Changing a default silently changes every classical spectrum in the repo.
    Pin both entry points, and the runner's, so it takes a deliberate edit."""
    import inspect

    from magcalc.sun import dynamics as sd

    for fn in (cd.dynamical_structure_factor, cd.sampled_correlations,
               sd.dynamical_structure_factor, sd.sampled_correlations):
        assert inspect.signature(fn).parameters["window"].default == "rectangular", fn
    assert cd.lag_window(16) is None            # bare call = no window


@pytest.mark.slow
def test_the_cosine_window_needs_subtract_elastic_on_an_ordered_magnet():
    """The trap, as a number, and the fix for it in the same test.

    Without `subtract_elastic` the windowed spectrum puts an enormous spike in the
    first inelastic bin (the smeared elastic delta, amplified by c2q); with it, the
    windowed and un-windowed whole-axis integrals agree -- the delta was the entire
    difference between them.
    """
    kT, Lc = 0.005, 24
    model = _model(1, field=20.0)
    params = [20.0, [0, 0, -1]]
    qs = np.array([[0.15, 0, 0]]) @ B_MAT
    got = {}
    for window in ("rectangular", "cosine"):
        for sub in (False, True):
            r = cd.sampled_correlations(model, params, qs, kT, supercell=(Lc, 1, 1),
                                        dt=0.02, n_steps=2048, n_traj=4,
                                        therm_sweeps=1500, seed=1, window=window,
                                        subtract_elastic=sub)
            got[(window, sub)] = (float(r.sqw[1, 0]),
                                  float(r.sqw[:, 0].sum() * (r.energies[1]
                                                             - r.energies[0])))
    # the trap: cosine alone dumps the smeared delta into bin 1
    assert got[("cosine", False)][0] > 100 * got[("rectangular", False)][0]
    # the fix: with the delta gone, the two windows agree on the total
    assert got[("cosine", True)][1] == pytest.approx(got[("rectangular", True)][1],
                                                     rel=0.02)
    assert got[("cosine", True)][0] < 10 * got[("rectangular", True)][0]


# --------------------------------------------------------------------------
# 5. The trap is REPORTED, not left to the documentation (item 15)
# --------------------------------------------------------------------------
#
# Section 4's table is the oracle: `check_elastic_leakage` must fire on
# (cosine, subtract_elastic=False, kT=0.005) and stay quiet on the other three rows.
# The engine reports rather than deciding -- making `cosine` imply `subtract_elastic`
# would silently change what the config asked for -- and the trigger is a computable
# number rather than a guess: the amplification IS c2q(dw) at the first bin, so the
# warning can name it.

# item 12's measurement grid: dw = 2pi/(2048 * 0.02) = 0.1534 meV, kT = 0.005 meV
GRID = dict(dt=0.02, n_t=2048)
DW = 2 * np.pi / (GRID["n_t"] * GRID["dt"])


def _check(window, sub, c2q=True, kT=0.005, mode="warn"):
    return cd.check_elastic_leakage(window, sub, c2q, kT, GRID["dt"], GRID["n_t"],
                                    mode)


def test_the_warning_fires_on_exactly_the_dangerous_row():
    """Section 4's four rows, as the guard's spec. Only `cosine` + no
    `subtract_elastic` is the combination that put 9.10 in one bin of a spectrum whose
    entire band sum is 0.5; the other three are the ones that behaved, and a guard
    that also fired on them would be noise."""
    rows = {(w, s): _check(w, s)
            for w in ("rectangular", "cosine") for s in (False, True)}
    assert rows[("cosine", False)] is not None
    assert [k for k, v in rows.items() if v is not None] == [("cosine", False)]


def test_the_warning_names_the_amplification_it_is_triggering_on():
    """The message has to carry the number, or it is just an opinion. That number is
    c2q at the first inelastic bin -- an identity with `classical_to_quantum_factor`,
    not a constant typed in here -- and at item 12's grid it is the ~31 the item
    measured."""
    msg = _check("cosine", False)
    amp = float(cd.classical_to_quantum_factor(np.array([DW]), 0.005)[0])
    assert f"{amp:.3g}" in msg and f"{DW:.3g}" in msg
    assert amp == pytest.approx(DW / 0.005, rel=1e-9)      # kT << dw limit
    assert amp == pytest.approx(30.7, rel=0.02)            # item 12's table


def test_it_is_kT_versus_the_grid_step_that_makes_it_dangerous():
    """The third condition, and the one that carries the physics: at kT >~ dw the
    smeared delta is not amplified, so the window costs no more than the one bin of
    Hann broadening `lag_window` documents -- which is the whole point of using it.
    Bracketed either side of the threshold, against the factor rather than a kT."""
    for x, expect_fire in ((1.4, False), (1.8, True)):
        kT = DW / x
        amp = float(cd.classical_to_quantum_factor(np.array([DW]), kT)[0])
        assert (amp >= 2.0) is expect_fire, amp        # the threshold, computed
        assert (_check("cosine", False, kT=kT) is not None) is expect_fire


def test_the_raw_classical_spectrum_is_not_flagged():
    """c2q is the amplifier: without it the smear is a quarter of the elastic line
    moved one bin, which is what the window advertises. Warning there would train the
    user to ignore it."""
    assert _check("cosine", False, c2q=False) is None
    assert _check("cosine", False, kT=None) is None


def test_the_report_mode_is_the_usual_three():
    assert _check("cosine", False, mode="off") is None
    with pytest.raises(RuntimeError, match="subtract_elastic"):
        _check("cosine", False, mode="error")


def test_both_samplers_run_the_check_from_their_own_settings(caplog):
    """End to end, so the wiring is pinned and not just the predicate: each sampler
    must derive dw from ITS OWN time grid (dt * record_every, ceil(n_steps /
    record_every)) and its own kT."""
    import logging

    from magcalc.sun import dynamics as sd

    qs = np.array([[0.15, 0, 0]]) @ B_MAT
    kw = dict(supercell=(4, 1, 1), dt=0.02, n_steps=64, n_traj=1, therm_sweeps=10,
              seed=1)
    with caplog.at_level(logging.WARNING, logger="magcalc.classical_dynamics"):
        cd.sampled_correlations(_model(1), [], qs, 0.005, window="cosine", **kw)
        assert "subtract_elastic" in caplog.text
        caplog.clear()
        cd.sampled_correlations(_model(1), [], qs, 0.005, window="cosine",
                                subtract_elastic=True, **kw)
        cd.sampled_correlations(_model(1), [], qs, 0.005, **kw)
        assert caplog.text == ""

        model = _sun_model(4)
        model.minimize_energy(n_restarts=2, seed=1)
        B = 2 * np.pi * np.linalg.inv(np.array(SUN_LAT, float)).T
        sd.sampled_correlations(model, np.array([[0.25, 0, 0]]) @ B, 0.005,
                                dt=0.02, n_steps=64, n_traj=1, therm_sweeps=20,
                                window="cosine", on_unequilibrated="off")
        assert "subtract_elastic" in caplog.text


def test_the_guard_is_on_by_default_and_reachable_from_a_config(tmp_path, caplog):
    """`on_elastic_leakage` is read from BOTH classical blocks by the runner, and
    defaults to warning in both entry points -- a guard that only exists in the Python
    API misses every user who drives the engine from YAML."""
    import inspect
    import logging
    import os

    import yaml

    from magcalc import runner
    from magcalc.sun import dynamics as sd

    for fn in (cd.sampled_correlations, sd.sampled_correlations):
        assert inspect.signature(fn).parameters["on_elastic_leakage"].default == "warn"

    cfg = {
        "crystal_structure": {
            "lattice_vectors": [[4.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]],
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": 1.0}]},
        "interactions": {"heisenberg": [
            {"pair": ["A", "A"], "rij_offset": [1, 0, 0], "value": -1.0},
            {"pair": ["A", "A"], "rij_offset": [-1, 0, 0], "value": -1.0}]},
        "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                               "directions": [[0, 0, 1]]},
        "parameters": {}, "parameter_order": [],
        "tasks": {"sampled_correlations": True},
        "sampled_correlations": {"temperature": 0.005, "supercell": [4, 1, 1],
                                 "n_steps": 64, "n_traj": 1, "therm_sweeps": 10,
                                 "window": "cosine"},
        "q_path": {"G": [0, 0, 0], "X": [0.5, 0, 0], "path": ["G", "X"],
                   "points_per_segment": 2},
        "plotting": {"enabled": False, "save_plot": False, "show_plot": False},
        "output": {"save_data": False},
    }
    path = os.path.join(str(tmp_path), "c.yaml")
    cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        with caplog.at_level(logging.WARNING, logger="magcalc.classical_dynamics"):
            yaml.safe_dump(cfg, open(path, "w"))
            runner.run_calculation(path)
            assert "subtract_elastic" in caplog.text
            caplog.clear()
            cfg["sampled_correlations"]["on_elastic_leakage"] = "off"
            yaml.safe_dump(cfg, open(path, "w"))
            runner.run_calculation(path)
            assert caplog.text == ""
    finally:
        os.chdir(cwd)


def test_subtract_elastic_removes_the_time_average():
    """Definitional: it is the time-mean of S(q, t) that is removed, so a static
    configuration must transform to exactly zero."""
    traj, pos = _traj()
    static = np.repeat(traj[:1], len(traj), axis=0)
    _e, out = cd.dynamical_structure_factor(static, pos, QS, DT, "trace", n_cells=L,
                                            two_sided=True, subtract_elastic=True)
    assert np.abs(out).max() < 1e-18
    _e, keep = cd.dynamical_structure_factor(static, pos, QS, DT, "trace", n_cells=L,
                                             two_sided=True, subtract_elastic=False)
    assert keep.max() > 0
