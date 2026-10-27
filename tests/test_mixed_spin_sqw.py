"""Mixed-spin dipole S(Q,w): a closed-form oracle, and the Fortran backend (K6).

The S(Q,w) prefactor is sqrt(S_i/2) PER SITE (linalg._spin_prefactor_vector).
fMagCalc applied one global sqrt(S/2) until K6 (2026-10-27), so a mixed-spin model
fell back to NumPy; it now takes the per-site vector (`fmagcalc.run_sqw(...,
spins=)`), and an fMagCalc older than that still falls back.

The oracle is the alternating-spin antiferromagnetic chain (S_A up, S_B down, NN
J, A at x = 0 and B at a/2). With A = 2 J S_B, B = 2 J S_A, C = 2 J sqrt(S_A S_B)
cos(pi h) and tanh(2 theta) = 2C / (A + B), LSWT gives in closed form

    w_alpha = E + (A - B)/2,  w_beta = E - (A - B)/2,  E = sqrt(((A+B)/2)^2 - C^2)
    I_alpha = 1/2 (f_A sqrt(S_A) cosh(theta) - f_B sqrt(S_B) sinh(theta))^2 (1 + qz^2/q^2)
    I_beta  = 1/2 (f_B sqrt(S_B) cosh(theta) - f_A sqrt(S_A) sinh(theta))^2 (1 + qz^2/q^2)

per chemical cell (the transverse correlation S^xx = S^yy of each mode, contracted
with the perp projector; the moments lie along z). The optical branch is gapped by
2 J |S_A - S_B| at h = 0, and the INTENSITIES weight the two sublattices by
sqrt(S_i) separately -- so any path that collapses the two spin lengths into one
fails it outright.
"""
import copy
import logging
import sys
import types

import numpy as np
import pytest

_SA, _SB, _J = 0.5, 2.0, 1.3
_LAT = [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]
_IONS = ("Cu2+", "Fe2+")
# h along the chain, with off-axis components so the (1 + qz^2/q^2) factor and the
# form factors at |q| both enter.
_RLU = np.array([[0.13, 0, 0], [0.31, 0.2, 0.0], [0.47, 0, 0.3],
                 [0.77, 0.1, 0.4], [1.21, 0, 0.0], [1.5, 0.3, 0.2]])


def _ferri_calc(SA=_SA, SB=_SB, J=_J, tag="ferri_cf", D=None):
    import magcalc as mc
    from magcalc.generic_model import GenericSpinModel

    cfg = {
        "crystal_structure": {"lattice_vectors": _LAT, "atoms_uc": [
            {"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": SA, "ion": _IONS[0]},
            {"label": "B", "pos": [0.5, 0.0, 0.0], "spin_S": SB, "ion": _IONS[1]}]},
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "distance": 3.0, "value": "J"}]},
        "parameters": {"J": J}, "parameter_order": ["J"],
        "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                               "directions": [[0, 0, 1], [0, 0, -1]]},
    }
    if D is not None:   # easy axis along the moments: gaps the acoustic branch
        cfg["interactions"]["single_ion_anisotropy"] = [
            {"value": D, "axis": [0, 0, 1], "atoms": ["A", "B"]}]
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    return mc.MagCalc(spin_model_module=m, spin_magnitude=SA, cache_mode="none",
                      cache_file_base=tag, hamiltonian_params=[J])


def _qs(rlu=_RLU):
    B = 2 * np.pi * np.linalg.inv(np.array(_LAT, float)).T
    return np.asarray(rlu, float) @ B


def _closed_form(q, h, SA=_SA, SB=_SB, J=_J):
    """(energies ascending, intensities in the same order)."""
    from magcalc.form_factors import get_form_factor

    A, B, C = 2 * J * SB, 2 * J * SA, 2 * J * np.sqrt(SA * SB) * np.cos(np.pi * h)
    E = np.sqrt(((A + B) / 2) ** 2 - C ** 2)
    w_a, w_b = E + (A - B) / 2, E - (A - B) / 2
    theta = 0.5 * np.arctanh(2 * C / (A + B))
    c, s = np.cosh(theta), np.sinh(theta)
    qn = np.linalg.norm(q)
    fA, fB = get_form_factor(_IONS[0], qn), get_form_factor(_IONS[1], qn)
    pol = 1 + (q[2] / qn) ** 2
    I_a = 0.5 * (fA * np.sqrt(SA) * c - fB * np.sqrt(SB) * s) ** 2 * pol
    I_b = 0.5 * (fB * np.sqrt(SB) * c - fA * np.sqrt(SA) * s) ** 2 * pol
    order = np.argsort([w_a, w_b])
    return np.array([w_a, w_b])[order], np.array([I_a, I_b])[order]


def _sorted(res):
    o = np.argsort(res.energies, axis=1)
    return (np.take_along_axis(res.energies, o, 1),
            np.take_along_axis(np.real(res.intensities), o, 1))


def _ran_on_fortran(caplog):
    return any("computed via fMagCalc" in r.getMessage() for r in caplog.records)


def test_ferrimagnetic_chain_matches_closed_form(backend, caplog):
    """Energies AND per-band intensities of the S = (1/2, 2) chain against the closed
    form, under both backends. The two bands never cross (w_alpha - w_beta = A - B),
    so per-band comparison is well defined."""
    caplog.set_level(logging.INFO, logger="magcalc.core")
    qs = _qs()
    res = _ferri_calc().calculate_sqw(qs, backend=backend)
    E, I = _sorted(res)
    for iq, (q, h) in enumerate(zip(qs, _RLU[:, 0])):
        E_cf, I_cf = _closed_form(q, h)
        np.testing.assert_allclose(E[iq], E_cf, rtol=0, atol=1e-10)
        np.testing.assert_allclose(I[iq], I_cf, rtol=0, atol=1e-10)
    # The optical gap only a mixed-spin chain has: 2 J |S_A - S_B| at h = 0.
    E0, _ = _closed_form(_qs([[1e-9, 0, 0]])[0], 1e-9)
    assert abs(E0[1] - 2 * _J * abs(_SA - _SB)) < 1e-9
    # A mixed-spin model must no longer fall back: the Fortran case ran in Fortran.
    assert _ran_on_fortran(caplog) == (backend == "fortran")


def _groups(e, tol=1e-6):
    out, start = [], 0
    for i in range(1, len(e) + 1):
        if i == len(e) or abs(e[i] - e[i - 1]) > tol:
            out.append((start, i))
            start = i
    return out


def _assert_parity(r_np, r_f, atol=1e-10):
    """Fortran vs NumPy: energies per band; intensities per band where the band is
    non-degenerate and per degenerate-group SUM otherwise (the split inside a group
    is a basis choice, not an observable). Modes with |E| < 1e-3 are excluded: the
    Fortran kernel zeroes their (ill-conditioned Goldstone) intensity by design."""
    E_np, I_np = _sorted(r_np)
    E_f, I_f = _sorted(r_f)
    np.testing.assert_allclose(E_f, E_np, rtol=0, atol=atol)
    for iq in range(E_np.shape[0]):
        for a, b in _groups(E_np[iq]):
            if np.any(np.abs(E_np[iq, a:b]) < 1e-3):
                continue
            if b - a == 1:
                assert abs(I_f[iq, a] - I_np[iq, a]) < atol, (iq, a)
            else:
                assert abs(I_f[iq, a:b].sum() - I_np[iq, a:b].sum()) < atol, (iq, a, b)


@pytest.mark.slow
def test_fortran_matches_numpy_on_mixed_spin_yig(fortran_backend):
    """A 20-site mixed-spin ferrimagnet with degenerate bands: SW21 YIG with the 12
    tetrahedral d sites at S = 0.8 and the 8 octahedral a sites at S = 1."""
    import yaml
    import os
    import magcalc as mc
    from magcalc.generic_model import GenericSpinModel

    here = os.path.dirname(os.path.abspath(__file__))
    cfg = yaml.safe_load(open(os.path.join(
        here, "..", "examples", "spinw_tutorials", "SW21_YIG", "config.yaml")))
    cfg.pop("tasks", None)
    for a in cfg["crystal_structure"]["atoms_uc"]:
        if a["label"].startswith("FeD"):
            a["spin_S"] = 0.8
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=1.0, cache_mode="none",
                      cache_file_base="yig_mixed",
                      hamiltonian_params=[cfg["parameters"][k] for k in cfg["parameter_order"]])
    t = np.linspace(0.0, 1.0, 41)
    lat = np.array(cfg["crystal_structure"]["lattice_vectors"], float)
    qs = np.stack([0.5 * t, 0.3 * t + 0.05, 0.2 * t], 1) @ (2 * np.pi * np.linalg.inv(lat).T)
    _assert_parity(calc.calculate_sqw(qs, backend="numpy"),
                   calc.calculate_sqw(qs, backend=fortran_backend))


def test_mixed_spin_powder_parity(fortran_backend):
    """The powder average goes through the same per-site kernel. Gapped by an
    easy-axis anisotropy: every sphere holds points with h = 0, where the bare
    chain's acoustic mode is a Goldstone mode whose intensity fMagCalc zeroes by
    design (|E| < 1e-3) and NumPy does not -- a pre-existing, documented difference
    that would mask the prefactor this test is about."""
    calc = _ferri_calc(tag="ferri_powder", D=-0.4)
    qm = np.linspace(0.3, 2.4, 5)
    _assert_parity(calc.calculate_powder_average(qm, num_samples=24, backend="numpy"),
                   calc.calculate_powder_average(qm, num_samples=24, backend=fortran_backend))


def _fake_fmagcalc(run_sqw):
    mod = types.ModuleType("fmagcalc")
    mod.backend = "ctypes"
    mod.run_sqw = run_sqw
    return mod


def test_pre_k6_fmagcalc_falls_back_for_mixed_spin_only(monkeypatch, caplog):
    """An fMagCalc without `spins=` applies one global sqrt(S/2): a mixed-spin model
    must fall back to NumPy with a warning, while a uniform one still uses it."""
    calls = []

    def old_run_sqw(h_plus, h_minus, ud, ff, S, q_grid, **_):   # the pre-K6 signature
        calls.append(S)
        nq, n = len(q_grid), h_plus.shape[1] // 2
        return dict(energies=np.ones((nq, n)), intensities=np.ones((nq, n)))

    monkeypatch.setitem(sys.modules, "fmagcalc", _fake_fmagcalc(old_run_sqw))
    caplog.set_level(logging.WARNING, logger="magcalc.core")
    qs = _qs()[:3]
    mixed = _ferri_calc(tag="ferri_old")
    res = mixed.calculate_sqw(qs, backend="fortran")
    assert calls == []
    assert any("predates per-site spins" in r.getMessage() for r in caplog.records)
    E, I = _sorted(res)
    for iq, (q, h) in enumerate(zip(qs, _RLU[:3, 0])):
        np.testing.assert_allclose(I[iq], _closed_form(q, h)[1], rtol=0, atol=1e-10)

    uniform = _ferri_calc(SA=1.0, SB=1.0, tag="afm_old")
    res = uniform.calculate_sqw(qs, backend="fortran")
    assert calls == [1.0]
    assert np.all(res.energies == 1.0)


def test_spins_reach_the_kernel_and_a_failing_kernel_falls_back(monkeypatch, caplog):
    """The per-site vector is what fMagCalc receives; a kernel that raises is logged
    and NumPy answers (the backend's fall-back contract)."""
    seen = []

    def failing_run_sqw(h_plus, h_minus, ud, ff, S, q_grid, spins=None, **_):
        seen.append(list(spins))
        raise RuntimeError("kernel exploded")

    monkeypatch.setitem(sys.modules, "fmagcalc", _fake_fmagcalc(failing_run_sqw))
    caplog.set_level(logging.WARNING, logger="magcalc.core")
    qs = _qs()[:2]
    res = _ferri_calc(tag="ferri_fail").calculate_sqw(qs, backend="fortran")
    assert seen == [[_SA, _SB]]
    assert any("fMagCalc backend failed" in r.getMessage() for r in caplog.records)
    E, I = _sorted(res)
    for iq, (q, h) in enumerate(zip(qs, _RLU[:2, 0])):
        np.testing.assert_allclose(I[iq], _closed_form(q, h)[1], rtol=0, atol=1e-10)
