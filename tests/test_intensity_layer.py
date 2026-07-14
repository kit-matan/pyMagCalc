"""Tests for the intensity/experiment layer:

- thermal Bose prefactor (calculate_sqw temperature=...)
- domain/twin averaging (calculate_sqw domains=...)
- cross-section selection (perp / trace / tensor components)
- energy-dependent resolution helpers (broaden_spectrum, resolve_de_fwhm,
  kinematic_q_bounds) and the plot path
- the 2-D q-grid constant-energy-cut runner task

The model is the SW01 FM chain (1 spin, J = -1 meV, S = 1/2): fast to build,
and every check against it is exact.
"""
import os

import numpy as np
import pytest
import yaml

import magcalc as mc
from magcalc.generic_model import GenericSpinModel
from magcalc.linalg import rotation_matrix
from magcalc.numerical import thermal_bose_prefactor, KB_MEV_PER_K
from magcalc.plotting import (
    broaden_spectrum,
    kinematic_q_bounds,
    plot_sqw_map,
    resolve_de_fwhm,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SW01 = os.path.join(HERE, "..", "examples", "spinw_tutorials",
                    "SW01_FM_chain", "config.yaml")


@pytest.fixture(scope="module")
def fm_chain(tmp_path_factory):
    cfg = yaml.safe_load(open(SW01))
    model = GenericSpinModel(cfg)
    pv = [cfg["parameters"][k] for k in (cfg.get("parameter_order") or [])]
    cache = str(tmp_path_factory.mktemp("cache") / "sw01")
    # SW01 declares spin_S = 1.0; keep S consistent with the config so
    # energies match what `magcalc run` computes.
    calc = mc.MagCalc(spin_model_module=model, spin_magnitude=1.0,
                      cache_mode="none", cache_file_base=cache,
                      hamiltonian_params=pv)
    th, ph = model.generate_magnetic_structure()
    calc.sm.set_magnetic_structure(th, ph)
    A = np.array(model.config["crystal_structure"]["lattice_vectors"], float)
    B = 2 * np.pi * np.linalg.inv(A).T
    qs = [np.array([0.15, 0, 0]) @ B, np.array([0.30, 0, 0]) @ B]
    base = calc.calculate_sqw(qs)
    return calc, B, qs, base


# ---------------------------------------------------------------- Bose factor

def test_bose_prefactor_matches_detailed_balance(fm_chain):
    calc, _, qs, base = fm_chain
    T = 10.0
    hot = calc.calculate_sqw(qs, temperature=T)
    expected = thermal_bose_prefactor(base.energies, T)
    assert np.allclose(hot.intensities, base.intensities * expected, rtol=1e-12)
    # n+1 for energy loss: 1/(1 - exp(-E/kT))
    E = base.energies
    analytic = 1.0 / (1.0 - np.exp(-E / (KB_MEV_PER_K * T)))
    assert np.allclose(expected, analytic, rtol=1e-12)


def test_bose_prefactor_limits():
    # T -> 0 (or unset): factor 1. Negative energies: n(|E|).
    E = np.array([0.5, 2.0])
    assert np.allclose(thermal_bose_prefactor(E, 0.0), 1.0)
    T = 20.0
    kT = KB_MEV_PER_K * T
    up = thermal_bose_prefactor(np.array([1.0]), T)[0]      # n+1
    down = thermal_bose_prefactor(np.array([-1.0]), T)[0]   # n
    assert np.isclose(up - down, 1.0, rtol=1e-12)           # detailed balance
    assert np.isclose(down, 1.0 / (np.exp(1.0 / kT) - 1.0), rtol=1e-12)
    # Goldstone floor keeps E ~ 0 finite
    assert np.isfinite(thermal_bose_prefactor(np.array([0.0]), T)[0])


def test_temperature_unset_is_identity(fm_chain):
    calc, _, qs, base = fm_chain
    again = calc.calculate_sqw(qs, temperature=None)
    assert np.allclose(again.intensities, base.intensities)


# ------------------------------------------------------------- cross-sections

def test_diagonal_components_sum_to_trace(fm_chain):
    calc, _, qs, _ = fm_chain
    tr = calc.calculate_sqw(qs, cross_section="trace")
    diag = sum(calc.calculate_sqw(qs, cross_section=c).intensities
               for c in ("xx", "yy", "zz"))
    assert np.allclose(tr.intensities, diag, atol=1e-10)


def test_perp_equals_trace_minus_projection(fm_chain):
    calc, _, qs, base = fm_chain
    tr = calc.calculate_sqw(qs, cross_section="trace")
    # perp <= trace always (projector removes non-negative diagonal weight)
    assert np.all(base.intensities <= tr.intensities + 1e-10)
    # For q along x: perp = S_yy + S_zz exactly
    yy = calc.calculate_sqw(qs, cross_section="yy").intensities
    zz = calc.calculate_sqw(qs, cross_section="zz").intensities
    assert np.allclose(base.intensities, yy + zz, atol=1e-10)


def test_unknown_cross_section_raises(fm_chain):
    calc, _, qs, _ = fm_chain
    with pytest.raises(ValueError):
        calc.calculate_sqw(qs, cross_section="nonsense")


# ------------------------------------------------------------------- domains

def test_domain_average_equals_manual_average(fm_chain):
    calc, _, qs, base = fm_chain
    doms = [{"axis": [0, 0, 1], "angle": 0},
            {"axis": [0, 0, 1], "angle": 90}]
    dres = calc.calculate_sqw(qs, domains=doms)
    R = rotation_matrix([0, 0, 1], 90)
    rot = calc.calculate_sqw([R.T @ q for q in qs])
    assert dres.energies.shape[1] == 2 * base.energies.shape[1]
    man_I = 0.5 * np.concatenate([base.intensities, rot.intensities], axis=1)
    man_E = np.concatenate([base.energies, rot.energies], axis=1)
    assert np.allclose(dres.intensities, man_I, atol=1e-12)
    assert np.allclose(dres.energies, man_E, atol=1e-12)


def test_domain_nfold_shorthand(fm_chain):
    calc, _, qs, base = fm_chain
    d3 = calc.calculate_sqw(qs, domains={"axis": [0, 0, 1], "n_fold": 3})
    assert d3.energies.shape[1] == 3 * base.energies.shape[1]
    # equal weights sum to 1: total spectral weight preserved for a
    # rotation-invariant quantity (trace)
    tr1 = calc.calculate_sqw(qs, cross_section="trace")
    tr3 = calc.calculate_sqw(qs, cross_section="trace",
                             domains={"axis": [0, 0, 1], "n_fold": 3})
    assert np.allclose(np.nansum(tr3.intensities, axis=1),
                       np.nansum(tr1.intensities, axis=1), rtol=1e-10)


def test_domains_with_component_cross_section_raises(fm_chain):
    calc, _, qs, _ = fm_chain
    with pytest.raises(ValueError):
        calc.calculate_sqw(qs, domains={"axis": [0, 0, 1], "n_fold": 2},
                           cross_section="xx")


def test_rotation_matrix_basics():
    R = rotation_matrix([0, 0, 1], 90)
    assert np.allclose(R @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-12)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    with pytest.raises(ValueError):
        rotation_matrix([0, 0, 0], 10)


# ----------------------------------------------------------------- resolution

def test_broaden_spectrum_per_mode_widths():
    grid = np.linspace(-2, 6, 401)
    centers = np.array([1.0, 3.0])
    weights = np.array([2.0, 1.0])
    widths = np.array([0.3, 0.8])
    combined = broaden_spectrum(centers, weights, grid, width=widths,
                                kind="gaussian")
    separate = (broaden_spectrum(centers[:1], weights[:1], grid, width=0.3,
                                 kind="gaussian")
                + broaden_spectrum(centers[1:], weights[1:], grid, width=0.8,
                                   kind="gaussian"))
    assert np.allclose(combined, separate, atol=1e-14)


def test_resolve_de_fwhm_polynomial():
    E = np.array([0.0, 5.0, 10.0])
    # constant fallback
    assert resolve_de_fwhm(E, None, 0.2) == 0.2
    assert resolve_de_fwhm(E, {"de_fwhm": 0.5}, 0.2) == 0.5
    # polyval convention: highest power first. FWHM(E) = -0.02 E + 1.0
    w = resolve_de_fwhm(E, {"de_fwhm": [-0.02, 1.0]}, 0.2)
    assert np.allclose(w, [1.0, 0.9, 0.8])
    # non-positive evaluations clipped
    w2 = resolve_de_fwhm(np.array([100.0]), {"de_fwhm": [-0.02, 1.0]}, 0.2)
    assert w2[0] == pytest.approx(1e-4)


def test_kinematic_q_bounds():
    ei = 25.0
    C = 2.0721
    ki = np.sqrt(ei / C)
    # elastic line, full coverage: [0, 2 ki]
    q_lo, q_hi = kinematic_q_bounds(np.array([0.0]), ei, (0.0, 180.0))
    assert q_lo[0] == pytest.approx(0.0, abs=1e-12)
    assert q_hi[0] == pytest.approx(2 * ki, rel=1e-12)
    # w > Ei is forbidden
    q_lo, q_hi = kinematic_q_bounds(np.array([30.0]), ei)
    assert np.isnan(q_lo[0]) and np.isnan(q_hi[0])
    # finite detector coverage at w = Ei/2
    w = ei / 2
    kf = np.sqrt((ei - w) / C)
    q_lo, q_hi = kinematic_q_bounds(np.array([w]), ei, (5.0, 130.0))
    expect_lo = np.sqrt(ki**2 + kf**2 - 2 * ki * kf * np.cos(np.deg2rad(5.0)))
    expect_hi = np.sqrt(ki**2 + kf**2 - 2 * ki * kf * np.cos(np.deg2rad(130.0)))
    assert q_lo[0] == pytest.approx(expect_lo, rel=1e-12)
    assert q_hi[0] == pytest.approx(expect_hi, rel=1e-12)


def test_plot_sqw_map_with_resolution(tmp_path, fm_chain):
    calc, _, qs, base = fm_chain
    out = tmp_path / "sqw_res.png"
    plot_sqw_map(
        q_vectors=np.array(qs),
        energies=base.energies,
        intensities=base.intensities,
        save_filename=str(out),
        resolution={"de_fwhm": [0.05, 0.1], "shape": "gaussian",
                    "dq_fwhm": 0.05, "ei": 1.0},
    )
    assert out.exists() and out.stat().st_size > 0


# --------------------------------------------------------- energy-cut task

def test_energy_cut_runner_task(tmp_path):
    from magcalc.runner import run_calculation

    cfg = yaml.safe_load(open(SW01))
    cfg["tasks"] = {"minimization": False, "energy_cut": True}
    cfg["energy_cut"] = {
        "origin": [0.0, 0.0, 0.0],
        "axis1": {"vec": [0.5, 0.0, 0.0], "points": 5},
        "axis2": {"vec": [0.0, 0.5, 0.0], "points": 4},
        "cuts": [{"center": 2.0, "fwhm": 0.5}, {"band": [1.0, 3.0]}],
    }
    cfg["output"] = {"save_data": True}
    cfg.setdefault("plotting", {})["save_plot"] = True
    cfg["plotting"]["show_plot"] = False
    cfg["plotting"]["plot_structure"] = False
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))

    run_calculation(str(cfg_file))

    data = np.load(tmp_path / "energy_cut_data.npz")
    assert data["panels"].shape == (2, 5, 4)
    assert data["q_rlu"].shape == (20, 3)
    assert np.all(np.isfinite(data["panels"]))
    assert (tmp_path / "energy_cut.png").exists()

    # Independent recomputation of the Gaussian-window cut (the SW10
    # companion-script math) on the same grid. S must come from the config's
    # spin_S (=1.0), exactly as the runner resolves it.
    src = yaml.safe_load(open(SW01))
    model = GenericSpinModel(src)
    pv = [src["parameters"][k] for k in src["parameter_order"]]
    spin_S = float(src["crystal_structure"]["atoms_uc"][0]["spin_S"])
    calc = mc.MagCalc(spin_model_module=model, spin_magnitude=spin_S,
                      cache_mode="none",
                      cache_file_base=str(tmp_path / "chk"),
                      hamiltonian_params=pv)
    th, ph = model.generate_magnetic_structure()
    calc.sm.set_magnetic_structure(th, ph)
    A = np.array(model.config["crystal_structure"]["lattice_vectors"], float)
    B = 2 * np.pi * np.linalg.inv(A).T
    s1 = np.linspace(0, 1, 5)
    s2 = np.linspace(0, 1, 4)
    S1, S2 = np.meshgrid(s1, s2, indexing="ij")
    q_rlu = np.column_stack([0.5 * S1.ravel(), 0.5 * S2.ravel(),
                             np.zeros(S1.size)])
    assert np.allclose(q_rlu, data["q_rlu"], atol=1e-12)
    res = calc.calculate_sqw(q_rlu @ B)
    sigma = 0.5 / 2.3548200450309493
    Z = np.nansum(res.intensities
                  * np.exp(-((res.energies - 2.0) ** 2) / (2 * sigma**2)),
                  axis=1)
    assert np.allclose(data["panels"][0], Z.reshape(5, 4), rtol=1e-10)


# ==========================================================================
# Tier-1 gaps: mixed-spin intensity prefactor, and fitting seeing the
# measurement model. Both silently biased results before.
# ==========================================================================
import copy as _copy

_LAT_MS = [[4.0, 0, 0], [0, 12.0, 0], [0, 0, 12.0]]


def _chain_cfg(atoms, J=-1.0):
    """Independent FM chain(s) along a; sublattices 6 A apart in y are decoupled."""
    return {
        "crystal_structure": {"lattice_vectors": _LAT_MS, "atoms_uc": atoms},
        "interactions": {"heisenberg": [
            {"pair": [a["label"], a["label"]], "rij_offset": o, "value": J}
            for a in atoms for o in ([1, 0, 0], [-1, 0, 0])]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
    }


def _weights(atoms, S_ref, qs, tag):
    import magcalc as mc
    from magcalc.generic_model import GenericSpinModel
    m = GenericSpinModel(_copy.deepcopy(_chain_cfg(atoms)))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=S_ref, cache_mode="none",
                      cache_file_base=tag, hamiltonian_params=[])
    B = 2 * np.pi * np.linalg.inv(np.array(_LAT_MS, float)).T
    res = calc.calculate_sqw([np.array(q) @ B for q in qs])
    return np.sum(np.real(res.intensities), axis=1)


_CU = {"label": "Cu", "pos": [0.0, 0.0, 0.0], "spin_S": 0.5, "ion": "Cu2+"}
_FE = {"label": "Fe", "pos": [0.0, 0.5, 0.0], "spin_S": 2.0, "ion": "Cu2+"}
_QS_MS = [[0.12, 0, 0], [0.3, 0, 0], [0.45, 0, 0]]


def test_mixed_spin_intensity_is_additive_over_decoupled_sublattices():
    """The S(Q,w) prefactor is sqrt(S_i/2) PER SITE. With a single global sqrt(S/2)
    (the old code) the weight of every site whose S differs from the reference is wrong
    by sqrt(S_i/S_ref) -- a 60% error on this model. Exact identity: two DECOUPLED
    sublattices must give exactly the sum of the two separate single-spin models."""
    i_both = _weights([_CU, _FE], 0.5, _QS_MS, "ms_both")
    i_cu = _weights([_CU], 0.5, _QS_MS, "ms_cu")
    i_fe = _weights([_FE], 2.0, _QS_MS, "ms_fe")
    assert np.allclose(i_both, i_cu + i_fe, atol=1e-9)


def test_mixed_spin_relative_weight_follows_S():
    """A ferromagnetic magnon's spectral weight scales with S, so Fe(S=2) must carry
    4x the weight of Cu(S=1/2)."""
    i_cu = _weights([_CU], 0.5, _QS_MS, "r_cu")
    i_fe = _weights([_FE], 2.0, _QS_MS, "r_fe")
    assert np.allclose(i_fe / i_cu, 4.0, rtol=1e-9)


def _ferri(J, temperature):
    """Ferrimagnetic chain (S=1/2 + S=2): acoustic AND optic branches, so the Bose
    factor reweights modes RELATIVE to each other and a free `scale` cannot absorb it.
    (On a plain AFM chain I ~ 1/(J f(q)), so J only rescales I and the intensity
    carries no information about J -- no bias is even possible there.)"""
    import magcalc as mc
    from magcalc.generic_model import GenericSpinModel
    lat = [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]
    cfg = {
        "crystal_structure": {"lattice_vectors": lat, "atoms_uc": [
            {"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": 0.5, "ion": "Fe2+"},
            {"label": "B", "pos": [0.5, 0.0, 0.0], "spin_S": 2.0, "ion": "Fe2+"}]},
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "distance": 3.0, "value": "J"}]},
        "parameters": {"J": J}, "parameter_order": ["J"],
        "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                               "directions": [[0, 0, 1], [0, 0, -1]]},
    }
    m = GenericSpinModel(_copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=0.5, cache_mode="none",
                      cache_file_base="ferri", hamiltonian_params=[J])
    B = 2 * np.pi * np.linalg.inv(np.array(lat, float)).T
    qs = [np.array([h, 0, 0]) @ B for h in np.linspace(0.05, 0.5, 6)]
    res = calc.calculate_sqw(qs, temperature=temperature)
    return np.real(res.intensities)


def test_ignoring_temperature_biases_an_intensity_fit():
    """Fitting intensities at T=0 when the data were measured warm biases the exchange.
    Synthetic data at 40 K from J = 1.30; profile out a scale nuisance (what a real
    fit's `scale` parameter does) and scan J. Ignoring T recovers ~1.07 (a 17% error);
    modelling T recovers 1.30 exactly."""
    J_true, T = 1.30, 40.0
    data = _ferri(J_true, T)

    def best_J(model_T):
        # Coarse grid on purpose: the bias is ~0.2 meV, an order of magnitude bigger
        # than the spacing, so a fine scan only burns time.
        best, chi_best = None, np.inf
        for J in np.linspace(1.1, 1.5, 9):
            I = _ferri(J, model_T)
            den = float(np.sum(I * I))
            scale = float(np.sum(I * data)) / den if den > 0 else 0.0
            chi = float(np.sum((scale * I - data) ** 2))
            if chi < chi_best:
                chi_best, best = chi, J
        return best

    assert abs(best_J(T) - J_true) < 1e-9            # models T: exact
    assert abs(best_J(None) - J_true) > 0.1          # ignores T: biased


def test_fitproblem_forward_model_uses_the_measurement_options():
    """FitProblem must actually forward temperature/cross_section to calculate_sqw --
    a fit that silently ignores them is the bug this closes."""
    import inspect

    from magcalc.fitting import FitProblem
    sig = inspect.signature(FitProblem.__init__)
    for key in ("temperature", "domains", "cross_section"):
        assert key in sig.parameters, f"FitProblem does not accept {key}"
    src = inspect.getsource(FitProblem._model_modes)
    assert "self.temperature" in src and "self.cross_section" in src
    assert "self.domains" in src
