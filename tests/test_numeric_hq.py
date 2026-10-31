"""The numeric H(q) builder (K8, `magcalc/numeric_hq.py` + rsMagCalc `DipoleHq`)
against SymPy's `gen_HM`, entry by entry (2026-10-31).

`calculation.hamiltonian_builder: numeric` resolves the terms with the spin
model's own builders and leaves the Holstein-Primakoff algebra, the Fourier
placement and 2g to rsMagCalc. The two builders compute the same matrix, so the
oracle is an identity: H(q) and Ud agree to rounding, at ANY rotation (no ground
state is needed -- the guards run later), on

  * a fast subset of shipped configs covering DM, a full interaction matrix, a
    rotating-frame spiral, a field with g-tensors, single-ion anisotropy, a
    magnetic supercell, mixed spin and Ewald, and every buildable dipole config
    (slow);
  * random rotations in place of the configured structure;
  * a triclinic three-site model with spins 1, 5/2 and 1/2 carrying every term the
    examples do not: a biquadratic on an intercell bond between differently
    rotated sites and on a bond with no bilinear term, sia_matrix, Stevens of rank
    2, 4 and 6 (with and without rcs), Kitaev, all three g-tensor forms in a
    field, and parameter expressions;
  * a rotation that depends on a parameter (`transformations`), after a parameter
    change.

Then the wiring: the dispersion and S(Q,w) a MagCalc returns, the fit evaluator
at new parameters, the per-q process pool (which ships the bond model to its
workers), and the fallbacks to gen_HM.
"""
import copy
import glob
import logging
import os

import numpy as np
import pytest
import yaml

import magcalc as mc
from magcalc import numeric_hq, runner
from magcalc.core import reciprocal_b_matrix
from magcalc.generic_model import GenericSpinModel

pytestmark = pytest.mark.skipif(numeric_hq.rust_dipole_hq() is None,
                                reason="rsmagcalc with K8 (DipoleHq) not installed")

HERE = os.path.dirname(__file__)
EX = os.path.join(HERE, "..", "examples")
REL = 1e-12          # algebra: a tolerance far above 1e-15 would itself be a finding


def _ex(*parts):
    return os.path.join(EX, *parts)


FAST = [
    _ex("spinw_tutorials", "SW02_AFM_chain", "config.yaml"),
    _ex("spinw_tutorials", "SW09_kagome_AFM_DM", "config.yaml"),
    _ex("spinw_tutorials", "SW29_AFM_chain_field", "config.yaml"),
    _ex("spinw_tutorials", "SW03_frustrated_chain", "config_spiral.yaml"),
    _ex("spinw_tutorials", "SW12_triangular_easyplane", "config.yaml"),
    _ex("spinw_tutorials", "SW16_Na2IrO3_Kitaev", "config.yaml"),
    _ex("spinw_tutorials", "SW36_anisotropic_FM_chain", "config.yaml"),
    _ex("spinw_tutorials", "SW19_different_ions", "config.yaml"),
    _ex("sunny_tutorials", "S08_momentum_conventions", "config.yaml"),
]

SLOW_FAST = [   # cheap enough for the default run? no: 5-40 s of gen_HM each
    _ex("spinw_tutorials", "SW20_Yb2Ti2O7", "config_field.yaml"),
    _ex("spinw_tutorials", "SW21_YIG", "config.yaml"),
    _ex("sunny_tutorials", "S07_dipole_dipole", "config.yaml"),
    _ex("materials", "FeI2", "config_fei2.yaml"),
]


def _load(path_or_cfg):
    if isinstance(path_or_cfg, dict):
        return copy.deepcopy(path_or_cfg), "."
    return yaml.safe_load(open(path_or_cfg)), os.path.dirname(path_or_cfg)


def _model(path_or_cfg, rotations=None):
    cfg, base = _load(path_or_cfg)
    cfg.setdefault("calculation", {})["cache_mode"] = "none"
    sm = GenericSpinModel(cfg, base_path=base)
    if rotations is not None:
        sm.optimized_matrices = rotations(len(sm.atom_pos()))
    else:
        try:
            th, ph = sm.generate_magnetic_structure()
        except Exception:
            th = None
        if th is not None:
            sm.set_magnetic_structure(th, ph)
    return sm


def _calc(sm, builder, monkeypatch):
    monkeypatch.setenv(numeric_hq.ENV, builder)
    S, p = runner.resolve_S_and_params(sm.config)
    calc = mc.MagCalc(spin_model_module=sm, spin_magnitude=S, hamiltonian_params=p,
                      cache_file_base="numeric_hq_test", cache_mode="none")
    assert calc.hamiltonian_builder == builder
    return calc


def _qs(sm, n=24, seed=1):
    B = reciprocal_b_matrix(np.asarray(sm.unit_cell(), dtype=float))
    rng = np.random.default_rng(seed)
    q = np.vstack([rng.uniform(-1, 1, (n, 3)), np.zeros((1, 3)),
                   [[0.5, 0, 0], [0.5, 0.5, 0], [1 / 3, 1 / 3, 0], [0.5, 0.5, 0.5]]])
    return q @ B


def _random_rotations(seed):
    def make(n):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            q, r = np.linalg.qr(rng.normal(size=(3, 3)))
            q = q * np.sign(np.diag(r))
            if np.linalg.det(q) < 0:
                q[:, 0] = -q[:, 0]
            out.append(q)
        return out
    return make


def _assert_parity(sm, monkeypatch, params=None):
    sym = _calc(sm, "symbolic", monkeypatch)
    num = _calc(sm, "numeric", monkeypatch)
    if params is not None:
        sym.update_hamiltonian_params(params)
        num.update_hamiltonian_params(params)
    S = float(sym.spin_magnitude)
    q = _qs(sm)
    hs, hn = sym._build_h_stack(q, S), num._build_h_stack(q, S)
    scale = np.abs(hs).max()
    assert scale > 0
    assert np.abs(hs - hn).max() <= REL * scale, np.abs(hs - hn).max() / scale
    hm_s, hm_n = sym._build_h_stack(q, S, negate=True), num._build_h_stack(q, S, negate=True)
    assert np.abs(hm_s - hm_n).max() <= REL * scale
    assert np.abs(sym.Ud_numeric - num.Ud_numeric).max() < 1e-13
    return sym, num


@pytest.mark.parametrize("path", FAST, ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_parity_on_shipped_configs(path, monkeypatch):
    _assert_parity(_model(path), monkeypatch)


@pytest.mark.slow
@pytest.mark.parametrize("path", SLOW_FAST, ids=lambda p: os.path.basename(p))
def test_parity_on_larger_configs(path, monkeypatch):
    _assert_parity(_model(path), monkeypatch)


@pytest.mark.parametrize("path", FAST[:4], ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_parity_at_random_rotations(path, monkeypatch):
    _assert_parity(_model(path, rotations=_random_rotations(3)), monkeypatch)


def _every_dipole_config():
    out = []
    for p in sorted(glob.glob(os.path.join(EX, "**", "*.yaml"), recursive=True)):
        if any(s in p for s in ("/plots/", "/cache/", "/data/", "/reference/",
                                "future_exmaples", "SpinW_examples", "_params.yaml")):
            continue
        cfg = yaml.safe_load(open(p))
        if not (isinstance(cfg, dict) and cfg.get("crystal_structure")) \
                or cfg.get("python_model_file"):
            continue
        mode = str((cfg.get("calculation") or {}).get("mode", "dipole")).lower()
        if mode == "dipole":
            out.append(p)
    return out


@pytest.mark.slow
@pytest.mark.parametrize("path", _every_dipole_config(),
                         ids=lambda p: os.path.relpath(p, EX))
def test_parity_on_every_dipole_config(path, monkeypatch):
    _assert_parity(_model(path), monkeypatch)


# --------------------------------------------------------------------------
# Every term, in a triclinic cell with mixed spins and random moment directions
# --------------------------------------------------------------------------
def _both(t, a, b, off, val, val_rev=None, **kw):
    return [dict(type=t, pair=[a, b], rij_offset=list(off), value=val, **kw),
            dict(type=t, pair=[b, a], rij_offset=[-o for o in off],
                 value=val if val_rev is None else val_rev, **kw)]


def _kitchen_sink(extra=(), rcs=False, field=True):
    rng = np.random.default_rng(7)
    d = rng.normal(size=(3, 3))
    d = (d / np.linalg.norm(d, axis=1)[:, None]).tolist()
    inter = []
    inter += _both("heisenberg", "A", "B", [0, 0, 0], "J1")
    inter += _both("heisenberg", "A", "A", [1, 0, 0], "0.5*J2")
    inter += _both("dm", "A", "B", [0, 0, 0], ["D1", 0.11, "Dz"], ["-D1", -0.11, "-Dz"])
    inter += _both("anisotropic_exchange", "B", "C", [0, 1, 0], [0.1, -0.2, "Kb"])
    Jm = [[0.3, 0.05, -0.07], [0.02, -0.1, 0.04], [0.09, 0.01, 0.2]]
    inter += _both("interaction_matrix", "C", "A", [0, 0, 1], Jm, np.array(Jm).T.tolist())
    inter += _both("kitaev", "B", "B", [1, 0, 0], "Kk", axis="x")
    inter += [dict(type="sia", value="Dsia", axis=[1, 1, 0], atoms=["A"])]
    inter += list(extra)
    params = {"J1": 1.3, "J2": -0.4, "D1": 0.21, "Dz": -0.13, "Kb": 0.07, "Kk": -0.25,
              "Dsia": -0.35}
    order = list(params)
    if field:
        params.update({"H_mag": 2.7, "H_dir": [0.3, -0.5, 0.81]})
        order += ["H_mag", "H_dir"]
    atoms = [
        {"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0, "ion": "Fe2+",
         "g": {"g_par": 1.9, "g_perp": 2.3, "axis": [1, 0, 1]}},
        {"label": "B", "pos": [0.31, 0.17, 0.05], "spin_S": 2.5, "ion": "Mn2+",
         "g": [[2.0, 0.1, 0.0], [0.1, 2.2, 0.05], [0.0, 0.05, 1.95]]},
        {"label": "C", "pos": [0.62, 0.55, 0.43], "spin_S": 0.5, "ion": "Cu2+", "g": 2.1},
    ]
    calc = {"on_imaginary": "off"}
    if rcs:
        calc["anisotropy_renormalization"] = "rcs"
    return {
        "crystal_structure": {"lattice_vectors": [[4.1, 0, 0], [0.7, 5.3, 0],
                                                  [0.4, -0.6, 6.2]], "atoms_uc": atoms},
        "interactions": inter, "parameters": params, "parameter_order": order,
        "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                               "directions": d},
        "calculation": calc,
    }


QUARTIC = [
    dict(type="sia_matrix", matrix=[[0.1, 0.02, -0.03], [0.02, -0.2, 0.05],
                                    [-0.03, 0.05, 0.15]], atoms=["B"]),
    dict(type="stevens", B={"2,0": 0.04, "2,2": -0.03, "4,0": 0.002, "4,3": -0.004,
                            "6,0": 1e-4, "6,6": 2e-4}, atoms=["B"]),
    dict(type="stevens", B={"4,0": 0.01}, atoms=["A"]),
]
BIQUADRATIC = (_both("biquadratic", "A", "B", [0, 0, 0], -0.05)
               + _both("biquadratic", "A", "A", [1, 0, 0], 0.03)
               # an intercell bond between differently rotated sites ...
               + _both("heisenberg", "A", "B", [1, 0, 0], 0.4)
               + _both("biquadratic", "A", "B", [1, 0, 0], -0.06)
               # ... and a bond with no bilinear term at all
               + _both("biquadratic", "C", "C", [0, 1, 0], 0.04))


@pytest.mark.parametrize("extra,rcs", [((), False), (QUARTIC, False), (QUARTIC, True),
                                       (BIQUADRATIC, False), (BIQUADRATIC, True)],
                         ids=["bilinear_sia_g_field", "onsite_quartic", "onsite_quartic_rcs",
                              "biquadratic", "biquadratic_rcs"])
def test_parity_every_term(extra, rcs, monkeypatch):
    sm = _model(_kitchen_sink(extra, rcs))
    sym, num = _assert_parity(sm, monkeypatch)
    assert num._hq_terms.n_nonbilinear > 0


def test_parity_after_a_parameter_change(monkeypatch):
    sm = _model(_kitchen_sink(BIQUADRATIC))
    _, p = runner.resolve_S_and_params(sm.config)
    flat = []
    for v in p:
        flat.extend(v if isinstance(v, (list, tuple)) else [v])
    p2 = list(np.array(flat) * np.linspace(0.7, 1.3, len(flat)))
    _assert_parity(sm, monkeypatch, params=p2)


def test_a_parameter_dependent_rotation(monkeypatch):
    """`transformations` keeps a rotation symbolic in a parameter: gen_HM bakes it
    in symbolically, the numeric builder re-evaluates it per parameter vector."""
    cfg = {
        "crystal_structure": {"lattice_vectors": [[3.0, 0, 0], [0, 7, 0], [0, 0, 7]],
                              "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": 1.0},
                                           {"label": "B", "pos": [0.5, 0, 0],
                                            "spin_S": 1.0}]},
        "interactions": (_both("heisenberg", "A", "B", [0, 0, 0], "J")
                         + _both("heisenberg", "B", "A", [1, 0, 0], "J")
                         + _both("dm", "A", "B", [0, 0, 0], [0, 0, "D"], [0, 0, "-D"])),
        "parameters": {"J": 1.0, "D": 0.2, "th": 2.6},
        "parameter_order": ["J", "D", "th"],
        "transformations": {"atom_frames": [{"atom": 0, "rotation": "RotY(0.4)"},
                                            {"atom": 1, "rotation": "RotY(th)"}]},
        "calculation": {"on_imaginary": "off"},
    }
    sm = _model(cfg, rotations=None)
    sm.optimized_matrices = None
    sym, num = _assert_parity(sm, monkeypatch)
    assert num._hq_terms._rot_const is None          # the rotation really is symbolic
    _assert_parity(sm, monkeypatch, params=[1.1, -0.3, 2.9])


# --------------------------------------------------------------------------
# The wiring: what a MagCalc returns under either builder
# --------------------------------------------------------------------------
def _same_modes(a, b, atol=1e-10):
    """Energies equal to rounding, except a Goldstone mode of a Jordan block
    (the 120-degree state at Gamma here): an O(eps) change in H moves it by
    O(sqrt(eps)) times the band scale, so both builders give noise there --
    5e-16 and 1e-7 meV on SW09 -- and only "both below the noise floor" is an
    identity."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    floor = 1e-6 * np.nanmax(np.abs(a))
    zero = (np.abs(a) < floor) & (np.abs(b) < floor)
    assert np.allclose(a[~zero], b[~zero], rtol=0, atol=atol)


def test_dispersion_sqw_and_fit_evaluator_agree(monkeypatch):
    path = _ex("spinw_tutorials", "SW09_kagome_AFM_DM", "config.yaml")
    sm = _model(path)
    sym = _calc(sm, "symbolic", monkeypatch)
    num = _calc(sm, "numeric", monkeypatch)
    q = _qs(sm, n=10)
    es = np.asarray(sym.calculate_dispersion(q, serial=True).energies)
    en = np.asarray(num.calculate_dispersion(q, serial=True).energies)
    _same_modes(es, en)
    rs, rn = sym.calculate_sqw(q), num.calculate_sqw(q)
    _same_modes(rs.energies, rn.energies)
    # per-band intensities can split differently inside a degenerate group; the
    # band sum cannot
    assert np.allclose(np.nansum(rs.intensities, axis=1), np.nansum(rn.intensities, axis=1),
                       rtol=1e-8, atol=1e-10)
    p2 = [v * 1.1 if isinstance(v, float) else v for v in sym.hamiltonian_params]
    fs, fn = sym.compile_dispersion_evaluator(), num.compile_dispersion_evaluator()
    _same_modes(fs.energies(q, p2), fn.energies(q, p2))


def test_the_process_pool_ships_the_bond_model(monkeypatch):
    path = _ex("spinw_tutorials", "SW02_AFM_chain", "config.yaml")
    sm = _model(path)
    num = _calc(sm, "numeric", monkeypatch)
    sym = _calc(sm, "symbolic", monkeypatch)
    q = _qs(sm, n=6)
    en = np.asarray(num.calculate_dispersion(q, serial=False, processes=2).energies)
    es = np.asarray(sym.calculate_dispersion(q, serial=True).energies)
    assert np.allclose(es, en, rtol=0, atol=1e-10)


def test_bond_hamiltonian_refuses_other_parameters(monkeypatch):
    sm = _model(_ex("spinw_tutorials", "SW02_AFM_chain", "config.yaml"))
    num = _calc(sm, "numeric", monkeypatch)
    f, _ = num._worker_h_source()
    S = float(num.spin_magnitude)
    p = list(num.hamiltonian_params)
    f(0.1, 0.0, 0.0, S, *p)
    with pytest.raises(ValueError, match="BondHamiltonian built for"):
        f(0.1, 0.0, 0.0, S + 0.5, *p)


def test_hmat_sym_is_built_on_demand_and_matches(monkeypatch):
    import sympy as sp
    sm = _model(_ex("spinw_tutorials", "SW02_AFM_chain", "config.yaml"))
    num = _calc(sm, "numeric", monkeypatch)
    assert num.__dict__.get("_HMat_sym") is None
    H = num.HMat_sym                       # explicit request: gen_HM runs now
    assert isinstance(H, sp.MatrixBase)
    q = _qs(sm, n=4)
    f = sp.lambdify(num.full_symbol_list, H, modules="numpy")
    S = float(num.spin_magnitude)
    for qq, h in zip(q, num._build_h_stack(q, S)):
        assert np.allclose(np.array(f(*qq, S, *num.hamiltonian_params), complex), h,
                           rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# Selection and fallbacks
# --------------------------------------------------------------------------
def test_an_unknown_builder_name_raises(monkeypatch):
    monkeypatch.setenv(numeric_hq.ENV, "fast")
    with pytest.raises(ValueError, match="hamiltonian_builder"):
        numeric_hq.requested_builder({})


def test_config_key_selects_the_builder(monkeypatch):
    monkeypatch.delenv(numeric_hq.ENV, raising=False)
    assert numeric_hq.requested_builder({}) == "auto"
    assert numeric_hq.requested_builder(
        {"calculation": {"hamiltonian_builder": "Symbolic"}}) == "symbolic"


def test_numeric_without_rsmagcalc_warns_and_falls_back(monkeypatch, caplog):
    monkeypatch.setattr(numeric_hq, "rust_dipole_hq", lambda: None)
    sm = _model(_ex("spinw_tutorials", "SW02_AFM_chain", "config.yaml"))
    monkeypatch.setenv(numeric_hq.ENV, "numeric")
    S, p = runner.resolve_S_and_params(sm.config)
    with caplog.at_level(logging.WARNING, logger="magcalc.core"):
        calc = mc.MagCalc(spin_model_module=sm, spin_magnitude=S, hamiltonian_params=p,
                          cache_file_base="numeric_hq_test", cache_mode="none")
    assert calc.hamiltonian_builder == "symbolic"
    assert "numeric unavailable" in caplog.text and "DipoleHq" in caplog.text


def test_auto_falls_back_quietly_for_a_custom_model(monkeypatch, caplog):
    """A custom Python spin model has no bond table: auto uses gen_HM, at INFO."""
    sm = _model(_ex("spinw_tutorials", "SW02_AFM_chain", "config.yaml"))

    class Custom:
        def __getattr__(self, name):
            return getattr(sm, name)

    monkeypatch.setenv(numeric_hq.ENV, "auto")
    S, p = runner.resolve_S_and_params(sm.config)
    with caplog.at_level(logging.INFO, logger="magcalc.core"):
        calc = mc.MagCalc(spin_model_module=Custom(), spin_magnitude=S,
                          hamiltonian_params=p, cache_file_base="numeric_hq_test",
                          cache_mode="none")
    assert calc.hamiltonian_builder == "symbolic"
    assert "custom Python module" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING
                and "numeric" in r.getMessage()]


def test_pair_operator_is_refused_by_both_builders(monkeypatch):
    cfg = yaml.safe_load(open(_ex("spinw_tutorials", "SW02_AFM_chain", "config.yaml")))
    inter = cfg["interactions"]
    entry = {"type": "pair_operator", "pair": ["Cu1", "Cu2"], "rij_offset": [0, 0, 0],
             "poly": [0, 0, -0.4]}
    if isinstance(inter, dict):
        inter.setdefault("pair_operator", []).append(entry)
    else:
        inter.append(entry)
    for builder in ("numeric", "symbolic"):
        sm = _model(copy.deepcopy(cfg))
        monkeypatch.setenv(numeric_hq.ENV, builder)
        S, p = runner.resolve_S_and_params(sm.config)
        with pytest.raises((ValueError, RuntimeError), match="pair_operator"):
            mc.MagCalc(spin_model_module=sm, spin_magnitude=S, hamiltonian_params=p,
                       cache_file_base="numeric_hq_test", cache_mode="none")
