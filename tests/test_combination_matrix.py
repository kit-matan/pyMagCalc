"""Engine mode x field x anisotropy x structure type (coverage-audit item 4).

The 2026-08-04 audit closed with the observation that a KEY-level sweep gives a lower
bound on the coverage gap, not its size, because both shipped field bugs were
present-key / absent-COMBINATION failures:

  * the Zeeman term was silently dropped in `mode: SUN` -- `H_mag` appeared in plenty
    of tests, never together with `mode: SUN`;
  * every field was forced along +z -- `H_dir` appeared in tests, never off the z axis.

Neither is visible to any audit that asks "is this key used somewhere?". This file
sweeps the cross-product instead, and leans on identities that hold across the WHOLE
grid rather than on per-cell reference numbers:

  1. an ISOTROPIC Hamiltonian is rotationally invariant, so rotating the field and the
     structure together must leave the spectrum unchanged -- in every engine. This is
     the exact identity the `H_dir`-forced-to-z bug violated.
  2. at S=1/2 with no anisotropy, SU(N) IS dipole LSWT (N=2 -> one boson), so the two
     engines must agree to machine precision at every point of the field grid.
  3. a magnetic supercell folds the bands of the chemical cell: the [2,1,1] spectrum
     at q is exactly {omega(q), omega(q+1/2)}.
  4. the exact dimer triplon sqrt(J^2 - J J' cos 2pi q) splits into omega, omega +/-
     gamma mu_B H under a field, with NO change from the field's direction.

THE TRAP THIS FILE ALMOST FELL INTO, and the reason `_calculator` exists:
`calculation.mode` is dispatched by the RUNNER (`runner.py`, "if mode == 'SUN'"), not
by `MagCalc`. A helper that builds `mc.MagCalc` directly and passes the config through
therefore runs the DIPOLE engine whatever `mode` says -- so a "SU(N) vs dipole"
comparison written that way comes back 0.000e+00 for every cell of the grid. A perfect
score, comparing the dipole engine with itself. It was caught only because the
entangled case has a distinctive expected answer (three triplon branches at the exact
bond-operator energy) and produced two wrong ones instead; the SU(N) comparison had no
such tell and looked flawless. `test_the_helper_really_dispatches_by_mode` now pins
the dispatch itself, because every other test here is worthless if it regresses.
"""
import copy
import logging

import numpy as np
import pytest

import magcalc as mc
from magcalc.generic_model import GenericSpinModel

LAT = [[3.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]
DIMER_LAT = [[3.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]]
QS = [[0.0, 0, 0], [0.17, 0, 0], [0.33, 0, 0], [0.5, 0, 0]]
MU_B, GAMMA = 5.788e-2, 2.0          # the engine's constants (see the test below)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _calculator(cfg, tag):
    """Build the calculator the way `runner.py` does -- mode dispatch included."""
    cfg = copy.deepcopy(cfg)
    mode = str(cfg.get("calculation", {}).get("mode", "dipole")).upper()
    model = GenericSpinModel(cfg)
    th, ph = model.generate_magnetic_structure()
    model.set_magnetic_structure(th, ph)

    params = []
    pvals = cfg.get("parameters", {})
    for key in cfg.get("parameter_order", []):
        v = pvals[key]
        if isinstance(v, (list, tuple)):
            params.extend(float(x) for x in v)
        else:
            params.append(float(v))

    if mode == "SUN":
        from magcalc.sun.adapter import SUNCalculator
        return SUNCalculator(model, cfg, params)
    if mode == "ENTANGLED":
        from magcalc.sun.entangled import EntangledCalculator
        return EntangledCalculator(model, cfg, params)
    S = cfg["crystal_structure"]["atoms_uc"][0]["spin_S"]
    return mc.MagCalc(spin_model_module=model, spin_magnitude=S, cache_mode="none",
                      cache_file_base=tag, hamiltonian_params=params)


def _bands(cfg, tag, qs=None, lat=None):
    calc = _calculator(cfg, tag)
    lat = np.array(lat if lat is not None else LAT, float)
    B = 2 * np.pi * np.linalg.inv(lat).T
    q = [np.array(x) @ B for x in (qs if qs is not None else QS)]
    return np.sort(np.real(calc.calculate_dispersion(q).energies), axis=1)


def _rot_to(n):
    """Rotation carrying +z onto the unit vector n."""
    n = np.array(n, float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(z, n), float(z @ n)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def _afm(mode="dipole", S=0.5, dirs=None, H=0.0, hdir=(0, 0, 1), sia=None, rcs=None):
    """Two-site antiferromagnetic chain -- the workhorse of the grid."""
    dirs = dirs if dirs is not None else [[0, 0, 1.0], [0, 0, -1.0]]
    cfg = {
        "crystal_structure": {"lattice_vectors": LAT,
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S},
                         {"label": "B", "pos": [0.5, 0, 0], "spin_S": S}]},
        "interactions": {"heisenberg": [
            {"pair": ["A", "B"], "rij_offset": [0, 0, 0], "value": 1.0},
            {"pair": ["B", "A"], "rij_offset": [0, 0, 0], "value": 1.0},
            {"pair": ["B", "A"], "rij_offset": [1, 0, 0], "value": 1.0},
            {"pair": ["A", "B"], "rij_offset": [-1, 0, 0], "value": 1.0}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                               "directions": [list(map(float, d)) for d in dirs]},
        "calculation": {"on_imaginary": "off", "mode": mode, "cache_mode": "none"},
        "tasks": {}}
    if sia is not None:
        cfg["interactions"]["single_ion_anisotropy"] = [
            {"value": sia, "axis": [0, 0, 1], "atoms": ["A", "B"]}]
    if rcs:
        cfg["calculation"]["anisotropy_renormalization"] = rcs
    if H:
        cfg["parameters"] = {"H_mag": H, "H_dir": list(map(float, hdir))}
        cfg["parameter_order"] = ["H_mag", "H_dir"]
    return cfg


def _rotated_afm(n, H, **kw):
    """The AFM chain with BOTH its spins and its field rotated onto the axis n."""
    u = (np.array(n, float) / np.linalg.norm(n)).tolist()
    R = _rot_to(u)
    dirs = [(R @ np.array([0, 0, 1.0])).tolist(), (R @ np.array([0, 0, -1.0])).tolist()]
    return _afm(dirs=dirs, H=H, hdir=u, **kw)


def _fm_chain(supercell=None, S=0.5, H=0.0, hdir=(0, 0, 1)):
    """One-atom ferromagnetic chain, for the structure-type axis."""
    cfg = {
        "crystal_structure": {"lattice_vectors": LAT,
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S}]},
        "interactions": {"heisenberg": [
            {"pair": ["A", "A"], "rij_offset": [1, 0, 0], "value": -1.0},
            {"pair": ["A", "A"], "rij_offset": [-1, 0, 0], "value": -1.0}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
        "calculation": {"on_imaginary": "off", "cache_mode": "none"}, "tasks": {}}
    if supercell:
        cfg["crystal_structure"]["magnetic_supercell"] = supercell
    if H:
        cfg["parameters"] = {"H_mag": H, "H_dir": list(map(float, hdir))}
        cfg["parameter_order"] = ["H_mag", "H_dir"]
    return cfg


J_D, JP_D = 4.0, 1.0


def _dimer(H=0.0, hdir=(0, 0, 1), S=0.5, sia=None, sia_axis=(0, 0, 1)):
    """Alternating chain of dimers -- the entangled-mode leg of the grid."""
    cfg = {
        "calculation": {"mode": "entangled", "cache_mode": "none",
                        "on_imaginary": "off"},
        "units": [["A", "B"]],
        "crystal_structure": {"lattice_vectors": DIMER_LAT,
            "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0], "spin_S": S},
                         {"label": "B", "pos": [0.4, 0, 0], "spin_S": S}]},
        "interactions": {"heisenberg": [
            {"pair": ["A", "B"], "rij_offset": [0, 0, 0], "value": J_D},
            {"pair": ["B", "A"], "rij_offset": [0, 0, 0], "value": J_D},
            {"pair": ["B", "A"], "rij_offset": [1, 0, 0], "value": JP_D},
            {"pair": ["A", "B"], "rij_offset": [-1, 0, 0], "value": JP_D}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
        "tasks": {}}
    if sia is not None:
        cfg["interactions"]["single_ion_anisotropy"] = [
            {"value": sia, "axis": list(map(float, sia_axis)), "atoms": ["A", "B"]}]
    if H:
        cfg["parameters"] = {"H_mag": H, "H_dir": list(map(float, hdir))}
        cfg["parameter_order"] = ["H_mag", "H_dir"]
    return cfg


def _dimer_bands(cfg, tag, qs):
    return _bands(cfg, tag, qs=qs, lat=DIMER_LAT)


# ---------------------------------------------------------------------------
# 0. the helper itself
# ---------------------------------------------------------------------------

def test_the_helper_really_dispatches_by_mode():
    """EVERY other test here is vacuous if `_calculator` silently returns the dipole
    engine for `mode: SUN` -- see the module docstring. S=1 with a single-ion
    anisotropy is the sharpest discriminator: dipole gives ONE boson per site (2 bands
    for 2 sites) while SU(N) gives N-1 = 2 (4 bands), because the single-ion
    (multipolar) excitations exist only in SU(N)."""
    nd = _bands(_afm("dipole", S=1.0, sia=-0.8), "disp_d").shape[1]
    ns = _bands(_afm("SUN", S=1.0, sia=-0.8), "disp_s").shape[1]
    assert (nd, ns) == (2, 4), f"dipole gave {nd} bands, SU(N) gave {ns}"


# ---------------------------------------------------------------------------
# 1. field DIRECTION x engine mode -- the combination that hid the +z bug
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [[1, 0, 0], [0, 1, 0], [1, 1, 1], [2, -1, 3]])
def test_dipole_is_rotationally_invariant_in_field(n):
    """An isotropic Heisenberg AFM in a field has no preferred lab axis: rotate the
    spins and the field together and the spectrum cannot move. The `H_dir`-forced-to-z
    bug broke exactly this -- the structure rotated, the field did not."""
    ref = _bands(_rotated_afm([0, 0, 1], 3.0), "inv_ref")
    got = _bands(_rotated_afm(n, 3.0), f"inv_{n}")
    assert got == pytest.approx(ref, abs=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("n", [[1, 0, 0], [1, 1, 1]])
def test_sun_is_rotationally_invariant_in_field(n):
    ref = _bands(_rotated_afm([0, 0, 1], 3.0, mode="SUN"), "sinv_ref")
    got = _bands(_rotated_afm(n, 3.0, mode="SUN"), f"sinv_{n}")
    assert got == pytest.approx(ref, abs=1e-6)


def test_entangled_field_direction_matters_only_against_an_anisotropy():
    """A cautionary cell of the grid. The ISOTROPIC dimer's reference is a singlet
    with no ordered moment, so its spectrum depends on |H| alone -- an invariance test
    written on it CANNOT FAIL for a direction bug, and indeed one written that way
    survived a deliberate `_resolve_field`-forced-to-+z mutation untouched.

    An easy-axis anisotropy gives the unit a direction, and only then is the check
    real: field parallel to the SIA axis differs from field perpendicular to it, while
    rotating the SIA axis AND the field together is a symmetry.
    """
    qs = [[0.0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]]
    kw = dict(S=1.0, sia=-1.2)
    par = _dimer_bands(_dimer(3.0, [0, 0, 1], sia_axis=[0, 0, 1], **kw), "ep", qs)
    perp = _dimer_bands(_dimer(3.0, [1, 0, 0], sia_axis=[0, 0, 1], **kw), "eq", qs)
    assert np.abs(par - perp).max() > 0.1        # the direction is now observable

    for n in ([1, 0, 0], [1, 1, 1]):
        u = (np.array(n, float) / np.linalg.norm(n)).tolist()
        got = _dimer_bands(_dimer(3.0, u, sia_axis=u, **kw), f"er_{n}", qs)
        assert got == pytest.approx(par, abs=1e-8)


def test_entangled_mode_applies_on_site_anisotropy_at_all():
    """The regression for a term SILENTLY DROPPED until 2026-08-05: the entangled
    builder assembled each unit's on-site block from the bilinear pair terms plus the
    Zeeman and never read `single_ion_anisotropy` / `sia_matrix` / `stevens`, while
    `_reject_unsupported_terms` let them through. A D = -5 meV anisotropy on an S=1
    dimer moved the triplon by EXACTLY 0.000.

    Found by this file's mode x anisotropy axis -- no key-level audit could see it,
    since every one of those keys is used in plenty of dipole and SU(N) tests.
    Correctness (as opposed to mere presence) is pinned against exact diagonalization
    of the isolated dimer in tests/test_entangled_units.py.
    """
    qs = [[0.0, 0, 0], [0.5, 0, 0]]
    bare = _dimer_bands(_dimer(S=1.0), "drop_bare", qs)
    for kw in ({"sia": -5.0},):
        got = _dimer_bands(_dimer(S=1.0, **kw), "drop_sia", qs)
        assert np.abs(got - bare).max() > 1.0, f"{kw} was dropped"


def test_spin_half_anisotropy_is_correctly_inert_in_entangled_mode():
    """The other half of the story, and the reason the drop above was easy to miss:
    at S=1/2 an (S.n)^2 anisotropy IS a constant (Sz^2 = I/4), so it cannot shift any
    excitation. "No effect at S=1/2" is correct physics, not a dropped term -- which
    is exactly why the S=1 case above is the one that carries the regression."""
    qs = [[0.0, 0, 0], [0.5, 0, 0]]
    bare = _dimer_bands(_dimer(S=0.5), "inert_bare", qs)
    got = _dimer_bands(_dimer(S=0.5, sia=-5.0), "inert_sia", qs)
    assert got == pytest.approx(bare, abs=1e-9)


def test_a_transverse_field_is_not_the_same_as_a_longitudinal_one():
    """Anti-vacuity for all of the above: if the field were dropped or forced onto the
    structure's axis, every invariance test would pass trivially. Here the field is
    turned WITHOUT turning the spins, and the spectrum must move."""
    longitudinal = _bands(_afm(H=3.0, hdir=[0, 0, 1]), "long")
    transverse = _bands(_afm(H=3.0, hdir=[1, 0, 0]), "trans")
    assert np.abs(longitudinal - transverse).max() > 0.1


def test_H_dir_is_not_normalised_so_only_the_product_matters():
    """A sharp edge worth pinning: `H_dir` is deliberately NOT normalised, so the
    field is H_mag * |H_dir|. `H_mag: 1` with `H_dir: [0,0,3]` is a 3 T field, not a
    1 T one. Normalising it "for safety" would silently rescale every config that
    relies on this."""
    a = _bands(_afm(H=3.0, hdir=[0, 0, 1]), "nn_a")
    b = _bands(_afm(H=1.0, hdir=[0, 0, 3]), "nn_b")
    assert a == pytest.approx(b, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. engine mode x field -- S=1/2 SU(N) IS dipole LSWT
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("n,H", [([0, 0, 1], 0.0), ([0, 0, 1], 3.0),
                                 ([1, 0, 0], 3.0), ([1, 1, 1], 3.0),
                                 ([2, -1, 3], 5.0)])
def test_sun_equals_dipole_at_spin_half_across_the_field_grid(n, H):
    """The load-bearing identity of the whole matrix. At S=1/2 there is one boson per
    site either way, so the engines must agree to machine precision -- at zero field,
    along z, and off z. The SU(N)-Zeeman-dropped bug fails the H>0 cells; the
    direction bug fails the off-z ones."""
    d = _bands(_rotated_afm(n, H, mode="dipole"), f"eq_d{n}{H}")
    s = _bands(_rotated_afm(n, H, mode="SUN"), f"eq_s{n}{H}")
    assert d == pytest.approx(s, abs=1e-6)


def test_spin_half_equivalence_is_quick_at_least_once():
    """Keeps the identity in the FAST suite (CLAUDE.md 5f2: every feature needs one
    quick pinned test outside `slow`)."""
    d = _bands(_rotated_afm([1, 0, 0], 3.0, mode="dipole"), "q_d")
    s = _bands(_rotated_afm([1, 0, 0], 3.0, mode="SUN"), "q_s")
    assert d == pytest.approx(s, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. engine mode x ANISOTROPY -- where the engines must NOT agree
# ---------------------------------------------------------------------------

def test_dipole_and_sun_differ_once_anisotropy_meets_spin_one():
    """The other side of the identity, and the documented reason `mode: SUN` exists:
    with S >= 1 and an anisotropy the single-ion bands are structurally absent from
    dipole LSWT. If these ever agreed, the S=1/2 tests above would be measuring
    nothing."""
    d = _bands(_afm("dipole", S=1.0, sia=-0.8), "an_d")
    s = _bands(_afm("SUN", S=1.0, sia=-0.8), "an_s")
    assert d.shape[1] == 2 and s.shape[1] == 4
    assert np.abs(d - s[:, :2]).max() > 0.1


def test_rcs_restores_the_spin_half_identity_under_anisotropy():
    """lambda_2(1/2) = 0 is the cleanest statement of the whole RCS story: (S.n)^2 IS
    a constant at S=1/2, so a quadratic anisotropy can have no effect -- SU(N) knows
    that, the un-renormalized classical polynomial does not. So dipole+SIA disagrees
    with SU(N), and dipole+SIA+`anisotropy_renormalization: rcs` agrees EXACTLY.
    A three-way check across mode x anisotropy x renormalization."""
    sun = _bands(_afm("SUN", S=0.5, sia=-0.8), "rcs_s")
    raw = _bands(_afm("dipole", S=0.5, sia=-0.8), "rcs_d")
    rcs = _bands(_afm("dipole", S=0.5, sia=-0.8, rcs="rcs"), "rcs_r")
    assert np.abs(raw - sun).max() > 0.5          # uncorrected really is different
    assert rcs == pytest.approx(sun, abs=1e-9)    # and RCS lands exactly on SU(N)


# ---------------------------------------------------------------------------
# 4. STRUCTURE TYPE -- a supercell must fold the chemical bands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("H", [0.0, 4.0])
def test_magnetic_supercell_folds_the_chemical_bands(H):
    """[2,1,1] doubles the cell, so its spectrum at q is exactly the pair
    {omega(q), omega(q+1/2)} of the chemical cell -- q_path stays in CHEMICAL RLU.
    Run with and without a field, since the supercell replication has to carry the
    Zeeman term across replicas too."""
    qs = [[0.0, 0, 0], [0.13, 0, 0], [0.25, 0, 0], [0.4, 0, 0]]
    shifted = [[q[0] + 0.5, 0, 0] for q in qs]
    chem = _bands(_fm_chain(H=H), f"fold_c{H}", qs)[:, 0]
    chem_shift = _bands(_fm_chain(H=H), f"fold_s{H}", shifted)[:, 0]
    super_ = _bands(_fm_chain(supercell=[2, 1, 1], H=H), f"fold_2{H}", qs)
    folded = np.sort(np.stack([chem, chem_shift], axis=1), axis=1)
    assert super_.shape == (len(qs), 2)
    assert super_ == pytest.approx(folded, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. field MAGNITUDE, consistently across engines
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("H", [2.0, 5.0])
def test_ferromagnet_zeeman_gap_is_gamma_mu_B_H(H):
    """A saturated ferromagnet's magnon gap in a longitudinal field is exactly
    gamma*mu_B*H with gamma = 2 -- the calibration CLAUDE.md quotes. Checks the
    magnitude, not just the presence, of the Zeeman term."""
    q0 = [[0.0, 0, 0]]
    gap0 = _bands(_fm_chain(), f"z0_{H}", q0)[0, 0]
    gapH = _bands(_fm_chain(H=H), f"zH_{H}", q0)[0, 0]
    assert gap0 == pytest.approx(0.0, abs=1e-6)
    assert gapH - gap0 == pytest.approx(GAMMA * MU_B * H, rel=1e-6)


def test_entangled_field_splits_the_exact_triplon():
    """Entangled x field against a CLOSED FORM: the dimer triplon
    omega(q) = sqrt(J^2 - J J' cos 2pi q) is threefold degenerate at zero field and
    splits into omega and omega +/- gamma*mu_B*H (the S^z = 0, -+1 members). Pins the
    zero-field energies, the splitting, and the multiplicity in one shot."""
    qs = [[0.0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]]
    exact = np.sqrt(J_D ** 2 - J_D * JP_D
                    * np.cos(2 * np.pi * np.array([q[0] for q in qs])))
    b0 = _dimer_bands(_dimer(), "tri0", qs)
    assert b0.shape == (3, 3)
    assert b0 == pytest.approx(np.repeat(exact[:, None], 3, axis=1), abs=1e-8)

    H = 3.0
    z = GAMMA * MU_B * H
    pred = np.sort(np.stack([exact - z, exact, exact + z], axis=1), axis=1)
    assert _dimer_bands(_dimer(H), "triH", qs) == pytest.approx(pred, abs=1e-8)


# The modules that carry a Zeeman term, and the names they bind mu_B / gamma to.
# Every one of these must be the object from magcalc.constants, not a copy.
_ZEEMAN_MODULES = {
    "magcalc.generic_model": ("MU_B", "GAMMA_ELECTRON"),
    "magcalc.spiral_opt": ("MU_B", "GAMMA"),
    "magcalc.thermal_mc": ("MU_B", "GAMMA"),
    "magcalc.sun.lswt": ("_MU_B", "_GAMMA"),
    "magcalc.sun.entangled": ("MU_B", "GAMMA"),
    "magcalc.sun.dimer_series": ("MU_B", "GAMMA"),
}


def test_every_engine_uses_the_same_bohr_magneton():
    """mu_B USED TO BE a magic number duplicated across six modules (generic_model x2,
    spiral_opt, thermal_mc, sun/lswt, sun/entangled, sun/dimer_series), four of them
    as a function-local. Nothing tied them together, so a "fix" in one place would
    have desynced the engines silently -- the field would then differ between dipole
    and SU(N) by a few parts in 10^4, which is far too small to notice in a spectrum
    and far too large to be rounding.

    They now all import `magcalc.constants`. This test pins BOTH halves of that:
    (a) every engine really does bind the shared object, and (b) nobody has
    reintroduced a literal of their own. (b) is the one that matters -- a stray
    `mu_B = 5.788e-2` back inside a function would restore the original hazard while
    (a) still passed, since the module-level import would sit there unused.

    Also pins the value against CODATA. The engine's 5.788e-2 meV/T is the CODATA
    value truncated to four figures (5.7883818e-2), i.e. 6.6e-5 relative -- fine, but
    it should be a deliberate 6.6e-5, not a drifting one.
    """
    import importlib
    import pathlib
    import re

    from magcalc import constants, spiral_opt

    # (a) one object, shared. `is` rather than `==`, so a re-typed literal that
    # happens to be equal today still fails.
    for mod_name, (mu_name, gamma_name) in _ZEEMAN_MODULES.items():
        mod = importlib.import_module(mod_name)
        assert getattr(mod, mu_name) is constants.MU_B, f"{mod_name}.{mu_name}"
        assert getattr(mod, gamma_name) is constants.GAMMA_ELECTRON, \
            f"{mod_name}.{gamma_name}"

    # (b) no module has re-typed the number. Search the whole package, not just the
    # six above -- a NEW engine with its own literal is exactly the regression this
    # is here to catch.
    root = pathlib.Path(spiral_opt.__file__).parent
    literal = re.compile(r"(?<![\w.])5\.788\d*e-0?2")
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "constants.py":
            continue            # the one place it is allowed to appear
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]        # comments may quote the value
            if literal.search(code):
                offenders.append(f"{path.relative_to(root)}:{n}: {line.strip()}")
    assert not offenders, (
        "mu_B literal re-introduced; import it from magcalc.constants instead:\n  "
        + "\n  ".join(offenders)
    )

    codata_mu_b = 5.7883817982e-2        # meV/T
    assert constants.MU_B == pytest.approx(codata_mu_b, rel=2e-4)
    # the constant this file asserts the physics against must be that same one
    assert MU_B == pytest.approx(constants.MU_B, rel=1e-12)
    assert GAMMA == pytest.approx(constants.GAMMA_ELECTRON, rel=1e-12)
