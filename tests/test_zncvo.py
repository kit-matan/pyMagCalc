"""ZnCu2V2O7 (ZnCVO) declarative config regression.

The old config_modern.yaml was broken THREE ways at once (it produced either a wrong
minimised state or an all-zero Hamiltonian). config_zncvo.yaml fixes them; this pins each
fix at the model level (fast -- no eigensolve). The full dispersion was validated
band-by-band against the reference spin_model.py + disp_ZnCVO.py to ~5e-5 meV by hand.

The three bugs:
  1. no `magnetic_structure` -> the runner minimised into the wrong state;
  2. atom `pos` were CARTESIAN while the declarative engine treats pos as FRACTIONAL, so
     every bond landed ~50 A away and NOTHING coupled (all-zero Hamiltonian);
  3. the anisotropic_exchange values pre-baked a 1/2 that the engine also applies ->
     half-strength exchange anisotropy.
"""
import os
from collections import Counter

import numpy as np
import yaml

from magcalc.generic_model import GenericSpinModel

HERE = os.path.dirname(__file__)
ZNCVO = os.path.join(HERE, "..", "examples", "materials", "ZnCVO")
CFG = os.path.join(ZNCVO, "config_zncvo.yaml")
# reference params (disp_ZnCVO.py): [J1,J2,J3,J4,J5,J6,J7,G,H]
REF_P = [8.497751, 0, 0, 0, 5.261605, 1.873546, 0.5095509, 0.00447892, 0]


def _model_and_params():
    cfg = yaml.safe_load(open(CFG))
    m = GenericSpinModel(cfg)
    order = cfg["parameter_order"]
    P = cfg["parameters"]
    pv = []
    for k in order:
        v = P[k]
        pv.extend(v) if isinstance(v, (list, tuple)) else pv.append(v)
    return cfg, m, pv


def test_uses_the_symmetry_model():
    """Structure via space group + Wyckoff (one representative Cu on 8f expands to 8),
    interactions via symmetry_rules -- the repo's 'symmetry model first' convention."""
    cfg = yaml.safe_load(open(CFG))
    cs = cfg["crystal_structure"]
    assert cs["lattice_parameters"]["space_group"] == 15          # C2/c
    assert len(cs["wyckoff_atoms"]) == 1                          # one Cu, 8f orbit
    assert "atoms_uc" not in cs                                   # not hand-listed
    assert len(cfg["interactions"]["symmetry_rules"]) == 4        # J1,J5,J6,J7 distance rules

    m = GenericSpinModel(cfg)
    assert len(m.config["crystal_structure"]["atoms_uc"]) == 8    # Wyckoff expanded to 8


def test_magnetic_structure_is_the_pm_c_afm():
    """Bug 1: the +/- c collinear AFM must be supplied (8 directions, alternating,
    net-zero moment)."""
    cfg = yaml.safe_load(open(CFG))
    ms = cfg["magnetic_structure"]
    dirs = np.array(ms["directions"])
    assert dirs.shape == (8, 3)
    assert np.allclose(dirs.sum(axis=0), 0, atol=1e-6)        # antiferromagnetic
    assert np.allclose(np.abs(dirs[0]), np.abs(dirs[1]))      # up/down same axis


def test_exchange_bonds_match_the_reference_model():
    """The declarative Heisenberg bond set must be identical to spin_model.py's."""
    # Load THIS folder's spin_model.py by path -- several materials ship a
    # `spin_model.py`, so a bare `import spin_model` would return whichever the
    # suite imported first (sys.modules cache collision).
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "zncvo_spin_model", os.path.join(ZNCVO, "spin_model.py"))
    sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm)
    _, m, pv = _model_and_params()
    Jd, _, _ = m.spin_interactions(pv)
    Js, _, _ = sm.spin_interactions(REF_P)

    def counts(J):
        return Counter(round(float(J[i, j]), 4)
                       for i in range(J.shape[0]) for j in range(J.shape[1]) if J[i, j] != 0)
    assert counts(Jd) == counts(Js)
    assert dict(counts(Jd)) == {8.4978: 8, 0.5096: 16, 5.2616: 16, 1.8735: 8}


def test_anisotropy_is_full_strength_not_half():
    """Bug 3: the per-bond anisotropic exchange must be FULL strength. The config must NOT
    pre-bake the engine's own 1/2, so the raw K-vector the model returns is G*J*const
    (the engine then applies its 1/2 to reach the physical 0.5*G*J*const, matching
    spin_model). If the 0.5 were pre-baked, this raw value would be half of the below."""
    import math
    _, m, pv = _model_and_params()
    _, _, Kd = m.spin_interactions(pv)
    G, J1, beta = 0.00447892, 8.497751, 110.251999
    sB, cB = math.sin(math.radians(beta)), math.cos(math.radians(beta))
    expect = sorted([G * J1 * (cB - sB), G * J1 * (-(sB + cB)), G * J1 * (sB - cB)])
    # find a J1 bond's Kex (the largest-magnitude anisotropy)
    best = None
    for i in range(len(Kd)):
        for j in range(len(Kd[0])):
            K = Kd[i][j]
            if K is None or (hasattr(K, "is_zero_matrix") and K.is_zero_matrix):
                continue
            vals = sorted(float(K[a]) for a in range(3))
            if best is None or abs(vals[0]) > abs(best[0]):
                best = vals
    assert np.allclose(best, expect, atol=1e-5), f"{best} vs {expect} (half-strength bug?)"


def test_config_validates():
    from magcalc.schema import MagCalcConfig
    MagCalcConfig.model_validate(yaml.safe_load(open(CFG)))
