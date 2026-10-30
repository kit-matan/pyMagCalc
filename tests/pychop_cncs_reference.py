"""Generator of tests/data_pychop_cncs_reference.json: Mantid PyChop's CNCS resolution
(the vanadium FWHM vs energy transfer, every component it adds in quadrature, and the
chopper and moderator time widths it used) for the three chopper-4 variants, four Ei
and four (resolution disk, chopper 1) frequency pairs.

Needs Mantid (PyChop ships in its scripts/ directory); run with `mantidpython`, then
keep the fields test_resolution.py reads. Not collected by pytest.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/usr/local/pixi/mantid/.pixi/envs/default/scripts")
import mantid                                    # noqa: E402
from pychop.Instruments import Instrument       # noqa: E402

CASES = []
for variant in ("High Flux", "Intermediate", "High Resolution"):
    for ei in (12.0, 6.25, 3.3, 25.0):
        for f_res, f1 in ((300.0, 60.0), (240.0, 60.0), (300.0, 180.0), (180.0, 60.0)):
            CASES.append((variant, ei, f_res, f1))

out = {"mantid_version": mantid.__version__, "cases": []}
for variant, ei, f_res, f1 in CASES:
    inst = Instrument("CNCS")
    inst.setChopper(variant)
    inst.setFrequency([f_res, f1])
    inst.setEi(ei)
    et = np.linspace(-0.5 * ei, 0.9 * ei, 15)
    fwhm = np.asarray(inst.getResolution(et), float)
    v_van, comps, modchop = inst.getVanVar(ei, None, et)
    wd = inst.chopper_system.getWidth(ei)
    case = {"variant": variant, "ei": ei, "f_res": f_res, "f1": f1,
            "etrans": et.tolist(), "fwhm": fwhm.tolist(),
            "v_van_s2": np.atleast_1d(v_van).tolist(),
            "components_s2": {k: np.atleast_1d(v).tolist() for k, v in comps.items()},
            "tsqmod_tsqchp_x0": [float(x) for x in modchop],
            "chop_width_s": [float(w) for w in wd],
            "moderator_fwhm_s": float(inst.moderator.getWidth(ei)),
            "distances": [float(d) for d in inst.chopper_system.getDistances()]}
    out["cases"].append(case)
    print(f"case {variant} Ei {ei} f_res {f_res} f1 {f1}: FWHM(0) {np.interp(0, et, fwhm):.4f}",
          flush=True)
json.dump(out, open("pychop_cncs_reference.json", "w"), indent=1)
print("saved pychop_cncs_reference.json", len(out["cases"]))
