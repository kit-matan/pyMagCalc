#!/usr/bin/env python3
"""Field dependence of the Rb2Cu3SnF12 pinwheel-dimer triplet (paper Fig. 4d).

Sweeps a c-axis field and plots the three triplon gaps of a single J1 dimer with
out-of-plane DM. The Stot^z = +/-1 doublet Zeeman-splits linearly (slope -/+ g mu_B),
while the Stot^z = 0 branch is field-independent -- exactly Matan et al., Nat. Phys.
6, 865 (2010), Fig. 4. Run:  python field_sweep.py
"""
import copy
import os

import numpy as np
import yaml

from magcalc.generic_model import GenericSpinModel
from magcalc.sun.entangled import build_entangled_model

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))


def gaps(B):
    cfg = copy.deepcopy(BASE)
    cfg["parameters"]["H_mag"] = float(B)
    m = GenericSpinModel(cfg)
    pv = [cfg["parameters"][k] for k in cfg["parameter_order"]]
    sm = build_entangled_model(m, pv, units=[[0, 1]])
    return np.sort(np.real(sm.dispersion(np.zeros(3))))


fields = np.linspace(0, 40, 41)
G = np.array([gaps(B) for B in fields])          # (nB, 3): [Sz=+1, Sz=0, Sz=-1]

# g from the Zeeman slope of the split branches (meV/T -> g via mu_B = 5.788e-2)
slope = np.polyfit(fields, G[:, 2] - G[:, 0], 1)[0] / 2.0     # (E(-1)-E(+1)) = 2 g mu_B B
print(f"Zeeman slope -> g = {slope / 5.788e-2:.3f}  (input g = 2)")
print(f"Stot^z = 0 branch is flat: {G[:,1].std():.2e} meV spread")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(fields, G[:, 0], "-", label=r"$S^z_{tot}=+1$")
    plt.plot(fields, G[:, 1], "-", label=r"$S^z_{tot}=0$")
    plt.plot(fields, G[:, 2], "-", label=r"$S^z_{tot}=-1$")
    plt.xlabel("Field H (T)"); plt.ylabel(r"$\hbar\omega$ (meV)")
    plt.title("Rb2Cu3SnF12 pinwheel dimer: triplet vs field (cf. Fig. 4d)")
    plt.legend(); plt.tight_layout()
    out = os.path.join(HERE, "Rb2Cu3SnF12_field_sweep.png")
    plt.savefig(out, dpi=120)
    print(f"saved {out}")
except Exception as e:
    print(f"(plot skipped: {e})")
