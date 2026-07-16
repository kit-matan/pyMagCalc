#!/usr/bin/env python3
"""Rb2Cu3SnF12 pinwheel VBS: high-order dimer series expansion + Dlog-Pade.

Builds the FULL 6-dimer deformed-kagome pinwheel from the published CIF (R-3,
a = 13.8771 A, c = 20.2392 A), assigns the four bond families J1..J4 by their
Cu-F-Cu superexchange angles (J increases with angle, as established for this
family), constructs the alternating out-of-plane DM pattern (D_z uniform when
every triangle is traversed counterclockwise; d_z = D_z/J = 0.18), and computes
the one-triplon dispersion by the linked-cluster dimer series expansion with
Dlog-Pade resummation -- the method of Matan et al., Nat. Phys. 6, 865 (2010)
and PRB 89, 024414 (2014).

    python series_dispersion.py [order]

Experimental gaps at the 2D zone centre: Delta_1 = 2.35(7) / 2.4(3) meV
(Stot^z = +/-1 doublet) and Delta_2 = 7.3(3) / 6.9(3) meV (Stot^z = 0).
The papers needed the series to EIGHTH order; run the highest order you can
afford (order 5 is minutes; order 6 is hours) and watch the convergence trend.
"""
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from magcalc.config_builder import MagCalcConfigBuilder          # noqa: E402
from magcalc.sun.dimer_series import DimerSeriesModel, resummed  # noqa: E402

# ------------------------------------------------------------------ constants
A_HEX, C_HEX = 13.8771, 20.2392
CU1 = [0.40529, 0.09039, -0.15637]
CU2 = [0.40529, -0.16590, -0.16837]
F_SITES = [  # from the CIF (F7B/F8B are partially occupied Sn-cap sites; not bridging)
    [0.3504, -0.0656, -0.1705], [0.2655, 0.0558, -0.1220],
    [0.2577, -0.2784, -0.1439], [0.4623, 0.2470, -0.1461],
    [0.4454, -0.3411, -0.2779], [0.5536, -0.0854, 0.0458],
]
J1 = 18.6            # meV (2010 paper fit)
J_RATIOS = {"J1": 1.0, "J2": 0.95, "J3": 0.85, "J4": 0.55}
DZ = 0.18            # D_z / J per bond


def _lattice():
    g = np.radians(120.0)
    return np.array([[A_HEX, 0, 0],
                     [A_HEX * np.cos(g), A_HEX * np.sin(g), 0],
                     [0, 0, C_HEX]])


def _expand_R3(pos):
    """All images of a fractional position under R-3 (hexagonal setting)."""
    x, y, z = pos
    rots = [(x, y, z), (-y, x - y, z), (-x + y, -x, z),
            (-x, -y, -z), (y, -x + y, -z), (x - y, x, -z)]
    cents = [(0, 0, 0), (2 / 3., 1 / 3., 1 / 3.), (1 / 3., 2 / 3., 2 / 3.)]
    out = []
    for r in rots:
        for c in cents:
            p = np.mod(np.array(r) + np.array(c), 1.0)
            if not any(np.allclose(p, q, atol=1e-4) or
                       np.allclose(np.abs(p - q), [1, 0, 0], atol=1e-4)
                       for q in out):
                out.append(p)
    return np.array(out)


def build_layer():
    """One kagome layer: 12 Cu (frac), all NN bonds with distance + Cu-F-Cu angle."""
    lat = _lattice()
    cfg = {"crystal_structure": {
        "lattice_parameters": {"a": A_HEX, "b": A_HEX, "c": C_HEX,
                               "alpha": 90, "beta": 90, "gamma": 120,
                               "space_group": 148},
        "wyckoff_atoms": [
            {"label": "Cu1", "pos": CU1, "spin_S": 0.5, "ion": "Cu2+"},
            {"label": "Cu2", "pos": CU2, "spin_S": 0.5, "ion": "Cu2+"}]}}
    b = MagCalcConfigBuilder.from_config(cfg)
    all_frac = np.array([a["pos"] for a in b.atoms_uc]) % 1.0
    # pick the layer around z ~ 0.84 (= -0.16 mod 1): exactly 12 Cu
    zs = all_frac[:, 2]
    z0 = zs[0]
    layer = all_frac[np.abs((zs - z0 + 0.5) % 1.0 - 0.5) < 0.06]
    assert len(layer) == 12, f"expected 12 Cu in a layer, got {len(layer)}"

    # all F images (any layer; the bridging test selects the right ones)
    fs = np.vstack([_expand_R3(f) for f in F_SITES])

    def cart(p):
        return np.asarray(p) @ lat

    # NN Cu-Cu bonds (3.30-3.60 A) with in-plane image search
    bonds = []
    for i in range(12):
        for j in range(12):
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    if j == i and ox == 0 and oy == 0:
                        continue
                    d = np.linalg.norm(cart(layer[j] + [ox, oy, 0]) - cart(layer[i]))
                    if 3.30 < d < 3.60:
                        key = (i, j, ox, oy)
                        rkey = (j, i, -ox, -oy)
                        if rkey not in [bb[0] for bb in bonds]:
                            bonds.append((key, d))
    # superexchange angle for each bond: bridging F within 2.35 A of both Cu
    out = []
    for (i, j, ox, oy), d in bonds:
        r1, r2 = cart(layer[i]), cart(layer[j] + [ox, oy, 0])
        best = None
        for f in fs:
            for fx in (-1, 0, 1):
                for fy in (-1, 0, 1):
                    for fz in (-1, 0, 1):
                        rf = cart(f + [fx, fy, fz])
                        d1, d2 = np.linalg.norm(rf - r1), np.linalg.norm(rf - r2)
                        if d1 < 2.35 and d2 < 2.35:
                            v1, v2 = r1 - rf, r2 - rf
                            ang = np.degrees(np.arccos(
                                np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                                        -1, 1)))
                            if best is None or ang > best:
                                best = ang
        out.append(((i, j, ox, oy), d, best))
    return lat, layer, out


def assign_families(bonds):
    """Group bonds by distance; map families to J1..J4 by descending mean angle."""
    fams = {}
    for (key, d, ang) in bonds:
        dk = round(d, 3)
        fams.setdefault(dk, []).append((key, ang))
    stats = sorted(((dk, np.mean([a for _k, a in v]), v) for dk, v in fams.items()),
                   key=lambda x: -x[1])   # descending angle
    names = ["J1", "J2", "J3", "J4"]
    table = {}
    print("bond families (by descending Cu-F-Cu angle):")
    for name, (dk, ang, v) in zip(names, stats):
        print(f"  {name}: d = {dk:.3f} A, angle = {ang:.1f} deg, {len(v)} bonds/cell")
        for key, _a in v:
            table[key] = name
    return table


def build_model(order_hint=None):
    lat, layer, bonds = build_layer()
    fam = assign_families(bonds)

    def cart(p):
        return np.asarray(p) @ lat

    # dimer partition on the J1 family
    used, units = set(), []
    for (key, d, _a) in bonds:
        i, j, ox, oy = key
        if fam[key] == "J1" and i not in used and j not in used:
            units.append([i, [j, [ox, oy, 0]]] if (ox, oy) != (0, 0) else [i, j])
            used.update([i, j])
    assert len(units) == 6 and len(used) == 12, "J1 must perfectly dimerize the layer"

    # triangles (for the DM circulation): triples pairwise NN-bonded
    bond_set = {}
    for (key, d, _a) in bonds:
        i, j, ox, oy = key
        bond_set[(i, j, ox, oy)] = fam[key]
        bond_set[(j, i, -ox, -oy)] = fam[key]

    # directed bond list with CCW-consistent DM: find for each undirected bond its
    # triangle, orient the triangle CCW (normal along +z), set D = (0,0, DZ*J_bond).
    Jval = {k: J1 * J_RATIOS[k] for k in J_RATIOS}
    directed = []
    for (key, d, _a) in bonds:
        i, j, ox, oy = key
        ri, rj = cart(layer[i]), cart(layer[j] + [ox, oy, 0])
        # third site of the triangle: NN of both ends
        third = None
        for (kk, dd, _aa) in bonds:
            pass
        for m in range(12):
            for mx in (-1, 0, 1):
                for my in (-1, 0, 1):
                    rm = cart(layer[m] + [mx, my, 0])
                    if (3.30 < np.linalg.norm(rm - ri) < 3.60
                            and 3.30 < np.linalg.norm(rm - rj) < 3.60):
                        third = rm
                        break
                if third is not None:
                    break
            if third is not None:
                break
        Jb = Jval[fam[key]]
        M = Jb * np.eye(3)
        if third is not None:
            nz = np.cross(rj - ri, third - rj)[2]
            sgn = 1.0 if nz > 0 else -1.0      # CCW as listed? flip D if not
            Dz = sgn * DZ * Jb
            M = M + np.array([[0, Dz, 0], [-Dz, 0, 0], [0, 0, 0]])
        directed.append((i, j, (ox, oy, 0), M))

    spins = [0.5] * 12
    pos_frac = layer.copy()
    dsm = DimerSeriesModel.from_spin_arrays(lat, spins, pos_frac, directed, units)
    return dsm, lat


def main():
    order = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    dsm, lat = build_model()
    B = 2 * np.pi * np.linalg.inv(lat).T
    qpts = {"Gamma": np.zeros(3), "M": np.array([0.5, 0, 0]) @ B,
            "K": np.array([1 / 3., 1 / 3., 0]) @ B}
    print(f"\n=== dimer series, orders 2..{order} (exp: 2.35/2.4 and 7.3/6.9 meV at Gamma) ===")
    for n in range(2, order + 1):
        t0 = time.time()
        rows = []
        for lbl, q in qpts.items():
            bands = dsm.band_series(q, n)
            vals_s = sorted(resummed(s, 1.0, "sum")[0] for s in bands)
            res = [resummed(s, 1.0, "dlog_pade") for s in bands]
            vals_d = sorted(v for v, _ in res)
            rows.append((lbl, vals_s[0], vals_s[-1], vals_d[0], vals_d[-1]))
        dt = time.time() - t0
        print(f"order {n} ({dt:.0f}s):")
        for lbl, s0, s1, d0, d1 in rows:
            print(f"  {lbl:6s} lowest: sum={s0:7.3f}  dlogPade={d0:7.3f}   "
                  f"highest: sum={s1:7.3f} dlogPade={d1:7.3f}")


if __name__ == "__main__":
    main()
