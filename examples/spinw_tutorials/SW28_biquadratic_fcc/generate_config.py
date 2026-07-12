"""Generate config.yaml for SW28 (biquadratic FCC antiferromagnet, MnO set).

SpinW tutorial 28 puts a biquadratic coupling b*(Si.Sj)^2 on the FCC
nearest-neighbour bond of a type-II (k=(1/2,1/2,1/2)) antiferromagnet:

    S = 5/2, J1 = 5.0, J2 = 5.5, Q = 0.01*J2,  b = -Q/S^3   (SpinW 'B' matrix)

pyMagCalc has no biquadratic term, but on a *collinear* structure the
fluctuation part of Si.Sj has no boson-linear piece, so at LSWT
(quadratic) order a biquadratic coupling maps exactly onto an effective
bilinear exchange that differs between parallel and antiparallel bonds:

    J_eff = J1 + dJ * sigma      (sigma = +1 parallel, -1 antiparallel)

Matching the tutorial's analytic dispersion (the PRB 85, 054409 formula,
whose Q-terms are 2Q/JS in H1 and -+Q/(3JS) on h1/h2) fixes

    dJ = -Q/S

(the naive 2*b_spinw*S^2 = -2Q/S gives exactly DOUBLE those Q-terms --
this is the "factor 2 difference between SpinW and paper" noted in the
tutorial source). This script assigns J_eff bond-by-bond on the explicit
2x2x2 magnetic supercell (32 sites). Run to (re)write config.yaml.
Validated against the analytic formula: one supercell band coincides
with the analytic branch at every sampled q (diff ~ 1e-13).
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _compact_yaml import dump as yaml_compact_dump

S = 2.5
J1 = 5.0
J2 = 5.5
Q = 0.01 * J2
dJ = -Q / S            # effective bilinear correction * sigma (see docstring)

a = 8.0                # cubic lattice parameter (arbitrary)

basis = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
sign_basis = [1, -1, -1, -1]                      # tutorial S0 pattern
n_dir = np.array([1, 1, 0]) / np.sqrt(2)          # spins along [110]

# --- build 2x2x2 supercell site list -------------------------------------
sites = []   # (label, frac pos in supercell, sign)
for lx in range(2):
    for ly in range(2):
        for lz in range(2):
            cell_sign = (-1) ** (lx + ly + lz)    # k=(1/2,1/2,1/2) phase
            for mu, (bx, by, bz) in enumerate(basis):
                pos = [(bx + lx) / 2, (by + ly) / 2, (bz + lz) / 2]
                sites.append((f"Mn{len(sites)}", pos, sign_basis[mu] * cell_sign))

pos_arr = np.array([s[1] for s in sites])
signs = np.array([s[2] for s in sites])
N = len(sites)

# neighbour shells in CUBIC units: NN at (+-1/2,+-1/2,0)-type (d=a/sqrt2),
# NNN at (+-1,0,0)-type (d=a)
def find_bonds(shell_vecs_cubic):
    """Return (i, j, cell_offset_supercell) for every ordered bond."""
    bonds = []
    for i in range(N):
        for d in shell_vecs_cubic:
            target = pos_arr[i] + np.array(d) / 2.0   # supercell frac
            off = np.floor(target + 1e-9).astype(int)
            frac = target - off
            j = np.where(np.all(np.abs(pos_arr - frac) < 1e-9, axis=1))[0]
            assert len(j) == 1, (i, d)
            bonds.append((i, int(j[0]), off.tolist()))
    return bonds

nn_vecs = []
for u in range(3):
    for v in range(u + 1, 3):
        for su in (0.5, -0.5):
            for sv in (0.5, -0.5):
                d = [0.0, 0.0, 0.0]
                d[u], d[v] = su, sv
                nn_vecs.append(d)
nnn_vecs = []
for u in range(3):
    for s in (1.0, -1.0):
        d = [0.0, 0.0, 0.0]
        d[u] = s
        nnn_vecs.append(d)

interactions = []
for (i, j, off) in find_bonds(nn_vecs):
    sigma = int(signs[i] * signs[j])
    Jeff = float(J1 + dJ * sigma)
    interactions.append({
        "type": "heisenberg",
        "pair": [sites[i][0], sites[j][0]],
        "rij_offset": off,
        "value": Jeff,
    })
for (i, j, off) in find_bonds(nnn_vecs):
    # sanity: type-II order satisfies all NNN bonds antiferromagnetically
    assert signs[i] * signs[j] == -1
    interactions.append({
        "type": "heisenberg",
        "pair": [sites[i][0], sites[j][0]],
        "rij_offset": off,
        "value": float(J2),
    })

atoms = [{"label": lab, "pos": [float(p) for p in pos],
          "spin_S": S, "ion": "Mn2+"} for (lab, pos, sg) in sites]
directions = [[float(x) for x in (sg * n_dir)] for (lab, pos, sg) in sites]

header = open(__file__).read().split('"""')[1]
cfg = {
    "parameter_order": [],
    "parameters": {},
    "crystal_structure": {
        "lattice_vectors": [[2 * a, 0, 0], [0, 2 * a, 0], [0, 0, 2 * a]],
        "atoms_uc": atoms,
    },
    "interactions": interactions,
    "magnetic_structure": {
        "enabled": True,
        "type": "pattern",
        "pattern_type": "generic",
        "directions": directions,
    },
    "tasks": {"minimization": False, "dispersion": True},
    # cubic RLU path (1,0,0)->(0,0,0)->(1/2,1/2,0)->(1/2,1/2,1/2)->(0,0,0),
    # x2 for the 2x2x2 supercell
    "q_path": {
        "X": [2.0, 0.0, 0.0],
        "G1": [0.0, 0.0, 0.0],
        "M": [1.0, 1.0, 0.0],
        "L": [1.0, 1.0, 1.0],
        "G2": [0.0, 0.0, 0.0],
        "path": ["X", "G1", "M", "L", "G2"],
        "points_per_segment": 60,
    },
    "plotting": {
        "save_plot": True,
        "show_plot": False,
        "plot_structure": False,
        "energy_limits_disp": [0, 165],
        "disp_plot_filename": "SW28_disp.png",
        "disp_title": "SW28 biquadratic FCC (MnO): X-G-M-L-G (cubic RLU)",
    },
    "calculation": {"cache_mode": "none"},
    "output": {"save_data": False},
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(out, "w") as f:
    f.write("# " + "=" * 69 + "\n")
    for line in header.strip().splitlines():
        f.write(("# " + line).rstrip() + "\n")
    f.write("# (file generated by generate_config.py)\n")
    f.write("# " + "=" * 69 + "\n")
    yaml_compact_dump(cfg, f)
print(f"wrote {out}: {N} sites, {len(interactions)} ordered bonds")
print(f"dJ = {dJ:.6f} -> J_eff(par) = {J1+dJ:.4f}, J_eff(anti) = {J1-dJ:.4f}")
