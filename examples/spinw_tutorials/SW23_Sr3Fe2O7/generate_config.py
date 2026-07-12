"""Generate config.yaml for SW23 (Sr3Fe2O7 bilayer helix, SpinW tutorial 23).

I4/mmm, a=3.8, c=20.4, Fe4+ S=2 at (0,0,0.0972) -> 4 Fe per conventional
cell in two "columns": column (0,0) hosts the bilayer around z=0
(z = +-0.0972) and column (1/2,1/2) the bilayer around z=1/2. Exchanges
(meV, positive = AFM) are entered as DISTANCE-BASED symmetry rules:
    J1  = -7.20  in-plane NN            (3.800 A, SpinW bond 1)
    Jc1 = -5.10  intra-bilayer vertical (3.966 A, bond 2)
    J2  =  1.05  in-plane diagonal      (5.374 A, bond 3)
    Jc3 =  0.01  inter-bilayer diagonal (6.789 A, bond 6)
    J3  =  2.10  in-plane 3rd NN        (7.600 A, bond 7)
The tutorial's easy-axis D = diag(0,0,-0.06) is NOT rotationally
invariant about the helix axis [110]; SpinW's incommensurate mode drops
such terms, and it is omitted here as well (|D| << J).

Magnetic structure: helix with k = (1/7, 1/7, 1), rotation axis
n = [1,1,0]. SpinW's genmagstr rotates by 2*pi*k.l over CELL translations
with per-atom S0; pyMagCalc's rotating frame uses the FULL atomic
positions, so the local directions are S0 back-rotated per site:
    n_i = R(-2*pi*k.d_i, [110]) . S0_i
(this script computes them). S0 follows the tutorial ([0,0,2] /
[-1.162,1.162,-1.140] alternating between the two COLUMNS, which keeps
every FM intra-bilayer pair exactly parallel; the 124.75-degree turn
happens between adjacent bilayers).

NOTE: the tutorial computes this spectrum with 'hermit',false, i.e. the
supplied helix is not the exact classical ground state of the truncated
model; the same caveat applies here (small imaginary parts near the
Goldstone points are truncated by the solver).
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _compact_yaml import dump as yaml_compact_dump

a, c = 3.8, 20.4
S = 2.0
zFe = 0.0972

k_vec = np.array([1 / 7, 1 / 7, 1.0])
axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)

# sites: label, frac pos, column (0 -> S0_a, 1 -> S0_b)
sites = [
    ("Fe1", np.array([0.0, 0.0, zFe]), 0),        # upper layer, bilayer @ z=0
    ("Fe2", np.array([0.0, 0.0, 1 - zFe]), 0),    # lower layer, bilayer @ z=1
    ("Fe3", np.array([0.5, 0.5, 0.5 + zFe]), 1),  # upper layer, bilayer @ z=1/2
    ("Fe4", np.array([0.5, 0.5, 0.5 - zFe]), 1),  # lower layer, bilayer @ z=1/2
]
S0 = [np.array([0.0, 0.0, 1.0]),
      np.array([-1.162, 1.162, -1.140]) / 2.0]
S0 = [v / np.linalg.norm(v) for v in S0]

def rot(theta, u):
    ux, uy, uz = u
    K = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

local_dirs = []
for lab, d, col in sites:
    n = rot(-2 * np.pi * float(np.dot(k_vec, d)), axis) @ S0[col]
    local_dirs.append([float(x) for x in n])

# bond distances (A) for the symmetry rules
rules = [
    (float(a), "J1", -7.20),
    (float(2 * zFe * c), "Jc1", -5.10),
    (float(np.sqrt(2) * a), "J2", 1.05),
    (float(np.hypot(np.sqrt(2) * a / 2, (0.5 - 2 * zFe) * c)), "Jc3", 0.01),
    (float(2 * a), "J3", 2.10),
]

header = open(__file__).read().split('"""')[1]
cfg = {
    "parameter_order": [name for _, name, _ in rules],
    "parameters": {name: val for _, name, val in rules},
    # Fe positions = the I4/mmm 4e Wyckoff orbit of (0,0,zFe); the engine
    # expands it (orbit order = the `sites` order used for local_directions,
    # verified band-by-band against explicit positions)
    "crystal_structure": {
        "lattice_parameters": {"a": a, "b": a, "c": c, "alpha": 90.0,
                               "beta": 90.0, "gamma": 90.0, "space_group": 139},
        "wyckoff_atoms": [{"label": "Fe", "pos": [0.0, 0.0, zFe],
                           "spin_S": S, "ion": "Fe3+"}],
    },
    "interactions": {
        "symmetry_rules": [
            {"type": "heisenberg", "distance": float(round(d, 5)), "value": name}
            for d, name, _ in rules
        ],
    },
    "magnetic_structure": {
        "enabled": True,
        "type": "spiral",
        "k": [float(x) for x in k_vec],
        "axis": [float(x) for x in axis],
        "local_directions": local_dirs,
    },
    "tasks": {"minimization": False, "dispersion": True},
    # tutorial Fig 3(f): (0.6,0.14,0) -> (1.4,0.14,0)
    "q_path": {
        "A": [0.6, 0.14, 0.0],
        "B": [1.4, 0.14, 0.0],
        "path": ["A", "B"],
        "points_per_segment": 240,
    },
    "plotting": {
        "save_plot": True,
        "show_plot": False,
        "plot_structure": False,
        "energy_limits_disp": [0, 35],
        "disp_plot_filename": "SW23_disp.png",
        "disp_title": "SW23 Sr3Fe2O7 helix: (0.6,0.14,0) -> (1.4,0.14,0)",
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
print(f"wrote {out}")
print("local_directions:", np.round(local_dirs, 4))
