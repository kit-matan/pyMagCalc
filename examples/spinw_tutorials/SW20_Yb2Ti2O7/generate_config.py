"""Generate config.yaml for SW20 (Yb2Ti2O7, SpinW tutorial 20), ZERO FIELD.

Yb2Ti2O7 pyrochlore, Fd-3m (origin choice 2 = the tutorial's 'F d -3 m Z'),
a = 10.0307, Yb3+ (S_eff = 1/2) on 16d (1/2,1/2,1/2). Anisotropic NN
exchange with the Hamiltonian of Ross et al., PRX 1, 021002 (2011):
J1 = -0.09, J2 = -0.22, J3 = -0.29, J4 = 0.01 meV.

The tutorial's published getmatrix() output defines the reference bond
r_i = (1/2,1/2,1/2) -> r_j = (1/2,1/4,1/4) (d = 3.546 A, centre
(1/2,3/8,3/8)) with allowed form [[C,0,0],[0,A,B],[0,B,A]] plus DM
(0,D1,-D1); setmatrix('pref',[J1 J3 J2 -J4]) therefore builds

    J_ref = [ J2  J4  J4 ]
            [-J4  J1  J3 ]
            [-J4  J3  J1 ]   (global cubic axes)

exactly the Ross et al. exchange matrix. The config carries J_ref as a
single symmetry rule (ref_pair Yb0 -> Yb4, the bond above): pyMagCalc
detects Fd-3m with spglib and propagates J -> R J R^T over all 96
ordered NN bonds itself, exactly as SpinW does. (This script still
builds the explicit matrices internally, but only to find the classical
ground state; verified identical to the engine's propagation.)

The classical ground state at B=0 is found here by steepest descent from
many random starts: a SPLAYED FERROMAGNET, net moment along a cubic axis
with the four sublattices canted symmetrically away from it (the known
zero-field ground state of these parameters), supplied as directions.

LIMITATION (documented in the README): the tutorial computes spectra in
B = 5 T and 2 T fields along [1,-1,0] with the anisotropic g-tensor
(gxy = 4.32, gz = 1.8 in the local <111> frames). pyMagCalc's Zeeman
term is a global isotropic H.S, so the in-field part is not ported;
this config gives the zero-field spin waves of the same Hamiltonian
along the tutorial's first path (-1/2,-1/2,-1/2) -> (2,2,2).

NOTE: the zero-field splayed FM has a nearly degenerate manifold of
states; its four soft pseudo-Goldstone modes carry small imaginary
parts at every q (truncated by the solver, same for explicit and
symmetry-propagated Hamiltonians).
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _compact_yaml import dump as yaml_compact_dump
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

a = 10.0307
J1, J2, J3, J4 = -0.09, -0.22, -0.29, 0.01
S = 0.5

J_ref = np.array([[J2, J4, J4],
                  [-J4, J1, J3],
                  [-J4, J3, J1]])
ri_ref = np.array([0.5, 0.5, 0.5])
rj_ref = np.array([0.5, 0.25, 0.25])

# 16d orbit (origin choice 2): 4 tetrahedron corners + F-centring
base = [(0.5, 0.5, 0.5), (0.5, 0.25, 0.25), (0.25, 0.5, 0.25), (0.25, 0.25, 0.5)]
fcc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
frac = np.array([[(b[k] + t[k]) % 1.0 for k in range(3)]
                 for b in base for t in fcc])
s = Structure(Lattice.cubic(a), ["Yb"] * 16, frac)
assert len(frac) == 16
ops = SpacegroupAnalyzer(s).get_symmetry_operations()   # fractional ops
assert any(True for _ in ops)

def site_index(f):
    d = (frac - f + 0.5) % 1.0 - 0.5
    hit = np.where(np.all(np.abs(d) < 1e-6, axis=1))[0]
    return int(hit[0]) if len(hit) else None

# enumerate ordered NN bonds (d = a*sqrt(2)/4)
d_nn = a * np.sqrt(2) / 4
from itertools import product
offsets = [np.array(o) for o in product((-1, 0, 1), repeat=3)]
bonds = []
for i in range(16):
    for j in range(16):
        for off in offsets:
            if i == j and not off.any():
                continue
            if abs(np.linalg.norm(frac[j] + off - frac[i]) * a - d_nn) < 1e-3:
                bonds.append((i, j, off))
assert len(bonds) == 96, len(bonds)

# map the reference bond onto each bond with a space-group op
def bond_matrix(i, j, off):
    target_i, target_j = frac[i], frac[j] + off
    for op in ops:
        gi = op.operate(ri_ref)
        gj = op.operate(rj_ref)
        t = target_i - gi                       # required lattice shift
        if np.all(np.abs(t - np.round(t)) < 1e-6) and \
           np.all(np.abs((gj + np.round(t)) - target_j) < 1e-6):
            R = op.rotation_matrix               # cubic: frac == cart
            return R @ J_ref @ R.T
    return None

interactions = []
n_forward = 0
for (i, j, off) in bonds:
    M = bond_matrix(i, j, off)
    if M is None:
        # reversed orientation: transpose of the partner bond's matrix
        M = bond_matrix(j, i, [-x for x in off])
        assert M is not None, (i, j, off)
        M = M.T
    else:
        n_forward += 1
    interactions.append({
        "type": "interaction_matrix",
        "pair": [f"Yb{i}", f"Yb{j}"],
        "rij_offset": [int(x) for x in off],
        "value": [[float(x) for x in row] for row in M],
    })
print(f"{n_forward}/96 bonds matched directly, rest via transpose")

# --- classical ground state by steepest descent -------------------------
Jmats = {}
for entry, (i, j, off) in zip(interactions, bonds):
    Jmats.setdefault(i, []).append((j, np.array(entry["value"])))

rng = np.random.default_rng(7)
best = (np.inf, None)
for attempt in range(40):
    n = rng.normal(size=(16, 3))
    n /= np.linalg.norm(n, axis=1)[:, None]
    for _ in range(4000):
        for i in range(16):
            h = np.zeros(3)
            for (j, M) in Jmats[i]:
                h += M @ n[j]
            nrm = np.linalg.norm(h)
            if nrm > 1e-12:
                n[i] = -h / nrm
    E = 0.5 * sum(n[i] @ M @ n[j] for i in range(16) for (j, M) in Jmats[i]) \
        * S * S
    if E < best[0] - 1e-10:
        best = (E, n.copy())
E0, n_gs = best
# orient the net moment along +z (degenerate domains)
mz = n_gs.sum(axis=0)
ax = int(np.argmax(np.abs(mz)))
perm = {0: [1, 2, 0], 1: [2, 0, 1], 2: [0, 1, 2]}[ax]
n_gs = n_gs[:, perm] * np.sign(mz[ax])
print(f"classical E/site = {E0 / 16:.6f} meV, net m/site = "
      f"{np.linalg.norm(n_gs.sum(axis=0)) / 16:.4f}")

header = open(__file__).read().split('"""')[1]
cfg = {
    "parameter_order": [],
    "parameters": {},
    "crystal_structure": {
        "lattice_vectors": [[a, 0, 0], [0, a, 0], [0, 0, a]],
        "atoms_uc": [{"label": f"Yb{i}", "pos": [float(x) for x in frac[i]],
                      "spin_S": S, "ion": "Cu2+"} for i in range(16)],
    },
    # single symmetry-propagated reference matrix (the tutorial's setmatrix):
    # the engine detects Fd-3m and expands to all 96 ordered NN bonds
    "interactions": {
        "symmetry_rules": [{
            "type": "interaction_matrix",
            "ref_pair": ["Yb0", "Yb4"],
            "offset": [0, 0, 0],
            "value": [[float(x) for x in row] for row in J_ref],
        }],
    },
    "magnetic_structure": {
        "enabled": True,
        "type": "pattern",
        "pattern_type": "generic",
        "directions": [[float(x) for x in v] for v in n_gs],
    },
    "tasks": {"minimization": False, "dispersion": True},
    "q_path": {
        "A": [-0.5, -0.5, -0.5],
        "B": [2.0, 2.0, 2.0],
        "path": ["A", "B"],
        "points_per_segment": 250,
    },
    "plotting": {
        "save_plot": True,
        "show_plot": False,
        "plot_structure": False,
        "energy_limits_disp": [0, 1.0],
        "disp_plot_filename": "SW20_disp.png",
        "disp_title": "SW20 Yb2Ti2O7 (B=0): (-1/2,-1/2,-1/2) -> (2,2,2)",
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
print(f"wrote {out}: 16 sites, {len(interactions)} interaction matrices")
print("ground-state directions:")
print(np.round(n_gs, 4))
