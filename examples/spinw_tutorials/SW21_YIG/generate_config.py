"""Generate config.yaml for SW21 (yttrium iron garnet, SpinW tutorial 21).

YIG, Ia-3d (#230), a = 12.376 A (10 K): Fe3+ on 16a (0,0,0) (octahedral)
and 24d (3/8,0,1/4) (tetrahedral). Like the tutorial (spinw.newcell),
the 20-atom PRIMITIVE BCC cell is used, so the spectrum shows the 20
physical bands along N-G-H (the conventional cube would fold H onto G).
Ferrimagnet: the 12 d spins point +z, the 8 a spins -z. Spins are
normalized to S=1 and the exchanges rescaled by S_cl = sqrt(S0(S0+1)),
S0=5/2, following the paper J. Barker & G. E. W. Bauer, PRL 117, 217201
(2016). Exchange values (converted J -> THz, then /S_cl), positive = AFM:

    Jad = 9.60e-21 J -> 14.4882 THz -> 4.8979  (a-d NN, 3.46 A, bond 1)
    Jdd = 3.24e-21 J ->  4.8898 THz -> 1.6531  (d-d NN, 3.79 A, bond 2)
    Jaa = 0.92e-21 J ->  1.3884 THz -> 0.4694  (a-a NN, 5.36 A, bond 3)

All energies in THz (pyMagCalc is unit-agnostic).

The a-d and d-d shells are distance symmetry rules. The a-a shell CANNOT
be a plain distance rule: the a-a bonds at 5.359 A split into two
symmetry-INEQUIVALENT families told apart by their midpoints (48g vs 16b
Wyckoff) -- the tutorial's own teaching point -- and quickham puts Jaa
only on the 48g family (SpinW bond 3), leaving the 16b family (bond 4)
uncoupled. Jaa therefore enters as a ref_pair symmetry rule: ONE
reference 48g bond that the engine propagates over the detected Ia-3d
operations, which picks out exactly that family (verified band-by-band
identical to the explicit 48-bond list; the 16b variant was tested and
clearly disagrees with the published figure).

The tutorial's tiny orienting field (0.01 meV) only fixes the overall
moment direction and is omitted. Path N-G-H in primitive RLU:
N = (1/2,0,0), H = (-1/2,1/2,1/2) (the conventional (1/2,1/2,0) and
(0,0,1) mapped with the tutorial's transformation T).
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _compact_yaml import dump as yaml_compact_dump
from pymatgen.core import Structure, Lattice

a = 12.376
Jad, Jdd, Jaa = 4.8979, 1.6531, 0.4694   # THz (see docstring)

lat = Lattice.cubic(a)
s = Structure.from_spacegroup("Ia-3d", lat, ["Fe", "Fe"],
                              [[0, 0, 0], [3 / 8, 0, 1 / 4]])
frac_conv = np.array([site.frac_coords % 1.0 for site in s])
assert len(frac_conv) == 40
kind_conv = ["a"] * 16 + ["d"] * 24

# 16b midpoints orbit (for classifying the two a-a bond families),
# in conventional fractional coordinates
s16b = Structure.from_spacegroup("Ia-3d", lat, ["X"], [[1 / 8, 1 / 8, 1 / 8]])
mid16b = np.array([site.frac_coords % 1.0 for site in s16b])
assert len(mid16b) == 16

def is_16b(m):
    d = (mid16b - m + 0.5) % 1.0 - 0.5
    return np.any(np.all(np.abs(d) < 1e-6, axis=1))

# --- primitive BCC cell (tutorial's newcell basis) ---------------------
pBV = np.array([[0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, -0.5, 0.5]])
P = a * pBV                       # primitive lattice vectors (rows, cart)
Pinv = np.linalg.inv(P)

cart = frac_conv @ (a * np.eye(3))
fp_all = (cart @ Pinv) % 1.0
frac, site_kind = [], []
for f, k in zip(fp_all, kind_conv):
    if not any(np.all(np.abs(((f - g + 0.5) % 1.0) - 0.5) < 1e-6) for g in frac):
        frac.append(f)
        site_kind.append(k)
frac = np.array(frac)
assert len(frac) == 20 and site_kind.count("a") == 8, (len(frac), site_kind)

# enumerate a-a bonds at 5.359 A (primitive frame) and keep the
# 48g-midpoint family
d_aa = a * np.sqrt(3) / 4
offsets = [np.array(o) for o in
           [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]]
a_idx = [n for n, k in enumerate(site_kind) if k == "a"]
aa_bonds = []
for i in a_idx:
    for j in a_idx:
        for off in offsets:
            if i == j and not off.any():
                continue
            dvec_cart = (frac[j] + off - frac[i]) @ P
            if abs(np.linalg.norm(dvec_cart) - d_aa) < 1e-3:
                mid_cart = (frac[i] @ P + frac[j] @ P + off @ P) / 2
                mid_conv = (mid_cart / a) % 1.0
                if not is_16b(mid_conv):
                    aa_bonds.append((i, j, [int(x) for x in off]))
assert len(aa_bonds) == 48, len(aa_bonds)   # 24 unordered x 2

labels = [f"Fe{'A' if k == 'a' else 'D'}{n}" for n, k in enumerate(site_kind)]
atoms = [{"label": labels[n], "pos": [float(x) for x in frac[n]],
          "spin_S": 1.0, "ion": "Fe3+"} for n in range(20)]
directions = [[0.0, 0.0, -1.0] if k == "a" else [0.0, 0.0, 1.0]
              for k in site_kind]

ref_i, ref_j, ref_off = aa_bonds[0]     # one 48g-family reference bond
interactions = {
    "symmetry_rules": [
        {"type": "heisenberg", "distance": float(round(a * np.sqrt(5) / 8, 5)),
         "value": "Jad"},                                    # 3.4595 A
        {"type": "heisenberg", "distance": float(round(a * np.sqrt(6) / 8, 5)),
         "value": "Jdd"},                                    # 3.7893 A
        # Jaa: reference 48g bond, propagated by the detected Ia-3d ops
        {"type": "heisenberg", "ref_pair": [labels[ref_i], labels[ref_j]],
         "offset": ref_off, "value": "Jaa"},
    ],
}

header = open(__file__).read().split('"""')[1]
cfg = {
    "parameter_order": ["Jad", "Jdd", "Jaa"],
    "parameters": {"Jad": Jad, "Jdd": Jdd, "Jaa": Jaa},
    "crystal_structure": {
        "lattice_vectors": [[float(x) for x in row] for row in P],
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
    "q_path": {
        "N": [0.5, 0.0, 0.0],
        "G": [0.0, 0.0, 0.0],
        "H": [-0.5, 0.5, 0.5],
        "path": ["N", "G", "H"],
        "points_per_segment": 100,
    },
    "plotting": {
        "save_plot": True,
        "show_plot": False,
        "plot_structure": False,
        "energy_limits_disp": [0, 28],
        "disp_plot_filename": "SW21_disp.png",
        "disp_title": "SW21 YIG: N-G-H (energies in THz)",
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
print(f"wrote {out}: {len(frac)} sites, 2 distance rules + 1 ref_pair rule "
      f"(reference 48g bond {labels[ref_i]}->{labels[ref_j]} {ref_off})")
