#!/usr/bin/env python
"""Generate and validate examples/CCSF/config_ccsf_symmetry.yaml.

The legacy example (config_ccsf.yaml) lists all 6 Cu atoms and all 104
interaction entries explicitly in an orthorhombic "kagome-frame" cell.
This script rebuilds the same spin model in the true crystallographic
cell -- the 10 K P2_1/n FullProf refinement expressed in the standard
P2_1/c (b1) setting that MagCalcConfigBuilder loads for space group 14 --
so that the web app (pyMagCalc Studio, ./start_magcalc.sh) can generate
all spin positions from two Wyckoff sites and all interactions from a
handful of symmetry rules.

Steps
  1. Expand Cu1/Cu2 Wyckoff sites with the SG-14 operators (exactly the
     code path the GUI server uses in _hydrate_builder / /expand-config).
  2. Find the proper rotation M + translation mapping the legacy
     orthorhombic frame onto the new monoclinic frame (all 6 atoms match
     to < 0.002 A).
  3. Map every legacy directed bond into the new frame; derive a minimal
     set of symmetry rules (one reference bond per orbit) that reproduces
     the legacy interaction list exactly, incl. DM vectors (axial
     transform D' = det(M) M D) and the 120-degree ground state.
  4. Write config_ccsf_symmetry.yaml (web-app importable).
  5. Validate: symmetry-expand the rules and compare every generated
     Heisenberg/DM entry against the mapped legacy model numerically.
     With --run, also run both dispersions and compare energies.

Usage:  python examples/CCSF/make_symmetry_config.py [--run]
"""
import os
import sys
import argparse
from itertools import product

import numpy as np
import sympy as sp
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from magcalc.config_builder import MagCalcConfigBuilder  # noqa: E402
import spglib  # noqa: E402
from ase.data import atomic_numbers  # noqa: E402

OLD_CONFIG = os.path.join(HERE, "config_ccsf.yaml")
OUT_CONFIG = os.path.join(HERE, "config_ccsf_symmetry.yaml")

# 10 K P2_1/n FullProf refinement (SuperHRPD), re-expressed in the
# standard P2_1/c (b1, Hall -P 2ybc) setting via
#   a' = a, b' = -b, c' = -(a+c)   =>   x' = x - z, y' = -y, z' = -z
CELL = dict(a=7.88873, b=7.08597, c=12.30511, alpha=90.0, beta=121.72456, gamma=90.0)
WYCKOFF = [
    ("Cu1", [0.0, 0.0, 0.5]),          # P2_1/n (1/2, 0, 1/2)
    ("Cu2", [0.01129, 0.27045, 0.25761]),  # P2_1/n (0.24632, 0.27045, 0.25761)
]


def make_builder():
    """Reproduce the GUI server's _hydrate_builder for this model."""
    b = MagCalcConfigBuilder()
    b.set_lattice(space_group=14, **CELL)
    for label, pos in WYCKOFF:
        b.add_wyckoff_atom(label=label, pos=pos, spin=0.5, ion="Cu2+")
    positions = [a["pos"] for a in b.atoms_uc]
    numbers = [atomic_numbers["Cu"]] * len(positions)
    ds = spglib.get_symmetry_dataset((b.lattice_vectors, positions, numbers),
                                     symprec=1e-3)
    b.set_symmetry_ops(ds.rotations, ds.translations)
    return b


def find_frame_map(builder, old):
    """Proper rotation M and translation T with r_new = M r_old + T."""
    L_new = np.array(builder.lattice_vectors, float)
    L_old = np.array(old["crystal_structure"]["lattice_vectors"], float)
    new_atoms = [(a["label"], np.array(a["pos"], float)) for a in builder.atoms_uc]
    old_atoms = [(a["label"], np.array(a["pos"], float))
                 for a in old["crystal_structure"]["atoms_uc"]]
    cart_old = {l: p @ L_old for l, p in old_atoms}

    e1 = L_new[1] / np.linalg.norm(L_new[1])   # kagome plane: span(b', c')
    e2 = L_new[2] / np.linalg.norm(L_new[2])
    n = np.cross(e1, e2)
    inv_L = np.linalg.inv(L_new)

    def try_map(M, T):
        pairs = {}
        for lbl_o, _ in old_atoms:
            f = ((M @ cart_old[lbl_o] + T) @ inv_L) % 1.0
            best, bd = None, 1e9
            for lbl_n, p_n in new_atoms:
                d = np.abs(f - p_n)
                d = np.minimum(d, 1 - d)
                dist = np.linalg.norm(d @ L_new)
                if dist < bd:
                    bd, best = dist, lbl_n
            if bd > 0.05 or lbl_o[:3] != best[:3]:
                return None
            pairs[lbl_o] = best
        return pairs if len(set(pairs.values())) == len(pairs) else None

    for s1, s2, s3 in product([1, -1], repeat=3):
        M = np.column_stack([s1 * e1, s2 * e2, s3 * n])
        if np.linalg.det(M) < 0:      # keep the handedness (chirality!)
            continue
        for lbl_n, p_n in new_atoms:
            if not lbl_n.startswith("Cu1"):
                continue
            for extra in product([0, 1], repeat=3):
                T = (p_n + np.array(extra)) @ L_new - M @ cart_old["Cu1_1"]
                pairs = try_map(M, T)
                if pairs:
                    return M, T, pairs
    raise RuntimeError("No proper-rotation frame map found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="also run legacy & symmetry dispersions and compare")
    args = ap.parse_args()

    old = yaml.safe_load(open(OLD_CONFIG))
    builder = make_builder()
    L_new = np.array(builder.lattice_vectors, float)
    L_old = np.array(old["crystal_structure"]["lattice_vectors"], float)
    new_atoms = [(a["label"], np.array(a["pos"], float)) for a in builder.atoms_uc]
    new_pos = dict(new_atoms)
    old_pos = {a["label"]: np.array(a["pos"], float)
               for a in old["crystal_structure"]["atoms_uc"]}

    M, T, amap = find_frame_map(builder, old)
    print("Atom map (legacy -> symmetry-generated):", amap)

    inv_L = np.linalg.inv(L_new)
    cart_new = {l: p @ L_new for l, p in new_atoms}

    def map_bond(lbl_i, lbl_j, off):
        """Legacy directed bond -> (new_i, new_j, new_offset)."""
        v_old = (old_pos[lbl_j] + np.array(off) - old_pos[lbl_i]) @ L_old
        k = amap[lbl_i]
        f_end = (cart_new[k] + M @ v_old) @ inv_L
        for lbl_n, p_n in new_atoms:
            r = np.round(f_end - p_n)
            if np.linalg.norm((f_end - p_n - r) @ L_new) < 0.05:
                return (k, lbl_n, tuple(int(x) for x in r))
        raise RuntimeError(f"no image for {lbl_i}->{lbl_j}+{off}")

    def bond_dist(key):
        k, l, off = key
        return float(np.linalg.norm((new_pos[l] + np.array(off) - new_pos[k]) @ L_new))

    # ---- map the whole legacy interaction list ----------------------
    heis_old, dm_old = {}, {}
    for it in old["interactions"]:
        key = map_bond(it["pair"][0], it["pair"][1], it["rij_offset"])
        (heis_old if it["type"] == "heisenberg" else dm_old)[key] = it["value"]
    print(f"Mapped {len(heis_old)} Heisenberg + {len(dm_old)} DM directed bonds")

    # ---- DM vectors into the new frame (axial: det(M)=+1) -----------
    def dm_to_new(val):
        D = sp.Matrix([sp.sympify(v) for v in val])
        return sp.expand(sp.Matrix(M) * D)

    def dm_string(expr_vec, jname):
        Dy, Dz = sp.symbols("Dy Dz")
        J = sp.Symbol(jname)
        out = []
        for comp in expr_vec:
            c = sp.expand(comp / J)
            cy, cz = float(c.coeff(Dy)), float(c.coeff(Dz))
            terms = ""
            if abs(cy) > 1e-12:
                terms += f"{cy:+.8f}*Dy"
            if abs(cz) > 1e-12:
                terms += f"{cz:+.8f}*Dz"
            out.append(f"{jname}*({terms})" if terms else "0")
        return out

    # ---- greedy minimal rule set -------------------------------------
    def covered_keys(kind):
        lst = builder.config["interactions"][kind]
        return {(e["pair"][0], e["pair"][1], tuple(e["rij_offset"])) for e in lst}

    heis_rules, dm_rules = [], []
    for key in sorted(heis_old, key=lambda k: (round(bond_dist(k), 4), k)):
        if key in covered_keys("heisenberg"):
            continue
        d = bond_dist(key)
        builder.add_symmetry_interaction(
            type="heisenberg", ref_pair=[key[0], key[1]], value=heis_old[key],
            distance=round(d, 4), offset=list(key[2]))
        heis_rules.append(dict(type="heisenberg", ref_pair=[key[0], key[1]],
                               offset=list(key[2]), distance=round(d, 4),
                               value=heis_old[key]))
    for key in sorted(dm_old, key=lambda k: (round(bond_dist(k), 4), k)):
        if key in covered_keys("dm_interaction"):
            continue
        d = bond_dist(key)
        jname = heis_old[key]           # DM lives on the J11/J12/J13 bonds
        val = dm_string(dm_to_new(dm_old[key]), jname)
        builder.add_symmetry_interaction(
            type="dm", ref_pair=[key[0], key[1]], value=val,
            distance=round(d, 4), offset=list(key[2]))
        dm_rules.append(dict(type="dm", ref_pair=[key[0], key[1]],
                             offset=list(key[2]), distance=round(d, 4),
                             value=val))
    print(f"Derived {len(heis_rules)} Heisenberg + {len(dm_rules)} DM symmetry rules")

    # ---- validation: expanded set == mapped legacy set ---------------
    params = {k: float(v) for k, v in old["parameters"].items()}
    subs = {sp.Symbol(k): v for k, v in params.items()}
    errors = []

    new_heis = {(e["pair"][0], e["pair"][1], tuple(e["rij_offset"])): e["value"]
                for e in builder.config["interactions"]["heisenberg"]}
    new_dm = {(e["pair"][0], e["pair"][1], tuple(e["rij_offset"])): e["value"]
              for e in builder.config["interactions"]["dm_interaction"]}

    if set(new_heis) != set(heis_old):
        errors.append(f"Heisenberg bond sets differ: "
                      f"missing={set(heis_old) - set(new_heis)} "
                      f"extra={set(new_heis) - set(heis_old)}")
    else:
        for key, v in heis_old.items():
            a = float(sp.sympify(v).subs(subs))
            b = float(sp.sympify(new_heis[key]).subs(subs))
            if abs(a - b) > 1e-9:
                errors.append(f"Heisenberg value mismatch on {key}: {v} vs {new_heis[key]}")

    if set(new_dm) != set(dm_old):
        errors.append(f"DM bond sets differ: missing={set(dm_old) - set(new_dm)} "
                      f"extra={set(new_dm) - set(dm_old)}")
    else:
        for key, v in dm_old.items():
            want = np.array([float(sp.sympify(c).subs(subs)) for c in dm_to_new(v)])
            got = np.array([float(sp.sympify(c).subs(subs)) for c in new_dm[key]])
            if np.max(np.abs(want - got)) > 1e-6:
                errors.append(f"DM value mismatch on {key}: want {want}, got {got}")

    if errors:
        for e in errors:
            print("ERROR:", e)
        sys.exit(1)
    print("VALIDATION OK: all", len(heis_old), "Heisenberg and", len(dm_old),
          "DM directed bonds reproduced exactly by the symmetry rules.")

    # ---- ground state (axial vectors, det(M)=+1) ---------------------
    inv_amap = {v: k for k, v in amap.items()}
    old_dirs = {old["crystal_structure"]["atoms_uc"][i]["label"]: np.array(d, float)
                for i, d in enumerate(old["magnetic_structure"]["directions"])}
    directions = []
    for lbl_n, _ in new_atoms:
        S = M @ old_dirs[inv_amap[lbl_n]]
        directions.append(S / np.linalg.norm(S))

    # ---- write the web-app config ------------------------------------
    def rule_line(r):
        if isinstance(r["value"], list):
            val = "[" + ", ".join(f'"{v}"' for v in r["value"]) + "]"
        else:
            val = f'"{r["value"]}"'
        return ("  - { type: %s, ref_pair: [%s, %s], offset: [%d, %d, %d], "
                "distance: %.4f, value: %s }" %
                (r["type"], r["ref_pair"][0], r["ref_pair"][1], *r["offset"],
                 r["distance"], val))

    nn_rules = [r for r in heis_rules if r["distance"] < 4.0]
    j2_rules = [r for r in heis_rules if r["distance"] >= 4.0]
    dir_lines = "\n".join("    - [%+.8f, %+.8f, %+.8f]" % tuple(d) for d in directions)

    yaml_text = f"""# =====================================================================
# Cs2Cu3SnF12 -- distorted spin-1/2 kagome antiferromagnet
# Symmetry-driven model for the pyMagCalc web app (pyMagCalc Studio):
#   ./start_magcalc.sh  ->  Import YAML  ->  this file
# The GUI expands the Wyckoff sites and symmetry rules into the full
# 6-spin / 104-bond model (via /expand-config) and runs LSWT. For the
# explicit, CLI-runnable version of the same model use config_ccsf.yaml.
#
# Crystal structure: 10 K P2_1/n FullProf refinement (SuperHRPD data,
# a=7.88873, b=7.08597, c=10.5622, beta=97.7168), re-expressed in the
# standard P2_1/c (b1) setting used by the builder for space group 14:
#   a' = a, b' = -b, c' = -(a+c)  =>  x' = x - z, y' = -y, z' = -z
# Only the two inequivalent Cu sites are given; the space-group symmetry
# generates all 6 spins of the unit cell and propagates every exchange
# and Dzyaloshinskii-Moriya interaction from one reference bond per
# orbit (physics identical to the legacy examples/CCSF/config_ccsf.yaml;
# regenerate/verify with examples/CCSF/make_symmetry_config.py).
#
# Kagome geometry in this cell: the layers are spanned by b and c
# (b ~ 7.086 A = kagome axis, |c| ~ 12.305 A = sqrt(3) x 7.10); the
# layer normal is a* . Nearest-neighbour bonds split into three orbits
# J11 (3.522 A), J12 (3.548 A), J13 (3.586 A); the weak J2 covers the
# 2nd/3rd-neighbour shells (5.99-7.17 A) exactly as in the legacy model.
# DM vectors are given in the Cartesian frame of this cell (a || x,
# b || y, c in the xz plane) and already include the J_ij prefactor.
# Parameters: variant-B refit of the AMATERAS INS data (see README.md).
# The coplanar 120-degree, negative-vector-chirality ground state is
# supplied via magnetic_structure, so no minimization is needed.
# =====================================================================

parameter_order: [J11, J12, J13, J2, Dy, Dz]
parameters:
  J11: {params['J11']}
  J12: {params['J12']}
  J13: {params['J13']}
  J2: {params['J2']}
  Dy: {params['Dy']}
  Dz: {params['Dz']}

crystal_structure:
  lattice_parameters:
    a: {CELL['a']}
    b: {CELL['b']}
    c: {CELL['c']}
    alpha: {CELL['alpha']}
    beta: {CELL['beta']}
    gamma: {CELL['gamma']}
    space_group: 14        # P2_1/c (b1 setting, Hall -P 2ybc)
  atom_mode: symmetry
  magnetic_elements: [Cu]
  # Asymmetric unit only -- SG 14 expands these to 6 Cu sites per cell,
  # labelled Cu10, Cu11 (from Cu1) and Cu20..Cu23 (from Cu2).
  wyckoff_atoms:
    - {{ label: Cu1, pos: [{WYCKOFF[0][1][0]}, {WYCKOFF[0][1][1]}, {WYCKOFF[0][1][2]}], spin_S: 0.5, ion: Cu2+ }}
    - {{ label: Cu2, pos: [{WYCKOFF[1][1][0]}, {WYCKOFF[1][1][1]}, {WYCKOFF[1][1][2]}], spin_S: 0.5, ion: Cu2+ }}

interactions:
  # One reference bond per symmetry orbit; ref_pair uses the expanded
  # labels. The builder propagates each rule with the P2_1/c operators
  # (DM as an axial vector, D' = det(R) R D).
  symmetry_rules:
  # --- nearest-neighbour kagome bonds (one rule per orbit) ---
{chr(10).join(rule_line(r) for r in nn_rules)}
  # --- DM on the nearest-neighbour bonds ---
{chr(10).join(rule_line(r) for r in dm_rules)}
  # --- weak 2nd/3rd-neighbour coupling J2 (one rule per orbit) ---
{chr(10).join(rule_line(r) for r in j2_rules)}

magnetic_structure:
  enabled: true
  type: pattern
  pattern_type: generic   # minimized negative-chirality 120-degree state,
                          # one direction per expanded spin (Cu10..Cu23)
  directions:
{dir_lines}

tasks:
  minimization: false      # ground state supplied above
  dispersion: true
  sqw_map: true

# Same physical path as the legacy example (kagome-frame Gamma-X-M-Y-Gamma):
# in this cell the in-plane reciprocal axes are b* and c*, so
# X = (0, 1/2, 0), M = (0, 1/2, 1/2), Y = (0, 0, 1/2); the a* component
# is out-of-plane and does not disperse (layers are decoupled).
q_path:
  Gamma: [0.0, 0.0, 0.0]
  X:     [0.0, 0.5, 0.0]
  M:     [0.0, 0.5, 0.5]
  Y:     [0.0, 0.0, 0.5]
  path: ["Gamma", "X", "M", "Y", "Gamma"]
  points_per_segment: 100

plotting:
  save_plot: true
  show_plot: false
  plot_structure: true
  disp_plot_filename: CCSF_symmetry_disp.png
  sqw_plot_filename: CCSF_symmetry_sqw.png
  energy_limits_disp: [0, 16]
  energy_limits_sqw: [0, 16]
  broadening: 0.4
  broadening_width: 0.4
  energy_resolution: 0.04
  disp_title: "Cs2Cu3SnF12 P2_1/n neg-chirality LSWT dispersion (symmetry-generated)"
  sqw_title: "Cs2Cu3SnF12 S(Q,w)"

calculation:
  cache_mode: "none"

output:
  save_data: false
"""
    with open(OUT_CONFIG, "w") as f:
        f.write(yaml_text)
    print("Wrote", OUT_CONFIG)

    # sanity: the file must round-trip through YAML
    yaml.safe_load(open(OUT_CONFIG))

    if args.run:
        run_and_compare(builder, old, yaml.safe_load(open(OUT_CONFIG)))


def run_and_compare(builder, old, newdoc):
    """Run legacy and symmetry-expanded dispersions on the same physical
    q-path and compare the magnon energies."""
    import tempfile
    from magcalc.runner import run_calculation

    tmp = tempfile.mkdtemp(prefix="ccsf_check_")
    npts = 10

    # legacy, trimmed
    legacy = yaml.safe_load(open(OLD_CONFIG))
    legacy["tasks"] = {"minimization": False, "dispersion": True, "sqw_map": False}
    legacy["plotting"] = {"save_plot": False, "show_plot": False, "plot_structure": False}
    legacy["q_path"]["points_per_segment"] = npts
    legacy["calculation"] = {"cache_mode": "none"}
    legacy["output"] = {"save_data": True,
                        "disp_data_filename": os.path.join(tmp, "legacy.npz")}

    # symmetry-expanded, mirroring /expand-config output
    final_inters = []
    for kind in ["heisenberg", "dm_interaction"]:
        final_inters.extend(builder.config["interactions"][kind])
    atoms = [{"label": a["label"], "pos": [float(x) for x in a["pos"]],
              "spin_S": a["spin_S"], "ion": a.get("ion")} for a in builder.atoms_uc]
    newcfg = {
        "parameters": {k: float(v) for k, v in old["parameters"].items()},
        "crystal_structure": {
            "lattice_parameters": dict(CELL, space_group=14),
            "atoms_uc": atoms,
            "magnetic_elements": ["Cu"],
            "dimensionality": 3,
        },
        "interactions": final_inters,
        "magnetic_structure": newdoc["magnetic_structure"],
        "tasks": {"minimization": False, "dispersion": True, "sqw_map": False},
        "q_path": dict(newdoc["q_path"], points_per_segment=npts),
        "plotting": {"save_plot": False, "show_plot": False, "plot_structure": False},
        "calculation": {"cache_mode": "none"},
        "output": {"save_data": True,
                   "disp_data_filename": os.path.join(tmp, "symmetry.npz")},
    }

    p_old = os.path.join(tmp, "legacy.yaml")
    p_new = os.path.join(tmp, "symmetry.yaml")
    yaml.safe_dump(legacy, open(p_old, "w"), sort_keys=False)
    yaml.safe_dump(newcfg, open(p_new, "w"), sort_keys=False)

    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        print("\nRunning legacy dispersion ...")
        run_calculation(p_old)
        print("Running symmetry-generated dispersion ...")
        run_calculation(p_new)
    finally:
        os.chdir(cwd)

    e_old = np.sort(np.load(os.path.join(tmp, "legacy.npz"))["energies"], axis=1)
    e_new = np.sort(np.load(os.path.join(tmp, "symmetry.npz"))["energies"], axis=1)
    diff = np.abs(e_old - e_new)
    print(f"\nDispersion comparison over {e_old.shape[0]} q-points, "
          f"{e_old.shape[1]} branches:")
    print(f"  max |dE| = {diff.max():.6f} meV,  mean |dE| = {diff.mean():.6f} meV")
    print(f"  legacy band top {e_old.max():.4f} meV, symmetry {e_new.max():.4f} meV")
    if diff.max() < 0.05:
        print("DISPERSION MATCH OK")
    else:
        print("WARNING: dispersion deviates more than 0.05 meV")


if __name__ == "__main__":
    main()
