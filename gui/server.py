import os
import io
import yaml
import numpy as np
import spglib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ase.io import read
from typing import List, Dict, Any

import sys
# Add parent directory to sys.path to find magcalc package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from magcalc.config_builder import MagCalcConfigBuilder
except ImportError:
    # If not found (e.g. running in a way where it's not available), we'll handle it in the endpoint
    MagCalcConfigBuilder = None

app = FastAPI(title="MagCalc Designer Backend")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/parse-cif")
async def parse_cif(file: UploadFile = File(...)):
    """
    Parse an uploaded CIF file and extract lattice parameters + Wyckoff positions.
    """
    try:
        content = await file.read()
        # ASE can read from a file-like object
        f = io.StringIO(content.decode('utf-8'))
        atoms = read(f, format="cif")
        
        # Prepare spglib cell
        cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
        dataset = spglib.get_symmetry_dataset(cell)
        
        if not dataset:
            raise HTTPException(status_code=400, detail="Could not determine symmetry for this CIF.")
            
        unique_indices = np.unique(dataset.equivalent_atoms)
        
        lattice_params = atoms.get_cell().cellpar()
        
        basis_atoms = []
        for idx in unique_indices:
            atom = atoms[int(idx)]
            basis_atoms.append({
                "label": atom.symbol,
                "pos": atom.scaled_position.tolist(),
                "spin_S": 0.5 # Default fallback
            })
            
        return {
            "lattice": {
                "a": float(lattice_params[0]),
                "b": float(lattice_params[1]),
                "c": float(lattice_params[2]),
                "alpha": float(lattice_params[3]),
                "beta": float(lattice_params[4]),
                "gamma": float(lattice_params[5]),
                "space_group": int(dataset.number)
            },
            "wyckoff_atoms": basis_atoms,
            "hall_number": int(dataset.hall_number),
            "international": dataset.international
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-config")
async def save_config(config: Dict[str, Any]):
    """
    Save the designer state to a YAML file in the workspace.
    """
    try:
        filename = config.get("filename", "config_pure.yaml")
        # Sanitize filename to be relative to project root
        filename = filename.lstrip("/") 
        
        # Save to the project root (parent directory of gui/)
        gui_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(gui_dir)
        save_path = os.path.join(project_root, filename)
        
        print(f"Saving config to: {save_path}")
        
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(config["data"], f, sort_keys=False)
            
        return {"message": f"Saved successfully to project root as {filename}", "path": save_path}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-neighbors")
async def get_neighbors(config: Dict[str, Any]):
    """
    Find unique neighbor distances for the given crystal structure.
    """
    if MagCalcConfigBuilder is None:
        raise HTTPException(status_code=500, detail="MagCalcConfigBuilder not found on server.")
        
    try:
        builder = MagCalcConfigBuilder()
        data = config.get("data", {})
        
        # 1. Lattice
        lattice = data.get("crystal_structure", {}).get("lattice_parameters", {})
        builder.set_lattice(
            a=lattice.get("a", 1.0),
            b=lattice.get("b", 1.0),
            c=lattice.get("c", 1.0),
            alpha=lattice.get("alpha", 90),
            beta=lattice.get("beta", 90),
            gamma=lattice.get("gamma", 90),
            space_group=lattice.get("space_group")
        )
        
        # 2. Wyckoff Atoms
        wyckoff_atoms = data.get("crystal_structure", {}).get("wyckoff_atoms", [])
        for atom in wyckoff_atoms:
            builder.add_wyckoff_atom(
                label=atom.get("label", "Atom"),
                pos=atom.get("pos", [0,0,0]),
                spin=atom.get("spin_S", 0.5)
            )
            
        # 3. Calculate Distances
        labels = [a['label'] for a in builder.atoms_uc]
        positions = [np.array(a['pos']) for a in builder.atoms_uc]
        lattice_vecs = builder.lattice_vectors
        
        max_dist = 10.0
        unique_bonds = []
        
        from itertools import product
        # Search a slightly larger cube of cells
        offsets = list(product([-1, 0, 1], repeat=3))
        
        for i in range(len(positions)):
            for j in range(len(positions)):
                for off in offsets:
                    off_vec = np.array(off)
                    pos_j_img = positions[j] + off_vec
                    dist = np.linalg.norm((pos_j_img - positions[i]) @ lattice_vecs)
                    
                    if 0.1 < dist < max_dist:
                        unique_bonds.append({
                            "distance": round(float(dist), 4),
                            "ref_pair": [labels[i], labels[j]],
                            "offset": off_vec.tolist()
                        })
        
        # 4. Group by symmetry
        if not builder.symmetry_ops:
            # Fallback to simple distance grouping
            unique_shells = {}
            for b in unique_bonds:
                key = (b["distance"], tuple(sorted(b["ref_pair"])))
                if key not in unique_shells:
                    unique_shells[key] = {**b, "multiplicity": 1}
                else:
                    unique_shells[key]["multiplicity"] += 1
            sorted_shells = sorted(list(unique_shells.values()), key=lambda x: x["distance"])
        else:
            # Full Symmetry Grouping
            rots = builder.symmetry_ops['rotations']
            trans = builder.symmetry_ops['translations']
            processed_bonds = set()
            orbits = []

            def find_atom_idx(pos):
                p = pos % 1.0
                for idx, a_pos in enumerate(positions):
                    diff = np.abs(a_pos - p)
                    diff = np.minimum(diff, 1.0 - diff)
                    if np.linalg.norm(diff) < 1e-3: return idx
                return None

            # Prepare list for sorting/processing
            bond_keys = []
            for b in unique_bonds:
                bond_keys.append((
                    labels.index(b["ref_pair"][0]),
                    labels.index(b["ref_pair"][1]),
                    tuple(b["offset"]),
                    b["distance"]
                ))

            for i, j, off, dist in bond_keys:
                if (i, j, off) in processed_bonds: continue
                
                current_orbit = set()
                p_i = positions[i]
                p_j_off = positions[j] + np.array(off)
                
                for R, t in zip(rots, trans):
                    p_i_p = R @ p_i + t
                    p_j_o_p = R @ p_j_off + t
                    idx_i_p = find_atom_idx(p_i_p)
                    idx_j_p = find_atom_idx(p_j_o_p)
                    
                    if idx_i_p is not None and idx_j_p is not None:
                        off_i_p = np.round(p_i_p - positions[idx_i_p]).astype(int)
                        off_j_p = np.round(p_j_o_p - positions[idx_j_p]).astype(int)
                        final_off = tuple(off_j_p - off_i_p)
                        current_orbit.add((idx_i_p, idx_j_p, final_off))
                        current_orbit.add((idx_j_p, idx_i_p, tuple(-np.array(final_off))))
                
                processed_bonds.update(current_orbit)
                orbits.append({
                    "distance": dist,
                    "ref_pair": [labels[i], labels[j]],
                    "offset": list(off),
                    "multiplicity": len(current_orbit) // 2
                })
            sorted_shells = sorted(orbits, key=lambda x: x["distance"])

        # 5. Assign 1st, 2nd, 3rd labels based on distance
        unique_distances = sorted(list(set(s["distance"] for s in sorted_shells)))
        dist_to_rank = {d: i + 1 for i, d in enumerate(unique_distances)}
        
        def get_rank_label(rank):
            if 10 <= rank % 100 <= 20: return f"{rank}th"
            else:
                mapping = {1: "st", 2: "nd", 3: "rd"}
                return f"{rank}{mapping.get(rank % 10, 'th')}"

        for s in sorted_shells:
            rank = dist_to_rank[s["distance"]]
            s["shell_label"] = get_rank_label(rank)
            s["rank"] = rank

        return sorted_shells[:40]
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/expand-config")
async def expand_config(config: Dict[str, Any]):
    """
    Take a symmetry-based config and return the expanded version.
    """
    if MagCalcConfigBuilder is None:
        raise HTTPException(status_code=500, detail="MagCalcConfigBuilder not found on server.")
        
    try:
        builder = MagCalcConfigBuilder()
        data = config.get("data", {})
        
        # 1. Lattice
        lattice = data.get("crystal_structure", {}).get("lattice_parameters", {})
        builder.set_lattice(
            a=lattice.get("a", 1.0),
            b=lattice.get("b", 1.0),
            c=lattice.get("c", 1.0),
            alpha=lattice.get("alpha", 90),
            beta=lattice.get("beta", 90),
            gamma=lattice.get("gamma", 90),
            space_group=lattice.get("space_group")
        )
        
        # 2. Wyckoff Atoms
        wyckoff_atoms = data.get("crystal_structure", {}).get("wyckoff_atoms", [])
        for atom in wyckoff_atoms:
            builder.add_wyckoff_atom(
                label=atom.get("label", "Atom"),
                pos=atom.get("pos", [0,0,0]),
                spin=atom.get("spin_S", 0.5)
            )
            
        # 3. Symmetry Interactions
        # The GUI provides 'symmetry_rules' in 'interactions'
        rules = data.get("interactions", {}).get("symmetry_rules", [])
        for rule in rules:
            builder.add_symmetry_interaction(
                type=rule.get("type", "heisenberg"),
                ref_pair=rule.get("ref_pair"),
                distance=rule.get("distance"),
                value=rule.get("value")
            )
            
        # 4. Global Parameters & Tasks
        builder.config["parameters"] = data.get("parameters", {})
        builder.config["tasks"] = data.get("tasks", {})
        
        # 5. Expand & Return
        # We use builder.save logic but captured in a dict
        builder._expand_heisenberg_rules()
        builder._expand_anisotropic_exchange_rules()
        
        # Build the final atoms list (atoms_uc)
        config_atoms = []
        for a in builder.atoms_uc:
            config_atoms.append({
                "label": a["label"],
                "pos": [float(x) for x in a["pos"]],
                "spin_S": a["spin_S"]
            })
        builder.config["crystal_structure"]["atoms_uc"] = config_atoms
        
        # Derive magnetic_elements
        species = sorted(list(set(a["species"] for a in builder.atoms_uc if a.get("species"))))
        builder.config["crystal_structure"]["magnetic_elements"] = species
        
        # Flatten interactions into a single list for the runner if desired, 
        # or keep them grouped. config_acvo_pure uses a flat list under 'interactions'.
        # However, MagCalcConfigRunner usually expects a flat list.
        # Let's flatten to match config_acvo_pure.yaml
        final_inters = []
        final_inters.extend(builder.config["interactions"].get("heisenberg", []))
        final_inters.extend(builder.config["interactions"].get("dm_interaction", []))
        final_inters.extend(builder.config["interactions"].get("anisotropic_exchange", []))
        
        expanded_config = {
            "crystal_structure": builder.config["crystal_structure"],
            "interactions": final_inters,
            "parameters": builder.config["parameters"],
            "tasks": builder.config["tasks"],
            "q_path": data.get("q_path", {}),
            "minimization": data.get("minimization", {}),
            "plotting": data.get("plotting", {})
        }
        
        return expanded_config
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
