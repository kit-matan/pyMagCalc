import os
import io
import yaml
import numpy as np
import spglib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from ase.io import read
from typing import List, Dict, Any
import asyncio
import logging
import uuid
from starlette.websockets import WebSocket, WebSocketDisconnect

import sys
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for thread safety
# Add parent directory to sys.path to find magcalc package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from magcalc.config_builder import MagCalcConfigBuilder
except ImportError:
    # If not found (e.g. running in a way where it's not available), we'll handle it in the endpoint
    MagCalcConfigBuilder = None

from fastapi.staticfiles import StaticFiles
from magcalc.runner import run_calculation

app = FastAPI(title="MagCalc Designer Backend")

# Mount the project root to serve generated files (e.g. plots)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
app.mount("/files", StaticFiles(directory=project_root), name="files")

# --- Log Streaming Setup ---
log_queue = asyncio.Queue()

class BroadcastingLogHandler(logging.Handler):
    """
    Pushes log records to an asyncio Queue for WebSocket broadcasting.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            # Schedule the put operation on the event loop
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(log_queue.put_nowait, msg)
            except RuntimeError:
                # Loop might not be running yet
                pass
        except Exception:
            self.handleError(record)

@app.on_event("startup")
async def startup_event():
    # Setup global logger to capture magcalc output
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Add our broadcasting handler
    handler = BroadcastingLogHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Attach to root logger
    logger.addHandler(handler)
    
    # Explicitly attach to magcalc logger and ensure level is INFO
    mc_logger = logging.getLogger("magcalc")
    mc_logger.setLevel(logging.INFO)
    mc_logger.addHandler(handler)
    mc_logger.propagate = False # Prevent double logging if root also handles it, but here we want to be sure logging works.
    # Actually, keep propagate=True mostly, but if duplicate, we set to False. 
    # Let's just set level and attach handler. Use propagate=False to avoid root capturing it IF root has handler.
    # But wait, root HAS handler. So we will get duplicates if we attach to both and propagate is True.
    # But currently we get NOTHING. So propagate might be False by default or blocked?
    # Safest: Attach to magcalc, set propagate=False.
    mc_logger.propagate = False
    
    # Also attach to uvicorn loggers to capture access logs if desired, or ensure magcalc logs propagated
    # logging.getLogger("uvicorn").addHandler(handler)

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    # Send a welcome message to confirm connection
    await websocket.send_text("Connected to Log Stream. Waiting for activity...")
    try:
        while True:
            # Wait for next log entry
            log_entry = await log_queue.get()
            await websocket.send_text(log_entry)
            log_queue.task_done()
    except WebSocketDisconnect:
        print("Log client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

# ... (rest of code)

@app.post("/run-calculation")
async def trigger_calculation(config: Dict[str, Any]):
    """
    Save config to a temporary run file and trigger magcalc.runner.run_calculation.
    """
    try:
        # 0. Expand the Config (Crucial step: generates atoms_uc, bonds, etc.)
        # We can reuse the logic from expand_config endpoint
        expanded_data = await expand_config(config)
        
        # 1. Save Config
        gui_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(gui_dir)
        run_config_path = os.path.join(project_root, "config_gui_run.yaml")
        
        with open(run_config_path, 'w') as f:
            yaml.dump(expanded_data, f, sort_keys=False)
            
        logging.getLogger().info(f"Starting calculation with config: {run_config_path}")
        
        # 2. Run Calculation: Non-blocking execution
        # This function is blocking, so we run it in a thread pool to keep the loop free for WS
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, run_calculation, run_config_path)
        
        # 3. Check for outputs
        # We expect disp_plot.png and sqw_plot.png in project root if plotting was enabled
        results = {
            "message": "Calculation completed successfully.",
            "plots": []
        }
        
        if os.path.exists(os.path.join(project_root, "disp_plot.png")):
            results["plots"].append("/files/disp_plot.png")
            
        if os.path.exists(os.path.join(project_root, "sqw_plot.png")):
            results["plots"].append("/files/sqw_plot.png")
            
        return results

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
        
        # 2. Basis Atoms
        struct_data = data.get("crystal_structure", {})
        wyckoff_atoms = struct_data.get("wyckoff_atoms", [])
        atom_mode = struct_data.get("atom_mode", "symmetry")

        if atom_mode == "explicit":
            # Treat Wyckoff atoms as the full unit cell directly
            builder.atoms_uc = []
            for atom in wyckoff_atoms:
                builder.atoms_uc.append({
                    "label": atom.get("label", "Atom"),
                    "pos": atom.get("pos", [0, 0, 0]),
                    "spin_S": atom.get("spin_S", 0.5)
                })
        else:
            # Standard Wyckoff expansion
            for atom in wyckoff_atoms:
                builder.add_wyckoff_atom(
                    label=atom.get("label", "Atom"),
                    pos=atom.get("pos", [0, 0, 0]),
                    spin=atom.get("spin_S", 0.5)
                )
            
        # 3. Calculate Distances
        labels = [a['label'] for a in builder.atoms_uc]
        positions = [np.array(a['pos']) for a in builder.atoms_uc]
        lattice_vecs = builder.lattice_vectors
        
        dimensionality = data.get("crystal_structure", {}).get("dimensionality", "3D")
        
        max_dist = 10.0
        candidate_bonds = []
        
        from itertools import product
        # Search a slightly larger cube of cells
        if dimensionality == "2D":
            offsets = [ (u, v, 0) for u, v in product([-1, 0, 1], repeat=2) ]
        else:
            offsets = list(product([-1, 0, 1], repeat=3))
        
        for i in range(len(positions)):
            for j in range(len(positions)):
                for off in offsets:
                    off_vec = np.array(off)
                    pos_j_img = positions[j] + off_vec
                    dist = np.linalg.norm((pos_j_img - positions[i]) @ lattice_vecs)
                    
                    if 0.1 < dist < max_dist:
                        candidate_bonds.append({
                            "distance": float(dist),
                            "ref_pair": [labels[i], labels[j]],
                            "offset": off_vec.tolist(),
                            "offset_mag": float(np.linalg.norm(off_vec)),
                            "indices": (i, j)
                        })
        
        # Sort candidates to prioritize:
        # 1. Distance (rounded to 5 decimals for shell grouping)
        # 2. Offset Magnitude (prefer zero offset)
        # 3. Atom indices (consistency)
        candidate_bonds.sort(key=lambda b: (round(b["distance"], 5), b["offset_mag"], b["indices"]))

        # 4. Group by symmetry
        unique_shells = []
        if not builder.symmetry_ops:
            # Fallback to simple distance grouping
            processed_keys = set()
            for b in candidate_bonds:
                # Use sorted pair and rounded distance as shell key
                key = (round(b["distance"], 5), tuple(sorted(b["ref_pair"])))
                if key not in processed_keys:
                    unique_shells.append({**b, "multiplicity": 0}) # Logic below will count
                    processed_keys.add(key)
            
            # Count multiplicities
            for shell in unique_shells:
                count = 0
                for b in candidate_bonds:
                    if round(b["distance"], 5) == round(shell["distance"], 5) and \
                       tuple(sorted(b["ref_pair"])) == tuple(sorted(shell["ref_pair"])):
                        count += 1
                shell["multiplicity"] = count
            sorted_shells = unique_shells
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
            for b in candidate_bonds:
                bond_keys.append((
                    b["indices"][0],
                    b["indices"][1],
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
                    "offset": list(map(int, off)),
                    "multiplicity": len(current_orbit) // 2,
                    "equivalent_bonds": [
                        {"pair": [labels[int(b[0])], labels[int(b[1])]], "offset": list(map(int, b[2]))}
                        for b in sorted(list(current_orbit), key=lambda x: (x[0], x[1], x[2]))
                        if b[0] <= b[1] # Deduplicate list
                    ]
                })
            sorted_shells = sorted(orbits, key=lambda x: x["distance"])

        # 5. Assign 1st, 2nd, 3rd labels based on distance
        unique_distances = sorted(list(set(round(s["distance"], 5) for s in sorted_shells)))
        dist_to_rank = {d: i + 1 for i, d in enumerate(unique_distances)}
        
        def get_rank_label(rank):
            if 10 <= rank % 100 <= 20: return f"{rank}th"
            else:
                mapping = {1: "st", 2: "nd", 3: "rd"}
                return f"{rank}{mapping.get(rank % 10, 'th')}"

        for s in sorted_shells:
            rank = dist_to_rank[round(s["distance"], 5)]
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
        
        # 2. Basis Atoms
        struct_data = data.get("crystal_structure", {})
        wyckoff_atoms = struct_data.get("wyckoff_atoms", [])
        atom_mode = struct_data.get("atom_mode", "symmetry")

        if atom_mode == "explicit":
            # Treat Wyckoff atoms as the full unit cell directly
            config_atoms = []
            for a in wyckoff_atoms:
                config_atoms.append({
                    "label": a.get("label", "Atom"),
                    "pos": [float(x) for x in a.get("pos", [0, 0, 0])],
                    "spin_S": a.get("spin_S", 0.5),
                    "species": a.get("label", "Atom")
                })
            builder.config["crystal_structure"]["atoms_uc"] = config_atoms
            builder.atoms_uc = config_atoms # Ensure builder also knows them for neighbor search
        else:
            # Standard Wyckoff expansion
            for atom in wyckoff_atoms:
                builder.add_wyckoff_atom(
                    label=atom.get("label", "Atom"),
                    pos=atom.get("pos", [0, 0, 0]),
                    spin=atom.get("spin_S", 0.5)
                )

        # 2.5 Set Dimensionality
        builder.dimensionality = data.get("crystal_structure", {}).get("dimensionality", "3D")
        
        # 4. Global Parameters & Tasks
        builder.config["parameters"] = data.get("parameters", {})
        builder.config["tasks"] = data.get("tasks", {})

        # 3. Interactions
        if "list" in data.get("interactions", {}):
            # Explicit interactions provided
            final_inters = data["interactions"]["list"]
        else:
            # Symmetry-based expansion
            rules = data.get("interactions", {}).get("symmetry_rules", [])
            for rule in rules:
                try:
                    builder.add_symmetry_interaction(
                        type=rule.get("type", "heisenberg"),
                        ref_pair=rule.get("ref_pair"),
                        distance=rule.get("distance"),
                        value=rule.get("value"),
                        offset=rule.get("offset")
                    )
                except KeyError as e:
                    print(f"Warning: Skipping invalid interaction rule referencing missing atom: {e}")
                except Exception as e:
                     print(f"Warning: Skipping rule due to error: {e}")
            
            # Expand rules based on current atoms_uc (which might be explicit or symmetry-expanded)
            builder._expand_heisenberg_rules()
            builder._expand_anisotropic_exchange_rules()
            
            # Filter and flatten interactions
            final_inters = []
            final_inters.extend(builder.config["interactions"].get("heisenberg", []))
            final_inters.extend(builder.config["interactions"].get("dm_interaction", []))
            final_inters.extend(builder.config["interactions"].get("anisotropic_exchange", []))

        # Re-build atoms list for response if symmetry was used
        if atom_mode != "explicit":
            config_atoms = []
            for a in builder.atoms_uc:
                config_atoms.append({
                    "label": a["label"],
                    "pos": [float(x) for x in a["pos"]],
                    "spin_S": a["spin_S"]
                })
            builder.config["crystal_structure"]["atoms_uc"] = config_atoms
        
        expanded_config = {
            "crystal_structure": {
                **builder.config["crystal_structure"],
                "dimensionality": builder.dimensionality
            },
            "interactions": final_inters,
            "magnetic_structure": data.get("magnetic_structure", {}),
            "parameters": builder.config["parameters"],
            "tasks": builder.config["tasks"],
            "q_path": data.get("q_path", {}),
            "minimization": data.get("minimization", {}),
            "plotting": data.get("plotting", {}),
            "calculation": data.get("calculation", {"cache_mode": "none"}),
            "output": data.get("output", {"export_csv": False})
        }
        
        return expanded_config
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-unit-cell-atoms")
async def get_unit_cell_atoms(config: Dict[str, Any]):
    """
    Expand Wyckoff positions or return explicit atoms for visualizer preview.
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
        
        # 2. Basis Atoms
        struct_data = data.get("crystal_structure", {})
        wyckoff_atoms = struct_data.get("wyckoff_atoms", [])
        atom_mode = struct_data.get("atom_mode", "symmetry")
        builder.dimensionality = struct_data.get("dimensionality", "3D")

        if atom_mode == "explicit":
            # Treat Wyckoff atoms as the full unit cell directly
            config_atoms = []
            for a in wyckoff_atoms:
                config_atoms.append({
                    "label": a.get("label", "Atom"),
                    "pos": [float(x) for x in a.get("pos", [0, 0, 0])],
                    "spin_S": a.get("spin_S", 0.5)
                })
            return config_atoms
        else:
            # Standard Wyckoff expansion
            for atom in wyckoff_atoms:
                builder.add_wyckoff_atom(
                    label=atom.get("label", "Atom"),
                    pos=atom.get("pos", [0, 0, 0]),
                    spin=atom.get("spin_S", 0.5)
                )
            
            # Re-build atoms list
            config_atoms = []
            for a in builder.atoms_uc:
                config_atoms.append({
                    "label": a["label"],
                    "pos": [float(x) for x in a["pos"]],
                    "spin_S": a["spin_S"]
                })
            return config_atoms
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-visualizer-data")
async def get_visualizer_data(config: Dict[str, Any]):
    """
    Return atoms and expanded bonds for visualization.
    Uses full symmetry expansion and rule expansion.
    """
    if MagCalcConfigBuilder is None:
        raise HTTPException(status_code=500, detail="MagCalcConfigBuilder not found on server.")
        
    try:
        builder = MagCalcConfigBuilder()
        data = config.get("data", {})
        
        # 2. Basis Atoms
        struct_data = data.get("crystal_structure", {})
        wyckoff_atoms = struct_data.get("wyckoff_atoms", [])
        atom_mode = struct_data.get("atom_mode", "symmetry")
        builder.dimensionality = struct_data.get("dimensionality", "3D")

        # 1. Lattice
        lattice = data.get("crystal_structure", {}).get("lattice_parameters", {})
        builder.set_lattice(
            a=lattice.get("a", 1.0),
            b=lattice.get("b", 1.0),
            c=lattice.get("c", 1.0),
            alpha=lattice.get("alpha", 90),
            beta=lattice.get("beta", 90),
            gamma=lattice.get("gamma", 90),
            space_group=lattice.get("space_group") if atom_mode == "symmetry" else None
        )

        if atom_mode == "explicit":
            config_atoms = []
            for a in wyckoff_atoms:
                config_atoms.append({
                    "label": a.get("label", "Atom"),
                    "pos": [float(x) for x in a.get("pos", [0, 0, 0])],
                    "spin_S": a.get("spin_S", 0.5),
                    "species": a.get("label", "Atom")
                })
            builder.config["crystal_structure"]["atoms_uc"] = config_atoms
            builder.atoms_uc = config_atoms
        else:
            for atom in wyckoff_atoms:
                builder.add_wyckoff_atom(
                    label=atom.get("label", "Atom"),
                    pos=atom.get("pos", [0, 0, 0]),
                    spin=atom.get("spin_S", 0.5)
                )

        # 3. Interactions
        # Copy interactions from input
        builder.config["interactions"] = {
            "heisenberg": [],
            "dm_interaction": [],
            "single_ion_anisotropy": [],
            "anisotropic_exchange": [],
            "applied_field": {}
        }
        
        if "list" in data.get("interactions", {}):
            # Explicit List Mode
            explicit_list = data["interactions"]["list"]
            for item in explicit_list:
                rtype = item.get("type", "heisenberg")
                if rtype == "heisenberg":
                    builder.config["interactions"]["heisenberg"].append(item)
                elif rtype in ["dm", "dm_interaction", "dm_manual"]:
                    builder.config["interactions"]["dm_interaction"].append(item)
                elif rtype == "anisotropic_exchange":
                    builder.config["interactions"]["anisotropic_exchange"].append(item)
        else:
            # Symmetry Rules Mode
            rules = data.get("interactions", {}).get("symmetry_rules", [])
            for rule in rules:
                 builder.add_symmetry_interaction(
                     type=rule.get("type"),
                     ref_pair=rule.get("ref_pair"),
                     value=rule.get("value"),
                     distance=rule.get("distance"),
                     offset=rule.get("offset")
                 )
        
        # 3. Expand Rules (Only in symmetry mode)
        # We don't need _expand_*_rules anymore if using add_symmetry_interaction,
        # but for BACKWARDS COMPATIBILITY with rules without ref_pair:
        if atom_mode == "symmetry":
            builder._expand_heisenberg_rules()
            builder._expand_anisotropic_exchange_rules()
            builder._expand_dm_rules()

        # 4. Prepare Response
        # Atoms
        resp_atoms = []
        for i, a in enumerate(builder.atoms_uc):
            resp_atoms.append({
                "label": a["label"],
                "pos": [float(x) for x in a["pos"]],
                "spin_S": a["spin_S"],
                "idx": i
            })
            
        # Bonds
        resp_bonds = []
        
        # Helper to find index by label (assuming unique labels in atoms_uc)
        label_map = {a["label"]: i for i, a in enumerate(builder.atoms_uc)}
        
        # Collect all types
        all_inters = []
        for r in builder.config["interactions"]["heisenberg"]:
             r["interaction_type"] = "heisenberg"
             all_inters.append(r)
        for r in builder.config["interactions"]["dm_interaction"]:
             r["interaction_type"] = "dm"
             all_inters.append(r)
        for r in builder.config["interactions"]["anisotropic_exchange"]:
             r["interaction_type"] = "anisotropic"
             all_inters.append(r)
             
        for rule in all_inters:
            idx_i = -1
            idx_j = -1
            
            if "pair" in rule:
                lbl_i = rule["pair"][0]
                lbl_j = rule["pair"][1]
                
                if lbl_i in label_map and lbl_j in label_map:
                    idx_i = label_map[lbl_i]
                    idx_j = label_map[lbl_j]
            elif "atom_i" in rule and "atom_j" in rule:
                # Direct indices (explicit mode)
                idx_i = rule["atom_i"]
                idx_j = rule["atom_j"]
            
            if idx_i == -1 or idx_j == -1:
                continue
            
            offset = rule.get("rij_offset", [0, 0, 0])
             
            # Handle dm_manual specialized offset keys if present
            if "offset_j" in rule:
                offset = rule["offset_j"]
            
            # Format value for label
            val = rule.get("value")
            label_text = ""
            dm_vec = None
            
            if rule["interaction_type"] == "heisenberg":
                label_text = str(val)
            elif rule["interaction_type"] == "dm":
                label_text = "DM"
                # If val is vector, store it
                # It might be symbolic strings, e.g. ["0", "-Dy", "-Dz"]
                # We need to evaluate them using config['parameters']
                if isinstance(val, list) and len(val) == 3:
                     # Get parameters dict
                     params = config.get("data", {}).get("parameters", {})
                     # Simple safe eval context
                     ctx = {k: float(v) for k, v in params.items() if isinstance(v, (int, float))}
                     
                     eval_vec = []
                     for comp in val:
                         if isinstance(comp, (int, float)):
                             eval_vec.append(float(comp))
                         elif isinstance(comp, str):
                             try:
                                 # Very basic eval: simple arithmetic + params
                                 # We use python's eval but restricted globals
                                 # Using simple replacements or eval with restricted scope
                                 allowed_names = ctx
                                 # Safe enough for local tool - standard eval with restricted context
                                 res = eval(comp, {"__builtins__": {}}, allowed_names)
                                 eval_vec.append(float(res))
                             except Exception:
                                 # If eval fails (e.g. unknown param), default to 0 or keep as is?
                                 # Visualizer needs numbers.
                                 eval_vec.append(0.0)
                         else:
                             eval_vec.append(0.0)
                     dm_vec = eval_vec
            elif rule["interaction_type"] == "anisotropic":
                label_text = "Aniso"

            resp_bonds.append({
                "atom_i": idx_i,
                "atom_j": idx_j,
                "offset": offset,
                "type": rule["interaction_type"],
                "label": label_text,
                "dm_vector": dm_vec,
                "distance": rule.get("distance", 0)
            })

        return {
            "atoms": resp_atoms,
            "bonds": resp_bonds
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Enable reload to pick up code changes automatically!
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
