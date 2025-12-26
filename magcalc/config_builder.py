
import numpy as np
import spglib
import yaml
import logging
from typing import List, Dict, Tuple, Optional, Union, Any

# Configure logger
logger = logging.getLogger(__name__)

class MagCalcConfigBuilder:
    """
    Builder class for generating MagCalc configuration YAML files.
    Supports symmetry operations via spglib.
    """
    def __init__(self):
        self.config = {
            "crystal_structure": {},
            "calculation": {},
            "interactions": {
                "heisenberg": [],
                "dm_interaction": [],
                "single_ion_anisotropy": [],
                "anisotropic_exchange": [],
                "interaction_matrix": [],
                "applied_field": {}
            },
            "parameters": {},
            "minimization": {},
            "q_path": {},
            "output": {},
            "plotting": {},
            "tasks": {},
            "magnetic_structure": {}
        }
        
        # Internal State
        self.lattice_parameters = {}
        self.space_group_number = None
        self.symmetry_ops = None # {'rotations': [], 'translations': []}
        
        # Atom Storage: List of dicts {label, species, pos (frac), spin_S, wyckoff_label}
        # We store the *full expanded* unit cell here.
        self.atoms_uc = [] 
        self.dimensionality = "3D" 
        
        # Validation mapping
        self._atom_label_to_idx = {}

    def set_lattice(self, a: float, b: float = None, c: float = None, 
                   alpha: float = 90.0, beta: float = 90.0, gamma: float = 90.0,
                   space_group: int = None):
        """
        Define lattice parameters and optional space group.
        :param space_group: Integer Number (1-230). 
        """
        if b is None: b = a
        if c is None: c = a
        
        self.lattice_parameters = {
            "a": a, "b": b, "c": c,
            "alpha": alpha, "beta": beta, "gamma": gamma
        }

        if space_group:
            self.space_group_number = space_group
            self._load_symmetry_ops(space_group)

        # Calculate lattice vectors
        # Calculate lattice vectors
        self.lattice_vectors = self._params_to_vectors(a, b, c, alpha, beta, gamma)
        self.config["crystal_structure"]["lattice_vectors"] = self.lattice_vectors.tolist()
        self.config["crystal_structure"]["lattice_parameters"] = self.lattice_parameters

    def _params_to_vectors(self, a, b, c, alpha_deg, beta_deg, gamma_deg):
        """Convert lattice parameters to Cartesian vectors (a || x)."""
        alpha, beta, gamma = np.radians([alpha_deg, beta_deg, gamma_deg])
        
        va = np.array([a, 0, 0])
        vb = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])
        
        cx = c * np.cos(beta)
        cy = (c * b * np.cos(alpha) - vb[0] * cx) / vb[1]
        cz = np.sqrt(c**2 - cx**2 - cy**2)
        
        vc = np.array([cx, cy, cz])
        return np.array([va, vb, vc])

    def add_interaction_rule(self, type: str, distance: float, value: Any, 
                             pair: Tuple[str, str] = None, name: str = None):
        """
        Add a simple interaction rule. 
        If 'pair' is None, applies to ALL pairs at that distance.
        """
        rule = {
            "type": type,
            "distance": float(distance),
            "value": value
        }
        if pair:
            rule["pair"] = list(pair)
            
        # Target list
        target_list = None
        if type == "heisenberg":
            target_list = self.config["interactions"]["heisenberg"]
        elif type == "anisotropic_exchange":
             # Value should be list [Jxx, Jyy, Jzz]
             target_list = self.config["interactions"]["anisotropic_exchange"]
             
        if target_list is not None:
            target_list.append(rule)
        else:
            logger.warning(f"Unknown interaction type {type}")

    def add_symmetry_interaction(self, type: str, ref_pair: Tuple[str, str], value: Any, 
                                 distance: float = None, offset: List[int] = None):
        """
        Add an interaction and propagate it by symmetry.
        Especially for DM (dm_manual) and Anisotropic Exchange.
        
        :param value: The interaction value (e.g. D vector [Dx, Dy, Dz] or matrix).
        :param offset: Optional explicit offset [u, v, w] for the reference bond j relative to i.
        """
        if not self.symmetry_ops:
             logger.warning("No symmetry operations loaded. Cannot propagate.")
             return

        # 1. Identify Reference Bond Vector
        def _resolve_atom_idx(lbl):
            if lbl in self._atom_label_to_idx:
                return self._atom_label_to_idx[lbl]
            # Fallback: try appending "0" (common renaming pattern for single/multiplicity handling)
            lbl_0 = f"{lbl}0"
            if lbl_0 in self._atom_label_to_idx:
                return self._atom_label_to_idx[lbl_0]
            raise KeyError(lbl)

        try:
            idx_i = _resolve_atom_idx(ref_pair[0])
            idx_j = _resolve_atom_idx(ref_pair[1])
        except KeyError as e:
            logger.error(f"Could not find atom {e} in unit cell for interaction rule.")
            return

        pos_i = np.array(self.atoms_uc[idx_i]["pos"])
        pos_j_uc = np.array(self.atoms_uc[idx_j]["pos"])
        
        best_offset = None
        if offset is not None:
            best_offset = np.array(offset)
        else:
            # Find the specific image of j that matches "distance" (or assume nearest)
            # We need the relative vector r_ij in CARTESIAN to check distance, but fractional for symmetry.
            # Let's search nearest images.
            best_dist = 1e9
            
            from itertools import product
            if self.dimensionality == "2D":
                offsets_to_check = [ (u, v, 0) for u, v in product([-1, 0, 1], repeat=2) ]
            else:
                offsets_to_check = list(product([-1, 0, 1], repeat=3))

            for off in offsets_to_check:
                off_vec = np.array(off)
                pos_j_img = pos_j_uc + off_vec
                
                # Cartesian distance
                d_vec_cart = (pos_j_img - pos_i) @ self.lattice_vectors
                dist = np.linalg.norm(d_vec_cart)
                
                if distance:
                    if abs(dist - distance) < 0.05:
                         best_offset = off_vec
                         break
                else:
                    if dist < best_dist and dist > 0.01:
                        best_dist = dist
                        best_offset = off_vec
        
        if best_offset is None:
             raise ValueError(f"Could not identify reference bond for {ref_pair} near distance {distance}")

        # Final bond distance
        final_dist = float(distance if distance else best_dist)

        # Reference Bond
        ref_frac_diff = (pos_j_uc + best_offset) - pos_i # vector from i to j
        ref_center_pos = pos_i # bond starts at i
        
        # 2. Apply Symmetry to find Orbit
        # We transform the bond vector and the bond center
        rotations = self.symmetry_ops['rotations']
        
        orbit_bonds = [] # List of (atom_i_lbl, atom_j_lbl, offset_j, transformed_value)
        added_bonds_keys = set()
        
        # Check if value is symbolic (contains strings)
        # Check if value is symbolic (contains strings)
        if isinstance(value, str):
            # Parse CSV string if needed (e.g. "0,-Dy,Dz" or "[0, -Dy, 0]")
            # Remove brackets and parentheses
            cleaned_val = value.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
            if ',' in cleaned_val:
                value = [v.strip() for v in cleaned_val.split(',') if v.strip()]
            
            # Auto-pad 2D vectors to 3D if likely intended (e.g. "0, -Dy" -> "0, -Dy, 0")
            # Only if it looks like a vector (list of length 2)
            if isinstance(value, list) and len(value) == 2:
                value.append("0")
        
        is_symbolic = any(isinstance(v, str) for v in value) if isinstance(value, (list, tuple, np.ndarray)) else isinstance(value, str)
        if is_symbolic:
            import sympy as sp
            # Wrap in list if scalar, handling the processed value
            mat_input = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
            try:
                val_vec = sp.Matrix(mat_input)
            except Exception:
                # Fallback: try sympifying limits or handle flat lists for matrix
                formatted_input = []
                for v in mat_input:
                    if isinstance(v, str):
                         formatted_input.append(sp.sympify(v) if v.strip() and v.strip() != "-" else 0)
                    else:
                         formatted_input.append(v)
                val_vec = sp.Matrix(formatted_input)
        else:
            val_vec = np.array(value, dtype=float)
        
        # RESTART LOOP with zip
        for R, t in zip(self.symmetry_ops['rotations'], self.symmetry_ops['translations']):
             # 1. Map Start Atom
             p_i_mapped = R @ pos_i + t
             p_i_wrapped = self._wrap_pos(p_i_mapped)
             atom_k, _ = self._find_atom_at_pos(p_i_wrapped)
             if not atom_k: continue
             
             # 2. Map Bond Vector
             # bond' = R * bond (vector doesn't translate)
             vec_prime = R @ ref_frac_diff
             
             # 3. Determine End Atom position (unwrapped)
             # p_end_prime = p_i_mapped + vec_prime
             p_end_prime = p_i_mapped + vec_prime # Absolute fractional
             
             # 4. Find End Atom Label (wrapped)
             p_end_wrapped = self._wrap_pos(p_end_prime)
             atom_l, _ = self._find_atom_at_pos(p_end_wrapped)
             if not atom_l: continue
             
             # 5. Calculate Relative Offset
             # p_i_mapped = pos_k_uc + off_k
             # p_end_prime = pos_l_uc + off_l
             # Correct bond in YAML (from image 0 of k) is k -> l(off_l - off_k)
             pos_k_uc = np.array(self.atoms_uc[self._atom_label_to_idx[atom_k['label']]]['pos'])
             pos_l_uc = np.array(self.atoms_uc[self._atom_label_to_idx[atom_l['label']]]['pos'])
             
             off_k = np.round(p_i_mapped - pos_k_uc).astype(int)
             off_l = np.round(p_end_prime - pos_l_uc).astype(int)
             offset_final = off_l - off_k
             
             # SANITY CHECK: distance & Dimensionality
             if self.dimensionality == "2D" and abs(offset_final[2]) > 0.01:
                 continue

             d_final_vec = (pos_l_uc + offset_final - pos_k_uc) @ self.lattice_vectors
             d_final = np.linalg.norm(d_final_vec)
             if abs(d_final - final_dist) > 0.1:
                 logger.warning(f"Symmetry propagation distance mismatch: {atom_k['label']}->{atom_l['label']} dist={d_final:.4f} (target {final_dist:.4f})")
                 continue
             
             # 6. Transform Value
             if type == "dm":
                 # DM Vector: D' = det(R) * R * D (Axial vector)
                 R_cart = self._lattice_rotation(R)
                 det_R = float(np.linalg.det(R))
                 
                 if is_symbolic:
                     import sympy as sp
                     R_cart_sym = sp.Matrix(R_cart)
                     val_p = det_R * R_cart_sym * val_vec
                     val_p_list = [val_p[0], val_p[1], val_p[2]]
                 else:
                     val_p = det_R * (R_cart @ val_vec)
                     val_p_list = val_p.tolist()
                 
                 # Key tuple for uniqueness: (label_k, label_l, offset_final)
                 bond_key = (atom_k['label'], atom_l['label'], tuple(offset_final))
                 rev_bond_key = (atom_l['label'], atom_k['label'], tuple(-offset_final))
                 
                 if bond_key not in added_bonds_keys and rev_bond_key not in added_bonds_keys:
                     # Add k->l entry
                     self._add_dm_entry(atom_k['label'], atom_l['label'], offset_final, val_p_list, final_dist)
                     added_bonds_keys.add(bond_key)
                     added_bonds_keys.add(rev_bond_key)
                     
                     # Add l->k entry (Reverse)
                     if is_symbolic:
                         val_p_inv = [-v for v in val_p_list]
                     else:
                         val_p_inv = (-np.array(val_p_list)).tolist()
                         
                     self._add_dm_entry(atom_l['label'], atom_k['label'], -offset_final, val_p_inv, final_dist)

             elif type == "heisenberg":
                 # Scalar: value doesn't change with rotation
                 bond_key = (atom_k['label'], atom_l['label'], tuple(offset_final))
                 rev_bond_key = (atom_l['label'], atom_k['label'], tuple(-offset_final))
                 if bond_key not in added_bonds_keys and rev_bond_key not in added_bonds_keys:
                     self._add_heisenberg_entry(atom_k['label'], atom_l['label'], offset_final, value, final_dist)
                     added_bonds_keys.add(bond_key)
                     added_bonds_keys.add(rev_bond_key)
                     self._add_heisenberg_entry(atom_l['label'], atom_k['label'], -offset_final, value, final_dist)

             elif type == "anisotropic_exchange":
                 # diagonal tensor J' = R J R^T
                 R_cart = self._lattice_rotation(R)
                 
                 # Construct diagonal matrix
                 if is_symbolic:
                     import sympy as sp
                     J_diag = sp.diag(*val_vec)
                     R_cart_sym = sp.Matrix(R_cart)
                     J_prime = R_cart_sym * J_diag * R_cart_sym.T
                     # Resulting diagonal? Usually for these models we keep it simple
                     # But let's extract the diagonal if it's diagonal or just store the matrix if supported.
                     # Config schema for anisotropic_exchange usually expects list [vxx, vyy, vzz] or full matrix.
                     # For now, let's extract the diagonal as a list.
                     val_p_list = [J_prime[0,0], J_prime[1,1], J_prime[2,2]]
                 else:
                     J_diag = np.diag(val_vec)
                     J_prime = R_cart @ J_diag @ R_cart.T
                     val_p_list = np.diag(J_prime).tolist()
                 
                 bond_key = (atom_k['label'], atom_l['label'], tuple(offset_final))
                 rev_bond_key = (atom_l['label'], atom_k['label'], tuple(-offset_final))
                 if bond_key not in added_bonds_keys and rev_bond_key not in added_bonds_keys:
                     self._add_anisotropic_entry(atom_k['label'], atom_l['label'], offset_final, val_p_list, final_dist)
                     added_bonds_keys.add(bond_key)
                     added_bonds_keys.add(rev_bond_key)
                     # Symmetric
                     self._add_anisotropic_entry(atom_l['label'], atom_k['label'], -offset_final, val_p_list, final_dist)
                 
             elif type == "interaction_matrix":
                 # Full 3x3 interaction tensor J' = R J R^T
                 R_cart = self._lattice_rotation(R)
                 
                 if is_symbolic:
                     import sympy as sp
                     R_cart_sym = sp.Matrix(R_cart)
                     J_mat = sp.Matrix(val_vec) # Assuming val is passed as list of lists
                     J_prime = R_cart_sym * J_mat * R_cart_sym.T
                     val_p_list = J_prime.tolist()
                     val_p_list_T = J_prime.T.tolist()
                 else:
                     J_mat = np.array(value) 
                     J_prime = R_cart @ J_mat @ R_cart.T
                     val_p_list = J_prime.tolist()
                     val_p_list_T = J_prime.T.tolist()
                 
                 bond_key = (atom_k['label'], atom_l['label'], tuple(offset_final))
                 rev_bond_key = (atom_l['label'], atom_k['label'], tuple(-offset_final))
                 
                 if bond_key not in added_bonds_keys and rev_bond_key not in added_bonds_keys:
                     self._add_interaction_matrix_entry(atom_k['label'], atom_l['label'], offset_final, val_p_list, final_dist)
                     added_bonds_keys.add(bond_key)
                     added_bonds_keys.add(rev_bond_key)
                     
                     # Reverse bond: J_ji = J_ij^T
                     self._add_interaction_matrix_entry(atom_l['label'], atom_k['label'], -offset_final, val_p_list_T, final_dist)
                     
    def _add_dm_entry(self, lbl_i, lbl_j, offset, val, distance=None):
        # Ensure val is a list of strings or floats (not sympy objects)
        clean_val = [str(v) if not isinstance(v, (float, int)) else v for v in val]
        
        # Deduplicate
        for entry in self.config["interactions"]["dm_interaction"]:
            if entry["pair"] == [lbl_i, lbl_j] and entry["rij_offset"] == list(map(int, offset)):
                return

        # Generic spin model expects 'pair': [lbl_i, lbl_j]
        entry = {
            "type": "dm_manual",
            "pair": [lbl_i, lbl_j],
            "atom_i": self._atom_label_to_idx[lbl_i],
            "atom_j": self._atom_label_to_idx[lbl_j],
            "offset_j": list(map(int, offset)),
            "rij_offset": list(map(int, offset)),
            "value": clean_val
        }
        if distance:
            entry["distance"] = distance
        self.config["interactions"]["dm_interaction"].append(entry)

    def _add_heisenberg_entry(self, lbl_i, lbl_j, offset, val, distance=None):
        # Deduplicate
        for entry in self.config["interactions"]["heisenberg"]:
             if entry["pair"] == [lbl_i, lbl_j] and entry.get("rij_offset") == list(map(int, offset)):
                 return

        entry = {
            "type": "heisenberg",
            "pair": [lbl_i, lbl_j],
            "rij_offset": list(map(int, offset)),
            "value": val
        }
        if distance:
            entry["distance"] = distance
        self.config["interactions"]["heisenberg"].append(entry)

    def _add_anisotropic_entry(self, lbl_i, lbl_j, offset, val, distance=None):
        clean_val = [str(v) if not isinstance(v, (float, int)) else v for v in val]
        # Deduplicate
        for entry in self.config["interactions"]["anisotropic_exchange"]:
             if entry["pair"] == [lbl_i, lbl_j] and entry.get("rij_offset") == list(map(int, offset)):
                 return
                 
        entry = {
            "type": "anisotropic_exchange",
            "pair": [lbl_i, lbl_j],
            "rij_offset": list(map(int, offset)),
            "value": clean_val
        }
        if distance:
            entry["distance"] = distance
        self.config["interactions"]["anisotropic_exchange"].append(entry)

    def _add_interaction_matrix_entry(self, lbl_i, lbl_j, offset, val, distance=None):
        # val is 3x3 matrix (list of lists)
        clean_val = []
        for row in val:
            clean_row = [str(v) if not isinstance(v, (float, int)) else v for v in row]
            clean_val.append(clean_row)
            
        # Deduplicate
        for entry in self.config["interactions"]["interaction_matrix"]:
             if entry["pair"] == [lbl_i, lbl_j] and entry.get("rij_offset") == list(map(int, offset)):
                 return

        entry = {
            "type": "interaction_matrix",
            "pair": [lbl_i, lbl_j],
            "rij_offset": list(map(int, offset)),
            "value": clean_val
        }
        if distance:
            entry["distance"] = distance
        self.config["interactions"]["interaction_matrix"].append(entry)

    def _find_atom_at_pos(self, pos, tol=1e-3):
        for a in self.atoms_uc:
            p = np.array(a['pos'])
            diff = np.abs(p - pos)
            diff = np.minimum(diff, 1.0 - diff)
            if np.linalg.norm(diff) < tol:
                return a, np.linalg.norm(diff)
        return None, None

    def _get_translation_for_rot(self, R_target):
        # Helper to find t for a given R in cached ops
        # This assumes unique R, or we just rely on the main loop iteration
        # The main loop iterates zip(R, t), so strict mapping is preserved there.
        return None

    def _lattice_rotation(self, R_frac):
        """Convert fractional rotation matrix to Cartesian rotation matrix."""
        A = self.lattice_vectors.T
        A_inv = np.linalg.inv(A)
        return A @ R_frac @ A_inv

    def set_calculation(self, neighbor_shells: List[int] = [1, 1, 1], **kwargs):
        """Set calculation settings like neighbor_shells."""
        if "calculation_settings" not in self.config:
            self.config["calculation_settings"] = {}
        
        # Note: config structure puts 'neighbor_shells' in 'calculation_settings' 
        # (based on generic_spin_model.py usage) or 'calculation'?
        # generic_spin_model.py uses config_data.get("calculation_settings")
        self.config["calculation_settings"]["neighbor_shells"] = neighbor_shells
        self.config["calculation_settings"].update(kwargs)


    def add_single_ion_anisotropy(self, label: str, value: Any, axis: str = "z"):
        """
        Add Uniaxial SIA: D(Sz)^2.
        Propagated to all equivalent atoms (same label prefix).
        Current magcalc limitation: Only global Z axis or simple scalar D.
        """
        # Find all atoms with this label prefix/match?
        # If we use add_wyckoff_atom, atoms are "Cu0", "Cu1".
        # If user says label="Cu", applies to all "CuX".
        base_label = label
        
        interactions = []
        for a in self.atoms_uc:
            if a['label'].startswith(base_label): # Simple prefix match
                interactions.append({
                    "type": "sia",
                    "atom_label": a['label'],
                    "K_global": value # Assuming value is string "D" or float
                })
        
        self.config["interactions"]["single_ion_anisotropy"].extend(interactions)

    def set_field(self, magnitude: Any, direction: Union[List[float], str, List[str]]):
        """Set external magnetic field."""
        self.config["interactions"]["applied_field"] = {
            "H_magnitude_symbol": magnitude if isinstance(magnitude, str) else None,
            "H_vector": direction
        }
        if not isinstance(magnitude, str):
             # Fixed magnitude logic if needed, but schema usually expects symbol or map
             # Storing fixed mag in parameters?
             # For now, just store what is passed.
             if "parameters" not in self.config: self.config["parameters"] = {}
             # If magnitude is float, we might need to handle it. 
             # generic_model expects H_mag symbol.
             pass

    def add_wyckoff_atom(self, label: str, pos: List[float], spin: float, 
                         species: str = None, wyckoff_label: str = ""):
        """
        Add an atom defined by a Wyckoff position. 
        Expands to all equivalent positions using the loaded space group.
        
        :param label: Prefix for atom labels (e.g. "Cu"). Result will be "Cu0", "Cu1"... 
                      OR if unique logic is needed, will append index.
        :param pos: Fractional coordinates [u, v, w] of the representative site.
        """
        if not self.symmetry_ops:
            # No symmetry? Just add the one atom.
            # Use label exactly as provided (don't append 0)
            self._add_atom_raw(label, pos, spin, species)
            return

        pos_array = np.array(pos, dtype=float)
        rotations = self.symmetry_ops['rotations']
        translations = self.symmetry_ops['translations']
        
        orbit_positions = []
        
        for R, t in zip(rotations, translations):
            # Apply Op: x' = R x + t
            new_pos = R @ pos_array + t
            new_pos = self._wrap_pos(new_pos)
            
            # Check duplicates
            if not self._is_duplicate(new_pos, orbit_positions):
                orbit_positions.append(new_pos)
        
        # Add all generated atoms
        if len(orbit_positions) == 1:
            # Single site? Try to preserve the label exactly as given
            atom_name = label
            if atom_name in [a['label'] for a in self.atoms_uc]:
                # Collision? Fall back to indexed
                atom_name = f"{label}0"
            self._add_atom_raw(atom_name, orbit_positions[0], spin, species)
        else:
            # Determine starting index based on existing atoms
            start_idx = 0
            existing_labels = [a['label'] for a in self.atoms_uc]
            while f"{label}{start_idx}" in existing_labels:
                start_idx += 1
                
            for i, p in enumerate(orbit_positions):
                atom_name = f"{label}{start_idx + i}"
                self._add_atom_raw(atom_name, p, spin, species)

    def _add_atom_raw(self, label, pos, spin, species):
        if species is None:
            # Guess from label (remove digits)
            species = "".join([c for c in label if c.isalpha()])
            
        self.atoms_uc.append({
            "label": label,
            "species": species,
            "pos": pos,
            "spin_S": spin
        })
        self._atom_label_to_idx[label] = len(self.atoms_uc) - 1

    def _wrap_pos(self, pos, tol=1e-5):
        """Wrap to [0, 1) and snap near 0/1."""
        p = pos % 1.0
        # Snap small eps to 0
        p[np.abs(p) < tol] = 0.0
        p[np.abs(p - 1.0) < tol] = 0.0
        return p

    def _is_duplicate(self, pos, pos_list, tol=1e-4):
        for p in pos_list:
            diff = np.abs(pos - p)
            # Check periodic closeness (0 and 1 are same)
            diff = np.minimum(diff, 1.0 - diff)
            if np.linalg.norm(diff) < tol:
                return True
        return False

    def align_atoms(self, reference_positions: List[List[float]], tolerance: float = 1e-3):
        """
        Reorder self.atoms_uc to match the order of reference_positions (fractional).
        Useful for aligning with legacy models that have a specific atom indexing.
        """
        new_atoms_uc = []
        old_atoms = list(self.atoms_uc)
        
        for ref_pos in reference_positions:
            ref_pos = np.array(ref_pos) % 1.0 # Ensure [0, 1)
            found_idx = -1
            best_dist = tolerance
            
            for i, atom in enumerate(old_atoms):
                p = np.array(atom['pos']) % 1.0
                diff = np.abs(p - ref_pos)
                diff = np.minimum(diff, 1.0 - diff)
                dist = np.linalg.norm(diff)
                
                if dist < best_dist:
                    best_dist = dist
                    found_idx = i
            
            if found_idx != -1:
                atom = old_atoms.pop(found_idx)
                # Re-label to match the new index in this reference sequence
                new_idx = len(new_atoms_uc)
                prefix = "".join([c for c in atom['label'] if c.isalpha()])
                atom['label'] = f"{prefix}{new_idx}"
                new_atoms_uc.append(atom)
            else:
                logger.warning(f"Align atoms: Could not find atom at {ref_pos} within tol {tolerance}")
                
        if old_atoms:
            logger.info(f"Align atoms: {len(old_atoms)} atoms leftover not matched to reference.")
            new_atoms_uc.extend(old_atoms)
            
        self.atoms_uc = new_atoms_uc
        # Re-build label mapping
        self._atom_label_to_idx = {a['label']: i for i, a in enumerate(self.atoms_uc)}
        logger.info(f"Aligned {len(new_atoms_uc)} atoms to provided reference positions.")

    def _load_symmetry_ops(self, space_group: int):
        """
        Load symmetry operations from spglib database.
        """
        # Heuristic: Iterate Hall numbers to find the first one matching the SG number.
        # This usually corresponds to the standard setting.
        found = False
        for hall in range(1, 531):
            sg_info = spglib.get_spacegroup_type(hall)
            # sg_info['number'] is integer
            if sg_info['number'] == space_group:
                # Prefer "standard" setting? Usually the first one or specific choice.
                # Just take the first one found for now.
                # Actually, for Fdd2 (43), we want standard setting.
                self.symmetry_ops = spglib.get_symmetry_from_database(hall)
                if self.symmetry_ops:
                    found = True
                    logger.info(f"Loaded symmetry for SG {space_group}: Hall {hall} ({sg_info['international_short']})")
                    break 
        
        if not found:
            logger.error(f"FAILED to find Hall number for Space Group {space_group}")

    def set_symmetry_ops(self, rotations, translations):
        """Manually set symmetry operations."""
        self.symmetry_ops = {
            'rotations': rotations,
            'translations': translations
        }
        logger.info(f"Manually set {len(list(rotations))} symmetry operations.")

    def set_minimization(self, enabled: bool = True, method: str = "L-BFGS-B", maxiter: int = 500, initial_configuration: List[Dict] = None):
        """Set minimization parameters."""
        self.config["minimization"] = {
            "enabled": enabled,
            "method": method,
            "maxiter": maxiter
        }
        if initial_configuration:
            self.config["minimization"]["initial_configuration"] = initial_configuration

    def set_q_path(self, start: List[float], end: List[float], steps: int = 100):
        """Set a simple linear Q-path."""
        self.config["q_path"] = {
            "Start": start,
            "End": end,
            "path": ["Start", "End"],
            "points_per_segment": steps
        }
        
    def set_tasks(self, **kwargs):
        """Set task flags (e.g. run_minimization=True)."""
        # Default keys usually needed
        default_tasks = {
            "minimization": True,
            "dispersion": False,
            "sqw_map": False,
            "export_csv": False
        }
        default_tasks.update(kwargs)
        self.config["tasks"] = default_tasks
        
    def set_plotting(self, **kwargs):
        """Set plotting options."""
        self.config["plotting"] = kwargs
        
    def set_output(self, **kwargs):
        """Set output paths."""
        self.config["output"] = kwargs

    def set_magnetic_structure(self, **kwargs):
        """Set magnetic structure (e.g. type='pattern', directions=[...])."""
        self.config["magnetic_structure"] = kwargs

    def _expand_heisenberg_rules(self):
        """
        Expand distance-based Heisenberg rules into explicit pair interactions.
        Modifies self.config["interactions"]["heisenberg"] in place.
        """
        rules = self.config["interactions"].get("heisenberg", [])
        if not rules:
            return

        # Separate explicit pairs vs generic distance rules
        explicit_rules = [r for r in rules if "pair" in r]
        generic_rules = [r for r in rules if "pair" not in r]

        if not generic_rules:
            return

        # Pre-calc positions
        positions_uc = [np.array(a["pos"]) for a in self.atoms_uc]
        labels = [a["label"] for a in self.atoms_uc]
        
        # Neighbor offsets to check
        from itertools import product
        if self.dimensionality == "2D":
            offsets = [ (u, v, 0) for u, v in product([-1, 0, 1], repeat=2) ]
        else:
            offsets = list(product([-1, 0, 1], repeat=3))
        
        expanded_rules = []
        
        # For each generic rule, find all matching pairs
        for rule in generic_rules:
            target_dist = rule["distance"]
            val_sym = rule["value"]
            
            # Find all (i, j) pairs with this distance
            for idx_i, pos_i in enumerate(positions_uc):
                for idx_j, pos_j in enumerate(positions_uc):
                    # Check images of j
                    for off in offsets:
                        # BIDIRECTIONAL: include both (i, j, offset) and (j, i, -offset)
                        # because Hamiltonian has 0.5 factor and sums over all i, j.

                        off_vec = np.array(off)
                        pos_j_img = pos_j + off_vec
                        
                        d_vec_cart = (pos_j_img - pos_i) @ self.lattice_vectors
                        dist = np.linalg.norm(d_vec_cart)
                        
                        if abs(dist - target_dist) < 0.01:
                            # Found a match
                            # UNIDIRECTIONAL deduplication for Heisenberg
                            # Only add if (idx_i < idx_j) or (idx_i == idx_j and offset comparison)
                            # This matches 'manual' style and avoids 2x energy factor if unwanted.
                            # However, GenericSpinModel expects symmetric Jex for full 1.0*J energy.
                            # If we want to match MODERN's -36.87, we should check its behavior.
                            
                            # For consistency with builder's add_symmetry_interaction: 
                            # we'll keep both but allow a global 'bidirectional' flag?
                            # Actually, let's just make it correctly bidirectional first.
                            
                            new_rule = {
                                "type": "heisenberg",
                                "pair": [labels[idx_i], labels[idx_j]],
                                "value": val_sym,
                                "rij_offset": list(off),
                                "distance": float(dist)
                            }
                            expanded_rules.append(new_rule)
        
        # Combine
        self.config["interactions"]["heisenberg"] = explicit_rules + expanded_rules
        logger.info(f"Expanded {len(generic_rules)} generic rules into {len(expanded_rules)} explicit Heisenberg pairs.")

    def analyze_bond_symmetry(self, max_distance: float = 8.0, verbose: bool = False) -> List[Dict]:
        """
        Analyze crystal structure to find all unique bond orbits up to max_distance.
        Returns a list of 'orbits', where each orbit contains:
        - representative: {atom_i, atom_j, offset_j, distance} (The System-Recommended Reference)
        - members: List of all equivalent bonds in the orbit
        - distance: bond length
        
        This solves the ambiguity of distance-based rules by providing explicit reference bonds.
        """
        if not self.atoms_uc:
            logger.warning("No atoms defined. Cannot analyze bonds.")
            return []

        # 1. Generate ALL pairs up to max_distance
        positions = [np.array(a["pos"]) for a in self.atoms_uc]
        labels = [a["label"] for a in self.atoms_uc]
        
        # Determine search range for offsets
        # Estimate from max_distance and lattice vectors? 
        # For now, use fixed robust range or simple heuristic.
        # If lattice is 5A and max_dist is 8A, range 2 is enough.
        # Safe bet: [-2, 2] for typical inputs.
        search_range = 2 
        from itertools import product
        
        if self.dimensionality == "2D":
            offsets = [ (u, v, 0) for u, v in product(range(-search_range, search_range+1), repeat=2) ]
        else:
            offsets = list(product(range(-search_range, search_range+1), repeat=3))

        # Store all bonds as (idx_i, idx_j, tuple(offset))
        all_bonds = []
        
        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                for off in offsets:
                    off_vec = np.array(off)
                    pos_j_img = pos_j + off_vec
                    
                    # Distance check
                    d_vec_cart = (pos_j_img - pos_i) @ self.lattice_vectors
                    dist = np.linalg.norm(d_vec_cart)
                    
                    if 0.01 < dist <= max_distance:
                        all_bonds.append({
                            "indices": (i, j),
                            "offset": off,
                            "distance": dist,
                            "vector_frac": (pos_j + off_vec) - pos_i
                        })
        
        # 2. Group into Orbits
        # We need to apply symmetry to each bond and see which ones map to each other.
        # Bond B1 is equivalent to B2 if there exists (R, t) such that:
        # B1_start -> B2_start AND B1_end -> B2_end (modulo lattice)
        
        orbits = []
        processed_bonds = set() # Store unique IDs of processed bonds
        
        # Helper to get unique ID for a bond (i, j, offset)
        def get_bond_id(i, j, off):
            return (i, j, tuple(off))

        # Sort bonds by distance first to keep orbits ordered efficiently
        all_bonds.sort(key=lambda x: x["distance"])

        for bond in all_bonds:
            bid = get_bond_id(*bond["indices"], bond["offset"])
            if bid in processed_bonds:
                continue
                
            # New Orbit found!
            orbit_members = []
            
            # Start atom and bond vector
            # (Note: we need robust mapping. The simplest way is to apply ALL symops to this bond
            # and collect the generated bonds. This ensures we find the FULL orbit.)
            
            p_i = positions[bond["indices"][0]]
            offset_initial = np.array(bond["offset"])
            # p_j_full = positions[bond["indices"][1]] + offset_initial
            
            # The bond vector in fractional coords attached to p_i
            # We want to map: Start -> Start', End -> End'
            
            for R, t in zip(self.symmetry_ops['rotations'], self.symmetry_ops['translations']):
                 # Transform Start
                 p_start_map = R @ p_i + t
                 p_start_wrap = self._wrap_pos(p_start_map)
                 atom_k, _ = self._find_atom_at_pos(p_start_wrap)
                 if not atom_k: continue
                 idx_k = self._atom_label_to_idx[atom_k['label']]
                 
                 # Transform Bond Vector (vector part only rotates)
                 # vec = (p_j + off) - p_i
                 # vec' = R vec
                 vec_initial = (positions[bond["indices"][1]] + offset_initial) - p_i
                 vec_prime = R @ vec_initial
                 
                 # New End Pos (absolute) = New Start (absolute) + vec'
                 # But we need New Start (absolute) = p_start_map
                 p_end_prime = p_start_map + vec_prime
                 
                 # Identify Atom L at End
                 p_end_wrap = self._wrap_pos(p_end_prime)
                 atom_l, _ = self._find_atom_at_pos(p_end_wrap)
                 if not atom_l: continue
                 idx_l = self._atom_label_to_idx[atom_l['label']]
                 
                 # Calculate Offset L
                 # p_end_prime = pos_l + off_l
                 off_l = np.round(p_end_prime - positions[idx_l]).astype(int)
                 
                 # Calculate Offset K (mapped start)
                 # p_start_map = pos_k + off_k
                 off_k = np.round(p_start_map - positions[idx_k]).astype(int)
                 
                 # Relative offset for the bond (k -> l) in the config convention
                 # The config stores "offset_j" relative to i. 
                 # Here, the bond connects k (at pos_k_uc) to l (at pos_l_uc + relative_offset)
                 # Absolute: P_start -> P_end
                 # P_start = pos_k + off_k
                 # P_end = pos_l + off_l
                 # vector = P_end - P_start = (pos_l - pos_k) + (off_l - off_k)
                 # So relative offset = off_l - off_k
                 
                 final_offset = off_l - off_k
                 
                 # Add to orbit
                 member_id = (idx_k, idx_l, tuple(final_offset))
                 
                 # Avoid duplicates within loop if symops are redundant or map to same
                 if member_id not in [m["id"] for m in orbit_members]:
                     orbit_members.append({
                         "id": member_id,
                         "atom_i": labels[idx_k],
                         "atom_j": labels[idx_l],
                         "offset": list(final_offset),
                         "ref_id": bid # Link back to generator
                     })
                     processed_bonds.add(member_id)

            if not orbit_members:
                continue

            # 3. Select Representative
            # Criterion 1: Offset Magnitude (Smallest L1 norm)
            # Criterion 2: Atom Indicies (i then j)
            # Criterion 3: Lexicographical offset (to be deterministic)
            
            def sort_key(m):
                off = m["offset"]
                idx_i = self._atom_label_to_idx[m["atom_i"]]
                idx_j = self._atom_label_to_idx[m["atom_j"]]
                mag = sum(abs(x) for x in off)
                return (mag, idx_i, idx_j, off[0], off[1], off[2])
            
            orbit_members.sort(key=sort_key)
            representative = orbit_members[0]
            
            orbits.append({
                "representative": representative,
                "members": orbit_members,
                "distance": bond["distance"], # Approx, from the seed bond
                "multiplicity": len(orbit_members)
            })

        return orbits

    def _expand_anisotropic_exchange_rules(self):
        """
        Expand distance-based anisotropic exchange rules into explicit pair interactions.
        Symmetry-aware: picks an orbit representative and transforms the matrix.
        """
        rules = self.config["interactions"].get("anisotropic_exchange", [])
        if not rules:
            return

        explicit_rules = [r for r in rules if "pair" in r]
        generic_rules = [r for r in rules if "pair" not in r]

        if not generic_rules:
            return

        # Start fresh for expansion (preserving explicit ones)
        self.config["interactions"]["anisotropic_exchange"] = explicit_rules
        
        # Neighbor offsets
        from itertools import product
        if self.dimensionality == "2D":
            offsets = [ (u, v, 0) for u, v in product([-1, 0, 1], repeat=2) ]
        else:
            offsets = list(product([-1, 0, 1], repeat=3))
            
        labels = [a["label"] for a in self.atoms_uc]
        positions = [np.array(a["pos"]) for a in self.atoms_uc]

        for rule in generic_rules:
            target_dist = rule["distance"]
            value = rule["value"]
            
            # Find all potential bonds at this distance
            pool = []
            for i, pos_i in enumerate(positions):
                for j, pos_j in enumerate(positions):
                    for off in offsets:
                        off_vec = np.array(off)
                        pos_j_img = pos_j + off_vec
                        d_cart = (pos_j_img - pos_i) @ self.lattice_vectors
                        dist = np.linalg.norm(d_cart)
                        if abs(dist - target_dist) < 0.05:
                            pool.append((labels[i], labels[j], tuple(off)))
            
            # Group into orbits via symmetry
            while pool:
                ref_i, ref_j, ref_off = pool[0]
                # add_symmetry_interaction will add the whole orbit and transform values
                self.add_symmetry_interaction("anisotropic_exchange", (ref_i, ref_j), value, 
                                             distance=target_dist, offset=list(ref_off))
                
                # Remove added bonds from pool
                expanded = self.config["interactions"]["anisotropic_exchange"]
                pool = [b for b in pool if not any(
                    (e["pair"] == [b[0], b[1]] and e.get("rij_offset") == list(b[2]))
                    for e in expanded
                )]

    def _expand_dm_rules(self):
        """
        Expand distance-based DM rules into explicit pair interactions.
        Symmetry-aware: picks an orbit representative and transforms the DM vector.
        """
        rules = self.config["interactions"].get("dm_interaction", [])
        if not rules:
            return

        explicit_rules = [r for r in rules if "pair" in r]
        generic_rules = [r for r in rules if "pair" not in r]

        if not generic_rules:
            return

        # Start fresh for expansion (preserving explicit ones)
        self.config["interactions"]["dm_interaction"] = explicit_rules
        
        # Neighbor offsets
        from itertools import product
        if self.dimensionality == "2D":
            offsets = [ (u, v, 0) for u, v in product([-1, 0, 1], repeat=2) ]
        else:
            offsets = list(product([-1, 0, 1], repeat=3))
            
        labels = [a["label"] for a in self.atoms_uc]
        positions = [np.array(a["pos"]) for a in self.atoms_uc]

        for rule in generic_rules:
            target_dist = rule["distance"]
            value = rule["value"]
            
            # Find all potential bonds at this distance
            pool = []
            for i, pos_i in enumerate(positions):
                for j, pos_j in enumerate(positions):
                    for off in offsets:
                        off_vec = np.array(off)
                        pos_j_img = pos_j + off_vec
                        d_cart = (pos_j_img - pos_i) @ self.lattice_vectors
                        dist = np.linalg.norm(d_cart)
                        if abs(dist - target_dist) < 0.05:
                            pool.append((labels[i], labels[j], tuple(off)))
            
            # Group into orbits via symmetry
            while pool:
                ref_i, ref_j, ref_off = pool[0]
                
                # Check if this exact bond is already failing to be added
                # to prevent infinite loop if add_symmetry_interaction fails logic.
                initial_pool_size = len(pool)
                
                # add_symmetry_interaction will add the whole orbit and transform values
                try:
                    self.add_symmetry_interaction("dm", (ref_i, ref_j), value, 
                                                 distance=target_dist, offset=list(ref_off))
                except Exception as e:
                    logger.error(f"Error propagating DM interaction for {ref_i}->{ref_j}: {e}")
                
                # Remove added bonds from pool
                expanded = self.config["interactions"]["dm_interaction"]
                pool_after = [b for b in pool if not any(
                    (e["pair"] == [b[0], b[1]] and e.get("rij_offset") == list(b[2]))
                    for e in expanded
                )]
                
                # Loop breaker: if pool didn't shrink (and not empty), force remove first item
                if len(pool_after) >= initial_pool_size and pool_after:
                     logger.warning(f"Preventing infinite loop: forcing removal of {pool_after[0]}")
                     pool_after.pop(0)
                     
                pool = pool_after

    def save(self, filename: str):
        """Export configuration to YAML file."""
        # 1. Ensure atoms are in config
        config_atoms = []
        for a in self.atoms_uc:
            config_atoms.append({
                "label": a["label"],
                "pos": [float(x) for x in a["pos"]],
                "spin_S": a["spin_S"]
            })
        self.config["crystal_structure"]["atoms_uc"] = config_atoms
        
        # 2. Derive magnetic_elements list (unique species)
        species = sorted(list(set(a["species"] for a in self.atoms_uc if a.get("species"))))
        if not species: 
            # Fallback to labels if species not set, or first 1-2 chars
            species = [] # Logic refinement needed if user doesn't provide
            
        self.config["crystal_structure"]["magnetic_elements"] = species
        self.config["crystal_structure"]["dimensionality"] = self.dimensionality

        # 3. Expand Interaction Rules
        self._expand_heisenberg_rules()
        self._expand_anisotropic_exchange_rules()
        self._expand_dm_rules()

        # WRITE

    def save(self, filename: str):
        """Export configuration to YAML file."""
        # 1. Ensure atoms are in config
        config_atoms = []
        for a in self.atoms_uc:
            config_atoms.append({
                "label": a["label"],
                "pos": [float(x) for x in a["pos"]],
                "spin_S": a["spin_S"]
            })
        self.config["crystal_structure"]["atoms_uc"] = config_atoms
        
        # 2. Derive magnetic_elements list (unique species)
        species = sorted(list(set(a["species"] for a in self.atoms_uc if a.get("species"))))
        if not species: 
            # Fallback to labels if species not set, or first 1-2 chars
            species = [] # Logic refinement needed if user doesn't provide
            
        self.config["crystal_structure"]["magnetic_elements"] = species
        self.config["crystal_structure"]["dimensionality"] = self.dimensionality

        # 3. Expand Interaction Rules
        self._expand_heisenberg_rules()
        self._expand_anisotropic_exchange_rules()
        self._expand_dm_rules()

        # WRITE
        with open(filename, 'w') as f:
            yaml.dump(self.config, f, sort_keys=False)
        logger.info(f"Configuration saved to {filename}")

    def get_bond_constraints(self, bond: Dict) -> Dict:
        """
        Determine the allowed Exchange Matrix form for a given bond based on its Little Group.
        Returns:
            - allowed_matrix: 3x3 array (strings/0)
            - independent_vars: list of free parameters (e.g. ['J1', 'D', 'K1'])
            - little_group_ops: list of ops that preserve the bond
        """
        import sympy as sp
        
        # 1. Reconstruct Bond Geometery
        # Bond is defined by representative: atom_i, atom_j, offset_j
        lbl_i = bond["representative"]["atom_i"]
        lbl_j = bond["representative"]["atom_j"]
        offset = np.array(bond["representative"]["offset"])
        
        idx_i = self._atom_label_to_idx[lbl_i]
        idx_j = self._atom_label_to_idx[lbl_j]
        
        pos_i = np.array(self.atoms_uc[idx_i]["pos"])
        pos_j_uc = np.array(self.atoms_uc[idx_j]["pos"])
        
        # Fractional bond vector V = P_end - P_start
        # P_start = pos_i
        # P_end = pos_j_uc + offset
        vec_frac = (pos_j_uc + offset) - pos_i
        midpoint = pos_i + 0.5 * vec_frac
        
        # 2. Find Little Group
        # Op R, t is in Little Group if:
        # A) Maps bond to itself: R(V) = V  AND  Maps i->i (or j->j)
        # B) Maps bond to reverse: R(V) = -V AND Maps i->j (or j->i)
        
        little_group_ops = []
        is_bond_symmetric = False # If we have operations that swap i and j
        
        for i_op, (R, t) in enumerate(zip(self.symmetry_ops['rotations'], self.symmetry_ops['translations'])):
            # Apply R to vector
            vec_prime = R @ vec_frac
            
            # Case A: Does it preserve direction?
            if np.allclose(vec_prime, vec_frac, atol=1e-3):
                # Must also map start point to start point (modulo lattice)
                # p_i_map = R @ pos_i + t
                # diff = p_i_map - pos_i
                # is_integer = np.allclose(diff - np.round(diff), 0, atol=1e-3)
                
                # Check mapping of atom i -> atom i
                # (pos_i is fractional position of atom in Unit Cell)
                p_i_map = R @ pos_i + t
                # does this land on pos_i?
                # Check if p_i_map - pos_i is integer
                diff = p_i_map - pos_i
                map_i_i = np.allclose(diff - np.round(diff), 0, atol=1e-3)
                
                if map_i_i:
                    little_group_ops.append({"op": (R, t), "swap": False})
                continue
                
            # Case B: Does it reverse direction?
            if np.allclose(vec_prime, -vec_frac, atol=1e-3):
                # Must map start point (i) to end point (j)
                # p_i_map = R @ pos_i + t
                # Goal: p_i_map == pos_j_uc + offset (modulo lattice?? No, pos_j_uc + offset IS the absolute position)
                # Actually, p_i is mapped to p_j_absolute?
                # Let's check diff = p_i_map - (pos_j_uc + offset)
                
                # Wait. p_i maps to SOME equivalent of j.
                # If V' = -V, it means bond flips.
                # Start -> End.
                # i -> j.
                
                p_i_map = R @ pos_i + t
                p_end_abs = pos_j_uc + offset
                
                diff = p_i_map - p_end_abs
                map_i_j = np.allclose(diff - np.round(diff), 0, atol=1e-3)
                
                if map_i_j:
                     little_group_ops.append({"op": (R, t), "swap": True})
                     is_bond_symmetric = True

        # 3. Solve for Allowed Matrix J
        # J is 3x3 matrix. Elements J_ab.
        # We construct a linear system for the 9 elements.
        
        J_vars = sp.Matrix(sp.symbols('j0:9')).reshape(3, 3) # j0..j8
        # Flattened for solver: [j0, ..., j8]
        # Constraints: Coeffs * vars = 0
        
        constraints = [] # List of equations (expressions that must equal 0)
        
        for item in little_group_ops:
            R_frac = item["op"][0]
            swap = item["swap"]
            
            # Convert R to Cartesian for spin interactions
            R_cart = self._lattice_rotation(R_frac)
            
            # SANITIZE R_cart to help sympy
            # 1. Round
            R_cart_clean = np.round(R_cart, decimals=10)
            # 2. Sympy nsimplify (handles 0.5 -> 1/2, 1.0 -> 1)
            R_sym = sp.Matrix(R_cart_clean).applyfunc(sp.nsimplify)
            
            # Constraint Eq: J = R J R^T (if no swap)
            #               J = R J^T R^T (if swap)
            
            term1 = J_vars
            if swap:
                 term2 = R_sym * J_vars.T * R_sym.T
            else:
                 term2 = R_sym * J_vars * R_sym.T
                 
            diff = term1 - term2
            for elem in diff:
                constraints.append(elem)
                
        # Solve
        # We need to find the basis of the null space of these linear equations
        # But sympy solve can handle it directly if we pass the system
        
        # Collect all equations
        # Solves for j0..j8 in terms of free parameters
        sol = sp.solve(constraints, list(J_vars))
        
        # Reconstruct matrix from solution
        J_solved = J_vars.subs(sol)
        
        # 4. Extract free parameters and clean up
        free_symbols = sorted(list(J_solved.free_symbols), key=lambda x: x.name)
        
        # Map weird sympy symbols (j0, j5...) to nice names (J1, J2, D...)?
        # For user display, we want a clean string matrix
        return {
            "symbolic_matrix": [[str(e) for e in row] for row in J_solved.tolist()],
            "free_parameters": [str(s) for s in free_symbols],
            "little_group_size": len(little_group_ops),
            "is_centrosymmetric": is_bond_symmetric # Roughly
        }

    def calc_interaction_matrix(self, ref_bond: Dict, params: Dict[str, float]) -> np.ndarray:
        """
        Calculate numeric 3x3 interaction matrix from free parameters and symmetry constraints.
        """
        import sympy as sp
        
        # Get symbolic form (or re-compute? caching better)
        # For robustness, recompute or pass in constraints.
        # Assuming we just call get_bond_constraints again for now or use cached logic.
        constraints = self.get_bond_constraints({"representative": ref_bond})
        # Note: ref_bond dict format must match whatever get_bond_constraints expects.
        # Ideally get_bond_constraints should cache based on bond ID.
        
        sym_matrix = sp.Matrix(constraints["symbolic_matrix"]) # Str to Sympy
        
        # Substitute params
        # params dict keys must match the free_parameters (j0, j1...)
        # WARN: If get_bond_constraints generates new symbols each time, keys won't match!
        # Symbols are static 'j0:9' so they should be consistent IF structure is same.
        # But sympy solve output depends on algorithm.
        
        # Better approach: Pass 'constraints' object which contains the solved matrix expression if possible.
        # But since we return strings, we re-parse.
        
        subs_dict = {}
        for k, v in params.items():
            subs_dict[sp.Symbol(k)] = float(v)
            
        # J_numeric = sym_matrix.subs(subs_dict) # This might leave symbols if params missing
        
        # Evaluate to numpy
        # Use lambdify or eval
        J_num = np.zeros((3, 3))
        for r in range(3):
            for c in range(3):
                expr = sym_matrix[r, c]
                val = expr.subs(subs_dict)
                try:
                    J_num[r, c] = float(val)
                except TypeError:
                    # Still symbolic?
                    logger.error(f"Parameter missing for interaction matrix element {r},{c}: {expr}")
                    
        return J_num

