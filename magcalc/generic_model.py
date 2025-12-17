import logging
import os
import sys
from itertools import product
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
from numpy import linalg as la
import sympy as sp
from scipy.optimize import minimize
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from ase.io import read as read_cif
except ImportError:
    read_cif = None
    logger.warning("ASE not installed. CIF loading will be disabled.")

# Ensure we can import cif_utils
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import cif_utils
except ImportError:
    pass # Will handle inside if needed

# Helper functions for rotations
def RotX(angle):
    return sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(angle), -sp.sin(angle)],
        [0, sp.sin(angle), sp.cos(angle)]
    ])

def RotY(angle):
    return sp.Matrix([
        [sp.cos(angle), 0, sp.sin(angle)],
        [0, 1, 0],
        [-sp.sin(angle), 0, sp.cos(angle)]
    ])

def RotZ(angle):
    return sp.Matrix([
        [sp.cos(angle), -sp.sin(angle), 0],
        [sp.sin(angle), sp.cos(angle), 0],
        [0, 0, 1]
    ])

# Custom rotation from spin_model.py (Z-like but permutation)
# [[0, sin, cos], [0, -cos, sin], [1, 0, 0]]
# Let's call it RotZ_KFe3J for convenience? 
# Or just let user define it with Matrix([[...]]) in YAML?
# User might prefer defining it explicitly in YAML if it's non-standard.
# But providing these standard ones is helpful.

def safe_eval(expr, context):
    """Safely evaluate a mathematical expression using sympy/numpy symbols."""
    # Allow simple math
    allowed_names = {
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "asin": sp.asin,
        "acos": sp.acos,
        "tan": sp.tan,
        "atan": sp.atan,
        "pi": sp.pi,
        "exp": sp.exp,
        "abs": abs,
        "Matrix": sp.Matrix,
        "eye": sp.eye,
        "zeros": sp.zeros,
        "RotX": RotX,
        "RotY": RotY,
        "RotZ": RotZ,
    }
    allowed_names.update(context)
    return eval(expr, {"__builtins__": {}}, allowed_names)

class GenericSpinModel:
    """
    A generic spin model that implements the required interface (unit_cell, atom_pos, Hamiltonian)
    by reading from a configuration dictionary.
    """
    def __init__(self, config, base_path="."):
        self.config = config
        self.base_path = base_path
        
        self.crystal_config = config.get('crystal_structure', {})
        self.interactions_config = config.get('interactions', [])
        self.optimized_matrices = None
        
        # Pre-load structure data
        self._load_structure()
        
        
        # Pre-calc neighbors
        self._atoms_ouc = self._generate_atom_pos_ouc()
        
        # Mimic module attribute for logging
        self.__name__ = "GenericSpinModel"

        # --- VALIDATION ---
        try:
            from .schema import MagCalcConfig
            try:
                # model_validate allows extra fields if ConfigDict(extra='allow') was not set?
                # Actually our schema uses extra='allow' in sub-configs but strict in others?
                # Using model_validate(config)
                validated = MagCalcConfig.model_validate(self.config)
                # Dump back to dict to preserve existing logic (which expects dicts)
                # mode='json' or 'python'? 'python' keeps objects like datetime, but we want primitives if possible.
                # However, original config was dict of primitives.
                self.config = validated.model_dump()
                logger.info("Configuration validation passed.")
            except Exception as e:
                logger.warning(f"Configuration validation failed: {e}")
                # We can choose to raise or just warn. 
                # For backward compatibility with partial configs (like sw_KFe3J.py might produce?), maybe warn?
                # But creating a robust system implies failing on invalid data.
                # Let's WARN for now to avoid breaking existing hybrid scripts, 
                # but eventually we want to Enforce.
                # Actually, if we are in 'legacy' mode (hybrid), the config might be partial.
                pass
        except ImportError:
            logger.warning("Could not import MagCalcConfig schema. Validation skipped.")

        
    def _load_structure(self):
        """
        Load crystal structure from config (Explicit or CIF).
        Sets self._uc_vectors and self._r_pos (Cartesian).
        """
        crystal_struct = self.config.get('crystal_structure', {})
        
        # 1. CIF File (Priority 1)
        cif_file = crystal_struct.get('cif_file')
        if cif_file:
            if not os.path.isabs(cif_file) and self.base_path:
                cif_file = os.path.join(self.base_path, cif_file)
            
            if read_cif is None:
                raise ImportError("ASE not installed or read_cif import failed.")
                
            try:
                self.atoms = read_cif(cif_file)
                # Filter by magnetic elements
                mag_elements = crystal_struct.get('magnetic_elements')
                if mag_elements:
                     self.atoms = self.atoms[
                         [atom.symbol in mag_elements for atom in self.atoms]
                     ]
                self._uc_vectors = self.atoms.cell[:]
                self._r_pos = self.atoms.get_positions()
                return 
            except Exception as e:
                raise ValueError(f"Failed to load CIF structure from {cif_file}: {e}")

        # 2. Explicit Definition
        # Determine Unit Cell Vectors
        if 'lattice_parameters' in crystal_struct:
            lp = crystal_struct['lattice_parameters']
            # Dict or object? Config is dict.
            a, b, c = lp['a'], lp['b'], lp['c']
            alpha = np.deg2rad(lp.get('alpha', 90.0))
            beta = np.deg2rad(lp.get('beta', 90.0))
            gamma = np.deg2rad(lp.get('gamma', 90.0))
            
            # Standard conversion (a || x, b in xy)
            v_a = np.array([a, 0, 0])
            v_b = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])
            
            # c vector components
            # cos(alpha) = (b.c) / bc = (bx*cx + by*cy)/bc
            # cos(beta) = (a.c) / ac = cx / c
            cx = c * np.cos(beta)
            cy = (c * b * np.cos(alpha) - v_b[0] * cx) / v_b[1] 
            # cz = sqrt(c^2 - cx^2 - cy^2)
            cz_sq = c**2 - cx**2 - cy**2
            cz = np.sqrt(cz_sq) if cz_sq > 0 else 0
            
            v_c = np.array([cx, cy, cz])
            self._uc_vectors = np.array([v_a, v_b, v_c])
            
        elif 'lattice_vectors' in crystal_struct:
             self._uc_vectors = np.array(crystal_struct['lattice_vectors'], dtype=float)
        else:
             raise ValueError("No crystal structure defined (requires 'lattice_parameters', 'lattice_vectors', or 'cif_file').")
             
        # Process Atoms (Fractional -> Cartesian)
        if 'atoms_uc' in crystal_struct:
             atoms = crystal_struct['atoms_uc']
             # Interpret 'pos' as fractional coordinates
             frac_pos_list = [a['pos'] for a in atoms]
             frac_pos = np.array(frac_pos_list, dtype=float)
             
             # Convert to Cartesian: r = u*a + v*b + w*c
             # Matrix algebra: R_cart = Frac * UC_matrix (if UC rows are vectors)
             # Frac shape (N, 3). UC shape (3, 3) rows a,b,c.
             # R_i = u_i * a + v_i * b + w_i * c
             #     = [u v w] . [a b c]^T ? No.
             #     = [u v w] * [a; b; c]
             self._r_pos = np.dot(frac_pos, self._uc_vectors)
             
        elif 'atom_positions' in crystal_struct:
             # Legacy/Flat support - Assume Cartesian if only this is provided? 
             # Or assume fractional now? 
             # To be consistent with "Revise to use fractional", let's assume fractional IF lattice params logic matches expectation?
             # But 'atom_positions' is raw array. 
             # Let's assume 'atoms_uc' is the modern way (schema). 'atom_positions' was generic support.
             # I will treat 'atom_positions' as Cartesian for backward compat if anyone uses it directly without atoms_uc.
             # But warned in plan.
             # Let's assume fractional for consistency or CARTESIAN for safety?
             # Given atoms_uc is preferred, let's treat atom_positions as raw Cartesian override if someone bypasses atoms_uc.
             self._r_pos = np.array(crystal_struct['atom_positions'], dtype=float)
        else:
             raise ValueError("Must provide 'atoms_uc' defining atom positions.")

    def unit_cell(self):
        """Find the unit cell vectors."""
        return self._uc_vectors

    def atom_pos(self):
        """Find the atom positions in the unit cell."""
        return self._r_pos

    def atom_pos_ouc(self):
        """Returns neighbor positions."""
        return self._atoms_ouc

    def _generate_atom_pos_ouc(self):
        """Generate neighbors (internal helper)."""
        uc = self.unit_cell()
        apos = self.atom_pos()
        apos_len = len(apos)
        
        # Standard -1 to 1 supercell
        r_pos_ouc = [apos[k] for k in range(apos_len)]
        
        from itertools import product
        
        # Check config for dimensionality limit (default 3)
        # 2 = Plane (a, b) only. 3 = Full 3D (a, b, c)
        c_struct = self.config.get('crystal_structure', {})
        dims = c_struct.get('dimensionality', 3)
        
        neighbor_offsets = []
        rng = range(-1, 2)
        
        if dims == 2:
             # Loop i, j. l=0 fixed.
             loop_iter = product(rng, rng, [0])
        else:
             # Loop i, j, l
             loop_iter = product(rng, rng, rng)

        neighbors = [
            apos[k] + i * uc[0] + j * uc[1] + l * uc[2]
            for i, j, l in loop_iter
            if (i, j, l) != (0, 0, 0)
            for k in range(apos_len)
        ]
        r_pos_ouc.extend(neighbors)
        return np.array(r_pos_ouc)

    def rot_mat(self, atom_list, p):
        # Default identity
        return [sp.eye(3) for _ in atom_list]

    def _parse_transformations(self, p):
        """Parse transformations section to generate rotation matrices per atom."""
        logger.debug("Entering _parse_transformations")
        trans_config = self.config.get('transformations', {})
        if not trans_config:
            logger.debug("No 'transformations' found in config.")
            return None # Use default
        
        logger.debug(f"trans_config keys: {list(trans_config.keys())}")
            
        param_map = self._resolve_param_map(p)
        logger.debug(f"Param Map for Transformations: {param_map}")
        
        # 1. Variables
        variables = trans_config.get('variables', {})
        # Evaluate in order (assuming dict preserves insertion order in Py3.7+)
        for name, expr in variables.items():
            val = safe_eval(str(expr), param_map)
            param_map[name] = val
            logger.debug(f"Variable '{name}' = {val}")
        logger.debug(f"Variables evaluated: {list(param_map.keys())}")
            
        # 2. Atom Frames
        atom_frames = trans_config.get('atom_frames', [])
        logger.debug(f"atom_frames content: {atom_frames}")
        
        # Initialize with Identity
        apos = self.atom_pos()
        rot_matrices = [sp.eye(3) for _ in range(len(apos))]
        
        for frame in atom_frames:
            atom_idx = frame.get('atom')
            rot_expr = frame.get('rotation')
            
            if atom_idx is not None and rot_expr:
                mat = safe_eval(rot_expr, param_map)
                # Ensure it's a matrix?
                if isinstance(mat, (list, sp.Matrix, np.ndarray)):
                     # If it's a list of lists, convert to Matrix
                     if isinstance(mat, list):
                         mat = sp.Matrix(mat)
                     rot_matrices[atom_idx] = mat
                else:
                    # Scalar? Error.
                    raise ValueError(f"Rotation expression for atom {atom_idx} did not return a matrix.")
            
            logger.debug(f"Rotation Matrix for Atom {atom_idx}:\n{rot_matrices[atom_idx]}")
        
        return rot_matrices

    def mpr(self, p):
        # Check for optimized matrices
        if self.optimized_matrices is not None:
             logger.debug("mpr returning runtime optimized matrices.")
             return self.optimized_matrices
             
        # Check for declarative transformations
        logger.debug("Calling mpr()")
        matrices = self._parse_transformations(p)
        if matrices:
            logger.debug("mpr returning parsed matrices.")
            return matrices
            
        # Default identity
        logger.debug("mpr returning default Identity matrices.")
        return [sp.eye(3) for _ in self.atom_pos()]

    def _resolve_param_map(self, p):
        """
        Map parameter names to values in p based on 'parameters' config.
        """
        param_names = self.config.get('parameters')
        if not param_names:
            logger.debug("No 'parameters' list in config. Cannot resolve params.")
            return {} # Cannot resolve names
        
        # Ensure p is long enough
        param_map = {}
        
        # Filter out 'S' from keys since runner.py excludes it from p
        keys = [k for k in param_names if k != 'S']
        
        for i, name in enumerate(keys):
            if i < len(p):
                param_map[name] = p[i]
        return param_map

    def spin_interactions(self, p):
        """
        Generate Jex and DM matrices based on interactions config.
        """
        apos = self.atom_pos()
        N_atom = len(apos)
        apos_ouc = self.atom_pos_ouc()
        N_atom_ouc = len(apos_ouc)
        
        Jex = sp.zeros(N_atom, N_atom_ouc)
        # DM cannot be sp.zeros (matrix of scalars) if we store vectors. Use list of lists.
        DM = [[None for _ in range(N_atom_ouc)] for _ in range(N_atom)]
        # Anisotropic Exchange (Diagonal tensors for now)
        Kex = [[None for _ in range(N_atom_ouc)] for _ in range(N_atom)]
        
        dist_tol = 0.05
        
        # Resolve parameters for symbolic expressions
        param_map = self._resolve_param_map(p)
        
        param_counter = 0
        
        for interaction in self.interactions_config:
            itype = interaction.get('type')
            
            if itype == 'heisenberg':
                # Dist-based
                target_dist = interaction.get('distance')
                # If 'shell' provided, we'd need shell analysis inside here. 
                # Let's assume explicit distance for robustness or allow tolerance.
                
                # J symbol/value comes from p
                val = interaction.get('value')
                # J symbol/value comes from p
                # J_val = p[param_counter] -> REPLACED BELOW
                if isinstance(val, str) and param_map:
                    J_val = safe_eval(val, param_map)
                elif isinstance(val, (int, float)):
                    J_val = val
                else: 
                     # Fallback to positional
                     if param_counter < len(p):
                         J_val = p[param_counter]
                         param_counter += 1
                     else:
                         raise ValueError(f"Not enough parameters in p for interaction {interaction}")
                
                atom_labels = [a.get('label') for a in self.config.get('crystal_structure').get('atoms_uc')]
                target_pair = interaction.get('pair') # Optional [Label1, Label2]

                for i in range(N_atom):
                    # Check first label
                    if target_pair and atom_labels[i] != target_pair[0]:
                        continue
                        
                    for j in range(N_atom_ouc):
                        # Check second label (mapped to UC)
                        if target_pair:
                            j_uc = j % N_atom
                            if atom_labels[j_uc] != target_pair[1]:
                                continue
                                
                        d = la.norm(apos[i] - apos_ouc[j])
                        if abs(d - target_dist) < dist_tol:
                            Jex[i, j] += J_val # Additive to support multiple terms
                            
            elif itype == 'dm':
                target_dist = interaction.get('distance')
                # DM has 3 params (Dx, Dy, Dz)
                # This assumes positional splitting if not explicit? 
                # Or explicit value?
                val = interaction.get('value')
                
                if isinstance(val, list) and len(val) == 3 and param_map:
                     # Check if elements are strings
                     vals = []
                     for v in val:
                         if isinstance(v, str):
                             vals.append(safe_eval(v, param_map))
                         else:
                             vals.append(v)
                     Dx, Dy, Dz = vals
                else:
                    Dx = p[param_counter]
                    Dy = p[param_counter+1]
                    Dz = p[param_counter+2]
                    param_counter += 3
                
                D_vec = sp.Matrix([Dx, Dy, Dz])
                
                for i in range(N_atom):
                    for j in range(N_atom_ouc):
                        d = la.norm(apos[i] - apos_ouc[j])
                        if abs(d - target_dist) < dist_tol:
                             # Initialize or Add? Usually initialize. 
                             # If none, set. If exists, add? 
                             # For simplicity, overwrite or set if None.
                             if DM[i][j] is None:
                                 DM[i][j] = D_vec
                             else:
                                 DM[i][j] += D_vec
                             
            elif itype == 'dm_manual':
                # Manual entry: atom_i, atom_j, cell_j, value (list of exprs)
                i = interaction.get('atom_i')
                target_j_uc = interaction.get('atom_j')
                offset = interaction.get('offset_j', [0,0,0])
                val_exprs = interaction.get('value') # [dx_str, dy_str, dz_str]
                
                # Evaluate expressions
                d_vals = []
                for v in val_exprs:
                    if isinstance(v, str):
                        d_vals.append(safe_eval(v, param_map))
                    else:
                        d_vals.append(v)
                
                dx, dy, dz = d_vals
                D_vec = sp.Matrix([dx, dy, dz])
                
                # Find the j index in apos_ouc that matches target_j_uc + offset
                # This could be slow if many manual entries, but totally fine for N~100.
                found_j = False
                
                # Calculate target position
                target_pos = apos[target_j_uc] + offset[0]*self.unit_cell()[0] + \
                             offset[1]*self.unit_cell()[1] + offset[2]*self.unit_cell()[2]
                
                for j_idx in range(N_atom_ouc):
                    # Check distance to target pos (should be near zero)
                    if la.norm(apos_ouc[j_idx] - target_pos) < 0.001:
                        if DM[i][j_idx] is None:
                            DM[i][j_idx] = D_vec
                        else:
                            DM[i][j_idx] += D_vec
                        found_j = True
                        break
                
                if not found_j:
                     # Warn with details
                     min_dist = min([la.norm(pos - target_pos) for pos in apos_ouc])
                     print(f"WARNING: DM Interaction i={i}->j_uc={target_j_uc} offset={offset} NOT FOUND.")
                     print(f"  Target Pos: {target_pos}")

                     print(f"  Min Dist to OUC: {min_dist}")
            
            elif itype == 'anisotropic_exchange':
                target_dist = interaction.get('distance')
                val = interaction.get('value') # [Jxx, Jyy, Jzz] strings/floats
                
                k_vals = []
                # If explicit list
                if isinstance(val, list):
                     for v in val:
                         if isinstance(v, str) and param_map:
                             k_vals.append(safe_eval(v, param_map))
                         else:
                             k_vals.append(v)
                else:
                     # Fallback positional? Not implemented for array
                     raise ValueError("Anisotropic Exchange value must be a list [Jxx, Jyy, Jzz]")
                
                atom_labels = [a.get('label') for a in self.config.get('crystal_structure').get('atoms_uc')]
                target_pair = interaction.get('pair') 
                
                K_vec = sp.Matrix(k_vals)
                
                for i in range(N_atom):
                    if target_pair and atom_labels[i] != target_pair[0]:
                        continue
                        
                    for j in range(N_atom_ouc):
                        if target_pair:
                            j_uc = j % N_atom
                            if atom_labels[j_uc] != target_pair[1]:
                                continue
                                
                        d = la.norm(apos[i] - apos_ouc[j])
                        if abs(d - target_dist) < dist_tol:
                             if Kex[i][j] is None:
                                 Kex[i][j] = K_vec
                             else:
                                 Kex[i][j] += K_vec
                     
        # Fill None with zeros
        dnull = sp.Matrix([0, 0, 0])
        for i in range(N_atom):
            for j in range(N_atom_ouc):
                if DM[i][j] is None:
                    DM[i][j] = dnull
                if Kex[i][j] is None:
                    Kex[i][j] = dnull
                    
        return Jex, DM, Kex

    def Hamiltonian(self, Sxyz: List[Any], pr: List[Any]) -> sp.Expr:
        """
        Define Hamiltonian using config logic.
        """
        # Parse params
        Jex, DM, Kex, p_rest, param_map = self._parse_hamiltonian_params(pr)
        
        HM = 0
        gamma = 2.0
        mu_B = 5.788e-2
        
        # 1. Exchange Terms (Heisenberg + DM + Anisotropic)
        HM += self._compute_heisenberg_dm_terms(Sxyz, Jex, DM, Kex)
        
        # 2. Extra Terms (SIA)
        HM += self._compute_sia_terms(Sxyz, p_rest)

        # 3. Zeeman
        HM += self._compute_zeeman_terms(Sxyz, p_rest, param_map, gamma, mu_B)

        # 4. Substitution and Filtering
        HM = self._apply_substitution_and_filter(HM, pr)
        
        return HM

    def _parse_hamiltonian_params(self, pr: List[Any]) -> Tuple[Any, Any, List[Any], Dict[str, Any]]:
        """Helper to parse parameters into Jex, DM, and remaining params."""
        if self.config.get('parameters'):
            # New Named Mode
            Jex, DM, Kex = self.spin_interactions(pr)
            # Param map...
            param_map = self._resolve_param_map(pr)
            p_rest = [] 
        else:
            # Old Positional Mode
            param_counter = 0
            for interaction in self.interactions_config:
                itype = interaction.get('type')
                if itype == 'heisenberg':
                    param_counter += 1
                elif itype == 'dm':
                    param_counter += 3
            
            p_ex = pr[0:param_counter]
            p_rest = pr[param_counter:]
            Jex, DM, Kex = self.spin_interactions(p_ex)   
            param_map = {}
        
        return Jex, DM, Kex, p_rest, param_map

    def _compute_heisenberg_dm_terms(self, Sxyz: List[Any], Jex: Any, DM: Any, Kex: Any) -> sp.Expr:
        """Compute Heisenberg and Dzyaloshinskii-Moriya terms."""
        HM = 0
        apos = self.atom_pos()
        N_uc = len(apos)
        atoms_ouc = self.atom_pos_ouc()
        N_ouc = len(atoms_ouc)
        
        logger.debug(f"Hamiltonian START. N_uc={N_uc}, N_ouc={N_ouc}")
        
        terms_added = 0
        for i in range(N_uc):
            for j in range(N_ouc):
                # Heisenberg
                if Jex[i, j] != 0:
                     if terms_added < 3:
                          logger.debug(f"Adding Jex[{i},{j}] = {Jex[i,j]}")
                     
                     term_heis = 0.5 * Jex[i, j] * (
                        Sxyz[i][0] * Sxyz[j][0] + 
                        Sxyz[i][1] * Sxyz[j][1] + 
                        Sxyz[i][2] * Sxyz[j][2]
                     )
                     HM += term_heis
                     terms_added += 1
                
                # DM Interaction
                D_vec = DM[i][j]
                if D_vec is not None:
                     is_zero = False
                     if hasattr(D_vec, 'is_zero_matrix'):
                         is_zero = D_vec.is_zero_matrix
                     elif D_vec == sp.Matrix([0,0,0]):
                         is_zero = True
                         
                     if not is_zero:
                        logger.debug(f"Adding DM[{i},{j}] = {D_vec}")
                        
                        Sc_x = Sxyz[i][1]*Sxyz[j][2] - Sxyz[i][2]*Sxyz[j][1]
                        Sc_y = Sxyz[i][2]*Sxyz[j][0] - Sxyz[i][0]*Sxyz[j][2]
                        Sc_z = Sxyz[i][0]*Sxyz[j][1] - Sxyz[i][1]*Sxyz[j][0]
                        HM += 0.5 * (
                            D_vec[0] * Sc_x +
                            D_vec[1] * Sc_y +
                            D_vec[2] * Sc_z
                        )
                        terms_added += 1

                # Anisotropic Exchange (Diagonal K . S . S)
                K_vec = Kex[i][j]
                if K_vec is not None:
                     is_zero = False
                     if hasattr(K_vec, 'is_zero_matrix'):
                         is_zero = K_vec.is_zero_matrix
                     elif K_vec == sp.Matrix([0,0,0]):
                         is_zero = True
                         
                     if not is_zero:
                         # K_vec = [Kxx, Kyy, Kzz]
                         HM += 0.5 * (
                             K_vec[0] * Sxyz[i][0] * Sxyz[j][0] +
                             K_vec[1] * Sxyz[i][1] * Sxyz[j][1] +
                             K_vec[2] * Sxyz[i][2] * Sxyz[j][2]
                         )
                         terms_added += 1
        return HM

    def _compute_sia_terms(self, Sxyz: List[Any], p_rest: List[Any]) -> sp.Expr:
        """Compute Single Ion Anisotropy terms."""
        HM = 0
        N_uc = len(self.atom_pos())
        rest_idx = 0
        for interaction in self.interactions_config:
            itype = interaction.get('type')
            if itype == 'sia':
                if rest_idx < len(p_rest):
                    D_sia = p_rest[rest_idx]
                    rest_idx += 1
                    # Assume Sz^2 for now
                    for i in range(N_uc):
                         HM += D_sia * (Sxyz[i][2])**2
        return HM

    def _compute_zeeman_terms(self, Sxyz: List[Any], p_rest: List[Any], param_map: Dict[str, Any], gamma: float, mu_B: float) -> sp.Expr:
        """Compute Zeeman energy terms."""
        HM = 0
        N_uc = len(self.atom_pos())
        H_mag = None
        
        if self.config.get('parameters'):
            if 'H' in param_map:
                H_mag = param_map['H']
        else:
             # Heuristic: H is last parameter if not consumed by SIA
             # Re-counting logic needed or just trust p_rest remainder?
             # Implementation above passed p_rest. 
             # We need to know how many SIA were consumed.
             # Actually, simpler to just check if p_rest has unused items.
             # This is tricky because _compute_sia consumed items but didn't modify p_rest in place.
             # Let's count SIA items again.
             sia_count = 0
             for interaction in self.interactions_config:
                 if interaction.get('type') == 'sia':
                     sia_count += 1
             
             if len(p_rest) > sia_count:
                 H_mag = p_rest[-1]

        # Try to find vector components Hx, Hy, Hz
        Hx = param_map.get('Hx')
        Hy = param_map.get('Hy')
        Hz = param_map.get('Hz')
        
        # Check for H_dir / H_mag
        # Check for H_dir / H_mag
        H_dir = param_map.get('H_dir')
        H_mag_val = param_map.get('H_mag')
        
        # If H_dir is a Symbol (passed from MagCalc params), we can't use it as a vector directly.
        # Fallback to config value if available and valid.
        if isinstance(H_dir, sp.Symbol):
             config_params = self.config.get('parameters', self.config.get('model_params', {}))
             orig_H_dir = config_params.get('H_dir')
             if isinstance(orig_H_dir, (list, tuple, np.ndarray)) and len(orig_H_dir) == 3:
                 logger.debug(f"H_dir is symbolic ({H_dir}), using static value from config: {orig_H_dir}")
                 H_dir = orig_H_dir

        if H_dir is not None and H_mag_val is not None:
             if isinstance(H_dir, (list, tuple, np.ndarray)) and len(H_dir) == 3:
                  Hx = H_mag_val * H_dir[0]
                  Hy = H_mag_val * H_dir[1]
                  Hz = H_mag_val * H_dir[2]
        
        if Hx is not None or Hy is not None or Hz is not None:
             # Vector Zeeman
             # Handle missing components as 0
             if Hx is None: Hx = 0
             if Hy is None: Hy = 0
             if Hz is None: Hz = 0
             
             for i in range(N_uc):
                 term = Hx*Sxyz[i][0] + Hy*Sxyz[i][1] + Hz*Sxyz[i][2]
                 HM += gamma * mu_B * term
                 
        elif H_mag is not None:
             # Check if vector (legacy check for runtime list passing, though discourage)
             is_vector = isinstance(H_mag, (list, tuple, np.ndarray))
             
             for i in range(N_uc):
                  if is_vector and len(H_mag) == 3:
                      # Dot product
                      term = H_mag[0]*Sxyz[i][0] + H_mag[1]*Sxyz[i][1] + H_mag[2]*Sxyz[i][2]
                      HM += gamma * mu_B * term
                  else:
                      # Scalar - assume Z
                      HM += gamma * mu_B * Sxyz[i][2] * H_mag
        
        return HM

    def _apply_substitution_and_filter(self, HM: sp.Expr, pr: List[Any]) -> sp.Expr:
        """Substitute numerical parameters and filter for quadratic terms."""
        if self.config.get('model_params'):
            p_names = self.config.get('parameters', [])
            p_values_dict = self.config.get('model_params', {})
            
            subs_map = {}
            for i, p_sym in enumerate(pr):
                if i < len(p_names):
                    name = p_names[i]
                    if name in p_values_dict:
                         subs_map[p_sym] = p_values_dict[name]
            
            free_syms = list(HM.free_symbols)
            for sym in free_syms:
                if sym.name in p_values_dict:
                     if sym not in subs_map:
                          subs_map[sym] = p_values_dict[sym.name]
            
            if subs_map:
                logger.debug(f"Substituting parameters with values: {subs_map}")
                HM = HM.subs(subs_map)

        logger.debug("Using HM.expand()")
        HM = HM.expand()
        
        # Check if we are doing LSWT (presence of 'c' operators) or Classical (no 'c')
        # If any symbol starts with 'c', we assume LSWT and filter for quadratic terms.
        # Otherwise, return full Hamiltonian (classical energy).
        # Note: 'c' prefix is standard for bosonic operators in this codebase.
        
        has_c_ops = False
        all_syms = HM.free_symbols
        for s in all_syms:
            if s.name.startswith('c'):
                has_c_ops = True
                break
        
        if has_c_ops:
             if hasattr(HM, 'as_ordered_terms'):
                  terms = HM.as_ordered_terms()
                  kept = []
                  for term in terms:
                      syms = term.atoms(sp.Symbol)
                      pow_dict = term.as_powers_dict()
                      degree = 0
                      for s in syms:
                          if s.name.startswith('c'):
                              degree += pow_dict.get(s, 0)
                      
                      if degree == 2:
                          kept.append(term)
                  
                  if len(kept) == 0:
                      logger.warning("All terms filtered out! HM will be zero.")
                      
                  HM = sp.Add(*kept)
        else:
             logger.debug("No bosonic 'c' operators found. Skipping LSWT filtering (Classical Mode).")
             
        return HM

    def minimize_energy(self, p_num):
        """
        Perform classical energy minimization to find the magnetic ground state.
        Updates self.optimized_matrices with the Rotation Matrices for the ground state.
        """
        min_config = self.config.get('minimization', {})
        if not min_config or not min_config.get('enabled', False):
            logger.debug("Minimization not enabled in config. Skipping.")
            return

        logger.info("Minimization Enabled. Finding classical ground state...")
        
        # Get numerical interaction matrices
        Jex_sym, DM_sym, Kex_sym = self.spin_interactions(p_num)
        
        # Convert Jex to numpy float array
        if hasattr(Jex_sym, 'tolist'):
            Jex = np.array(Jex_sym.tolist(), dtype=float)
        else:
            # Fallback list of lists
            Jex = np.array(Jex_sym, dtype=float)
            
        N = Jex.shape[0]
        N_ouc = Jex.shape[1]
        
        # Convert DM to numpy (N, N_ouc, 3)
        DM = np.zeros((N, N_ouc, 3))
        for i in range(N):
            for j in range(N_ouc):
                val = DM_sym[i][j]
                if val is not None:
                    if hasattr(val, 'tolist'): val = np.array(val.tolist(), dtype=float).flatten()
                    elif hasattr(val, 'evalf'): val = np.array(val.evalf()).flatten().astype(float)
                    DM[i, j] = val
                    
        # Convert Kex to numpy (N, N_ouc, 3)
        Kex = np.zeros((N, N_ouc, 3))
        for i in range(N):
            for j in range(N_ouc):
                val = Kex_sym[i][j]
                if val is not None:
                     if hasattr(val, 'tolist'): val = np.array(val.tolist(), dtype=float).flatten()
                     Kex[i, j] = val

        # Prepare params
        params_dict = self._resolve_param_map(p_num)
        
        # Determine H vector if present (Zeeman)
        H_vec = None
        
        if 'Hx' in params_dict or 'Hy' in params_dict or 'Hz' in params_dict:
             hx = float(params_dict.get('Hx', 0.0))
             hy = float(params_dict.get('Hy', 0.0))
             hz = float(params_dict.get('Hz', 0.0))
             H_vec = np.array([hx, hy, hz])
             
        elif 'H' in params_dict:
            h_val = params_dict['H']
            # Assume H along z if scalar, or check if vector
            if isinstance(h_val, (list, tuple, np.ndarray)):
                 H_vec = np.array(h_val, dtype=float)
            else:
                 # Scalar H. Assume Z-axis by default
                 H_vec = np.array([0.0, 0.0, float(h_val)])
        else:
             # Heuristic: last param? Or from p_rest?
             # GenericSpinModel logic for H is complex. 
             # Let's rely on 'H' being in params dict.
             pass

        S_val = float(params_dict.get('S', 0.5))
        logger.info(f"Minimize Energy using S_val={S_val}")
             
        # Initial guess 
        # Random or aligned?
        # Use initial guess from config if provided?
        # Or start with Ferromagnetic along Z (or some direction)
        N_atom = len(self.atom_pos())
        
        # x0: [theta_0, phi_0, theta_1, phi_1, ...]
        # Default start: theta=pi/2, phi=0 (in plane)
        # Add random perturbation to avoid getting stuck in high-symmetry stationary points (like FM for AFM model)
        x0 = np.zeros(2 * N_atom)
        rng = np.random.default_rng(seed=42) # Fixed seed for reproducibility
        
        # Smart Initial Guess: align against field (for electrons)
        # If H_vec is present and non-zero, set initial theta/phi to point opposite to H.
        if H_vec is not None and np.linalg.norm(H_vec) > 1e-4:
             # Direction of H
             h_dir = H_vec / np.linalg.norm(H_vec)
             # Target spin direction S = -h_dir
             s_target = -h_dir
             # Convert to theta, phi
             # z = cos(theta) -> theta = acos(z)
             # x = sin(theta)cos(phi), y = sin(theta)sin(phi) -> phi = atan2(y, x)
             t_target = np.arccos(s_target[2])
             p_target = np.arctan2(s_target[1], s_target[0])
             
             for i in range(N_atom):
                 # Add small noise
                 x0[2*i] = t_target + rng.normal(0, 0.1)
                 x0[2*i+1] = p_target + rng.normal(0, 0.1)
        else:
             # Random around XY plane
             for i in range(N_atom):
                 x0[2*i] = np.pi/2.0 + rng.normal(0, 0.2) 
                 x0[2*i+1] = rng.uniform(0, 2*np.pi)
            
        # Optimization
        method = min_config.get('method', 'L-BFGS-B')
        ftol = min_config.get('ftol', 1e-9)
        maxiter = min_config.get('maxiter', 5000)
        
        # Bounds: theta [0, pi], phi [-inf, inf] (or [0, 2pi])
        bounds = []
        for i in range(N_atom):
            bounds.append((0, np.pi))
            bounds.append((None, None))
            
        args = (Jex, DM, Kex, H_vec, S_val)
        
        with tqdm(total=maxiter, desc="Minimizing Energy", leave=False) as pbar:
             def callback(xk):
                 pbar.update(1)
                 
             res = minimize(
                self._classical_energy_func, 
                x0, 
                args=args, 
                method=method, 
                bounds=bounds,
                tol=ftol,
                options={'maxiter': maxiter},
                callback=callback
             )
        
        if res.success:
            logger.info(f"Minimization converged. Energy: {res.fun:.6f}")
        else:
            logger.warning(f"Minimization did not converge: {res.message}")
            
        # Construct Rotation Matrices
        opt_angles = res.x
        self.optimized_matrices = []
        for i in range(N_atom):
            th = opt_angles[2*i]
            ph = opt_angles[2*i+1]
            
            # Construct Rotation Matrix R such that R * z_local = S_global
            # S_global = [sin(th)cos(ph), sin(th)sin(ph), cos(th)]
            
            # Standard construction: Y-Z-Y convention or similar
            # GenericSpinModel expects a matrix R.
            # R should rotate [0,0,1] to [Sx, Sy, Sz].
            # Common choice:
            # R = Rz(phi) * Ry(theta)
            # R * [0,0,1] 
            # Ry(theta) * z = [sin(th), 0, cos(th)]
            # Rz(phi) * [sin, 0, cos] = [cos(ph)sin(th), sin(ph)sin(th), cos(th)] -> Correct.
            
            # Ry(theta) = [[cos(th), 0, sin(th)], [0, 1, 0], [-sin(th), 0, cos(th)]]
            # Rz(phi) = [[cos(ph), -sin(ph), 0], [sin(ph), cos(ph), 0], [0, 0, 1]]
            
            # Using sp.Matrix for internal consistency with symbolic parts (though these are numeric)
            ct, st = np.cos(th), np.sin(th)
            cp, sp_ = np.cos(ph), np.sin(ph)
            
            Ry = sp.Matrix([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
            Rz = sp.Matrix([[cp, -sp_, 0], [sp_, cp, 0], [0, 0, 1]])
            
            
            R = Rz * Ry
            self.optimized_matrices.append(R)
            
        logger.info("Optimized rotation matrices updated.")

    def set_magnetic_structure(self, thetas, phis):
        """
        Update the magnetic structure (rotation matrices) from given angles.
        Args:
            thetas (array-like): Theta angles (radians), length N_atom
            phis (array-like): Phi angles (radians), length N_atom
        """
        N_atom = len(self.atom_pos())
        if len(thetas) != N_atom or len(phis) != N_atom:
            raise ValueError(f"Length of angles ({len(thetas)}, {len(phis)}) must match N_atom ({N_atom})")

        self.optimized_matrices = []
        for i in range(N_atom):
            th = float(thetas[i])
            ph = float(phis[i])
            
            # Construct Rotation Matrix R such that R * z_local = S_global
            # Using same convention as minimize_energy: R = Rz(phi) * Ry(theta)
            
            ct, st = np.cos(th), np.sin(th)
            cp, sp_ = np.cos(ph), np.sin(ph)
            
            # Use sympy matrices if downstream expects them, or numpy if supported.
            # Using sympy for consistency.
            Ry = sp.Matrix([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
            Rz = sp.Matrix([[cp, -sp_, 0], [sp_, cp, 0], [0, 0, 1]])
            
            R = Rz * Ry
            self.optimized_matrices.append(R)
            
        logger.info("GenericSpinModel: Magnetic structure updated via set_magnetic_structure.")


    def generate_magnetic_structure(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Generate magnetic structure (theta, phi) based on 'magnetic_structure' in config.
        Returns:
            (thetas, phis): Lists of angles, or (None, None) if config missing.
        """
        struct_config = self.config.get('magnetic_structure')
        if not struct_config:
            return None, None

        method = struct_config.get('type')
        if method == 'explicit':
            return self._generate_structure_explicit(struct_config)
        elif method == 'propagation_vector':
            return self._generate_structure_from_k(struct_config)
        elif method == 'pattern':
            return self._generate_structure_from_pattern(struct_config)
        else:
             logger.warning(f"Unknown magnetic_structure type: {method}")
             return None, None

    def _generate_structure_explicit(self, config):
        """Parse explicit angle list."""
        # Use same logic as runner.py legacy parser but cleaner
        atoms = self.atom_pos()
        nspins = len(atoms)
        thetas = [0.0] * nspins
        phis = [0.0] * nspins
        
        entries = config.get('explicit_list', config.get('configuration', []))
        for item in entries:
             idx = item.get('atom_index')
             if idx is not None and 0 <= idx < nspins:
                 if 'theta' in item: thetas[idx] = float(item['theta'])
                 if 'phi' in item: phis[idx] = float(item['phi'])
        return thetas, phis

    def _generate_structure_from_pattern(self, config):
        """Parse high-level pattern."""
        pattern = config.get('pattern_type')
        atoms = self.atom_pos()
        nspins = len(atoms)
        thetas = [0.0] * nspins
        phis = [0.0] * nspins
        
        if pattern == 'ferromagnetic':
            # Direction vector or angles
            direction = config.get('direction', [0, 0, 1]) # Default z
            # Convert vector to angles
            # Simplistic conversion
            v = np.array(direction, dtype=float)
            norm = np.linalg.norm(v)
            if norm > 1e-9:
                v /= norm
                # th = acos(z), ph = atan2(y, x)
                th = np.arccos(v[2])
                ph = np.arctan2(v[1], v[0])
                thetas = [float(th)] * nspins
                phis = [float(ph)] * nspins
        elif pattern == 'antiferromagnetic':
             # Need sublattices
             # Simplest: list of directions applied cyclically or by index mapping
             directions = config.get('directions', [])
             if not directions:
                 # Default Neel for 2 sublattices?
                 directions = [[0, 0, 1], [0, 0, -1]]
             
             for i in range(nspins):
                 d = directions[i % len(directions)]
                 v = np.array(d, dtype=float)
                 norm = np.linalg.norm(v)
                 if norm > 1e-9:
                      v /= norm
                      th = np.arccos(v[2])
                      ph = np.arctan2(v[1], v[0])
                      thetas[i] = float(th)
                      phis[i] = float(ph)
        
        return thetas, phis

    def _generate_structure_from_k(self, config):
        """Generate spiral from propagation vector."""
        k_vec = np.array(config.get('k', [0, 0, 0]), dtype=float)
        # Type: 'planar', 'conical', 'complex'
        stype = config.get('subtype', 'planar')
        
        # Basis vectors u, v (and normal n for conical?)
        # Default planar in ab plane: u=[1,0,0], v=[0,1,0]
        u_vec = np.array(config.get('u', [1, 0, 0]), dtype=float)
        v_vec = np.array(config.get('v', [0, 1, 0]), dtype=float)
        
        atoms = self.atom_pos()
        thetas = []
        phis = []
        
        for i, pos in enumerate(atoms):
            phase = np.dot(k_vec, pos) # k . r
            
            if stype == 'planar':
                # S = u * cos(phase) + v * sin(phase)
                # Assumes u, v orthogonal and S_mag matches
                S_vec = u_vec * np.cos(phase) + v_vec * np.sin(phase)
            elif stype == 'conical':
                 # Add offset component?
                 # n + u cos + v sin
                 n_vec = np.array(config.get('n', [0, 0, 1]), dtype=float)
                 cone_angle = np.radians(config.get('cone_angle_deg', 0)) # 0 = flat?
                 # Interpretation: S = n * cos(cone) + (u cos(ph) + v sin(ph)) * sin(cone)
                 # Wait, usually cone angle is deviation from n.
                 # Let's assume standard cons.
                 S_vec = n_vec * np.cos(cone_angle) + (u_vec * np.cos(phase) + v_vec * np.sin(phase)) * np.sin(cone_angle)
            else:
                S_vec = np.array([0, 0, 1])

            # Normalize to safe guard
            norm = np.linalg.norm(S_vec)
            if norm > 1e-9:
                S_vec /= norm
                
            th = np.arccos(S_vec[2])
            ph = np.arctan2(S_vec[1], S_vec[0])
            thetas.append(float(th))
            phis.append(float(ph))
            
        return thetas, phis
    def _classical_energy_func(self, x, Jex, DM, Kex, H_vec, S_val=0.5):
        """
        Calculate total classical energy for angles x.
        x = [th0, ph0, th1, ph1, ...]
        Assumes Jex, DM, Kex are numpy arrays.
        """
        N = Jex.shape[0]
        N_ouc = Jex.shape[1]
        
        # Vectorized S construction
        theta = x[0::2]
        phi = x[1::2]
        
        # S_vecs_uc: (N, 3)
        st, ct = np.sin(theta), np.cos(theta)
        sp_, cp = np.sin(phi), np.cos(phi)
        
        S_vecs_uc = np.stack([st*cp, st*sp_, ct], axis=1)
        
        # Expand to OUC
        # Assuming OUC maps j -> j%N
        # (N_ouc, 3)
        if N_ouc > N:
            # Create full array
             indices = np.arange(N_ouc) % N
             S_vecs = S_vecs_uc[indices]
        else:
             S_vecs = S_vecs_uc
            
        E = 0.0
        
        # Optimized Energy summation
        # We can just iterate.
        
        # Zeeman
        if H_vec is not None:

            # E += gamma * mu * sum(S . H)
            gamma = 2.0
            mu_B = 5.788e-2
            # Sum S . H for all i in UC
            # Using vectorized dot. Scale by S_val
            E += gamma * mu_B * np.sum(S_vecs_uc @ H_vec) * S_val

        # Interactions
        for i in range(N):
             Si = S_vecs[i]
             
             for j in range(N_ouc):
                  Sj = S_vecs[j]
                  
                  # Heisenberg
                  J_val = Jex[i, j]
                  if J_val != 0:
                       E += 0.5 * J_val * np.dot(Si, Sj) * (S_val * S_val)
                       
                  # DM
                  D_vec = DM[i, j]
                  # Check norm > epsilon
                  if abs(D_vec[0]) > 1e-9 or abs(D_vec[1]) > 1e-9 or abs(D_vec[2]) > 1e-9:
                       # D . (Si x Sj)
                       E += 0.5 * np.dot(D_vec, np.cross(Si, Sj)) * (S_val * S_val)
                       
                  # Anisotropic (Kex)
                  K_vec = Kex[i, j]
                  if abs(K_vec[0]) > 1e-9 or abs(K_vec[1]) > 1e-9 or abs(K_vec[2]) > 1e-9:
                       term = K_vec[0]*Si[0]*Sj[0] + K_vec[1]*Si[1]*Sj[1] + K_vec[2]*Si[2]*Sj[2]
                       E += 0.5 * term * (S_val * S_val)
                       
        return E
