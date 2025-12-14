import logging
import os
import sys
from itertools import product
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
from numpy import linalg as la
import sympy as sp

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
        Sets self._uc_vectors and self._r_pos.
        """
        crystal_struct = self.config.get('crystal_structure', {})
        
        # 1. Explicit Definition
        if 'lattice_vectors' in crystal_struct and 'atom_positions' in crystal_struct:
            self._uc_vectors = np.array(crystal_struct['lattice_vectors'], dtype=float)
            self._r_pos = np.array(crystal_struct['atom_positions'], dtype=float)
            # Create a dummy atoms object for ASE compatibility if needed? 
            # Not strictly needed if we don't access self.atoms elsewhere.
            self.atoms = None 
            return

        # 2. CIF File
        cif_file = crystal_struct.get('cif_file')
        if cif_file:
            if not os.path.isabs(cif_file) and self.base_path:
                cif_file = os.path.join(self.base_path, cif_file)
            
            if read_cif is None:
                raise ImportError("ASE not installed or read_cif import failed.")
                
            try:
                # Use cif_utils to handle scaling and selection if available
                # Or just basic read
                self.atoms = read_cif(cif_file)
                
                # Filter by magnetic elements
                mag_elements = crystal_struct.get('magnetic_elements')
                if mag_elements:
                     self.atoms = self.atoms[
                         [atom.symbol in mag_elements for atom in self.atoms]
                     ]
                
                self._uc_vectors = self.atoms.cell[:]
                self._r_pos = self.atoms.get_positions()
                
            except Exception as e:
                raise ValueError(f"Failed to load CIF structure from {cif_file}: {e}")
        else:
             raise ValueError("No crystal structure defined (lattice_vectors+atom_positions or cif_file).")

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
        for i, name in enumerate(param_names):
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
                
                for i in range(N_atom):
                    for j in range(N_atom_ouc):
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
                dx = safe_eval(val_exprs[0], param_map)
                dy = safe_eval(val_exprs[1], param_map)
                dz = safe_eval(val_exprs[2], param_map)
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
                     
        # Fill None with zeros
        dnull = sp.Matrix([0, 0, 0])
        for i in range(N_atom):
            for j in range(N_atom_ouc):
                if DM[i][j] is None:
                    DM[i][j] = dnull
                    
        return Jex, DM

    def Hamiltonian(self, Sxyz: List[Any], pr: List[Any]) -> sp.Expr:
        """
        Define Hamiltonian using config logic.
        """
        # Parse params
        Jex, DM, p_rest, param_map = self._parse_hamiltonian_params(pr)
        
        HM = 0
        gamma = 2.0
        mu_B = 5.788e-2
        
        # 1. Exchange Terms (Heisenberg + DM)
        HM += self._compute_heisenberg_dm_terms(Sxyz, Jex, DM)
        
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
            Jex, DM = self.spin_interactions(pr)
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
            Jex, DM = self.spin_interactions(p_ex)   
            param_map = {}
            
        return Jex, DM, p_rest, param_map

    def _compute_heisenberg_dm_terms(self, Sxyz: List[Any], Jex: Any, DM: Any) -> sp.Expr:
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

        if H_mag is not None:
             for i in range(N_uc):
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
             
        return HM
