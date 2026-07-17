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

from .stevens import stevens_polynomial

# mu0 * muB^2 / (4*pi), in meV * Angstrom^3. Cross-checked against Sunny 0.8.1,
# whose Units(:meV, :angstrom).vacuum_permeability = 0.6745817653 is mu0*muB^2
# (no 4pi): 0.6745817653 / (4*pi) = 0.05368216.
DIPOLE_PREFACTOR_MEV_A3 = 0.05368216

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

def rotation_about_axis(theta, axis):
    """Numeric Rodrigues rotation matrix: rotate by ``theta`` (rad) about ``axis``."""
    n = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError("Rotation axis must be non-zero.")
    n = n / norm
    K = np.array([
        [0.0, -n[2], n[1]],
        [n[2], 0.0, -n[0]],
        [-n[1], n[0], 0.0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def spiral_propagation_case(k, tol=1e-8):
    """Classify a propagation vector like Sunny's spiral_propagation_case.

    Returns 1 if every component of k is an integer, 2 if every component of
    2k is an integer (k = 1/2-type), else 3 (generic, incommensurate).
    """
    k = np.asarray(k, dtype=float)
    if np.linalg.norm(k - np.round(k)) < tol:
        return 1
    if np.linalg.norm(2 * k - np.round(2 * k)) < 2 * tol:
        return 2
    return 3


def resolve_supercell_dims(spec, k_rlu=None, max_size=100, tol=1e-6):
    """Resolve a magnetic_supercell spec to integer dims [n1, n2, n3].

    Accepts [n1, n2, n3], {'matrix': [n1, n2, n3]}, or 'auto'. 'auto' derives
    the minimal diagonal supercell commensurate with the propagation vector k
    (per-component denominator, like the diagonal case of Sunny's
    suggest_magnetic_supercell); it raises if k is incommensurate within
    max_size.

    `k_rlu` may be a single k, or a LIST of k vectors (a multi-k structure): the
    cell must then be commensurate with EVERY k, so each axis takes the least
    common multiple of the per-k denominators.
    """
    from fractions import Fraction
    from math import lcm

    if isinstance(spec, dict):
        spec = spec.get('matrix', spec.get('dims'))
    if isinstance(spec, str):
        if spec != 'auto':
            raise ValueError(f"Unknown magnetic_supercell spec: {spec!r}")
        if k_rlu is None:
            raise ValueError("magnetic_supercell: 'auto' requires a "
                             "magnetic_structure with a propagation vector k.")
        k_arr = np.atleast_2d(np.asarray(k_rlu, dtype=float))
        dims = [1, 1, 1]
        for k_vec in k_arr:
            for axis, comp in enumerate(k_vec):
                frac = Fraction(comp).limit_denominator(max_size)
                if abs(float(frac) - comp) > tol:
                    raise ValueError(
                        f"magnetic_supercell: 'auto' — k component {comp} is "
                        f"incommensurate (no denominator <= {max_size}); use the "
                        "rotating-frame single_k mode instead."
                    )
                dims[axis] = lcm(dims[axis], frac.denominator)
        if any(d > max_size for d in dims):
            raise ValueError(
                f"magnetic_supercell: 'auto' derived {dims}, which exceeds "
                f"max_size={max_size}. The k vectors are mutually near-incommensurate."
            )
        return [int(d) for d in dims]
    dims = [int(d) for d in spec]
    if len(dims) != 3 or any(d < 1 for d in dims):
        raise ValueError(f"magnetic_supercell must be three integers >= 1, got {spec}")
    return dims


def supercell_site_label(label, cell):
    """Label of a replicated site. Cell (0,0,0) keeps the original label."""
    i, j, l = cell
    if (i, j, l) == (0, 0, 0):
        return label
    return f"{label}@{i}_{j}_{l}"


def interactions_to_numpy(Jex_sym, DM_sym, Kex_sym):
    """Convert (possibly symbolic) interaction matrices to numeric arrays.

    Returns (Jex (N, N_ouc), DM (N, N_ouc, 3), Kex (N, N_ouc, 3, 3)).
    """
    if hasattr(Jex_sym, 'tolist'):
        Jex = np.array(Jex_sym.tolist(), dtype=float)
    else:
        Jex = np.array(Jex_sym, dtype=float)

    N, N_ouc = Jex.shape

    DM = np.zeros((N, N_ouc, 3))
    for i in range(N):
        for j in range(N_ouc):
            val = DM_sym[i][j]
            if val is not None:
                if hasattr(val, 'tolist'):
                    val = np.array(val.tolist(), dtype=float).flatten()
                elif hasattr(val, 'evalf'):
                    val = np.array(val.evalf()).flatten().astype(float)
                DM[i, j] = val

    Kex = np.zeros((N, N_ouc, 3, 3))
    for i in range(N):
        for j in range(N_ouc):
            val = Kex_sym[i][j]
            if val is not None:
                if hasattr(val, 'tolist'):
                    val = np.array(val.tolist(), dtype=float)
                elif hasattr(val, 'evalf'):
                    val = np.array(val.evalf().tolist(), dtype=float)
                val_flat = val.flatten()
                if val_flat.size == 9:
                    Kex[i, j] = val_flat.reshape(3, 3)
                elif val_flat.size == 3:
                    Kex[i, j] = np.diag(val_flat)
                else:
                    logger.warning(f"Unexpected Kex size {val_flat.size} at ({i},{j})")
    return Jex, DM, Kex


_LEGACY_MS_WARNED = set()


def normalize_magnetic_structure(ms_cfg, quiet=False):
    """Normalize the 'magnetic_structure' config onto the unified 'single_k' form.

    - 'spiral' (legacy rotating-frame) -> 'single_k' (pure rename: axis/n,
      local_directions pass through unchanged).
    - 'propagation_vector' (legacy real-space generator) -> 'single_k' with
      real_space: true (angles are generated in the lab frame from the u/v
      basis with the corrected 2*pi*k.r_frac phases; interactions are NOT
      rotated, preserving the legacy semantics).
    - 'single_k' passes through with defaults filled ('axis' from alias 'n').
    - Other types ('explicit', 'pattern', ...) pass through untouched.
    """
    if not ms_cfg or not isinstance(ms_cfg, dict):
        return {}
    cfg = dict(ms_cfg)
    mtype = cfg.get('type')

    if mtype == 'spiral':
        if 'spiral' not in _LEGACY_MS_WARNED:
            _LEGACY_MS_WARNED.add('spiral')
            logger.warning(
                "magnetic_structure type 'spiral' is deprecated; use "
                "'single_k' (same fields; 'axis' unchanged). Proceeding with "
                "the unified single_k path."
            )
        cfg['type'] = 'single_k'
    elif mtype == 'propagation_vector':
        if 'propagation_vector' not in _LEGACY_MS_WARNED:
            _LEGACY_MS_WARNED.add('propagation_vector')
            logger.warning(
                "magnetic_structure type 'propagation_vector' is deprecated; "
                "use 'single_k'. Mapping to single_k with real_space: true "
                "(lab-frame angles from the u/v basis; note the phase "
                "convention is now the corrected 2*pi*k.r_frac)."
            )
        cfg['type'] = 'single_k'
        if cfg.get('real_space') is None:
            cfg['real_space'] = True
        if cfg.get('u') is None:
            cfg['u'] = [1, 0, 0]
        if cfg.get('v') is None:
            cfg['v'] = [0, 1, 0]

    if cfg.get('type') == 'single_k':
        # Treat explicit None (e.g. from pydantic model_dump) as missing.
        if cfg.get('axis') is None:
            cfg['axis'] = cfg['n'] if cfg.get('n') is not None else [0, 0, 1]
        if cfg.get('k') is None:
            cfg['k'] = [0, 0, 0]
        k_case = spiral_propagation_case(cfg['k'])
        cfg['k_case'] = k_case
        if k_case == 2 and not cfg.get('real_space', False) and not quiet:
            logger.warning(
                "single_k structure with 2k integer (k = 1/2-type, k=%s): the "
                "helical description may double count a collinear structure; "
                "consider an explicit magnetic supercell if the structure is "
                "collinear.", cfg['k']
            )
    return cfg


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
    try:
        from sympy.parsing.sympy_parser import parse_expr
        # Using parse_expr instead of eval for security against arbitrary code execution
        return parse_expr(str(expr), local_dict=allowed_names)
    except Exception as e:
        logger.error(f"Failed to safely evaluate expression '{expr}': {e}")
        raise ValueError(f"Invalid mathematical expression: {expr}")

class GenericSpinModel:
    """
    A generic spin model that implements the required interface (unit_cell, atom_pos, Hamiltonian)
    by reading from a configuration dictionary.
    
    This class acts as a bridge between the declarative YAML configuration and the 
    MagCalc calculation engine.
    
    Attributes:
        config (Dict): The configuration dictionary.
        atoms (Optional[ase.Atoms]): ASE atoms object if loaded from CIF.
    """
    def __init__(self, config, base_path="."):
        self.config = config
        self.base_path = base_path

        # A declarative model needs `crystal_structure` at the TOP level. Fail here with
        # an actionable message rather than crashing later with a bare
        # KeyError('crystal_structure') deep inside expansion. The usual causes:
        #   * the config keys are nested one level down (e.g. everything under a
        #     `cvo_model:` wrapper) -- a fragment meant to be embedded, not run;
        #   * it is a LEGACY python-model config that belongs at the runner level via
        #     `python_model_file:` / `spin_model_module:` (those never reach here).
        if not isinstance(config, dict) or 'crystal_structure' not in config:
            top = list(config.keys()) if isinstance(config, dict) else type(config).__name__
            raise ValueError(
                "Config has no top-level `crystal_structure`. A declarative model needs "
                "`crystal_structure` (lattice + atoms) at the top level. "
                f"Found top-level keys: {top}. If your model keys are nested under another "
                "key, un-nest them; if this is a legacy `python_model_file` config, that is "
                "handled by the runner, not GenericSpinModel.")

        # We always attempt in-place expansion to standardize.
        self._expand_config_inplace()

        # Magnetic supercell (SpinW nExt / Sunny resize_supercell analogue):
        # replicate the chemical cell and remap interactions + magnetic
        # structure. Must run after rule expansion (explicit pair/offset
        # bonds) and before structure loading.
        self.supercell_dims = [1, 1, 1]
        self._apply_magnetic_supercell()

        self.crystal_config = self.config.get('crystal_structure', {})
        self.interactions_config = self.config.get('interactions', [])
        self.optimized_matrices = None
        self.parameter_order = self.config.get('parameter_order', [])
        
        # Ion list for form factors -- must be initialized BEFORE _load_structure,
        # which populates it from the atoms. (It used to be reset to [] AFTER the
        # structure load, silently dropping the magnetic form factor from every
        # intensity calculation -- caught because the Cu5SbO6 powder map carried
        # far too much weight at high |Q| compared to PRR 8, 013247 Fig. 5.)
        self._ion_list = []

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

        # Normalized magnetic-structure config (legacy 'spiral' /
        # 'propagation_vector' types are mapped onto the unified 'single_k'
        # form). All downstream code reads this instead of the raw dict.
        self.mag_struct_cfg = normalize_magnetic_structure(
            self.config.get('magnetic_structure')
        )

    @property
    def is_single_k(self):
        """True when the model carries a single-k (propagation-vector) structure."""
        return (self.mag_struct_cfg.get('type') == 'single_k'
                and self.mag_struct_cfg.get('enabled', True))

    @property
    def use_rotating_frame(self):
        """True when interactions must be transformed into the rotating frame."""
        return self.is_single_k and not self.mag_struct_cfg.get('real_space', False)

    @property
    def k_rlu(self):
        """Propagation vector in RLU, or None if not a single-k structure."""
        if not self.is_single_k:
            return None
        return np.asarray(self.mag_struct_cfg.get('k', [0, 0, 0]), dtype=float)

    @property
    def spiral_axis(self):
        """Unit rotation axis (normal to the polarization plane), or None."""
        if not self.is_single_k:
            return None
        axis = np.asarray(self.mag_struct_cfg.get('axis', [0, 0, 1]), dtype=float)
        norm = np.linalg.norm(axis)
        return axis / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])

    @property
    def k_case(self):
        """Sunny-style propagation case (1: k integer, 2: 2k integer, 3: generic)."""
        if not self.is_single_k:
            return None
        return self.mag_struct_cfg.get('k_case',
                                       spiral_propagation_case(self.k_rlu))

    def _expand_config_inplace(self):
        """
        Use MagCalcConfigBuilder to expand Wyckoff atoms and symmetry rules.
        This allows direct 'magcalc run' on designer-exported YAML files.
        """
        try:
            from .config_builder import MagCalcConfigBuilder
        except ImportError:
            # Fallback for if relative import fails
            try:
                from magcalc.config_builder import MagCalcConfigBuilder
            except ImportError:
                logger.warning("MagCalcConfigBuilder not found. Automatic expansion skipped.")
                return

        logger.info("GenericSpinModel: Expanding configuration via MagCalcConfigBuilder...")
        builder = MagCalcConfigBuilder()
        
        # 1. Lattice & Dimensionality
        crystal_struct = self.config.get('crystal_structure', {})
        lattice = crystal_struct.get('lattice_parameters', {})
        if lattice:
             builder.set_lattice(
                a=lattice.get('a', 1.0),
                b=lattice.get('b', 1.0),
                c=lattice.get('c', 1.0),
                alpha=lattice.get('alpha', 90.0),
                beta=lattice.get('beta', 90.0),
                gamma=lattice.get('gamma', 90.0),
                space_group=int(lattice.get('space_group')) if lattice.get('space_group') else None
             )
        elif 'lattice_vectors' in crystal_struct:
             builder.lattice_vectors = np.array(crystal_struct['lattice_vectors'], dtype=float)
        elif 'unit_cell_vectors' in crystal_struct:
             builder.lattice_vectors = np.array(crystal_struct['unit_cell_vectors'], dtype=float)
             
        builder.dimensionality = crystal_struct.get('dimensionality', '3D')
        
        # 2. Atoms
        wyckoff_atoms = crystal_struct.get('wyckoff_atoms', [])
        atom_mode = crystal_struct.get('atom_mode', 'symmetry')
        
        if atom_mode == 'explicit':
             # Just pass them through
             builder.atoms_uc = wyckoff_atoms
             builder.config['crystal_structure']['atoms_uc'] = wyckoff_atoms
        elif not wyckoff_atoms and 'atoms_uc' in crystal_struct:
             builder.atoms_uc = crystal_struct['atoms_uc']
             # Ensure builder knows species/labels for interaction matching
             builder._atom_label_to_idx = {
                 atom['label']: i for i, atom in enumerate(builder.atoms_uc)
             }
        else:
             for atom in wyckoff_atoms:
                 builder.add_wyckoff_atom(
                     label=atom.get('label', 'Atom'),
                     pos=atom.get('pos', [0, 0, 0]),
                     spin=atom.get('spin_S', 0.5),
                     species=atom.get('element', atom.get('label', 'Atom'))
                 )
                 
        # 3. Interactions
        interactions_data = self.config.get('interactions', {})
        
        # Pre-load manual interactions into builder to ensure they persist
        # and can be expanded if needed or merged.
        if isinstance(interactions_data, dict):
             valid_inter_keys = [
                 "heisenberg", "dm_interaction", "single_ion_anisotropy",
                 "anisotropic_exchange", "interaction_matrix", "kitaev"
             ]
             for key in valid_inter_keys:
                 if key in interactions_data and isinstance(interactions_data[key], list):
                     # Copy to builder
                     # We use slicing [:] or list() to copy to avoid shared reference if builder modifies it?
                     # Builder usually appends.
                     builder.config["interactions"][key] = list(interactions_data[key])

        if isinstance(interactions_data, dict):
             # Symmetry Rules Mode
             if 'symmetry_rules' in interactions_data:
                 # Check if we need to detect symmetry first
                 if not builder.space_group_number and builder.atoms_uc:
                      logger.info("Detecting symmetry for interaction expansion...")
                      builder.detect_symmetry_from_structure()

                 rules = interactions_data.get('symmetry_rules', [])
                 for rule in rules:
                     # A rule that cannot be applied must NOT be swallowed: the run
                     # would otherwise "succeed" with a term missing from H, i.e. a
                     # physically wrong spectrum that looks like a good result.
                     try:
                         rtype = rule.get('type')
                         if rtype == 'heisenberg' and not rule.get('ref_pair'):
                             builder.add_interaction_rule(
                                 type='heisenberg',
                                 distance=rule.get('distance'),
                                 value=rule.get('value')
                             )
                         else:
                             builder.add_symmetry_interaction(
                                 type=rtype,
                                 ref_pair=rule.get('ref_pair'),
                                 value=rule.get('value'),
                                 distance=rule.get('distance'),
                                 offset=rule.get('offset')
                             )
                     except Exception as e:
                         raise ValueError(
                             f"Failed to expand symmetry rule {rule}: {e}"
                         ) from e
             
             if atom_mode == 'symmetry':
                 # Standard distance-based propagation for rules without ref_pair
                 builder._expand_heisenberg_rules()
                 builder._expand_anisotropic_exchange_rules()
                 builder._expand_dm_rules()
                 builder._expand_interaction_matrix_rules()
        
        # 4. Integrate back to self.config
        # Update atoms_uc
        self.config['crystal_structure']['atoms_uc'] = builder.atoms_uc
        if 'lattice_vectors' not in self.config['crystal_structure']:
            if hasattr(builder, 'lattice_vectors'):
                 self.config['crystal_structure']['lattice_vectors'] = builder.lattice_vectors.tolist()

        # Update interactions to a list of explicit entries
        if isinstance(interactions_data, dict):
            all_inters = []
            # Map builder internal lists to the formats GenericSpinModel expects
            # We map to keys like 'dm_manual' where needed.
            inter_map = [
                ("heisenberg", "heisenberg"),
                ("dm_interaction", "dm_manual"),
                ("anisotropic_exchange", "anisotropic_exchange"),
                ("interaction_matrix", "interaction_matrix"),
                ("kitaev", "kitaev"),
                ("single_ion_anisotropy", "sia"),
                # On-site / higher-order terms. These are not touched by the
                # symmetry-rule expanders, but they MUST be carried through this
                # dict->list conversion or they would silently vanish.
                ("sia_matrix", "sia_matrix"),
                ("stevens", "stevens"),
                ("biquadratic", "biquadratic"),
            ]
            for extra_key in ("sia_matrix", "stevens", "biquadratic"):
                if extra_key in interactions_data:
                    builder.config["interactions"].setdefault(
                        extra_key, list(interactions_data[extra_key]))
            for itype_key, key_in_model in inter_map:
                if itype_key in builder.config["interactions"]:
                    for r in builder.config["interactions"][itype_key]:
                         r["type"] = key_in_model
                         all_inters.append(r)
            
            # Carry over SIA if present in original but not added via builder
            # Note: builder always has the key as an empty list by default.
            if ("sia" in interactions_data or "single_ion_anisotropy" in interactions_data) and \
               not builder.config["interactions"].get("single_ion_anisotropy"):
                 sia_list = interactions_data.get("sia", interactions_data.get("single_ion_anisotropy", []))
                 for r in sia_list:
                      all_inters.append(r)

            self.config['interactions'] = all_inters

            # DEBUG: Check for strings in interactions
            for idx, item in enumerate(self.config['interactions']):
                if not isinstance(item, dict):
                    logger.error(f"DEBUG: Found non-dict item in interactions at index {idx}: {item} (Type: {type(item)})")

        # Long-range dipole-dipole -> explicit per-bond interaction matrices.
        # Runs last so it sees the final atoms_uc/lattice_vectors, and before the
        # magnetic supercell so the replication remaps the generated bonds.
        dd_spec = None
        if isinstance(interactions_data, dict):
            dd_spec = interactions_data.get('dipole_dipole')
        if dd_spec is None:
            dd_spec = self.config.get('dipole_dipole')
        if dd_spec:
            self._expand_dipole_dipole(dd_spec)

    def _expand_dipole_dipole(self, spec):
        """Expand dipolar coupling into explicit 3x3 `interaction_matrix` bonds.

        H_dip = (mu0 g_i g_j muB^2 / 4pi) * sum_{i<j}
                    [ S_i.S_j - 3 (S_i.rhat)(S_j.rhat) ] / r^3

        i.e. a bilinear bond matrix  J_ij = A_ij (I - 3 rhat rhat^T) / r^3  with
        A_ij = (mu0 muB^2/4pi) g_i g_j. Emitting it as ordinary bond matrices (both
        directions, matching the 1/2-ordered-pairs convention) means the rotating
        frame, magnetic supercell and symmetry checks all handle it for free.

        This is a REAL-SPACE TRUNCATED sum inside `cutoff` (Angstrom) -- the
        analogue of Sunny's `modify_exchange_with_truncated_dipole_dipole!`, not
        its Ewald-summed `enable_dipole_dipole!`. The dipolar sum is only
        conditionally convergent, so results depend on the cutoff: increase it
        until the quantity you care about stops moving.

        spec: {cutoff: <Angstrom>, g: <optional override, default per-site g or 2>}
        """
        # Stash the spec at the top level: `interactions` is about to be flattened to a
        # list, and MagCalc needs to find this later.
        if isinstance(spec, dict):
            self.config['dipole_dipole'] = dict(spec)

        method = str(spec.get('method', 'truncated')).lower() \
            if isinstance(spec, dict) else 'truncated'
        if method == 'ewald':
            # Ewald sums ALL images exactly, so there are no finite bonds to generate --
            # A(q) is added to the dynamical matrix directly (core._ewald_nambu).
            # Generating truncated bonds here as well would double-count.
            logger.info("dipole_dipole: Ewald summation (no real-space bonds generated).")
            return
        if method != 'truncated':
            raise ValueError(
                f"dipole_dipole.method must be 'truncated' or 'ewald', got {method!r}.")

        cutoff = float(spec.get('cutoff', 0.0)) if isinstance(spec, dict) else float(spec)
        if cutoff <= 0:
            raise ValueError(
                f"dipole_dipole needs a positive `cutoff` in Angstrom, got {spec!r}.")

        cs = self.config['crystal_structure']
        lat = np.asarray(cs['lattice_vectors'], dtype=float)
        atoms = cs['atoms_uc']
        labels = [a['label'] for a in atoms]
        frac = np.asarray([a['pos'] for a in atoms], dtype=float)
        cart = frac @ lat
        n = len(atoms)

        # Per-site isotropic g for the dipolar prefactor. A full g-tensor makes the
        # dipolar coupling anisotropic in a way this scalar form cannot express, so
        # be explicit rather than silently using some projection of it.
        g_override = spec.get('g') if isinstance(spec, dict) else None
        g_site = []
        for a in atoms:
            g = g_override if g_override is not None else a.get('g', 2.0)
            if not isinstance(g, (int, float)):
                raise ValueError(
                    f"dipole_dipole needs a scalar g per site (site {a.get('label')} "
                    f"has g={g!r}). Pass `g:` in the dipole_dipole block to override.")
            g_site.append(float(g))

        # Cell images that can hold a bond of length <= cutoff.
        vol = abs(np.linalg.det(lat))
        heights = [vol / np.linalg.norm(np.cross(lat[(k+1) % 3], lat[(k+2) % 3]))
                   for k in range(3)]
        ranges = [range(-m, m + 1)
                  for m in (max(1, int(np.ceil(cutoff / h))) for h in heights)]

        entries = []
        for i in range(n):
            for j in range(n):
                for off in product(*ranges):
                    r_vec = (cart[j] + np.asarray(off, dtype=float) @ lat) - cart[i]
                    r = float(np.linalg.norm(r_vec))
                    if r < 1e-8 or r > cutoff:
                        continue
                    rhat = r_vec / r
                    A = DIPOLE_PREFACTOR_MEV_A3 * g_site[i] * g_site[j]
                    J = A * (np.eye(3) - 3.0 * np.outer(rhat, rhat)) / r**3
                    entries.append({
                        'type': 'interaction_matrix',
                        'pair': [labels[i], labels[j]],
                        'rij_offset': list(off),
                        'value': [[float(x) for x in row] for row in J],
                        'distance': r,
                    })

        if not entries:
            raise ValueError(
                f"dipole_dipole with cutoff {cutoff} A matched no bonds -- the cutoff "
                f"is shorter than the nearest neighbour distance.")

        inters = self.config.get('interactions')
        if isinstance(inters, list):
            inters.extend(entries)
        else:
            self.config['interactions'] = entries
        logger.info(
            f"dipole_dipole: generated {len(entries)} bond matrices within "
            f"{cutoff} A (truncated real-space sum).")


    def _apply_magnetic_supercell(self):
        """Expand the chemical cell into a diagonal magnetic supercell.

        SpinW nExt / Sunny resize_supercell analogue, driven by
        ``crystal_structure.magnetic_supercell: [n1, n2, n3]`` (or ``'auto'``
        to derive the minimal cell commensurate with the propagation vector).

        Performs, in place on ``self.config``:
        - replicates ``atoms_uc`` over the n1*n2*n3 cells (cell-major, atom
          index fastest; cell (0,0,0) keeps the original labels, replicas are
          labelled ``<label>@i_j_l``) and rescales the lattice;
        - remaps explicit pair/offset interactions onto the replicated sites
          with periodic wrapping (offsets in supercell units); distance-only
          rules and SIA entries are expanded accordingly;
        - converts the magnetic structure to explicit per-site directions:
          a ``single_k`` structure becomes the commensurate real-space spin
          pattern (replicas rotated by R(2*pi*k.c, axis) like Sunny's
          repeat_periodically_as_spiral), so the LSWT runs on the supercell
          with unrotated interactions (folded bands, SpinW convention).

        Q-vectors remain in CHEMICAL-cell RLU: the runner converts them with
        the chemical B-matrix (see ``chemical_unit_cell``).
        """
        cs = self.config.get('crystal_structure', {}) or {}
        spec = cs.get('magnetic_supercell')
        if not spec:
            return

        # A NON-DIAGONAL (3x3) supercell is applied by the SU(N) engine, which handles
        # it natively (SUNModel._replicate); the dipole engine cannot express it, so
        # refuse rather than silently fall back to the chemical cell.
        _mat = spec.get('matrix') if isinstance(spec, dict) else spec
        if _mat is not None and np.asarray(_mat, dtype=object).shape == (3, 3):
            mode = str((self.config.get('calculation') or {}).get('mode', 'dipole')).upper()
            if mode != 'SUN':
                raise ValueError(
                    "A non-diagonal magnetic_supercell (3x3 matrix) is only supported by "
                    "the SU(N) engine. Set `calculation: {mode: SUN}`, or give a diagonal "
                    "[n1, n2, n3].")
            logger.info("Non-diagonal magnetic_supercell: applied by the SU(N) engine.")
            return

        # quiet=True: the k=1/2 double-count warning is moot when the user is
        # already requesting a real-space supercell.
        ms_raw = normalize_magnetic_structure(self.config.get('magnetic_structure'),
                                              quiet=True)
        if ms_raw.get('type') == 'single_k':
            k_rlu = np.asarray(ms_raw.get('k', [0, 0, 0]), dtype=float)
        elif ms_raw.get('type') == 'multi_k':
            # Every component's k must fit the cell -> pass them all.
            k_rlu = np.asarray([c['k'] for c in ms_raw.get('components', [])],
                               dtype=float)
        else:
            k_rlu = None

        dims = resolve_supercell_dims(spec, k_rlu=k_rlu)
        if dims == [1, 1, 1]:
            logger.info("magnetic_supercell [1,1,1]: nothing to expand.")
            return
        n1, n2, n3 = dims
        n_cells = n1 * n2 * n3
        cells = [np.array(c, dtype=int) for c in product(range(n1), range(n2), range(n3))]

        atoms = cs.get('atoms_uc')
        if not atoms:
            raise ValueError("magnetic_supercell requires explicit or expanded "
                             "'atoms_uc' (Wyckoff atoms are expanded first).")
        if 'cif_file' in cs:
            raise ValueError("magnetic_supercell is not supported with cif_file "
                             "structures; list atoms_uc explicitly.")
        interactions = self.config.get('interactions', [])
        if not isinstance(interactions, list):
            raise ValueError("magnetic_supercell requires the interactions list "
                             "form (builder expansion should have produced it).")

        n_chem = len(atoms)
        logger.info(f"Applying magnetic supercell {dims}: {n_chem} -> "
                    f"{n_chem * n_cells} sites.")

        # --- 1. Replicate atoms (cell-major, atom index fastest) ---
        dims_f = np.asarray(dims, dtype=float)
        new_atoms = []
        for cell in cells:
            for atom in atoms:
                rep = dict(atom)
                rep['label'] = supercell_site_label(atom['label'], tuple(cell))
                pos = (np.asarray(atom['pos'], dtype=float) + cell) / dims_f
                rep['pos'] = [float(x) for x in pos]
                new_atoms.append(rep)
        cs['atoms_uc'] = new_atoms
        cs.pop('wyckoff_atoms', None)

        # --- 2. Rescale the lattice ---
        if 'lattice_vectors' in cs:
            lv = np.asarray(cs['lattice_vectors'], dtype=float)
            cs['lattice_vectors'] = (lv * dims_f[:, None]).tolist()
            if 'lattice_parameters' in cs:
                # lattice_vectors takes priority in _load_structure; drop the
                # now-inconsistent parameters to avoid ambiguity.
                cs.pop('lattice_parameters', None)
        elif 'lattice_parameters' in cs:
            lp = cs['lattice_parameters']
            for key, n in zip(('a', 'b', 'c'), dims):
                if lp.get(key) is not None:
                    lp[key] = float(lp[key]) * n
            # The supercell breaks the space group; symmetry expansion already
            # ran on the chemical cell.
            lp.pop('space_group', None)
        else:
            raise ValueError("magnetic_supercell requires lattice_vectors or "
                             "lattice_parameters.")

        # --- 3. Remap interactions ---
        new_interactions = []
        for entry in interactions:
            itype = entry.get('type')
            pair = entry.get('pair')
            offset = entry.get('rij_offset', entry.get('offset_j'))

            if itype in ('sia', 'single_ion_anisotropy'):
                rep = dict(entry)
                targets = rep.get('atoms') or rep.get('atom_labels')
                if targets:
                    expanded = [supercell_site_label(lbl, tuple(c))
                                for c in cells for lbl in targets]
                    if 'atoms' in rep or 'atom_labels' not in rep:
                        rep['atoms'] = expanded
                        rep.pop('atom_labels', None)
                    else:
                        rep['atom_labels'] = expanded
                new_interactions.append(rep)
            elif pair and offset is not None:
                m = np.asarray(offset, dtype=int)
                for cell in cells:
                    target = cell + m
                    wrapped = np.mod(target, dims)
                    new_off = np.floor_divide(target, dims)
                    rep = dict(entry)
                    rep['pair'] = [supercell_site_label(pair[0], tuple(cell)),
                                   supercell_site_label(pair[1], tuple(wrapped))]
                    rep['rij_offset'] = [int(x) for x in new_off]
                    rep.pop('offset_j', None)
                    new_interactions.append(rep)
            elif pair and entry.get('distance') is not None:
                # Pair + distance (no offset): replicate the pair labels over
                # all cell combinations; the distance check does the matching.
                for c_i in cells:
                    for c_j in cells:
                        rep = dict(entry)
                        rep['pair'] = [supercell_site_label(pair[0], tuple(c_i)),
                                       supercell_site_label(pair[1], tuple(c_j))]
                        new_interactions.append(rep)
            else:
                # Distance-only rules (no pair) match every site pair at that
                # distance in the supercell at runtime; keep unchanged.
                new_interactions.append(dict(entry))
        self.config['interactions'] = new_interactions

        # --- 4. Magnetic structure -> explicit per-site directions ---
        self._remap_magnetic_structure_supercell(ms_raw, atoms, cells, dims_f)

        self.supercell_dims = dims

    def _remap_magnetic_structure_supercell(self, ms_raw, chem_atoms, cells, dims_f):
        """Convert the magnetic structure to per-supercell-site directions."""
        mtype = ms_raw.get('type')
        if not mtype:
            return
        n_chem = len(chem_atoms)

        if mtype == 'single_k':
            k = np.asarray(ms_raw.get('k', [0, 0, 0]), dtype=float)
            axis = np.asarray(ms_raw.get('axis', [0, 0, 1]), dtype=float)
            n_ax = axis / (np.linalg.norm(axis) or 1.0)
            d_frac = [np.asarray(a['pos'], dtype=float) for a in chem_atoms]

            # Commensurability: the pattern must be periodic in the supercell.
            kn = k * dims_f
            if np.max(np.abs(kn - np.round(kn))) > 1e-6:
                logger.warning(
                    "magnetic_supercell %s is not commensurate with k=%s "
                    "(k*dims not integer); the spin pattern will not be "
                    "periodic — use the rotating-frame single_k mode instead.",
                    [int(d) for d in dims_f], k.tolist())

            # Lab-frame direction of each chemical site in cell 0.
            cone_deg = float(ms_raw.get('cone_angle_deg', 0.0) or 0.0)
            lab0 = []
            if ms_raw.get('S0'):
                S0 = ms_raw['S0']
                if len(S0) == 1:
                    S0 = S0 * n_chem
                lab0 = [np.asarray(s, dtype=float) for s in S0]
            else:
                # local_directions / u,v basis / default -> lab frame via the
                # full-position phase (CLAUDE.md convention).
                if ms_raw.get('local_directions') or ms_raw.get('directions'):
                    dirs = ms_raw.get('local_directions', ms_raw.get('directions'))
                    u_loc = [np.asarray(dirs[i % len(dirs)], dtype=float)
                             for i in range(n_chem)]
                elif ms_raw.get('u') is not None or ms_raw.get('v') is not None:
                    u_vec = np.asarray(ms_raw.get('u', [1, 0, 0]), dtype=float)
                    v_vec = np.asarray(ms_raw.get('v', [0, 1, 0]), dtype=float)
                    u_loc = []
                    for i in range(n_chem):
                        ph = 2 * np.pi * float(d_frac[i] @ k)
                        u_loc.append(rotation_about_axis(-ph, n_ax) @ (
                            u_vec * np.cos(ph) + v_vec * np.sin(ph)))
                else:
                    trial = np.array([1.0, 0.0, 0.0]) if abs(n_ax[0]) < 0.9 \
                        else np.array([0.0, 1.0, 0.0])
                    u0 = trial - np.dot(trial, n_ax) * n_ax
                    u0 /= np.linalg.norm(u0)
                    u_loc = [u0.copy() for _ in range(n_chem)]
                if cone_deg > 0.0:
                    c = np.radians(cone_deg)
                    coned = []
                    for v in u_loc:
                        u_in = v - np.dot(v, n_ax) * n_ax
                        nrm = np.linalg.norm(u_in)
                        if nrm < 1e-12:
                            raise ValueError("Conical structure needs a non-zero "
                                             "in-plane component.")
                        coned.append(np.cos(c) * n_ax + np.sin(c) * (u_in / nrm))
                    u_loc = coned
                for i in range(n_chem):
                    ph = 2 * np.pi * float(d_frac[i] @ k)
                    lab0.append(rotation_about_axis(ph, n_ax) @ u_loc[i])

            # Replicas: rotate by R(2*pi*k.cell) — Sunny's
            # repeat_periodically_as_spiral cell-offset convention.
            directions = []
            for cell in cells:
                R_cell = rotation_about_axis(2 * np.pi * float(cell @ k), n_ax)
                for i in range(n_chem):
                    v = R_cell @ lab0[i]
                    nrm = np.linalg.norm(v)
                    if nrm > 1e-12:
                        v = v / nrm
                    directions.append([float(x) for x in v])

            self.config['magnetic_structure'] = {
                'enabled': True, 'type': 'pattern', 'pattern_type': 'generic',
                'directions': directions,
            }
            logger.info(
                "single_k + magnetic_supercell: converted to a real-space "
                "commensurate pattern (%d sites); LSWT runs on the supercell "
                "with unrotated interactions (folded bands).", len(directions))
        elif mtype == 'pattern':
            rep = dict(ms_raw)
            dirs = rep.get('directions')
            if dirs:
                rep['directions'] = [dirs[i % len(dirs)] for _ in cells
                                     for i in range(n_chem)]
            self.config['magnetic_structure'] = rep
        elif mtype == 'explicit':
            entries = ms_raw.get('explicit_list', ms_raw.get('configuration', []))
            new_entries = []
            for ci, _cell in enumerate(cells):
                for item in entries:
                    idx = item.get('atom_index')
                    if idx is None or not (0 <= idx < n_chem):
                        continue
                    rep = dict(item)
                    rep['atom_index'] = ci * n_chem + idx
                    new_entries.append(rep)
            self.config['magnetic_structure'] = {
                'enabled': True, 'type': 'explicit', 'explicit_list': new_entries,
            }
        # other/absent types: leave unchanged

    def chemical_unit_cell(self):
        """Lattice vectors of the CHEMICAL cell (rows), undoing any magnetic
        supercell scaling. Q-vectors in configs are interpreted in chemical
        RLU (SpinW/Sunny convention)."""
        uc = np.asarray(self.unit_cell(), dtype=float)
        dims = np.asarray(self.supercell_dims, dtype=float)
        return uc / dims[:, None]

    def _load_structure(self):
        """
        Loads the crystal structure from the configuration.

        This method handles two modes:
        1. CIF File: Loads structure using ASE from a .cif file.
        2. Explicit Definition: constructing unit cell vectors from lattice parameters
           and atom positions from fractional or Cartesian coordinates.

        Raises:
            ValueError: If structure definition is missing or invalid.
            ImportError: If ASE is required for CIF loading but not installed.
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
                # Use elements from CIF
                self._ion_list = list(self.atoms.get_chemical_symbols())
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
             
             # Extract ions
             self._ion_list = [a.get('ion', a.get('element', a.get('label', 'Fe3+'))) for a in atoms]

             # Extract per-atom spin magnitudes (enables mixed-spin models).
             self._spin_list = [float(a.get('spin_S', 0.5)) for a in atoms]
             
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

    def spin_magnitudes(self):
        """Per-atom spin magnitudes S_i for the unit-cell atoms.

        Enables mixed-spin models: the LSWT layer scales each site's
        Holstein-Primakoff expansion by S_i. Returns a list of floats in the
        same order as ``atom_pos()``.
        """
        return list(getattr(self, '_spin_list', []))

    def atom_pos_ouc(self):
        """Returns neighbor positions."""
        return self._atoms_ouc

    def ion_list(self):
        """Returns list of ion names for Each atom in the unit cell."""
        return self._ion_list

    def _required_neighbor_shells(self):
        """Smallest per-axis neighbor shell that covers every explicit bond offset.

        Bonds whose ``rij_offset`` reaches beyond the OUC used to be SILENTLY
        dropped (the offset match in ``spin_interactions`` simply never fired),
        corrupting any model with |offset| > 1 (e.g. 2nd-neighbor bonds along
        one axis, FeI2's J3/J'2a, spiral chains). Scan both the raw config and
        the builder-expanded interaction list and size the shell accordingly.
        """
        shells = [1, 1, 1]
        items = []
        ints = self.config.get('interactions')
        if isinstance(ints, dict):
            for v in ints.values():
                if isinstance(v, list):
                    items.extend(x for x in v if isinstance(x, dict))
        elif isinstance(ints, list):
            items.extend(x for x in ints if isinstance(x, dict))
        items.extend(x for x in getattr(self, 'interactions_config', []) or []
                     if isinstance(x, dict))
        for it in items:
            off = it.get('rij_offset', it.get('offset'))
            if isinstance(off, (list, tuple)) and len(off) == 3:
                try:
                    for i in range(3):
                        shells[i] = max(shells[i], abs(int(off[i])))
                except (TypeError, ValueError):
                    continue
        return shells

    def _generate_atom_pos_ouc(self):
        """Generate neighbors (internal helper)."""
        uc = self.unit_cell()
        apos = self.atom_pos()
        apos_len = len(apos)

        # Standard -1 to 1 supercell
        r_pos_ouc = [apos[k] for k in range(apos_len)]

        from itertools import product

        # Check config for dimensionality limit (default 3)
        c_struct = self.config.get('crystal_structure', {})
        dims = c_struct.get('dimensionality', 3)
        calc_settings = self.config.get('calculation_settings', {})
        neighbor_shells = calc_settings.get('neighbor_shells')

        neighbor_offsets = []

        if not neighbor_shells:
            # Auto-size the shell so every explicit bond offset is reachable.
            required = self._required_neighbor_shells()
            if required != [1, 1, 1]:
                logger.info(f"Extending neighbor shell to {required} to cover bond offsets.")
                neighbor_shells = required

        if neighbor_shells:
             rx = range(-neighbor_shells[0], neighbor_shells[0]+1)
             ry = range(-neighbor_shells[1], neighbor_shells[1]+1)
             rz = range(-neighbor_shells[2], neighbor_shells[2]+1)
             if dims == 2:
                  rz = [0]
             loop_iter = product(rx, ry, rz)
        else:
             rng = range(-1, 2)
             if dims == 2:
                  loop_iter = product(rng, rng, [0])
             else:
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
        
        if self.config.get('parameter_order'):
            # Use explicit order if provided
            keys = [k for k in self.config['parameter_order'] if k in param_names and k != 'S']
            # Also append any missing keys? Or assume parameter_order is complete?
            # MagCalc core assumes parameter_order defines the list.
        else:
            # Match runner.py/core.py fallback: insertion order excluding 'S'
            keys = [k for k in param_names if k != 'S']
        
        for i, name in enumerate(keys):
            if i < len(p):
                param_map[name] = p[i]
        return param_map

    def spin_interactions(self, p):
        """
        Generates interaction matrices (Heisenberg, DM, Anisotropic) based on the
        configuration. For a single-k (spiral) structure the lab-frame matrices
        are transformed into the rotating frame.
        """
        Jex, DM, Kex = self._spin_interactions_lab(p)

        if self.use_rotating_frame:
            logger.info("Single-k magnetic structure detected. Computing effective rotated interactions...")
            self._check_rotational_symmetry(Jex, DM, Kex)
            Jex, DM, Kex = self._compute_rotated_interactions(Jex, DM, Kex, self.mag_struct_cfg)

        return Jex, DM, Kex

    def _spin_interactions_lab(self, p):
        """
        Generates the lab-frame interaction matrices (Heisenberg, DM, Anisotropic)
        based on the configuration, without any rotating-frame transformation.
        """
        apos = self.atom_pos()
        N_atom = len(apos)
        apos_ouc = self.atom_pos_ouc()
        N_atom_ouc = len(apos_ouc)
        
        Jex = sp.zeros(N_atom, N_atom_ouc)
        DM = [[None for _ in range(N_atom_ouc)] for _ in range(N_atom)]
        Kex = [[None for _ in range(N_atom_ouc)] for _ in range(N_atom)]
        
        dist_tol = 0.1 # Increased slightly for robustness
        param_map = self._resolve_param_map(p)
        param_counter = 0
        
        atom_labels = [a.get('label') for a in self.config.get('crystal_structure').get('atoms_uc')]
        label_to_idx = {lbl: idx for idx, lbl in enumerate(atom_labels)}
    
        for interaction in self.interactions_config:
            itype = interaction.get('type')
            if not itype: continue
            
            # 1. Resolve Value
            val = interaction.get('value')
            if val is None:
                val = interaction.get('K') # Fallback for some builder/designer formats
            resolved_val = None
            
            if itype == 'heisenberg':
                if isinstance(val, str) and param_map:
                    resolved_val = safe_eval(val, param_map)
                elif isinstance(val, (int, float)):
                    resolved_val = val
                elif param_counter < len(p):
                    resolved_val = p[param_counter]
                    param_counter += 1
            if itype in ['dm', 'dm_manual', 'anisotropic_exchange', 'interaction_matrix', 'kitaev']:
                if isinstance(val, list):
                    resolved_val = []
                    for v in val:
                        if isinstance(v, str) and param_map: 
                            resolved_val.append(safe_eval(v, param_map))
                        elif isinstance(v, (list, tuple)):
                            nested_res = []
                            for sub_v in v:
                                if isinstance(sub_v, str) and param_map:
                                    nested_res.append(safe_eval(sub_v, param_map))
                                else:
                                    nested_res.append(sub_v)
                            resolved_val.append(nested_res)
                        else: 
                            resolved_val.append(v)
                elif isinstance(val, (int, float, str)) and param_map:
                    # Single symbolic expression for vector or scalar
                    resolved_val = safe_eval(str(val), param_map)
                elif isinstance(val, (int, float)):
                    resolved_val = val

            # 2. Matching logic
            target_pair = interaction.get('pair')
            offset = interaction.get('rij_offset') or interaction.get('offset_j')
            target_dist = interaction.get('distance')
            
            # Determine candidate i indices
            if target_pair:
                i_candidates = [label_to_idx[target_pair[0]]] if target_pair[0] in label_to_idx else []
            else:
                i_candidates = range(N_atom)
                
            for i in i_candidates:
                # Determine candidate j indices (in OUC)
                for j in range(N_atom_ouc):
                    j_uc = j % N_atom
                    
                    # Label check
                    if target_pair and atom_labels[j_uc] != target_pair[1]:
                        continue
                        
                    # Offset check (if provided)
                    if offset is not None:
                        target_pos = apos[j_uc] + offset[0]*self.unit_cell()[0] + \
                                     offset[1]*self.unit_cell()[1] + offset[2]*self.unit_cell()[2]
                        if la.norm(apos_ouc[j] - target_pos) > 0.001:
                            continue
                    
                    # Distance check (if no offset or as extra verification)
                    if target_dist is not None:
                        d = la.norm(apos[i] - apos_ouc[j])
                        if abs(d - target_dist) > dist_tol:
                            continue
                    elif offset is None:
                        # If neither offset nor distance provided, we can't match.
                        continue

                    # 3. Populate Matrices
                    if itype == 'heisenberg':
                        Jex[i, j] += resolved_val
                    elif itype in ['dm', 'dm_manual']:
                        D_vec = sp.Matrix(resolved_val)
                        if DM[i][j] is None: DM[i][j] = D_vec
                        else: DM[i][j] += D_vec
                    elif itype == 'anisotropic_exchange':
                        K_vec = sp.Matrix(resolved_val)
                        if Kex[i][j] is None: Kex[i][j] = K_vec
                        else: Kex[i][j] += K_vec
                    elif itype == 'interaction_matrix':
                        # resolved_val could be 3x3 list of lists or flattened
                        if len(resolved_val) == 3: J_mat = sp.Matrix(resolved_val) # nested
                        else: J_mat = sp.Matrix(resolved_val).reshape(3, 3) # flattened
                        if Kex[i][j] is None: Kex[i][j] = J_mat
                        else: Kex[i][j] += J_mat
                    elif itype == 'kitaev':
                        # Standard Kitaev: bond along x, y, or z
                        # Kitaev rules in builder usually specify axis or bond_direction.
                        axis = (interaction.get('axis') or interaction.get('bond_direction') or 'z').lower()
                        k_val = resolved_val[0] if isinstance(resolved_val, list) else resolved_val
                        if k_val is None:
                            logger.warning(f"Kitaev interaction value resolved to None for {interaction}. Skipping.")
                            continue
                        k_mat = sp.zeros(3,3)
                        ax_idx = {'x':0, 'y':1, 'z':2}.get(axis, 2)
                        k_mat[ax_idx, ax_idx] = k_val
                        if Kex[i][j] is None: Kex[i][j] = k_mat
                        else: Kex[i][j] += k_mat

        # 4. Fill None with zeros
        dnull_vec = sp.Matrix([0, 0, 0])
        dnull_mat = sp.zeros(3, 3)
        for i in range(N_atom):
            for j in range(N_atom_ouc):
                if DM[i][j] is None: DM[i][j] = dnull_vec
                if Kex[i][j] is None: Kex[i][j] = dnull_mat

        return Jex, DM, Kex

    def _match_bond_pairs(self, interaction):
        """(i, j_ouc) pairs matched by an interaction's pair/offset/distance keys.

        Same matching rules as _spin_interactions_lab, factored out for the
        non-bilinear terms (biquadratic) that do not populate Jex/DM/Kex.
        """
        apos = self.atom_pos()
        apos_ouc = self.atom_pos_ouc()
        N_atom, N_atom_ouc = len(apos), len(apos_ouc)
        dist_tol = 0.1
        atom_labels = [a.get('label') for a in
                       self.config.get('crystal_structure').get('atoms_uc')]
        label_to_idx = {lbl: idx for idx, lbl in enumerate(atom_labels)}

        target_pair = interaction.get('pair')
        offset = interaction.get('rij_offset') or interaction.get('offset_j')
        target_dist = interaction.get('distance')

        if target_pair:
            i_candidates = [label_to_idx[target_pair[0]]] \
                if target_pair[0] in label_to_idx else []
        else:
            i_candidates = range(N_atom)

        pairs = []
        for i in i_candidates:
            for j in range(N_atom_ouc):
                j_uc = j % N_atom
                if target_pair and atom_labels[j_uc] != target_pair[1]:
                    continue
                if offset is not None:
                    target_pos = apos[j_uc] + offset[0]*self.unit_cell()[0] + \
                                 offset[1]*self.unit_cell()[1] + offset[2]*self.unit_cell()[2]
                    if la.norm(apos_ouc[j] - target_pos) > 0.001:
                        continue
                if target_dist is not None:
                    d = la.norm(apos[i] - apos_ouc[j])
                    if abs(d - target_dist) > dist_tol:
                        continue
                elif offset is None:
                    continue
                pairs.append((i, j))
        return pairs

    def _compute_biquadratic_terms(self, Sxyz, param_map):
        """`type: biquadratic` -- H = (1/2) sum_ij B_ij (S_i . S_j)^2.

        The 1/2-over-ordered-pairs convention matches the Heisenberg term, so
        every bond must appear in BOTH directions (symmetry rules do this
        automatically). Unlike SW28's collinear J_eff = J +/- dJ workaround, this
        is the genuine operator and is therefore valid for non-collinear
        structures too.

        The quadratic-boson part of (S_i.S_j)^2 carries S^3; it survives only
        because the LSWT truncation is by boson degree (see
        symbolic._prepare_hamiltonian) -- the legacy S-power filter deleted it.
        """
        HM = 0
        for interaction in self.interactions_config:
            if interaction.get('type') not in ('biquadratic', 'biq'):
                continue
            B = self._resolve_scalar(
                interaction.get('value', interaction.get('B')), param_map)
            if B is None:
                raise ValueError(f"biquadratic entry needs a `value`: {interaction}")
            pairs = self._match_bond_pairs(interaction)
            if not pairs:
                raise ValueError(
                    f"biquadratic entry matched no bonds: {interaction}. A term that "
                    f"matches nothing would silently vanish from the Hamiltonian.")
            for i, j in pairs:
                dot = (Sxyz[i][0]*Sxyz[j][0] + Sxyz[i][1]*Sxyz[j][1]
                       + Sxyz[i][2]*Sxyz[j][2])
                HM += 0.5 * B * dot**2
        return HM

    def _check_rotational_symmetry(self, Jex, DM, Kex, theta=0.01):
        """Verify the Hamiltonian is invariant under rotations about the spiral axis.

        Mirrors Sunny's check_rotational_symmetry: the single-k rotating-frame
        method is only exact when every bond matrix satisfies R^T J R = J for
        rotations R about the axis (DM vectors parallel to the axis, exchange
        matrices uniaxial about it), single-ion anisotropy axes are parallel to
        it, and any external field is parallel to it. Symbolic entries are
        tested at random parameter values. Behavior on failure follows
        magnetic_structure.enforce_rotational_symmetry: 'warn' (default),
        'error', or 'off'.
        """
        mode = self.mag_struct_cfg.get('enforce_rotational_symmetry', 'warn')
        if mode == 'off':
            return

        n = self.spiral_axis
        R = rotation_about_axis(theta, n)
        problems = []

        apos = self.atom_pos()
        N_atom = len(apos)
        N_ouc = len(self.atom_pos_ouc())
        rng = np.random.default_rng(seed=7)

        def _to_numeric(mat):
            """Evaluate a (possibly symbolic) sympy matrix at random param values."""
            m = sp.Matrix(mat)
            syms = list(m.free_symbols)
            if syms:
                subs = {s: float(rng.uniform(0.5, 1.5)) for s in syms}
                m = m.subs(subs)
            return np.array(m.evalf().tolist(), dtype=complex).real

        checked = 0
        for i in range(N_atom):
            for j in range(N_ouc):
                D_vec = DM[i][j]
                K_mat = Kex[i][j]
                d_nonzero = not (hasattr(D_vec, 'is_zero_matrix') and D_vec.is_zero_matrix)
                k_nonzero = not (hasattr(K_mat, 'is_zero_matrix') and K_mat.is_zero_matrix)
                if not d_nonzero and not k_nonzero:
                    continue
                if checked > 200:  # enough bonds to trust the pattern
                    break
                checked += 1
                # J_global = Jex*I + skew(D) + Kex; the isotropic part is
                # always invariant, so only test the DM + matrix part.
                J_aniso = np.zeros((3, 3))
                if d_nonzero:
                    d = _to_numeric(D_vec).flatten()
                    J_aniso = J_aniso + np.array([
                        [0.0, d[2], -d[1]],
                        [-d[2], 0.0, d[0]],
                        [d[1], -d[0], 0.0],
                    ])
                if k_nonzero:
                    km = _to_numeric(K_mat)
                    if km.size == 3:
                        km = np.diag(km.flatten())
                    J_aniso = J_aniso + km.reshape(3, 3)
                if np.max(np.abs(R.T @ J_aniso @ R - J_aniso)) > 1e-8:
                    problems.append(
                        f"bond ({i},{j}): interaction matrix not invariant "
                        f"under rotation about axis {np.round(n, 6).tolist()}"
                    )

        # Single-ion anisotropy axes must be parallel to the spiral axis.
        for interaction in self.interactions_config:
            if interaction.get('type') != 'sia':
                continue
            axis_sia = np.asarray(interaction.get('axis', [0, 0, 1]), dtype=float)
            nrm = np.linalg.norm(axis_sia)
            if nrm > 1e-12 and np.linalg.norm(np.cross(axis_sia / nrm, n)) > 1e-8:
                problems.append(
                    f"single-ion anisotropy axis {axis_sia.tolist()} not parallel to spiral axis"
                )

        # External field must be parallel to the spiral axis.
        params = self.config.get('parameters') or {}
        h_dir = params.get('H_dir')
        h_mag = next((params.get(k) for k in ('H', 'H_mag', 'H_field') if k in params), None)
        if h_mag is not None and isinstance(h_mag, (int, float)) and abs(h_mag) > 1e-12:
            h_vec = np.asarray(h_dir, dtype=float) if h_dir is not None else np.array([0.0, 0.0, 1.0])
            nrm = np.linalg.norm(h_vec)
            if nrm > 1e-12 and np.linalg.norm(np.cross(h_vec / nrm, n)) > 1e-8:
                problems.append("external field not parallel to spiral axis")

        if problems:
            msg = ("Single-k rotating-frame Hamiltonian is NOT invariant under "
                   "rotations about the spiral axis; the spiral LSWT result is "
                   "unreliable:\n  - " + "\n  - ".join(problems[:5]))
            if mode == 'error':
                raise ValueError(msg)
            logger.warning(msg)

    def _compute_rotated_interactions(self, Jex, DM, Kex, mag_struct):
        """
        Compute effective interactions in the local rotating frame for a spiral.
        R_i^T * J_{ij} * R_j  =>  Effective J_{ij}
        """
        k_vec = np.array(mag_struct.get('k', [0, 0, 0]), dtype=float)
        axis = np.array(mag_struct.get('axis', [0, 0, 1]), dtype=float) # Normal to spiral plane
        # Normalize axis
        if la.norm(axis) > 1e-9: axis /= la.norm(axis)
        
        # We assume R_i is identity (or fixed reference).
        # We model the relative rotation R_{ij} = R(Q . r_{ij}) around 'axis'.
        
        dnull_vec = sp.Matrix([0, 0, 0]) # Fix: Define dnull_vec here
        
        apos = self.atom_pos()
        apos_ouc = self.atom_pos_ouc()
        
        N_atom = len(apos)
        N_ouc = len(apos_ouc)
        
        # Sympy rotation generator
        def get_rot_matrix(theta, axis_vec):
            # General rotation matrix around axis by theta
            # Rodrigues formula: I + sin(th) K + (1-cos(th)) K^2
            # But we need Sympy expression? Or numeric?
            # Interactions can be symbolic. Let's stick to Sympy for consistency?
            # Or numeric if K, Axis are numeric.
            # Assuming K and Axis numeric.
            ux, uy, uz = axis_vec
            K = sp.Matrix([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
            I = sp.eye(3)
            # theta is a float/value
            # We construct numeric matrix if possible
            R = I + sp.sin(theta) * K + (1 - sp.cos(theta)) * (K @ K)
            return R

        # Iterate and rotate
        # Effective J_{ij} = R_i^T @ J_{ij} @ R_j
        # Relative displacement r = r_j - r_i
        # Phase difference phi = k . r
        # R_j = R(phi) * R_i.  So R_i^T R_j = R(phi).
        # Thus J_eff = J_{ij} @ R(phi). (Assuming J is matrix form)
        
        # For Heisenberg scalar J: J_eff matrix = J * R(phi)
        # For DM D: D . (Si x Sj) -> R rotates spins?
        # DM energy: D . (Ri Si x Rj Sj). 
        # This is complex. Standard LSWT codes usually rotate the Interaction Matrix Kex.
        # Heisenberg J becomes J * Identity.
        # Full interaction K_{total} = J*I + S(D) + Kex
        # We transform K_{total}.
        
        # Phase convention: k in RLU, positions Cartesian (Angstrom), so
        # phase = 2*pi * k . r_frac with r_frac = r_cart @ inv(uc).
        # NOTE: pyMagCalc uses FULL atomic positions here (2*pi*k.(r_j - r_i)),
        # not just cell offsets; see CLAUDE.md section 3.
        inv_uc = la.inv(self.unit_cell())

        for i in range(N_atom):
            for j in range(N_ouc):
                r_vec = apos_ouc[j] - apos[i]
                r_frac = r_vec @ inv_uc # vector * matrix_inv

                phase = 2 * np.pi * np.dot(k_vec, r_frac)
                
                R_rel = get_rot_matrix(phase, axis)
                
                # Total Interaction Matrix in Global Frame
                # J_tot = J * I + Kex + DM_term
                # DM term D . (S x S) = S . (Skew(D)) . S? 
                # No, D . (Si x Sj) = Si . Skew(D) . Sj ?
                # Skew(D) = [[0, Dz, -Dy], [-Dz, 0, Dx], [Dy, -Dx, 0]] ?
                # Yes, Si . Skew(D) . Sj = sum Si_mu Skew_mu_nu Sj_nu
                
                # Construct Full J_global
                J_scalar = Jex[i, j]
                D_vec = DM[i][j]
                K_mat = Kex[i][j]
                
                J_global = sp.eye(3) * J_scalar + K_mat
                
                if not (hasattr(D_vec, 'is_zero_matrix') and D_vec.is_zero_matrix) and D_vec != sp.Matrix([0,0,0]):
                    Dx, Dy, Dz = D_vec[0], D_vec[1], D_vec[2]
                    D_skew = sp.Matrix([[0, Dz, -Dy], [-Dz, 0, Dx], [Dy, -Dx, 0]])
                    J_global += D_skew
                    
                # Effective J_local = J_global * R_rel
                # (Assuming R_i = I, R_j = R_rel)
                # J_eff = J_global * R_rel
                
                J_eff = J_global * R_rel
                
                # Decompose back?
                # GenericSpinModel expects Jex, DM, Kex.
                # But Effective Interaction might be fully anisotropic.
                # So we put everything into Kex and zero out Jex/DM.
                
                Jex[i, j] = 0
                DM[i][j] = dnull_vec
                Kex[i][j] = J_eff
                
        return Jex, DM, Kex

    def Hamiltonian(self, Sxyz: List[Any], pr: List[Any]) -> sp.Expr:
        """
        Constructs the symbolic Hamiltonian expression.

        This is the core method called by the MagCalc engine to build the Hamiltonian.
        It aggregates Heisenberg, DM, Anisotropic Exchange, Single-Ion Anisotropy,
        and Zeeman terms.

        Args:
            Sxyz (List[List[sp.Symbol]]): List of spin vector symbols [[Sx0, Sy0, Sz0], ...].
            pr (List[Any]): List of symbolic parameters.

        Returns:
            sp.Expr: The full symbolic Hamiltonian expression.
        """
        # Parse params
        Jex, DM, Kex, p_rest, param_map = self._prepare_interaction_matrices(pr)
        
        
        HM = 0
        mu_B = 5.788e-2
        
        # 1. Exchange Terms (Heisenberg + DM + Anisotropic)
        HM += self._compute_heisenberg_dm_terms(Sxyz, Jex, DM, Kex)
        
        # 1b. Biquadratic exchange
        HM += self._compute_biquadratic_terms(Sxyz, param_map)

        # 2. Extra Terms (SIA)
        HM += self._compute_sia_terms(Sxyz, p_rest, param_map)

        # 3. Zeeman
        # Default gamma=1.0. If users want g=2, they can provide g separately 
        # or specify H such that it includes factors. 
        # Actually, in most LSWT contexts, H = g*mu_B*H_field.
        # If we use gamma=1.0, the shift will be 1 * mu_B * H.
        # Given the reported doubling, setting gamma=1.0 should fix it.
        gamma = 1.0 
        HM += self._compute_zeeman_terms(Sxyz, p_rest, param_map, gamma, mu_B)

        # 4. Substitution and Filtering
        HM = self._apply_substitution_and_filter(HM, pr)
        
        return HM

    def _prepare_interaction_matrices(self, pr):
        """Prepare interaction matrices for Hamiltonian construction."""
        if self.config.get('parameters'):
            # New Named Mode
            Jex, DM, Kex = self.spin_interactions(pr)
            # Param map...
            param_map = self._resolve_param_map(pr)
            p_rest = pr # In Named mode, we can just pass everything to SIA/Zeeman
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
                         if K_vec.shape == (3, 3):
                             # Full Interaction Matrix
                             Si = sp.Matrix(Sxyz[i])
                             Sj = sp.Matrix(Sxyz[j])
                             term_mat = Si.T * K_vec * Sj
                             HM += 0.5 * term_mat[0]
                             terms_added += 1
                         else:
                             HM += 0.5 * (
                                 K_vec[0] * Sxyz[i][0] * Sxyz[j][0] +
                                 K_vec[1] * Sxyz[i][1] * Sxyz[j][1] +
                                 K_vec[2] * Sxyz[i][2] * Sxyz[j][2]
                             )
                             terms_added += 1
        return HM

    def _compute_sia_terms(self, Sxyz: List[Any], p_rest: List[Any], param_map: Dict[str, Any] = None) -> sp.Expr:
        """Compute Single Ion Anisotropy terms K (S.n)^2 for each targeted site.

        The strength ``value`` may be a numeric literal, a named parameter, or a
        parameter expression; a legacy positional fallback (consume from
        ``p_rest``) is used only if none of those resolve. Each SIA entry is
        applied *only* to the sites named in its ``atoms`` field (matched by
        label), defaulting to every site when ``atoms`` is omitted -- so listing
        one entry per site no longer multiplies the anisotropy by the site count.
        """
        HM = 0
        N_uc = len(self.atom_pos())
        # Map atom label -> unit-cell index (for the 'atoms' targeting field).
        try:
            atom_labels = [a.get('label') for a in
                           self.config.get('crystal_structure', {}).get('atoms_uc', [])]
        except Exception:
            atom_labels = []
        label_to_idx = {lbl: i for i, lbl in enumerate(atom_labels) if lbl is not None}

        rest_idx = 0
        for interaction in self.interactions_config:
            itype = interaction.get('type')
            if itype != 'sia':
                continue

            # --- Resolve the anisotropy strength D_sia ---
            D_sia = None
            val = interaction.get('value')
            if isinstance(val, (int, float)):
                # Numeric literal (e.g. value: 0.2). Honour it directly.
                D_sia = val
            elif isinstance(val, str) and param_map and val in param_map:
                # Named parameter (e.g. value: "D").
                D_sia = param_map[val]
            elif isinstance(val, str):
                # Parameter expression or numeric string.
                try:
                    D_sia = float(sp.sympify(val, locals=param_map or {}))
                except Exception:
                    D_sia = None
            if D_sia is None and rest_idx < len(p_rest):
                # Legacy positional fallback.
                D_sia = p_rest[rest_idx]
                rest_idx += 1
            if D_sia is None:
                continue

            # --- Determine which sites this entry applies to ---
            target_labels = interaction.get('atoms') or interaction.get('atom_labels')
            if target_labels:
                target_idx = [label_to_idx[l] for l in target_labels if l in label_to_idx]
            else:
                target_idx = list(range(N_uc))

            # --- Build the anisotropy axis ---
            axis = interaction.get('axis', [0, 0, 1])
            if isinstance(axis, (list, tuple, np.ndarray)) and len(axis) == 3:
                n = np.array(axis, dtype=float)
                norm = np.linalg.norm(n)
                if norm > 0:
                    n /= norm
                for i in target_idx:
                    S_dot_n = Sxyz[i][0]*n[0] + Sxyz[i][1]*n[1] + Sxyz[i][2]*n[2]
                    HM += D_sia * (S_dot_n)**2
            else:
                for i in target_idx:
                    HM += D_sia * (Sxyz[i][2])**2

        # --- General 3x3 anisotropy tensor: sum_ab A_ab S^a S^b ---
        HM += self._compute_sia_matrix_terms(Sxyz, param_map, label_to_idx, N_uc)
        # --- Stevens operators: sum_kq B_k^q O_k^q(S) ---
        HM += self._compute_stevens_terms(Sxyz, param_map, label_to_idx, N_uc)
        return HM

    def _resolve_scalar(self, val, param_map):
        """Resolve a config scalar: number, named parameter, or safe expression."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, str):
            if param_map and val in param_map:
                return param_map[val]
            return safe_eval(val, param_map or {})
        return val

    def _sia_target_indices(self, interaction, label_to_idx, N_uc):
        labels = interaction.get('atoms') or interaction.get('atom_labels')
        if labels:
            return [label_to_idx[l] for l in labels if l in label_to_idx]
        return list(range(N_uc))

    def _compute_sia_matrix_terms(self, Sxyz, param_map, label_to_idx, N_uc):
        """`type: sia_matrix` -- full 3x3 single-ion anisotropy tensor A.

        H_i = S_i^T A S_i, summed over the entry's target sites. Only the
        symmetric part of A contributes (the antisymmetric part contracts to
        zero against S_i S_i), so A is symmetrized on input.
        """
        HM = 0
        for interaction in self.interactions_config:
            if interaction.get('type') not in ('sia_matrix', 'anisotropy_matrix'):
                continue
            val = interaction.get('matrix', interaction.get('value'))
            if val is None:
                raise ValueError(f"sia_matrix entry needs a 3x3 `matrix`: {interaction}")
            rows = [[self._resolve_scalar(v, param_map) for v in row] for row in val]
            A = sp.Matrix(rows)
            if A.shape != (3, 3):
                raise ValueError(
                    f"sia_matrix `matrix` must be 3x3, got {A.shape}: {interaction}")
            A = (A + A.T) / 2  # only the symmetric part survives S_i^T A S_i
            for i in self._sia_target_indices(interaction, label_to_idx, N_uc):
                Si = sp.Matrix(Sxyz[i])
                HM += (Si.T * A * Si)[0]
        return HM

    def _compute_stevens_terms(self, Sxyz, param_map, label_to_idx, N_uc):
        """`type: stevens` -- crystal-field terms sum_{k,q} B_k^q O_k^q(S_i).

        Accepts either a single {k, q, value} or a `B` mapping of "k,q" -> value
        (e.g. {"2,0": B20, "4,3": B43}). O_k^q are the classical (large-s)
        Stevens polynomials, Sunny's `stevens_matrices(Inf)` convention.

        Quartic/sextic operators contribute quadratic-boson terms at order S^3 /
        S^5; these survive because the model truncates by boson degree (see
        symbolic._prepare_hamiltonian).
        """
        HM = 0
        for interaction in self.interactions_config:
            if interaction.get('type') != 'stevens':
                continue
            terms = {}
            if 'B' in interaction:
                for key, v in (interaction['B'] or {}).items():
                    k_str, q_str = str(key).replace(' ', '').split(',')
                    terms[(int(k_str), int(q_str))] = v
            elif 'k' in interaction:
                terms[(int(interaction['k']), int(interaction.get('q', 0)))] = \
                    interaction.get('value')
            else:
                raise ValueError(
                    f"stevens entry needs either `B: {{'k,q': value}}` or `k`/`q`/`value`: "
                    f"{interaction}")

            targets = self._sia_target_indices(interaction, label_to_idx, N_uc)
            for (k, q), raw in terms.items():
                B = self._resolve_scalar(raw, param_map)
                if B is None:
                    raise ValueError(
                        f"stevens B_{k}^{q} resolved to None in {interaction}")
                for i in targets:
                    HM += B * stevens_polynomial(k, q, Sxyz[i][0], Sxyz[i][1], Sxyz[i][2])
        return HM

    def _compute_zeeman_terms(self, Sxyz: List[Any], p_rest: List[Any], param_map: Dict[str, Any], gamma: float, mu_B: float) -> sp.Expr:
        """Compute Zeeman energy terms."""
        HM = 0
        N_uc = len(self.atom_pos())
        H_mag = None
        H_dir = None
        
        if self.config.get('parameters') or self.config.get('model_params'):
            params_vals = self.config.get('parameters', self.config.get('model_params', {}))
            for h_name in ['H', 'H_mag', 'H_field']:
                if h_name in param_map:
                    H_mag = param_map[h_name]
                    break
            
            # If still None or zero, check if it's in the values dict directly
            if H_mag is None:
                for h_name in ['H', 'H_mag', 'H_field']:
                    if h_name in params_vals:
                         H_mag = params_vals[h_name]
                         break
            
            # Explicitly check for H_dir if symbolic
            H_dir = param_map.get('H_dir')
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
             # Unwrap H_mag_val if list (e.g. from flattened params)
             if isinstance(H_mag_val, (list, tuple, np.ndarray)) and len(H_mag_val) == 1:
                  H_mag_val = H_mag_val[0]
             if isinstance(H_dir, (list, tuple, np.ndarray)) and len(H_dir) == 3:
                  Hx = H_mag_val * H_dir[0]
                  Hy = H_mag_val * H_dir[1]
                  Hz = H_mag_val * H_dir[2]
        
        # Per-site g-tensors (anisotropic / sublattice-dependent Zeeman). When
        # present, the field couples as mu_B * B . g_i . S_i (SpinW `addg`,
        # Sunny `Moment(g=...)`), so g_xy != g_z and per-sublattice local frames
        # (rare-earth pyrochlores) are expressible. Without them the legacy
        # global isotropic gamma*mu_B*H.S is used unchanged.
        g_tensors = self._resolve_g_tensors()

        # Resolve the field VECTOR once, from whichever form is available. Doing
        # this in one place matters: the classical-energy path (used by the
        # minimizer) binds numeric parameters, and H_dir then does not survive
        # param_map as a 3-vector. The old code fell through to a branch that
        # both ignored the g-tensor AND assumed the field pointed along z, so the
        # minimizer optimized a DIFFERENT Hamiltonian than LSWT diagonalized --
        # a wrong ground state, showing up as imaginary magnon energies.
        B_vec = None
        if Hx is not None or Hy is not None or Hz is not None:
            B_vec = [Hx or 0, Hy or 0, Hz or 0]
        elif H_mag is not None:
            if isinstance(H_mag, (list, tuple, np.ndarray)) and len(H_mag) == 3:
                B_vec = list(H_mag)
            else:
                # Scalar magnitude: take the direction from the config (static),
                # falling back to z only when no direction was given at all.
                cfg_params = self.config.get(
                    'parameters', self.config.get('model_params', {})) or {}
                d = cfg_params.get('H_dir')
                if isinstance(d, (list, tuple, np.ndarray)) and len(d) == 3:
                    B_vec = [H_mag * d[0], H_mag * d[1], H_mag * d[2]]
                else:
                    B_vec = [0, 0, H_mag]

        if B_vec is None:
            return HM

        if g_tensors is not None:
            Bm = sp.Matrix(B_vec)
            for i in range(N_uc):
                Si = sp.Matrix(Sxyz[i])
                # The legacy global term is gamma*mu_B*H.S with gamma=1, calibrated
                # (SW29) so that H_mag = B[Tesla] reproduces the electron g=2
                # Zeeman. Scaling by g/2 makes an isotropic g=2 tensor reduce
                # EXACTLY to the legacy term (asserted in tests), while g_xy != g_z
                # now works.
                HM += (mu_B / 2.0) * (Bm.T * g_tensors[i] * Si)[0]
        else:
            for i in range(N_uc):
                HM += gamma * mu_B * (
                    B_vec[0]*Sxyz[i][0] + B_vec[1]*Sxyz[i][1] + B_vec[2]*Sxyz[i][2])

        return HM

    def _resolve_g_tensors(self):
        """Per-site 3x3 g-tensors from `crystal_structure.atoms_uc[i].g`, or None.

        Returns None when NO atom declares a `g` (the legacy global isotropic
        Zeeman then applies unchanged). Otherwise every site gets a tensor;
        sites without a `g` fall back to the isotropic electron value g = 2.

        Accepted per-atom forms (values may be numbers, parameter names, or
        expressions):
          g: 2.0                                   isotropic
          g: [gxx, gyy, gzz]                       diagonal, lab frame
          g: [[...], [...], [...]]                 full 3x3, lab frame
          g: {g_par: 1.8, g_perp: 4.32,            uniaxial about a LOCAL axis:
              axis: [1, 1, 1]}                     g = g_par*zz^T + g_perp*(I - zz^T)

        The last form is what rare-earth pyrochlores need: each sublattice
        carries its own local <111> axis (SW20's Yb2Ti2O7, g_xy=4.32/g_z=1.8).
        """
        atoms = (self.config.get('crystal_structure', {}) or {}).get('atoms_uc') or []
        if not any(isinstance(a, dict) and a.get('g') is not None for a in atoms):
            return None

        param_map = self._resolve_param_map(self.parameter_order or [])
        N_uc = len(self.atom_pos())
        tensors = []
        for i in range(N_uc):
            spec = atoms[i].get('g') if i < len(atoms) and isinstance(atoms[i], dict) else None
            tensors.append(self._build_g_tensor(spec, param_map, i))
        return tensors

    def _build_g_tensor(self, spec, param_map, site_idx):
        if spec is None:
            return sp.eye(3) * 2.0  # electron g; matches the legacy calibration
        if isinstance(spec, (int, float, str)):
            return sp.eye(3) * self._resolve_scalar(spec, param_map)
        if isinstance(spec, dict):
            axis = np.asarray(spec.get('axis', [0, 0, 1]), dtype=float)
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                raise ValueError(
                    f"g-tensor `axis` must be non-zero for site {site_idx}: {spec}")
            z = sp.Matrix((axis / norm).tolist())
            g_par = self._resolve_scalar(
                spec.get('g_par', spec.get('g_z', spec.get('g_parallel'))), param_map)
            g_perp = self._resolve_scalar(
                spec.get('g_perp', spec.get('g_xy', spec.get('g_perpendicular'))), param_map)
            if g_par is None or g_perp is None:
                raise ValueError(
                    f"axial g-tensor for site {site_idx} needs both `g_par` and "
                    f"`g_perp` (aliases g_z/g_xy): {spec}")
            zzT = z * z.T
            return g_par * zzT + g_perp * (sp.eye(3) - zzT)
        if isinstance(spec, (list, tuple)):
            if len(spec) == 3 and all(not isinstance(v, (list, tuple)) for v in spec):
                vals = [self._resolve_scalar(v, param_map) for v in spec]
                return sp.diag(*vals)
            if len(spec) == 3:
                rows = [[self._resolve_scalar(v, param_map) for v in row] for row in spec]
                M = sp.Matrix(rows)
                if M.shape == (3, 3):
                    return M
        raise ValueError(
            f"Unrecognized g-tensor spec for site {site_idx}: {spec!r}. Use a scalar, "
            f"a 3-vector (diagonal), a 3x3 matrix, or {{g_par, g_perp, axis}}.")

    def _apply_substitution_and_filter(self, HM: sp.Expr, pr: List[Any]) -> sp.Expr:
        """Substitute numerical parameters and filter for quadratic terms."""
        config_params = self.config.get('parameters', self.config.get('model_params', {}))
        if config_params:
            p_names = self.parameter_order # Use stored order
            p_values_dict = config_params
            
            subs_map = {}
            # Do NOT substitute primary parameters from 'pr' (p0, p1...)
            # We keep them symbolic for the LSWT engine.
            pass
            

            
            free_syms = list(HM.free_symbols)
            for sym in free_syms:
                if sym.name in p_values_dict and sym.name not in p_names:
                     subs_map[sym] = p_values_dict[sym.name]
            
            if subs_map:
                logger.debug(f"Substituting parameters with values: {subs_map}")
                HM = HM.subs(subs_map)

        if isinstance(HM, (int, float)) and HM == 0:
            return sp.Integer(0)

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
        Jex, DM, Kex = interactions_to_numpy(Jex_sym, DM_sym, Kex_sym)
        N = Jex.shape[0]
        N_ouc = Jex.shape[1]

        # Prepare params
        params_dict = self._resolve_param_map(p_num)
        
        # Determine H vector if present (Zeeman)
        H_vec = None
        
        if 'Hx' in params_dict or 'Hy' in params_dict or 'Hz' in params_dict:
             hx = float(params_dict.get('Hx', 0.0))
             hy = float(params_dict.get('Hy', 0.0))
             hz = float(params_dict.get('Hz', 0.0))
             H_vec = np.array([hx, hy, hz])
             
        elif any(k in params_dict for k in ['H', 'H_mag', 'H_field']):
            h_val = next(params_dict[k] for k in ['H', 'H_mag', 'H_field'] if k in params_dict)
            h_dir = params_dict.get('H_dir')
            # Assume H along z if scalar, or check if vector
            if isinstance(h_val, (list, tuple, np.ndarray)):
                 H_vec = np.array(h_val, dtype=float)
            elif h_dir is not None and isinstance(h_dir, (list, tuple, np.ndarray)):
                 H_vec = np.array(h_dir, dtype=float) * float(h_val)
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
        struct_config = self.mag_struct_cfg or self.config.get('magnetic_structure')
        if not struct_config:
            return None, None

        method = struct_config.get('type')
        if method == 'explicit':
            return self._generate_structure_explicit(struct_config)
        elif method == 'single_k':
            return self._generate_structure_single_k(struct_config)
        elif method == 'propagation_vector':
            return self._generate_structure_from_k(struct_config)
        elif method == 'pattern':
            return self._generate_structure_from_pattern(struct_config)
        elif method == 'spiral':
            return self._generate_structure_spiral_local(struct_config)
        elif method == 'multi_k':
            return self._generate_structure_multi_k(struct_config)
        else:
             logger.warning(f"Unknown magnetic_structure type: {method}")
             return None, None

    def _generate_structure_multi_k(self, config):
        """Real-space multi-k structure on a commensurate magnetic supercell.

        S_i  =  sum_m  m_m * cos(2*pi * k_m . r_i + phi_m)         (then normalized)

        with r_i the site position in CHEMICAL fractional coordinates. Each
        component gives an amplitude vector `m` (Cartesian), a propagation vector
        `k` (chemical RLU) and an optional phase.

        This is a REAL-SPACE construction: it requires a magnetic supercell
        commensurate with every k (`crystal_structure.magnetic_supercell: auto`
        derives it via the per-axis LCM of the k denominators). There is no
        rotating-frame multi-k theory -- SpinW and Sunny also require a supercell
        here -- so all k must be commensurate.

        `normalize: true` (default) rescales every site to |S| = 1, which is the
        usual convention for a multi-k *spin* structure; set it false to keep a
        genuinely amplitude-modulated (sinusoidal) structure, whose sites then
        have unequal moment lengths.
        """
        comps = config.get('components') or []
        if not comps:
            raise ValueError(
                "multi_k magnetic_structure needs a `components` list of "
                "{k: [...], m: [...], phase_deg: <optional>}.")
        if self.supercell_dims == [1, 1, 1]:
            logger.warning(
                "multi_k structure without a magnetic supercell: unless every k is "
                "a reciprocal-lattice vector this is not commensurate with the cell. "
                "Set crystal_structure.magnetic_supercell: auto.")

        dims = np.asarray(self.supercell_dims, dtype=float)
        atoms = self.config['crystal_structure']['atoms_uc']
        normalize = config.get('normalize', True)

        thetas, phis = [], []
        for a in atoms:
            # Supercell-fractional -> chemical-fractional (the supercell lattice is
            # diag(dims) x chemical, so multiplying by dims undoes the rescaling).
            r_chem = np.asarray(a['pos'], dtype=float) * dims
            S = np.zeros(3)
            for c in comps:
                k = np.asarray(c['k'], dtype=float)
                m = np.asarray(c['m'], dtype=float)
                phase = np.deg2rad(float(c.get('phase_deg', 0.0)))
                S += m * np.cos(2.0 * np.pi * float(np.dot(k, r_chem)) + phase)
            norm = np.linalg.norm(S)
            if norm < 1e-9:
                raise ValueError(
                    f"multi_k: site {a.get('label')} has zero net moment "
                    f"(the components cancel there). Adjust the phases/amplitudes.")
            if normalize:
                S = S / norm
            thetas.append(float(np.arccos(np.clip(S[2] / np.linalg.norm(S), -1, 1))))
            phis.append(float(np.arctan2(S[1], S[0])))
        logger.info(
            f"multi_k structure: {len(comps)} components on a "
            f"{self.supercell_dims} supercell ({len(atoms)} sites).")
        return thetas, phis

    def _generate_structure_single_k(self, config):
        """Structure angles for the unified single-k (propagation-vector) type.

        Spin-direction input, exactly one of:
          - ``local_directions``: rotating-frame directions (legacy 'spiral'
            semantics), applied cyclically over the sites;
          - ``S0``: lab-frame directions of the cell-0 spins (SpinW genmagstr
            'helical' / Sunny convention). Back-rotated per site by
            R(-2*pi*k.r_frac_i, axis) into the rotating frame;
          - ``u``/``v`` basis (legacy 'propagation_vector' semantics):
            S0_i = u*cos(phi_i) + v*sin(phi_i), phi_i = 2*pi*k.r_frac_i.
        ``cone_angle_deg`` > 0 makes a conical structure:
        S = cos(c)*axis + sin(c)*u_i with u_i the in-plane component.

        Returns lab-frame angles when ``real_space: true`` (no rotating frame,
        legacy propagation_vector semantics); rotating-frame angles otherwise.
        """
        atoms = self.atom_pos()
        nspins = len(atoms)
        inv_uc = la.inv(self.unit_cell())
        r_frac = np.asarray(atoms, dtype=float) @ inv_uc

        k = np.asarray(config.get('k', [0, 0, 0]), dtype=float)
        axis = np.asarray(config.get('axis', [0, 0, 1]), dtype=float)
        n = axis / (np.linalg.norm(axis) or 1.0)
        do_normalize = config.get('normalize', True)
        real_space = bool(config.get('real_space', False))
        phases = 2.0 * np.pi * (r_frac @ k)

        # --- Resolve per-site rotating-frame directions u_i ---
        local_dirs = None
        if config.get('local_directions') or config.get('directions'):
            dirs = config.get('local_directions', config.get('directions'))
            local_dirs = [np.asarray(dirs[i % len(dirs)], dtype=float) for i in range(nspins)]
        elif config.get('S0'):
            S0 = config['S0']
            if len(S0) == 1:
                S0 = S0 * nspins
            if len(S0) != nspins:
                raise ValueError(
                    f"magnetic_structure.S0 must list 1 or {nspins} directions, got {len(S0)}"
                )
            local_dirs = [
                rotation_about_axis(-phases[i], n) @ np.asarray(S0[i], dtype=float)
                for i in range(nspins)
            ]
        elif config.get('u') is not None or config.get('v') is not None:
            u_vec = np.asarray(config.get('u', [1, 0, 0]), dtype=float)
            v_vec = np.asarray(config.get('v', [0, 1, 0]), dtype=float)
            S0 = [u_vec * np.cos(phases[i]) + v_vec * np.sin(phases[i]) for i in range(nspins)]
            local_dirs = [rotation_about_axis(-phases[i], n) @ S0[i] for i in range(nspins)]
        else:
            # Default: one common in-plane direction perpendicular to the axis.
            trial = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = trial - np.dot(trial, n) * n
            u /= np.linalg.norm(u)
            local_dirs = [u.copy() for _ in range(nspins)]

        # --- Cone angle (deviation from the axis) ---
        cone_deg = float(config.get('cone_angle_deg', 0.0))
        if cone_deg > 0.0:
            c = np.radians(cone_deg)
            coned = []
            for v in local_dirs:
                u_in = v - np.dot(v, n) * n  # in-plane component
                nrm = np.linalg.norm(u_in)
                if nrm < 1e-12:
                    raise ValueError(
                        "Conical single_k structure needs directions with a "
                        "non-zero in-plane component."
                    )
                coned.append(np.cos(c) * n + np.sin(c) * (u_in / nrm))
            local_dirs = coned
        elif not real_space:
            # Planar spiral: rotating-frame directions must be perpendicular to
            # the axis, otherwise LSWT is built on a non-ground state and the
            # phason mode is spuriously gapped.
            for i, v in enumerate(local_dirs):
                nrm = np.linalg.norm(v)
                if nrm > 1e-12 and abs(np.dot(v / nrm, n)) > 1e-6:
                    logger.warning(
                        "single_k: direction for site %d is not perpendicular "
                        "to the rotation axis; the phason mode will be "
                        "spuriously gapped.", i
                    )

        # --- Emit angles ---
        thetas, phis = [], []
        for i in range(nspins):
            v = local_dirs[i]
            if real_space:
                # Lab-frame structure: apply the k.r phase to each site.
                v = rotation_about_axis(phases[i], n) @ v
            if do_normalize:
                nrm = np.linalg.norm(v)
                if nrm > 1e-12:
                    v = v / nrm
            thetas.append(float(np.arccos(np.clip(v[2], -1.0, 1.0))))
            phis.append(float(np.arctan2(v[1], v[0])))
        return thetas, phis

    def _generate_structure_spiral_local(self, config):
        """Local (rotating-frame) structure for a single-k spiral.

        For ``type: spiral`` the interactions are transformed into the
        rotating frame by ``_compute_rotated_interactions`` (each bond picks
        up R(2*pi*k.r_ij) about ``axis``), so the spin directions stored on
        the model are the LOCAL-frame ones. For a planar spiral these must
        lie in the plane PERPENDICULAR to the rotation axis; without this the
        LSWT is built on a non-ground state and the phason is spuriously
        gapped. Optional ``local_directions`` (one per spin, applied
        cyclically) sets the intra-cell pattern (e.g. a 120-degree
        arrangement); the default is a single common direction perpendicular
        to ``axis``.
        """
        atoms = self.atom_pos()
        nspins = len(atoms)
        axis = np.array(config.get('axis', [0, 0, 1]), dtype=float)
        n = axis / (np.linalg.norm(axis) or 1.0)
        dirs = config.get('local_directions', config.get('directions'))
        if not dirs:
            trial = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = trial - np.dot(trial, n) * n
            u /= np.linalg.norm(u)
            dirs = [u.tolist()]
        thetas, phis = [], []
        for i in range(nspins):
            v = np.array(dirs[i % len(dirs)], dtype=float)
            v /= np.linalg.norm(v)
            thetas.append(float(np.arccos(np.clip(v[2], -1.0, 1.0))))
            phis.append(float(np.arctan2(v[1], v[0])))
        return thetas, phis

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
        elif pattern in ('antiferromagnetic', 'generic', 'custom'):
             # A list of spin-direction unit vectors applied to the spins.
             #   - 'antiferromagnetic': typically two sublattice directions applied
             #     cyclically (defaults to a Neel pattern if none are given).
             #   - 'generic'/'custom': one direction per spin (e.g. a structure
             #     imported from an energy minimization). Applied 1:1 when the list
             #     length matches the spin count, otherwise cyclically.
             directions = config.get('directions', [])
             if not directions:
                 # Default Neel for 2 sublattices (only sensible for AFM).
                 directions = [[0, 0, 1], [0, 0, -1]]

             for i in range(nspins):
                 d = directions[i % len(directions)]
                 v = np.array(d, dtype=float)
                 norm = np.linalg.norm(v)
                 if norm > 1e-9:
                      v /= norm
                      th = np.arccos(np.clip(v[2], -1.0, 1.0))
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
                       
                  # Anisotropic (Kex) - Full 3x3 support
                  K_mat = Kex[i, j]
                  if np.any(np.abs(K_mat) > 1e-9):
                       # Si^T * K * Sj
                       term = np.dot(Si, np.dot(K_mat, Sj))
                       E += 0.5 * term * (S_val * S_val)
                       
        return E
