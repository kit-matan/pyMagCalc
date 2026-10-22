
import numpy as np
import spglib
import yaml
import logging
from typing import List, Dict, Tuple, Optional, Union, Any

# Configure logger
logger = logging.getLogger(__name__)

# Reference-bond resolution tolerances (Angstrom).
# DEGEN: two images this close in length are the same bond length physically;
# the choice between them must never be made by floating-point noise.
# DIST: window a `distance:` key must match within. It has to stay loose enough
# for hand-typed distances, which is why a window spanning two distinct bond
# lengths is an error rather than a silent first-match.
_REF_BOND_DEGEN_TOL = 1e-6
_REF_BOND_DIST_TOL = 0.05

# Rule types whose `value` carries a direction, so that *which* member of a
# degenerate set of reference bonds is picked changes the Hamiltonian:
# D_ji = -D_ij, and symmetry-related bonds carry point-group images of D.
_DIRECTIONAL_RULE_TYPES = frozenset({
    "dm", "dm_manual", "anisotropic_exchange", "interaction_matrix", "kitaev",
})

_CLOSE_BRACKET = {"[": "]", "(": ")"}


def _bracket_partner(s, start):
    """Index of the bracket closing the one at `s[start]`, in a balanced `s`."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] in _CLOSE_BRACKET:
            depth += 1
        elif s[i] in ")]":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_rule_value(value):
    """
    A string rule `value` as its list of component strings ("0,-Dy,Dz",
    "[0, -Dy, Dz]", "(0, -Dy, Dz)"; a "[[..], [..], [..]]" string gives rows), or
    `value` unchanged when it holds a single expression.

    Only ONE bracket pair enclosing the whole value is removed, and only commas at
    bracket depth 0 split it, so every component reaches sympify intact. Until
    2026-10-18 every `[`, `]`, `(` and `)` in the string was deleted before
    splitting on every comma: `sqrt(3)*D1x/3` became `sqrt3*D1x/3`, a new free
    symbol `sqrt3` -- and that is exactly the form `magcalc symmetry` prints for
    the allowed DM on any hexagonal bond (KFe3J's kagome NN bond: (D1x,
    sqrt(3)*D1x/3, D1z)). Nothing binds `sqrt3`, so the bond table carried an
    unbound symbol and the dispersion came out all-NaN; `(J1+J2)/2, 0, 0` lost its
    grouping and `atan2(Dy, Dx)` split in two.
    """
    s = value.strip()
    stack = []
    for c in s:
        if c in _CLOSE_BRACKET:
            stack.append(_CLOSE_BRACKET[c])
        elif c in ")]" and (not stack or stack.pop() != c):
            stack.append(None)
            break
    if stack:
        raise ValueError(f"unbalanced brackets in rule value {value!r}")

    inner = s
    if s[:1] in _CLOSE_BRACKET and _bracket_partner(s, 0) == len(s) - 1:
        inner = s[1:-1]

    parts, depth, start = [], 0, 0
    for i, c in enumerate(inner):
        if c in _CLOSE_BRACKET:
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(inner[start:i].strip())
            start = i + 1
    parts.append(inner[start:].strip())
    if len(parts) == 1:
        return value                # one expression, e.g. "J1" or "(J1+J2)/2"
    if parts[-1] == "":
        parts.pop()                 # a trailing comma
    if any(p == "" for p in parts):
        # Dropping it would shift every later component one axis over.
        raise ValueError(f"empty component in rule value {value!r}")

    rows = [p.startswith("[") and _bracket_partner(p, 0) == len(p) - 1 for p in parts]
    if all(rows):
        return [_split_rule_value(p) for p in parts]
    return parts


#: How far an allowed-matrix coefficient may sit from an exact value and still
#: be snapped to it. Allowed-matrix coefficients come from point-group algebra,
#: so they are O(1) ratios like 1, 1/2, sqrt(3)/2 -- there is no such thing as a
#: genuine coefficient of 3e-7. What produces one is lattice noise: a CIF that
#: is cubic to six decimals gives rotation matrices that are off by ~1e-6, and
#: those propagate here. `detect_symmetry_from_structure` already accepted the
#: space group at symprec = 1e-3, so snapping two orders of magnitude tighter
#: than that adds no assumption it has not already made.
_SYMMETRY_COEFF_TOL = 1e-5


#: Relative tolerance of the propagation-consistency check in
#: `add_symmetry_interaction` on an exactly specified lattice (relative to the
#: largest entry of the value). It is widened to `_PROPAGATION_NOISE_FACTOR`
#: times the lattice's own noise -- the non-orthogonality of the Cartesian
#: symmetry operations, ~1e-6 for a Materials Project cell that is cubic to six
#: decimals -- so measurement noise is never reported as a symmetry conflict,
#: while on an exact lattice a forbidden component is caught down to 1e-8.
_PROPAGATION_TOL = 1e-8
_PROPAGATION_NOISE_FACTOR = 100.0


class SymmetryConflictError(ValueError):
    """A `symmetry_rules` value that the reference bond's own site symmetry forbids.

    Raised by `MagCalcConfigBuilder.add_symmetry_interaction` when the space-group
    operations reaching one bond propagate different values to it -- a DM component
    forbidden by Moriya's rules, a symmetric off-diagonal the bond's 2-fold axis
    excludes, and so on. No consistent bond table exists for such a value, so none
    is built; the message names the component and the operation that excludes it.
    """


def _op_to_string(R, t):
    """A space-group operation as its Jones-faithful symbol, e.g. '-x,-y+1/2,z+1/4'."""
    from fractions import Fraction

    out = []
    for row, tr in zip(np.asarray(R), np.asarray(t, dtype=float)):
        s = ""
        for c, name in zip(row, "xyz"):
            c = int(round(float(c)))
            if c == 0:
                continue
            sign = "-" if c < 0 else ("+" if s else "")
            s += sign + (name if abs(c) == 1 else f"{abs(c)}{name}")
        fr = Fraction(float(tr)).limit_denominator(24)
        if fr != 0:
            s += ("+" if fr > 0 and s else "") + str(fr)
        out.append(s or "0")
    return ",".join(out)


def _chop_expr(x, tol=1e-10):
    """A propagated sympy expression with its float noise removed, for display only:
    `1.0*Dx + 5.4e-33*Dy - 1.2e-16*Dz` reads as `Dx`."""
    import sympy as sp

    if not isinstance(x, sp.Basic):
        return x
    repl = {}
    for f in x.atoms(sp.Float):
        if abs(f) < tol:
            repl[f] = sp.Integer(0)
        elif abs(f - round(f)) < tol:
            repl[f] = sp.Integer(int(round(f)))
    return x.xreplace(repl) if repl else x


def _fmt_rule_value(v):
    """One rule value (3-vector or 3x3; numeric or sympy) for an error message."""
    def one(x):
        if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
            return f"{float(x):.6g}"
        return str(_chop_expr(x))

    v = list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v]
    if v and isinstance(v[0], (list, tuple, np.ndarray)):
        return "[" + ", ".join("[" + ", ".join(one(x) for x in row) + "]" for row in v) + "]"
    return "(" + ", ".join(one(x) for x in v) + ")"


def _propagate_numeric(rule_type, R_cart, det_R, V):
    """
    (forward, reverse) numeric values of one bond rule under a Cartesian operation.

    `forward` is what the operation carries the reference value to; `reverse` is
    the same bond read the other way (D_ji = -D_ij, J_ji = J_ij^T, a diagonal is
    its own reverse). These are the exact expressions the emitted table has
    always used, so a symmetry-allowed value propagates bit-identically.
    """
    V = np.asarray(V, dtype=float)
    if rule_type == "dm":
        # DM Vector: D' = det(R) * R * D (Axial vector)
        fwd = det_R * (R_cart @ V)
        return fwd, -np.array(fwd)
    if rule_type == "anisotropic_exchange":
        # diagonal tensor J' = R J R^T, diagonal kept
        J_prime = R_cart @ np.diag(V) @ R_cart.T
        fwd = np.diag(J_prime)
        return fwd, fwd
    if rule_type == "interaction_matrix":
        # Full 3x3 interaction tensor J' = R J R^T
        J_prime = R_cart @ V @ R_cart.T
        return J_prime, J_prime.T
    raise ValueError(f"no numeric propagation for rule type {rule_type!r}")


def _propagate_symbolic(rule_type, R_cart, det_R, V):
    """Sympy twin of `_propagate_numeric` (lists of expressions, as emitted)."""
    import sympy as sp

    R_sym = sp.Matrix(R_cart)
    if rule_type == "dm":
        val_p = det_R * R_sym * V
        fwd = [val_p[0], val_p[1], val_p[2]]
        return fwd, [-v for v in fwd]
    if rule_type == "anisotropic_exchange":
        J_prime = R_sym * sp.diag(*V) * R_sym.T
        fwd = [J_prime[0, 0], J_prime[1, 1], J_prime[2, 2]]
        return fwd, fwd
    if rule_type == "interaction_matrix":
        J_prime = R_sym * sp.Matrix(V) * R_sym.T
        return J_prime.tolist(), J_prime.T.tolist()
    raise ValueError(f"no symbolic propagation for rule type {rule_type!r}")


def _rule_value_samples(val_vec, is_symbolic, n_points=2, seed=20261018):
    """
    Numeric evaluations of a rule's reference value, for the consistency check.

    A numeric value is its own single sample. A symbolic one is evaluated at
    `n_points` random points of its free parameters (fixed seed), so the check
    sees the structure of the value rather than any particular number: a
    forbidden `Dz` is caught whether Dz is later 1 meV or 1e-3.
    """
    if not is_symbolic:
        return [np.asarray(val_vec, dtype=float)]
    import sympy as sp

    def as_array(M):
        arr = np.array(sp.Matrix(M).evalf().tolist(), dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr

    syms = sorted(val_vec.free_symbols, key=str)
    if not syms:
        return [as_array(val_vec)]
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.5, 1.5, size=(n_points, len(syms)))
    return [as_array(val_vec.subs(dict(zip(syms, pt)))) for pt in pts]


def _rref(A, tol=_SYMMETRY_COEFF_TOL):
    """
    Reduced row echelon form with partial pivoting, plus the pivot columns.

    Used to turn an (arbitrary) orthonormal null-space basis into the canonical
    one in which each basis vector owns a distinct entry of J -- which is what
    makes "j3" in the reported allowed matrix mean "the (1,0) entry is free".
    """
    A = np.array(A, dtype=float, copy=True)
    m, n = A.shape
    pivots = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        k = r + int(np.argmax(np.abs(A[r:, c])))
        if abs(A[k, c]) < tol:
            continue
        A[[r, k]] = A[[k, r]]
        A[r] = A[r] / A[r, c]
        for i in range(m):
            if i != r and abs(A[i, c]) > 0:
                A[i] = A[i] - A[i, c] * A[r]
        pivots.append(c)
        r += 1
    A[np.abs(A) < tol] = 0.0
    return A[:r], pivots


def _nearest_orthogonal(R):
    """
    The orthogonal matrix closest to `R`, via the polar decomposition.

    A crystallographic point-group operation IS orthogonal in Cartesian
    coordinates; `A R_frac A^-1` only fails to be so because the lattice A
    carries measurement noise. A real CIF has plenty -- the Materials Project
    NiO primitive cell is cubic to 6 decimals, not exactly -- and that noise
    propagates straight into the allowed exchange matrix, which is then reported
    with 1.41421377829 where sqrt(2) belongs and with ties that do not quite
    close. spglib has already decided (within its symprec) that this op is in
    the group, so idealising it is consistent with that decision, not an extra
    assumption on top of it.
    """
    U, _s, Vt = np.linalg.svd(np.asarray(R, dtype=float))
    return U @ Vt


#: Irrationals a crystallographic rotation matrix actually produces, so that a
#: tie like 1/sqrt(3) prints as `sqrt(3)/3` rather than `0.57735`.
_SYMMETRY_CONSTANTS = None


def _nice_coefficient(x, tol=_SYMMETRY_COEFF_TOL):
    """
    A float coefficient of the allowed-matrix basis, as an exact sympy number
    when it plainly is one, and as a plain Float otherwise.

    Deliberately narrow: it snaps only to simple rationals and to the handful of
    surds a point-group rotation can produce. Anything else is left as a float,
    because a wrong "exact" value here is precisely the failure this replaced.
    """
    import sympy as sp

    global _SYMMETRY_CONSTANTS
    if _SYMMETRY_CONSTANTS is None:
        _SYMMETRY_CONSTANTS = [sp.sqrt(2), sp.sqrt(3), sp.sqrt(6),
                               sp.sqrt(2) / 2, sp.sqrt(3) / 2, sp.sqrt(3) / 3,
                               sp.sqrt(6) / 3, sp.sqrt(2) / 4, sp.sqrt(6) / 4]

    x = float(x)
    rat = sp.Rational(x).limit_denominator(24)
    if abs(float(rat) - x) < tol:
        return rat
    for const in _SYMMETRY_CONSTANTS:
        c = float(const)
        for sign in (1, -1):
            for mult in (sp.Integer(1), sp.Rational(1, 2), sp.Rational(1, 3),
                         sp.Rational(3, 2), sp.Integer(2)):
                cand = sign * float(mult) * c
                if abs(cand - x) < tol:
                    return sign * mult * const
    return sp.Float(x, 12)


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
                "kitaev": [],
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
        # No dimensionality state needed anymore, everything is 3D

        
        # Validation mapping
        self._atom_label_to_idx = {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MagCalcConfigBuilder':
        """Hydrate a builder (lattice, expanded atoms, symmetry ops) from a run config
        dict -- the structural half of what `GenericSpinModel` does, without the
        interactions/model. Used by the `magcalc symmetry` CLI and any caller that wants
        the crystal-symmetry analysis (`analyze_bond_symmetry`, `get_bond_constraints`).
        """
        b = cls()
        cs = config.get('crystal_structure', {}) or {}
        lp = cs.get('lattice_parameters') or {}
        sg = lp.get('space_group')

        if lp:
            b.set_lattice(a=lp.get('a', 1.0), b=lp.get('b', lp.get('a', 1.0)),
                          c=lp.get('c', lp.get('a', 1.0)), alpha=lp.get('alpha', 90.0),
                          beta=lp.get('beta', 90.0), gamma=lp.get('gamma', 90.0),
                          space_group=int(sg) if sg is not None else None)
        elif cs.get('lattice_vectors') is not None:
            b.lattice_vectors = np.array(cs['lattice_vectors'], dtype=float)
            b.config['crystal_structure']['lattice_vectors'] = b.lattice_vectors.tolist()
            if sg is not None:
                b.space_group_number = int(sg)
                b._load_symmetry_ops(int(sg))

        # Atoms. Explicit atoms_uc are the full cell already; wyckoff_atoms need the
        # orbit expanded over the loaded space group (add_wyckoff_atom does that).
        atom_mode = cs.get('atom_mode')
        explicit_atoms = cs.get('atoms_uc')
        wyckoff = cs.get('wyckoff_atoms')
        if atom_mode == 'explicit' or (explicit_atoms and not wyckoff):
            atoms = explicit_atoms or wyckoff or []
            b.atoms_uc = [dict(a) for a in atoms]
            b._atom_label_to_idx = {a.get('label', f'Atom{i}'): i
                                    for i, a in enumerate(b.atoms_uc)}
        else:
            for a in (wyckoff or []):
                b.add_wyckoff_atom(label=a.get('label', 'Atom'), pos=a.get('pos', [0, 0, 0]),
                                   spin=a.get('spin_S', 0.5),
                                   species=a.get('element', a.get('label', 'Atom')),
                                   ion=a.get('ion'), element=a.get('element'),
                                   charge=a.get('charge'))

        # No space group given (lattice_vectors path): detect it from the structure so
        # the analyzer has operations to work with.
        if not b.symmetry_ops and b.atoms_uc:
            b.detect_symmetry_from_structure()
        return b

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

    def detect_symmetry_from_structure(self, symprec: float = 1e-3):
        """
        Detect symmetry operations from the current structure using spglib.
        Useful when lattice_vectors are provided but space_group is not.
        """
        if not self.atoms_uc:
             logger.warning("No atoms defined. Cannot detect symmetry.")
             return

        # Prepare spglib cell input: (lattice, positions, numbers)
        positions = np.array([a['pos'] for a in self.atoms_uc], dtype=float)
        
        # Numbers: assign dummy atomic numbers based on species/element/ion.
        # Fall back to the label with trailing digits stripped, so per-site
        # labels (Fe0, Fe1, ...) don't degrade the detected group to P1.
        def _species_key(a):
            for k in ('species', 'element', 'ion'):
                if a.get(k):
                    return str(a[k])
            import re
            return re.sub(r'\d+$', '', str(a.get('label', 'X'))) or 'X'
        species_list = [_species_key(a) for a in self.atoms_uc]
        unique_species = sorted(list(set(species_list)))
        species_map = {s: i+1 for i, s in enumerate(unique_species)}
        numbers = np.array([species_map[s] for s in species_list], dtype=int)
        
        lattice = self.lattice_vectors
        
        cell = (lattice, positions, numbers)
        
        # Get symmetry
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        
        if dataset:
             self.space_group_number = dataset.number
             # Extract rotations and translations
             rotations = dataset.rotations
             translations = dataset.translations
             self.symmetry_ops = {'rotations': rotations, 'translations': translations}
             logger.info(f"Detected Space Group: {dataset.number} ({dataset.international})")
             logger.info(f"Loaded {len(rotations)} symmetry operations.")
        else:
             logger.warning("Spglib failed to detect symmetry.")

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
                                 distance: float = None, offset: List[int] = None,
                                 axis: str = None):
        """
        Add an interaction and propagate it by symmetry.
        Especially for DM (dm_manual) and Anisotropic Exchange.
        
        :param value: The interaction value (e.g. D vector [Dx, Dy, Dz] or matrix).
        :param offset: Optional explicit offset [u, v, w] for the reference bond j relative to i.
        """
        if ref_pair is None:
            raise ValueError(
                f"symmetry rule type '{type}' requires a `ref_pair` (one reference bond, "
                f"propagated over the detected space group). Only `type: heisenberg` may be "
                f"given as a bare `distance` rule; a non-scalar rule without `ref_pair` would "
                f"otherwise expand to zero bonds and silently drop the interaction."
            )

        if not self.symmetry_ops:
            raise ValueError(
                f"No symmetry operations loaded, cannot propagate the '{type}' rule for "
                f"{ref_pair}. Provide a `space_group` in `lattice_parameters`, or list the "
                f"bonds explicitly."
            )

        # 1. Identify Reference Bond Vector
        def _resolve_atom_idx(lbl):
            if lbl in self._atom_label_to_idx:
                return self._atom_label_to_idx[lbl]
            # Fallback: try appending "0" (common renaming pattern for single/multiplicity handling)
            lbl_0 = f"{lbl}0"
            if lbl_0 in self._atom_label_to_idx:
                return self._atom_label_to_idx[lbl_0]
            
            # Robust normalization: Match "Label" + "Index" and try to align
            # e.g. Cu01 -> Cu1, Cu00 -> Cu0
            try:
                import re
                match = re.match(r"([A-Za-z]+)(\d+)", lbl)
                if match:
                    prefix, num = match.groups()
                    normalized = f"{prefix}{int(num)}"
                    if normalized in self._atom_label_to_idx:
                        return self._atom_label_to_idx[normalized]
            except Exception:
                pass

            raise KeyError(f"Atom '{lbl}' not found in unit cell (keys: {list(self._atom_label_to_idx.keys())})")

        idx_i = _resolve_atom_idx(ref_pair[0])
        idx_j = _resolve_atom_idx(ref_pair[1])

        pos_i = np.array(self.atoms_uc[idx_i]["pos"])
        pos_j_uc = np.array(self.atoms_uc[idx_j]["pos"])
        
        best_offset = None
        if offset is not None:
            best_offset = np.array(offset)
        else:
            # Find the specific image of j that matches "distance" (or assume
            # nearest). We need the relative vector r_ij in CARTESIAN to check
            # distance, but fractional for symmetry.
            #
            # Every candidate is enumerated before one is chosen: a first-match
            # `break` (or a running minimum with a strict `<`) resolves ties by
            # enumeration order and by ULP-level rounding in the Cartesian
            # distance, so a symmetry-degenerate pair of reference bonds would
            # be picked between silently and unreproducibly.

            # Sized from the target distance so a reference bond beyond one
            # cell image is still found (an explicit `offset` skips this).
            offsets_to_check = self._offset_shell(distance)

            images = []
            for off in offsets_to_check:
                off_vec = np.array(off)
                pos_j_img = pos_j_uc + off_vec

                # Cartesian distance
                d_vec_cart = (pos_j_img - pos_i) @ self.lattice_vectors
                dist_img = float(np.linalg.norm(d_vec_cart))

                if dist_img > 0.01:
                    images.append((dist_img, tuple(int(v) for v in off)))

            if images:
                if distance:
                    candidates = [im for im in images
                                  if abs(im[0] - distance) < _REF_BOND_DIST_TOL]
                else:
                    d_min = min(im[0] for im in images)
                    candidates = [im for im in images
                                  if im[0] - d_min < _REF_BOND_DEGEN_TOL]

                if candidates:
                    # Canonical order: shortest bond, then the image closest to
                    # the home cell, then lexicographic. Rounding the distance
                    # before sorting keeps exactly-degenerate images from being
                    # ordered by floating-point noise.
                    candidates.sort(key=lambda im: (round(im[0], 9),
                                                    sum(abs(v) for v in im[1]),
                                                    im[1]))
                    win_dist, win_off = candidates[0]
                    best_offset = np.array(win_off)

                    lengths = sorted({round(im[0], 6) for im in candidates})
                    tied = [im[1] for im in candidates
                            if abs(im[0] - win_dist) < _REF_BOND_DEGEN_TOL]

                    if len(lengths) > 1:
                        # Distinct bond lengths inside the match window: these
                        # are different orbits, so the rule would attach `value`
                        # to whichever the search happened to reach first. Wrong
                        # for every rule type, scalar ones included.
                        raise ValueError(
                            f"Ambiguous reference bond for {ref_pair} "
                            f"(rule type '{type}'): distance={distance} matches "
                            f"{len(lengths)} distinct bond lengths {lengths} A "
                            f"within the {_REF_BOND_DIST_TOL} A window, i.e. more "
                            f"than one symmetry orbit. Give an exact `distance:` "
                            f"or pin the reference bond with an explicit "
                            f"`offset: [u, v, w]`. Candidate images: "
                            f"{[im[1] for im in candidates]}."
                        )

                    if len(tied) > 1 and str(type).lower() in _DIRECTIONAL_RULE_TYPES:
                        # Same length, several images: one symmetry orbit, so a
                        # scalar J is unaffected, but a direction-carrying value
                        # is not. Attaching D to the other member realizes a
                        # point-group image of it (for a 2_1 screw or glide,
                        # D -> -C2.D, i.e. a sign flip on one component) with no
                        # visible symptom - identical bond count, plausible
                        # spectrum, silently different Hamiltonian.
                        raise ValueError(
                            f"Ambiguous reference bond for {ref_pair} "
                            f"(rule type '{type}'): {len(tied)} symmetry-equivalent "
                            f"images are all at d = {win_dist:.6f} A, offsets "
                            f"{tied}. Which one carries `value` changes the sign "
                            f"or orientation of the propagated vector, so the "
                            f"choice cannot be made automatically. Pin it with an "
                            f"explicit `offset: [u, v, w]` "
                            f"(e.g. `offset: {list(tied[0])}`)."
                        )

        if best_offset is None:
             raise ValueError(f"Could not identify reference bond for {ref_pair} near distance {distance}")

        # Final bond distance: actual length of the chosen reference bond
        # (an explicit `offset` skips the distance search, so it cannot be
        # derived from the search's running minimum).
        final_dist = float(np.linalg.norm(
            ((pos_j_uc + best_offset) - pos_i) @ self.lattice_vectors))

        # Reference Bond
        ref_frac_diff = (pos_j_uc + best_offset) - pos_i # vector from i to j
        
        # A Kitaev term IS an interaction matrix: K on one Cartesian diagonal entry
        # and nothing else. Propagating it means R K R^T, which is in general no
        # longer of Kitaev form -- on a honeycomb the C3 about [111] permutes the
        # axes, so the z-bond's diag(0,0,K) becomes the x- and y-bonds' diag(K,0,0)
        # and diag(0,K,0), which is precisely the Kitaev model. So convert here and
        # let the tested `interaction_matrix` transform do the work.
        #
        # Until 2026-08 there was no `kitaev` branch in the dispatch below at all:
        # the rule ran the whole reference-bond search, looped over every symmetry
        # op, and added ZERO bonds without a word -- while CLAUDE.md documented
        # `ref_pair` as REQUIRED for exactly this type.
        if str(type).lower() == "kitaev":
            k_axis = str(axis or "z").lower()
            if k_axis not in ("x", "y", "z"):
                raise ValueError(
                    f"kitaev symmetry rule for {ref_pair} has axis {k_axis!r}; "
                    f"expected one of 'x', 'y', 'z' (Cartesian spin components)")
            if isinstance(value, (list, tuple, np.ndarray)):
                raise ValueError(
                    f"kitaev symmetry rule for {ref_pair} expects a SCALAR `value` "
                    f"(the coupling K) plus an `axis`; got {value!r}. For a full "
                    f"tensor use `type: interaction_matrix`.")
            a = {"x": 0, "y": 1, "z": 2}[k_axis]
            mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            mat[a][a] = value
            value = mat
            type = "interaction_matrix"

        # Check if value is symbolic (contains strings)
        if isinstance(value, str):
            # Parse CSV string if needed (e.g. "0,-Dy,Dz" or "[0, -Dy, 0]"),
            # keeping brackets inside components: sqrt(3)*D1x/3 stays intact.
            value = _split_rule_value(value)

            # Auto-pad 2D vectors to 3D if likely intended (e.g. "0, -Dy" -> "0, -Dy, 0")
            # Only if it looks like a vector (list of length 2)
            if isinstance(value, list) and len(value) == 2:
                value.append("0")
        
        if isinstance(value, (list, tuple, np.ndarray)):
            # Flatten to check for strings
            flat_val = np.array(value).flatten()
            is_symbolic = any(isinstance(v, str) for v in flat_val)
        else:
            is_symbolic = isinstance(value, str)
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
        
        # 2. Apply Symmetry to find Orbit -- in two passes.
        #
        # Pass 1 records what EVERY operation does to the reference bond: the
        # directed bond it lands on and the value it carries there. Pass 2 checks
        # that operations landing on the same bond agree. Only then is the table
        # emitted, in operation order, keeping the first operation's value for
        # each bond exactly as before -- so a value the symmetry allows produces
        # a bit-identical table, and one it forbids produces an error instead of
        # a Hamiltonian of lower symmetry than the crystal.
        #
        # Until 2026-10-18 there was no pass 2. The first operation to reach a
        # bond won, and when the bond's own site symmetry forbids part of the
        # value (Moriya's rules for a DM vector) the operations reaching it
        # disagree on the sign of that part, so the winner depended on spglib's
        # enumeration order. Measured on alpha-Cu2V2O7 (Fdd2, the J2 bond, whose
        # midpoint lies on the 2-fold axis along c): a D_z on that bond is
        # forbidden, yet the engine placed D_z = +-1 on all 8 bonds with signs
        # that differed from a hand propagation on 8 of 16 directed entries and
        # were NOT invariant under the F-centring translations (32 of 48
        # translated entries differed) -- a bond table with lower symmetry than
        # the lattice, whose ground state then broke translation symmetry (spins
        # on centring-related sites differed by up to 0.12). Nothing downstream
        # can notice: the bond count is right and the spectrum is plausible.
        user_type = str(type)
        rot_ops = self.symmetry_ops['rotations']
        trans_ops = self.symmetry_ops['translations']
        ref_key = (self.atoms_uc[idx_i]['label'], self.atoms_uc[idx_j]['label'],
                   tuple(int(v) for v in best_offset))

        # Numeric evaluations of the reference value for the check. A symbolic
        # value is sampled at random points of its parameters, so the check is
        # on the STRUCTURE of the value (a forbidden `Dz` is caught whatever
        # number Dz is later given), and the transform is linear, so sampling
        # before propagating equals propagating before sampling.
        samples = None
        if type != "heisenberg":
            try:
                samples = _rule_value_samples(val_vec, is_symbolic)
            except (TypeError, ValueError) as e:  # pragma: no cover - exotic value
                logger.warning(
                    f"Cannot evaluate the '{type}' rule value {value!r} for {ref_pair} "
                    f"numerically ({e}); skipping the symmetry-consistency check.")

        hits = []
        lattice_noise = 0.0
        for gi, (R, t) in enumerate(zip(rot_ops, trans_ops)):
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

             d_final_vec = (pos_l_uc + offset_final - pos_k_uc) @ self.lattice_vectors
             d_final = np.linalg.norm(d_final_vec)
             if abs(d_final - final_dist) > 0.1:
                 # If we have a distance mismatch, it's a configuration error (e.g. wrong reference bond).
                 # We log a warning so the user can correct their input.
                 logger.warning(f"Symmetry propagation distance mismatch: {atom_k['label']}->{atom_l['label']} dist={d_final:.4f} (target {final_dist:.4f})")
                 continue

             # 6. Transform Value
             if type == "heisenberg":
                 # Scalar: value doesn't change with rotation
                 fwd, rev = value, value
                 fwd_chk = rev_chk = None
             elif type in ("dm", "anisotropic_exchange", "interaction_matrix"):
                 # DM vector: D' = det(R) R D (axial vector); tensors: J' = R J R^T.
                 R_cart = self._lattice_rotation(R)
                 det_R = float(np.linalg.det(R))
                 if is_symbolic:
                     fwd, rev = _propagate_symbolic(type, R_cart, det_R, val_vec)
                 else:
                     fwd, rev = _propagate_numeric(type, R_cart, det_R, val_vec)
                 fwd_chk = rev_chk = None
                 if samples is not None:
                     pairs = [_propagate_numeric(type, R_cart, det_R, s) for s in samples]
                     fwd_chk = [p[0] for p in pairs]
                     rev_chk = [p[1] for p in pairs]
                     lattice_noise = max(lattice_noise, float(np.abs(
                         R_cart @ R_cart.T - np.eye(3)).max()))
             else:
                 # No branch matched, so this op contributes no bond -- and if NO op
                 # ever matches, the rule expands to nothing and the term is simply
                 # absent from H. That is how `type: kitaev` went missing for as long
                 # as it existed. Fail loudly instead of returning a Hamiltonian that
                 # is quietly short a term.
                 raise ValueError(
                     f"symmetry rule type '{type}' has no propagation branch, so the "
                     f"rule for {ref_pair} would expand to zero bonds and the "
                     f"interaction would be silently dropped. Supported: "
                     f"heisenberg, dm, anisotropic_exchange, interaction_matrix, "
                     f"kitaev.")

             key = (atom_k['label'], atom_l['label'], tuple(int(v) for v in offset_final))
             rkey = (atom_l['label'], atom_k['label'], tuple(-int(v) for v in offset_final))
             hits.append({
                 "op": gi, "R": R, "t": t,
                 "k": atom_k['label'], "l": atom_l['label'], "off": offset_final,
                 "key": key, "rkey": rkey,
                 "fwd": fwd, "rev": rev, "fwd_chk": fwd_chk, "rev_chk": rev_chk,
             })

        # Pass 2: every operation reaching a bond must agree on its value.
        if samples is not None:
            self._check_propagation_consistency(
                user_type, type, ref_pair, ref_key, value, hits, samples,
                lattice_noise, final_dist)

        # Pass 3: emit, first operation to reach each bond wins (as before).
        added_bonds_keys = set()
        for h in hits:
             key, rkey = h["key"], h["rkey"]
             if key in added_bonds_keys or rkey in added_bonds_keys:
                 continue
             k, l, offset_final = h["k"], h["l"], h["off"]
             fwd, rev = h["fwd"], h["rev"]
             if type != "heisenberg" and not is_symbolic:
                 fwd, rev = fwd.tolist(), rev.tolist()

             if type == "dm":
                 # Add k->l entry, then l->k (reverse, D_ji = -D_ij)
                 self._add_dm_entry(k, l, offset_final, fwd, final_dist, original_value=value)
                 added_bonds_keys.add(key)
                 added_bonds_keys.add(rkey)
                 self._add_dm_entry(l, k, -offset_final, rev, final_dist, original_value=value)

             elif type == "heisenberg":
                 self._add_heisenberg_entry(k, l, offset_final, fwd, final_dist)
                 added_bonds_keys.add(key)
                 added_bonds_keys.add(rkey)
                 self._add_heisenberg_entry(l, k, -offset_final, rev, final_dist)

             elif type == "anisotropic_exchange":
                 # Config schema for anisotropic_exchange expects the diagonal
                 # [vxx, vyy, vzz]; the reverse bond carries the same diagonal.
                 self._add_anisotropic_entry(k, l, offset_final, fwd, final_dist, original_value=value)
                 added_bonds_keys.add(key)
                 added_bonds_keys.add(rkey)
                 self._add_anisotropic_entry(l, k, -offset_final, rev, final_dist, original_value=value)

             elif type == "interaction_matrix":
                 # Reverse bond: J_ji = J_ij^T
                 self._add_interaction_matrix_entry(k, l, offset_final, fwd, final_dist, original_value=value)
                 added_bonds_keys.add(key)
                 added_bonds_keys.add(rkey)
                 self._add_interaction_matrix_entry(l, k, -offset_final, rev, final_dist, original_value=value)

    def _check_propagation_consistency(self, user_type, rule_type, ref_pair, ref_key,
                                       value, hits, samples, lattice_noise, final_dist):
        """
        Refuse a rule value that the reference bond's own site symmetry forbids.

        Two operations g1, g2 that carry the reference bond onto the same directed
        bond differ by g2^-1 g1, an element of the bond's little group (an operation
        mapping the bond onto itself or onto its reverse). They agree on the value
        iff the value is invariant under that element -- D -> det(R) R D = D for a
        bond-preserving operation, = -D for a bond-reversing one; J -> R J R^T = J
        or J^T. So "all operations reaching each bond agree" is exactly "the value
        has the symmetry of its bond", and a disagreement anywhere in the orbit is a
        disagreement at the reference bond, which is where it is reported.

        The comparison is numeric, on `samples` (the value at random points of its
        parameters for a symbolic value, the value itself otherwise), relative to the
        largest entry. Its tolerance is `_PROPAGATION_TOL` on an exact lattice and
        is widened to `_PROPAGATION_NOISE_FACTOR` times the lattice's own noise --
        the non-orthogonality of the Cartesian operations -- so a CIF that is cubic
        to six decimals is not reported as a symmetry conflict (its operations
        disagree at the 1e-6 level on EVERY value, allowed or not), while a genuinely
        forbidden component disagrees by twice its size and is caught.
        """
        scale = max(float(np.abs(s).max()) for s in samples)
        if scale == 0.0 or rule_type == "heisenberg":
            return
        tol = max(_PROPAGATION_TOL, _PROPAGATION_NOISE_FACTOR * lattice_noise) * scale

        def mismatch(a, b):
            """Boolean mask of the entries on which two sample sets differ."""
            return np.any([np.abs(x - y) > tol for x, y in zip(a, b)], axis=0)

        first = {}       # directed key -> check samples of the first op to reach it
        conflicting = set()
        for h in hits:
            for key, chk in ((h["key"], h["fwd_chk"]), (h["rkey"], h["rev_chk"])):
                if key not in first:
                    first[key] = chk
                elif key not in conflicting and mismatch(first[key], chk).any():
                    conflicting.add(key)
        if not conflicting:
            return

        # Explain it at the reference bond, against the rule's own value: the
        # first operation (in spglib order) that carries the value back onto the
        # reference bond as something else.
        culprit = None
        union = None     # entries some little-group operation disagrees on
        for h in hits:
            for key, flag, chk, shown in ((h["key"], False, h["fwd_chk"], h["fwd"]),
                                          (h["rkey"], True, h["rev_chk"], h["rev"])):
                if key == ref_key:
                    mask = mismatch(samples, chk)
                    if mask.any():
                        if culprit is None:
                            culprit = (h, flag, shown, mask)
                        union = mask if union is None else (union | mask)
        if culprit is None:  # pragma: no cover - cannot happen for a group
            key = sorted(conflicting)[0]
            raise SymmetryConflictError(
                f"Symmetry rule '{user_type}' for ref_pair {list(ref_pair)}: the "
                f"operations reaching bond {key[0]}->{key[1]} {list(key[2])} propagate "
                f"different values to it, so the value {_fmt_rule_value(value)} is not "
                f"compatible with the bond's site symmetry.")

        h, reversed_bond, shown, mask = culprit
        is_vector = np.ndim(mask) == 1

        def named(m):
            if is_vector:
                names = [n for n, flag in zip("xyz", m) if flag]
                word = "component" + ("s " if len(names) > 1 else " ")
            else:
                names = [f"{'xyz'[a]}{'xyz'[b]}" for a in range(3) for b in range(3) if m[a, b]]
                word = "entr" + ("ies " if len(names) > 1 else "y ")
            return names, word + ", ".join(names)

        names, comp = named(mask)
        _all_names, comp_all = named(union)
        if is_vector:
            rule_part = ", ".join(str(v) for v, m in zip(list(value), mask) if m)
        else:
            rule_part = ", ".join(str(np.array(value, dtype=object)[a, b])
                                  for a in range(3) for b in range(3) if mask[a, b])
        reverse_law = {"dm": "D_ji = -D_ij", "anisotropic_exchange": "J_ji = J_ij",
                       "interaction_matrix": "J_ji = J_ij^T"}[rule_type]
        if reversed_bond:
            how = (f"maps the reference bond {ref_key[0]}->{ref_key[1]} {list(ref_key[2])} "
                   f"onto its own reverse, so with {reverse_law} it carries the value "
                   f"back onto the reference bond as")
        else:
            how = (f"maps the reference bond {ref_key[0]}->{ref_key[1]} {list(ref_key[2])} "
                   f"onto itself and carries the value back onto it as")
        if rule_type == "dm":
            why = ("Moriya's rules forbid that component here: a 2-fold axis "
                   "perpendicular to the bond through its midpoint, or a mirror plane "
                   "containing the bond, excludes the D component along it; an "
                   "inversion centre at the midpoint excludes all three.")
        else:
            why = "The bond's little group excludes that part of the tensor."
        fix = "Set the forbidden part to 0 in `value`"
        if is_vector and user_type != "kitaev":
            zeroed = ["0" if m else (f"{v:g}" if isinstance(v, (int, float)) else str(v))
                      for v, m in zip(list(value), union)]
            if any(isinstance(v, (int, float)) for v in value):
                fix += f" (e.g. `value: [{', '.join(zeroed)}]`)"
            else:
                fix += f" (e.g. `value: {','.join(zeroed)}`)"
        target = {"dm": "dm_interaction", "anisotropic_exchange": "anisotropic_exchange",
                  "interaction_matrix": "interaction_matrix"}[rule_type]
        n_directed = len(first)
        raise SymmetryConflictError(
            f"Symmetry rule '{user_type}' for ref_pair {list(ref_pair)} (offset "
            f"{list(ref_key[2])}, d = {final_dist:.5f} A): the value "
            f"{_fmt_rule_value(value)} is not invariant under the bond's own site "
            f"symmetry, which forbids its {comp_all}. "
            f"Operation {_op_to_string(h['R'], h['t'])} (op {h['op'] + 1} of "
            f"{len(self.symmetry_ops['rotations'])}) {how} {_fmt_rule_value(shown)}: "
            f"{comp} differ{'s' if len(names) == 1 else ''} from the rule's "
            f"{rule_part}. {why} No single value can be propagated consistently: the "
            f"operations disagree on {len(conflicting)} of the {n_directed} directed "
            f"bonds in this orbit, and keeping whichever operation reached each bond "
            f"first (what this engine did before 2026-10-18) gives a bond table that "
            f"is not invariant under the lattice's own (centring) translations, i.e. "
            f"a Hamiltonian of lower symmetry than the crystal. {fix}; when the "
            f"constraint ties components instead of zeroing one (a forbidden direction "
            f"off the Cartesian axes), use the allowed form `magcalc symmetry <config>` "
            f"prints; or list the bonds explicitly under `interactions.{target}` if "
            f"the lower symmetry is intended."
        )


    def _add_dm_entry(self, lbl_i, lbl_j, offset, val, distance=None, original_value=None):
        # Ensure val is a list of strings or floats (not sympy objects)
        clean_val = [str(v) if not isinstance(v, (float, int)) else v for v in val]
        
        # Deduplicate
        for entry in self.config["interactions"]["dm_interaction"]:
            if entry.get("pair") == [lbl_i, lbl_j] and entry["rij_offset"] == list(map(int, offset)):
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
        if original_value is not None:
             entry["original_value"] = original_value
        self.config["interactions"]["dm_interaction"].append(entry)

    def _add_kitaev_entry(self, lbl_i, lbl_j, offset, val, bond_dir, distance=None):
        # Deduplicate
        for entry in self.config["interactions"]["kitaev"]:
            if entry.get("pair") == [lbl_i, lbl_j] and entry["rij_offset"] == list(map(int, offset)):
                return
        
        entry = {
            "type": "kitaev",
            "pair": [lbl_i, lbl_j],
            "rij_offset": list(map(int, offset)),
            "K": val,
            "bond_direction": bond_dir
        }
        if distance:
            entry["distance"] = distance
        self.config["interactions"]["kitaev"].append(entry)

    def _add_heisenberg_entry(self, lbl_i, lbl_j, offset, val, distance=None):
        # Deduplicate
        for entry in self.config["interactions"]["heisenberg"]:
             if entry.get("pair") == [lbl_i, lbl_j] and entry.get("rij_offset") == list(map(int, offset)):
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

    def _add_anisotropic_entry(self, lbl_i, lbl_j, offset, val, distance=None, original_value=None):
        clean_val = [str(v) if not isinstance(v, (float, int)) else v for v in val]
        # Deduplicate
        for entry in self.config["interactions"]["anisotropic_exchange"]:
             if entry.get("pair") == [lbl_i, lbl_j] and entry.get("rij_offset") == list(map(int, offset)):
                 return
                 
        entry = {
            "type": "anisotropic_exchange",
            "pair": [lbl_i, lbl_j],
            "rij_offset": list(map(int, offset)),
            "value": clean_val
        }
        if distance:
            entry["distance"] = distance
        if original_value is not None:
             entry["original_value"] = original_value
        self.config["interactions"]["anisotropic_exchange"].append(entry)

    def _add_interaction_matrix_entry(self, lbl_i, lbl_j, offset, val, distance=None, original_value=None):
        # val is 3x3 matrix (list of lists)
        clean_val = []
        for row in val:
            clean_row = [str(v) if not isinstance(v, (float, int)) else v for v in row]
            clean_val.append(clean_row)
            
        # Deduplicate
        for entry in self.config["interactions"]["interaction_matrix"]:
             if entry.get("pair") == [lbl_i, lbl_j] and entry.get("rij_offset") == list(map(int, offset)):
                 return

        entry = {
            "type": "interaction_matrix",
            "pair": [lbl_i, lbl_j],
            "rij_offset": list(map(int, offset)),
            "value": clean_val
        }
        if distance:
            entry["distance"] = distance
        if original_value is not None:
             entry["original_value"] = original_value
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


    def add_single_ion_anisotropy(self, label: str, value: Any, axis: List[float] = None):
        """
        Add Uniaxial SIA: D(S . n)^2.
        Propagated to all equivalent atoms (same label prefix).
        """
        base_label = label
        
        interactions = []
        for a in self.atoms_uc:
            if a['label'].startswith(base_label): # Simple prefix match
                interactions.append({
                    "type": "sia",
                    "atom_label": a['label'],
                    "K": value,
                    "axis": axis if axis is not None else [0,0,1]
                })
        
        self.config["interactions"]["single_ion_anisotropy"].extend(interactions)

    def add_kitaev_interaction(self, ref_pair: Tuple[str, str], K: Any, bond_direction: str, 
                               distance: float = None, offset: List[int] = None):
        """
        Add Kitaev interaction and propagate it by symmetry.
        """
        if not self.symmetry_ops:
             logger.warning("No symmetry operations loaded. Cannot propagate.")
             return

        # Simple manual entry for now (similar to others)
        # We might want to use full symmetry propagation eventually,
        # but for Kitaev it's often better to specify them by bond type.
        # However, for consistency, let's treat it like Heisenberg/DM.
        
        # 1. Resolve atoms and find best offset
        idx_i = self._atom_label_to_idx[ref_pair[0]]
        idx_j = self._atom_label_to_idx[ref_pair[1]]
        pos_i = np.array(self.atoms_uc[idx_i]["pos"])
        pos_j_uc = np.array(self.atoms_uc[idx_j]["pos"])
        
        if offset is not None:
            best_offset = np.array(offset)
        else:
            # Simple nearest image search
            best_offset = [0,0,0]
            best_d = 1e9
            from itertools import product
            rng = [-1, 0, 1]
            for u, v, w in product(rng, repeat=3):
                p_j = pos_j_uc + np.array([u, v, w])
                d = np.linalg.norm((p_j - pos_i) @ self.lattice_vectors)
                if 0.01 < d < best_d:
                    best_d = d
                    best_offset = [u, v, w]
            if distance is None: distance = best_d

        self._add_kitaev_entry(ref_pair[0], ref_pair[1], best_offset, K, bond_direction, distance)

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
                         species: str = None, wyckoff_label: str = "",
                         ion: str = None, element: str = None, charge=None):
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
            self._add_atom_raw(label, pos, spin, species, ion=ion, element=element,
                               charge=charge)
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
            self._add_atom_raw(atom_name, orbit_positions[0], spin, species, ion=ion,
                               element=element, charge=charge)
        else:
            # Determine starting index based on existing atoms
            start_idx = 0
            existing_labels = [a['label'] for a in self.atoms_uc]
            while f"{label}{start_idx}" in existing_labels:
                start_idx += 1
                
            for i, p in enumerate(orbit_positions):
                atom_name = f"{label}{start_idx + i}"
                self._add_atom_raw(atom_name, p, spin, species, ion=ion, element=element,
                                   charge=charge)

    def _add_atom_raw(self, label, pos, spin, species, ion=None, element=None,
                      charge=None):
        if species is None:
            # Guess from label (remove digits)
            species = "".join([c for c in label if c.isalpha()])
            
        self.atoms_uc.append({
            "label": label,
            "species": species,
            "pos": pos,
            "spin_S": spin,
            "ion": ion,
            "element": element,
            # `charge` rides along so `element: Fe` + `charge: 2` can reconstruct the
            # form-factor ion (GenericSpinModel._resolve_ion). Dropping it here is how
            # FeI2 ended up asking for a NEUTRAL-iron form factor.
            "charge": charge,
        })
        self.config["crystal_structure"]["atoms_uc"] = self.atoms_uc
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
            # sg_info.number is integer
            if sg_info.number == space_group:
                # Prefer "standard" setting? Usually the first one or specific choice.
                # Just take the first one found for now.
                # Actually, for Fdd2 (43), we want standard setting.
                self.symmetry_ops = spglib.get_symmetry_from_database(hall)
                if self.symmetry_ops:
                    found = True
                    logger.info(f"Loaded symmetry for SG {space_group}: Hall {hall} ({sg_info.international_short})")
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

    def set_minimization(self, enabled: bool = True, method: str = "L-BFGS-B", maxiter: int = 500, early_stopping: int = 10, initial_configuration: List[Dict] = None):
        """Set minimization parameters."""
        self.config["minimization"] = {
            "enabled": enabled,
            "method": method,
            "maxiter": maxiter,
            "early_stopping": early_stopping
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

    def _offset_shell(self, target_dist: Optional[float] = None) -> List[Tuple[int, int, int]]:
        """Cell-image offsets that can hold a bond of length `target_dist`.

        A bond of length d reaches at most ceil(d / h_i) cells along axis i,
        where h_i is the cell's perpendicular height along that axis. Sizing the
        shell from the distance (rather than a fixed +-1) is what lets rules
        find 2nd-neighbour bonds (a chain bond at 2a, FeI2's J3 at [2,0,0]).

        With no target distance, falls back to the +-1 shell.
        """
        from itertools import product

        if not target_dist:
            return list(product([-1, 0, 1], repeat=3))

        lat = np.asarray(self.lattice_vectors, dtype=float)
        volume = abs(np.linalg.det(lat))
        heights = [volume / np.linalg.norm(np.cross(lat[(i + 1) % 3], lat[(i + 2) % 3]))
                   for i in range(3)]
        ranges = [range(-n, n + 1) for n in
                  (max(1, int(np.ceil((target_dist + 0.01) / h))) for h in heights)]
        return list(product(*ranges))

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
        
        expanded_rules = []

        # For each generic rule, find all matching pairs
        for rule in generic_rules:
            target_dist = rule["distance"]
            val_sym = rule["value"]

            offsets = self._offset_shell(target_dist)
            n_before = len(expanded_rules)

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

            if len(expanded_rules) == n_before:
                raise ValueError(
                    f"Heisenberg rule {rule} matched no bonds: no pair of sites is "
                    f"{target_dist} A apart (within 0.01 A) anywhere in the searched "
                    f"{len(offsets)} cell images. Check the distance against the actual "
                    f"bond lengths -- a rule that matches nothing would silently drop "
                    f"the interaction from the Hamiltonian."
                )

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
                 
                 # Validation: Distance must be preserved (within tolerance)
                 d_vec_p = (positions[idx_l] + final_offset - positions[idx_k]) @ self.lattice_vectors
                 if abs(bond["distance"] - np.linalg.norm(d_vec_p)) > 0.1:
                     continue

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

        labels = [a["label"] for a in self.atoms_uc]
        positions = [np.array(a["pos"]) for a in self.atoms_uc]

        for rule in generic_rules:
            target_dist = rule["distance"]
            value = rule["value"]

            offsets = self._offset_shell(target_dist)

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

            if not pool:
                raise ValueError(
                    f"Distance rule {rule} matched no bonds: no pair of sites is "
                    f"{target_dist} A apart (within 0.05 A) anywhere in the searched "
                    f"{len(offsets)} cell images. Check the distance against the actual "
                    f"bond lengths -- a rule that matches nothing would silently drop "
                    f"the interaction from the Hamiltonian."
                )

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

    def _expand_interaction_matrix_rules(self):
        """
        Expand generic interaction matrix rules.
        """
        rules = self.config["interactions"]["interaction_matrix"]
        if not rules:
            return

        explicit_rules = [r for r in rules if "pair" in r]
        generic_rules = [r for r in rules if "pair" not in r]

        if not generic_rules:
            return

        # Start fresh
        self.config["interactions"]["interaction_matrix"] = explicit_rules

        labels = [a["label"] for a in self.atoms_uc]
        positions = [np.array(a["pos"]) for a in self.atoms_uc]

        for rule in generic_rules:
            target_dist = rule.get("distance", 0.0)
            value = rule.get("value") # Expecting 3x3 list of lists

            offsets = self._offset_shell(target_dist)

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

            if not pool:
                raise ValueError(
                    f"Distance rule {rule} matched no bonds: no pair of sites is "
                    f"{target_dist} A apart (within 0.05 A) anywhere in the searched "
                    f"{len(offsets)} cell images. Check the distance against the actual "
                    f"bond lengths -- a rule that matches nothing would silently drop "
                    f"the interaction from the Hamiltonian."
                )

            # Group into orbits via symmetry
            while pool:
                ref_i, ref_j, ref_off = pool[0]
                
                initial_pool_size = len(pool)
                try:
                    self.add_symmetry_interaction("interaction_matrix", (ref_i, ref_j), value, 
                                                 distance=target_dist, offset=list(ref_off))
                except SymmetryConflictError:
                    raise
                except Exception as e:
                    logger.error(f"Error propagating Matrix interaction: {e}")
                
                # Remove added bonds from pool
                expanded = self.config["interactions"]["interaction_matrix"]
                pool_after = [b for b in pool if not any(
                    (e["pair"] == [b[0], b[1]] and e.get("rij_offset") == list(b[2]))
                    for e in expanded
                )]
                
                # Loop breaker
                if len(pool_after) >= initial_pool_size and pool_after:
                     pool_after.pop(0)
                     
                pool = pool_after

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

        labels = [a["label"] for a in self.atoms_uc]
        positions = [np.array(a["pos"]) for a in self.atoms_uc]

        for rule in generic_rules:
            # Skip invalid rules
            if "distance" not in rule:
                continue

            target_dist = rule.get("distance", 0.0)
            value = rule.get("value", ["0","0","0"])

            # Sized from the target distance, not a fixed +-1 shell (which
            # silently found nothing for bonds beyond one cell image).
            offsets = self._offset_shell(target_dist)

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

            if not pool:
                raise ValueError(
                    f"Distance rule {rule} matched no bonds: no pair of sites is "
                    f"{target_dist} A apart (within 0.05 A) anywhere in the searched "
                    f"{len(offsets)} cell images. Check the distance against the actual "
                    f"bond lengths -- a rule that matches nothing would silently drop "
                    f"the interaction from the Hamiltonian."
                )

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
                except SymmetryConflictError:
                    # A value the bond's symmetry forbids has no consistent table;
                    # swallowing it here would drop the orbit in silence.
                    raise
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
        # Dimensionality removed


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

        # 3. Solve for the allowed matrix J: the null space of {J - R J R^T = 0}
        #
        # The constraint is LINEAR in the nine entries of J, so this is a null
        # space, and it is computed as one -- numerically, with an SVD.
        #
        # It used to be handed to `sp.solve` after "sanitizing" R_cart with
        # `np.round(R_cart, 10)` + `nsimplify`, and that was wrong twice over.
        # R_cart = A R_frac A^-1 inherits the lattice's numerical noise, and a
        # real CIF has plenty: nsimplify of a 10-decimal float is not 2/3, it is
        # the exact rational 1666666499/2500000000. Such an R is no longer
        # orthogonal, so `J = R J R^T` over-constrains and the reported allowed
        # form COLLAPSES -- measured on KFe3J, where perturbing `a` by 1e-7 A
        # took the NN bond from 6 free parameters to 1 (every off-diagonal
        # forced to zero, i.e. "symmetry forbids DM here", silently). On a
        # Materials Project NiO primitive cell the same giant rationals instead
        # made sympy grind for over ten minutes without finishing.
        #
        # Numerically it is a 0.2 ms SVD whose answer does not move at 1e-7, and
        # the tolerance is explicit rather than implied by float noise.
        J_vars = sp.Matrix(sp.symbols('j0:9')).reshape(3, 3)  # j0..j8

        # vec(J) is row-major, so J -> R J R^T is kron(R, R), and the transpose
        # (the `swap` case) is that composed with the permutation vec(J^T).
        P_T = np.zeros((9, 9))
        for a in range(3):
            for c in range(3):
                P_T[3 * a + c, 3 * c + a] = 1.0

        rows = []
        for item in little_group_ops:
            R_cart = _nearest_orthogonal(self._lattice_rotation(item["op"][0]))
            K = np.kron(R_cart, R_cart)
            if item["swap"]:
                K = K @ P_T
            rows.append(np.eye(9) - K)

        if rows:
            M = np.vstack(rows)
            _u, sv, Vt = np.linalg.svd(M)
            # Relative tolerance: the constraint rows are O(1) by construction
            # (R is orthogonal up to lattice noise), so this separates a genuine
            # constraint from the noise that motivated this rewrite.
            rank = int(np.sum(sv > 1e-6 * max(1.0, float(sv[0]))))
            null = Vt[rank:].T                      # (9, p) orthonormal basis
        else:
            null = np.eye(9)                        # no ops: nothing constrained

        basis, pivots = _rref(null.T)

        # Each pivot column of the reduced basis is a free entry of J; every
        # other entry is a fixed linear combination of them. That is the same
        # presentation the sympy solve produced (free symbols named after the
        # entries they are), so `symbolic_matrix` reads as it always did.
        exprs = []
        for entry in range(9):
            terms = []
            for a, col in enumerate(pivots):
                coeff = basis[a][entry]
                if abs(coeff) > _SYMMETRY_COEFF_TOL:
                    terms.append(_nice_coefficient(coeff) * J_vars[col])
            # NOT nsimplify'd: any coefficient `_nice_coefficient` declined to
            # recognise stays a Float on purpose, and nsimplify would turn it
            # back into the sort of exact-looking giant rational this rewrite
            # exists to get rid of.
            exprs.append(sp.together(sum(terms)) if terms else sp.Integer(0))
        J_solved = sp.Matrix(3, 3, exprs)

        free_symbols = sorted(J_solved.free_symbols, key=lambda s: int(s.name[1:]))

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

