"""Read a magnetic CIF (mCIF) into a pyMagCalc magnetic structure.

An mCIF encodes an experimentally-determined magnetic structure through a magnetic
space group: a set of magnetic symmetry operations (each a crystallographic op plus a
time-reversal parity p = +/-1) acting on the moments of a few asymmetric-unit atoms.
Applying every operation expands those to the full magnetic cell.

Two transforms do the work (matching Sunny 0.8.1's MCIF.jl / MSymOp.jl):

  position:  r' = R r + T                               (fractional coords)
  moment:    m' = det(R) * p * (R m)                    (m is an axial vector, so it is
                                                         invariant under spatial inversion
                                                         -- hence det(R) -- and flips under
                                                         time reversal -- hence p)

The moment components in an mCIF (`_atom_site_moment.crystalaxis_*`) are coefficients on
the lattice vectors, so the Cartesian moment is  m_cart = m' @ A  (A rows = lattice
vectors). LSWT needs only the DIRECTION, so the magnitude/g handling is left to
`spin_S` downstream.

Validated against Sunny on TbSb (test/cifs/TbSb_isodistort.mcif): see tests/test_mcif.py.
"""
import logging
import re
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- CIF parsing
def _lattice_from_params(a, b, c, alpha, beta, gamma) -> np.ndarray:
    """Lattice vectors as ROWS (Angstrom), standard crystallographic setting."""
    al, be, ga = np.radians([alpha, beta, gamma])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c**2 - cx**2 - cy**2, 0.0))
    return np.array([
        [a, 0.0, 0.0],
        [b * np.cos(ga), b * np.sin(ga), 0.0],
        [cx, cy, cz],
    ])


def _parse_xyz_component(expr: str) -> Tuple[np.ndarray, float]:
    """One component of a symop string, e.g. 'x-y+1/2' -> (row [1,-1,0], const 0.5)."""
    row = np.zeros(3)
    const = 0.0
    # split into signed tokens
    for tok in re.findall(r'[+-]?[^+-]+', expr.replace(' ', '')):
        if not tok:
            continue
        sign = -1.0 if tok[0] == '-' else 1.0
        t = tok.lstrip('+-')
        m = re.match(r'^(\d+/\d+|\d*\.?\d*)?\*?([xyz])$', t)
        if m:
            coeff = m.group(1)
            val = 1.0 if not coeff else (
                float(coeff.split('/')[0]) / float(coeff.split('/')[1])
                if '/' in coeff else float(coeff))
            row['xyz'.index(m.group(2))] += sign * val
        else:                                   # pure constant (maybe a fraction)
            const += sign * (float(t.split('/')[0]) / float(t.split('/')[1])
                             if '/' in t else float(t))
    return row, const


def parse_magnetic_symop(s: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """'x-y,-x,z+1/2,+1' -> (R 3x3, T 3-vector, p in {+1,-1})."""
    parts = [p.strip() for p in s.split(',')]
    if len(parts) != 4:
        raise ValueError(f"magnetic symop must have 4 comma-separated parts: {s!r}")
    R = np.zeros((3, 3))
    T = np.zeros(3)
    for i in range(3):
        R[i], T[i] = _parse_xyz_component(parts[i])
    p = int(parts[3])
    if abs(p) != 1:
        raise ValueError(f"time-reversal parity must be +/-1, got {p} in {s!r}")
    return R, T, p


def _compose(s1, s2):
    """s1 . s2 for magnetic symops (R, T, p)."""
    R1, T1, p1 = s1
    R2, T2, p2 = s2
    return (R1 @ R2, T1 + R1 @ T2, p1 * p2)


def _read_cif_loops(text: str):
    """Minimal CIF reader: returns (tags dict of scalar key->value, list of loops).

    Each loop is (header_tags, rows) with rows a list of token lists. Enough for the
    fields an mCIF needs; not a general CIF parser.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    scalars: Dict[str, str] = {}
    loops = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i].strip()
        if not ln or ln.startswith('#'):
            i += 1
            continue
        if ln.lower() == 'loop_':
            i += 1
            headers = []
            while i < n and lines[i].strip().startswith('_'):
                headers.append(lines[i].strip().split()[0])
                i += 1
            rows = []
            while i < n:
                s = lines[i].strip()
                if not s or s.startswith('_') or s.lower() == 'loop_' or s.startswith('#'):
                    break
                rows.append(_split_cif_row(s))
                i += 1
            loops.append((headers, rows))
        elif ln.startswith('_'):
            toks = _split_cif_row(ln)
            if len(toks) >= 2:
                scalars[toks[0]] = toks[1]
            i += 1
        else:
            i += 1
    return scalars, loops


def _split_cif_row(s: str) -> List[str]:
    """Tokenise a CIF row, respecting single/double quotes."""
    return [t for t in re.findall(r"'[^']*'|\"[^\"]*\"|\S+", s)]


def _cif_float(v: str) -> float:
    """Parse a CIF number, dropping any (uncertainty) suffix."""
    return float(re.sub(r'\(.*?\)', '', v.strip().strip('"\'')))


def _find_loop(loops, tag):
    for headers, rows in loops:
        if tag in headers:
            return headers, rows
    return None, None


# --------------------------------------------------------------------------- public API
def read_mcif(path: str, tol: float = 1e-4) -> Dict:
    """Parse an mCIF file. Returns a dict with:

        lattice_vectors : (3,3) rows, Angstrom
        sites           : list of {label, pos (frac, in the MAGNETIC cell),
                                    moment (Cartesian), direction (unit)}

    Every magnetic-atom orbit is expanded by the magnetic space group. Non-magnetic
    atoms (no listed moment) are omitted -- LSWT only needs the magnetic sublattice.
    """
    with open(path) as f:
        scalars, loops = _read_cif_loops(f.read())

    A = _lattice_from_params(
        _cif_float(scalars['_cell_length_a']), _cif_float(scalars['_cell_length_b']),
        _cif_float(scalars['_cell_length_c']), _cif_float(scalars['_cell_angle_alpha']),
        _cif_float(scalars['_cell_angle_beta']), _cif_float(scalars['_cell_angle_gamma']))

    def _ops(*tags):
        for t in tags:
            h, rows = _find_loop(loops, t)
            if rows is not None:
                ci = h.index(t)
                return [parse_magnetic_symop(r[ci]) for r in rows]
        return [(np.eye(3), np.zeros(3), 1)]

    operations = _ops('_space_group_symop_magn_operation.xyz',
                      '_space_group_symop.magn_operation_xyz')
    centerings = _ops('_space_group_symop_magn_centering.xyz',
                      '_space_group_symop.magn_centering_xyz')

    # atom positions (by label)
    hpos, rpos = _find_loop(loops, '_atom_site_label')
    li = hpos.index('_atom_site_label')
    xi = hpos.index('_atom_site_fract_x')
    yi = hpos.index('_atom_site_fract_y')
    zi = hpos.index('_atom_site_fract_z')
    pos_by_label = {r[li]: np.array([_cif_float(r[xi]), _cif_float(r[yi]),
                                     _cif_float(r[zi])]) for r in rpos}

    # moments (by label)
    hm, rm = _find_loop(loops, '_atom_site_moment.label')
    if rm is None:
        hm, rm = _find_loop(loops, '_atom_site_moment_label')
    mli = hm.index('_atom_site_moment.label') if '_atom_site_moment.label' in hm \
        else hm.index('_atom_site_moment_label')

    def _mi(*names):
        for nm in names:
            if nm in hm:
                return hm.index(nm)
        raise KeyError(names)
    mx = _mi('_atom_site_moment.crystalaxis_x', '_atom_site_moment_crystalaxis_x')
    my = _mi('_atom_site_moment.crystalaxis_y', '_atom_site_moment_crystalaxis_y')
    mz = _mi('_atom_site_moment.crystalaxis_z', '_atom_site_moment_crystalaxis_z')

    sites = []
    seen = []
    for r in rm:
        label = r[mli]
        m0 = np.array([_cif_float(r[mx]), _cif_float(r[my]), _cif_float(r[mz])])
        r0 = pos_by_label[label]
        for s1 in operations:
            for s2 in centerings:
                R, T, p = _compose(s1, s2)
                r_new = (R @ r0 + T) % 1.0
                m_new = float(np.linalg.det(R)) * p * (R @ m0)      # axial + time reversal
                key = tuple(np.round(r_new / tol).astype(int) % int(round(1 / tol)))
                match = next((k for k in seen if k[0] == key), None)
                m_cart = m_new @ A
                if match is None:
                    seen.append((key, len(sites)))
                    d = np.linalg.norm(m_cart)
                    sites.append({
                        'label': label, 'pos': r_new, 'moment': m_cart,
                        'direction': (m_cart / d) if d > 1e-9 else np.zeros(3)})
                else:
                    prev = sites[match[1]]['moment']
                    if np.linalg.norm(prev) > 1e-9 and \
                            np.linalg.norm(m_cart - prev) > 1e-3 * max(np.linalg.norm(prev), 1):
                        raise ValueError(
                            f"mCIF internally inconsistent at {label} {r_new}: two symmetry "
                            f"images give different moments ({prev} vs {m_cart}).")
    return {'lattice_vectors': A, 'sites': sites}


def mcif_to_config_fragment(path: str, spin_S: float = 1.0,
                            ion: str = None) -> Dict:
    """A config fragment (crystal_structure + magnetic_structure) from an mCIF.

    Merge into a `magcalc run` config, add `interactions`/`parameters`, and run. The
    magnetic cell and per-site spin DIRECTIONS come straight from the file; magnitudes
    are set by `spin_S` (LSWT uses the direction, not the mCIF moment length).
    """
    data = read_mcif(path)
    atoms, dirs = [], []
    for i, s in enumerate(data['sites']):
        a = {'label': f"{s['label']}_{i}", 'pos': [float(x) for x in s['pos']],
             'spin_S': spin_S}
        if ion:
            a['ion'] = ion
        atoms.append(a)
        dirs.append([float(x) for x in s['direction']])
    return {
        'crystal_structure': {
            'lattice_vectors': [[float(x) for x in row] for row in data['lattice_vectors']],
            'atoms_uc': atoms},
        'magnetic_structure': {'type': 'pattern', 'pattern_type': 'generic',
                               'directions': dirs},
    }
