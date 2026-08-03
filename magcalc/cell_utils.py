"""Cell-level crystal utilities (Gap 4 #27): primitive / standardized cells,
sub-crystals, and irreducible Brillouin-zone paths.

The Sunny analogues are `primitive_cell`, `standardize`, `subcrystal` and
`print_irreducible_bz_paths`. The first three are thin wrappers over spglib, which
is already a hard dependency here and is the same library Sunny's own conventions
are checked against; `irreducible_bz_path` needs the high-symmetry-point tables and
uses `seekpath`, which is OPTIONAL -- a missing install raises a clear message
rather than a traceback.

Everything works on a plain `cell` tuple `(lattice, positions, numbers)` in spglib's
convention (lattice ROWS are the vectors, positions FRACTIONAL), with adapters to and
from a pyMagCalc `crystal_structure` block so the CLI can round-trip a config.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import spglib

logger = logging.getLogger(__name__)

Cell = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _species_key(atom: Dict[str, Any]) -> str:
    """Chemical identity of an atom entry, ignoring the per-site index.

    Same rule as `MagCalcConfigBuilder.detect_symmetry_from_structure`: without it,
    per-site labels (Fe0, Fe1, ...) would make every site inequivalent and collapse
    the detected group to P1.
    """
    for k in ("species", "element", "ion"):
        if atom.get(k):
            return str(atom[k])
    return re.sub(r"\d+$", "", str(atom.get("label", "X"))) or "X"


def cell_from_structure(crystal_structure: Dict[str, Any]) -> Tuple[Cell, List[str]]:
    """(spglib cell, species names in atom order) from a `crystal_structure` block.

    Requires `lattice_vectors` and `atoms_uc` -- i.e. an already-expanded structure.
    Use `MagCalcConfigBuilder.from_config` first if the config is in Wyckoff form.
    """
    lat = crystal_structure.get("lattice_vectors")
    atoms = crystal_structure.get("atoms_uc")
    if lat is None or not atoms:
        raise ValueError(
            "cell_from_structure needs an expanded crystal_structure with "
            "`lattice_vectors` and `atoms_uc` (build it with "
            "MagCalcConfigBuilder.from_config first).")
    lattice = np.asarray(lat, dtype=float)
    positions = np.asarray([a["pos"] for a in atoms], dtype=float)
    species = [_species_key(a) for a in atoms]
    order = sorted(set(species))
    numbers = np.asarray([order.index(s) + 1 for s in species], dtype=int)
    return (lattice, positions, numbers), species


def _named(cell: Cell, species_order: Sequence[str]) -> List[str]:
    """Map spglib's `numbers` back onto species names."""
    return [species_order[int(n) - 1] for n in cell[2]]


def primitive_cell(cell: Cell, symprec: float = 1e-4) -> Cell:
    """The primitive cell (spglib `find_primitive`; Sunny `primitive_cell`).

    Raises rather than returning None when spglib cannot reduce the cell -- a silent
    None here would look like "already primitive".
    """
    out = spglib.find_primitive(cell, symprec=symprec)
    if out is None:
        raise ValueError(
            f"spglib could not find a primitive cell at symprec={symprec}. The "
            f"structure may be inconsistent; try a looser symprec.")
    lattice, positions, numbers = out
    return np.asarray(lattice, float), np.asarray(positions, float), \
        np.asarray(numbers, int)


def standardize_cell(cell: Cell, to_primitive: bool = False,
                     no_idealize: bool = False, symprec: float = 1e-4) -> Cell:
    """The standardized (conventional, or primitive) cell -- Sunny `standardize`."""
    out = spglib.standardize_cell(cell, to_primitive=to_primitive,
                                  no_idealize=no_idealize, symprec=symprec)
    if out is None:
        raise ValueError(
            f"spglib could not standardize the cell at symprec={symprec}.")
    lattice, positions, numbers = out
    return np.asarray(lattice, float), np.asarray(positions, float), \
        np.asarray(numbers, int)


def subcrystal(crystal_structure: Dict[str, Any],
               species: Sequence[str]) -> Dict[str, Any]:
    """A copy of `crystal_structure` keeping only atoms of the named species.

    Sunny's `subcrystal`. Matching is on the chemical identity (`species`/`element`/
    `ion`, else the label with its trailing site index stripped), so asking for "Cu"
    keeps Cu0, Cu1, ... The lattice is untouched. Raises if a requested species is
    absent -- a typo that silently produced an empty magnetic sublattice would be a
    very expensive way to get a featureless spectrum.
    """
    import copy

    atoms = crystal_structure.get("atoms_uc") or []
    if not atoms:
        raise ValueError("subcrystal needs an expanded `atoms_uc`.")
    wanted = {str(s) for s in species}
    present = {_species_key(a) for a in atoms}
    missing = wanted - present
    if missing:
        raise ValueError(
            f"subcrystal: species {sorted(missing)} are not in the structure "
            f"(present: {sorted(present)}).")
    out = copy.deepcopy(crystal_structure)
    out["atoms_uc"] = [copy.deepcopy(a) for a in atoms if _species_key(a) in wanted]
    return out


def irreducible_bz_path(cell: Cell, reference_distance: float = 0.025,
                        symprec: float = 1e-4) -> Dict[str, Any]:
    """Suggested high-symmetry path through the irreducible BZ (Sunny
    `print_irreducible_bz_paths`), via `seekpath`.

    Returns {'point_coords': {label: [h,k,l]}, 'path': [(label, label), ...],
             'spacegroup_number': int, 'bravais': str}, all in RECIPROCAL LATTICE
    UNITS of the primitive cell seekpath chooses -- which is not necessarily the
    input cell, so the returned `primitive_lattice` is included to make that explicit.
    """
    try:
        import seekpath
    except ImportError as exc:                     # pragma: no cover - env-dependent
        raise ImportError(
            "irreducible_bz_path needs the optional `seekpath` package "
            "(pip install seekpath). The rest of magcalc.cell_utils does not.") from exc

    lattice, positions, numbers = cell
    res = seekpath.get_path((np.asarray(lattice, float).tolist(),
                             np.asarray(positions, float).tolist(),
                             [int(n) for n in numbers]),
                            with_time_reversal=True,
                            reference_distance=reference_distance,
                            symprec=symprec)
    return {
        "point_coords": {k: list(map(float, v))
                         for k, v in res["point_coords"].items()},
        "path": [tuple(seg) for seg in res["path"]],
        "spacegroup_number": int(res["spacegroup_number"]),
        "bravais": str(res.get("bravais_lattice_extended", "")),
        "primitive_lattice": np.asarray(res["primitive_lattice"], float),
    }


def describe_cell(cell: Cell, species_order: Optional[Sequence[str]] = None
                  ) -> Dict[str, Any]:
    """Human-readable summary: lattice parameters, volume, site count, formula."""
    lattice, positions, numbers = cell
    a, b, c = (float(np.linalg.norm(v)) for v in lattice)

    def ang(u, v):
        cosv = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))

    names = _named(cell, species_order) if species_order is not None else \
        [str(int(n)) for n in numbers]
    counts: Dict[str, int] = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    return {
        "a": a, "b": b, "c": c,
        "alpha": ang(lattice[1], lattice[2]),
        "beta": ang(lattice[0], lattice[2]),
        "gamma": ang(lattice[0], lattice[1]),
        "volume": float(abs(np.linalg.det(lattice))),
        "n_sites": int(len(numbers)),
        "composition": counts,
    }
