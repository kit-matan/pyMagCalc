
import ase.io
import numpy as np
import logging
import os
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def load_cif_structure(cif_path: str, magnetic_elements: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load crystal structure from a CIF file using ASE.
    
    Args:
        cif_path (str): Path to the CIF file.
        magnetic_elements (Optional[List[str]]): List of element symbols to include (e.g. ["Fe", "Ni"]).
                                                If None, all atoms are included.
                                                
    Returns:
        Dict[str, Any]: A dictionary compatible with the 'crystal_structure' config section.
                        Contains 'unit_cell_vectors' and 'atoms_uc'.
    
    Raises:
        FileNotFoundError: If the CIF file does not exist.
        ImportError: If ASE is not installed.
        Exception: If parsing fails.
    """
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
        
    try:
        atoms = ase.io.read(cif_path)
    except Exception as e:
        logger.error(f"Failed to read CIF file {cif_path}: {e}")
        raise
        
    # Extract unit cell vectors (3x3 matrix)
    # atoms.cell is an object, convert to numpy array
    uc_vectors = np.array(atoms.get_cell())
    
    # Filter atoms
    symbols = atoms.get_chemical_symbols()
    positions_frac = atoms.get_scaled_positions()
    
    # Create list of (index, symbol, pos_frac) tuples for sorting/filtering
    atom_data = []
    for i in range(len(atoms)):
        sym = symbols[i]
        pos = positions_frac[i]
        
        # Filter
        if magnetic_elements:
            if sym not in magnetic_elements:
                continue
                
        atom_data.append({'sym': sym, 'pos': pos, 'orig_index': i})
        
    if not atom_data:
        logger.warning(f"No atoms found in CIF {cif_path} matching magnetic elements: {magnetic_elements}")
        return {
            "unit_cell_vectors": uc_vectors.tolist(),
            "atoms_uc": []
        }

    # Sort atoms to ensure deterministic ordering
    # Sort by Symbol, then by Z, Y, X coordinates
    # Using a small tolerance for float comparison isn't easy in simple sort key,
    # but standard tuple comparison works for "exact" sorting which is usually fine 
    # for determinism if input is consistent.
    atom_data.sort(key=lambda x: (x['sym'], x['pos'][2], x['pos'][1], x['pos'][0]))
    
    # Generate labels
    # Count occurrences of each symbol to generate Fe1, Fe2, etc.
    sym_counts = {}
    atoms_uc_list = []
    
    for item in atom_data:
        sym = item['sym']
        pos = item['pos']
        
        count = sym_counts.get(sym, 0) + 1
        sym_counts[sym] = count
        
        label = f"{sym}{count}"
        
        # Ensure position is in [0, 1) range? 
        # CIF usually puts them there, but ASE might wrap. 
        # pyMagCalc logic handles generic fractional coords, but keeping them normalized is good practice.
        pos_norm = pos % 1.0 
        
        atoms_uc_list.append({
            "label": label,
            "pos": pos_norm.tolist()
        })
        
    logger.info(f"Loaded {len(atoms_uc_list)} atoms from CIF {cif_path}")
    
    return {
        "unit_cell_vectors": uc_vectors.tolist(),
        "atoms_uc": atoms_uc_list
    }
