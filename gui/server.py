import os
import io
import yaml
import numpy as np
import spglib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ase.io import read
from typing import List, Dict, Any

app = FastAPI(title="MagCalc Designer Backend")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        # Ensure we don't write outside the allowed project directories
        # For simplicity, we save to the project root for now.
        save_path = os.path.join(os.getcwd(), filename)
        
        with open(save_path, 'w') as f:
            yaml.dump(config["data"], f, sort_keys=False)
            
        return {"message": f"Saved successfully to {save_path}", "path": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
