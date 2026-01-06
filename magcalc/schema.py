from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Any, Optional, Union

class LatticeParameters(BaseModel):
    model_config = ConfigDict(extra='allow')
    a: float
    b: Optional[float] = None
    c: Optional[float] = None
    alpha: Optional[float] = 90.0
    beta: Optional[float] = 90.0
    gamma: Optional[float] = 90.0
    space_group: Optional[int] = None

class AtomUC(BaseModel):
    model_config = ConfigDict(extra='allow')
    label: str
    pos: List[float]
    spin_S: float
    species: Optional[str] = None

class WyckoffAtom(BaseModel):
    model_config = ConfigDict(extra='allow')
    label: str
    pos: List[float]
    spin_S: float

class CrystalStructure(BaseModel):
    model_config = ConfigDict(extra='allow')
    lattice_parameters: Optional[LatticeParameters] = None
    lattice_vectors: Optional[List[List[float]]] = None
    atoms_uc: Optional[List[AtomUC]] = None
    wyckoff_atoms: Optional[List[WyckoffAtom]] = None
    atom_mode: Optional[str] = "symmetry"

class Interaction(BaseModel):
    model_config = ConfigDict(extra='allow')
    type: str
    pair: Optional[List[str]] = None
    rij_offset: Optional[List[int]] = None

class InteractionsDict(BaseModel):
    model_config = ConfigDict(extra='allow')
    heisenberg: Optional[List[Dict[str, Any]]] = []
    dm_interaction: Optional[List[Dict[str, Any]]] = []
    single_ion_anisotropy: Optional[List[Dict[str, Any]]] = []
    anisotropic_exchange: Optional[List[Dict[str, Any]]] = []
    interaction_matrix: Optional[List[Dict[str, Any]]] = []
    kitaev: Optional[List[Dict[str, Any]]] = []
    symmetry_rules: Optional[List[Dict[str, Any]]] = []

class MagCalcConfig(BaseModel):
    model_config = ConfigDict(extra='allow')
    crystal_structure: CrystalStructure
    interactions: Union[List[Dict[str, Any]], InteractionsDict]
    parameters: Optional[Dict[str, Any]] = {}
    parameter_order: Optional[List[str]] = None
    tasks: Optional[Dict[str, Any]] = {}
    minimization: Optional[Dict[str, Any]] = {}
    q_path: Optional[Dict[str, Any]] = {}
    output: Optional[Dict[str, Any]] = {}
    plotting: Optional[Dict[str, Any]] = {}
    calculation: Optional[Dict[str, Any]] = {}
    magnetic_structure: Optional[Dict[str, Any]] = {}
