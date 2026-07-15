from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import List, Dict, Any, Optional, Union, Literal
from typing_extensions import Annotated

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
    # Per-site g-tensor: scalar | [gxx,gyy,gzz] | 3x3 | {g_par, g_perp, axis}.
    # Absent on every atom => legacy global isotropic Zeeman.
    g: Optional[Union[float, str, List[Any], Dict[str, Any]]] = None

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
    # Magnetic supercell (SpinW nExt / Sunny resize_supercell): [n1, n2, n3],
    # {'matrix': [n1, n2, n3]}, or 'auto' (minimal diagonal cell commensurate
    # with magnetic_structure.k). Chemical cell is replicated; q_path stays in
    # chemical RLU.
    magnetic_supercell: Optional[Union[List[int], Dict[str, Any], str]] = None

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
    # Higher-order / on-site terms (see CLAUDE.md 'Hamiltonian terms').
    sia_matrix: Optional[List[Dict[str, Any]]] = []       # full 3x3 anisotropy tensor
    stevens: Optional[List[Dict[str, Any]]] = []          # crystal field B_k^q O_k^q
    biquadratic: Optional[List[Dict[str, Any]]] = []      # B (S_i.S_j)^2
    # Long-range dipolar coupling: {cutoff: <Angstrom>, g: <optional>}
    dipole_dipole: Optional[Dict[str, Any]] = None

class SingleKStructure(BaseModel):
    """Unified propagation-vector magnetic structure (SpinW genmagstr 'helical' /
    Sunny set_spiral analogue). k in RLU; axis (alias n) in global Cartesian
    coordinates, normal to the polarization plane. Exactly one of S0
    (lab-frame cell-0 directions), local_directions (rotating-frame), or a
    u/v basis should be given."""
    model_config = ConfigDict(extra='allow')
    type: Literal['single_k']
    k: List[float]
    axis: Optional[List[float]] = None          # alias: n
    n: Optional[List[float]] = None
    S0: Optional[List[List[float]]] = None
    local_directions: Optional[List[List[float]]] = None
    u: Optional[List[float]] = None
    v: Optional[List[float]] = None
    cone_angle_deg: float = 0.0
    normalize: bool = True
    satellites: Optional[bool] = None           # default: True for S(q,w), False for dispersion
    real_space: bool = False                    # generate lab-frame angles, no rotating frame
    enforce_rotational_symmetry: Literal['warn', 'error', 'off'] = 'warn'
    enabled: bool = True

    @model_validator(mode='after')
    def _axis_alias(self):
        if self.axis is None and self.n is not None:
            self.axis = self.n
        return self

class MultiKComponent(BaseModel):
    model_config = ConfigDict(extra='allow')
    k: List[float]                 # propagation vector, chemical RLU
    m: List[float]                 # amplitude vector, Cartesian
    phase_deg: float = 0.0


class MultiKStructure(BaseModel):
    """Real-space multi-k structure: S_i = sum_m m_m cos(2 pi k_m . r_i + phi_m).

    Requires a magnetic supercell commensurate with every k (use
    crystal_structure.magnetic_supercell: 'auto', which takes the per-axis LCM of
    the k denominators). There is no rotating-frame multi-k theory -- SpinW and
    Sunny likewise require a supercell -- so all k must be commensurate.
    """
    model_config = ConfigDict(extra='allow')
    type: Literal['multi_k']
    components: List[MultiKComponent]
    normalize: bool = True         # rescale each site to |S| = 1
    enabled: bool = True


class ExplicitStructure(BaseModel):
    model_config = ConfigDict(extra='allow')
    type: Literal['explicit']
    explicit_list: Optional[List[Dict[str, Any]]] = None
    configuration: Optional[List[Dict[str, Any]]] = None
    enabled: bool = True

class PatternStructure(BaseModel):
    model_config = ConfigDict(extra='allow')
    type: Literal['pattern']
    pattern_type: Optional[str] = None
    direction: Optional[List[float]] = None
    directions: Optional[List[List[float]]] = None
    enabled: bool = True

class LegacySpiralStructure(BaseModel):
    """Deprecated: use type 'single_k' (same fields)."""
    model_config = ConfigDict(extra='allow')
    type: Literal['spiral']
    k: Optional[List[float]] = None
    axis: Optional[List[float]] = None
    local_directions: Optional[List[List[float]]] = None
    enabled: bool = True

class LegacyPropKStructure(BaseModel):
    """Deprecated: use type 'single_k' (mapped with real_space: true)."""
    model_config = ConfigDict(extra='allow')
    type: Literal['propagation_vector']
    k: Optional[List[float]] = None
    subtype: Optional[str] = None
    u: Optional[List[float]] = None
    v: Optional[List[float]] = None
    n: Optional[List[float]] = None
    cone_angle_deg: Optional[float] = None
    enabled: bool = True

MagneticStructure = Annotated[
    Union[SingleKStructure, MultiKStructure, ExplicitStructure, PatternStructure,
          LegacySpiralStructure, LegacyPropKStructure],
    Field(discriminator='type'),
]

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
    magnetic_structure: Optional[Union[MagneticStructure, Dict[str, Any]]] = {}
    fitting: Optional[Dict[str, Any]] = {}
    powder_average: Optional[Dict[str, Any]] = {}
    # 2-D q-grid constant-energy cuts (tasks: {energy_cut: true}):
    # origin/axis1/axis2 define the grid in RLU, cuts list the energy windows.
    energy_cut: Optional[Dict[str, Any]] = {}
