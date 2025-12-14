from typing import List, Dict, Optional, Union, Any, Literal, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


# --- Primitive Types ---
Vector3 = Union[List[float], Tuple[float, float, float]]
# J/D values can be float or string (symbolic)
ParamValue = Union[float, str]
# D Vector can be list of values/strings
DVector = Union[List[ParamValue], Tuple[ParamValue, ParamValue, ParamValue]]

# --- Crystal Structure ---
class LatticeParameters(BaseModel):
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

class AtomConfig(BaseModel):
    label: str
    element: str = "X"
    pos: Vector3
    spin_S: float
    magmom_classical: Union[Vector3, List[float], str] = Field(
        default=[0.0, 0.0, 1.0], 
        description="Classical magnetic moment direction [mx, my, mz], [theta, phi], or special string."
    )

class CrystalStructureConfig(BaseModel):
    lattice_parameters: Optional[LatticeParameters] = None
    # Support raw vector list [[a,0,0], ...]
    lattice_vectors: Optional[List[Vector3]] = None
    atoms_uc: List[AtomConfig] = Field(default_factory=list)
    cif_file: Optional[str] = None
    magnetic_elements: Optional[List[str]] = None
    dimensionality: int = 3

    @model_validator(mode='after')
    def check_structure_source(self):
        if not self.cif_file and not (self.lattice_parameters or self.lattice_vectors):
            raise ValueError("Must provide either 'cif_file' or 'lattice_parameters'/'lattice_vectors'.")
        if not self.cif_file and not self.atoms_uc:
             raise ValueError("Must provide 'atoms_uc' if not using 'cif_file'.")
        return self

# --- Interactions ---
class InteractionBase(BaseModel):
    type: str # Discriminator field
    model_config = ConfigDict(extra='allow') # Allow extra fields for flexibility

class HeisenbergInteraction(InteractionBase):
    type: Literal['heisenberg']
    pair: List[str] # [atom_i, atom_j]
    J: ParamValue
    rij_offset: Vector3 = [0.0, 0.0, 0.0]
    distance: Optional[float] = None
    value: Optional[ParamValue] = None # Alias for J if used

    @model_validator(mode='after')
    def set_value_alias(self):
        # Allow 'value' or 'J'
        if self.value is not None and self.J is None:
            self.J = self.value
        return self

class DMInteraction(InteractionBase):
    type: Literal['dm']
    pair: List[str]
    D_vector: Optional[DVector] = None
    value: Optional[DVector] = None # Alias
    rij_offset: Vector3 = [0.0, 0.0, 0.0]
    distance: Optional[float] = None

class DMManualInteraction(InteractionBase):
    type: Literal['dm_manual']
    atom_i: int
    atom_j: int
    offset_j: Vector3 = [0.0, 0.0, 0.0]
    value: List[str] # Symbolic expressions [Dx, Dy, Dz]

class SIAInteraction(InteractionBase):
    type: Literal['sia']
    atom_label: Optional[str] = None
    K: Optional[ParamValue] = None
    axis: Optional[Vector3] = None

# Union for validation
InteractionType = Union[HeisenbergInteraction, DMInteraction, DMManualInteraction, SIAInteraction, InteractionBase]

# --- Transformations ---
class TransformationFrame(BaseModel):
    atom: int
    rotation: str # Expression string evaluation to matrix

class TransformationsConfig(BaseModel):
    variables: Dict[str, str] = Field(default_factory=dict)
    atom_frames: List[TransformationFrame] = Field(default_factory=list)

# --- Other Sections ---
class CalculationConfig(BaseModel):
    cache_mode: str = 'r'
    cache_file_base: str = 'magcalc_cache'

class QPathConfig(BaseModel):
    points_per_segment: int = 50
    path: List[str]
    # Allow extra fields for point definitions (Dynamic keys)
    model_config = ConfigDict(extra='allow') 

class OutputConfig(BaseModel):
    disp_data_filename: str = 'disp_data.npz'
    sqw_data_filename: str = 'sqw_data.npz'

class PlottingConfig(BaseModel):
    save_plot: bool = True
    plot_structure: bool = False
    disp_plot_filename: str = 'disp_plot.png'
    sqw_plot_filename: str = 'sqw_plot.png'
    show_plot: bool = False
    disp_title: str = "Dispersion"
    sqw_title: str = "S(Q,w)"
    energy_limits_disp: Optional[List[float]] = None
    energy_limits_sqw: Optional[List[float]] = None
    cmap: str = 'PuBu_r'
    broadening_width: float = 0.2
    
    model_config = ConfigDict(extra='allow')

class TasksConfig(BaseModel):
    run_dispersion: bool = False
    calculate_dispersion_new: bool = True
    plot_dispersion: bool = False
    run_sqw_map: bool = False
    calculate_sqw_map_new: bool = True
    plot_sqw_map: bool = False

# --- Main Configuration ---
class MagCalcConfig(BaseModel):
    crystal_structure: CrystalStructureConfig
    interactions: List[InteractionType] = Field(default_factory=list)
    parameters: Dict[str, float] = Field(default_factory=dict) # Name -> Value
    model_params: Optional[Dict[str, float]] = None # Alias/Legacy support
    
    transformations: Optional[TransformationsConfig] = None
    
    calculation: CalculationConfig = Field(default_factory=CalculationConfig)
    q_path: Optional[QPathConfig] = None
    output: OutputConfig = Field(default_factory=OutputConfig)
    plotting: PlottingConfig = Field(default_factory=PlottingConfig)
    tasks: TasksConfig = Field(default_factory=TasksConfig)

    @model_validator(mode='before')
    def normalize_parameters(cls, values):
        # Support either 'parameters' (dict) or strictly specific list order logic?
        # The schema uses 'parameters' as dict of values.
        # But 'examples/KFe3J/config.yaml' has 'model_params' as the dict.
        # Let's map model_params to parameters if needed.
        if 'model_params' in values and 'parameters' not in values:
             values['parameters'] = values['model_params']
        return values
