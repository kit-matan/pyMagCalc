import typer
import yaml
import os
import logging
from typing import Optional
from typing_extensions import Annotated
from magcalc import runner

# Import Schema for validation
try:
    from magcalc.schema import MagCalcConfig
except ImportError:
    MagCalcConfig = None

app = typer.Typer(help="PyMagCalc: Linear Spin-Wave Theory Calculator CLI")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("magcalc")

@app.command()
def init(
    filename: Annotated[str, typer.Argument(help="Filename for the new config")] = "config.yaml"
):
    """
    Generate a template configuration file.
    """
    if os.path.exists(filename):
        typer.confirm(f"{filename} already exists. Overwrite?", abort=True)
    
    # Minimal but runnable template: 1D ferromagnetic chain along a.
    # Demonstrates the exact config shape that runner.run_calculation accepts:
    #   - interactions use `value:` to reference parameter names
    #   - tasks use the keys `runner.py` actually reads (dispersion / sqw_map / ...)
    template = """\
crystal_structure:
  lattice_parameters:
    a: 4.0
    b: 4.0
    c: 4.0
    alpha: 90
    beta: 90
    gamma: 90
  atoms_uc:
    - label: "Fe1"
      pos: [0.0, 0.0, 0.0]
      spin_S: 1.0

interactions:
  - type: heisenberg
    pair: ["Fe1", "Fe1"]
    value: "J1"          # references parameters.J1 below
    rij_offset: [1, 0, 0]

parameters:
  J1: 1.0
  S: 1.0

q_path:
  Gamma: [0.0, 0.0, 0.0]
  X:     [0.5, 0.0, 0.0]
  points_per_segment: 50
  path: ["Gamma", "X"]

tasks:
  dispersion: true       # master switch read by runner.py
  plot_dispersion: true

plotting:
  save_plot: true
  show_plot: false
  disp_plot_filename: "disp_plot.png"
  disp_title: "Spin Wave Dispersion"

calculation:
  cache_mode: "auto"

output:
  disp_data_filename: "disp_data.npz"
"""

    with open(filename, "w") as f:
        f.write(template)
    typer.echo(f"Created template config: {filename}")

# Task keys understood by runner.run_calculation. Anything outside this set
# in `tasks:` is most likely a typo, so we surface it as a warning during
# `validate` even though the schema is otherwise permissive (extra='allow').
_KNOWN_TASK_KEYS = {
    "minimization",
    "dispersion",
    "sqw_map",
    "powder_average",
    "calculate_dispersion",
    "calculate_sqw_map",
    "export_csv",
    "plot_dispersion",
    "plot_sqw_map",
    "plot_structure",
    "run_plotting",
    "run_powder_average",
}


@app.command()
def validate(
    config_file: Annotated[str, typer.Argument(help="Path to the config.yaml file")]
):
    """
    Validate a configuration file against the schema.

    Two formats are accepted:
      - "modern" configs with a top-level `crystal_structure:` block (full
        pydantic schema check);
      - "legacy" configs that drive a user-written spin model via
        `python_model_file:` or `spin_model_module:` (the schema is skipped,
        we only check the minimum keys that runner.run_calculation requires).

    Unknown task keys are reported as warnings (likely typos) — e.g.
    `run_dispersion:` instead of `dispersion:`.
    """
    if not os.path.exists(config_file):
        typer.secho(f"Error: File {config_file} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        typer.secho(f"YAML Parse Failed:", fg=typer.colors.RED)
        typer.echo(str(e))
        raise typer.Exit(code=1)

    # Identify the model section, allowing a single wrapper key
    # (e.g. examples/aCVO/config.yaml wraps everything under `cvo_model:`).
    section = data
    if "crystal_structure" not in section and "python_model_file" not in section \
            and "spin_model_module" not in section and len(section) == 1:
        inner = next(iter(section.values()))
        if isinstance(inner, dict):
            section = inner

    is_modern = "crystal_structure" in section
    is_legacy = "python_model_file" in section or "spin_model_module" in section

    if is_modern:
        if MagCalcConfig is None:
            typer.secho("Error: Schema module not found.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        try:
            MagCalcConfig.model_validate(section)
        except Exception as e:
            typer.secho("Validation Failed:", fg=typer.colors.RED)
            typer.echo(str(e))
            raise typer.Exit(code=1)
    elif is_legacy:
        # Legacy configs drive a user spin_model; runner.run_calculation only
        # needs `parameters` (or `model_params`) plus the spin model itself.
        if "parameters" not in section and "model_params" not in section:
            typer.secho(
                "Validation Failed: legacy config has neither 'parameters' nor 'model_params'.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
    else:
        typer.secho(
            "Validation Failed: config has neither 'crystal_structure' (modern) "
            "nor 'python_model_file'/'spin_model_module' (legacy).",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # Soft check: surface unknown task keys as a warning.
    tasks = section.get("tasks") or {}
    unknown = sorted(k for k in tasks if k not in _KNOWN_TASK_KEYS)
    if unknown:
        typer.secho(
            f"Warning: unknown task key(s) {unknown} will be ignored by the runner. "
            f"Did you mean one of: dispersion, sqw_map, plot_dispersion, ...?",
            fg=typer.colors.YELLOW,
        )

    typer.secho(f"Success: {config_file} is valid.", fg=typer.colors.GREEN)

@app.command()
def run(
    config_file: Annotated[str, typer.Argument(help="Path to the config.yaml file")]
):
    """
    Run calculations defined in the configuration file.
    """
    try:
        runner.run_calculation(config_file)
        typer.secho("Calculation completed successfully.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Calculation failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

# Entry point for setuptools
def main():
    app()

if __name__ == "__main__":
    app()
