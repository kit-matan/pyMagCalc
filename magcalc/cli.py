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
    
    # Template content (Simplified for brevity, could read from file)
    template = """
crystal_structure:
  lattice_parameters:
    a: 5.0
    b: 5.0
    c: 10.0
    alpha: 90
    beta: 90
    gamma: 90
  atoms_uc:
    - label: "Fe1"
      pos: [0, 0, 0]
      spin_S: 1.0
      magmom_classical: [0, 0, 1]

interactions:
  - type: heisenberg
    pair: ["Fe1", "Fe1"]
    J: "J1"
    rij_offset: [1, 0, 0]

parameters:
  J1: 1.0
  S: 1.0

tasks:
  run_dispersion: true
  plot_dispersion: true
    """.strip()

    with open(filename, "w") as f:
        f.write(template)
    typer.echo(f"Created template config: {filename}")

@app.command()
def validate(
    config_file: Annotated[str, typer.Argument(help="Path to the config.yaml file")]
):
    """
    Validate a configuration file against the schema.
    """
    if not os.path.exists(config_file):
        typer.secho(f"Error: File {config_file} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if MagCalcConfig is None:
        typer.secho("Error: Schema module not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        MagCalcConfig.model_validate(data)
        typer.secho(f"Success: {config_file} is valid.", fg=typer.colors.GREEN)
    
    except Exception as e:
        typer.secho(f"Validation Failed:", fg=typer.colors.RED)
        typer.echo(str(e))
        raise typer.Exit(code=1)

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
