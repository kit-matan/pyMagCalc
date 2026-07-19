import typer
import yaml
import os
import logging
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
def mcif(
    filename: Annotated[str, typer.Argument(help="Path to the .mcif file")],
    out: Annotated[str, typer.Option(help="Write a runnable config fragment here")] = None,
    spin_s: Annotated[float, typer.Option(help="Spin magnitude S for every site")] = 1.0,
    ion: Annotated[str, typer.Option(help="Ion label for form factors, e.g. Fe2+")] = None,
):
    """
    Import a magnetic CIF (mCIF): expand the magnetic space group into the full magnetic
    cell and print (or write) the per-site spin directions.
    """
    from magcalc.mcif import mcif_to_config_fragment, read_mcif
    import numpy as _np

    data = read_mcif(filename)
    typer.secho(f"{len(data['sites'])} magnetic sites in the cell:", fg=typer.colors.GREEN)
    for s in data['sites']:
        typer.echo(f"  {s['label']:8s} pos={_np.round(s['pos'], 4).tolist()}  "
                   f"dir={_np.round(s['direction'], 4).tolist()}")
    if out:
        import yaml
        frag = mcif_to_config_fragment(filename, spin_S=spin_s, ion=ion)
        frag['interactions'] = {'symmetry_rules': [
            {'type': 'heisenberg', 'distance': 0.0, 'value': 'J1  # set the bond distance + value'}]}
        frag['parameters'] = {'J1': 1.0}
        frag['parameter_order'] = ['J1']
        frag['tasks'] = {'dispersion': True}
        with open(out, 'w') as f:
            yaml.safe_dump(frag, f, sort_keys=False)
        typer.secho(f"Wrote config fragment to {out} "
                    f"(fill in `interactions` before running).", fg=typer.colors.GREEN)


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
  auto_scale_disp: true  # auto-scale dispersion y-axis to data (set false to use energy_limits_disp)

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
    "corrections",
    "energy_cut",
    "calculate_dispersion",
    "calculate_sqw_map",
    "export_csv",
    "plot_dispersion",
    "plot_sqw_map",
    "plot_structure",
    "run_plotting",
    "run_powder_average",
    "fit",
    "plot_fit",
}


@app.command()
def symmetry(
    config_file: Annotated[str, typer.Argument(help="Path to the config.yaml file")],
    max_distance: Annotated[float, typer.Option(help="Search bonds out to this distance (Å)")] = 8.0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the analysis as JSON")] = False,
):
    """
    Crystal-symmetry analysis of a config: the space group, the symmetry-inequivalent
    bond orbits out to `max_distance`, and the symmetry-ALLOWED exchange matrix for each
    (the analogue of Sunny `print_symmetry_table` / SpinW's bond symmetry table).

    Use it to choose `ref_pair` reference bonds and to see which exchange-matrix entries
    symmetry forces to zero or ties together, before writing `interactions.symmetry_rules`.
    """
    if not os.path.exists(config_file):
        typer.secho(f"Error: File {config_file} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    try:
        with open(config_file) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        typer.secho(f"YAML Parse Failed:\n{e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Accept a single wrapper key (e.g. `my_model:`), matching `validate`.
    section = data
    if "crystal_structure" not in section and len(section) == 1:
        inner = next(iter(section.values()))
        if isinstance(inner, dict):
            section = inner
    if "crystal_structure" not in section:
        typer.secho("Error: config has no `crystal_structure` to analyze.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        from magcalc.config_builder import MagCalcConfigBuilder
    except ImportError:
        typer.secho("Error: MagCalcConfigBuilder unavailable.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    b = MagCalcConfigBuilder.from_config(section)
    if not b.atoms_uc:
        typer.secho("Error: no atoms could be built from the config.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    sg = b.space_group_number
    given_sg = (section.get("crystal_structure", {}).get("lattice_parameters", {}) or {}).get("space_group")
    sg_source = "given" if given_sg is not None else "detected from structure"
    n_ops = len(b.symmetry_ops["rotations"]) if b.symmetry_ops else 0

    def _relabel(matrix_rows, free):
        # j0, j5, ... -> p1, p2, ... in order of first appearance, for a readable table.
        # (p-names, not A/B/C, so they never collide with single-letter atom labels.)
        names = {sym: f"p{i + 1}" for i, sym in enumerate(free)}
        out = []
        for row in matrix_rows:
            out.append([names.get(e, e) if e in names else
                        _sub_names(e, names) for e in row])
        return out, [names[s] for s in free]

    def _sub_names(expr, names):
        s = expr
        # replace bare and negated symbols (e.g. "-j6") token-wise
        for sym, nm in names.items():
            s = s.replace(sym, nm)
        return s

    orbits = b.analyze_bond_symmetry(max_distance=max_distance)
    results = []
    for o in orbits:
        c = b.get_bond_constraints(o)
        rows, free = _relabel(c["symbolic_matrix"], c["free_parameters"])
        rep = o["representative"]
        results.append({
            "distance": round(float(o["distance"]), 4),
            "multiplicity": int(o["multiplicity"]),
            "atom_i": rep["atom_i"], "atom_j": rep["atom_j"],
            "offset": [int(x) for x in rep["offset"]],
            "little_group_order": int(c["little_group_size"]),
            "allowed_matrix": rows,
            "free_parameters": free,
        })

    if as_json:
        import json
        typer.echo(json.dumps({
            "space_group": int(sg) if sg is not None else None,
            "space_group_source": sg_source, "n_symmetry_ops": n_ops,
            "n_atoms": len(b.atoms_uc), "max_distance": max_distance,
            "bond_orbits": results}, indent=2))
        return

    typer.secho(f"Crystal symmetry analysis: {config_file}", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  Space group: {sg} ({sg_source}), {n_ops} symmetry operations")
    typer.echo(f"  Magnetic atoms in cell: {len(b.atoms_uc)} "
               f"({', '.join(a['label'] for a in b.atoms_uc)})")
    typer.echo(f"\nSymmetry-inequivalent bond orbits up to {max_distance:g} Å "
               f"({len(results)} found):")
    for n, r in enumerate(results, 1):
        typer.secho(f"\n  #{n}  d = {r['distance']:.4f} Å   multiplicity {r['multiplicity']}",
                    fg=typer.colors.CYAN)
        typer.echo(f"      representative: {r['atom_i']} -> {r['atom_j']}  "
                   f"offset {r['offset']}")
        typer.echo(f"      symmetry-allowed exchange matrix (little group order "
                   f"{r['little_group_order']}):")
        w = max((len(e) for row in r["allowed_matrix"] for e in row), default=1)
        for row in r["allowed_matrix"]:
            typer.echo("        [ " + "  ".join(e.rjust(w) for e in row) + " ]")
        typer.echo(f"      free parameters: {', '.join(r['free_parameters']) or '(none)'}")


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

    # Identify the model section, allowing a single wrapper key (a config whose model
    # keys are all nested one level down under a single wrapper, e.g. `cvo_model:`).
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

@app.command()
def fit(
    config_file: Annotated[str, typer.Argument(help="Path to the config.yaml file")]
):
    """
    Fit the model to neutron data defined in the config's `fitting:` block.

    Reads the same config as `run`, but forces the `fit` task on so the fitting
    engine drives the calculation. The config must contain a `fitting:` section
    (type / data_file / vary / ...). On success the best-fit parameters, an lmfit
    report, and a data-vs-model comparison plot are written.
    """
    if not os.path.exists(config_file):
        typer.secho(f"Error: File {config_file} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Inject `tasks.fit = True` into a temporary copy so the runner takes the
    # fit branch even if the on-disk config left it off.
    try:
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        typer.secho(f"YAML Parse Failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if 'fitting' not in data and not any(
        isinstance(v, dict) and 'fitting' in v for v in data.values()
    ):
        typer.secho(
            "Error: config has no `fitting:` block. See TUTORIAL.md for the format.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    data.setdefault('tasks', {})['fit'] = True
    # Write the temp config next to the original so relative paths (data_file,
    # output filenames) still resolve against the config's directory.
    import tempfile
    cfg_dir = os.path.dirname(os.path.abspath(config_file))
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False, dir=cfg_dir) as tf:
        yaml.safe_dump(data, tf)
        tmp_path = tf.name

    try:
        runner.run_calculation(tmp_path)
        typer.secho("Fit completed successfully.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Fit failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

# Entry point for setuptools
def main():
    app()

if __name__ == "__main__":
    app()
