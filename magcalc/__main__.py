"""
Module entry point so that `python -m magcalc ...` behaves identically to the
installed `magcalc` console script. Previously this module hosted a separate
argparse-based CLI with a divergent config schema; that path is now gone in
favor of a single Typer-based CLI defined in magcalc.cli.
"""
from magcalc.cli import app


if __name__ == "__main__":
    app()
