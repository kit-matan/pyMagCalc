"""Opening a config in the Studio and running it == `magcalc run <that file>`.

`test_gui_passthrough.py` pins the SERVER half (it starts from "the config as the
editor holds it" and checks the run path does not filter it). This pins the half
upstream of that, which nothing tested: what the editor holds after you OPEN a
file. The editor used to rebuild the config from a whitelist of keys it knew
about, so a config could run perfectly from the CLI and, opened in the app, run a
quietly different model:

  * `atoms_uc` + the `atom_mode: explicit` the app adds -> the cell came out EMPTY
    (fixed in generic_model, see test_atom_mode_explicit.py);
  * `magnetic_structure` without an explicit `enabled: true` -> DELETED, because
    the runner defaults that key to True and the editor defaulted it to False
    (ZnCVO's bands moved 3.54/3.59/11.00 meV -> 0.62/10.94/11.10/21.42 meV, and
    the run still exited 0);
  * `tasks: {fit: ...}`, `energy_cut`, `static_correlations` -> dropped, so the
    headline task never ran;
  * `from_mcif` -> dropped along with the whole crystal;
  * the anneal-only `minimization.n_sweeps` injected into a `method: TNC` config
    -> "minimize() got an unexpected keyword argument 'n_sweeps'", after which the
    run died at the ground-state guard blaming the magnetic structure;
  * parameters rounded to 5 decimals, degrading every fitted value on open.

The browser-side logic lives in gui/src/lib/configIO.js as pure functions exactly
so it can be driven from here without a browser.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
GUI = os.path.join(ROOT, "gui")

node = shutil.which("node")
needs_node = pytest.mark.skipif(
    node is None or not os.path.isdir(os.path.join(GUI, "node_modules")),
    reason="needs node + `npm install` in gui/ (the Studio's own toolchain)")


@needs_node
def test_every_example_config_survives_open_then_run():
    """The whole examples/ tree, through the app's open -> run transform."""
    proc = subprocess.run([node, os.path.join(GUI, "tests", "roundtrip.test.mjs")],
                          capture_output=True, text=True, cwd=GUI)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _emit_app_config(src, dest):
    """The config the Studio would RUN for `src` (open, then press Run)."""
    proc = subprocess.run(
        [node, os.path.join(GUI, "tests", "emit_run_config.mjs"), src, dest],
        capture_output=True, text=True, cwd=GUI)
    assert proc.returncode == 0, proc.stderr
    return dest


def _run(cfg_path, workdir):
    return subprocess.run(
        [sys.executable, "-c",
         "import sys;from magcalc.runner import run_calculation;run_calculation(sys.argv[1])",
         cfg_path],
        cwd=workdir, capture_output=True, text=True)


def _prepare(example_dir, name, tmp_path, tag, as_app):
    d = tmp_path / tag
    shutil.copytree(example_dir, d)
    cfg_path = str(d / name)
    if as_app:
        cfg_path = _emit_app_config(cfg_path, str(d / "_app_run.yaml"))
    cfg = yaml.safe_load(open(cfg_path))
    # Same forcing on both sides: keep the run headless and make it write the
    # band energies we compare.
    cfg.setdefault("output", {}).update(save_data=True, disp_data_filename="disp.npz")
    cfg.setdefault("plotting", {}).update(save_plot=False, show_plot=False,
                                          plot_structure=False)
    yaml.safe_dump(cfg, open(cfg_path, "w"))
    return cfg_path, d


# One config per failure class. The mCIF one is the cheap representative and
# stays in the fast suite (it also covers the atom_mode/explicit path, since the
# mCIF expands to an explicit cell); the rest are full model builds.
@pytest.mark.parametrize("rel", [
    # from_mcif: no crystal_structure in the file at all (~9 s)
    "examples/materials/mcif/config_afm_inplane.yaml",
    # atoms_uc + atom_mode: explicit (the empty-cell crash), and tasks.energy_cut
    pytest.param("examples/spinw_tutorials/SW10_energy_cut/config.yaml",
                 marks=pytest.mark.slow),
    # magnetic_structure with no `enabled` key (the deleted-order case)
    pytest.param("examples/materials/ZnCVO/config_zncvo.yaml",
                 marks=pytest.mark.slow),
    # minimization method: TNC (the injected anneal-only n_sweeps)
    pytest.param("examples/materials/KFe3J/config_kfe3j.yaml",
                 marks=pytest.mark.slow),
])
@needs_node
def test_app_run_reproduces_cli_run(rel, tmp_path):
    """Not "the YAML looks similar" -- the same bands, from the same engine."""
    src = os.path.join(ROOT, rel)
    example_dir, name = os.path.dirname(src), os.path.basename(src)

    cli_cfg, cli_dir = _prepare(example_dir, name, tmp_path, "cli", as_app=False)
    app_cfg, app_dir = _prepare(example_dir, name, tmp_path, "app", as_app=True)

    cli, app = _run(cli_cfg, cli_dir), _run(app_cfg, app_dir)
    assert cli.returncode == 0, f"the CLI baseline itself failed:\n{cli.stderr[-2000:]}"
    assert app.returncode == 0, (
        f"{rel} runs from the CLI but not from the app:\n{app.stderr[-2000:]}")

    cli_npz, app_npz = cli_dir / "disp.npz", app_dir / "disp.npz"
    assert cli_npz.exists() and app_npz.exists()
    a, b = np.load(cli_npz), np.load(app_npz)
    np.testing.assert_allclose(b["energies"], a["energies"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(b["q_vectors"], a["q_vectors"], rtol=0, atol=1e-12)
