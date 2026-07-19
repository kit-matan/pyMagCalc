"""The Studio (web/native) run path must not drop the beyond-LSWT config blocks.

Regression for a real integration bug: the server's old run path rebuilt the run
config from a WHITELIST of top-level keys, silently dropping `scga`, `thermal_mc`,
`sampled_correlations`, `kpm`, `corrections`, `energy_cut`, and `units` — so those
features worked from the CLI but ran with defaults (or not at all) from the apps.

The server is now "config-as-source": /run-calculation takes the editor's config
dict, applies ONLY `_pin_gui_outputs` (plot/fit output filenames), writes it as
canonical YAML, and hands the file to the same `magcalc.runner.run_calculation`
the CLI uses. These tests drive exactly that transform (no HTTP) and assert every
block survives into the config the runner reads — pinning that no whitelist /
translation layer creeps back in.
"""
import copy
import importlib.util
import io
import os
import sys

import pytest
import yaml

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))


@pytest.fixture(scope="module")
def server():
    spec = importlib.util.spec_from_file_location(
        "gui_server_under_test", os.path.join(ROOT, "gui", "server.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gui_server_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _through_run_path(server, cfg):
    """Exactly what /run-calculation (config mode) hands to the runner:
    _pin_gui_outputs on the client's config, canonical-YAML write, YAML
    read-back (run_calculation re-reads the file)."""
    pinned = server._pin_gui_outputs(copy.deepcopy(cfg))
    buf = io.StringIO()
    server.compact_yaml_dump(pinned, buf)
    return yaml.safe_load(buf.getvalue())


def _client_payload():
    """A config with every beyond-LSWT block, as the editor holds it."""
    return {
        "crystal_structure": {
            "lattice_parameters": {"a": 5, "b": 5, "c": 5, "alpha": 90, "beta": 90,
                                   "gamma": 90, "space_group": 1},
            "wyckoff_atoms": [{"label": "Cu", "pos": [0, 0, 0], "spin_S": 0.5}],
            "atom_mode": "explicit",
        },
        "interactions": {"symmetry_rules": []},
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
        "parameters": {"J1": 1.0},
        "parameter_order": ["J1"],
        "tasks": {"scga": True, "thermal_mc": True, "sampled_correlations": True,
                  "kpm_sqw": True, "corrections": True, "dispersion": False,
                  "sqw_map": False, "minimization": False},
        "q_path": {"G": [0, 0, 0], "X": [0.5, 0, 0], "path": ["G", "X"],
                   "points_per_segment": 5},
        "plotting": {"save_plot": False},
        "calculation": {"mode": "entangled", "series_order": 4,
                        "cache_mode": "none"},
        "scga": {"temperature": 1.5, "mesh_density": 20},
        "thermal_mc": {"temperatures": [0.5, 1.0], "supercell": [3, 3, 1]},
        "sampled_correlations": {"temperature": 0.5, "supercell": [8, 1, 1]},
        "kpm": {"e_min": 0, "e_max": 8, "e_step": 0.1, "fwhm": 0.3},
        "corrections": {"k_mesh": [8, 8, 8]},
        "energy_cut": {"origin": [0, 0, 0]},
        "units": [[0, 1]],
    }


def test_run_config_keeps_every_beyond_lswt_block(server):
    data = _client_payload()
    final = _through_run_path(server, data)

    for key in ("scga", "thermal_mc", "sampled_correlations", "kpm",
                "corrections", "energy_cut", "units", "parameter_order",
                "interactions", "parameters", "magnetic_structure", "q_path"):
        assert final.get(key) == data[key], f"block {key!r} was dropped or mangled"

    # tasks and calculation extras must survive too
    for t in ("scga", "thermal_mc", "sampled_correlations", "kpm_sqw", "corrections"):
        assert final["tasks"].get(t) is True, f"task flag {t!r} lost"
    assert final["calculation"]["mode"] == "entangled"
    assert final["calculation"]["series_order"] == 4


def test_cu5sbo6_entangled_runs_through_the_app_path(server, tmp_path):
    """Regression for a user-reported failure: importing the Cu5SbO6 config in the web
    app and running it died with "entangled mode needs a `units:` list" -- the old
    run path never carried the top-level `units:` block. With config-as-source the
    editor sends the whole config file; this drives the REAL Cu5SbO6 config through
    the server transform and asserts the entangled dispersion matches the CLI run
    bit-for-bit."""
    import numpy as np

    doc = yaml.safe_load(open(os.path.join(
        ROOT, "examples", "entangled", "Cu5SbO6", "config.yaml")))
    qp = {"G": [0, 0, 0], "X": [0.5, 0, 0.5], "path": ["G", "X"],
          "points_per_segment": 4}

    app_cfg_dict = copy.deepcopy(doc)
    app_cfg_dict["tasks"] = {"dispersion": True, "sqw_map": False,
                             "minimization": False}
    app_cfg_dict["q_path"] = qp
    app_cfg_dict["output"] = {"disp_data_filename": str(tmp_path / "app.npz"),
                              "save_data": True}

    final = _through_run_path(server, app_cfg_dict)
    assert final["units"] == doc["units"]
    assert final["calculation"]["mode"] == "entangled"

    from magcalc.runner import run_calculation
    # Disable plotting for the actual run (hermetic test); _pin_gui_outputs only
    # pins plot filenames and never touches the physics blocks.
    final["plotting"] = {"save_plot": False, "show_plot": False}
    app_cfg = tmp_path / "app.yaml"
    yaml.safe_dump(final, open(app_cfg, "w"))
    run_calculation(str(app_cfg))
    app = np.sort(np.load(tmp_path / "app.npz")["energies"], axis=1)

    cli_cfg = dict(doc)
    cli_cfg["tasks"] = {"dispersion": True, "minimization": False}
    cli_cfg["q_path"] = qp
    cli_cfg["plotting"] = {"save_plot": False, "show_plot": False}
    cli_cfg["output"] = {"disp_data_filename": str(tmp_path / "cli.npz"),
                         "save_data": True}
    cli_file = tmp_path / "cli.yaml"
    yaml.safe_dump(cli_cfg, open(cli_file, "w"))
    run_calculation(str(cli_file))
    cli = np.sort(np.load(tmp_path / "cli.npz")["energies"], axis=1)

    assert np.abs(app - cli).max() == 0.0


def test_absent_blocks_are_not_invented(server):
    data = _client_payload()
    for key in ("scga", "thermal_mc", "sampled_correlations", "kpm",
                "corrections", "energy_cut", "units"):
        data.pop(key)
    final = _through_run_path(server, data)
    for key in ("scga", "thermal_mc", "sampled_correlations", "kpm",
                "corrections", "energy_cut", "units"):
        assert key not in final, f"block {key!r} appeared from nowhere"
