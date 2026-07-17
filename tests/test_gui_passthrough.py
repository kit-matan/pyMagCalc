"""The Studio (web/native) run path must not drop the beyond-LSWT config blocks.

Regression for a real integration bug: the server's /expand-config built the run
config from a WHITELIST of top-level keys, silently dropping `scga`, `thermal_mc`,
`sampled_correlations`, `kpm`, `corrections`, `energy_cut`, and `units` — so those
features worked from the CLI but ran with defaults (or not at all) from the apps.

These tests call the server's expand_config + _faithful_run_config directly (no
HTTP) on a payload shaped exactly like the apps' /run-calculation body, and assert
every block survives into the config handed to the runner.
"""
import asyncio
import importlib.util
import os
import sys

import pytest

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


def _client_payload():
    """A /run-calculation body with every beyond-LSWT block, as the apps send it."""
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
    expanded = asyncio.run(server.expand_config({"data": data}))
    final = server._faithful_run_config(expanded, data)

    for key in ("scga", "thermal_mc", "sampled_correlations", "kpm",
                "corrections", "energy_cut", "units", "parameter_order"):
        assert final.get(key) == data[key], f"block {key!r} was dropped or mangled"

    # tasks and calculation extras must survive too
    for t in ("scga", "thermal_mc", "sampled_correlations", "kpm_sqw", "corrections"):
        assert final["tasks"].get(t) is True, f"task flag {t!r} lost"
    assert final["calculation"]["mode"] == "entangled"
    assert final["calculation"]["series_order"] == 4


def test_absent_blocks_are_not_invented(server):
    data = _client_payload()
    for key in ("scga", "thermal_mc", "sampled_correlations", "kpm",
                "corrections", "energy_cut", "units"):
        data.pop(key)
    expanded = asyncio.run(server.expand_config({"data": data}))
    final = server._faithful_run_config(expanded, data)
    for key in ("scga", "thermal_mc", "sampled_correlations", "kpm",
                "corrections", "energy_cut", "units"):
        assert key not in final, f"block {key!r} appeared from nowhere"
