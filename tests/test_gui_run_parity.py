"""The GUI backend run-path must reproduce `magcalc run` (CLI) exactly.

Both Studio clients (web + native) POST to /run-calculation, which is now
"config-as-source": the client's config dict is passed through `_pin_gui_outputs`
(plot/fit output filenames only), written as canonical YAML, and handed to the
SAME `magcalc.runner.run_calculation` the CLI uses. There is deliberately NO
builder-side re-derivation in the run path any more: the old `expand_config`
route silently diverged from the runner's own expansion for advanced
Hamiltonians / non-standard cells (e.g. SW38's symmetry-breaking
interaction_matrix was dropped entirely -> 35 meV wrong).

These tests pin:
  1. the physics blocks survive the run-path transform verbatim for SW38, the
     config the old route corrupted;
  2. end-to-end numeric parity (app-path dispersion == CLI dispersion) for a
     small advanced config that exercises the previously-broken path.
"""
import os
import sys
import copy
import io

import numpy as np
import pytest
import yaml

pytest.importorskip("fastapi")

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
EX = os.path.join(ROOT, "examples")


@pytest.fixture(scope="module")
def server():
    sys.path.insert(0, os.path.join(ROOT, "gui"))
    import server as srv
    return srv


def _through_run_path(server, cfg):
    """Exactly what /run-calculation (config mode) hands to the runner:
    _pin_gui_outputs on the client's config, canonical-YAML write, YAML
    read-back (run_calculation re-reads the file)."""
    pinned = server._pin_gui_outputs(copy.deepcopy(cfg))
    buf = io.StringIO()
    server.compact_yaml_dump(pinned, buf)
    return yaml.safe_load(buf.getvalue())


def test_run_path_keeps_symmetry_breaking_exchange(server):
    """SW38's interaction_matrix deliberately breaks the lattice symmetry, so any
    builder-side re-expansion drops it (the run config would have NO exchange --
    the old expand_config-based route did exactly that). The run-path transform
    must pass the client's physics blocks through verbatim."""
    cfg = yaml.safe_load(open(os.path.join(EX, "spinw_tutorials/SW38_Ca2RuO4/config.yaml")))
    final = _through_run_path(server, cfg)
    assert final["interactions"] == cfg["interactions"]
    assert final["parameters"] == cfg["parameters"]
    assert final["magnetic_structure"] == cfg["magnetic_structure"]
    assert final["crystal_structure"] == cfg["crystal_structure"]


@pytest.mark.slow
def test_run_path_matches_cli_dispersion_for_advanced_config(server, tmp_path):
    """End-to-end: the server run-path config must give the SAME dispersion as
    `magcalc run` on the original. Uses SW09 (DM interaction_matrix on kagome) -- an
    advanced case that goes through the previously-divergent path -- kept small."""
    from magcalc.runner import run_calculation

    src = os.path.join(EX, "spinw_tutorials/SW09_kagome_AFM_DM/config.yaml")
    cfg = yaml.safe_load(open(src))
    cfgdir = os.path.dirname(src)

    def prep(c):
        c = copy.deepcopy(c)
        c["tasks"] = {"dispersion": True}
        c["plotting"] = {"save_plot": False, "show_plot": False}
        c["calculation"] = {**(c.get("calculation") or {}), "cache_mode": "none"}
        qp = c.get("q_path") or {}
        if "points_per_segment" in qp:
            qp["points_per_segment"] = 12
        c["q_path"] = qp
        return c

    def run(c, tag):
        npz = os.path.join(str(tmp_path), f"{tag}.npz")
        c = copy.deepcopy(c)
        c["output"] = {"disp_data_filename": npz, "save_data": True}
        p = os.path.join(cfgdir, f".parity_{tag}.yaml")   # in cfg dir so rel paths resolve
        yaml.safe_dump(c, open(p, "w"), sort_keys=False)
        try:
            run_calculation(p)
        finally:
            os.remove(p)
        d = np.load(npz)
        return np.sort(np.real(np.array(d["energies"])), axis=-1)

    cli = run(prep(cfg), "cli")

    # App path: the editor's config through the run-path transform. Re-apply
    # prep afterwards: _pin_gui_outputs pins plot filenames / save_plot, which
    # the hermetic test overrides (it never touches the physics blocks).
    app_cfg = prep(_through_run_path(server, prep(cfg)))
    app = run(app_cfg, "app")

    assert cli.shape == app.shape
    assert np.max(np.abs(cli - app)) < 1e-9, f"app vs CLI diff = {np.max(np.abs(cli - app))}"
