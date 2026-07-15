"""The GUI backend run-path must reproduce `magcalc run` (CLI) exactly.

Both Studio clients (web + native) POST to /run-calculation, which calls
`expand_config` and then runs the SAME `magcalc.runner.run_calculation` the CLI
uses. The only variable is the config transform. `expand_config` re-derives
crystal_structure + interactions with builder-side symmetry, which SILENTLY
DIVERGES from the runner's own expansion for advanced Hamiltonians / non-standard
cells (e.g. SW38's symmetry-breaking interaction_matrix was dropped entirely ->
35 meV wrong). `_faithful_run_config` overlays the client's blocks so the runner
does its own (CLI-identical) expansion.

These tests pin:
  1. the regression itself -- expand_config alone drops SW38's exchange, and
     _faithful_run_config restores it;
  2. end-to-end numeric parity (app-path dispersion == CLI dispersion) for a
     small advanced config that exercises the previously-broken path.
"""
import os
import sys
import copy
import asyncio

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


def _frontend_payload(cfg):
    """Mimic App.jsx buildStructPayload (raw-import branch) + the run-calculation body."""
    cs = cfg.get("crystal_structure", {})
    atoms = cs.get("atoms_uc") or cs.get("wyckoff_atoms") or []
    csp = {
        **({"lattice_vectors": cs["lattice_vectors"]} if cs.get("lattice_vectors")
           else {"lattice_parameters": cs.get("lattice_parameters")}),
        "atoms_uc": atoms, "wyckoff_atoms": atoms,
        "atom_mode": cs.get("atom_mode") or (
            "explicit" if cs.get("atoms_uc") else
            ("symmetry" if cs.get("wyckoff_atoms") else
             ("explicit" if cs.get("lattice_vectors") else "symmetry"))),
        "magnetic_elements": cs.get("magnetic_elements", ["Cu"]), "dimensionality": 3,
    }
    if cs.get("magnetic_supercell") is not None:
        csp["magnetic_supercell"] = cs["magnetic_supercell"]
    return {"crystal_structure": csp, "interactions": cfg.get("interactions"),
            "magnetic_structure": cfg.get("magnetic_structure"),
            "parameters": cfg.get("parameters", {}), "tasks": cfg.get("tasks", {}),
            "q_path": cfg.get("q_path", {}), "minimization": cfg.get("minimization", {}),
            "calculation": cfg.get("calculation", {})}


def test_expand_config_drops_symmetry_breaking_exchange_but_fix_restores_it(server):
    """SW38's interaction_matrix breaks the lattice symmetry, so the builder-side
    re-expansion silently drops it -- the run config would have NO exchange. The fix
    must restore the client's interactions verbatim."""
    cfg = yaml.safe_load(open(os.path.join(EX, "spinw_tutorials/SW38_Ca2RuO4/config.yaml")))
    data = _frontend_payload(cfg)

    naive = asyncio.run(server.expand_config({"data": copy.deepcopy(data)}))
    # The bug: the exchange (symmetry_rules / interaction_matrix) is gone.
    ni = naive["interactions"]
    naive_has_exchange = bool(
        (isinstance(ni, dict) and ni.get("interaction_matrix"))
        or (isinstance(ni, list) and any(x.get("type") == "interaction_matrix" for x in ni)))
    assert not naive_has_exchange, "expected the pre-fix drop; test premise is stale"

    fixed = server._faithful_run_config(copy.deepcopy(naive), data)
    # After the fix the client's interactions (with the exchange) are back verbatim.
    assert fixed["interactions"] == cfg["interactions"]


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

    data = prep(_frontend_payload(cfg))
    exp = asyncio.run(server.expand_config({"data": copy.deepcopy(data)}))
    exp = server._faithful_run_config(exp, data)
    exp = prep(exp)
    app = run(exp, "app")

    assert cli.shape == app.shape
    assert np.max(np.abs(cli - app)) < 1e-9, f"app vs CLI diff = {np.max(np.abs(cli - app))}"
