"""A config with a relative reference must run from the web app as it does from
the CLI.

`from_mcif:`, `fitting.data_file:`, `cif_file:` and `python_model_file:` are all
resolved against the CONFIG FILE's own directory. The native Studio has a real
`URL` for the opened file and sends it as `config_dir`, so the server runs there;
the browser's File System Access API exposes only `handle.name`, so the web app
could not, and a run of such a config died on FileNotFoundError in the project
root while `magcalc run <that file>` worked. The web app now opens files through
the server (`/load-config`, which returns an ABSPATH) and sends `config_dir` too.

The oracle is the CLI: same config, same directory, band for band. The control
matters as much as the assertion -- the same payload sent with the WRONG
directory must fail, or the test would pass on a server that ignored `config_dir`
entirely and happened to find the mCIF some other way.
"""
import asyncio
import importlib.util
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml
from fastapi import HTTPException

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
# The shipped config whose crystal comes from a SIBLING file by relative path.
EXAMPLE = os.path.join(ROOT, "examples", "materials", "mcif")


@pytest.fixture(scope="module")
def server():
    spec = importlib.util.spec_from_file_location(
        "gui_server_relpaths", os.path.join(ROOT, "gui", "server.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gui_server_relpaths"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _isolate_recent_file(server, tmp_path, monkeypatch):
    """Opening a config records it in ~/.magcalc/recent_configs.json. A test must
    not push tmp paths into the user's own picker."""
    monkeypatch.setattr(server, "RECENT_FILE", str(tmp_path / "recent.json"))


def _prepare(tmp_path, tag):
    """A copy of the example, patched to save the band energies."""
    d = tmp_path / tag
    shutil.copytree(EXAMPLE, d)
    cfg_path = d / "config_afm_inplane.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    cfg["output"] = {"save_data": True, "disp_data_filename": "disp.npz"}
    cfg["plotting"] = {"save_plot": False, "show_plot": False, "plot_structure": False}
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return d, cfg_path, cfg


def _bands(directory):
    with np.load(os.path.join(directory, "disp.npz")) as z:
        return np.sort(z["energies"], axis=-1)


def _run_cli(cfg_path, workdir):
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys;from magcalc.runner import run_calculation;run_calculation(sys.argv[1])",
         str(cfg_path)],
        cwd=str(workdir), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


def test_web_app_run_in_the_config_directory_matches_the_cli(server, tmp_path):
    cli_dir, cli_cfg, _ = _prepare(tmp_path, "cli")
    app_dir, app_cfg, _ = _prepare(tmp_path, "app")

    _run_cli(cli_cfg, cli_dir)

    # Exactly what the app does: open through the server (which is where the
    # absolute path comes from), then run the parsed config with `config_dir`.
    loaded = asyncio.run(server.load_config({"path": str(app_cfg)}))
    assert loaded["dir"] == str(app_dir)
    asyncio.run(server.trigger_calculation(
        {"config": loaded["config"], "config_dir": loaded["dir"]}))

    np.testing.assert_allclose(_bands(app_dir), _bands(cli_dir), rtol=0, atol=1e-12)


def test_the_same_payload_in_the_wrong_directory_fails(server, tmp_path, monkeypatch):
    """The control. Without `config_dir` the run happens in the project root,
    where `afm_inplane.mcif` does not exist -- which is precisely the failure the
    web app used to hit on every config with a relative reference."""
    app_dir, app_cfg, cfg = _prepare(tmp_path, "control")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(server, "project_root", str(elsewhere))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.trigger_calculation({"config": cfg}))
    assert exc.value.status_code == 500
    assert not (elsewhere / "disp.npz").exists()


def test_load_config_returns_an_abspath_and_remembers_it(server, tmp_path):
    """`dir` is what the client sends back as `config_dir`; the recent list is how
    the user finds the file again, since a browser dialog cannot supply a path."""
    d, cfg_path, _ = _prepare(tmp_path, "recent")

    # A relative path must come back absolute -- a relative one is useless as a
    # run directory, since the server's cwd is not the client's.
    loaded = asyncio.run(server.load_config(
        {"path": os.path.relpath(str(cfg_path), os.getcwd())}))
    assert loaded["path"] == str(cfg_path)
    assert loaded["dir"] == str(d)

    recents = asyncio.run(server.recent_configs())["files"]
    assert recents[0]["path"] == str(cfg_path)
    assert recents[0]["dir"] == str(d)
    assert recents[0]["exists"] is True

    # Newest first, no duplicates.
    other = shutil.copy(str(cfg_path), str(d / "other.yaml"))
    asyncio.run(server.load_config({"path": other}))
    asyncio.run(server.load_config({"path": str(cfg_path)}))
    paths = [f["path"] for f in asyncio.run(server.recent_configs())["files"]]
    assert paths == [str(cfg_path), other]


def test_browse_lists_directories_and_yaml_files(server, tmp_path):
    """The first open has no recent entry, so the picker has to walk the tree."""
    d, cfg_path, _ = _prepare(tmp_path, "browse")
    listing = asyncio.run(server.browse_configs({"dir": str(d)}))
    assert listing["dir"] == str(d)
    assert os.path.basename(str(cfg_path)) in [f["name"] for f in listing["files"]]
    # The mCIF is not a config; only YAML is offered.
    assert "afm_inplane.mcif" not in [f["name"] for f in listing["files"]]

    parent = asyncio.run(server.browse_configs({"dir": str(tmp_path)}))
    assert "browse" in [x["name"] for x in parent["dirs"]]
    assert parent["parent"] == os.path.dirname(str(tmp_path))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.browse_configs({"dir": str(tmp_path / "nope")}))
    assert exc.value.status_code == 404
