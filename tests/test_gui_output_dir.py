"""An app run with no opened config file must not scatter its outputs.

Both Studio clients POST to /run-calculation. When the user has opened a real
file the client sends `config_dir` and the run happens THERE, so its outputs
land beside the config -- correct, and unchanged by these tests. When there is
no opened file (a config built in the editor, or the browser's file picker,
which hands over only a name) the run falls back to the project root, and every
plot and .npz it produced used to be written loose into the checkout.

The plots were at least cleaned before each run; the DATA files were not pinned
at all, so `disp_data.npz`, `thermal_mc.npz` and friends simply accumulated. Such
a run now writes into `GUI_OUTPUT_SUBDIR` instead.

What must NOT change is where the run itself executes: the runner resolves a
config's relative references against the config file's own directory, so the run
config stays at the run root and only the OUTPUT paths move. The last test is
that control -- it is the `config_dir` bug (FileNotFoundError on `from_mcif:`)
which this folder must not reintroduce.
"""
import asyncio
import glob
import importlib.util
import os
import sys

import pytest
import yaml

pytest.importorskip("fastapi")

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))


@pytest.fixture(scope="module")
def server():
    spec = importlib.util.spec_from_file_location(
        "gui_server_outdir", os.path.join(ROOT, "gui", "server.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gui_server_outdir"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _isolate_recent_file(server, tmp_path, monkeypatch):
    """Opening a config records it in ~/.magcalc/recent_configs.json; a test must
    not push tmp paths into the user's own picker."""
    monkeypatch.setattr(server, "RECENT_FILE", str(tmp_path / "recent.json"))


def _chain_config():
    """SW01's FM chain, cut down: dispersion over a handful of q, data saved."""
    return {
        "parameter_order": ["J"],
        "parameters": {"J": -1.0},
        "crystal_structure": {
            "lattice_vectors": [[3.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            "atoms_uc": [{"label": "Cu", "pos": [0.0, 0.0, 0.0],
                          "spin_S": 1.0, "ion": "Cu2+"}],
        },
        "interactions": {
            "symmetry_rules": [{"type": "heisenberg", "distance": 3.0, "value": "J"}],
        },
        "magnetic_structure": {
            "enabled": True, "type": "pattern",
            "pattern_type": "ferromagnetic", "direction": [0.0, 0.0, 1.0],
        },
        "tasks": {"minimization": False, "dispersion": True},
        "q_path": {"Gamma": [0.0, 0.0, 0.0], "H100": [1.0, 0.0, 0.0],
                   "path": ["Gamma", "H100"], "points_per_segment": 5},
        "plotting": {"show_plot": False},
        "calculation": {"cache_mode": "none"},
        "output": {"save_data": True},
    }


def _loose_outputs(directory):
    """Generated files sitting directly in `directory` (the clutter under test).
    The run config is a dotfile and belongs at the run root by design."""
    return sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(directory, "*"))
        if os.path.isfile(p) and os.path.splitext(p)[1] in (".npz", ".png", ".csv", ".txt")
    )


def test_a_run_with_no_open_file_keeps_the_project_root_clean(server, tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(server, "project_root", str(root))

    asyncio.run(server.trigger_calculation({"config": _chain_config()}))

    out = root / server.GUI_OUTPUT_SUBDIR
    assert _loose_outputs(root) == [], (
        "app run scattered outputs into the project root")
    # Both halves land there: the plot (pinned before) and the data (was not).
    assert (out / "disp_plot.png").is_file()
    assert (out / "disp_data.npz").is_file()
    # The runnable record of the run stays at the root -- see the last test.
    assert (root / ".config_gui_run.yaml").is_file()


def test_the_data_filenames_a_config_chose_are_kept_inside_the_folder(server, tmp_path,
                                                                     monkeypatch):
    """Relocating outputs must not rename them: a config asking for its own
    filename gets that file, in the folder. (`G1_h00_d.npz` at the checkout root
    is exactly how an unpinned custom name escaped.)"""
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(server, "project_root", str(root))

    cfg = _chain_config()
    cfg["output"] = {"save_data": True, "disp_data_filename": "G1_h00_d.npz"}
    asyncio.run(server.trigger_calculation({"config": cfg}))

    assert (root / server.GUI_OUTPUT_SUBDIR / "G1_h00_d.npz").is_file()
    assert _loose_outputs(root) == []


def test_run_artifact_serves_from_the_output_folder(server, tmp_path, monkeypatch):
    """The UI fetches plots by bare name; it must not care that they moved."""
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(server, "project_root", str(root))

    results = asyncio.run(server.trigger_calculation({"config": _chain_config()}))
    assert "/api/run-artifact/disp_plot.png" in results["plots"]

    served = asyncio.run(server.run_artifact("disp_plot.png"))
    assert os.path.dirname(served.path) == str(root / server.GUI_OUTPUT_SUBDIR)


def test_an_opened_file_still_writes_beside_its_config(server, tmp_path, monkeypatch):
    """The `config_dir` path is unchanged: outputs belong next to the config the
    user opened (that is how the manuscript-figure directories are organised)."""
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(server, "project_root", str(root))
    work = tmp_path / "work"
    work.mkdir()

    asyncio.run(server.trigger_calculation(
        {"config": _chain_config(), "config_dir": str(work)}))

    assert (work / "disp_plot.png").is_file()
    assert not (work / server.GUI_OUTPUT_SUBDIR).exists()
    assert _loose_outputs(root) == []


def test_the_run_config_stays_where_relative_references_resolve(server, tmp_path,
                                                               monkeypatch):
    """The control. `from_mcif:`, `fitting.data_file:` and friends resolve against
    the config file's OWN directory, so writing the run config into the output
    folder would silently re-root every one of them one level down. Only the
    output paths may move."""
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(server, "project_root", str(root))

    asyncio.run(server.trigger_calculation({"config": _chain_config()}))

    assert not (root / server.GUI_OUTPUT_SUBDIR / ".config_gui_run.yaml").exists()
    written = yaml.safe_load(open(root / ".config_gui_run.yaml"))
    # And the paths inside it are relative to that directory, not absolute.
    assert written["plotting"]["disp_plot_filename"] == os.path.join(
        server.GUI_OUTPUT_SUBDIR, "disp_plot.png")


def test_the_gui_spawns_its_calculations_headless():
    """
    `matplotlib.use('Agg')` at the top of server.py binds the SERVER process, not
    the children it spawns -- and the children are what actually run the config.
    Seven shipped configs set `plotting: {show_plot: true}`, so without an
    explicit env the child calls `plt.show()` on the interactive macOS backend
    and blocks forever on a native window. From the UI that is a calculation
    that starts and then hangs, with /stop-calculation the only way out.

    Asserted on the source rather than by launching a server: the failure is a
    HANG, so a test that reproduced it would hang too.
    """
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "gui" / "server.py").read_text()
    spawn = re.search(r"create_subprocess_exec\((.*?)\)\n", src, re.S)
    assert spawn, "the calculation spawn moved; re-check this guard"
    assert "env=child_env" in spawn.group(1), (
        "the GUI spawns calculations without an explicit environment")
    assert 'MPLBACKEND="Agg"' in src and 'MAGCALC_NO_GUI="1"' in src
