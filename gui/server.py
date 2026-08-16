import os
import io
import yaml
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response, FileResponse
from ase.io import read
from typing import List, Dict, Any
import asyncio
import logging
import uuid
from itertools import product
from starlette.websockets import WebSocket, WebSocketDisconnect
import glob
import json
import re

import sys
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for thread safety
# Add parent directory to sys.path to find magcalc package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from magcalc.config_builder import MagCalcConfigBuilder
except ImportError:
    # If not found (e.g. running in a way where it's not available), we'll handle it in the endpoint
    MagCalcConfigBuilder = None

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from starlette.staticfiles import StaticFiles

# Whitelist of SymPy names safe to expose to parse_expr from untrusted input.
_SAFE_EVAL_FUNCS = {
    "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "atan2": sp.atan2,
    "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "pi": sp.pi, "E": sp.E, "Abs": sp.Abs, "abs": sp.Abs,
}

# Minimal sympy globals needed by parse_expr's transformations (Symbol,
# Integer, etc.) while denying Python's __builtins__.
_SAFE_EVAL_GLOBALS = {
    "__builtins__": {},
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Function": sp.Function,
}

def _safe_eval(expr, ctx):
    """Safely evaluate a mathematical expression using SymPy.

    Uses parse_expr with an explicit whitelist instead of sympify(str(...)),
    which can fall back to eval and execute arbitrary Python code on
    untrusted input. Additionally:

    - rejects any input containing dunder tokens (e.g. ``__class__``,
      ``__bases__``) so attackers cannot walk the Python type hierarchy
      to reach dangerous classes;
    - denies builtins via global_dict so names like ``__import__`` cannot
      resolve to the real built-in.
    """
    if isinstance(expr, (int, float)):
        return float(expr)
    s = str(expr)
    # Block introspection-based escapes outright.
    if "__" in s:
        return 0.0
    try:
        local_dict = dict(_SAFE_EVAL_FUNCS)
        # Convert numeric ctx values into SymPy numbers so they substitute cleanly.
        for k, v in ctx.items():
            if isinstance(v, (int, float)):
                local_dict[str(k)] = sp.Float(v)
        parsed = parse_expr(
            s,
            local_dict=local_dict,
            global_dict=_SAFE_EVAL_GLOBALS,
            evaluate=True,
        )
        return float(parsed.evalf())
    except Exception:
        return 0.0

from magcalc.runner import run_calculation
from magcalc.yaml_io import dump as compact_yaml_dump

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app):
    """FastAPI lifespan handler (replaces the deprecated @app.on_event("startup")).
    `_startup_event` is defined later in the module; it exists by the time the
    server actually starts."""
    await _startup_event()
    yield


app = FastAPI(title="MagCalc Designer Backend", lifespan=_lifespan)

# Mount the project root to serve generated files (e.g. plots)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
app.mount("/files", StaticFiles(directory=project_root), name="files")


def _hydrate_builder(data: Dict[str, Any]) -> 'MagCalcConfigBuilder':
    """
    Shared helper to create and hydrate a MagCalcConfigBuilder from a request payload.
    Handles lattice setup, atom addition (symmetry or explicit), and symmetry detection.
    """
    builder = MagCalcConfigBuilder()

    # 1. Lattice
    struct_data = data.get("crystal_structure", {})
    lattice = struct_data.get("lattice_parameters") or {}
    lattice_vectors = struct_data.get("lattice_vectors")
    atom_mode = struct_data.get("atom_mode", "symmetry")

    if lattice_vectors:
        # Explicit lattice vectors (the CLI / config_loader path also accepts
        # these). Feed them straight to the builder so symmetry detection,
        # neighbour finding and the run config use the CORRECT cell instead of
        # a default 1x1x1. Previously only lattice_parameters was read, so any
        # config written with `lattice_vectors:` was silently given an identity
        # cell in the GUI/native apps.
        import numpy as _np
        lv = _np.array(lattice_vectors, dtype=float)
        builder.lattice_vectors = lv
        builder.config["crystal_structure"]["lattice_vectors"] = lv.tolist()
        # Derive parameters for display / any downstream a,b,c usage.
        a, b, c = (float(_np.linalg.norm(lv[i])) for i in range(3))
        ang = lambda u, v: float(_np.degrees(_np.arccos(
            _np.clip(_np.dot(lv[u], lv[v]) / (_np.linalg.norm(lv[u]) * _np.linalg.norm(lv[v])), -1, 1))))
        builder.lattice_parameters = {"a": a, "b": b, "c": c,
                                      "alpha": ang(1, 2), "beta": ang(0, 2), "gamma": ang(0, 1)}
        builder.config["crystal_structure"]["lattice_parameters"] = builder.lattice_parameters
        if lattice.get("space_group") is not None:
            try:
                builder.space_group_number = int(lattice["space_group"])
                builder._load_symmetry_ops(int(lattice["space_group"]))
            except Exception:
                pass
    else:
        builder.set_lattice(
            a=lattice.get("a", 1.0),
            b=lattice.get("b", 1.0),
            c=lattice.get("c", 1.0),
            alpha=lattice.get("alpha", 90),
            beta=lattice.get("beta", 90),
            gamma=lattice.get("gamma", 90),
            space_group=int(lattice.get("space_group")) if lattice.get("space_group") is not None else None
        )

    # 2. Basis Atoms
    # Accept both the GUI's `wyckoff_atoms` and the CLI/config-file `atoms_uc`
    # key so a raw example config can be passed straight through. When atoms are
    # given as explicit unit-cell positions (atoms_uc, or lattice_vectors with
    # no space group) default to explicit mode.
    wyckoff_atoms = struct_data.get("wyckoff_atoms")
    if not wyckoff_atoms and struct_data.get("atoms_uc"):
        wyckoff_atoms = struct_data.get("atoms_uc")
        if struct_data.get("atom_mode") is None:
            atom_mode = "explicit"
    wyckoff_atoms = wyckoff_atoms or []

    if atom_mode == "explicit":
        config_atoms = []
        for a in wyckoff_atoms:
            config_atoms.append({
                "label": a.get("label", "Atom"),
                "pos": [float(x) for x in a.get("pos", [0, 0, 0])],
                "spin_S": a.get("spin_S", 0.5),
                # `species` is only carried when the config actually gives one.
                # It used to be filled with the per-site LABEL, which made every
                # site of an explicit cell (Cu1, Cu2, Cu3, Cu4) a DISTINCT
                # species to the symmetry detector -- the group collapsed to the
                # identity and no bond rule could propagate. The builder's
                # detector falls back to ion, then to the label with trailing
                # digits stripped, which is what the CLI resolves for the same
                # config.
                "species": a.get("species") or a.get("element"),
                "ion": a.get("ion"),
                "element": a.get("element")
            })
        builder.config["crystal_structure"]["atoms_uc"] = config_atoms
        builder.atoms_uc = config_atoms
        # Bond rules resolve their `ref_pair` labels through this map; the
        # explicit branch never built it, so every rule failed with
        # "Atom 'Cu1' not found in unit cell (keys: [])". `generic_model` builds
        # exactly this for the same branch.
        builder._atom_label_to_idx = {a["label"]: i for i, a in enumerate(config_atoms)}
    else:
        for atom in wyckoff_atoms:
            builder.add_wyckoff_atom(
                label=atom.get("label", "Atom"),
                pos=atom.get("pos", [0, 0, 0]),
                spin=atom.get("spin_S", 0.5),
                ion=atom.get("ion"),
                element=atom.get("element")
            )

    # 3. Symmetry operations, for `interactions.symmetry_rules` propagation.
    #
    # Mirror `generic_model._build_from_config` exactly: a declared `space_group`
    # wins, otherwise detect from the structure. Two ways this used to diverge
    # from the CLI on the same file:
    #   * detection was skipped for `atom_mode: explicit`. How the ATOMS were
    #     given says nothing about whether a bond RULE may be propagated, and
    #     `lattice_vectors` + `atoms_uc` + `symmetry_rules` is ordinary usage
    #     (the whole NaCVO manuscript set). Every rule then failed with "No
    #     symmetry operations loaded" and the visualiser drew ZERO bonds, while
    #     `magcalc run` on the same file expanded them fine.
    #   * the detection here keyed species off `a["species"]`, which the explicit
    #     branch above sets to the per-site LABEL ("Cu1", "Cu2", ...), so every
    #     site fell through `atomic_numbers.get(sym, 1)` to Z=1 -- distinct
    #     sublattices merged into one species and the detected group could be too
    #     high. The builder's own detector keys off species/element/ion and only
    #     falls back to the label with trailing digits stripped.
    try:
        if not builder.space_group_number and builder.atoms_uc:
            builder.detect_symmetry_from_structure()
    except Exception as e:
        logging.getLogger().warning(f"Warning: Failed to detect symmetry: {e}")

    return builder, atom_mode

# --- Log Streaming Setup ---
class LogBroadcaster:
    def __init__(self):
        self.queues = set()

    def add_queue(self, queue):
        self.queues.add(queue)

    def remove_queue(self, queue):
        self.queues.discard(queue)

    def broadcast(self, msg):
        for q in self.queues:
            try:
                q.put_nowait(msg)
            except Exception:
                pass

broadcaster = LogBroadcaster()
MAIN_LOOP = None # Reference to the main event loop
LOGGING_INITIALIZED = False

# Handle to the currently running calculation subprocess (if any). Calculations
# run in a separate, terminable process so the UI can stop them mid-flight; a
# background thread (the previous approach) cannot be killed cooperatively.
CURRENT_PROC = None

# The directory the last run EXECUTED in (its cwd, and the directory its config
# file sits in). File-mode runs execute in the config file's own directory, like
# the CLI, so this can be anywhere -- not just under the /files mount. Outputs
# are served from LAST_OUTPUT_DIR below, which is not always the same place.
LAST_RUN_DIR = None

# Where the last run's OUTPUT files went. Usually the run directory itself, but
# a config-mode run with no `config_dir` falls back to the project root, and
# dropping a dozen plots/.npz into the checkout root is pure clutter -- there is
# no config there they belong to. Such a run writes into this subdirectory
# instead (see `_pin_gui_outputs`). Tracked separately from LAST_RUN_DIR because
# the run still EXECUTES in the run directory: the config's own relative
# references (`from_mcif:`, `fitting.data_file:`) must keep resolving exactly as
# they do for the CLI, which is the whole reason `config_dir` exists.
LAST_OUTPUT_DIR = None

# The dedicated folder for app-generated outputs, relative to the run directory.
GUI_OUTPUT_SUBDIR = "app_runs"

class StreamToLogger:
    """
    Redirects writes to a stream (stdout/stderr) to both the original stream and the broadcaster.
    This is the single entry point for all UI logs (print, tqdm, logging).
    """
    def __init__(self, original_stream):
        self.original_stream = original_stream

    def write(self, buf):
        # Write to original stream (terminal)
        self.original_stream.write(buf)
        self.original_stream.flush()
        
        # Send to broadcaster
        if buf and MAIN_LOOP:
           MAIN_LOOP.call_soon_threadsafe(broadcaster.broadcast, buf)
               
    def flush(self):
        self.original_stream.flush()

async def _startup_event():
    global MAIN_LOOP, LOGGING_INITIALIZED
    MAIN_LOOP = asyncio.get_running_loop()

    # Robust guard: Check if already initialized via flag OR if stdout is already wrapped
    if LOGGING_INITIALIZED or getattr(sys.stdout, '__class__', None).__name__ == 'StreamToLogger':
        return
    LOGGING_INITIALIZED = True

    # 1. Redirect stdout/stderr BEFORE configuring logging
    # This ensures that even StreamHandler(sys.stdout) is intercepted by us.
    sys.stdout = StreamToLogger(sys.stdout)
    sys.stderr = StreamToLogger(sys.stderr)

    # 2. Setup global logger to use sys.stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    
    # Add a standard StreamHandler that writes to the redirected sys.stdout
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Ensure magcalc logger propagates to root
    mc_logger = logging.getLogger("magcalc")
    mc_logger.setLevel(logging.INFO)
    mc_logger.propagate = True

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    
    # Create a unique queue for this connection
    client_queue = asyncio.Queue()
    broadcaster.add_queue(client_queue)
    
    # Send a welcome message
    await websocket.send_text("Connected to Log Stream. Waiting for activity...")
    
    try:
        while True:
            # Wait for next log entry specific to this client
            log_entry = await client_queue.get()
            await websocket.send_text(log_entry)
            client_queue.task_done()
    except WebSocketDisconnect:
        pass
    finally:
        broadcaster.remove_queue(client_queue)

# ... (rest of code)

# The runner's `output:` data files, with its own defaults. Kept in step with
# magcalc/runner.py -- `tests/test_gui_output_dir.py` asserts a run leaves the
# run directory clean, so a key added there and missed here fails the test
# rather than quietly reintroducing a loose file.
GUI_DATA_OUTPUTS = {
    "disp_data_filename": "disp_data.npz",
    "sqw_data_filename": "sqw_data.npz",
    "powder_data_filename": "powder_data.npz",
    "corrections_filename": "corrections.npz",
    "thermal_mc_filename": "thermal_mc.npz",
    "sampled_correlations_filename": "sampled_correlations.npz",
    "scga_filename": "scga.npz",
    "kpm_filename": "kpm_sqw.npz",
    "disp_csv_filename": "disp_data.csv",
    "sqw_csv_filename": "sqw_data.csv",
}


def _pin_gui_outputs(cfg: Dict[str, Any], subdir: str = "") -> Dict[str, Any]:
    """Force the plot/fit output filenames the UI knows how to fetch.

    The ONLY transformation the GUI applies to a config before running it: it
    pins output filenames to fixed names so the frontend can locate and display
    the results. It does NOT touch the physics blocks (crystal_structure,
    interactions, magnetic_structure, parameters, tasks) -- those are written
    and run exactly as the CLI would read them. This is the whole point of the
    file-backed design: `.config_gui_run.yaml` is a real, runnable config.

    `subdir` (relative to the run directory) puts every output the run produces
    in one dedicated folder instead of loose in the run directory. It is a name
    change only -- the run still executes in the run directory, so relative
    references INSIDE the config resolve unchanged. The runner creates the
    parent directory of every output path it writes (`_safe_makedirs`), so a
    relative subdirectory needs nothing else from us.

    Data filenames the config already carries are PREFIXED, not replaced: a
    config asking for `G1_h00_d.npz` keeps that name, in the folder. Otherwise
    the runner's own defaults are pinned explicitly, because unnamed data files
    default relative to the config and would stay loose in the run directory --
    which is how loose `.npz` accumulated at the checkout root while the plots,
    being pinned, were cleaned before every run.
    """
    def _at(name: str) -> str:
        return os.path.join(subdir, name) if subdir else name

    plotting = cfg.setdefault("plotting", {})

    # The GUI drives plotting via the `plotting` block; strip any explicit
    # plot_* task toggles the editor may carry so they don't override it.
    if "tasks" in cfg and isinstance(cfg["tasks"], dict):
        for k in ["plot_structure", "plot_sqw_map", "plot_dispersion"]:
            cfg["tasks"].pop(k, None)

    plotting["save_plot"] = True
    plotting["disp_plot_filename"] = _at("disp_plot.png")
    plotting["sqw_plot_filename"] = _at("sqw_plot.png")
    plotting["powder_plot_filename"] = _at("powder_plot.png")
    plotting["structure_plot_filename"] = _at("mag_structure.png")
    plotting["scga_plot_filename"] = _at("scga_plot.png")
    plotting["thermal_mc_plot_filename"] = _at("thermal_mc_plot.png")
    plotting["sampled_correlations_plot_filename"] = _at("sampled_correlations_plot.png")
    plotting["kpm_plot_filename"] = _at("kpm_plot.png")
    plotting["energy_cut_plot_filename"] = _at("energy_cut.png")

    if cfg.get("tasks", {}).get("fit") or (cfg.get("fitting") or {}).get("enabled"):
        plotting["fit_plot_filename"] = _at("fit_comparison.png")
        output = cfg.setdefault("output", {})
        output["fit_report_filename"] = _at("fit_report.txt")
        output["fit_params_filename"] = _at("fit_params.yaml")

    if subdir:
        output = cfg.setdefault("output", {})
        for key, default in GUI_DATA_OUTPUTS.items():
            name = output.get(key) or default
            # An absolute path is the user saying exactly where it goes.
            if not os.path.isabs(str(name)):
                output[key] = _at(os.path.basename(str(name)))
    return cfg


def _clean_stale_outputs(out_dir: str) -> None:
    """Remove plots/structure JSON from a previous run so the UI never serves stale results."""
    stale = glob.glob(os.path.join(out_dir, "*_plot.png"))
    stale += [os.path.join(out_dir, f) for f in (
        "mag_structure.png", "mag_structure.json", "disp_plot.png", "sqw_plot.png",
        "scga_plot.png", "thermal_mc_plot.png", "sampled_correlations_plot.png",
        "kpm_plot.png", "energy_cut.png", "fit_comparison.png")]
    for p in stale:
        try:
            os.remove(p)
        except OSError:
            pass


@app.post("/run-calculation")
async def trigger_calculation(payload: Dict[str, Any]):
    """
    Run a calculation from a config FILE, exactly as `magcalc run <file>` does.

    The apps are editors of the config file: they send the config they hold and
    the server writes it verbatim (canonical YAML) to `.config_gui_run.yaml`,
    then hands that file to `magcalc.runner.run_calculation` -- the SAME entry
    point the CLI uses. No builder re-derivation, no config translation: a GUI
    run and `magcalc run .config_gui_run.yaml` are byte-for-byte the same run.

    Accepted payloads:
      - {"config": <config dict>}  -- write it and run it (the normal GUI path);
      - {"path": "<file.yaml>"}    -- run an existing on-disk file IN PLACE
                                      (its own directory as cwd), for when the
                                      user has opened a real file to edit.

    Optional with "config": {"config_dir": "<dir>"} -- the directory of the file
    the client has open. The run config is written and executed THERE instead of
    the project root, so relative references inside it resolve exactly as they do
    for `magcalc run <that file>`: `from_mcif`, `fitting.data_file`, `cif_file`,
    `python_model_file`. Without it, a config whose structure comes from a
    sibling mCIF ran fine from the CLI and died on FileNotFoundError in the app.
    Clients that cannot know the directory (the browser's file picker only hands
    over a name) simply omit it and keep the previous project-root behaviour.
    """
    try:
        # The module-level `project_root` (same value; it was recomputed here).
        # Reading the global lets a test point the default run directory
        # somewhere harmless instead of writing into the real checkout.
        default_run_dir = project_root

        # --- Determine what file to run and where ---------------------------
        run_path = payload.get("path")
        # Outputs land in the run directory unless we say otherwise; only the
        # project-root fallback below redirects them into a dedicated folder.
        out_subdir = ""
        if run_path:
            # File mode: run the on-disk file untouched, in its own directory.
            # The config is NOT rewritten here, so its own output filenames
            # stand and the outputs land beside it, exactly as for the CLI.
            run_config_path = os.path.abspath(run_path)
            if not os.path.exists(run_config_path):
                raise HTTPException(status_code=404,
                                    detail=f"Config file not found: {run_config_path}")
            run_dir = os.path.dirname(run_config_path)
        else:
            # Config mode: write the editor's config to the GUI run file and run it.
            cfg = payload.get("config")
            if cfg is None:
                raise HTTPException(status_code=400,
                                    detail="Payload must contain 'config' or 'path'.")
            run_dir = default_run_dir
            cfg_dir = payload.get("config_dir")
            if cfg_dir:
                cfg_dir = os.path.abspath(os.path.expanduser(cfg_dir))
                if not os.path.isdir(cfg_dir):
                    raise HTTPException(
                        status_code=400,
                        detail=f"config_dir is not a directory: {cfg_dir}")
                run_dir = cfg_dir
            else:
                # No opened file: the run falls back to the project root, where
                # nothing owns the outputs. Give them their own folder rather
                # than scattering them through the checkout.
                out_subdir = GUI_OUTPUT_SUBDIR
            cfg = _pin_gui_outputs(dict(cfg), out_subdir)
            # The run config itself stays in the run directory, NOT in the output
            # folder: the runner resolves a config's relative references against
            # `os.path.dirname(config_file)`, so moving it would silently
            # re-root `from_mcif:` / `fitting.data_file:` one level down.
            run_config_path = os.path.join(run_dir, ".config_gui_run.yaml")
            with open(run_config_path, 'w') as f:
                compact_yaml_dump(cfg, f)

        out_dir = os.path.join(run_dir, out_subdir) if out_subdir else run_dir
        os.makedirs(out_dir, exist_ok=True)

        global LAST_RUN_DIR, LAST_OUTPUT_DIR
        LAST_RUN_DIR = run_dir
        LAST_OUTPUT_DIR = out_dir

        _clean_stale_outputs(out_dir)

        logging.getLogger().info(f"Starting calculation with config: {run_config_path}")
        # Explicit info for the UI
        if MAIN_LOOP:
            MAIN_LOOP.call_soon_threadsafe(broadcaster.broadcast, "\n--- NEW CALCULATION STARTED ---\n")

        # 2. Run Calculation: Non-blocking execution in a separate process.
        # Running in a subprocess (rather than a thread) lets the UI terminate
        # the calculation mid-flight via /stop-calculation. We stream the child's
        # stdout/stderr to the log broadcaster.
        #
        # The child sets up the SAME logging format the in-process runner used
        # (timestamp - LEVEL - message) so the UI log console looks unchanged.
        global CURRENT_PROC
        if CURRENT_PROC is not None and CURRENT_PROC.returncode is None:
            raise HTTPException(status_code=409, detail="A calculation is already running.")

        # The import is retried because the project may live in cloud-synced
        # storage (OneDrive/Google Drive): reading a not-yet-hydrated module
        # file can raise TimeoutError ([Errno 60]) while the sync client
        # downloads it. TimeoutError is a subclass of OSError.
        child_script = (
            "import sys, time, logging\n"
            "logging.basicConfig(level=logging.INFO, "
            "format='%(asctime)s - %(levelname)s - %(message)s')\n"
            "logging.getLogger('magcalc').setLevel(logging.INFO)\n"
            "for _attempt in range(3):\n"
            "    try:\n"
            "        from magcalc.runner import run_calculation\n"
            "        break\n"
            "    except OSError as e:\n"
            "        if _attempt == 2:\n"
            "            raise\n"
            "        logging.warning('Import of magcalc failed (%s); file may be "
            "downloading from cloud storage, retrying in 5 s...', e)\n"
            "        time.sleep(5)\n"
            "run_calculation(sys.argv[1])\n"
        )
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-c", child_script,
            run_config_path,
            cwd=run_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        CURRENT_PROC = proc

        try:
            assert proc.stdout is not None
            # Stream raw chunks (not whole lines) so carriage-return progress
            # bars (tqdm) update live, matching the previous in-process behaviour.
            while True:
                chunk = await proc.stdout.read(1024)
                if not chunk:
                    break
                # We're on the main event loop, so broadcasting directly is safe.
                broadcaster.broadcast(chunk.decode(errors="replace"))
            returncode = await proc.wait()
        finally:
            CURRENT_PROC = None

        if returncode != 0:
            # Negative return code => terminated by a signal (i.e. user pressed Stop).
            if returncode < 0:
                raise HTTPException(status_code=499, detail="Calculation stopped by user.")
            raise HTTPException(status_code=500, detail="Calculation failed. See logs for details.")

        # 3. Collect outputs from the run's OUTPUT directory. They are served
        # through /run-artifact (keyed on LAST_OUTPUT_DIR) so a project-root
        # config run (outputs in their own folder), a run beside an opened file,
        # and an in-place file run in any directory all display correctly.
        results = {
            "message": "Calculation completed successfully.",
            "plots": [],
        }

        def _artifact(name: str) -> str:
            # Clean path (the frontend cache-busts <img> tags itself, and keys on
            # the trailing filename to tell PNG from JSON), served via /run-artifact.
            return f"/api/run-artifact/{name}"

        for name in ("disp_plot.png", "sqw_plot.png", "powder_plot.png",
                     "scga_plot.png", "thermal_mc_plot.png",
                     "sampled_correlations_plot.png", "kpm_plot.png",
                     "energy_cut.png"):
            if os.path.exists(os.path.join(out_dir, name)):
                results["plots"].append(_artifact(name))

        if os.path.exists(os.path.join(out_dir, "fit_comparison.png")):
            results["plots"].append(_artifact("fit_comparison.png"))
            # Surface the best-fit parameters to the UI alongside the plot.
            fit_params_path = os.path.join(out_dir, "fit_params.yaml")
            if os.path.exists(fit_params_path):
                try:
                    with open(fit_params_path) as fpf:
                        results["fit_params"] = (yaml.safe_load(fpf) or {}).get(
                            "best_fit_parameters", {}
                        )
                except Exception:
                    pass

        if os.path.exists(os.path.join(out_dir, "mag_structure.png")):
            # Send the JSON structure data too (the 3D viewer prefers it); the
            # frontend decides which to show.
            if os.path.exists(os.path.join(out_dir, "mag_structure.json")):
                results["plots"].append(_artifact("mag_structure.json"))
            results["plots"].append(_artifact("mag_structure.png"))

        return results

    except HTTPException:
        # Pass through deliberate status codes (e.g. 499 user-stop, 409 busy).
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop-calculation")
async def stop_calculation():
    """
    Terminate the currently running calculation subprocess, if any.
    """
    proc = CURRENT_PROC
    if proc is None or proc.returncode is not None:
        return {"stopped": False, "message": "No calculation is currently running."}

    broadcaster.broadcast("\n--- STOPPING CALCULATION (requested by user) ---\n")
    try:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            # Process ignored SIGTERM; force kill.
            proc.kill()
            await proc.wait()
    except ProcessLookupError:
        # Already exited between our check and terminate().
        pass

    return {"stopped": True, "message": "Calculation stopped."}


@app.get("/run-artifact/{name}")
async def run_artifact(name: str):
    """
    Serve a file (plot PNG, structure JSON, fit report) produced by the last run.

    Keyed on LAST_OUTPUT_DIR so it works whether the run wrote into the project
    root's dedicated output folder (config mode with no opened file) or into an
    opened file's own directory (file mode, and config mode with `config_dir`).
    Only a bare basename is honoured, so this cannot read arbitrary paths.
    """
    if LAST_OUTPUT_DIR is None:
        raise HTTPException(status_code=404, detail="No calculation has been run yet.")
    safe = os.path.basename(name)
    path = os.path.join(LAST_OUTPUT_DIR, safe)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Artifact not found: {safe}")
    return FileResponse(path)


# --- Server-side file picking ------------------------------------------------
#
# The browser cannot tell the app WHERE a picked file lives: the File System
# Access API hands over `handle.name` and nothing else, and `<input type=file>`
# is no better. That is not cosmetic -- a run of a config carrying `from_mcif:`,
# `fitting.data_file:` or `cif_file:` resolves those relative to the config's own
# directory, so without it the web app could only run in the project root and
# such a config died on FileNotFoundError while `magcalc run <file>` worked.
#
# The server runs on the same machine as the browser, so it can supply what the
# browser cannot: it opens the file (`/load-config`, which returns an ABSPATH),
# remembers it (`/recent-configs`), and lets the user walk the filesystem to find
# it the first time (`/browse-configs`). The client then sends `config_dir` on
# /run-calculation exactly as the native app does.
RECENT_FILE = os.path.join(os.path.expanduser("~"), ".magcalc", "recent_configs.json")
RECENT_LIMIT = 20


def _load_recent() -> List[str]:
    try:
        with open(RECENT_FILE, "r") as f:
            entries = json.load(f)
        return [p for p in entries if isinstance(p, str)][:RECENT_LIMIT]
    except (OSError, ValueError):
        return []


def _remember_recent(abspath: str) -> None:
    """Most-recent-first, deduplicated. Failure to persist is never fatal: the
    picker is a convenience, and a read-only home directory must not stop a run."""
    entries = [p for p in _load_recent() if p != abspath]
    entries.insert(0, abspath)
    try:
        os.makedirs(os.path.dirname(RECENT_FILE), exist_ok=True)
        with open(RECENT_FILE, "w") as f:
            json.dump(entries[:RECENT_LIMIT], f, indent=1)
    except OSError as e:
        logging.getLogger().warning("Could not record recent config (%s)", e)


@app.get("/recent-configs")
async def recent_configs():
    """Config files opened before, newest first, for the app's Open dialog."""
    out = []
    for p in _load_recent():
        out.append({"path": p, "name": os.path.basename(p),
                    "dir": os.path.dirname(p), "exists": os.path.isfile(p)})
    return {"files": out, "project_root": project_root}


@app.post("/browse-configs")
async def browse_configs(payload: Dict[str, Any]):
    """List sub-directories and YAML files of `dir` (default: the project root).

    This is how a first-time file is found, since the recent list starts empty.
    No path restriction: the server is a local tool bound to localhost, and
    /load-config already opens any absolute path the user names.
    """
    raw = payload.get("dir") or project_root
    directory = os.path.abspath(os.path.expanduser(raw))
    if not os.path.isdir(directory):
        raise HTTPException(status_code=404, detail=f"Not a directory: {directory}")
    dirs, files = [], []
    try:
        for name in sorted(os.listdir(directory), key=str.lower):
            if name.startswith("."):
                continue
            full = os.path.join(directory, name)
            if os.path.isdir(full):
                dirs.append({"name": name, "path": full})
            elif name.endswith((".yaml", ".yml")):
                files.append({"name": name, "path": full})
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Could not read {directory}: {e}")
    parent = os.path.dirname(directory)
    return {"dir": directory, "parent": parent if parent != directory else None,
            "dirs": dirs, "files": files}


@app.post("/load-config")
async def load_config(payload: Dict[str, Any]):
    """
    Read a config file from disk and return its parsed contents plus raw text.

    The app is an editor of the on-disk config file: this opens the exact file
    `magcalc run` would read. The returned `config` is the parsed YAML (what the
    editor populates its state from); `text` is the verbatim file for a raw view.
    `path` is absolute, and is what the client sends back as `config_dir` on
    /run-calculation so relative references inside the config resolve.
    """
    path = payload.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="Payload must contain 'path'.")
    abspath = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(abspath):
        raise HTTPException(status_code=404, detail=f"Config file not found: {abspath}")
    try:
        with open(abspath, "r") as f:
            text = f.read()
        config = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML in {abspath}: {e}")
    _remember_recent(abspath)
    return {"path": abspath, "dir": os.path.dirname(abspath),
            "config": config, "text": text}


@app.post("/save-config")
async def save_config(payload: Dict[str, Any]):
    """
    Write the editor's config to a file on disk, using the canonical serializer.

    This is the "Save" action: it produces exactly the file `magcalc run` reads,
    so a saved config is directly runnable from the CLI. Pass `config` (a dict,
    serialized with the compact dumper) or `text` (verbatim, already-formatted
    YAML) plus the target `path`.
    """
    path = payload.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="Payload must contain 'path'.")
    abspath = os.path.abspath(path)
    try:
        if payload.get("text") is not None:
            with open(abspath, "w") as f:
                f.write(payload["text"])
        elif payload.get("config") is not None:
            with open(abspath, "w") as f:
                compact_yaml_dump(payload["config"], f)
        else:
            raise HTTPException(status_code=400,
                                detail="Payload must contain 'config' or 'text'.")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not write {abspath}: {e}")
    return {"path": abspath, "saved": True}


@app.post("/upload-fit-data")
async def upload_fit_data(file: UploadFile = File(...)):
    """
    Save an uploaded experimental data file for fitting into the project root and
    return the filename the config's `fitting.data_file` should reference.

    The file is stored as ``.fit_data.<ext>`` next to ``.config_gui_run.yaml`` so
    that the runner (cwd = project root) resolves the relative path correctly.
    """
    try:
        content = await file.read()
        gui_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(gui_dir)
        ext = os.path.splitext(file.filename or "")[1].lower() or ".txt"
        if ext not in (".txt", ".csv", ".dat", ".npz"):
            ext = ".txt"
        target_name = f".fit_data{ext}"
        with open(os.path.join(project_root, target_name), "wb") as f:
            f.write(content)
        return {"data_file": target_name, "original_name": file.filename}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/parse-cif")
async def parse_cif(file: UploadFile = File(...)):
    """
    Parse a CIF file and return the crystal structure using Pymatgen.
    Returns the unique Wyckoff positions (asymmetric unit).
    """
    try:
        content = await file.read()
        filename = file.filename
        
        # Save to temporary file
        temp_filename = f"temp_{uuid.uuid4()}.cif"
        with open(temp_filename, "wb") as f:
            f.write(content)
            
        try:
            # Import Pymatgen
            from pymatgen.core import Structure
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            
            # Load Structure
            struct = Structure.from_file(temp_filename)
            
            # Analyze Symmetry
            # symprec=0.1 is robust for experimental data
            sga = SpacegroupAnalyzer(struct, symprec=0.1) 
            
            # Get Symmetrized Structure (groups equivalent atoms)
            symmetrized = sga.get_symmetrized_structure()
            
            # Get Lattice Parameters from the standardized structure or original?
            # Usually better to use the refined structure from SGA
            refined_struct = sga.get_refined_structure()
            lat = refined_struct.lattice
            
            # Get Space Group info
            sg_symbol = sga.get_space_group_symbol()
            sg_number = sga.get_space_group_number()
            
            # Build unique atoms list
            cif_atoms = []
            
            # equivalent_indices is a list of lists: [[0, 1, ...], [4, 5, ...]]
            # Each sublist contains indices of equivalent atoms.
            # We take the first index of each sublist as the representative.
            
            # We need to map these back to the refined structure positions
            # Actually, symmetrized structure preserves the sites of the input (or refined).
            # Let's iterate over the unique sites directly.
            
            for i, group in enumerate(symmetrized.equivalent_indices):
                # Representative index
                rep_idx = group[0]
                site = symmetrized[rep_idx]
                
                # Get Wyckoff symbol for this group
                wyckoff_label = ""
                try:
                    symbols = symmetrized.wyckoff_symbols
                    if rep_idx < len(symbols):
                        wyckoff_label = symbols[rep_idx]
                    else:
                        logging.getLogger().debug(f"DEBUG: rep_idx {rep_idx} out of range for symbols length {len(symbols)}")
                        wyckoff_label = "?"
                except Exception as ex:
                     logging.getLogger().debug(f"DEBUG: Error getting wyckoff: {ex}")
                     wyckoff_label = "?"
                
                # Species
                try:
                    # Try accessing .specie (Element) - older pymatgen or ordered sites
                    species_str = site.specie.symbol
                except AttributeError:
                    # Fallback to .species (Composition) - newer pymatgen or disordered sites
                    # Take the majority element or first element
                    s = site.species
                    if hasattr(s, "elements") and s.elements:
                        species_str = s.elements[0].symbol
                    else:
                        species_str = str(s)
                
                # Label
                if wyckoff_label and wyckoff_label != "?":
                    label = f"{species_str}_{wyckoff_label}"
                else:
                    label = f"{species_str}_{i+1}" # Fallback
                
                # Position (fractional)
                pos = site.frac_coords.tolist()
                
                cif_atoms.append({
                    "label": label,
                    "pos": pos,
                    "spin_S": 0.5,
                    "species": species_str
                })

            return {
                "lattice": {
                    "a": float(lat.a),
                    "b": float(lat.b),
                    "c": float(lat.c),
                    "alpha": float(lat.alpha),
                    "beta": float(lat.beta),
                    "gamma": float(lat.gamma),
                    "space_group": int(sg_number)
                },
                "international": sg_symbol,
                "wyckoff_atoms": cif_atoms
            }

        except ImportError:
             raise HTTPException(status_code=500, detail="Pymatgen not installed. Please install pymatgen.")
        except Exception as e:
             # Fallback or detailed error
             import traceback
             traceback.print_exc()
             raise HTTPException(status_code=500, detail=f"Pymatgen parsing failed: {str(e)}")
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/parse-mcif")
async def parse_mcif(file: UploadFile = File(...),
                     spin_s: float = Form(0.5), ion: str = Form("")):
    """
    Parse a magnetic CIF (mCIF) and return the FULLY EXPANDED magnetic cell:
    lattice, per-site positions, and per-site spin DIRECTIONS.

    Unlike a plain CIF, an mCIF encodes an experimentally-determined magnetic
    structure via a magnetic space group. `magcalc.mcif` expands it into the
    magnetic cell, so the returned atoms are EXPLICIT (atom_mode = explicit,
    P1) and the magnetic_structure is a `generic` pattern with those directions
    -- exactly what LSWT needs. The user still supplies the interactions.
    """
    try:
        from magcalc.mcif import read_mcif

        content = await file.read()
        temp_filename = f"temp_{uuid.uuid4()}.mcif"
        with open(temp_filename, "wb") as f:
            f.write(content)

        try:
            data = read_mcif(temp_filename)
        except Exception as e:
            # read_mcif raises a clear ValueError on an inconsistent file; surface it.
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=400,
                                detail=f"mCIF parsing failed: {str(e)}")
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        A = np.array(data["lattice_vectors"], float)          # rows are a1,a2,a3
        a, b, c = (float(np.linalg.norm(A[i])) for i in range(3))

        def _ang(u, v):
            cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
            return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))

        alpha, beta, gamma = _ang(A[1], A[2]), _ang(A[0], A[2]), _ang(A[0], A[1])

        wyckoff_atoms, directions = [], []
        for i, s in enumerate(data["sites"]):
            # element symbol = leading alphabetic run of the site label (Tb1_1 -> Tb)
            m = re.match(r"[A-Za-z]+", str(s["label"]))
            base = m.group() if m else str(s["label"])
            atom = {
                "label": f"{s['label']}_{i}",
                "pos": [float(x) for x in s["pos"]],
                "spin_S": float(spin_s),
                "species": base,     # web app field
                "element": base,     # native app (WyckoffAtom) field
            }
            if ion:
                atom["ion"] = ion
            wyckoff_atoms.append(atom)
            directions.append([float(x) for x in s["direction"]])

        magnetic_elements = sorted({a2["species"] for a2 in wyckoff_atoms})

        return {
            # P1 magnetic cell (already expanded) -> explicit atoms, no further symmetry
            "lattice": {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta,
                        "gamma": gamma, "space_group": 1},
            "international": "P1 (expanded magnetic cell)",
            "atom_mode": "explicit",
            "wyckoff_atoms": wyckoff_atoms,
            "magnetic_elements": magnetic_elements,
            "magnetic_structure": {
                "enabled": True,
                "type": "pattern",
                "pattern_type": "generic",
                "directions": directions,
            },
            "n_sites": len(wyckoff_atoms),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-neighbors")
async def get_neighbors(config: Dict[str, Any]):
    """
    Find unique neighbor distances for the given crystal structure.
    """
    if MagCalcConfigBuilder is None:
        raise HTTPException(status_code=500, detail="MagCalcConfigBuilder not found on server.")
        
    try:
        data = config.get("data", {})
        builder, atom_mode = _hydrate_builder(data)
        
            
        max_dist = config.get("max_distance", 10.0)
        candidate_bonds = []

        # 3. Calculate Distances
        labels = [a['label'] for a in builder.atoms_uc]
        positions = [np.array(a['pos']) for a in builder.atoms_uc]
        lattice_vecs = builder.lattice_vectors
        
        offsets = list(product([-1, 0, 1], repeat=3))
        
        for i in range(len(positions)):
            for j in range(len(positions)):
                for off in offsets:
                    off_vec = np.array(off)
                    pos_j_img = positions[j] + off_vec
                    dist = np.linalg.norm((pos_j_img - positions[i]) @ lattice_vecs)
                    
                    if 0.1 < dist < max_dist:
                        candidate_bonds.append({
                            "distance": float(dist),
                            "ref_pair": [labels[i], labels[j]],
                            "offset": off_vec.tolist(),
                            "offset_mag": float(np.linalg.norm(off_vec)),
                            "indices": (i, j)
                        })
        
        # Sort candidates to prioritize:
        # 1. Distance (rounded to 5 decimals for shell grouping)
        # 2. Offset Magnitude (prefer zero offset)
        # 3. Atom indices (consistency)
        candidate_bonds.sort(key=lambda b: (round(b["distance"], 5), b["offset_mag"], b["indices"]))

        # 4. Group by symmetry
        unique_shells = []
        if not builder.symmetry_ops:
            # Fallback to simple distance grouping
            processed_keys = set()
            for b in candidate_bonds:
                # Use sorted pair and rounded distance as shell key
                key = (round(b["distance"], 5), tuple(sorted(b["ref_pair"])))
                if key not in processed_keys:
                    unique_shells.append({**b, "multiplicity": 0}) # Logic below will count
                    processed_keys.add(key)
            
            # Count multiplicities
            for shell in unique_shells:
                count = 0
                for b in candidate_bonds:
                    if round(b["distance"], 5) == round(shell["distance"], 5) and \
                       tuple(sorted(b["ref_pair"])) == tuple(sorted(shell["ref_pair"])):
                        count += 1
                shell["multiplicity"] = count
            sorted_shells = unique_shells
        else:
            # Full Symmetry Grouping
            rots = builder.symmetry_ops['rotations']
            trans = builder.symmetry_ops['translations']
            processed_bonds = set()
            orbits = []

            def find_atom_idx(pos):
                p = pos % 1.0
                for idx, a_pos in enumerate(positions):
                    diff = np.abs(a_pos - p)
                    diff = np.minimum(diff, 1.0 - diff)
                    if np.linalg.norm(diff) < 1e-3: return idx
                return None

            # Prepare list for sorting/processing
            bond_keys = []
            for b in candidate_bonds:
                bond_keys.append((
                    b["indices"][0],
                    b["indices"][1],
                    tuple(b["offset"]),
                    b["distance"]
                ))

            for i, j, off, dist in bond_keys:
                if (i, j, off) in processed_bonds: continue
                
                current_orbit = set()
                p_i = positions[i]
                p_j_off = positions[j] + np.array(off)
                
                # Validation: orbit seed distance
                seed_dist = dist
                
                for R, t in zip(rots, trans):
                    p_i_p = R @ p_i + t
                    p_j_o_p = R @ p_j_off + t
                    idx_i_p = find_atom_idx(p_i_p)
                    idx_j_p = find_atom_idx(p_j_o_p)
                    
                    if idx_i_p is not None and idx_j_p is not None:
                        off_i_p = np.round(p_i_p - positions[idx_i_p]).astype(int)
                        off_j_p = np.round(p_j_o_p - positions[idx_j_p]).astype(int)
                        final_off = tuple(off_j_p - off_i_p)
                        
                        # VALIDATION: Check that this generated bond effectively matches the distance
                        # This prevents mismatched 3D ops from polluting 2D orbits
                        d_vec_p = (positions[idx_j_p] + np.array(final_off) - positions[idx_i_p]) @ lattice_vecs
                        d_p = np.linalg.norm(d_vec_p)
                        
                        if abs(d_p - seed_dist) < 0.1:
                            current_orbit.add((idx_i_p, idx_j_p, final_off))
                            # Add reverse too? Handled by symmetry if group includes inversion/-1, but safe to add logically
                            # Wait, reverse bond might not be explicitly in symmetry group ops if no inversion?
                            # Standard neighbor finding usually lists i->j and j->i
                            # We can leave it to symmetry ops to find the inverse if it exists.
                            # current_orbit.add((idx_j_p, idx_i_p, tuple(-np.array(final_off))))
                
                processed_bonds.update(current_orbit)
                orbits.append({
                    "distance": dist,
                    "ref_pair": [labels[i], labels[j]],
                    "offset": list(map(int, off)),
                    "multiplicity": len(current_orbit) // 2,
                    "equivalent_bonds": [
                        {"pair": [labels[int(b[0])], labels[int(b[1])]], "offset": list(map(int, b[2]))}
                        for b in sorted(list(current_orbit), key=lambda x: (x[0], x[1], x[2]))
                        if b[0] <= b[1] # Deduplicate list
                    ]
                })
            sorted_shells = sorted(orbits, key=lambda x: x["distance"])

        # 5. Assign 1st, 2nd, 3rd labels based on distance
        unique_distances = sorted(list(set(round(s["distance"], 5) for s in sorted_shells)))
        dist_to_rank = {d: i + 1 for i, d in enumerate(unique_distances)}
        
        def get_rank_label(rank):
            if 10 <= rank % 100 <= 20: return f"{rank}th"
            else:
                mapping = {1: "st", 2: "nd", 3: "rd"}
                return f"{rank}{mapping.get(rank % 10, 'th')}"

        for s in sorted_shells:
            rank = dist_to_rank[round(s["distance"], 5)]
            s["shell_label"] = get_rank_label(rank)
            s["rank"] = rank

        return sorted_shells[:40]
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-bonds")
async def analyze_bonds(config: Dict[str, Any]):
    """
    Analyze bond symmetry and return orbits.
    Input: { "data": full_config_json, "max_distance": float }
    """
    if MagCalcConfigBuilder is None:
        raise HTTPException(status_code=500, detail="MagCalcConfigBuilder not found on server.")
        
    try:
        data = config.get("data", {})
        max_dist = config.get("max_distance", 6.0)
        builder, atom_mode = _hydrate_builder(data)

        # 4. Analyze
        orbits = builder.analyze_bond_symmetry(max_distance=max_dist)
        
        # 5. Clean for JSON serialization (numpy types)
        def clean_orbit(orb):
            return {
                "distance": float(orb["distance"]),
                "multiplicity": int(orb["multiplicity"]),
                "representative": {
                    "atom_i": int(orb["representative"]["atom_i"]) if isinstance(orb["representative"]["atom_i"], (int, np.integer)) else orb["representative"]["atom_i"],
                    "atom_j": int(orb["representative"]["atom_j"]) if isinstance(orb["representative"]["atom_j"], (int, np.integer)) else orb["representative"]["atom_j"],
                    "offset": [int(x) for x in orb["representative"]["offset"]]
                },
                "members": [
                    {
                        "atom_i": m["atom_i"],
                        "atom_j": m["atom_j"],
                        "offset": [int(x) for x in m["offset"]],
                        "id": str(m["id"])
                    } for m in orb["members"]
                ]
            }
            
        return [clean_orbit(o) for o in orbits]
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bond-constraints")
async def bond_constraints(payload: Dict[str, Any]):
    """
    Get allowed matrix form for a bond.
    Input: { "data": config, "bond": bond_representative_obj }
    """
    if MagCalcConfigBuilder is None:
        raise HTTPException(status_code=500, detail="MagCalcConfigBuilder not found.")
        
    try:
        data = payload.get("data", {})
        bond_rep = payload.get("bond", {})
        builder, atom_mode = _hydrate_builder(data)

        # Config must match bond logic
        # Bond rep should have { "atom_i": "Cu1", "atom_j": "Cu2", "offset": [0,0,0] }
        # The get_bond_constraints expects a dict with "representative" key
        # We wrap it to match the signature
        bond_wrapper = { "representative": bond_rep }
        
        constraints = builder.get_bond_constraints(bond_wrapper)
        return constraints
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/expand-config")
async def expand_config(config: Dict[str, Any]):
    """
    Take a symmetry-based config and return the expanded version.
    """
    if MagCalcConfigBuilder is None:
        raise HTTPException(status_code=500, detail="MagCalcConfigBuilder not found on server.")
        
    try:
        data = config.get("data", {})
        builder, atom_mode = _hydrate_builder(data)
        struct_data = data.get("crystal_structure", {})

        # Global Parameters & Tasks
        builder.config["parameters"] = data.get("parameters", {})
        builder.config["tasks"] = data.get("tasks", {})

        # 3. Interactions
        # The CLI/config_loader accepts several equivalent forms; preserve all
        # of them faithfully instead of collapsing to a single Heisenberg list
        # (which previously dropped single_ion_anisotropy, interaction_matrix,
        # DM and bare-list interactions, so many example configs ran with the
        # wrong -- or empty -- Hamiltonian in the apps).
        EXPLICIT_KEYS = ["heisenberg", "dm_interaction", "single_ion_anisotropy",
                         "interaction_matrix", "anisotropic_exchange", "kitaev"]

        def _expand_rules(rules):
            for rule in rules:
                try:
                    if rule.get("type") == "heisenberg" and not rule.get("ref_pair"):
                        builder.add_interaction_rule(type="heisenberg",
                                                     distance=rule.get("distance"),
                                                     value=rule.get("value"))
                    else:
                        builder.add_symmetry_interaction(type=rule.get("type", "heisenberg"),
                                                         ref_pair=rule.get("ref_pair"),
                                                         distance=rule.get("distance"),
                                                         value=rule.get("value"),
                                                         offset=rule.get("offset"),
                                                         # kitaev needs the bond's
                                                         # Cartesian axis; dropping it
                                                         # silently meant z.
                                                         axis=(rule.get("axis")
                                                               or rule.get("bond_direction")))
                except KeyError as e:
                    print(f"Warning: Skipping invalid interaction rule referencing missing atom: {e}")
                except Exception as e:
                    print(f"Warning: Skipping rule due to error: {e}")
            try:
                builder._expand_heisenberg_rules()
                builder._expand_dm_rules()
                builder._expand_anisotropic_exchange_rules()
                builder._expand_interaction_matrix_rules()
            except Exception as e:
                logging.getLogger().error(f"Expansion Error: {e}")
                import traceback
                traceback.print_exc()

        interactions_in = data.get("interactions", {})
        if isinstance(interactions_in, list):
            # Bare explicit list, e.g. `interactions: [ {type: heisenberg, ...} ]`
            final_inters = interactions_in
        elif isinstance(interactions_in, dict) and "list" in interactions_in:
            # Designer "explicit" mode
            final_inters = interactions_in["list"]
        else:
            idict = interactions_in if isinstance(interactions_in, dict) else {}
            rules = idict.get("symmetry_rules", [])
            has_explicit = any(idict.get(k) for k in EXPLICIT_KEYS)
            if rules and not has_explicit:
                # Pure symmetry-designer workflow: flatten to a Heisenberg/DM/... list
                # (unchanged behaviour for the material examples).
                _expand_rules(rules)
                final_inters = []
                for itype in ["heisenberg", "dm_interaction", "anisotropic_exchange",
                              "interaction_matrix", "kitaev"]:
                    final_inters.extend(builder.config["interactions"].get(itype, []))
            else:
                # Explicit interactions (optionally alongside symmetry_rules):
                # keep every explicit key AND fold in any expanded rules -- as a
                # dict, which config_loader handles natively.
                final_inters = {}
                for k in EXPLICIT_KEYS:
                    if idict.get(k):
                        final_inters[k] = list(idict[k])
                if rules:
                    _expand_rules(rules)
                    for itype in ["heisenberg", "dm_interaction", "anisotropic_exchange",
                                  "interaction_matrix", "kitaev"]:
                        exp = builder.config["interactions"].get(itype, [])
                        if exp:
                            final_inters.setdefault(itype, []).extend(exp)

        # 4. Global Parameters & Tasks
        builder.config["parameters"] = data.get("parameters", {})
        builder.config["tasks"] = data.get("tasks", {})
        
        # Ensure calculation section exists and default to auto cache for GUI performance
        if "calculation" not in builder.config:
            builder.config["calculation"] = {}
        if "cache_mode" not in builder.config["calculation"]:
            builder.config["calculation"]["cache_mode"] = "auto"
        if "cache_file_base" not in builder.config["calculation"]:
             # Use a stable but specific cache name for GUI
             builder.config["calculation"]["cache_file_base"] = "gui_run_cache"

        # Re-build atoms list for response if symmetry was used
        if atom_mode != "explicit":
            config_atoms = []
            for a in builder.atoms_uc:
                config_atoms.append({
                    "label": a["label"],
                    "pos": [float(x) for x in a["pos"]],
                    "spin_S": a["spin_S"],
                    "ion": a.get("ion"),
                    "element": a.get("element")
                })
            builder.config["crystal_structure"]["atoms_uc"] = config_atoms

        # Copy magnetic_elements if present
        if "magnetic_elements" in struct_data:
            builder.config["crystal_structure"]["magnetic_elements"] = struct_data["magnetic_elements"]
        
        # Remove 'S' from parameters as it is defined in atoms_uc
        if "S" in builder.config["parameters"]:
            del builder.config["parameters"]["S"]

        # Restore Parameter Order: H_dir before H_mag (matches config_propagation.yaml)
        params = builder.config["parameters"]
        
        # Ensure H_mag and H_dir are floats
        if "H_mag" in params:
             try:
                 params["H_mag"] = float(params["H_mag"])
             except: pass
        if "H_dir" in params and isinstance(params["H_dir"], (list, tuple)):
             params["H_dir"] = [float(x) if not isinstance(x, str) else x for x in params["H_dir"]]

        ordered_params = {k: v for k, v in params.items() if k not in ["H_mag", "H_dir"]}
        if "H_dir" in params:
            ordered_params["H_dir"] = params["H_dir"]
        if "H_mag" in params:
            ordered_params["H_mag"] = params["H_mag"]
        builder.config["parameters"] = ordered_params

        # (Removed confusing applied_field injection)

        # Preserve the crystal geometry faithfully. When the client supplied
        # explicit lattice_vectors (as every example config does), emit those
        # and drop the derived lattice_parameters so the runner uses the exact
        # cell -- identical to `python -m magcalc run`. In the symmetry-designer
        # workflow (lattice_parameters + space_group) keep the previous shape.
        _had_vectors = bool(struct_data.get("lattice_vectors"))
        _struct_out = dict(builder.config["crystal_structure"])
        _struct_out["dimensionality"] = 3
        if _had_vectors:
            _struct_out["lattice_vectors"] = struct_data["lattice_vectors"]
            _struct_out.pop("lattice_parameters", None)
        else:
            _struct_out.pop("lattice_vectors", None)

        expanded_config = {
            "crystal_structure": _struct_out,
            "interactions": final_inters,
            "magnetic_structure": data.get("magnetic_structure", {}),
            "parameters": builder.config["parameters"],
            "tasks": builder.config["tasks"],
            "q_path": data.get("q_path", {}),
            "minimization": data.get("minimization", {}),
            "plotting": data.get("plotting", {}),
            # Default to 'auto' caching: gen_HM is deterministic per model
            # topology and pickle-cached, so 'none' needlessly regenerates the
            # symbolic matrix every run (~seconds of cold start). Respect an
            # explicit client choice when one is sent.
            "calculation": {"cache_mode": "auto", **(data.get("calculation") or {})},
            "output": data.get("output", {"export_csv": False}),
            "powder_average": data.get("powder_average", {}),
            "fitting": data.get("fitting", {})
        }

        # Beyond-LSWT / auxiliary blocks: pass through verbatim. These are consumed
        # by the runner's task blocks; without this they were silently DROPPED by the
        # whitelist above, so tasks like scga/thermal_mc ran with defaults (or not at
        # all) from the apps while working from the CLI.
        for k in ("scga", "thermal_mc", "sampled_correlations", "kpm",
                  "corrections", "energy_cut", "units", "parameter_order",
                  "hamiltonian_params"):
            if data.get(k) is not None:
                expanded_config[k] = data[k]

        return expanded_config
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-unit-cell-atoms")
async def get_unit_cell_atoms(config: Dict[str, Any]):
    """
    Expand Wyckoff positions or return explicit atoms for visualizer preview.
    """
    if MagCalcConfigBuilder is None:
        raise HTTPException(status_code=500, detail="MagCalcConfigBuilder not found on server.")
        
    try:
        data = config.get("data", {})
        builder, atom_mode = _hydrate_builder(data)

        # Build atoms list from builder
        config_atoms = []
        for a in builder.atoms_uc:
            config_atoms.append({
                "label": a["label"],
                "pos": [float(x) for x in a["pos"]],
                "spin_S": a["spin_S"]
            })
        return config_atoms
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-visualizer-data")
async def get_visualizer_data(config: Dict[str, Any]):
    """
    Return atoms and expanded bonds for visualization.
    Uses full symmetry expansion and rule expansion.
    """
    if MagCalcConfigBuilder is None:
        raise HTTPException(status_code=500, detail="MagCalcConfigBuilder not found on server.")
        
    try:
        data = config.get("data", {})
        builder, atom_mode = _hydrate_builder(data)

        # 3. Interactions.
        # Reset to the builder's OWN empty shape. Listing a subset here was a
        # KeyError waiting to happen: `_add_interaction_matrix_entry` indexes
        # `config["interactions"]["interaction_matrix"]` directly (and a kitaev
        # rule is converted to one), so an `interaction_matrix` symmetry rule
        # died with a bare `KeyError('interaction_matrix')` and its bonds --
        # usually the strongest exchange in the model -- were simply absent from
        # the visualiser.
        builder.config["interactions"] = {
            "heisenberg": [],
            "dm_interaction": [],
            "single_ion_anisotropy": [],
            "anisotropic_exchange": [],
            "interaction_matrix": [],
            "kitaev": [],
            "applied_field": {}
        }
        
        # `interactions` may arrive as a bare list, a {list: [...]}, a dict of
        # explicit type keys, and/or {symmetry_rules: [...]}. Handle them all so
        # example configs (which use lattice_vectors + interaction_matrix, etc.)
        # don't crash the visualiser (previously `.get()` was called on a list).
        def _consume_explicit(items):
            bins = {"heisenberg": "heisenberg", "dm": "dm_interaction",
                    "dm_interaction": "dm_interaction", "dm_manual": "dm_interaction",
                    "anisotropic_exchange": "anisotropic_exchange",
                    "interaction_matrix": "interaction_matrix", "kitaev": "kitaev",
                    "sia": "single_ion_anisotropy",
                    "single_ion_anisotropy": "single_ion_anisotropy"}
            for item in (items or []):
                if not isinstance(item, dict):
                    continue
                key = bins.get(item.get("type", "heisenberg"))
                if key:
                    builder.config["interactions"].setdefault(key, []).append(item)

        interactions_in = data.get("interactions", {})
        if isinstance(interactions_in, list):
            _consume_explicit(interactions_in)
        elif isinstance(interactions_in, dict) and "list" in interactions_in:
            _consume_explicit(interactions_in["list"])
        else:
            idict = interactions_in if isinstance(interactions_in, dict) else {}
            for k in ["heisenberg", "dm_interaction", "single_ion_anisotropy",
                      "interaction_matrix", "anisotropic_exchange", "kitaev"]:
                for e in (idict.get(k) or []):
                    if isinstance(e, dict):
                        builder.config["interactions"].setdefault(k, []).append(
                            {"type": e.get("type", k), **e})
            for rule in idict.get("symmetry_rules", []):
                try:
                    # A bare-`distance` heisenberg rule (no ref_pair) is the one
                    # form `add_symmetry_interaction` refuses -- it belongs to
                    # `add_interaction_rule` + the distance expanders below.
                    # Routing it here anyway (as this endpoint alone did) made
                    # every SW26-style config warn and draw no bonds. The CLI and
                    # /expand-config both split on exactly this.
                    if rule.get("type") == "heisenberg" and not rule.get("ref_pair"):
                        builder.add_interaction_rule(
                            type="heisenberg",
                            distance=rule.get("distance"),
                            value=rule.get("value"),
                        )
                    else:
                        builder.add_symmetry_interaction(
                            type=rule.get("type"),
                            ref_pair=rule.get("ref_pair"),
                            value=rule.get("value"),
                            distance=rule.get("distance"),
                            offset=rule.get("offset"),
                            # kitaev needs the bond's Cartesian axis; dropping it
                            # silently meant z.
                            axis=(rule.get("axis") or rule.get("bond_direction"))
                        )
                except Exception as e:
                    print(f"Warning: Failed to add symmetry rule {rule}: {e}")
        
        # 3. Distance-based expansion, for rules WITHOUT a ref_pair (both the
        # `symmetry_rules` entries routed above and entries placed directly under
        # `interactions.heisenberg` / `dm_interaction` / ...).
        #
        # Same four calls, unconditionally, as `generic_model._build_from_config`
        # -- see the comment there. Gating them on `atom_mode == "symmetry"`
        # conflates how the ATOMS were given with whether a bond RULE may be
        # propagated by distance, and the apps write `atom_mode: explicit`; the
        # `interaction_matrix` expander was missing from the list outright. Each
        # is a no-op when there is no distance rule of its type.
        builder._expand_heisenberg_rules()
        builder._expand_anisotropic_exchange_rules()
        builder._expand_dm_rules()
        builder._expand_interaction_matrix_rules()

        # 4. Prepare Response
        # Atoms
        resp_atoms = []
        for i, a in enumerate(builder.atoms_uc):
            resp_atoms.append({
                "label": a["label"],
                "pos": [float(x) for x in a["pos"]],
                "spin_S": a["spin_S"],
                "idx": i
            })
            
        # Bonds
        resp_bonds = []
        
        # Helper to find index by label (assuming unique labels in atoms_uc)
        label_map = {a["label"]: i for i, a in enumerate(builder.atoms_uc)}
        
        def resolve_idx(lbl):
            # 1. Exact match
            if lbl in label_map: return label_map[lbl]
            # 2. Try appending '0' (Cu -> Cu0)
            if f"{lbl}0" in label_map: return label_map[f"{lbl}0"]
            
            # 3. Try removing leading zeros from numeric part (Cu00 -> Cu0)
            try:
                import re
                match = re.match(r"([A-Za-z]+)(\d+)", lbl)
                if match:
                    prefix, num = match.groups()
                    normalized = f"{prefix}{int(num)}"
                    if normalized in label_map: return label_map[normalized]
            except Exception:
                pass
                
            return None

        # Collect all types
        all_inters = []
        for r in builder.config["interactions"]["heisenberg"]:
             r["interaction_type"] = "heisenberg"
             all_inters.append(r)
        for r in builder.config["interactions"]["dm_interaction"]:
             r["interaction_type"] = "dm"
             all_inters.append(r)
        for r in builder.config["interactions"]["anisotropic_exchange"]:
             r["interaction_type"] = "anisotropic"
             all_inters.append(r)
        # Full 3x3 exchange matrices (interaction_matrix) and Kitaev bonds also
        # define pairs to draw; treat them like anisotropic exchange for display.
        for r in builder.config["interactions"].get("interaction_matrix", []):
             r["interaction_type"] = "matrix"
             all_inters.append(r)
        for r in builder.config["interactions"].get("kitaev", []):
             r["interaction_type"] = "matrix"
             all_inters.append(r)

        for rule in all_inters:
            idx_i = -1
            idx_j = -1
            
            if "pair" in rule:
                lbl_i = rule["pair"][0]
                lbl_j = rule["pair"][1]
                
                idx_i = resolve_idx(lbl_i)
                idx_j = resolve_idx(lbl_j)
                
                # If resolution failed and we are in explicit mode fallback?
                # No, builder should have consistent labels.
            elif "atom_i" in rule and "atom_j" in rule:
                # Direct indices (explicit mode)
                idx_i = rule["atom_i"]
                idx_j = rule["atom_j"]
            
            if idx_i is None or idx_j is None or idx_i == -1 or idx_j == -1:
                continue
            
            offset = rule.get("rij_offset", [0, 0, 0])
             
            # Handle dm_manual specialized offset keys if present
            if "offset_j" in rule:
                offset = rule["offset_j"]
            
            # Format value for label and calculate matrix
            val = rule.get("value")
            label_text = ""
            dm_vec = None
            exchange_matrix = [[0.0]*3 for _ in range(3)]
            
            try:
                if rule["interaction_type"] == "heisenberg":
                    label_text = str(val)
                    # Try to evaluate params for matrix
                    try:
                         # Get parameters dict
                         params = config.get("data", {}).get("parameters", {})
                         ctx = {k: float(v) for k, v in params.items() if isinstance(v, (int, float))}
                         j_val = _safe_eval(val, ctx)
                         exchange_matrix = [
                             [j_val, 0.0, 0.0],
                             [0.0, j_val, 0.0],
                             [0.0, 0.0, j_val]
                         ]
                    except:
                        pass
                elif rule["interaction_type"] == "dm":
                    label_text = "DM"
                    # If val is vector, store it
                    if isinstance(val, list) and len(val) == 3:
                         # Get parameters dict
                         params = config.get("data", {}).get("parameters", {})
                         ctx = {k: float(v) for k, v in params.items() if isinstance(v, (int, float))}
                         
                         eval_vec = []
                         for comp in val:
                             if isinstance(comp, (int, float)):
                                 eval_vec.append(float(comp))
                             elif isinstance(comp, str):
                                 try:
                                     res = _safe_eval(comp, ctx)
                                     eval_vec.append(res)
                                 except Exception:
                                     eval_vec.append(0.0)
                             else:
                                 eval_vec.append(0.0)
                         dm_vec = eval_vec
                         
                         # Construct DM Matrix: [[0, Dz, -Dy], [-Dz, 0, Dx], [Dy, -Dx, 0]]
                         dx, dy, dz = dm_vec
                         exchange_matrix = [
                             [0.0, dz, -dy],
                             [-dz, 0.0, dx],
                             [dy, -dx, 0.0]
                         ]

                elif rule["interaction_type"] == "anisotropic":
                    label_text = "Aniso"
                    # Expecting [Jx, Jy, Jz] or full matrix? usually vector diagonal
                    if isinstance(val, list) and len(val) == 3:
                         params = config.get("data", {}).get("parameters", {})
                         ctx = {k: float(v) for k, v in params.items() if isinstance(v, (int, float))}
                         eval_vec = []
                         for comp in val:
                             try:
                                 v = _safe_eval(comp, ctx)
                                 eval_vec.append(v)
                             except:
                                 eval_vec.append(0.0)
                         
                         exchange_matrix = [
                             [eval_vec[0], 0.0, 0.0],
                             [0.0, eval_vec[1], 0.0],
                             [0.0, 0.0, eval_vec[2]]
                         ]

                elif rule["interaction_type"] == "matrix":
                    # Full 3x3 interaction_matrix / Kitaev bond: value is the matrix.
                    label_text = "Matrix"
                    params = config.get("data", {}).get("parameters", {})
                    ctx = {k: float(v) for k, v in params.items() if isinstance(v, (int, float))}
                    if isinstance(val, list) and len(val) == 3 and all(isinstance(row, list) for row in val):
                        m = []
                        for row in val:
                            out_row = []
                            for comp in row:
                                if isinstance(comp, (int, float)):
                                    out_row.append(float(comp))
                                else:
                                    try:
                                        out_row.append(float(_safe_eval(comp, ctx)))
                                    except Exception:
                                        out_row.append(0.0)
                            m.append(out_row)
                        exchange_matrix = m
            except Exception as e:
                print(f"Matrix Calc Error: {e}")

            resp_bonds.append({
                "atom_i": idx_i,
                "atom_j": idx_j,
                "offset": offset,
                "type": rule["interaction_type"],
                "value": rule.get("value"),
                "rule_value": rule.get("original_value", rule.get("value")),
                "label": label_text,
                "dm_vector": dm_vec,
                "distance": rule.get("distance", 0),
                "exchange_matrix": exchange_matrix
            })

        return {
            "atoms": resp_atoms,
            "bonds": resp_bonds
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/shutdown")
async def shutdown():
    """
    Shut down the server.
    """
    logging.getLogger().info("Shutdown requested. Closing MagCalc...")
    # Use a small delay to allow the response to reach the client if possible
    loop = asyncio.get_running_loop()
    loop.call_later(0.5, os._exit, 0)
    return {"message": "Shutting down..."}

if __name__ == "__main__":
    import uvicorn
    # Default to localhost-only to avoid exposing the API (which evaluates
    # user-supplied expressions) over the network. Set MAGCALC_HOST=0.0.0.0
    # explicitly to bind to all interfaces.
    host = os.environ.get("MAGCALC_HOST", "127.0.0.1")
    port = int(os.environ.get("MAGCALC_PORT", "8000"))
    # Disable reload by default to prevent infinite reload loops when writing result files/plots
    uvicorn.run("server:app", host=host, port=port, reload=False)
