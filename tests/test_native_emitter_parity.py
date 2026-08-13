"""The native Studio's config emitter must agree with the web app's, exactly.

`gui/src/lib/configIO.js` (web) and `MagCalcConfig.backendInput` (native) are two
implementations of ONE rule -- *the opened file is the base; write only the edits
the user actually made over it*. Only the first had a test:
`tests/test_gui_roundtrip.py` drives it over every shipped config and, for four of
them, runs the app's config through the real engine and compares the spectrum with
the CLI's band for band. The Swift side was checked by compiling it and reading it
next to the JS, and it had drifted on **every one of the 58 shipped configs**:

  * a settings block the file did NOT declare was emitted from the app's struct in
    full, so opening a config with no `minimization:` added `method: anneal,
    n_sweeps: 2000, num_starts: 4, ...` to the run and one with no `fitting:`
    gained a placeholder fit. This is the injected-default class that already cost
    this project a debugging session (OPEN_WORK item 6: the anneal-only `n_sweeps`
    in a `method: TNC` config crashed the minimizer, and the run then died at the
    ground-state guard blaming the magnetic structure);
  * `fitting.data_file`, `vary`, `bounds`, `scale`, `background` and
    `energy_broadening` had NO import branch but were re-emitted from the struct,
    so opening a fitting config and pressing Fit wrote the app's BLANK data file
    and EMPTY vary list over the real ones -- it fitted the wrong thing, or
    nothing. `minimization.n_sweeps` had the same hole;
  * the crystal structure was re-emitted from the file verbatim, so every edit made
    in the Structure panel was silently discarded after opening a config;
  * `parameter_order` was emitted only when the file carried one, and the global
    `S` (which belongs on the atoms) was not stripped.

Parity is the oracle rather than a golden file because the JS side is itself
pinned to `magcalc run` by test_gui_roundtrip -- so "equals the JS emitter" chains
back to the CLI, which is the thing both apps exist to reproduce. A hardcoded
expected payload would be a self-generated golden number and would drift with the
engine.

Needs macOS + Xcode + node; skipped elsewhere.
"""
import json
import os
import platform
import shutil
import subprocess
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(__file__))
import test_config_smoke  # noqa: E402  (the project's own config discovery)

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
GUI = os.path.join(ROOT, "gui")
NATIVE = os.path.join(ROOT, "native", "MagCalcStudio")
BIN = os.path.join(NATIVE, "build", "dd", "Build", "Products", "Release",
                   "magcalc-emit-config")

node = shutil.which("node")
xcodebuild = shutil.which("xcodebuild")
needs_both = pytest.mark.skipif(
    platform.system() != "Darwin" or xcodebuild is None or node is None
    or not os.path.isdir(os.path.join(GUI, "node_modules")),
    reason="needs macOS + Xcode (the Swift emitter) and node + `npm install` in gui/")

# Representative of each shape the emitter has a branch for, so the FAST suite
# still touches all of them: a plain symmetry-rule config, one whose crystal comes
# from an mCIF, one with per-site `g` tensors and explicit atoms, and the fitting
# example (whose data_file/vary/bounds the Swift side used to blank).
FAST_CONFIGS = [
    os.path.join("examples", "spinw_tutorials", "SW01_FM_chain", "config.yaml"),
    os.path.join("examples", "spinw_tutorials", "SW20_Yb2Ti2O7", "config_field.yaml"),
    os.path.join("examples", "fitting", "fit_dispersion.yaml"),
    os.path.join("examples", "materials", "FeI2", "config_fei2.yaml"),
]


def _sources_newer_than(binary):
    """True if any input to the emitter changed since it was last built."""
    if not os.path.isfile(binary):
        return True
    newest = os.path.getmtime(os.path.join(NATIVE, "project.yml"))
    for sub in ("Sources/Models", "Tools"):
        for dirpath, _dirs, files in os.walk(os.path.join(NATIVE, sub)):
            for f in files:
                if f.endswith(".swift"):
                    newest = max(newest, os.path.getmtime(os.path.join(dirpath, f)))
    return newest > os.path.getmtime(binary)


@pytest.fixture(scope="session")
def emitter():
    """Path to the built `magcalc-emit-config`, rebuilt only when stale.

    xcodebuild costs ~18 s even for a no-op, which is why this checks timestamps
    first rather than shelling out every session.

    It does NOT run `xcodegen generate`: the .xcodeproj is tracked and carries an
    explicit file list, so ADDING a Swift file under Sources/Models needs a manual
    regenerate (the documented workflow) before it is compiled here. A test must
    not rewrite a tracked file to paper over that.
    """
    if _sources_newer_than(BIN):
        proc = subprocess.run(
            [xcodebuild, "-project", os.path.join(NATIVE, "MagCalcStudio.xcodeproj"),
             "-scheme", "magcalc-emit-config", "-configuration", "Release",
             "-derivedDataPath", os.path.join(NATIVE, "build", "dd"), "build"],
            capture_output=True, text=True, cwd=NATIVE)
        if proc.returncode != 0:
            pytest.skip("could not build magcalc-emit-config:\n"
                        + proc.stdout[-3000:] + proc.stderr[-2000:])
    assert os.path.isfile(BIN), BIN
    return BIN


def _swift(emitter, config_path):
    proc = subprocess.run([emitter, config_path], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _js(config_path):
    proc = subprocess.run(
        [node, os.path.join(GUI, "tests", "emit_run_config.mjs"), config_path],
        capture_output=True, text=True, cwd=GUI)
    assert proc.returncode == 0, proc.stderr
    return yaml.safe_load(proc.stdout)


def _canon(o):
    """JSON has one number type and YAML has two; nothing else may differ."""
    if isinstance(o, dict):
        return {k: _canon(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_canon(v) for v in o]
    if isinstance(o, bool):
        return o
    if isinstance(o, (int, float)):
        return float(o)
    return o


def _diff(a, b, path=""):
    """Every disagreement, keyed by its config path -- one assert per config would
    otherwise report only the first."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: missing from native (web: {b[k]!r})")
            elif k not in b:
                out.append(f"{path}.{k}: extra in native (native: {a[k]!r})")
            else:
                out += _diff(a[k], b[k], f"{path}.{k}")
    elif a != b:
        out.append(f"{path}: native={a!r} web={b!r}")
    return out


def _assert_parity(emitter, config_path):
    d = _diff(_canon(_swift(emitter, config_path)), _canon(_js(config_path)))
    assert not d, ("native and web emitters disagree on "
                   + os.path.relpath(config_path, ROOT) + ":\n  " + "\n  ".join(d))


@needs_both
@pytest.mark.parametrize("rel", FAST_CONFIGS)
def test_native_and_web_emit_the_same_run_config_fast(emitter, rel):
    _assert_parity(emitter, os.path.join(ROOT, rel))


@needs_both
@pytest.mark.slow
@pytest.mark.parametrize("config_path", test_config_smoke._configs())
def test_native_and_web_emit_the_same_run_config(emitter, config_path):
    """Every config the smoke test runs, through both apps' open -> run transform."""
    _assert_parity(emitter, config_path)


# --- The specific losses that parity was blind to until it existed -----------
# Parity alone cannot catch "both sides wrong the same way", so the three edits
# that motivated the Swift changes are also pinned directly against the FILE.

@needs_both
def test_native_keeps_the_fitting_block_the_file_declared(emitter):
    """data_file / vary / bounds are the fit; blanking them fits nothing."""
    path = os.path.join(ROOT, "examples", "fitting", "fit_dispersion.yaml")
    src = yaml.safe_load(open(path))["fitting"]
    out = _swift(emitter, path)["fitting"]
    assert out["data_file"] == src["data_file"]
    assert out["vary"] == src["vary"]
    assert out["bounds"] == src["bounds"]


@needs_both
def test_native_keeps_per_site_keys_it_does_not_model(emitter):
    """A per-site `g` tensor is the Zeeman term; the app models five atom fields."""
    path = os.path.join(ROOT, "examples", "spinw_tutorials", "SW20_Yb2Ti2O7",
                        "config_field.yaml")
    src = yaml.safe_load(open(path))["crystal_structure"]
    src_atoms = src.get("atoms_uc") or src["wyckoff_atoms"]
    out = _swift(emitter, path)["crystal_structure"]
    out_atoms = out.get("atoms_uc") or out["wyckoff_atoms"]
    assert [a.get("g") for a in out_atoms] == [a.get("g") for a in src_atoms]
    assert any(a.get("g") for a in src_atoms), "config no longer exercises per-site g"


@needs_both
def test_native_does_not_invent_a_block_the_file_omitted(emitter):
    """The injected-default class: an anneal-only `n_sweeps` reaching a config that
    never asked for a minimization is what OPEN_WORK item 6 was."""
    path = os.path.join(ROOT, "examples", "spinw_tutorials", "SW01_FM_chain",
                        "config.yaml")
    assert "minimization" not in yaml.safe_load(open(path)), "config changed"
    assert _swift(emitter, path)["minimization"] == {"enabled": False}
