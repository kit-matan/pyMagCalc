"""The two Studios' 3D previews must draw the SAME bond network for a given file.

The web app and the native app each build their own payload for
`/get-visualizer-data` -- `buildStructPayload` in `gui/src/lib/configIO.js` and
`MagCalcConfig.structurePayload` in Swift -- and the endpoint expands the
`symmetry_rules` from it. Nothing checked that the two agreed, and they did not:

  * the web builds ONE `crystal_structure` for both its preview and its run
    config, and deliberately emits only the lattice keys the file declared, so
    the editor's placeholder `space_group: 1` is never injected;
  * the native had TWO builders. `backendInput` (run) carried that rule;
    `structurePayload` (preview) did not -- with no `crystal_structure` in
    `rawImport` it emitted `lattice_parameters` straight off `LatticeParameters`,
    whose `spaceGroup` DEFAULTS TO 1.

`space_group: 1` is P1: the identity alone. The backend honours a declared group
rather than detecting one from the structure (as the CLI does), so every
`ref_pair` rule expanded to its own reference bond and nothing else. On the NaCVO
manuscript configs the native preview drew **18 bonds / 6 out-of-cell partners**
where the web drew **72 / 30** -- of the same file, at the same moment.

The oracle is the endpoint itself: both payloads go through the real
`/get-visualizer-data` and the resulting NETWORKS are compared (bond type, the
two atom labels, and the cell offset). That is the user-visible invariant, and it
is insensitive to harmless payload differences -- key order, a `dimensionality`
one side omits -- that a raw payload diff would fail on.

Needs macOS + Xcode (to build the Swift emitter) + node; skipped elsewhere.
"""
import asyncio
import json
import os
import platform
import shutil
import subprocess
import sys

import pytest
import yaml

pytest.importorskip("fastapi")

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

# One config per shape the preview payload has a branch for: an explicit cell
# whose rules need a DETECTED group (the case that broke), a wyckoff cell with a
# declared space group, and a bare-distance rule set.
CONFIGS = [
    os.path.join("examples", "spinw_tutorials", "SW20_Yb2Ti2O7", "config.yaml"),
    os.path.join("examples", "spinw_tutorials", "SW09_kagome_AFM_DM", "config.yaml"),
    os.path.join("examples", "spinw_tutorials", "SW26_spiral_chain", "config.yaml"),
]


@pytest.fixture(scope="module")
def server():
    sys.path.insert(0, os.path.join(ROOT, "gui"))
    import server as srv
    return srv


@pytest.fixture(scope="session")
def emitter():
    """The built `magcalc-emit-config`; rebuilt only when its sources are newer."""
    def stale():
        if not os.path.isfile(BIN):
            return True
        newest = os.path.getmtime(os.path.join(NATIVE, "project.yml"))
        for sub in ("Sources/Models", "Tools"):
            for dirpath, _dirs, files in os.walk(os.path.join(NATIVE, sub)):
                for f in files:
                    if f.endswith(".swift"):
                        newest = max(newest, os.path.getmtime(os.path.join(dirpath, f)))
        return newest > os.path.getmtime(BIN)

    if stale():
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


def _native_payload(emitter, config_path, no_raw=False):
    cmd = [emitter, "--structure"] + (["--no-raw"] if no_raw else []) + [config_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _web_payload(config_path, no_raw=False):
    cmd = [node, os.path.join(GUI, "tests", "emit_struct_payload.mjs")]
    cmd += (["--no-raw"] if no_raw else []) + [config_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=GUI)
    assert proc.returncode == 0, proc.stderr
    return yaml.safe_load(proc.stdout)


def _network(server, payload):
    """{(type, label_i, label_j, offset)} -- the bonds the preview draws."""
    res = asyncio.run(server.get_visualizer_data({"data": payload}))
    labels = [a["label"] for a in res["atoms"]]
    return {(b["type"], labels[b["atom_i"]], labels[b["atom_j"]],
             tuple(int(v) for v in b["offset"])) for b in res["bonds"]}


@needs_both
@pytest.mark.parametrize("no_raw", [False, True], ids=["opened-file", "editor-state"])
@pytest.mark.parametrize("rel", CONFIGS)
def test_previews_draw_the_same_network(server, emitter, rel, no_raw):
    """Both branches of both emitters. `no_raw` drops the captured document, so
    the payload comes from the editor model alone -- the branch that broke, and
    the one an app-authored config always takes."""
    path = os.path.join(ROOT, rel)
    web = _network(server, _web_payload(path, no_raw))
    native = _network(server, _native_payload(emitter, path, no_raw))
    assert web, "the web preview drew no bonds at all"
    assert native == web, (
        f"the two Studios' 3D previews disagree on {rel} (no_raw={no_raw}):\n"
        f"  web    {len(web)} bonds\n  native {len(native)} bonds\n"
        f"  only web:    {sorted(web - native)[:6]}\n"
        f"  only native: {sorted(native - web)[:6]}")


@needs_both
@pytest.mark.parametrize("no_raw", [False, True], ids=["opened-file", "editor-state"])
def test_native_preview_does_not_inject_a_placeholder_space_group(emitter, no_raw):
    """The specific loss parity was blind to before it existed: a config that
    declares no space group must not acquire one on the way to the preview. P1
    is the identity alone, and the backend honours a declared group instead of
    detecting the real one, so the placeholder does not merely add noise -- it
    silently turns symmetry propagation off."""
    path = os.path.join(ROOT, "examples", "spinw_tutorials", "SW20_Yb2Ti2O7", "config.yaml")
    assert yaml.safe_load(open(path))["crystal_structure"].get("space_group") is None
    cs = _native_payload(emitter, path, no_raw)["crystal_structure"]
    assert "space_group" not in (cs.get("lattice_parameters") or {}), cs
