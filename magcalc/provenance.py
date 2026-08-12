"""Which copy of the engine is actually running.

Two stale full copies of this tree have lived on this machine under OneDrive, and
the shadowing they enable is silent. The mechanism is specific, and the obvious
mental model -- "the editable install wins" -- is wrong here:

`pip install -e .` writes `__editable__.magcalc-0.1.0.pth`, whose finder is
*appended* to `sys.meta_path`, i.e. AFTER the built-in `PathFinder`. So anything on
`sys.path` beats it. What lands on `sys.path[0]` depends on the entry point, and
that is what decides whether you are exposed:

* `python -c` / `python -m magcalc` -> `''`, the CWD. **Exposed.**
* `pytest` -> the rootdir it inserts. **Exposed** (running the suite inside a
  stale checkout tests the stale engine, with the stale tests, silently).
* `python scripts/foo.py` -> the script's directory. **Exposed** for any helper
  living inside a stale tree.
* the `magcalc` console script -> the interpreter's `bin/`. Not exposed to CWD
  shadowing, but still exposed via `PYTHONPATH`.

All of the above verified on this machine, and pinned in
`tests/test_install_provenance.py`.

Nothing announces this. You get a working `magcalc` that runs your config against
an OLD engine: your fix is "not applied", a documented key is "unsupported", a
pinned number moved. The symptom is indistinguishable from a real bug in the code
you are editing -- which is why the repo's own notes could only ever offer the
manual check `python -c "import magcalc; print(magcalc.__file__)"`, useless unless
you already suspect it.

So the check is no longer manual. `describe()` is logged once per
`run_calculation` (the record of the confusing run now contains its own
explanation) and printed by `magcalc where`.

THE LOAD-BEARING DETAIL: a `sys.path` scan alone is not enough, and gets the worst
case exactly backwards. When a stale copy wins, the LIVE tree is typically not on
`sys.path` at all -- it is reachable only through the editable finder on
`sys.meta_path` -- so a path-only scan sees a single copy and reports all clear at
the precise moment it should be shouting. `importable_roots()` therefore unions
three sources: the package actually imported, the `sys.path` entries, and the
editable installs' declared roots.
"""
import sys
from pathlib import Path

__all__ = ["package_root", "importable_roots", "describe", "startup_guard_active"]


def startup_guard_active() -> bool:
    """Is the interpreter-startup shadow guard installed on this interpreter?

    That guard (`tools/magcalc_shadow_guard.py`, installed into site-packages)
    is the only thing that can report a stale copy winning OUTRIGHT, because it
    lives outside every `magcalc` copy -- nothing in this file runs in that case.
    Checked against `sys.meta_path` rather than the filesystem, so it reflects
    what is actually loaded in this process.
    """
    return any(
        getattr(finder, "__name__", None) == "ShadowWatcher" for finder in sys.meta_path
    )


def package_root() -> Path:
    """Directory of the `magcalc` package that this process actually imported."""
    import magcalc

    return Path(magcalc.__file__).resolve().parent


def _syspath_roots() -> list:
    roots = []
    for entry in sys.path:
        candidate = Path(entry or ".").resolve() / "magcalc" / "__init__.py"
        if candidate.is_file():
            roots.append(candidate.parent)
    return roots


def _editable_install_roots() -> list:
    """Roots declared by setuptools editable installs, via their finder modules.

    The objects on `sys.meta_path` are classes defined inside a generated
    `__editable___<dist>_finder` module; the `MAPPING` they consult is a
    module-level global, not a class attribute, so it is read off the defining
    module. Best-effort by design -- this is a diagnostic, and an unrecognised
    packaging layout must not break `magcalc run`.
    """
    roots = []
    for finder in sys.meta_path:
        module = sys.modules.get(getattr(finder, "__module__", "") or "")
        mapping = getattr(module, "MAPPING", None)
        if isinstance(mapping, dict) and "magcalc" in mapping:
            try:
                roots.append(Path(mapping["magcalc"]).resolve())
            except (OSError, TypeError):
                continue
    return roots


def importable_roots() -> list:
    """Every `magcalc` package this interpreter could import, imported one first.

    More than one entry means which copy wins depends on the working directory and
    on `PYTHONPATH`, so the same command gives different answers from different
    places.
    """
    roots = [package_root()]
    for root in _syspath_roots() + _editable_install_roots():
        if root not in roots:
            roots.append(root)
    return roots


def _git_head(root: Path):
    """Short HEAD of the checkout containing `root`, or None if not a git repo.

    Reads `.git` directly rather than shelling out: this runs on the startup path
    of every calculation, and a subprocess against a cloud-synced (dataless)
    checkout can block for a long time -- exactly the situation this module
    exists to diagnose.
    """
    for base in [root, *root.parents]:
        git = base / ".git"
        if not git.is_dir():
            continue
        try:
            head = (git / "HEAD").read_text().strip()
            if not head.startswith("ref: "):
                return head[:12]
            ref = head[5:].strip()
            target = git / ref
            if target.is_file():
                return target.read_text().strip()[:12]
            packed = git / "packed-refs"
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    if line.endswith(" " + ref):
                        return line.split()[0][:12]
            return None
        except OSError:
            return None
    return None


def describe(verbose: bool = False) -> str:
    """One line naming the running engine; a short report when `verbose`."""
    root = package_root()
    head = _git_head(root)
    line = f"magcalc engine: {root}" + (f" (git {head})" if head else "")

    extras = importable_roots()[1:]
    if extras:
        line += (
            f"  [WARNING: {len(extras) + 1} importable copies -- also "
            + ", ".join(str(d) for d in extras)
            + "; which one wins depends on the working directory. Run `magcalc where`.]"
        )

    if not verbose:
        return line

    lines = [line, f"  python:      {sys.executable}", f"  sys.path[0]: {sys.path[0]!r}"]
    if startup_guard_active():
        lines.append("  startup guard: ACTIVE (catches a stale copy winning outright)")
    else:
        lines.append(
            "  startup guard: not installed on this interpreter — a stale copy that"
            "\n                 wins outright would go unreported. Install with:"
            "\n                 python tools/install_shadow_guard.py"
        )
    if not extras:
        lines.append("  no other importable magcalc copy")
    return "\n".join(lines)
