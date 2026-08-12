"""Interpreter-startup guard: shout when more than one `magcalc` is importable.

This file is the SOURCE. It is installed into the active interpreter's
site-packages by `tools/install_shadow_guard.py`, alongside a one-line
`zz_magcalc_shadow_guard.pth` that imports it. `site.py` executes `.pth` import
lines at interpreter startup, so this module is ARMED in every Python process on
this interpreter, independently of `import magcalc`. (Armed, not fired -- see the
timing trap below: the check itself happens later.)

That placement is the entire point, and it is what `magcalc/provenance.py` cannot
do. The in-package detector only runs if the good copy wins; when a stale checkout
wins outright it has no `provenance.py`, nothing reports, and the only tell is the
ABSENCE of the `magcalc engine:` line -- a signal you have to already be looking
for. Living outside every `magcalc` copy is what closes that hole. It also closes
the other one: a NEW second checkout is flagged the moment it is importable, so
re-arming the hazard is loud instead of silent.

THE TIMING TRAP, measured the hard way: **checking at `.pth` time does not work,**
and fails on precisely the case that matters. When `site.py` executes `.pth` lines,
`sys.path[0]` is NOT yet the working directory -- for `python -c` the `''` entry is
prepended *after* site initialization. A survey run at startup therefore sees
PYTHONPATH and site-packages but not the cwd, so `cd` into a stale checkout and it
reports all clear. The first version of this file did exactly that and passed a
PYTHONPATH test while missing the real hazard.

So the check is DEFERRED to the moment `magcalc` is actually imported, by parking
an observer at the front of `sys.meta_path`. Its `find_spec` runs with `sys.path`
fully assembled, surveys, warns, and always returns `None` -- it never claims the
module, it only watches the resolution happen.

Design constraints, in priority order:

1. **It must never break a Python process.** Everything is wrapped; any exception
   is swallowed. A diagnostic that can break `pip` is worse than the bug.
2. **It must not import `magcalc`.** That would put the whole engine (numpy,
   sympy, ...) on every `python -c` startup. It replicates the resolution order
   instead -- scan `sys.path` the way `PathFinder` does; the editable-install
   finder loses to all of it, so it only wins when `sys.path` has no copy.
3. **It must be silent when nothing is wrong**, and cheap. `find_spec` is called
   for every import in the process, so its fast path is one string comparison.

`MAGCALC_SHADOW_GUARD=off` silences it, for the legitimate case of deliberately
working inside a second checkout (a git worktree, a version comparison).
"""
import os
import sys

_ENV = "MAGCALC_SHADOW_GUARD"
_OFF = ("0", "off", "no", "silent", "false")


def _real(path):
    try:
        return os.path.realpath(path or os.curdir)
    except (OSError, ValueError):
        return None


def path_roots():
    """`magcalc` packages reachable from `sys.path`, in RESOLUTION order.

    Mirrors what `PathFinder` will do, without importing anything: the first
    entry here is the copy `import magcalc` would actually get.
    """
    roots, seen = [], set()
    for entry in sys.path:
        base = _real(entry)
        if base is None or base in seen:
            continue
        seen.add(base)
        root = os.path.join(base, "magcalc")
        if os.path.isfile(os.path.join(root, "__init__.py")):
            roots.append(root)
    return roots


def editable_roots():
    """Roots declared by setuptools editable installs.

    Read from the already-imported `__editable___*_finder` modules -- their own
    `.pth` runs before this one (`_` sorts before `z`), so they are in
    `sys.modules` already and this costs nothing. These finders live on
    `sys.meta_path` and are APPENDED, i.e. after `PathFinder`, so a root found
    here wins only when `path_roots()` is empty.
    """
    roots = []
    for name, module in list(sys.modules.items()):
        if not name.startswith("__editable__"):
            continue
        mapping = getattr(module, "MAPPING", None)
        if isinstance(mapping, dict) and "magcalc" in mapping:
            root = _real(mapping.get("magcalc"))
            if root is not None and root not in roots:
                roots.append(root)
    return roots


def survey():
    """`(winner, others)` when several copies are importable, else `None`."""
    on_path = path_roots()
    editable = editable_roots()

    everything = list(on_path)
    for root in editable:
        if root not in everything:
            everything.append(root)

    if len(everything) < 2:
        return None

    # sys.path beats the editable finder; hence on_path first.
    winner = on_path[0] if on_path else editable[0]
    return winner, [r for r in everything if r != winner]


class MagcalcShadowWarning(RuntimeWarning):
    """Raised through `warnings` so pytest surfaces it in the warnings summary."""


def _warn(winner, others):
    summary = "%d importable copies of `magcalc`: WINS -> %s; also -> %s" % (
        len(others) + 1, winner, ", ".join(others),
    )

    lines = [
        "",
        "!!! magcalc shadow warning: %d importable copies of `magcalc` !!!" % (len(others) + 1),
        "    WINS -> %s" % winner,
    ]
    lines += ["    also -> %s" % o for o in others]
    lines += [
        "    Which one wins depends on the working directory and PYTHONPATH, so the",
        "    same command can give different answers from different places. Run",
        "    `magcalc where`. Silence with %s=off if this is deliberate." % _ENV,
        "",
    ]
    sys.stderr.write("\n".join(lines) + "\n")

    # stderr alone is not enough: under `pytest` the import happens during
    # collection, inside the capture, so the banner above is buffered and shown
    # only if something else fails -- and "ran the suite inside a stale checkout"
    # is one of the cases this exists for. A real warning lands in pytest's
    # warnings summary, which is printed on a fully passing run.
    try:
        import warnings

        warnings.warn(summary, MagcalcShadowWarning, stacklevel=2)
    except Exception:
        pass


def silenced():
    return os.environ.get(_ENV, "").strip().lower() in _OFF


class ShadowWatcher(object):
    """A `sys.meta_path` entry that observes, and never claims, `magcalc`.

    Parked at index 0 so it is consulted first, but `find_spec` always returns
    `None`, so resolution proceeds exactly as it would have. Its only job is to
    be present at the instant `magcalc` is imported -- the earliest moment at
    which `sys.path` is complete enough to answer "how many copies are there?"
    """

    fired = False

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        # Called for EVERY import in the process: keep the miss path to one
        # string comparison.
        if fullname != "magcalc" or cls.fired:
            return None
        cls.fired = True
        try:
            if not silenced():
                found = survey()
                if found:
                    _warn(*found)
        except Exception:
            pass
        return None


def install():
    """Idempotent: never stack duplicate watchers on repeated .pth processing."""
    for finder in sys.meta_path:
        if getattr(finder, "__name__", None) == "ShadowWatcher":
            return
    sys.meta_path.insert(0, ShadowWatcher)


try:
    install()
except Exception:
    # A startup hook must never take down the interpreter it is protecting.
    pass
