#!/usr/bin/env python3
"""Install / remove the interpreter-startup shadow guard.

    python tools/install_shadow_guard.py [--status | --uninstall]

Copies `magcalc_shadow_guard.py` into the ACTIVE interpreter's site-packages and
writes a one-line `zz_magcalc_shadow_guard.pth` next to it. `site.py` executes
`.pth` import lines at startup, so from then on every Python process on this
interpreter checks for duplicate `magcalc` copies before any user code runs.

Two details that matter:

* **It is per-interpreter, not per-repo.** A new venv starts unprotected; re-run
  this there. `magcalc where` reports whether the guard is active, so the
  protection level is never a guess.
* **The `.pth` only ARMS the guard; it does not check anything.** The survey is
  deferred to the moment `magcalc` is imported (see the guard's module docstring
  for why a startup-time check is blind to cwd shadowing). So `.pth` ordering
  against `__editable__.magcalc-*.pth` is NOT load-bearing -- by import time the
  editable finder is in `sys.modules` regardless. The `zz_` prefix is kept only
  so the guard arms after any finder a `.pth` might install.

Uninstall removes exactly the two files it wrote, and nothing else.
"""
import argparse
import os
import shutil
import sys
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "magcalc_shadow_guard.py")

MODULE_NAME = "magcalc_shadow_guard.py"
PTH_NAME = "zz_magcalc_shadow_guard.pth"
PTH_LINE = "import magcalc_shadow_guard\n"


def site_packages():
    target = sysconfig.get_path("purelib")
    if not target or not os.path.isdir(target):
        raise SystemExit("cannot locate site-packages for %s" % sys.executable)
    return target


def paths():
    target = site_packages()
    return os.path.join(target, MODULE_NAME), os.path.join(target, PTH_NAME)


def status():
    module, pth = paths()
    active = os.path.isfile(module) and os.path.isfile(pth)
    print("interpreter:   %s" % sys.executable)
    print("site-packages: %s" % site_packages())
    print("guard module:  %s" % ("present" if os.path.isfile(module) else "ABSENT"))
    print("guard .pth:    %s" % ("present" if os.path.isfile(pth) else "ABSENT"))
    print("status:        %s" % ("ACTIVE" if active else "not installed"))
    return 0 if active else 1


def install():
    module, pth = paths()
    shutil.copyfile(SOURCE, module)
    with open(pth, "w") as handle:
        handle.write(PTH_LINE)
    print("installed %s" % module)
    print("installed %s" % pth)
    print("\nEvery Python process on this interpreter now warns when more than one")
    print("`magcalc` is importable. Silence with MAGCALC_SHADOW_GUARD=off.")
    return 0


def uninstall():
    removed = 0
    for path in paths():
        if os.path.isfile(path):
            os.remove(path)
            print("removed %s" % path)
            removed += 1
    if not removed:
        print("nothing to remove")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="report whether the guard is active")
    group.add_argument("--uninstall", action="store_true", help="remove the guard")
    args = parser.parse_args()

    if args.status:
        return status()
    if args.uninstall:
        return uninstall()
    return install()


if __name__ == "__main__":
    raise SystemExit(main())
