"""Install/remove the startup shadow guard. Importable so the `magcalc guard`
CLI works for any install; `tools/install_shadow_guard.py` is a thin wrapper
for use without the CLI.
"""
import argparse
import os
import shutil
import sys
import sysconfig

# The guard SOURCE ships inside the package (`magcalc/_shadow_guard.py`), not
# next to this script: `tools/` is not installed by a non-editable `pip install`,
# so a wheel user had no way to install the guard at all. magcalc never imports
# that module -- it is a template copied out to site-packages, where it must live
# OUTSIDE every magcalc copy to do its job.
import magcalc
SOURCE = os.path.join(os.path.dirname(os.path.abspath(magcalc.__file__)),
                      "_shadow_guard.py")

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


def install_quietly():
    """The write itself, with no output. Used by the CLI's first-run arming
    (`magcalc.cli._arm_shadow_guard`), which prints its own one-line notice."""
    module, pth = paths()
    shutil.copyfile(SOURCE, module)
    with open(pth, "w") as handle:
        handle.write(PTH_LINE)
    return module, pth


def install():
    module, pth = install_quietly()
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
