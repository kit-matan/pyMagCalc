"""The interpreter-startup shadow guard: `tools/magcalc_shadow_guard.py`.

This is the half of the shadowing defence that `magcalc/provenance.py` cannot
cover. When a stale checkout wins OUTRIGHT, none of the in-package code runs --
that copy has no `provenance.py` -- so the only in-package tell is the *absence*
of a log line. The guard lives in site-packages, outside every `magcalc` copy,
and is triggered by a `.pth` at interpreter startup, so it reports regardless of
which copy wins.

Everything here drives the guard in SUBPROCESSES against throwaway decoy trees.
Nothing installs, modifies or removes anything on the developer's interpreter,
and the tests pass whether or not the guard is actually installed here.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1] / "tools" / "magcalc_shadow_guard.py"
LIVE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def guard_dir(tmp_path_factory):
    """A directory holding the guard module, for use via PYTHONPATH."""
    d = tmp_path_factory.mktemp("guard")
    (d / "magcalc_shadow_guard.py").write_text(GUARD.read_text())
    return d


def make_decoy(root: Path) -> Path:
    """A stale checkout: has `magcalc/`, but NOT `provenance.py`.

    The missing `provenance.py` is the point -- it is what makes the in-package
    detector structurally unable to report this case.
    """
    pkg = root / "magcalc"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("VERSION = 'stale'\n")
    assert not (pkg / "provenance.py").exists()
    return pkg


def run_probe(cwd, guard_dir, extra_path=None, env_extra=None, code="import magcalc"):
    """Import `magcalc` in a subprocess with the guard armed, return stderr."""
    pythonpath = [str(guard_dir)]
    if extra_path:
        pythonpath.append(str(extra_path))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(pythonpath))
    env.pop("MAGCALC_SHADOW_GUARD", None)
    env.update(env_extra or {})

    # `import magcalc_shadow_guard` stands in for the installed .pth line.
    probe = "import magcalc_shadow_guard\n" + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", probe], cwd=str(cwd),
        capture_output=True, text=True, env=env,
    )


def test_guard_reports_a_stale_copy_that_wins_outright(tmp_path, guard_dir):
    """THE case the in-package detector cannot reach.

    cwd is the decoy, so `sys.path[0]` resolves `magcalc` to it; the live tree is
    reachable only via the editable finder. The decoy has no `provenance.py`, so
    nothing in the package could possibly report -- but the guard does.
    """
    make_decoy(tmp_path)
    proc = run_probe(tmp_path, guard_dir, code="import magcalc; print(magcalc.__file__)")

    assert str(tmp_path / "magcalc") in proc.stdout, "decoy did not actually win"
    assert "magcalc shadow warning" in proc.stderr
    assert str(tmp_path / "magcalc") in proc.stderr
    assert str(LIVE_ROOT / "magcalc") in proc.stderr


def test_guard_is_silent_when_only_one_copy_exists(tmp_path, guard_dir):
    proc = run_probe(tmp_path, guard_dir, code="import magcalc")
    assert "shadow warning" not in proc.stderr, proc.stderr
    assert proc.returncode == 0


def test_guard_is_silent_when_magcalc_is_never_imported(tmp_path, guard_dir):
    """It observes an import; it must not editorialise about unrelated processes."""
    make_decoy(tmp_path)
    proc = run_probe(tmp_path, guard_dir, code="print('unrelated program')")
    assert "shadow warning" not in proc.stderr, proc.stderr


def test_env_var_silences_the_guard(tmp_path, guard_dir):
    """The legitimate case: deliberately working inside a second checkout."""
    make_decoy(tmp_path)
    proc = run_probe(tmp_path, guard_dir, env_extra={"MAGCALC_SHADOW_GUARD": "off"},
                     code="import magcalc")
    assert "shadow warning" not in proc.stderr, proc.stderr


@pytest.mark.slow
def test_a_startup_time_check_would_miss_cwd_shadowing(tmp_path):
    """Pins the TIMING TRAP the first version of this guard fell into.

    At real `.pth` execution time `sys.path[0]` is not yet the working directory
    -- for `python -c` the `''` entry is prepended AFTER site initialisation. An
    eager survey at startup is therefore blind to cwd shadowing, which is the
    main hazard; only the deferred `sys.meta_path` observer sees it.

    Proving that needs a genuine `.pth` in a genuine site directory (PYTHONPATH
    dirs are not scanned for `.pth`), so this builds a throwaway venv and installs
    TWO `.pth` files: the guard, and an eager probe that prints what a startup
    check would have concluded. If a future CPython populates `sys.path[0]`
    earlier, `EAGER=True` will appear here and the deferral can be simplified.
    """
    venv = tmp_path / "v"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip",
                    "--system-site-packages", str(venv)], check=True,
                   capture_output=True)
    site_packages = next(venv.glob("lib/python*/site-packages"))

    (site_packages / "magcalc_shadow_guard.py").write_text(GUARD.read_text())
    (site_packages / "zz_magcalc_shadow_guard.pth").write_text(
        "import magcalc_shadow_guard\n")

    # Runs at startup, exactly where an eager check would have lived.
    (site_packages / "eager_probe.py").write_text(textwrap.dedent("""
        import sys
        try:
            import magcalc_shadow_guard as g
            sys.stderr.write("EAGER=%s\\n" % (g.survey() is not None))
        except Exception as exc:
            sys.stderr.write("EAGER=error %s\\n" % exc)
    """))
    (site_packages / "zzz_eager_probe.pth").write_text("import eager_probe\n")

    stale = tmp_path / "stale"
    make_decoy(stale)

    env = dict(os.environ)
    env.pop("MAGCALC_SHADOW_GUARD", None)
    proc = subprocess.run(
        [str(venv / "bin" / "python"), "-c", "import magcalc; print(magcalc.__file__)"],
        cwd=str(stale), capture_output=True, text=True, env=env,
    )

    assert str(stale / "magcalc") in proc.stdout, "decoy did not win"
    # The trap: at startup, cwd is invisible, so the survey finds nothing.
    assert "EAGER=False" in proc.stderr, proc.stderr
    # The fix: by import time sys.path is complete, and the observer fires.
    assert "magcalc shadow warning" in proc.stderr, proc.stderr


def test_guard_never_breaks_the_interpreter(tmp_path, guard_dir):
    """A diagnostic that can break `pip` is worse than the bug it reports."""
    make_decoy(tmp_path)
    proc = run_probe(tmp_path, guard_dir, code="""
        import magcalc
        print('EXIT-OK')
    """)
    assert proc.returncode == 0
    assert "EXIT-OK" in proc.stdout
    assert "Traceback" not in proc.stderr


def test_installing_twice_does_not_stack_watchers(tmp_path, guard_dir):
    proc = run_probe(tmp_path, guard_dir, code="""
        import sys, magcalc_shadow_guard as g
        g.install(); g.install()
        n = sum(1 for f in sys.meta_path
                if getattr(f, '__name__', None) == 'ShadowWatcher')
        print('WATCHERS', n)
    """)
    assert "WATCHERS 1" in proc.stdout, proc.stdout


def test_provenance_reports_whether_the_guard_is_active():
    """`magcalc where` must state the protection level rather than imply one."""
    from magcalc import provenance

    assert isinstance(provenance.startup_guard_active(), bool)
    text = provenance.describe(verbose=True)
    assert "startup guard:" in text
