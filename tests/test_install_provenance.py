"""The `magcalc` being tested must be the one in THIS checkout.

Background and the exact shadowing mechanism are in `magcalc/provenance.py`. The
short version: `pip install -e .` appends its finder to `sys.meta_path`, i.e.
AFTER the built-in `PathFinder`, so anything on `sys.path` -- including `sys.path[0]`,
the script dir or cwd -- beats it. Two stale full copies of this tree have lived on
this machine under OneDrive; working inside one silently swaps the engine.

These tests make that a named failure in the FAST suite instead of a lost
afternoon. They are deliberately unmarked (no `slow`), and derive the expected
root from their OWN location rather than any constant: whichever checkout
collected this file is the checkout whose `magcalc/` must be imported.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import magcalc
from magcalc import provenance

# tests/ -> the pyMagCalc checkout that owns this file.
CHECKOUT = Path(__file__).resolve().parents[1]


def test_magcalc_is_imported_from_this_checkout():
    """Catches the MIXED case: this checkout's tests against a different engine.

    Under pytest's default `prepend` import mode the checkout root goes on
    `sys.path[0]`, so in the ordinary case this is cheap. It earns its place on
    the mismatch: a non-editable `magcalc` winning from site-packages, an
    `--import-mode=importlib` run, or these tests executed from a tree whose
    package is not the one that answers `import magcalc`.
    """
    imported = Path(magcalc.__file__).resolve().parent
    expected = (CHECKOUT / "magcalc").resolve()
    assert imported == expected, (
        f"`import magcalc` resolved to {imported}, but these tests live in "
        f"{CHECKOUT}, whose package is {expected}.\n"
        "You are testing a DIFFERENT copy of the engine than the one you are "
        "editing. Usual causes: a stale tree earlier on sys.path (working from "
        "inside a cloud-synced copy), or `pip install -e .` last run from that "
        "copy. Fix with `pip install -e .` from this checkout, and see "
        "`magcalc where`.\n"
        f"sys.path[0] = {sys.path[0]!r}"
    )


def test_no_second_magcalc_package_is_importable():
    """Even a shadowed-but-unimported duplicate is worth failing on.

    `magcalc.__file__` only reports the winner. If a second `magcalc/` is
    reachable, which one wins depends on the working directory and `PYTHONPATH`
    -- so the suite passes here and the same command fails from another cwd.
    """
    roots = provenance.importable_roots()
    assert len(roots) <= 1, (
        "More than one `magcalc` package is importable: "
        + ", ".join(str(p) for p in roots)
        + ".\nWhich one wins depends on the working directory, so results are "
        "not reproducible. Remove or rename the extra copy (`magcalc where`)."
    )


def test_a_cwd_copy_really_does_shadow_the_editable_install(tmp_path):
    """The hazard is real, so the detector must be pinned against it.

    This asserts the *mechanism*, not just the detector: a bare `magcalc/` in the
    working directory wins over the editable install. If a future packaging change
    ever makes the editable finder take precedence (putting it ahead of
    `PathFinder` on `sys.meta_path`), this test fails and the elaborate warnings
    around it can be retired -- which is the only way anyone would find out.
    """
    decoy = tmp_path / "magcalc"
    decoy.mkdir()
    (decoy / "__init__.py").write_text("SENTINEL = 'stale-copy'\n")

    probe = textwrap.dedent(
        """
        import magcalc
        print(getattr(magcalc, 'SENTINEL', 'live-engine'))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert out == "stale-copy", (
        "A `magcalc/` package in the working directory no longer shadows the "
        f"installed one (got {out!r}). If that is a deliberate packaging change, "
        "the shadowing warnings in magcalc/provenance.py, OPEN_WORK.md and "
        "CLAUDE.md are now obsolete and should be removed."
    )


def test_detector_fires_on_a_shadowing_copy(tmp_path, monkeypatch):
    """The detector must fire on the situation above, not merely exist."""
    decoy = tmp_path / "magcalc"
    decoy.mkdir()
    (decoy / "__init__.py").write_text("")

    monkeypatch.syspath_prepend(str(tmp_path))
    roots = provenance.importable_roots()

    assert decoy.resolve() in roots
    assert len(roots) > 1
    assert "WARNING" in provenance.describe()


def test_detector_does_not_need_the_live_tree_on_sys_path(tmp_path, monkeypatch):
    """The blind spot a `sys.path`-only scan gets exactly backwards.

    In the worst case -- a stale copy winning via `PYTHONPATH` -- the LIVE tree is
    reachable only through the editable finder on `sys.meta_path`, and is absent
    from `sys.path` entirely. A scan of `sys.path` alone then finds exactly one
    copy and reports all clear at the moment it should be shouting. That was the
    first version of this module, and it was wrong.

    Simulated here by emptying `sys.path` of the live tree, leaving only a decoy.
    """
    if not provenance._editable_install_roots():
        pytest.skip("magcalc is not installed editable; no second source to union")

    decoy = tmp_path / "magcalc"
    decoy.mkdir()
    (decoy / "__init__.py").write_text("")

    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    roots = provenance.importable_roots()

    assert provenance.package_root() not in provenance._syspath_roots()
    assert decoy.resolve() in roots
    assert len(roots) >= 2, "sys.path-only blind spot is back"
