"""Optional compiled kernels for the beyond-LSWT loops (rsMagCalc, Rust via PyO3).

`calculation: {kernels: rust}` asks for them. The contract is the one
`calculation.backend: fortran` already has with fMagCalc, so the two read the
same way in a run log:

* absent package, or a package whose compiled core did not build
  (``rsmagcalc.backend != "pyo3"``) -> a WARNING, and the NumPy implementation
  runs. A run never fails because an optional accelerator is missing;
* a kernel that raises at run time -> the caller logs the exception and falls
  back to NumPy for that call (each kernel's call site does this; `resolve`
  only decides availability);
* the run log names the build (`kernels: rust (rsmagcalc 0.1.0, git abc, release)`)
  right after the engine line, so a wrong number can be attributed to the code
  that produced it. `magcalc where` prints the same line.

The NumPy paths are never removed: they are the fallback AND the parity reference
every kernel is pinned to (`tests/conftest.py`'s `kernels` fixture runs the oracle
tests under both). What exists, in what order, and what each kernel is pinned to:
`RUST_KERNELS_PLAN.md` at the workspace root.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["KERNELS", "resolve", "describe", "rust_module", "availability",
           "reset_cache"]

KERNELS = ("numpy", "rust")

# None = not probed yet; False = probed and unavailable; else the module.
_RUST = None


def rust_module():
    """The importable ``rsmagcalc`` with a compiled core, else None (cached)."""
    global _RUST
    if _RUST is None:
        try:
            import rsmagcalc
        except ImportError:
            _RUST = False
        else:
            _RUST = rsmagcalc if getattr(rsmagcalc, "backend", None) == "pyo3" else False
    return _RUST or None


def reset_cache() -> None:
    """Forget the probe result (tests that install or hide the package)."""
    global _RUST
    _RUST = None


def resolve(requested: Optional[str]) -> str:
    """The kernel set that will actually run for `calculation.kernels`.

    Returns "numpy" or "rust". An unknown name raises (a typo must not silently
    mean "numpy"); "rust" without an importable compiled core warns and returns
    "numpy".
    """
    name = "numpy" if requested is None else str(requested).strip().lower()
    if name not in KERNELS:
        raise ValueError(
            f"calculation.kernels must be one of {list(KERNELS)}, got {requested!r}."
        )
    if name == "rust" and rust_module() is None:
        logger.warning(
            "calculation.kernels: rust requested but the rsmagcalc package is not "
            "importable with a compiled core (install with `pip install "
            "/path/to/rsMagCalc`, or `pip install -e . --no-build-isolation` inside it). "
            "Falling back to the NumPy implementations for this run."
        )
        return "numpy"
    return name


def describe(kernel: str) -> str:
    """One line naming the kernel set in use, for the run log."""
    if kernel == "rust":
        mod = rust_module()
        if mod is None:  # resolve() would have downgraded; be honest anyway
            return "kernels: numpy (rust requested, rsmagcalc unavailable)"
        try:
            return f"kernels: rust {mod.describe()}"
        except Exception:  # an older package without describe()
            return f"kernels: rust (rsmagcalc {getattr(mod, '__version__', '?')})"
    return "kernels: numpy"


def availability() -> str:
    """For `magcalc where`: is the Rust kernel package importable, and which build."""
    mod = rust_module()
    if mod is None:
        return ("  rsmagcalc:   not installed (optional Rust kernels; "
                "`calculation: {kernels: rust}` would fall back to NumPy)")
    return f"  rsmagcalc:   available {mod.describe()}"
