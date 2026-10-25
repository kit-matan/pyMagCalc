"""`calculation.kernels` selects the optional Rust kernels (rsMagCalc) the way
`calculation.backend` selects fMagCalc: opt in, warn and fall back when absent, and
name the build in the run log so a number can be attributed to the code that made it.

M0 of RUST_KERNELS_PLAN.md: the selection, the fallback, the log line, and the
`kernels` fixture that later kernels' oracle tests will run under.
"""
import logging
import sys

import numpy as np
import pytest

from magcalc import kernels as K


@pytest.fixture(autouse=True)
def _fresh_probe():
    K.reset_cache()
    yield
    K.reset_cache()


def test_default_is_numpy():
    assert K.resolve(None) == "numpy"
    assert K.resolve("numpy") == "numpy"
    assert K.resolve(" NumPy ") == "numpy"
    assert K.describe("numpy") == "kernels: numpy"


def test_a_typo_is_an_error_not_a_silent_numpy_run():
    with pytest.raises(ValueError, match="calculation.kernels"):
        K.resolve("rsut")


def test_rust_falls_back_to_numpy_with_a_warning_when_not_installed(monkeypatch, caplog):
    # `sys.modules[name] = None` makes `import rsmagcalc` raise ImportError,
    # whether or not the package is really installed.
    monkeypatch.setitem(sys.modules, "rsmagcalc", None)
    K.reset_cache()
    with caplog.at_level(logging.WARNING, logger="magcalc.kernels"):
        assert K.resolve("rust") == "numpy"
    assert any("rsmagcalc" in r.getMessage() and "Falling back" in r.getMessage()
               for r in caplog.records)
    assert "unavailable" in K.describe("rust")


def test_a_package_without_its_compiled_core_counts_as_absent(monkeypatch):
    """A source checkout never built with maturin imports fine but has
    `backend = None`; that must read as 'not available', not as 'rust'."""
    import types

    shell = types.ModuleType("rsmagcalc")
    shell.backend = None
    monkeypatch.setitem(sys.modules, "rsmagcalc", shell)
    K.reset_cache()
    assert K.rust_module() is None
    assert K.resolve("rust") == "numpy"


def test_rust_is_selected_and_named_when_installed(kernels):
    """Runs under both fixture values; under 'rust' only when rsmagcalc is built."""
    if kernels == "numpy":
        assert K.describe(K.resolve("numpy")) == "kernels: numpy"
        return
    assert K.resolve("rust") == "rust"
    line = K.describe("rust")
    assert line.startswith("kernels: rust (rsmagcalc ")
    assert "\n" not in line


def test_where_reports_the_kernel_package():
    from typer.testing import CliRunner

    from magcalc.cli import app

    result = CliRunner().invoke(app, ["where"])
    assert "rsmagcalc:" in result.output, result.output
    assert ("available" in result.output) or ("not installed" in result.output)


def test_run_log_names_the_kernels_right_after_the_engine(tmp_path, caplog):
    """A `magcalc run` that asks for the kernels logs which set it actually used,
    so the record of a confusing run contains its own explanation (the same
    reason the engine line exists)."""
    import yaml

    from magcalc.runner import run_calculation

    cfg = {
        "crystal_structure": {
            "lattice_vectors": [[3.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
            "atoms_uc": [{"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0}],
        },
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "distance": 3.0, "value": -1.0}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
        "tasks": {"dispersion": True},
        "q_path": {"G": [0, 0, 0], "X": [0.5, 0, 0], "path": ["G", "X"],
                   "points_per_segment": 3},
        "plotting": {"save_plot": False, "show_plot": False, "plot_structure": False},
        "output": {"save_data": False},
        "calculation": {"cache_mode": "none", "kernels": "rust"},
    }
    p = tmp_path / "fm.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with caplog.at_level(logging.INFO):
        run_calculation(str(p))
    msgs = [r.getMessage() for r in caplog.records]
    engine = next(i for i, m in enumerate(msgs) if m.startswith("magcalc engine:"))
    kern = next(i for i, m in enumerate(msgs) if m.startswith("kernels: "))
    assert kern > engine
    expected = "kernels: rust" if K.rust_module() is not None else "kernels: numpy"
    assert msgs[kern].startswith(expected), msgs[kern]


def test_m0_kernel_is_pinned_to_the_numpy_energy():
    """The one kernel M0 ships: the classical energy, the quantity every later
    sampler kernel conserves or thermalizes. Named skip, like a missing fMagCalc."""
    mod = K.rust_module()
    if mod is None:
        pytest.skip("rsmagcalc not installed (optional Rust kernels)")
    from magcalc.classical_dynamics import energy

    rng = np.random.default_rng(0)
    n = 6
    A = rng.standard_normal((3 * n, 3 * n))
    H = 0.5 * (A + A.T)
    b = rng.standard_normal(3 * n)
    m = rng.standard_normal((n, 3))
    m *= 1.0 / np.linalg.norm(m, axis=1, keepdims=True)
    assert mod.classical_energy(H, b, m) == pytest.approx(energy(H, b, m), rel=1e-12)
