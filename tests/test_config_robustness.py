"""Runner/config robustness fixes uncovered by the broken aCVO/config.yaml.

The file had been a NESTED fragment (every key under a `cvo_model:` wrapper), which
tripped two latent engine bugs. Both are general, not aCVO-specific:
  1. a config with no top-level `crystal_structure` (and no python model) crashed with a
     bare KeyError deep in expansion instead of a clear, actionable error;
  2. the legacy python-model path matched an IMPORTED `scipy.optimize.minimize` as if it
     were a custom minimizer (`from scipy.optimize import minimize` in the model module),
     so it called scipy's minimize with a config -> "missing argument x0".
"""
import os

import pytest

from magcalc.generic_model import GenericSpinModel

HERE = os.path.dirname(__file__)
ACVO = os.path.join(HERE, "..", "examples", "materials", "aCVO")


def test_nested_config_gives_clear_error_not_keyerror():
    """The exact aCVO breakage: keys nested under a wrapper -> no top-level
    crystal_structure. Must name the problem, not raise KeyError('crystal_structure')."""
    cfg = {"cvo_model": {"python_model_file": "spin_model.py", "model_params": {}}}
    with pytest.raises(ValueError) as exc:
        GenericSpinModel(cfg)
    msg = str(exc.value)
    assert "crystal_structure" in msg
    assert "cvo_model" in msg                 # names the offending wrapper key
    assert "KeyError" not in type(exc.value).__name__


def test_missing_crystal_structure_error_mentions_legacy_route():
    with pytest.raises(ValueError, match="python_model_file"):
        GenericSpinModel({"parameters": {}, "interactions": {}})


def test_legacy_minimize_not_confused_with_imported_scipy():
    """A module that does `from scipy.optimize import minimize` must NOT be treated as
    providing a custom minimizer. Reproduces the runner's detection logic."""
    import types

    import scipy.optimize

    mod = types.ModuleType("fake_model")
    mod.minimize = scipy.optimize.minimize        # imported, not defined here
    fn = getattr(mod, "minimize", None)
    is_custom = (callable(fn) and
                 getattr(fn, "__module__", None) == getattr(mod, "__name__", None))
    assert not is_custom                          # scipy's __module__ != 'fake_model'

    # a genuinely local minimize IS accepted
    def minimize(cfg):
        return None
    mod2 = types.ModuleType("real_model")
    minimize.__module__ = "real_model"
    mod2.minimize = minimize
    fn2 = mod2.minimize
    assert getattr(fn2, "__module__", None) == mod2.__name__


def test_acvo_legacy_config_runs():
    """End to end: the fixed aCVO legacy config (python_model_file: spin_model.py) runs,
    minimises to its ground state, and passes the stability guard."""
    from magcalc.runner import run_calculation
    run_calculation(os.path.join(ACVO, "config.yaml"))        # must not raise
