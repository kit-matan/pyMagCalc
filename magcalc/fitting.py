"""
Data fitting for pyMagCalc.

Fits a spin Hamiltonian to inelastic-neutron-scattering data of three kinds:

  * ``dispersion`` - magnon peak positions E(Q) (single crystal);
  * ``sqw``        - single-crystal dynamic structure factor I(Q, w);
  * ``powder``     - powder-averaged I(|Q|, w).

The engine keeps **one** :class:`magcalc.MagCalc` instance alive for the whole
minimization. For ``dispersion`` fits it compiles a
:class:`magcalc.core.DispersionEvaluator` (the symbolic Hamiltonian lambdified
once over ``(q, S, params)``), so each iteration is a pure numerical
evaluation with no per-call symbolic substitution or re-lambdification --- for
large magnetic cells this is orders of magnitude faster than calling
:meth:`MagCalc.calculate_dispersion` per iteration. Set ``fitting.fast: false``
to force the legacy per-iteration path. Intensity fits (``sqw``/``powder``)
use :meth:`MagCalc.update_hamiltonian_params` each iteration. lmfit drives the
optimization, giving bounds, fixed/varied parameters, expression constraints,
uncertainties and a fit report for free.

Parameter ordering follows the same rule as
``GenericSpinModel._resolve_param_map``: ``parameter_order`` if present, else the
keys of ``parameters`` excluding ``S``. The fit varies a subset of those names
(scalars only in v1) and reassembles the full ordered list each iteration.

Intensity fits (``sqw`` / ``powder``) additionally fit three nuisance
parameters - a global intensity ``scale``, a flat ``background`` and a Gaussian/
Lorentzian ``energy_broadening`` width - using the shared
:func:`magcalc.plotting.broaden_spectrum` line-shape so the fit and the plots
agree.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from magcalc.plotting import broaden_spectrum

logger = logging.getLogger(__name__)

# Tolerance (in RLU / inverse-angstrom for powder) for grouping data rows that
# share a Q-point, so S(Q,w) is computed once per distinct Q.
_Q_GROUP_DECIMALS = 5


# --------------------------------------------------------------------------- #
# Parameter ordering
# --------------------------------------------------------------------------- #
def canonical_name_order(final_config: Dict[str, Any]) -> List[str]:
    """
    Return the ordered list of Hamiltonian parameter names the model consumes.

    Mirrors ``GenericSpinModel._resolve_param_map``: use ``parameter_order`` if
    present, otherwise the keys of ``parameters`` excluding ``S``.
    """
    parameters = final_config.get("parameters")
    if parameters is None:
        parameters = final_config.get("model_params", {})
    parameters = parameters or {}

    order = final_config.get("parameter_order")
    if order:
        return [k for k in order if k in parameters and k != "S"]
    return [k for k in parameters if k != "S"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_fit_data(fit_type: str, path: str) -> Dict[str, np.ndarray]:
    """
    Load experimental data for a given ``fit_type`` into a normalized dict.

    dispersion: comma-delimited text (``#`` comments), columns
        ``h, k, l, E, sigma[, mode]``. ``mode`` (1-based band index) is optional.
        Returns ``{hkl (N,3), E (N,), sigma (N,), mode (N,) or None}``.

    sqw: comma-delimited text or ``.npz`` with columns/keys
        ``h, k, l, energy, intensity, error``.
        Returns ``{hkl (N,3), energy (N,), intensity (N,), error (N,)}``.

    powder: like ``sqw`` but with a single ``|Q|`` magnitude column instead of hkl.
        Returns ``{q (N,), energy (N,), intensity (N,), error (N,)}``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fit data file not found: {path}")

    if fit_type == "dispersion":
        arr = np.loadtxt(path, comments="#", delimiter=",", ndmin=2)
        if arr.shape[1] < 5:
            raise ValueError(
                f"Dispersion data '{path}' needs >=5 columns (h,k,l,E,sigma[,mode]); "
                f"got {arr.shape[1]}."
            )
        out = {
            "hkl": arr[:, 0:3].astype(float),
            "E": arr[:, 3].astype(float),
            "sigma": arr[:, 4].astype(float),
            "mode": arr[:, 5].astype(int) if arr.shape[1] >= 6 else None,
        }
        return out

    if fit_type in ("sqw", "powder"):
        if path.endswith(".npz"):
            d = np.load(path)
            energy = np.asarray(d["energy"], dtype=float).ravel()
            intensity = np.asarray(d["intensity"], dtype=float).ravel()
            error = np.asarray(d["error"], dtype=float).ravel() if "error" in d \
                else np.ones_like(intensity)
            if fit_type == "sqw":
                hkl = np.asarray(d["hkl"], dtype=float).reshape(-1, 3)
                return {"hkl": hkl, "energy": energy, "intensity": intensity, "error": error}
            q = np.asarray(d["q"], dtype=float).ravel()
            return {"q": q, "energy": energy, "intensity": intensity, "error": error}

        arr = np.loadtxt(path, comments="#", delimiter=",", ndmin=2)
        if fit_type == "sqw":
            if arr.shape[1] < 6:
                raise ValueError(
                    f"sqw data '{path}' needs >=6 columns (h,k,l,energy,intensity,error)."
                )
            return {
                "hkl": arr[:, 0:3].astype(float),
                "energy": arr[:, 3].astype(float),
                "intensity": arr[:, 4].astype(float),
                "error": arr[:, 5].astype(float),
            }
        # powder
        if arr.shape[1] < 4:
            raise ValueError(
                f"powder data '{path}' needs >=4 columns (|Q|,energy,intensity,error)."
            )
        return {
            "q": arr[:, 0].astype(float),
            "energy": arr[:, 1].astype(float),
            "intensity": arr[:, 2].astype(float),
            "error": arr[:, 3].astype(float),
        }

    raise ValueError(f"Unknown fit type '{fit_type}'. Expected dispersion|sqw|powder.")


def _group_by_q(keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group rows that share a Q-point.

    Args:
        keys: (N, D) array of Q identifiers (hkl for sqw, |Q| reshaped to (N,1) for powder).

    Returns:
        (unique_keys (U, D), inverse (N,)) where ``unique_keys[inverse]`` reconstructs ``keys``.
    """
    rounded = np.round(keys, _Q_GROUP_DECIMALS)
    unique_keys, inverse = np.unique(rounded, axis=0, return_inverse=True)
    return unique_keys, inverse.ravel()


# --------------------------------------------------------------------------- #
# The fit problem
# --------------------------------------------------------------------------- #
class FitProblem:
    """
    Holds the calculator, data and parameter mapping for a single fit, and
    exposes :meth:`residual` for lmfit.
    """

    def __init__(
        self,
        fit_type: str,
        calculator,
        data: Dict[str, np.ndarray],
        name_order: List[str],
        base_values: Dict[str, Any],
        B_matrix: np.ndarray,
        match: str = "nearest",
        lineshape: str = "lorentzian",
        backend: str = "numpy",
        fast: bool = True,
        temperature: Optional[float] = None,
        domains=None,
        cross_section: str = "perp",
    ):
        self.fit_type = fit_type
        self.calc = calculator
        self.data = data
        self.name_order = name_order
        self.base_values = base_values
        self.B = B_matrix
        self.match = match
        self.lineshape = lineshape
        self.backend = backend
        # Measurement model -- the SAME options `magcalc run` applies (see
        # CLAUDE.md 'Intensity / experiment layer'). Fitting used to ignore these
        # entirely, so every intensity residual was computed at T = 0 for a single
        # domain: a systematic bias against any real (finite-T, twinned) data set.
        self.temperature = temperature
        self.domains = domains
        self.cross_section = cross_section
        self._neval = 0
        # Cache of the expensive forward-model output (mode energies/intensities)
        # keyed on the Hamiltonian parameter vector. Lets nuisance-only steps
        # (scale/background/broadening) skip the eigensolve entirely.
        self._cache_key = None
        self._cache_modes = None

        # Pre-compute Q-grouping and cartesian Q-vectors for intensity fits.
        if fit_type == "sqw":
            self._uniq_q, self._inv = _group_by_q(data["hkl"])
            self._q_cart = np.dot(self._uniq_q, self.B)
        elif fit_type == "powder":
            self._uniq_q, self._inv = _group_by_q(data["q"].reshape(-1, 1))
            self._q_mags = self._uniq_q.ravel()
        elif fit_type == "dispersion":
            self._q_cart = np.dot(data["hkl"], self.B)

        # Fast path for dispersion fits: compile the symbolic Hamiltonian once
        # over (q, S, params); each iteration is then a pure numerical
        # evaluation (~ms per q-point) instead of a per-call re-lambdify
        # (which can cost tens of seconds for large cells).
        self._fast_eval = None
        if fit_type == "dispersion" and fast:
            try:
                self._fast_eval = calculator.compile_dispersion_evaluator()
            except Exception:
                logger.warning(
                    "Fast dispersion evaluator unavailable; falling back to "
                    "calculate_dispersion per iteration.", exc_info=True
                )

    # -- parameter assembly -------------------------------------------------- #
    def _assemble_params(self, lmfit_params) -> List[Any]:
        """Build the full ordered Hamiltonian parameter list for this iteration."""
        p = []
        for name in self.name_order:
            if name in lmfit_params:
                p.append(float(lmfit_params[name].value))
            else:
                v = self.base_values[name]
                if isinstance(v, (list, tuple, np.ndarray)):
                    p.append([float(x) for x in v])
                else:
                    p.append(float(v))
        return p

    def _model_modes(self, p):
        """
        Return the forward-model mode output for parameter vector ``p``, calling
        the (expensive) eigensolve only when ``p`` differs from the cached vector.

        dispersion -> energies (N, nmodes).
        sqw/powder -> (energies, intensities) per unique Q (U, nmodes).
        """
        key = tuple(
            tuple(v) if isinstance(v, (list, tuple, np.ndarray)) else v for v in p
        )
        if key == self._cache_key and self._cache_modes is not None:
            return self._cache_modes

        if self.fit_type == "dispersion" and self._fast_eval is not None:
            modes = self._fast_eval.energies(self._q_cart, p)
            self._cache_key = key
            self._cache_modes = modes
            return modes

        self.calc.update_hamiltonian_params(p)
        if self.fit_type == "dispersion":
            res = self.calc.calculate_dispersion(self._q_cart, serial=True, backend=self.backend)
            modes = np.real(np.asarray(res.energies))
        elif self.fit_type == "powder":
            # Sample-resolved modes: each sphere direction keeps its own energies, so
            # the broadened model reproduces the true powder lineshape. The legacy
            # sphere-averaged representation collapsed a dispersive band to its
            # center, silently biasing powder fits (caught on Cu5SbO6, whose 10 meV
            # band became a ~1 meV blob -- see PRR 8, 013247 Fig. 5).
            from .numerical import powder_sample_modes
            E_smp, I_smp = powder_sample_modes(
                self.calc, self._q_mags, num_samples=self._powder_samples,
                backend=self.backend, temperature=self.temperature,
                cross_section=self.cross_section,
            )
            modes = (E_smp, I_smp)
        else:  # sqw
            res = self.calc.calculate_sqw(
                self._q_cart, backend=self.backend,
                temperature=self.temperature, domains=self.domains,
                cross_section=self.cross_section,
            )
            modes = (np.asarray(res.energies), np.asarray(res.intensities))

        self._cache_key = key
        self._cache_modes = modes
        return modes

    # -- residual ------------------------------------------------------------ #
    def residual(self, lmfit_params) -> np.ndarray:
        self._neval += 1
        p = self._assemble_params(lmfit_params)

        if self.fit_type == "dispersion":
            return self._residual_dispersion(self._model_modes(p))
        return self._residual_intensity(lmfit_params, self._model_modes(p))

    def _residual_dispersion(self, energies) -> np.ndarray:
        E_data = self.data["E"]
        sigma = self.data["sigma"]
        mode = self.data["mode"]

        model_E = np.empty_like(E_data)
        for i in range(len(E_data)):
            bands = energies[i]
            if self.match == "mode" and mode is not None:
                idx = int(mode[i]) - 1
                idx = min(max(idx, 0), len(bands) - 1)
                model_E[i] = bands[idx]
            else:  # nearest-band assignment
                model_E[i] = bands[np.argmin(np.abs(bands - E_data[i]))]

        return (model_E - E_data) / sigma

    # -- prediction (for plotting the data/model comparison) ----------------- #
    def predict(self, lmfit_params) -> Dict[str, np.ndarray]:
        """
        Return the best-fit model prediction aligned with the data, for plotting.

        dispersion -> {"x", "E_data", "sigma", "E_model"} (x = data-row index).
        sqw/powder -> {"x", "energy", "I_data", "I_model"} where x is the path
            index (sqw) or |Q| (powder) of each data row.
        """
        p = self._assemble_params(lmfit_params)
        modes = self._model_modes(p)

        if self.fit_type == "dispersion":
            energies = modes
            E_data = self.data["E"]
            mode = self.data["mode"]
            E_model = np.empty_like(E_data)
            for i in range(len(E_data)):
                bands = energies[i]
                if self.match == "mode" and mode is not None:
                    idx = min(max(int(mode[i]) - 1, 0), len(bands) - 1)
                    E_model[i] = bands[idx]
                else:
                    E_model[i] = bands[np.argmin(np.abs(bands - E_data[i]))]
            return {
                "x": np.arange(len(E_data)),
                "E_data": E_data,
                "sigma": self.data["sigma"],
                "E_model": E_model,
            }

        # intensity fits
        if self.fit_type == "powder":
            x_per_row = self.data["q"]
        else:
            x_per_row = self._inv.astype(float)  # path index of the unique Q

        scale = float(lmfit_params["scale"].value)
        background = float(lmfit_params["background"].value)
        width = float(lmfit_params["energy_broadening"].value)
        energies, intensities = modes
        e_data = self.data["energy"]
        I_model = np.empty_like(self.data["intensity"])
        for u in range(energies.shape[0]):
            sel = np.where(self._inv == u)[0]
            if sel.size == 0:
                continue
            spec = broaden_spectrum(
                energies[u], intensities[u], e_data[sel], width=width, kind=self.lineshape
            )
            I_model[sel] = scale * spec + background
        return {
            "x": x_per_row,
            "energy": e_data,
            "I_data": self.data["intensity"],
            "I_model": I_model,
        }

    def _residual_intensity(self, lmfit_params, modes) -> np.ndarray:
        scale = float(lmfit_params["scale"].value)
        background = float(lmfit_params["background"].value)
        width = float(lmfit_params["energy_broadening"].value)

        energies, intensities = modes  # (U, nmodes) each

        e_data = self.data["energy"]
        i_data = self.data["intensity"]
        err = self.data["error"]
        inv = self._inv

        model_I = np.empty_like(i_data)
        for u in range(energies.shape[0]):
            sel = np.where(inv == u)[0]
            if sel.size == 0:
                continue
            spec = broaden_spectrum(
                energies[u], intensities[u], e_data[sel], width=width, kind=self.lineshape
            )
            model_I[sel] = scale * spec + background

        return (model_I - i_data) / err


# --------------------------------------------------------------------------- #
# lmfit parameter construction
# --------------------------------------------------------------------------- #
def build_lmfit_params(fit_cfg: Dict[str, Any], parameters: Dict[str, Any],
                       name_order: List[str]):
    """
    Build the ``lmfit.Parameters`` object for the fit.

    Varies the names listed in ``fitting.vary`` (must be scalar entries of
    ``parameters``); applies ``fitting.bounds`` (``{name: [min, max]}``) and
    ``fitting.expr`` (``{name: "expression"}``). For sqw/powder fits, adds the
    ``scale``, ``background`` and ``energy_broadening`` nuisance parameters.
    """
    import lmfit

    params = lmfit.Parameters()
    vary = fit_cfg.get("vary", [])
    bounds = fit_cfg.get("bounds", {}) or {}
    exprs = fit_cfg.get("expr", {}) or {}

    for name in vary:
        if name not in name_order:
            raise ValueError(
                f"fitting.vary references '{name}', which is not a model parameter "
                f"name. Known names: {name_order}."
            )
        base = parameters.get(name)
        if isinstance(base, (list, tuple, np.ndarray)):
            raise ValueError(
                f"fitting.vary references vector parameter '{name}'. v1 only varies "
                f"scalar parameters; keep vectors fixed."
            )
        lo, hi = (bounds.get(name) or [-np.inf, np.inf])
        params.add(name, value=float(base), min=float(lo), max=float(hi))
        if name in exprs:
            params[name].set(expr=exprs[name])

    fit_type = fit_cfg.get("type", "dispersion")
    if fit_type in ("sqw", "powder"):
        def _nuis(key, default_value, default_vary):
            spec = fit_cfg.get(key, {}) or {}
            return (
                float(spec.get("value", default_value)),
                bool(spec.get("vary", default_vary)),
                spec.get("min"),
                spec.get("max"),
            )

        s_val, s_vary, s_min, s_max = _nuis("scale", 1.0, True)
        b_val, b_vary, b_min, b_max = _nuis("background", 0.0, True)
        w_val, w_vary, w_min, w_max = _nuis("energy_broadening", 0.3, False)
        params.add("scale", value=s_val, vary=s_vary,
                   min=0.0 if s_min is None else float(s_min),
                   max=np.inf if s_max is None else float(s_max))
        params.add("background", value=b_val, vary=b_vary,
                   min=-np.inf if b_min is None else float(b_min),
                   max=np.inf if b_max is None else float(b_max))
        params.add("energy_broadening", value=w_val, vary=w_vary,
                   min=1e-3 if w_min is None else float(w_min),
                   max=np.inf if w_max is None else float(w_max))

    return params


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_fit(
    final_config: Dict[str, Any],
    calculator,
    name_order: List[str],
    B_matrix: np.ndarray,
    config_dir: str = ".",
    backend: str = "numpy",
) -> Dict[str, Any]:
    """
    Run a data fit defined by ``final_config['fitting']`` using ``calculator``.

    Returns a dict with keys:
        ``result``        - the lmfit ``MinimizerResult``;
        ``report``        - ``lmfit.fit_report`` string;
        ``best_values``   - {param_name: value} for the varied model params;
        ``best_p``        - full ordered Hamiltonian parameter list at the optimum;
        ``problem``       - the :class:`FitProblem` (for plotting the comparison).
    """
    import lmfit

    fit_cfg = final_config.get("fitting", {}) or {}
    fit_type = fit_cfg.get("type", "dispersion")
    method = fit_cfg.get("method", "leastsq")
    lineshape = fit_cfg.get("lineshape", "lorentzian")
    match = fit_cfg.get("match", "nearest")
    fast = bool(fit_cfg.get("fast", True))

    data_file = fit_cfg.get("data_file")
    if not data_file:
        raise ValueError("fitting.data_file is required.")
    if not os.path.isabs(data_file):
        data_file = os.path.join(config_dir, data_file)

    logger.info(f"Loading {fit_type} fit data from {data_file}")
    data = load_fit_data(fit_type, data_file)

    parameters = final_config.get("parameters") or final_config.get("model_params", {}) or {}
    base_values = {k: parameters[k] for k in name_order}

    # The measurement model comes from the SAME `calculation:` block that `magcalc run`
    # uses, so a fit and a forward run of the same config model the experiment
    # identically. `fitting:` may override any of them locally.
    calc_cfg = final_config.get("calculation", {}) or {}
    temperature = fit_cfg.get("temperature", calc_cfg.get("temperature"))
    domains = fit_cfg.get("domains", calc_cfg.get("domains"))
    cross_section = fit_cfg.get(
        "cross_section", calc_cfg.get("cross_section", "perp"))
    if fit_type in ("sqw", "powder"):
        logger.info(
            f"Fit measurement model: temperature={temperature}, "
            f"domains={'yes' if domains else 'none'}, cross_section={cross_section}.")

    problem = FitProblem(
        fit_type=fit_type,
        calculator=calculator,
        data=data,
        name_order=name_order,
        base_values=base_values,
        B_matrix=B_matrix,
        match=match,
        lineshape=lineshape,
        backend=backend,
        fast=fast,
        temperature=temperature,
        domains=domains,
        cross_section=cross_section,
    )
    # Number of sphere samples for powder evaluations during the fit.
    problem._powder_samples = int(
        (final_config.get("powder_average", {}) or {}).get("num_samples", 50)
    )

    params = build_lmfit_params(fit_cfg, parameters, name_order)
    varied = [n for n in params if params[n].vary]
    logger.info(f"Fitting ({fit_type}, method={method}) varying: {varied}")

    # Extra keyword args forwarded to lmfit.minimize (e.g. differential_evolution
    # bounds, tolerances). For the default Levenberg-Marquardt (`leastsq`), the
    # forward model is comparatively expensive and its output precision is
    # limited (eigensolve + energy broadening), so the default ~1e-8 relative
    # finite-difference step can read a flat gradient. Use a larger `epsfcn` so
    # the Jacobian probes a meaningful parameter change.
    fit_kws = dict(fit_cfg.get("fit_kws", {}) or {})
    if method == "leastsq":
        fit_kws.setdefault("epsfcn", 1e-4)

    result = lmfit.minimize(problem.residual, params, method=method,
                            nan_policy="omit", **fit_kws)

    report = lmfit.fit_report(result)
    logger.info("Fit complete.\n" + report)

    # Cast to plain Python floats so downstream YAML/JSON serialization is clean.
    best_values = {n: float(result.params[n].value)
                   for n in name_order if n in result.params}
    best_p = problem._assemble_params(result.params)

    # Leave the shared calculator synchronized with the optimum (the fast
    # dispersion path never touches the calculator's parameters during the
    # fit, and even the slow path's last evaluation need not be the optimum).
    try:
        calculator.update_hamiltonian_params(best_p)
    except Exception:
        logger.warning("Could not synchronize calculator with best-fit parameters.",
                       exc_info=True)

    return {
        "result": result,
        "report": report,
        "best_values": best_values,
        "best_p": best_p,
        "problem": problem,
    }
