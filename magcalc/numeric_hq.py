"""The dipole engine's H(q) = 2 g H2(q), built numerically (K8).

`calculation.hamiltonian_builder` chooses how the LSWT matrix is built:

* ``symbolic`` -- `symbolic.gen_HM`: the SymPy Holstein-Primakoff expansion,
  Fourier transform and matrix extraction, then `lambdify`. 5-45 s for a cold
  16-site model, plus a `lambdify` in every pool worker. It is the reference this
  module is pinned to, and the only route for a custom Python spin model;
* ``numeric`` -- this module plus rsMagCalc's `DipoleHq`. The terms are resolved
  in Python by the spin model's OWN builders (`GenericSpinModel.classical_parts`,
  the same calls `Hamiltonian` makes), called once with commutative symbols for
  the spin components, so the meaning of every term stays in one place. That gives
  a polynomial in the components with coefficients in (S, parameters). rsMagCalc
  then does what `gen_HM` does symbolically -- the HP expansion truncated at boson
  degree 2, the Fourier placement, 2 g -- for each (S, parameters), in
  milliseconds, as a sparse bond sum H(q) = sum_d M_d exp(i q.d);
* ``auto`` (the default) -- numeric when rsMagCalc is installed and the model can
  be resolved this way, otherwise symbolic, with the reason in the run log.

The two builders agree entry by entry to rounding (`tests/test_numeric_hq.py`, on
every buildable shipped dipole config); every engine test that touches H(q) runs
under both through the `builder` fixture. `MAGCALC_HAMILTONIAN_BUILDER` in the
environment overrides the config (the test fixture uses it).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

logger = logging.getLogger(__name__)

BUILDERS = ("auto", "numeric", "symbolic")
ENV = "MAGCALC_HAMILTONIAN_BUILDER"


class NumericUnavailable(Exception):
    """The numeric builder cannot represent this model; the message says why."""


def requested_builder(config: Optional[dict]) -> str:
    """`calculation.hamiltonian_builder` (or $MAGCALC_HAMILTONIAN_BUILDER), checked."""
    raw = os.environ.get(ENV)
    if not raw:
        calc = config.get("calculation") if isinstance(config, dict) else None
        raw = calc.get("hamiltonian_builder", "auto") if isinstance(calc, dict) else "auto"
    name = str(raw).strip().lower()
    if name not in BUILDERS:
        raise ValueError(
            f"calculation.hamiltonian_builder must be one of {list(BUILDERS)}, got {raw!r}.")
    return name


def rust_dipole_hq():
    """rsmagcalc's `DipoleHq` when an rsMagCalc with K8 is installed, else None."""
    from .kernels import rust_module
    rs = rust_module()
    return getattr(rs, "DipoleHq", None) if rs is not None else None


def _params_key(S: float, params: Sequence[float]) -> Tuple[float, ...]:
    return (float(S),) + tuple(float(p) for p in params)


class HamiltonianTerms:
    """A dipole model's H(q) as monomial tables, ready to assemble for any (S, p).

    Built once per `MagCalc`. `model(S, params)` returns rsMagCalc's `DipoleHq`
    for those values (the last one is kept); `rotations(S, params)` the local-frame
    rotations that `gen_HM` bakes in. Raises `NumericUnavailable` when the model
    cannot be resolved this way (the caller then uses `gen_HM`).
    """

    def __init__(self, sm, S_sym: sp.Symbol, params_sym: list, params_sym_flat: list):
        from .generic_model import GenericSpinModel

        t0 = time.perf_counter()
        if not isinstance(sm, GenericSpinModel):
            raise NumericUnavailable(
                "the spin model is a custom Python module, not a config-driven "
                "GenericSpinModel (only gen_HM can expand an arbitrary Hamiltonian)")
        self._DipoleHq = rust_dipole_hq()
        if self._DipoleHq is None:
            raise NumericUnavailable(
                "rsmagcalc with the K8 builder (DipoleHq) is not installed")
        # Same refusal gen_HM makes, before anything else.
        sm.reject_dipole_unsupported()

        pos_uc = np.asarray(sm.atom_pos(), dtype=float)
        pos_ouc = np.asarray(sm.atom_pos_ouc(), dtype=float)
        n, n_ouc = len(pos_uc), len(pos_ouc)
        # gen_HM pairs cell site i with OUC site j through r_uc[i] - r_ouc[j] and
        # rotates OUC site j with cell site j % n: both assume this layout.
        if n == 0 or n_ouc % n or not np.array_equal(pos_ouc[:n], pos_uc):
            raise NumericUnavailable(
                "atom_pos_ouc() does not start with the cell sites in order")
        self.n, self.n_ouc, self.pos_ouc = n, n_ouc, np.ascontiguousarray(pos_ouc)

        self.args = [S_sym] + list(params_sym_flat)
        allowed = set(self.args)

        def subst(expr):
            # Exactly the substitution gen_HM's Hamiltonian applies to its total:
            # config parameters left out of parameter_order become numbers.
            expr = sp.sympify(expr)
            if expr.free_symbols:
                expr = sm._apply_substitution_and_filter(expr, params_sym)
            return expr

        # S itself is replaced by a number when the config's `parameters` carries
        # an S that parameter_order does not list; HP then uses that number.
        probe = sp.Symbol("_hq_probe")
        s_bound = subst(S_sym * probe) / probe
        self.S_fixed = None if s_bound.free_symbols else float(s_bound)

        # Per-site spin lengths S_i = ratio_i * S, the ratios as gen_HM forms them.
        self.ratios = np.ones(n)
        mags = getattr(sm, "spin_magnitudes", lambda: [])()
        if mags and len(mags) == n:
            s_ref = float(mags[0])
            if s_ref > 0 and any(abs(float(m) - s_ref) > 1e-12 for m in mags):
                self.ratios = np.array(
                    [1.0 if float(m) / s_ref == 1.0 else float(sp.nsimplify(float(m) / s_ref))
                     for m in mags])

        # The terms, from the model's own builders, with commutative components.
        comps = [[sp.Symbol(f"_hqS{j}_{a}") for a in range(3)] for j in range(n_ouc)]
        Svec = [sp.Matrix(c) for c in comps]
        Jex, DM, Kex, rest = sm.classical_parts(Svec, params_sym)

        exprs: List[sp.Expr] = []
        factors: List[Tuple[Tuple[int, int], ...]] = []

        def zero_like(x):
            if x is None:
                return True
            if hasattr(x, "is_zero_matrix"):
                return bool(x.is_zero_matrix)
            return x == 0

        n_pairs = 0
        for i in range(n):
            for j in range(n_ouc):
                J = sp.zeros(3, 3)
                jij = Jex[i, j]
                if jij != 0:
                    J += sp.eye(3) * jij
                D = DM[i][j]
                if not zero_like(D):
                    J += sp.Matrix([[0, D[2], -D[1]], [-D[2], 0, D[0]], [D[1], -D[0], 0]])
                K = Kex[i][j]
                if not zero_like(K):
                    J += K if K.shape == (3, 3) else sp.diag(K[0], K[1], K[2])
                if J.is_zero_matrix:
                    continue
                n_pairs += 1
                for a in range(3):
                    for b in range(3):
                        if J[a, b] != 0:
                            exprs.append(sp.Rational(1, 2) * J[a, b])
                            factors.append(((i, a), (j, b)))

        rest = sp.expand(sp.sympify(rest))
        n_rest = 0
        if rest != 0:
            flat = [c for row in comps for c in row]
            where = {c: (k // 3, k % 3) for k, c in enumerate(flat)}
            used = [c for c in flat if c in rest.free_symbols]
            if not used:
                raise NumericUnavailable("a non-bilinear term carries no spin component")
            for powers, coef in sp.Poly(rest, *used).terms():
                fac = []
                for c, e in zip(used, powers):
                    fac.extend([where[c]] * int(e))
                if not fac:
                    continue          # a constant has no boson word
                # A single factor (Zeeman) still has one: S^a carries -R[a][2] c^+ c.
                exprs.append(coef)
                factors.append(tuple(fac))
                n_rest += 1

        # Coefficients: constants once, parametric ones through one lambdify.
        uniq: Dict[sp.Expr, int] = {}
        uniq_list: List[sp.Expr] = []
        idx = np.empty(len(exprs), dtype=np.int64)
        for m, e in enumerate(exprs):
            e = subst(e)
            k = uniq.get(e)
            if k is None:
                k = uniq[e] = len(uniq_list)
                uniq_list.append(e)
            idx[m] = k
        self._coef_const = np.zeros(len(exprs))
        param_slots, param_uniq = [], []
        uniq_param_index: Dict[int, int] = {}
        for k, e in enumerate(uniq_list):
            extra = e.free_symbols - allowed
            if extra:
                raise NumericUnavailable(
                    f"a coefficient carries symbols the parameters do not bind: "
                    f"{sorted(map(str, extra))}")
            if not e.free_symbols:
                v = complex(e)
                if abs(v.imag) > 1e-14 * max(1.0, abs(v.real)):
                    raise NumericUnavailable(f"a coefficient is complex ({e})")
                self._coef_const[idx == k] = v.real
            else:
                uniq_param_index[k] = len(param_uniq)
                param_uniq.append(e)
        for m in range(len(exprs)):
            if int(idx[m]) in uniq_param_index:
                param_slots.append((m, uniq_param_index[int(idx[m])]))
        self._param_pos = np.array([p for p, _ in param_slots], dtype=np.int64)
        self._param_src = np.array([s for _, s in param_slots], dtype=np.int64)
        self._param_func = (sp.lambdify(self.args, param_uniq, modules="numpy", cse=True)
                            if param_uniq else None)

        self._ptr = np.zeros(len(factors) + 1, dtype=np.int64)
        np.cumsum([len(f) for f in factors], out=self._ptr[1:])
        self._site = np.array([s for f in factors for (s, _) in f], dtype=np.int64)
        self._comp = np.array([a for f in factors for (_, a) in f], dtype=np.int64)

        # The ground-state rotations gen_HM bakes in (mpr may depend on parameters
        # through `transformations`).
        rots = [sp.Matrix(r) for r in sm.mpr(params_sym)]
        if len(rots) != n or any(r.shape != (3, 3) for r in rots):
            raise NumericUnavailable("mpr() did not return one 3x3 rotation per cell site")
        rot_exprs = [subst(x) for r in rots for x in r]
        extra = set().union(*[e.free_symbols for e in rot_exprs]) - allowed
        if extra:
            raise NumericUnavailable(
                f"a rotation carries unbound symbols {sorted(map(str, extra))}")
        if any(e.free_symbols for e in rot_exprs):
            self._rot_const = None
            self._rot_func = sp.lambdify(self.args, rot_exprs, modules="numpy", cse=True)
        else:
            self._rot_const = np.array([float(e) for e in rot_exprs]).reshape(n, 3, 3)
            self._rot_func = None

        self.n_monomials = len(exprs)
        self.n_pairs = n_pairs
        self.n_nonbilinear = n_rest
        self._last = None
        self.setup_seconds = time.perf_counter() - t0

    # -- per (S, p) -----------------------------------------------------------
    def rotations(self, S: float, params: Sequence[float]) -> np.ndarray:
        """(n, 3, 3) local-to-lab rotations at these values."""
        if self._rot_const is not None:
            return self._rot_const
        vals = self._rot_func(float(S), *[float(p) for p in params])
        return np.array([float(v) for v in vals]).reshape(self.n, 3, 3)

    def coefficients(self, S: float, params: Sequence[float]) -> np.ndarray:
        coef = self._coef_const.copy()
        if self._param_func is not None:
            vals = self._param_func(float(S), *[float(p) for p in params])
            v = np.array([complex(x) for x in vals])
            if np.any(np.abs(v.imag) > 1e-14 * np.maximum(1.0, np.abs(v.real))):
                raise ValueError("a Hamiltonian coefficient evaluated to a complex number")
            coef[self._param_pos] = v.real[self._param_src]
        return coef

    def model(self, S: float, params: Sequence[float]):
        """rsMagCalc `DipoleHq` for these values (the last one is cached)."""
        key = _params_key(S, params)
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        s_hp = self.S_fixed if self.S_fixed is not None else float(S)
        m = self._DipoleHq(self.n, self.pos_ouc, self.ratios * s_hp,
                           self.rotations(S, params),
                           csr=(self.coefficients(S, params), self._ptr, self._site,
                                self._comp))
        self._last = (key, m)
        return m

    def ud(self, S: float, params: Sequence[float]) -> np.ndarray:
        """The block-diagonal rotation matrix Ud (3n x 3n), as `_build_ud_matrix`."""
        R = self.rotations(S, params)
        ud = np.zeros((3 * self.n, 3 * self.n), dtype=np.complex128)
        for i in range(self.n):
            ud[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R[i]
        return ud

    def h_function(self):
        """f(kx, ky, kz, S, *params) -> H, the signature of the lambdified
        `HMat_sym`, rebuilding the bond model only when (S, params) change."""
        def f(kx, ky, kz, S, *params):
            return self.model(S, params).hamiltonians(
                np.array([[kx, ky, kz]], dtype=float))[0]
        return f

    def fixed_function(self, S: float, params: Sequence[float]) -> "BondHamiltonian":
        """A picklable f(kx, ky, kz, S, *params) for the per-q process pools, fixed
        at these values (the pools never change them)."""
        return BondHamiltonian(self.model(S, params), _params_key(S, params))

    def fingerprint(self, S: float, params: Sequence[float]) -> bytes:
        """Bytes that change whenever H(q) does, for the numerical-results cache."""
        dvec, bond, row, col, val = self.model(S, params).entries()
        return b"".join(np.ascontiguousarray(a).tobytes()
                        for a in (dvec, bond, row, col, val))


class BondHamiltonian:
    """H(kx, ky, kz, S, *params) from one bond model, for a worker process.

    Refuses other (S, params): a pool is set up for one set of values, and a
    mismatch would mean evaluating a different Hamiltonian than was asked for."""

    def __init__(self, model, key):
        self.model, self.key = model, key

    def __call__(self, kx, ky, kz, S, *params):
        key = _params_key(S, params)
        if key != self.key:
            raise ValueError(
                f"BondHamiltonian built for (S, params) = {self.key} called with {key}")
        return self.model.hamiltonians(np.array([[kx, ky, kz]], dtype=float))[0]
