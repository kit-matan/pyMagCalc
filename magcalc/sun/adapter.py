"""Adapter that lets the ordinary runner drive the SU(N) engine.

`calculation: {mode: SUN}` in a config swaps the LSWT engine while everything else --
structure, symmetry propagation, q-paths, tasks, plotting, the ground-state guards --
stays exactly as it is. The adapter exposes the small slice of the MagCalc interface the
runner actually uses.
"""
import logging
from typing import List, Optional

import numpy as np

from ..numerical import DispersionResult, SqwResult, thermal_bose_prefactor
from .lswt import SUNModel

logger = logging.getLogger(__name__)


def _supercell_from_config(cfg) -> Optional[list]:
    """`magnetic_supercell` for SU(N): a diagonal [n1,n2,n3] OR a full 3x3 matrix
    (columns = the new lattice vectors, Sunny's reshape_supercell convention). The 3x3
    form is what FeI2 needs and is not expressible in the dipole engine."""
    spec = (cfg.get("crystal_structure") or {}).get("magnetic_supercell")
    if spec is None:
        return None
    if isinstance(spec, dict):
        spec = spec.get("matrix", spec.get("dims"))
    if spec is None or isinstance(spec, str):
        raise ValueError(
            f"SU(N) needs an explicit magnetic_supercell (a [n1,n2,n3] or a 3x3 matrix); "
            f"got {spec!r}. 'auto' is not supported here.")
    arr = np.asarray(spec)
    if arr.shape == (3,):
        return np.diag(arr.astype(int)).tolist()
    if arr.shape == (3, 3):
        return arr.astype(int).tolist()
    raise ValueError(f"magnetic_supercell must be 3 ints or a 3x3 matrix, got {spec!r}")


class SUNCalculator:
    """Quacks like MagCalc for the runner's dispersion / S(Q,w) / guard calls."""

    def __init__(self, spin_model, config, hamiltonian_params=None):
        self.sm = spin_model
        self.config = config
        self.hamiltonian_params = list(hamiltonian_params or [])

        supercell = _supercell_from_config(config)
        min_cfg = config.get("minimization", {}) or {}

        directions = None
        if not (config.get("tasks", {}) or {}).get("minimization", False):
            try:
                th, ph = spin_model.generate_magnetic_structure()
                if th is not None:
                    directions = [[np.sin(t) * np.cos(p), np.sin(t) * np.sin(p),
                                   np.cos(t)] for t, p in zip(th, ph)]
            except Exception:
                directions = None

        if directions is None:
            # No structure supplied (or minimisation requested): seed along z and let
            # the CP^(N-1) search find the ground state. It MUST be found in SU(N) --
            # with an anisotropy a coherent state has <Sz^2> != (S n_z)^2, so the SU(N)
            # ground state differs from the dipole one.
            directions = [[0.0, 0.0, 1.0]] * _n_sites(spin_model, supercell)
            seeded = True
        else:
            seeded = False

        nsite = _n_sites(spin_model, supercell)
        if len(directions) != nsite:
            raise ValueError(
                f"SU(N): the magnetic structure has {len(directions)} directions but the "
                f"model has {nsite} sites"
                + (f" ({len(spin_model.atom_pos())} chemical x "
                   f"{nsite // max(len(spin_model.atom_pos()), 1)} cells)"
                   if supercell is not None else "")
                + ". Give one direction per site of the MAGNETIC cell, or set "
                  "`tasks: {minimization: true}` and let the CP^(N-1) search find it.")

        self.model = SUNModel.from_generic_model(
            spin_model, params=self.hamiltonian_params,
            directions=directions, supercell=supercell)

        minimize = bool((config.get("tasks", {}) or {}).get("minimization", False))
        if minimize or seeded:
            E = self.model.minimize_energy(
                n_restarts=int(min_cfg.get("num_starts", 20)),
                seed=int(min_cfg.get("seed", 0) or 0))
            logger.info(f"SU(N) ground state: E/site = {E / self.model.L:.8f} meV "
                        f"({self.model.L} sites, {self.model.M} bosons/site)")
        else:
            self._audit_supplied_state(config)

        self.nspins = self.model.L
        self.spin_magnitude = self.model.S[0]

    def _audit_supplied_state(self, config):
        """A hand-supplied SU(N) reference state must actually BE the SU(N) ground state.

        This is the trap: the SU(N) ground state differs from the dipole one whenever an
        anisotropy is present, because a coherent state has <Sz^2> != (S n_z)^2. A user
        who runs the model in dipole mode, pastes the resulting `magnetic_structure` into
        the config and flips `mode: SUN` is expanding about the WRONG state -- and the
        imaginary-mode check CANNOT see it, because that state is typically a perfectly
        good LOCAL minimum (verified: |Im w| ~ 5e-16 on FeI2's collinear stripe, which is
        0.048 meV/site above the true ground state). The spectrum comes out real,
        plausible, and wrong. Only an ENERGY audit catches it.
        """
        calc_cfg = config.get("calculation", {}) or {}
        action = str(calc_cfg.get("on_imaginary", "error")).lower()
        if action == "off":
            return
        tol = float(calc_cfg.get("energy_tolerance", 1e-6))

        e_now = self.model.energy_per_site()
        Z = [z.copy() for z in self.model.Z]
        e_rel = self.model.minimize_energy(n_restarts=12, seed=7) / self.model.L
        self.model.Z = Z
        self.model._prepare()

        if e_rel < e_now - max(tol, 1e-9 * abs(e_now)):
            msg = (
                f"The supplied magnetic structure is NOT the SU(N) ground state: relaxing "
                f"the coherent states lowers the energy from {e_now:.8f} to {e_rel:.8f} "
                f"meV/site.\n"
                f"NOTE this will NOT show up as imaginary magnon energies -- such a state "
                f"is usually a perfectly good LOCAL minimum, so the spectrum comes out "
                f"real and plausible and wrong.\n"
                f"  * The SU(N) ground state DIFFERS from the dipole one whenever an "
                f"anisotropy is present (a coherent state has <Sz^2> != (S n_z)^2), so a "
                f"structure taken from a dipole-mode run is generally NOT valid here.\n"
                f"  * Fix: set `tasks: {{minimization: true}}` and let the CP^(N-1) search "
                f"find it, or supply the SU(N) ground state.\n"
                f"  * `calculation: {{on_imaginary: warn}}` downgrades this to a warning."
            )
            if action == "error":
                raise ValueError(msg)
            logger.warning(msg)
        else:
            logger.info(
                f"SU(N) supplied structure verified as a minimum "
                f"(E/site = {e_now:.8f} meV).")

    # -- the runner's calls -------------------------------------------------
    def calculate_dispersion(self, q_vectors, backend="numpy", satellites=None, **_):
        qs = np.asarray(q_vectors, dtype=float).reshape(-1, 3)
        e = np.array([self.model.dispersion(q) for q in qs])
        return DispersionResult(q_vectors=qs, energies=e)

    def calculate_sqw(self, q_vectors, backend="numpy", satellites=None,
                      temperature=None, domains=None, cross_section="perp", **_):
        if domains:
            raise NotImplementedError("SU(N) does not support domain averaging yet.")
        qs = np.asarray(q_vectors, dtype=float).reshape(-1, 3)
        ions = None
        try:
            ions = self.sm.ion_list()
        except Exception:
            pass
        ion = ions[0] if ions else None

        E, I = [], []
        for q in qs:
            w, inten = self.model.structure_factor(q, ion=ion,
                                                   cross_section=cross_section)
            E.append(w)
            I.append(inten)
        E, I = np.array(E), np.array(I)
        if temperature:
            I = I * thermal_bose_prefactor(E, temperature)
        return SqwResult(q_vectors=qs, energies=E, intensities=I)

    def calculate_powder_average(self, *a, **k):
        raise NotImplementedError("SU(N) powder averaging is not implemented yet.")

    # -- the ground-state guards -------------------------------------------
    def stability_report(self, n_q=16, seed=0, q_cart=None):
        B = _recip(self.sm)
        rng = np.random.default_rng(seed)
        qs = rng.uniform(-0.5, 0.5, size=(int(n_q), 3)) @ B
        if q_cart is not None and len(q_cart):
            extra = np.asarray(q_cart, float).reshape(-1, 3)
            if len(extra) > 128:
                extra = extra[np.linspace(0, len(extra) - 1, 128).astype(int)]
            qs = np.vstack([qs, extra])
        worst = band = 0.0
        for q in qs:
            ev = np.linalg.eigvals(self.model.hamiltonian(q))
            worst = max(worst, float(np.max(np.abs(np.imag(ev)))))
            band = max(band, float(np.max(np.abs(np.real(ev)))))
        return {"max_imag": worst, "band_scale": band,
                "relative": worst / band if band > 1e-12 else 0.0}

    def max_imaginary_energy(self, n_q=16, seed=0, q_cart=None):
        return self.stability_report(n_q, seed, q_cart)["max_imag"]

    def relax_from_current(self, params=None, tol=1e-6):
        """The SU(N) energy audit: relax the coherent states and see if the energy drops."""
        e_now = self.model.energy_per_site()
        Z = [z.copy() for z in self.model.Z]
        e_best = self.model.minimize_energy(n_restarts=4, seed=11) / self.model.L
        self.model.Z = Z
        self.model._prepare()
        return float(e_now), float(min(e_now, e_best))

    def update_hamiltonian_params(self, p):
        self.hamiltonian_params = list(p)
        self.__init__(self.sm, self.config, p)


def _n_sites(sm, supercell):
    L = len(sm.atom_pos())
    if supercell is None:
        return L
    return L * int(round(abs(np.linalg.det(np.asarray(supercell, float)))))


def _recip(sm):
    lat = np.asarray(sm.config["crystal_structure"]["lattice_vectors"], float)
    return 2 * np.pi * np.linalg.inv(lat).T
