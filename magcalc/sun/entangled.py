"""Entangled units (dimers / trimers / tetramers) for SU(N) LSWT.

A "unit" is a small cluster of spins treated as ONE effective SU(N) site whose Hilbert
space is the product of the constituents' (N = prod_k (2 S_k + 1)). The intra-unit
couplings are diagonalized EXACTLY and become the on-site term; the reference state is the
unit's ground state (e.g. a dimer SINGLET, which has zero dipole moment -- invisible to
dipole or single-site SU(N) LSWT). Inter-unit couplings become bonds between effective
sites, coupling the embedded constituent-spin operators. Excitations are transitions
within the unit's spectrum -- the triplon of a dimer magnet.

This is Sunny's `EntangledSystem` analogue. It reuses the generalized `SUNModel` engine:
each unit passes its embedded operators, its intra-unit Hamiltonian as `onsite`, and its
total spin as the neutron `moment_operators`.

Validated (tests/test_entangled_units.py) against exact / analytic references:
  * isolated S=1/2 dimer -> flat triplon at omega = J (the singlet-triplet gap);
  * coupled-dimer chain -> omega(k) = sqrt(J^2 - J J' cos k), the harmonic bond-operator
    (Sachdev-Bhatt) triplon dispersion, to machine precision.
"""
import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .lswt import SUNModel
from .operators import spin_matrices

logger = logging.getLogger(__name__)


def _embedded_spin_ops(spins: Sequence[float]) -> List[List[np.ndarray]]:
    """For a unit of spins [S_0, S_1, ...] return, per constituent, its 3 spin operators
    embedded (via Kronecker products) in the product Hilbert space."""
    mats = [spin_matrices(s) for s in spins]          # each (3, n_k, n_k)
    dims = [m.shape[1] for m in mats]
    ops = []
    for k in range(len(spins)):
        comp = []
        for a in range(3):
            blocks = [mats[m][a] if m == k else np.eye(dims[m]) for m in range(len(spins))]
            O = blocks[0]
            for B in blocks[1:]:
                O = np.kron(O, B)
            comp.append(O)
        ops.append(comp)                              # ops[k] = [S_k^x, S_k^y, S_k^z]
    return ops


def _num(x) -> float:
    import sympy as _sp
    return float(_sp.N(x)) if hasattr(x, "free_symbols") else float(x)


def _pair_matrix(Jex, DM, Kex, i, j) -> np.ndarray:
    """The 3x3 exchange matrix for physical pair (i, j), same assembly as the SU(N)
    dipole bridge: J*I + skew(D) + Kex."""
    M = np.zeros((3, 3), dtype=float)
    jv = Jex[i, j]
    if jv != 0:
        M += _num(jv) * np.eye(3)
    D = DM[i][j]
    if D is not None and not (hasattr(D, "is_zero_matrix") and D.is_zero_matrix):
        dx, dy, dz = _num(D[0]), _num(D[1]), _num(D[2])
        M += np.array([[0.0, dz, -dy], [-dz, 0.0, dx], [dy, -dx, 0.0]])
    K = Kex[i][j]
    if K is not None and not (hasattr(K, "is_zero_matrix") and K.is_zero_matrix):
        Km = np.asarray(K, dtype=object)
        if Km.shape == (3, 3):
            M += np.array([[_num(Km[a, b]) for b in range(3)] for a in range(3)])
        else:
            M += np.diag([_num(Km[a]) for a in range(3)])
    return M


def build_entangled_model(model, params: Optional[Sequence[float]] = None,
                          units: Optional[Sequence[Sequence[int]]] = None) -> SUNModel:
    """Build an entangled-unit `SUNModel` from a `GenericSpinModel`.

    `units` is a list of units, each a list of CHEMICAL-cell site indices (0-based, in the
    order of `crystal_structure.atoms_uc`). Every magnetic site must appear in exactly one
    unit, and all units must have the same product dimension N. `params` are the model's
    numeric parameters (as for the other engines).
    """
    params = list(params if params is not None else [])
    Jex, DM, Kex = model.spin_interactions(params)
    apos = np.asarray(model.atom_pos(), dtype=float)        # CARTESIAN site positions
    aouc = np.asarray(model.atom_pos_ouc(), dtype=float)    # CARTESIAN over-cell positions
    lat = np.asarray(model.unit_cell(), dtype=float)        # rows = lattice vectors
    inv_lat = np.linalg.inv(lat)
    L = len(apos)
    spins_all = [float(s) for s in model.spin_magnitudes()]

    if units is None:
        raise ValueError("entangled mode needs a `units` list (each unit a list of site "
                         "indices).")
    units = [[int(s) for s in u] for u in units]
    flat = [s for u in units for s in u]
    if sorted(flat) != list(range(L)):
        raise ValueError(f"`units` must partition all {L} magnetic sites exactly once; "
                         f"got {units}.")

    unit_of = {s: u for u, sites in enumerate(units) for s in sites}
    pos_in_unit = {s: p for sites in units for p, s in enumerate(sites)}
    U = len(units)

    # Per-unit embedded operators, moment terms (constituent offset + op indices), centroid.
    unit_ops, unit_moment_terms, unit_centroid, unit_dim = [], [], [], []
    for sites in units:
        s_list = [spins_all[s] for s in sites]
        emb = _embedded_spin_ops(s_list)                       # emb[p] = [Sx,Sy,Sz]
        ops = [emb[p][a] for p in range(len(sites)) for a in range(3)]  # 3*k operators
        centroid = np.mean([apos[s] for s in sites], axis=0)   # CARTESIAN centroid
        # neutron moment = sum_k e^{i q.(r_k - centroid)} S_k -> the q-dependent (staggered)
        # combination that the triplon actually couples to.
        mterms = [(apos[s] - centroid, (3 * p, 3 * p + 1, 3 * p + 2))
                  for p, s in enumerate(sites)]
        unit_ops.append(ops)
        unit_moment_terms.append(mterms)
        unit_dim.append(ops[0].shape[0])
        unit_centroid.append(centroid)
    if len(set(unit_dim)) != 1:
        raise NotImplementedError(f"all units must have the same dimension N; got {unit_dim}.")
    n_per_unit = [len(u) for u in units]

    def _op_index(unit, site):
        return 3 * pos_in_unit[site]        # first (x) operator index of `site` within `unit`

    # Split physical pairs into intra-unit (-> on-site A) and inter-unit (-> bonds).
    A_intra = [np.zeros((unit_dim[0], unit_dim[0]), dtype=complex) for _ in range(U)]
    bond_C: Dict[Any, np.ndarray] = {}      # (u_i, u_j, offset-tuple) -> coupling matrix
    for i in range(L):
        for j in range(len(aouc)):
            M = _pair_matrix(Jex, DM, Kex, i, j)
            if not np.any(M):
                continue
            jc = j % L
            # cartesian cell translation of site j -> fractional integer cell offset
            offset = np.round((aouc[j] - apos[jc]) @ inv_lat).astype(int)
            ui, uj = unit_of[i], unit_of[jc]
            if ui == uj and np.all(offset == 0):
                # intra-unit: fold J_ab S_i^a S_j^b into the on-site Hamiltonian.
                oi, oj = _op_index(ui, i), _op_index(ui, jc)
                emb = unit_ops[ui]
                for a in range(3):
                    for b in range(3):
                        if M[a, b] != 0.0:
                            A_intra[ui] += 0.5 * M[a, b] * (emb[oi + a] @ emb[oj + b])
                continue
            # inter-unit bond between effective sites ui, uj at this cell offset.
            key = (ui, uj, tuple(int(x) for x in offset))
            C = bond_C.setdefault(key, np.zeros((3 * n_per_unit[ui], 3 * n_per_unit[uj])))
            oi, oj = _op_index(ui, i), _op_index(uj, jc)
            C[oi:oi + 3, oj:oj + 3] += M

    # Reference state per unit = ground state of its on-site (intra-unit) Hamiltonian.
    coherent = []
    for u in range(U):
        w, V = np.linalg.eigh(A_intra[u])
        if abs(w[1] - w[0]) < 1e-9:
            logger.warning("Unit %d has a degenerate ground state (gap %.2e); the "
                           "reference is ambiguous.", u, w[1] - w[0])
        coherent.append(V[:, 0])

    # Assemble bonds with the CARTESIAN inter-unit displacement: centroid(uj) shifted by
    # the cell translation, minus centroid(ui).
    bonds = []
    for (ui, uj, offset), C in bond_C.items():
        dr = unit_centroid[uj] + (np.array(offset, float) @ lat) - unit_centroid[ui]
        bonds.append((ui, uj, dr, C))

    onsite = [(u, A_intra[u]) for u in range(U)]
    sm = SUNModel(spins=[1.0] * U, coherent_states=coherent, bonds=bonds, onsite=onsite,
                  operators=unit_ops, moment_terms=unit_moment_terms)
    sm.pos = np.array(unit_centroid)          # cartesian, for the structure factor phase
    return sm


def _resolve_units(config, spin_model) -> List[List[int]]:
    """Read the `units` spec (top-level or under `calculation`) and return lists of
    chemical-cell site INDICES. Each unit may be given by site label or by index."""
    spec = config.get("units")
    if spec is None:
        spec = (config.get("calculation") or {}).get("units")
    if not spec:
        raise ValueError(
            "entangled mode needs a `units:` list, e.g. `units: [[Cu0, Cu1]]` (each unit a "
            "list of site labels or indices forming one cluster).")
    atoms = spin_model.config["crystal_structure"]["atoms_uc"]
    lab2i = {a["label"]: i for i, a in enumerate(atoms)}
    out = []
    for unit in spec:
        idx = []
        for s in unit:
            if isinstance(s, int):
                idx.append(s)
            elif s in lab2i:
                idx.append(lab2i[s])
            else:
                raise ValueError(f"units: unknown site label {s!r}; "
                                 f"known labels: {list(lab2i)}")
        out.append(idx)
    return out


class EntangledCalculator:
    """Quacks like MagCalc/SUNCalculator for the runner, backed by an entangled-unit
    SUNModel. `calculation: {mode: entangled}` + `units:` selects it."""

    def __init__(self, spin_model, config, hamiltonian_params=None):
        from ..numerical import DispersionResult, SqwResult, thermal_bose_prefactor
        self._DispersionResult = DispersionResult
        self._SqwResult = SqwResult
        self._bose = thermal_bose_prefactor

        self.sm = spin_model
        self.config = config
        self.hamiltonian_params = list(hamiltonian_params or [])
        units = _resolve_units(config, spin_model)
        self.model = build_entangled_model(spin_model, self.hamiltonian_params, units)
        self.nspins = self.model.L
        self.spin_magnitude = 1.0
        logger.info("Entangled units: %d unit(s), N=%d per unit (%d bosons), %d bond(s).",
                    self.model.L, self.model.N, self.model.M, len(self.model.bonds))

    def calculate_dispersion(self, q_vectors, backend="numpy", satellites=None, **_):
        qs = np.asarray(q_vectors, dtype=float).reshape(-1, 3)
        e = np.array([self.model.dispersion(q) for q in qs])
        return self._DispersionResult(q_vectors=qs, energies=e)

    def calculate_sqw(self, q_vectors, backend="numpy", satellites=None,
                      temperature=None, domains=None, cross_section="perp", **_):
        if domains:
            raise NotImplementedError("entangled units do not support domain averaging yet.")
        qs = np.asarray(q_vectors, dtype=float).reshape(-1, 3)
        try:
            ions = self.sm.ion_list()
            ion = ions[0] if ions else None
        except Exception:
            ion = None
        E, I = [], []
        for q in qs:
            w, inten = self.model.structure_factor(q, ion=ion, cross_section=cross_section)
            E.append(w)
            I.append(inten)
        E, I = np.array(E), np.array(I)
        if temperature:
            I = I * self._bose(E, temperature)
        return self._SqwResult(q_vectors=qs, energies=E, intensities=I)

    def calculate_powder_average(self, *a, **k):
        raise NotImplementedError("entangled-unit powder averaging is not implemented yet.")

    def stability_report(self, n_q=16, seed=0, q_cart=None):
        lat = np.asarray(self.sm.config["crystal_structure"]["lattice_vectors"], float)
        B = 2 * np.pi * np.linalg.inv(lat).T
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
        # The reference is the EXACT intra-unit ground state; there is no in-cell coherent
        # relaxation to do. (An over-strong inter-unit coupling shows up instead as an
        # imaginary triplon in stability_report -- the dimer picture breaking down.)
        e = self.model.energy_per_site()
        return float(e), float(e)

    def update_hamiltonian_params(self, p):
        self.hamiltonian_params = list(p)
        self.__init__(self.sm, self.config, p)
