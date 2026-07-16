"""SU(N) (generalized) spin-wave theory.

Each site carries an N-level local Hilbert space and a coherent state |Z_i>. Choosing a
local basis U_i whose first column is Z_i, an on-site operator A becomes

    A~ = U~ A U,   and, to quadratic order in the N-1 bosons b_m,

    A  ~  A~_00 (1 - sum_m b~_m b_m)
        + sum_m ( A~_0m b_m + A~_m0 b~_m )
        + sum_mn A~_mn b~_m b_n

so that the ONE-boson (linear) part is A~_0m, and the QUADRATIC part is
Q_mn = A~_mn - delta_mn A~_00.

For a bond term  sum_ab J_ab A^a_i B^b_j  the quadratic-in-boson piece is

    <A^a>_i * Q[B^b]_j  +  <B^b>_j * Q[A^a]_i        (on-site "mean field", no phase)
  + (linear at i) x (linear at j)                     (hopping + anomalous, with phase)

which is exactly the dipole construction generalised from 1 boson to N-1.

CONVENTIONS (both verified against the host, see tests):
  * bonds are listed in BOTH directions, and there is NO 1/2 on the hopping -- this is
    how pyMagCalc encodes H = (1/2) sum_ordered;
  * the on-site mean-field term is the q=0 sum (NO phase);
  * the returned H(q) is g*H2, i.e. the Bogoliubov metric is already folded in (the
    host stores it that way -- see core._ewald_nambu), so its eigenvalues come in
    +/- omega pairs.

For S=1/2 (N=2) there is exactly one boson per site and this reduces IDENTICALLY to
dipole LSWT -- the load-bearing test.
"""
from typing import List, Optional, Sequence, Tuple

from itertools import product as _product

import numpy as np

from .operators import (coherent_from_direction, local_basis, spin_matrices,
                        stevens_matrices)


class SUNModel:
    """A minimal SU(N) LSWT model.

    sites   : list of (S, Z) -- Z is the coherent state (N-vector). Use
              `from_directions` to build the dipole-equivalent reference state.
    bonds   : list of (i, j, dr_cart, J) with J a real 3x3; BOTH directions must be
              listed (as everywhere else in pyMagCalc).
    onsite  : list of (i, A) with A an N_i x N_i Hermitian matrix (crystal field).
    """

    def __init__(
        self,
        spins: Sequence[float],
        coherent_states: Sequence[np.ndarray],
        bonds: Sequence[Tuple[int, int, np.ndarray, np.ndarray]],
        onsite: Optional[Sequence[Tuple[int, np.ndarray]]] = None,
        operators: Optional[Sequence[Sequence[np.ndarray]]] = None,
        moment_terms: Optional[Sequence[Sequence[Tuple[np.ndarray, Sequence[int]]]]] = None,
    ):
        """
        By default each site carries the 3 dipole operators `spin_matrices(S)` and a bond's
        matrix `J` couples them (S_i^a J_ab S_j^b) -- ordinary SU(N).

        For ENTANGLED UNITS a site is a small CLUSTER (dimer/trimer/...) whose N is the
        product Hilbert-space dimension. Pass `operators[i]` = the site's local operators
        (e.g. the embedded constituent spins S_1^x..S_2^z of a dimer) and give each bond a
        coupling matrix `C` of shape (n_ops_i, n_ops_j): the term is sum_{p,r} C[p,r]
        O_i^p O_j^r. `moment_terms[i]` describes the neutron magnetic-moment operator of the
        unit as a list of `(d_k, [ix, iy, iz])`: constituent k sits at CARTESIAN offset `d_k`
        from the unit position and contributes `operators[i][ix|iy|iz]`. The structure
        factor then builds the q-DEPENDENT moment sum_k e^{i q.d_k} S_k -- essential for
        entangled units, where the singlet->triplet triplon is excited by the STAGGERED
        combination (the total spin sum_k S_k has zero matrix element). It defaults to the
        first three operators at zero offset (the plain-dipole case). The reference
        `coherent_states[i]` is the intra-unit ground state (e.g. a dimer singlet -- zero
        dipole, so dipole LSWT cannot touch it), not a spin coherent state.
        """
        self.S = [float(s) for s in spins]
        self.Z = [np.asarray(z, dtype=complex) for z in coherent_states]
        self.bonds = [(int(i), int(j), np.asarray(dr, float), np.asarray(J, float))
                      for (i, j, dr, J) in bonds]
        self.onsite = [(int(i), np.asarray(A, dtype=complex))
                       for (i, A) in (onsite or [])]

        self.pos = None          # site positions (set by the bridge; needed for S(q,w))
        self.L = len(self.Z)
        self.Ns = [len(z) for z in self.Z]       # N = local Hilbert-space dimension
        if len(set(self.Ns)) != 1:
            raise NotImplementedError(
                "SU(N) currently requires all sites to have the same N.")
        self.N = self.Ns[0]
        self.M = self.N - 1                      # bosons per site

        # Local operators used by the bonds. Default: the 3 dipole components.
        if operators is None:
            self.ops = [[spin_matrices(s)[a] for a in range(3)] for s in self.S]
        else:
            self.ops = [[np.asarray(o, dtype=complex) for o in ol] for ol in operators]
        self.n_ops = len(self.ops[0])
        if any(len(ol) != self.n_ops for ol in self.ops):
            raise NotImplementedError("all sites must expose the same number of operators.")
        # Neutron magnetic-moment operator, as (cartesian offset d_k, [ix,iy,iz]) terms.
        # Default: the first 3 operators at zero offset (plain dipole).
        if moment_terms is None:
            self.moment_terms = [[(np.zeros(3), (0, 1, 2))] for _ in range(self.L)]
        else:
            self.moment_terms = [[(np.asarray(d, float), tuple(int(x) for x in idx))
                                  for (d, idx) in ml] for ml in moment_terms]
        self._prepare()

    # ---------------------------------------------------------------- setup
    @classmethod
    def from_directions(cls, spins, directions, bonds, onsite=None):
        """Reference state = the spin coherent state pointing along `directions` -- the
        SU(N) state that corresponds to the classical dipole."""
        Z = [coherent_from_direction(s, d) for s, d in zip(spins, directions)]
        return cls(spins, Z, bonds, onsite)

    def _prepare(self):
        """Rotate every local operator into the |Z> basis and cache the pieces."""
        L, M, nop = self.L, self.M, self.n_ops
        self.s0 = np.zeros((L, nop), dtype=complex)        # <Z|O^p|Z>
        self.t = np.zeros((L, nop, M), dtype=complex)      # <0|O^p|m>  (coeff of b_m)
        self.tb = np.zeros((L, nop, M), dtype=complex)     # <m|O^p|0>  (coeff of b_m^dag)
        self.Q = np.zeros((L, nop, M, M), dtype=complex)   # O~_mn - delta O~_00
        self.QA = np.zeros((L, M, M), dtype=complex)       # on-site (crystal field / intra-unit)

        for i in range(L):
            U = local_basis(self.Z[i])
            for p in range(nop):
                Ot = U.conj().T @ self.ops[i][p] @ U
                self.s0[i, p] = Ot[0, 0]
                self.t[i, p, :] = Ot[0, 1:]
                self.tb[i, p, :] = Ot[1:, 0]
                self.Q[i, p] = Ot[1:, 1:] - np.eye(M) * Ot[0, 0]

        for (i, A) in self.onsite:
            U = local_basis(self.Z[i])
            At = U.conj().T @ A @ U
            self.QA[i] += At[1:, 1:] - np.eye(M) * At[0, 0]

    # ---------------------------------------------------------------- energy
    def energy_per_site(self) -> float:
        """Classical energy PER SITE (compare with Sunny's `energy_per_site`)."""
        return self.classical_energy() / self.L

    def classical_energy(self) -> float:
        """TOTAL classical energy of the magnetic cell:
        E = (1/2) sum_ordered <S_i> J <S_j> + sum_i <A_i>.

        NOTE this is the total, not per site -- divide by L (or use `energy_per_site`)
        when comparing with Sunny. The two coincide only for a one-site cell, which is
        why the single-site validation gates did not catch the difference.
        """
        E = 0.0
        for (i, j, dr, J) in self.bonds:
            E += 0.5 * float(np.real(self.s0[i] @ J @ self.s0[j]))
        for (i, A) in self.onsite:
            Z = self.Z[i]
            E += float(np.real(Z.conj() @ A @ Z))
        return E

    # ---------------------------------------------------------------- H(q)
    def hamiltonian(self, q_cart: np.ndarray) -> np.ndarray:
        """g * H2(q), shape (2 L M, 2 L M). Eigenvalues come in +/- omega pairs."""
        L, M = self.L, self.M
        D = L * M
        H11 = np.zeros((D, D), dtype=complex)
        H22 = np.zeros((D, D), dtype=complex)
        H12 = np.zeros((D, D), dtype=complex)
        H21 = np.zeros((D, D), dtype=complex)
        q = np.asarray(q_cart, dtype=float)

        for (i, j, dr, J) in self.bonds:
            ph = np.exp(1j * float(np.dot(q, dr)))
            bi, bj = i * M, j * M
            for a in range(self.n_ops):
                for b in range(self.n_ops):
                    c = J[a, b]
                    if c == 0.0:
                        continue
                    ti, tbi = self.t[i, a], self.tb[i, a]
                    tj, tbj = self.t[j, b], self.tb[j, b]
                    # inter-site: hopping and anomalous (carry the phase)
                    H11[bi:bi + M, bj:bj + M] += c * np.outer(tbi, tj) * ph
                    H22[bi:bi + M, bj:bj + M] += c * np.outer(ti, tbj) * ph
                    H12[bi:bi + M, bj:bj + M] += c * np.outer(tbi, tbj) * ph
                    H21[bi:bi + M, bj:bj + M] += c * np.outer(ti, tj) * ph
                    # on-site mean field at i, weighted by <S^b>_j. NO phase: it is the
                    # q=0 sum. (Putting the phase here makes a ferromagnet's H(q)
                    # cancel to zero -- the same trap as in the dipole engine.)
                    mf = c * self.Q[i, a] * self.s0[j, b]
                    H11[bi:bi + M, bi:bi + M] += mf
                    H22[bi:bi + M, bi:bi + M] += mf.T

        # single-ion / crystal field
        for i in range(L):
            if np.any(self.QA[i]):
                bi = i * M
                H11[bi:bi + M, bi:bi + M] += self.QA[i]
                H22[bi:bi + M, bi:bi + M] += self.QA[i].T

        g = np.diag([1.0] * D + [-1.0] * D)
        return g @ np.block([[H11, H12], [H21, H22]])

    def dispersion(self, q_cart: np.ndarray) -> np.ndarray:
        """The L*M positive magnon energies at q (ascending)."""
        ev = np.linalg.eigvals(self.hamiltonian(q_cart))
        w = np.sort(np.real(ev))[self.L * self.M:]
        return np.sort(w)

    def max_imaginary(self, q_cart: np.ndarray) -> float:
        ev = np.linalg.eigvals(self.hamiltonian(q_cart))
        return float(np.max(np.abs(np.imag(ev))))

    # ------------------------------------------------------- host bridge
    @staticmethod
    def _replicate(bonds, onsite, spins, L_chem, lat_chem, pos_frac, M):
        """Replicate a chemical-cell bond list into a (possibly NON-DIAGONAL) supercell.

        `M` has the new lattice vectors as COLUMNS, in units of the chemical ones
        (Sunny's `reshape_supercell` convention): A_super = M^T @ A_chem.

        pyMagCalc's own `magnetic_supercell` only supports DIAGONAL cells, but FeI2's
        magnetic cell is [1 0 0; 0 1 -2; 0 1 2]. Doing the replication here keeps the
        (validated) symmetry propagation of the exchange matrices upstream and only
        rearranges sites -- which is the whole point: hand-rolling the symmetry is what
        went wrong before.
        """
        M = np.asarray(M, dtype=int)
        ncell = int(round(abs(np.linalg.det(M))))
        Minv = np.linalg.inv(M)

        # the chemical cells that lie inside the supercell
        R = int(np.max(np.abs(M))) + 2
        cells = [np.array(n) for n in _product(range(-R, R + 1), repeat=3)
                 if all(-1e-9 <= x < 1 - 1e-9 for x in Minv @ np.asarray(n, float))]
        if len(cells) != ncell:
            raise ValueError(
                f"supercell matrix {M.tolist()} has |det| = {ncell} but "
                f"{len(cells)} cells were enumerated.")

        def fold(n):
            """Fold a chemical cell index back into the supercell; return its slot."""
            base = np.floor(Minv @ np.asarray(n, float) + 1e-9)
            n_in = np.asarray(n, int) - (M @ base).astype(int)
            for k, c in enumerate(cells):
                if np.array_equal(c, n_in):
                    return k
            raise ValueError(f"cell {n} did not fold into the supercell")

        inv_lat = np.linalg.inv(lat_chem)
        new_bonds = []
        for (ci, ni) in enumerate(cells):
            for (i, j, dr, J) in bonds:
                # recover the integer chemical-cell offset of this bond
                off = np.round((dr + (pos_frac[i] - pos_frac[j]) @ lat_chem) @ inv_lat)
                cj = fold(ni + off.astype(int))
                new_bonds.append((ci * L_chem + i, cj * L_chem + j, dr, J))

        new_onsite = [(ci * L_chem + i, A)
                      for ci in range(ncell) for (i, A) in onsite]
        new_pos = [(ni @ lat_chem) + (pos_frac[i] @ lat_chem)
                   for ni in cells for i in range(L_chem)]
        return (new_bonds, new_onsite,
                [spins[i % L_chem] for i in range(ncell * L_chem)],
                np.array(new_pos))

    @classmethod
    def from_generic_model(cls, model, params=None, directions=None, supercell=None):
        """Build an SU(N) model from a pyMagCalc `GenericSpinModel`.

        This reuses the whole existing front end -- CIF/Wyckoff structure, space-group
        symmetry propagation of the exchange matrices, magnetic supercells -- and only
        swaps the LSWT engine. The bond list is taken from the SAME (Jex, DM, Kex)
        matrices the dipole engine uses, so a model that runs in dipole mode runs here.

        The reference state is the spin coherent state pointing along each site's
        classical direction (from the model's magnetic structure, or `directions`).
        That covers dipolar orders -- including FeI2's -- but not genuinely non-dipolar
        (spin-nematic) states, which need a CP^(N-1) search.
        """
        import numpy as _np

        params = list(params if params is not None else [])
        Jex, DM, Kex = model.spin_interactions(params)

        apos = _np.asarray(model.atom_pos(), dtype=float)
        aouc = _np.asarray(model.atom_pos_ouc(), dtype=float)
        L = len(apos)
        positions = apos.copy()
        spins = [float(s) for s in model.spin_magnitudes()]

        def _num(x):
            import sympy as _sp
            return float(_sp.N(x)) if hasattr(x, "free_symbols") else float(x)

        bonds = []
        for i in range(L):
            for j in range(len(aouc)):
                J = _np.zeros((3, 3), dtype=float)
                jv = Jex[i, j]
                if jv != 0:
                    J += _num(jv) * _np.eye(3)
                D = DM[i][j]
                if D is not None and not (hasattr(D, "is_zero_matrix")
                                          and D.is_zero_matrix):
                    dx, dy, dz = (_num(D[0]), _num(D[1]), _num(D[2]))
                    # D . (S_i x S_j) = S_i^T M S_j
                    J += _np.array([[0.0, dz, -dy], [-dz, 0.0, dx], [dy, -dx, 0.0]])
                K = Kex[i][j]
                if K is not None and not (hasattr(K, "is_zero_matrix")
                                          and K.is_zero_matrix):
                    Km = _np.asarray(K, dtype=object)
                    if Km.shape == (3, 3):
                        J += _np.array([[_num(Km[a, b]) for b in range(3)]
                                        for a in range(3)])
                    else:                      # diagonal anisotropic exchange
                        J += _np.diag([_num(Km[a]) for a in range(3)])
                if _np.any(J):
                    bonds.append((i, j % L, aouc[j] - apos[i], J))

        # --- on-site terms: SIA, 3x3 anisotropy tensor, Stevens ---------------
        onsite = []
        Sops = {s: spin_matrices(s) for s in set(spins)}
        pmap = model._resolve_param_map(params)
        for inter in model.interactions_config:
            t = inter.get("type")
            if t not in ("sia", "sia_matrix", "anisotropy_matrix", "stevens"):
                continue
            labels = inter.get("atoms") or inter.get("atom_labels")
            atoms_uc = model.config["crystal_structure"]["atoms_uc"]
            lab2i = {a["label"]: k for k, a in enumerate(atoms_uc)}
            targets = [lab2i[l] for l in labels if l in lab2i] if labels else range(L)

            for i in targets:
                Sx, Sy, Sz = Sops[spins[i]]
                if t == "sia":
                    val = model._resolve_scalar(inter.get("value"), pmap)
                    n = _np.asarray(inter.get("axis", [0, 0, 1]), dtype=float)
                    n = n / _np.linalg.norm(n)
                    nS = n[0] * Sx + n[1] * Sy + n[2] * Sz
                    onsite.append((i, _num(val) * (nS @ nS)))
                elif t in ("sia_matrix", "anisotropy_matrix"):
                    Amat = inter.get("matrix", inter.get("value"))
                    A = _np.zeros_like(Sx)
                    Svec = [Sx, Sy, Sz]
                    for a in range(3):
                        for b in range(3):
                            c = _num(model._resolve_scalar(Amat[a][b], pmap))
                            if c:
                                A = A + c * (Svec[a] @ Svec[b])
                    onsite.append((i, A))
                elif t == "stevens":
                    terms = {}
                    if "B" in inter:
                        for key, v in (inter["B"] or {}).items():
                            k_, q_ = str(key).replace(" ", "").split(",")
                            terms[(int(k_), int(q_))] = v
                    else:
                        terms[(int(inter["k"]), int(inter.get("q", 0)))] = \
                            inter.get("value")
                    A = _np.zeros_like(Sx)
                    for (k_, q_), v in terms.items():
                        B = _num(model._resolve_scalar(v, pmap))
                        A = A + B * stevens_matrices(spins[i], k_, q_)
                    onsite.append((i, A))

        if supercell is not None:
            lat = _np.asarray(model.config["crystal_structure"]["lattice_vectors"],
                              dtype=float)
            pf = _np.asarray([a["pos"] for a in
                              model.config["crystal_structure"]["atoms_uc"]], dtype=float)
            bonds, onsite, spins, positions = cls._replicate(
                bonds, onsite, spins, L, lat, pf, supercell)

        if directions is None:
            th, ph = model.generate_magnetic_structure()
            if th is None:
                raise ValueError("SU(N): the model has no magnetic_structure and no "
                                 "`directions` were given.")
            directions = [[_np.sin(t) * _np.cos(p), _np.sin(t) * _np.sin(p), _np.cos(t)]
                          for t, p in zip(th, ph)]
            if supercell is not None:
                # the chemical structure repeated; the CP^(N-1) search will relax it
                reps = len(spins) // len(directions)
                directions = directions * reps

        mdl = cls.from_directions(spins, directions, bonds, onsite)
        mdl.pos = _np.asarray(positions, dtype=float)
        return mdl

    # ------------------------------------------------- CP^(N-1) ground state
    def local_field(self, i: int, s0: np.ndarray) -> np.ndarray:
        """The local N x N Hamiltonian felt by site i, given the current expectations.

        E = (1/2) sum_bonds <S_i> J <S_j> + sum_i <A_i>, and with both bond directions
        listed (and J_ji = J_ij^T) the derivative collapses to

            h_i = sum_{bonds starting at i} sum_ab J_ab <S_j>^b S^a  +  A_i
        """
        Sxyz = spin_matrices(self.S[i])
        h = np.zeros((self.N, self.N), dtype=complex)
        for (bi, bj, dr, J) in self.bonds:
            if bi != i:
                continue
            for a in range(3):
                for b in range(3):
                    if J[a, b]:
                        h += J[a, b] * np.real(s0[bj, b]) * Sxyz[a]
        for (k, A) in self.onsite:
            if k == i:
                h += A
        return h

    def minimize_energy(self, n_restarts: int = 8, max_iter: int = 400,
                        tol: float = 1e-12, seed: int = 0):
        """Ground state on CP^(N-1) by self-consistent local-field diagonalisation.

        The SU(N) analogue of SpinW's `optmagsteep`: for a fixed environment the optimal
        |Z_i> is simply the LOWEST EIGENVECTOR of the local field h_i, so iterate that to
        self-consistency. Random restarts give the global search.

        This is NOT the same as the dipole ground state whenever an anisotropy is
        present: a coherent state has <Sz^2> != (S n_z)^2, so the two theories have
        genuinely different classical energies for a canted structure (FeI2's, for
        instance). Using the dipole ground state in SU(N) would be simply wrong.
        """
        rng = np.random.default_rng(seed)
        best_E, best_Z = np.inf, None

        for r in range(max(int(n_restarts), 1)):
            if r == 0:
                Z = [z.copy() for z in self.Z]                 # current state as a seed
            else:
                Z = []
                for i in range(self.L):
                    v = rng.normal(size=self.N) + 1j * rng.normal(size=self.N)
                    Z.append(v / np.linalg.norm(v))

            E_prev = np.inf
            for _ in range(max_iter):
                s0 = np.array([[Zi.conj() @ spin_matrices(self.S[i])[a] @ Zi
                                for a in range(3)] for i, Zi in enumerate(Z)])
                for i in range(self.L):
                    h = self.local_field(i, s0)
                    w, v = np.linalg.eigh(h)
                    Z[i] = v[:, int(np.argmin(w))]
                    s0[i] = [Z[i].conj() @ spin_matrices(self.S[i])[a] @ Z[i]
                             for a in range(3)]
                E = self._energy_of(Z)
                if abs(E_prev - E) < tol:
                    break
                E_prev = E

            E = self._energy_of(Z)
            if E < best_E - 1e-12:
                best_E, best_Z = E, [z.copy() for z in Z]

        self.Z = best_Z
        self._prepare()
        return best_E

    def _energy_of(self, Z) -> float:
        s0 = np.array([[Z[i].conj() @ spin_matrices(self.S[i])[a] @ Z[i]
                        for a in range(3)] for i in range(self.L)])
        E = 0.0
        for (i, j, dr, J) in self.bonds:
            E += 0.5 * float(np.real(s0[i] @ J @ s0[j]))
        for (i, A) in self.onsite:
            E += float(np.real(Z[i].conj() @ A @ Z[i]))
        return E

    @property
    def dipoles(self) -> np.ndarray:
        """<S_i> for the current coherent states."""
        return np.array([[float(np.real(self.Z[i].conj()
                                        @ spin_matrices(self.S[i])[a] @ self.Z[i]))
                          for a in range(3)] for i in range(self.L)])

    # ---------------------------------------------------------------- intensities
    def _bogoliubov(self, q_cart):
        """Colpa diagonalisation of H(q). Returns (omega, T) with Psi = T Phi,
        Psi = (b, b^dag), Phi = (beta, beta^dag), and omega the L*M positive energies."""
        D = self.L * self.M
        g = np.diag([1.0] * D + [-1.0] * D)
        H2 = g @ self.hamiltonian(q_cart)        # hamiltonian() returns g*H2
        H2 = 0.5 * (H2 + H2.conj().T)            # symmetrise away round-off

        try:
            C = np.linalg.cholesky(H2).conj().T   # H2 = C^dag C
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "SU(N) H(q) is not positive definite: the reference state is not a "
                "minimum (imaginary magnons). Minimise first.") from exc

        W = C @ g @ C.conj().T
        W = 0.5 * (W + W.conj().T)
        vals, U = np.linalg.eigh(W)
        # eigh gives ascending; we want the D positive ones first (descending), then the
        # D negative ones -- Colpa's ordering, so that E' = g E is all positive.
        order = np.concatenate([np.argsort(-vals)[:D], np.argsort(vals)[:D]])
        vals, U = vals[order], U[:, order]
        Ep = np.abs(vals)                         # = g E, all positive
        T = np.linalg.inv(C) @ U @ np.diag(np.sqrt(Ep))
        return Ep[:D], T

    def structure_factor(self, q_cart, ion=None, cross_section="perp"):
        """(omega, intensity) per magnon band at q.

        The one-magnon matrix element. Keeping only the LINEAR part of the spin operator,

            S^a(q) = sum_i e^{i q.r_i} [ sum_m t^a_i[m] b_im + tb^a_i[m] b^dag_im ]

        so in the Nambu basis Psi = (b, b^dag) it is a row vector v_a. With Psi = T Phi,
        the amplitude to create magnon nu is (v_a T)[D + nu], and

            S^{ab}(q, w_nu) = conj(M^a_nu) M^b_nu.

        Because the observables here are the FULL N x N spin operators, the multipolar
        (single-ion) bands carry weight -- which is exactly what dipole LSWT cannot do.
        """
        from ..numerical import contract_cross_section

        D = self.L * self.M
        w, T = self._bogoliubov(q_cart)

        v = np.zeros((3, 2 * D), dtype=complex)
        for i in range(self.L):
            base = np.exp(1j * float(np.dot(q_cart, self.pos[i]))) if self.pos is not None \
                else 1.0
            sl, slb = slice(i * self.M, (i + 1) * self.M), \
                slice(D + i * self.M, D + (i + 1) * self.M)
            # q-dependent moment: sum over constituents k, each at cartesian offset d_k,
            # contributing e^{i q.(pos_i + d_k)} S_k^a. For a plain dipole this is one term
            # at d=0; for a dimer the e^{i q.d} phases build the STAGGERED operator that the
            # triplon couples to (the total-spin term alone would give zero intensity).
            for (d_k, idx) in self.moment_terms[i]:
                ph = base * np.exp(1j * float(np.dot(q_cart, d_k)))
                for a in range(3):
                    v[a, sl] += ph * self.t[i, idx[a]]
                    v[a, slb] += ph * self.tb[i, idx[a]]

        M = (v @ T)[:, D:]                        # (3, D): amplitude per band
        # Normalise PER SITE, as Sunny's ssf does (without this the intensities come out
        # a factor L too large -- a constant factor, easy to mistake for "close enough").
        spin_corr = np.einsum("am,bm->abm", M.conj(), M) / self.L

        inten, clamp = contract_cross_section(spin_corr, q_cart, cross_section)
        inten = np.real(inten)
        if clamp:
            inten[inten < 0] = 0.0

        if ion:
            from ..form_factors import get_form_factor
            inten = inten * get_form_factor(ion, float(np.linalg.norm(q_cart))) ** 2
        return w, inten
