"""High-order dimer series expansion (linked-cluster) with Dlog-Pade resummation.

The entangled-units engine (entangled.py) is HARMONIC bond-operator theory: exact in
the weak-interdimer limit (Cu5SbO6), but far off at strong coupling (Rb2Cu3SnF12,
J2 = 0.95 J1, where the observed triplon sits ~4x below the harmonic band). The
quantitative tool for that regime is the DIMER SERIES EXPANSION -- the method of
Matan et al., Nat. Phys. 6, 865 (2010) / PRB 89, 024414 (2014): expand the
one-triplon effective Hamiltonian in powers of the interdimer couplings via a
linked-cluster expansion, then resum with Dlog-Pade approximants.

Structure (all validated against independent oracles -- see tests/test_dimer_series.py):

  * `block_effective_series` -- numerical Rayleigh-Schrodinger / Bloch perturbation
    theory with the des Cloizeaux (canonical) Hermitization, for a quasi-degenerate
    model space over a DIAGONAL H0. Validated: its eigenvalues match exact
    diagonalization to O(lambda^{n+1}).
  * `eigenvalue_series` -- eigenvalue series of a Hermitian matrix series
    H(lambda) = sum_k H_k lambda^k with recursive degenerate-block handling.
  * `DimerSeriesModel` -- the physics: partition the spins into dimer units (exactly
    as `mode: entangled`), diagonalize each unit exactly (intra coupling + intra DM +
    Zeeman in H0), treat every interdimer coupling as the perturbation, and build the
    one-triplon hopping amplitudes by a linked-cluster expansion: enumerate connected
    link-clusters up to `order` links, compute each cluster's effective one-particle
    block by the PT engine, subtract proper connected subclusters (the linked-cluster
    theorem: the subtracted weight of an L-link cluster starts at order L -- asserted
    numerically), and Fourier-sum into H_eff(k) per order.
  * `pade` / `dlog_pade_estimates` / `resummed` -- Pade and Dlog-Pade resummation;
    Dlog-Pade returns the family of near-diagonal approximants (defective ones with a
    real pole in (0, lambda] are discarded) and the spread is the uncertainty, as in
    the papers.

Cluster additivity note: the one-particle effective Hamiltonian minus the ground-state
energy, both from the canonical (des Cloizeaux) transformation, is cluster additive
(Gelfand, Solid State Comm. 98, 11 (1996)): on a disconnected cluster the wave operator
factorizes, so cross-cluster hopping vanishes and the subtraction scheme is exact. The
engine ASSERTS this numerically -- after subtraction, every weight must vanish below
order L(cluster).
"""
import itertools
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .operators import spin_matrices
from .entangled import _embedded_spin_ops, _pair_matrix

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- PT engine
def block_effective_series(E0, V_list, P_idx, order):
    """Effective-Hamiltonian series for a (quasi-degenerate) model space.

    H(lambda) = diag(E0) + sum_{j>=1} V_list[j-1] lambda^j.  Returns
    [H_eff_0, ..., H_eff_order] (each p x p, Hermitian) such that the eigenvalues of
    sum_k H_eff_k lambda^k reproduce the exact eigenvalues connected to the model
    space to O(lambda^{order+1}).

    Bloch recursion with column-dependent resolvents (quasi-degenerate model space),
    then the des Cloizeaux Hermitization H_eff = S^{-1/2} (Omega^+ H Omega) S^{-1/2}.
    """
    E0 = np.asarray(E0, dtype=float)
    N = len(E0)
    P_idx = np.asarray(P_idx, dtype=int)
    p = len(P_idx)
    comp = np.ones(N, dtype=bool)
    comp[P_idx] = False
    eps = E0[P_idx]

    if comp.any():
        scale = max(float(np.abs(E0).max()), 1.0)
        dmin = np.abs(eps[None, :] - E0[comp, None]).min()
        if dmin < 1e-9 * scale:
            raise ValueError(
                f"model space is not separated from the complement "
                f"(min gap {dmin:.2e} on scale {scale:.2e}); the perturbative block "
                f"diagonalization is ill-defined.")

    jmax = len(V_list)
    Om = [np.zeros((N, p), dtype=complex)]
    Om[0][P_idx, np.arange(p)] = 1.0

    # complement resolvent per model column: 1 / (eps_mu - E0_q), zero on model rows
    inv_den = np.zeros((N, p))
    if comp.any():
        inv_den[comp, :] = 1.0 / (eps[None, :] - E0[comp, None])

    B = [np.diag(eps).astype(complex)]
    for k in range(1, order + 1):
        Y = np.zeros((N, p), dtype=complex)
        for j in range(1, min(k, jmax) + 1):
            Y += V_list[j - 1] @ Om[k - j]
        Bk = Y[P_idx, :].copy()                      # Bloch H_eff at order k
        B.append(Bk)
        for j in range(1, k):
            Y -= Om[j] @ B[k - j]
        Om.append(Y * inv_den)                       # zero on model rows by construction

    # S_k = (Omega^+ Omega)_k ; M_k = (Omega^+ H Omega)_k
    D = E0
    S, M = [], []
    for k in range(order + 1):
        Sk = np.zeros((p, p), dtype=complex)
        Mk = np.zeros((p, p), dtype=complex)
        for i in range(k + 1):
            Sk += Om[i].conj().T @ Om[k - i]
            Mk += Om[i].conj().T @ (D[:, None] * Om[k - i])
        for l in range(1, min(k, jmax) + 1):
            for i in range(k - l + 1):
                Mk += Om[i].conj().T @ (V_list[l - 1] @ Om[k - l - i])
        S.append(Sk)
        M.append(Mk)

    # G = S^{1/2}, T = S^{-1/2} as series (all Hermitian)
    G = [np.eye(p, dtype=complex)]
    for k in range(1, order + 1):
        acc = S[k].copy()
        for i in range(1, k):
            acc -= G[i] @ G[k - i]
        G.append(0.5 * acc)
    T = [np.eye(p, dtype=complex)]
    for k in range(1, order + 1):
        acc = np.zeros((p, p), dtype=complex)
        for j in range(1, k + 1):
            acc -= G[j] @ T[k - j]
        T.append(acc)

    Heff = []
    for k in range(order + 1):
        Hk = np.zeros((p, p), dtype=complex)
        for a in range(k + 1):
            for b in range(k - a + 1):
                Hk += T[a] @ M[b] @ T[k - a - b]
        Heff.append(0.5 * (Hk + Hk.conj().T))
    return Heff


def eigenvalue_series(H_list, order, tol=1e-9):
    """Eigenvalue series of the Hermitian matrix series H(lambda) = sum_k H_k lambda^k.

    Returns a list of n real arrays (order+1,), one per band, such that
    sum_k e_k lambda^k matches the exact eigenvalues to O(lambda^{order+1}).
    Degenerate H_0 blocks are folded with `block_effective_series` and split
    recursively at the first order that lifts them; bands that never split share a
    series (e.g. the DM-protected Stot^z = +/-1 doublet).
    """
    H_list = [np.asarray(H, dtype=complex) for H in H_list]
    n = H_list[0].shape[0]
    w, W = np.linalg.eigh(H_list[0])
    scale = max(float(np.abs(w).max()), 1.0)
    Hrot = [W.conj().T @ H @ W for H in H_list]

    groups, start = [], 0
    for i in range(1, n + 1):
        if i == n or abs(w[i] - w[i - 1]) > tol * scale:
            groups.append(list(range(start, i)))
            start = i

    out = []
    for g in groups:
        eps = float(w[g[0]])
        V_list = [Hrot[k] for k in range(1, min(len(Hrot) - 1, order) + 1)]
        Heff = block_effective_series(w, V_list, g, order)
        if len(g) == 1:
            series = np.zeros(order + 1)
            series[0] = eps
            for k in range(1, order + 1):
                series[k] = float(Heff[k][0, 0].real)
            out.append(series)
            continue
        A = [Heff[k].copy() for k in range(order + 1)]
        A[0] -= eps * np.eye(len(g))
        r = None
        for k in range(1, order + 1):
            if np.abs(A[k]).max() > tol * scale:
                r = k
                break
        if r is None:
            base = np.zeros(order + 1)
            base[0] = eps
            out.extend([base.copy() for _ in g])
            continue
        sub = eigenvalue_series([A[k] for k in range(r, order + 1)], order - r, tol)
        for ssub in sub:
            series = np.zeros(order + 1)
            series[0] = eps
            series[r:] += ssub[: order + 1 - r]
            out.append(series)
    return out


# ------------------------------------------------------------------- resummation
def pade(c, m, nden):
    """[m/nden] Pade of the series c (len >= m+nden+1). Returns (a, b) with b[0]=1."""
    c = np.asarray(c, dtype=float)
    if nden == 0:
        return c[: m + 1].copy(), np.array([1.0])
    A = np.zeros((nden, nden))
    rhs = np.zeros(nden)
    for k in range(1, nden + 1):
        for j in range(1, nden + 1):
            idx = m + k - j
            A[k - 1, j - 1] = c[idx] if idx >= 0 else 0.0
        rhs[k - 1] = -c[m + k]
    b = np.concatenate([[1.0], np.linalg.solve(A, rhs)])
    a = np.array([sum(b[j] * c[i - j] for j in range(min(i, nden) + 1))
                  for i in range(m + 1)])
    return a, b


def _has_pole(b, x):
    if len(b) <= 1:
        return False
    roots = np.roots(b[::-1])
    return bool(np.any((np.abs(roots.imag) < 1e-8) & (roots.real > 1e-12)
                       & (roots.real <= x * (1 + 1e-9))))


def dlog_pade_estimates(c, x=1.0):
    """Dlog-Pade estimates of f(x) from its series c (c[0] > 0 required).

    Pade-approximates u = d/dl ln f and integrates: f = c0 exp(int_0^x u). Returns the
    surviving near-diagonal approximants; the spread is the uncertainty (papers'
    convention). Empty list if every approximant is defective.
    """
    c = np.asarray(c, dtype=float)
    n = len(c) - 1
    if n < 2 or c[0] <= 0:
        return []
    u = np.zeros(n)
    for k in range(n):
        s = (k + 1) * c[k + 1] - sum(u[j] * c[k - j] for j in range(k))
        u[k] = s / c[0]
    xs = np.linspace(0.0, x, 2001)
    ests = []
    for nden in range(1, n):
        m = (n - 1) - nden
        if m < 0:
            continue
        try:
            a, b = pade(u, m, nden)
        except np.linalg.LinAlgError:
            continue
        if _has_pole(b, x):
            continue
        vals = np.polyval(a[::-1], xs) / np.polyval(b[::-1], xs)
        ests.append(float(c[0] * np.exp(np.trapz(vals, xs))))
    return ests


def resummed(c, x=1.0, method="dlog_pade"):
    """Evaluate the series c at x with the chosen resummation. Returns (value, spread).

    `sum` -- plain truncated sum (spread 0).  `pade` / `dlog_pade` -- median of the
    surviving approximants, spread = max - min; falls back to the plain sum (spread
    inf) when every approximant is defective, so a meaningless number is never
    silently returned.
    """
    c = np.asarray(c, dtype=float)
    plain = float(np.polyval(c[::-1], x))
    if method == "sum" or len(c) < 3:
        return plain, 0.0
    vals = []
    if method == "pade":
        n = len(c) - 1
        for nden in range(1, n):
            m = n - nden
            try:
                a, b = pade(c, m, nden)
            except np.linalg.LinAlgError:
                continue
            if _has_pole(b, x):
                continue
            vals.append(float(np.polyval(a[::-1], x) / np.polyval(b[::-1], x)))
    elif method == "dlog_pade":
        vals = dlog_pade_estimates(c, x)
    else:
        raise ValueError(f"unknown resummation {method!r}")
    if not vals:
        return plain, float("inf")
    v = np.asarray(vals)
    return float(np.median(v)), float(v.max() - v.min())


# ----------------------------------------------------------------- dimer lattice
def _embed_pair_op(Vp, pa, pb, m, nloc):
    """Embed the two-site operator Vp (nloc^2 x nloc^2, index sa*nloc+sb) acting on
    cluster positions pa != pb into the nloc^m product space (little-endian digits)."""
    nz = np.argwhere(np.abs(Vp) > 1e-14)
    if len(nz) == 0:
        return sp.csr_matrix((nloc ** m, nloc ** m), dtype=complex)
    rows_ab, cols_ab = nz[:, 0], nz[:, 1]
    sa, sb = rows_ab // nloc, rows_ab % nloc
    ta, tb = cols_ab // nloc, cols_ab % nloc
    vals = Vp[rows_ab, cols_ab]

    other = [q for q in range(m) if q not in (pa, pb)]
    rest = nloc ** len(other)
    o = np.arange(rest)
    base = np.zeros(rest, dtype=np.int64)
    for idx, q in enumerate(other):
        base += ((o // nloc ** idx) % nloc) * nloc ** q
    row = (base[:, None] + sa[None, :] * nloc ** pa + sb[None, :] * nloc ** pb).ravel()
    col = (base[:, None] + ta[None, :] * nloc ** pa + tb[None, :] * nloc ** pb).ravel()
    data = np.broadcast_to(vals[None, :], (rest, len(vals))).ravel()
    return sp.csr_matrix((data, (row, col)), shape=(nloc ** m, nloc ** m))


def _cluster_pt(levels_list, links_local, order):
    """Ground-state and one-particle effective series on one cluster.

    levels_list: local eigen-energies (nloc,) per cluster node (ascending, [0] = gs).
    links_local: [(pa, pb, Vloc)] with pa < pb cluster-node positions, Vloc the
    (nloc^2 x nloc^2) coupling in the two nodes' LOCAL EIGENBASES.

    Returns (gs_int (order+1,), delta_int (order+1, m*nexc, m*nexc)):
    the INTERACTION parts -- local energies subtracted -- which are the
    cluster-additive quantities the linked-cluster subtraction acts on.
    """
    m = len(levels_list)
    nloc = len(levels_list[0])
    nexc = nloc - 1
    dims = nloc ** m

    E0 = np.zeros(dims)
    idx = np.arange(dims)
    for d in range(m):
        E0 += np.asarray(levels_list[d])[(idx // nloc ** d) % nloc]

    V = sp.csr_matrix((dims, dims), dtype=complex)
    for (pa, pb, Vloc) in links_local:
        V = V + _embed_pair_op(Vloc, pa, pb, m, nloc)

    e_loc = sum(float(l[0]) for l in levels_list)
    Heff_g = block_effective_series(E0, [V], [0], order)
    gs = np.array([float(Heff_g[k][0, 0].real) for k in range(order + 1)])
    gs_int = gs.copy()
    gs_int[0] -= e_loc

    onep = [alpha * nloc ** d for d in range(m) for alpha in range(1, nloc)]
    Heff_1 = block_effective_series(E0, [V], onep, order)
    p = len(onep)
    delta = np.array([Heff_1[k] - gs[k] * np.eye(p) for k in range(order + 1)])
    # subtract the purely local order-0 splittings -> delta_int order 0 == 0
    loc0 = np.array([levels_list[d][alpha] - levels_list[d][0]
                     for d in range(m) for alpha in range(1, nloc)])
    delta[0] -= np.diag(loc0)
    return gs_int, delta


class DimerSeriesModel:
    """Linked-cluster dimer series expansion for a lattice of entangled units."""

    def __init__(self, lat, levels, positions, link_types, nloc):
        self.lat = np.asarray(lat, dtype=float)
        self.levels = [np.asarray(l, dtype=float) for l in levels]   # per dimer (nloc,)
        self.pos = np.asarray(positions, dtype=float)                # cartesian centroids
        # link_types: list of (u, v, off(3-int-tuple), Vloc (nloc^2 x nloc^2))
        self.link_types = link_types
        self.nloc = int(nloc)
        self.nexc = self.nloc - 1
        self.D = len(self.levels)
        self._touch = defaultdict(list)   # d -> [(t, shift)]: node (d,c) in inst (t, c-shift)
        for t, (u, v, off, _V) in enumerate(self.link_types):
            self._touch[u].append((t, (0, 0, 0)))
            self._touch[v].append((t, off))
        self._t_cache = {}                # order -> hopping dict
        self._w_memo = {}                 # canonical cluster -> (nodes, gs_int, delta_int)

    # ------------------------------------------------------------- constructors
    @classmethod
    def from_spin_arrays(cls, lat, spin_S, pos_frac, bonds, units, field=None):
        """Direct constructor. `bonds`: [(i, j, off3, M(3x3))], each physical spin pair
        ONCE (i -> j at cell offset off).  `units`: as in mode entangled -- lists of
        site indices, per-member cell offsets allowed via [i, [j, [ox,oy,oz]]]."""
        directed = [(int(i), int(j), tuple(int(x) for x in off), np.asarray(M, float), 1.0)
                    for (i, j, off, M) in bonds]
        return cls._build(np.asarray(lat, float), [float(s) for s in spin_S],
                          np.asarray(pos_frac, float), directed, units, field)

    @classmethod
    def from_generic_model(cls, model, params=None, units=None):
        """Build from a GenericSpinModel exactly as `mode: entangled` does."""
        params = list(params if params is not None else [])
        Jex, DM, Kex = model.spin_interactions(params)
        apos = np.asarray(model.atom_pos(), dtype=float)
        aouc = np.asarray(model.atom_pos_ouc(), dtype=float)
        lat = np.asarray(model.unit_cell(), dtype=float)
        inv_lat = np.linalg.inv(lat)
        L = len(apos)
        spins = [float(s) for s in model.spin_magnitudes()]
        pos_frac = apos @ inv_lat

        directed = []
        for i in range(L):
            for j in range(len(aouc)):
                M = _pair_matrix(Jex, DM, Kex, i, j)
                if not np.any(M):
                    continue
                jc = j % L
                off = tuple(int(x) for x in
                            np.round((aouc[j] - apos[jc]) @ inv_lat).astype(int))
                directed.append((i, jc, off, M, 0.5))   # both directions listed -> 1/2

        field = None
        try:
            from .. import spiral_opt as _so
            field = _so._resolve_field(model, params)
        except Exception:
            field = None
        return cls._build(lat, spins, pos_frac, directed, units, field)

    @classmethod
    def _build(cls, lat, spins, pos_frac, directed, units, field):
        if units is None:
            raise ValueError("DimerSeriesModel needs a `units` partition.")

        def _member(mm):
            if isinstance(mm, (list, tuple)) and len(mm) == 2 \
                    and isinstance(mm[1], (list, tuple)):
                return int(mm[0]), tuple(int(x) for x in mm[1])
            return int(mm), (0, 0, 0)

        units = [[_member(mm) for mm in u] for u in units]
        L = len(spins)
        flat = sorted(s for u in units for (s, _o) in u)
        if flat != list(range(L)):
            raise ValueError(f"`units` must partition all {L} sites exactly once.")
        U = len(units)
        role = {s: pp for u in units for pp, (s, _o) in enumerate(u)}
        shift = {s: np.array(o, int) for u in units for (s, o) in u}
        unit_of = {s: k for k, u in enumerate(units) for (s, _o) in u}

        # embedded raw spin operators + frame positions per unit
        emb, centroids, nlocs = [], [], []
        for u in units:
            s_list = [spins[s] for (s, _o) in u]
            e = _embedded_spin_ops(s_list)
            emb.append(e)
            rpos = [pos_frac[s] @ lat + np.array(o, float) @ lat for (s, o) in u]
            centroids.append(np.mean(rpos, axis=0))
            nlocs.append(e[0][0].shape[0])
        if len(set(nlocs)) != 1:
            raise NotImplementedError("all units must share the same local dimension.")
        nloc = nlocs[0]

        # intra-unit Hamiltonians and inter-unit directed pair terms
        A = [np.zeros((nloc, nloc), dtype=complex) for _ in range(U)]
        inter = defaultdict(list)   # (u, v, cell) -> [(role_i, role_j, w*M)]
        for (i, j, off, M, w) in directed:
            ui, uj = unit_of[i], unit_of[j]
            cell = np.array(off, int) - shift[j] + shift[i]
            if ui == uj and np.all(cell == 0):
                Si, Sj = emb[ui][role[i]], emb[ui][role[j]]
                for a in range(3):
                    for b in range(3):
                        if M[a, b] != 0.0:
                            A[ui] += w * M[a, b] * (Si[a] @ Sj[b])
                continue
            inter[(ui, uj, tuple(int(x) for x in cell))].append((role[i], role[j], w * M))

        # Zeeman on each unit (gamma = 2 convention, matching the engines)
        if field is not None and np.linalg.norm(field) > 0:
            MU_B, GAMMA = 5.788e-2, 2.0
            for k, u in enumerate(units):
                Svec = [sum(emb[k][pp][a] for pp in range(len(u))) for a in range(3)]
                A[k] += GAMMA * MU_B * sum(field[a] * Svec[a] for a in range(3))

        # local eigenbases; rotate member operators
        levels, rot_ops = [], []
        for k in range(U):
            wv, Uk = np.linalg.eigh(0.5 * (A[k] + A[k].conj().T))
            levels.append(wv)
            rot_ops.append([[Uk.conj().T @ emb[k][pp][a] @ Uk for a in range(3)]
                            for pp in range(len(units[k]))])

        # merge directed inter-unit terms into canonical unordered links
        Vmap = {}
        for (ui, uj, cell), terms in inter.items():
            neg = tuple(-x for x in cell)
            k1, k2 = (ui, uj, cell), (uj, ui, neg)
            swap = k2 < k1
            key = k2 if swap else k1
            V = Vmap.setdefault(key, np.zeros((nloc * nloc, nloc * nloc), dtype=complex))
            for (ri, rj, M) in terms:
                for a in range(3):
                    for b in range(3):
                        if M[a, b] != 0.0:
                            Oa, Ob = rot_ops[ui][ri][a], rot_ops[uj][rj][b]
                            V += M[a, b] * (np.kron(Ob, Oa) if swap else np.kron(Oa, Ob))
        link_types = [(u, v, off, V) for (u, v, off), V in Vmap.items()
                      if np.abs(V).max() > 1e-12]
        logger.info("DimerSeriesModel: %d units (N=%d), %d link types/cell.",
                    U, nloc, len(link_types))
        return cls(lat, levels, np.array(centroids), link_types, nloc)

    # -------------------------------------------------------------- enumeration
    def _inst_nodes(self, inst):
        t, c = inst
        u, v, off, _ = self.link_types[t]
        return ((u, c), (v, (c[0] + off[0], c[1] + off[1], c[2] + off[2])))

    def _canonical(self, cluster):
        """Canonical (translation-normalized) form of a set of link instances.
        Returns (form, tau): form = sorted tuple of (t, cell - tau)."""
        cand = set()
        for inst in cluster:
            for (_d, c) in self._inst_nodes(inst):
                cand.add(c)
        best, best_tau = None, None
        for tau in cand:
            form = tuple(sorted(
                (t, (c[0] - tau[0], c[1] - tau[1], c[2] - tau[2])) for (t, c) in cluster))
            if best is None or form < best:
                best, best_tau = form, tau
        return best, best_tau

    def _neighbors(self, cluster):
        nodes = set()
        for inst in cluster:
            nodes.update(self._inst_nodes(inst))
        out = set()
        for (d, c) in nodes:
            for (t, sh) in self._touch[d]:
                inst = (t, (c[0] - sh[0], c[1] - sh[1], c[2] - sh[2]))
                if inst not in cluster:
                    out.add(inst)
        return out

    def _connected(self, links):
        if not links:
            return False
        links = list(links)
        seen = {0}
        nodes = set(self._inst_nodes(links[0]))
        grew = True
        while grew:
            grew = False
            for idx, inst in enumerate(links):
                if idx in seen:
                    continue
                a, b = self._inst_nodes(inst)
                if a in nodes or b in nodes:
                    seen.add(idx)
                    nodes.update((a, b))
                    grew = True
        return len(seen) == len(links)

    def _enumerate(self, order):
        levels = {1: {self._canonical(frozenset({(t, (0, 0, 0))}))[0]
                      for t in range(len(self.link_types))}}
        for size in range(2, order + 1):
            new = set()
            for form in levels[size - 1]:
                cl = frozenset(form)
                for nb in self._neighbors(cl):
                    new.add(self._canonical(cl | {nb})[0])
            levels[size] = new
        return levels

    # ------------------------------------------------------------------ weights
    def _weight(self, form, order):
        """Subtracted (linked) weight of a canonical cluster. Memoized per order."""
        if (form, order) in self._w_memo:
            return self._w_memo[(form, order)]
        cluster = list(form)
        nodes = sorted({n for inst in cluster for n in self._inst_nodes(inst)})
        node_pos = {n: i for i, n in enumerate(nodes)}
        m = len(nodes)
        nexc = self.nexc

        links_local = []
        for (t, c) in cluster:
            u, v, off, V = self.link_types[t]
            na = (u, c)
            nb = (v, (c[0] + off[0], c[1] + off[1], c[2] + off[2]))
            pa, pb = node_pos[na], node_pos[nb]
            if pa > pb:
                # re-express V (built on kron(u, v)) on kron(node_pb_first) ordering
                Vsw = V.reshape(self.nloc, self.nloc, self.nloc, self.nloc)
                Vsw = Vsw.transpose(1, 0, 3, 2).reshape(self.nloc ** 2, self.nloc ** 2)
                links_local.append((pb, pa, Vsw))
            else:
                links_local.append((pa, pb, V))

        lv = [self.levels[d] for (d, _c) in nodes]
        gs_raw, delta_raw = _cluster_pt(lv, links_local, order)

        # subtract proper connected subclusters
        Lc = len(cluster)
        for r in range(1, Lc):
            for subset in itertools.combinations(cluster, r):
                if not self._connected(frozenset(subset)):
                    continue
                sform, tau = self._canonical(frozenset(subset))
                s_nodes, s_gs, s_delta = self._weight(sform, order)
                gs_raw = gs_raw - s_gs
                # map subcluster nodes (canonical frame) back into this cluster's frame
                idx_map = []
                for (d, c) in s_nodes:
                    n_here = (d, (c[0] + tau[0], c[1] + tau[1], c[2] + tau[2]))
                    idx_map.append(node_pos[n_here])
                rows = np.concatenate([[i * nexc + a for a in range(nexc)]
                                       for i in idx_map])
                srows = np.arange(len(s_nodes) * nexc)
                for k in range(order + 1):
                    delta_raw[k][np.ix_(rows, rows)] -= s_delta[k][np.ix_(srows, srows)]

        # linked-cluster theorem check: weight vanishes below order L
        low = max((np.abs(delta_raw[k]).max() for k in range(min(Lc, order + 1))),
                  default=0.0)
        scale = max(np.abs(delta_raw).max(), 1.0)
        if low > 1e-7 * scale:
            logger.warning("cluster additivity violated for %s (below-order weight "
                           "%.2e vs scale %.2e)", form, low, scale)
        out = (tuple(nodes), gs_raw, delta_raw)
        self._w_memo[(form, order)] = out
        return out

    # ----------------------------------------------------------------- assembly
    def hopping_series(self, order):
        """Bulk one-triplon hoppings: {(dA, dB, off3): (order+1, nexc, nexc) complex},
        plus the order-0 local splittings handled in h_series_at. Cached per order."""
        if order in self._t_cache:
            return self._t_cache[order]
        levels = self._enumerate(order)
        n_cl = sum(len(v) for v in levels.values())
        logger.info("dimer series order %d: %d clusters (%s)", order, n_cl,
                    ", ".join(f"L{k}:{len(v)}" for k, v in sorted(levels.items())))
        t = defaultdict(lambda: np.zeros((order + 1, self.nexc, self.nexc), complex))
        nexc = self.nexc
        for size in sorted(levels):
            for form in levels[size]:
                nodes, _gs, delta = self._weight(form, order)
                for ia, (dA, cA) in enumerate(nodes):
                    for ib, (dB, cB) in enumerate(nodes):
                        off = (cB[0] - cA[0], cB[1] - cA[1], cB[2] - cA[2])
                        blk = delta[:, ia * nexc:(ia + 1) * nexc,
                                    ib * nexc:(ib + 1) * nexc]
                        if np.abs(blk).max() > 1e-14:
                            t[(dA, dB, off)] += blk
        t = dict(t)
        self._t_cache[order] = t
        return t

    def ground_state_energy(self, order):
        """Ground-state energy per unit cell: series (order+1,)."""
        self.hopping_series(order)   # populate memo
        levels = self._enumerate(order)
        gs = np.zeros(order + 1)
        gs[0] = sum(float(l[0]) for l in self.levels)
        for size in sorted(levels):
            for form in levels[size]:
                _n, g, _d = self._w_memo[(form, order)]
                gs += g
        return gs

    def h_series_at(self, q_cart, order):
        """One-triplon effective Hamiltonian series [H_0(k) .. H_order(k)]."""
        t = self.hopping_series(order)
        nb = self.D * self.nexc
        H = [np.zeros((nb, nb), dtype=complex) for _ in range(order + 1)]
        for d in range(self.D):
            for a in range(self.nexc):
                H[0][d * self.nexc + a, d * self.nexc + a] = \
                    self.levels[d][a + 1] - self.levels[d][0]
        q = np.asarray(q_cart, dtype=float)
        for (dA, dB, off), W in t.items():
            dr = self.pos[dB] + np.array(off, float) @ self.lat - self.pos[dA]
            ph = np.exp(1j * float(q @ dr))
            ra = slice(dA * self.nexc, (dA + 1) * self.nexc)
            rb = slice(dB * self.nexc, (dB + 1) * self.nexc)
            for k in range(order + 1):
                H[k][ra, rb] += ph * W[k]
        for k in range(order + 1):
            herm = np.abs(H[k] - H[k].conj().T).max()
            if herm > 1e-7 * max(np.abs(H[k]).max(), 1.0):
                logger.warning("H_%d(k) not Hermitian to tolerance (%.2e)", k, herm)
            H[k] = 0.5 * (H[k] + H[k].conj().T)
        return H

    def band_series(self, q_cart, order):
        """Per-band eigenvalue series at q: array (n_bands, order+1), sorted by e_0."""
        H = self.h_series_at(q_cart, order)
        series = eigenvalue_series(H, order)
        return np.array(sorted(series, key=lambda s: (s[0], s[1] if len(s) > 1 else 0)))

    def dispersion(self, q_cart, order, resum="dlog_pade"):
        """Resummed one-triplon energies at q. Returns (values sorted, spreads)."""
        bands = self.band_series(q_cart, order)
        vals, spreads = [], []
        for s in bands:
            v, dv = resummed(s, 1.0, method=resum)
            vals.append(v)
            spreads.append(dv)
        o = np.argsort(vals)
        return np.asarray(vals)[o], np.asarray(spreads)[o]
