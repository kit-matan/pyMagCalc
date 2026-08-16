"""Finite-temperature classical spin dynamics and SampledCorrelations S(q,ω).

LSWT gives S(q,ω) as an expansion about the ORDERED ground state; above T_N, or when
you want the full thermal lineshape (damping, multi-magnon continuum, paramagnons),
the classical route is real-time spin dynamics on a thermally sampled ensemble — the
`SampledCorrelations` idea (Sunny): thermalize, evolve the deterministic
Landau–Lifshitz equation, and Fourier transform the trajectory in space and time.

Landau–Lifshitz (undamped, microcanonical measurement):

    dS_i/dt = − S_i × B_i,    B_i = ∂E/∂S_i = (H S)_i + b_i,

with the SAME classical energy E = ½ SᵀH S + bᵀS as `thermal_mc`/`annealing`. This
conserves energy and |S_i|, and its small-amplitude normal modes ARE the spin-wave
frequencies (for a ferromagnet, exactly the LSWT dispersion). Thermal states are
drawn by Metropolis (`thermal_mc`), then evolved; averaging |Σ_r e^{-iq·r} S_r(t)|²
over trajectories gives

    S(q,ω) = ⟨ |S(q,ω)|² ⟩ / (N_t N),   S(q,ω) = Σ_t e^{iωt} Σ_r e^{-iq·r} S_r(t).

Validated (tests/test_classical_dynamics.py) against exact/independent results:
  * a single spin in a field precesses at the Larmor frequency ω = gμ_B B (the S(0,ω)
    peak) — pins the integrator, the time axis and the frequency convention;
  * the undamped integrator conserves energy to O(dt⁴) (RK4);
  * for a Heisenberg ferromagnet the low-T S(q,ω) peak positions fall on the exact
    LSWT magnon dispersion the pyMagCalc engine computes — tying the dynamics to the
    validated spin-wave engine.
"""
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def local_field(H, b, m):
    """B_i = (H m)_i + b_i, shape (N, 3)."""
    return (H @ m.ravel() + b).reshape(-1, 3)


def _deriv(H, b, m):
    return -np.cross(m, local_field(H, b, m))


def llg_rk4_step(H, b, m, dt, S):
    """One RK4 step of dS/dt = −S×B, renormalizing |S_i|=S (removes O(dt⁵) drift)."""
    k1 = _deriv(H, b, m)
    k2 = _deriv(H, b, m + 0.5 * dt * k1)
    k3 = _deriv(H, b, m + 0.5 * dt * k2)
    k4 = _deriv(H, b, m + dt * k3)
    m = m + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    m *= S / np.linalg.norm(m, axis=1, keepdims=True)
    return m



def implicit_midpoint_step(H, b, m, dt, S, tol=1e-12, max_iter=50):
    """One step of dS/dt = -S x B by the IMPLICIT MIDPOINT rule (Sunny's
    `ImplicitMidpoint`).

        S_{n+1} = S_n + dt * f((S_n + S_{n+1}) / 2)

    solved by fixed-point iteration. Unlike RK4 this is symplectic: it conserves the
    energy and every |S_i| to the solver tolerance rather than to O(dt^4), so the
    error stays BOUNDED over a long trajectory instead of drifting secularly. That
    matters for S(q,w), where a slow energy drift smears the very peaks being
    measured.

    Note no renormalization is applied afterwards: the midpoint rule preserves |S_i|
    exactly (the update is orthogonal to the midpoint spin), and rescaling would mask
    a genuine solver failure.
    """
    m_new = m.copy()
    for _ in range(max_iter):
        mid = 0.5 * (m + m_new)
        cand = m + dt * _deriv(H, b, mid)
        delta = float(np.abs(cand - m_new).max())
        m_new = cand
        if delta < tol:
            break
    else:
        logger.warning(
            f"implicit midpoint did not converge in {max_iter} iterations "
            f"(last step {delta:.2e}); reduce dt.")
    return m_new


def suggest_timestep(H, b, m, S, target=0.033):
    """A time step resolving the fastest precession (Sunny `suggest_timestep`).

    The fastest frequency in the system is set by the largest local field,
    omega_max ~ |B_i|_max, so dt = target / omega_max with `target` a fraction of a
    radian per step (Sunny's default accuracy knob is comparable). Returns a float in
    the same time units the rest of this module uses.
    """
    B = local_field(H, b, m)
    w_max = float(np.abs(B).max())
    if w_max <= 0:
        raise ValueError(
            "suggest_timestep: the local field vanishes everywhere, so there is no "
            "timescale to resolve (is the Hamiltonian empty?).")
    return float(target) / w_max


def langevin_step(H, b, m, dt, S, kT, damping, rng):
    """One step of the stochastic Landau-Lifshitz (Langevin) equation.

        dS/dt = -S x G + (lambda/S) S x (S x G) + noise,    G = dE/dS

    MIND THE SIGN. `local_field` returns the energy GRADIENT G = dE/dS, not the
    physical field B = -G, and the module's precession term is written -S x G to
    match. The damping must then be +(lambda/S) S x (S x G): using -(...) relaxes
    spins AWAY from the energy minimum, which shows up as a magnetization of the
    right magnitude and the WRONG SIGN -- plausible enough to miss without an exact
    reference. Check: for E = -h S^z (so G = -h zhat) and S along x,
    S x (S x G) = +S^2 h zhat, i.e. toward +z, which is the minimum.

    integrated by Heun's method. The noise amplitude is fixed by the
    fluctuation-dissipation theorem, so the stationary distribution is Boltzmann at
    `kT` -- which is the whole point, and what the tests check rather than assume.

    This is a THERMOSTAT (Sunny's `Langevin`), an alternative to Metropolis for
    preparing thermal states. Measurement trajectories should still be run
    undamped/deterministic; damping distorts the lineshape.
    """
    lam = float(damping)
    beta = 1.0 / float(kT)
    # <xi_a(t) xi_b(t')> = 2 lambda kT delta_ab delta(t-t') / S  (FDT)
    sigma = np.sqrt(2.0 * lam * float(kT) * dt / float(S))

    def drift(x):
        B = local_field(H, b, x)
        return -np.cross(x, B) + (lam / float(S)) * np.cross(x, np.cross(x, B))

    noise = sigma * rng.standard_normal(m.shape)
    f0 = drift(m)
    pred = m + dt * f0 - np.cross(m, noise)
    f1 = drift(pred)
    out = m + 0.5 * dt * (f0 + f1) - np.cross(0.5 * (m + pred), noise)
    # |S| is conserved by the continuum equation but not by a finite Heun step
    out *= float(S) / np.linalg.norm(out, axis=1, keepdims=True)
    return out


def langevin_thermalize(H, b, S, m, kT, dt, n_steps, damping=0.1, seed=0):
    """Run the Langevin thermostat for `n_steps`, returning the final configuration."""
    rng = np.random.default_rng(seed)
    for _ in range(int(n_steps)):
        m = langevin_step(H, b, m, dt, S, kT, damping, rng)
    return m


def evolve(H, b, S, m0, dt, n_steps, record_every=1, integrator="rk4"):
    """Deterministic LL trajectory. Returns (n_rec, N, 3) recorded configurations.

    `integrator`: "rk4" (default, historical) or "midpoint" (implicit midpoint --
    symplectic, bounded energy error over long runs).
    """
    step_fn = {"rk4": llg_rk4_step,
               "midpoint": implicit_midpoint_step}.get(str(integrator).lower())
    if step_fn is None:
        raise ValueError(
            f"integrator must be 'rk4' or 'midpoint', got {integrator!r}.")
    m = m0.copy()
    rec = []
    for step in range(n_steps):
        if step % record_every == 0:
            rec.append(m.copy())
        m = step_fn(H, b, m, dt, S)
    return np.array(rec)


def energy(H, b, m):
    mr = m.ravel()
    return 0.5 * float(mr @ (H @ mr)) + float(b @ mr)


@dataclass
class DynamicsResult:
    q_vectors: np.ndarray        # (Nq, 3) cartesian
    energies: np.ndarray         # (Nω,) meV
    sqw: np.ndarray              # (Nω, Nq) S(q,ω)
    temperature: float
    classical_to_quantum: bool = False   # was the c2q factor already applied?


def lag_window(n_t, window="rectangular"):
    """The window w(Δt) applied to the time CORRELATION before the ω transform.

    Returns None for `rectangular` (no window), else an (n_t,) array indexed the way
    `np.fft.fftfreq` indexes lags: w[k] is the weight of lag Δt = k (k < n_t/2) and
    of Δt = k − n_t (k > n_t/2), so it is symmetric in |Δt| by construction.

    `cosine` is Sunny's window (`add_sample!(...; window=:cosine)`, its default;
    NOT pyMagCalc's -- see "WHY IT IS OPT-IN" below, which is a measurement):

        w(Δt) = cos²(π|Δt|/n_t),   1 at Δt = 0, 0 at |Δt| = n_t/2.

    WHY A WINDOW AT ALL. Truncating a trajectory at length T multiplies C(Δt) by a
    rectangular window, whose transform is a Dirichlet kernel with sidelobes decaying
    only as 1/(ω−ω₀)². That is normally harmless — but the classical-to-quantum
    factor `classical_to_quantum_factor` grows LINEARLY in ω all the way to the
    Nyquist frequency π/dt (157 meV at the default dt = 0.02), so leakage from a
    4 meV band gets amplified across a 40× wider axis: measured at **+16 % on the
    whole-axis integral** of a gapped ferromagnetic chain, against +1.5 % when
    integrating ±1 meV about the band.

    WHAT IT COSTS, exactly and not approximately. cos²(x) = ½ + ¼e^{2ix} + ¼e^{−2ix},
    so multiplying the correlation by this window is IDENTICAL to convolving the
    spectrum with the 3-point kernel [¼, ½, ¼] — one bin Δω = 2π/T of Hann
    broadening, no more (`tests/test_classical_window.py` pins that identity). Two
    consequences worth relying on: the kernel is non-negative, so a windowed S(q,ω)
    cannot go negative where the raw periodogram was positive; and it sums to 1, so
    **the two-sided ω-integral, hence every sum rule, is preserved EXACTLY** — which
    is also why `tests/test_classical_absolute_normalization.py` cannot see this
    change and a separate oracle was needed for it.

    **WHY IT IS OPT-IN HERE, AND SUNNY'S DEFAULT THERE.** Because that same one-bin
    smear lands on the ELASTIC LINE, and `classical_to_quantum_factor` then amplifies
    what it moved. An ordered magnet at kT << ω carries a huge delta at ω = 0; c2q is
    1 there and |ω|/kT one bin away, which at kT = 0.005 and Δω = 0.153 meV is **31**.
    Measured on the gapped ferromagnetic chain of
    `tests/test_classical_absolute_normalization.py` (L = 24, q = 0.15, LSWT band
    sum 0.5):

        window        first inelastic bin, c2q-corrected
        rectangular   0.00006
        cosine        9.10          <- 18x the entire band sum, from ONE bin

    So switching the default on would trade a ~16 % leakage error for a
    catastrophically wrong low-energy bin on exactly the ordered magnets this engine
    is normally pointed at. The combination that works is
    `window: cosine` WITH `subtract_elastic: true`, which removes the delta before it
    can be smeared; use it together, and read `two_sided=True` integrals if you want
    the sum rule back (the one-sided integral drops the ω < 0 half of what the window
    moved out of the ω = 0 bin). Asking for one without the other where it matters is
    reported by `check_elastic_leakage` rather than left to this docstring.
    """
    name = str(window or "rectangular").lower()
    if name in ("rectangular", "none", "off", "false"):
        return None
    if name != "cosine":
        raise ValueError(
            f"window must be 'cosine' or 'rectangular', got {window!r}.")
    k = np.arange(int(n_t))
    return np.cos(np.pi * k / int(n_t))**2


def _window_correlation(tensor, win):
    """Apply the lag window to a (n_ω, …) spectrum: transform to the lag domain,
    multiply, transform back (the Blackman–Tukey form, and Sunny's).

    Windowing the CORRELATION rather than tapering the trajectory is what keeps the
    sum rule exact: the ω-integral is C(Δt = 0), and w(0) = 1.
    """
    if win is None:
        return tensor
    C = np.fft.ifft(tensor, axis=0)
    return np.fft.fft(C * win.reshape((-1,) + (1,) * (tensor.ndim - 1)), axis=0)


def dynamical_structure_factor(traj, pos, q_cart, dt, cross_section="perp", *,
                               n_cells, two_sided=False, window="rectangular",
                               subtract_elastic=False):
    """S(q,ω) from one trajectory `traj` (n_t, N, 3) at cartesian q-vectors.

    Returns (energies, sqw (n_ω, n_q)) on the ABSOLUTE scale LSWT and Sunny use,

        S^ab(q,ω) = (1/2π) ∫dt e^{-iωt} ⟨S^a(q,0)* S^b(q,t)⟩ / n_cells,
        S^a(q,t)  = Σ_r e^{-iq·r} S^a_r(t),

    whose defining property is the equal-time sum rule ∫dω S^ab(q,ω) =
    ⟨S^a(q)* S^b(q)⟩ / n_cells. `n_cells` is the number of CHEMICAL cells in the
    simulated box (Sunny's `prod(sys.dims)`), NOT the number of spins: LSWT S(q,ω)
    is per chemical cell in both codes, so a cell with two magnetic atoms scatters
    twice as much. It is keyword-only and has no default deliberately — the two
    differ only when the cell holds more than one site, i.e. silently on exactly the
    models a one-site test cannot reach.

    `two_sided` returns the whole frequency axis (ascending, ω < 0 included) rather
    than the ω ≥ 0 half; the classical spectrum is symmetric in ω, and it is the
    two-sided integral that the sum rule above is a statement about. The one-sided
    default is what `classical_to_quantum_factor` turns into a quantum S(q,ω ≥ 0).

    Both normalizing factors were missing until 2026-08-13: the FFT was left as a
    bare sum (so the result was 2π/dt too large — 314× at the default dt = 0.02) and
    the spatial sum was divided by the site count rather than the cell count.

    `window` tapers the time correlation before the ω transform. The default
    `rectangular` is the un-windowed behaviour this function has always had; `cosine`
    is Sunny's, and `lag_window` documents both what it fixes (leakage, ~16 % on the
    whole-axis integral) and the measured reason it is NOT the default here (it
    smears the elastic line into a bin the c2q factor then multiplies by ~31).

    `subtract_elastic` removes the time-average of S(q, t), i.e. the ELASTIC line, so
    the window has no delta to smear. Off by default, matching Sunny and the SU(N)
    path (`sun/dynamics.sampled_correlations`, which has had this option all along);
    `window: cosine` + `subtract_elastic: true` is the combination that behaves.
    """
    n_t, N, _ = traj.shape
    if n_cells is None or int(n_cells) <= 0:
        raise ValueError(f"n_cells must be a positive cell count, got {n_cells!r}.")
    qs = np.asarray(q_cart, float).reshape(-1, 3)
    phase = np.exp(-1j * (qs @ pos.T))                  # (n_q, N)
    # S^a(q, t) = Σ_r e^{-iq·r} S^a_r(t)  -> (n_t, n_q, 3)
    Sqt = np.einsum("qr,tra->tqa", phase, traj)
    if subtract_elastic:
        Sqt = Sqt - Sqt.mean(axis=0, keepdims=True)
    Sqw = np.fft.fft(Sqt, axis=0)                       # (n_t, n_q, 3)
    energies = 2 * np.pi * np.fft.fftfreq(n_t, d=dt)
    # |FFT|²/n_t is the DISCRETE spectral sum Σ_Δt C(Δt) e^{-iωΔt}; the continuous
    # transform in the definition above needs the sampling interval and the 1/2π.
    norm = dt / (2 * np.pi * n_t * int(n_cells))
    win = lag_window(n_t, window)
    sl = slice(None) if two_sided else slice(0, n_t // 2)
    out = np.zeros((n_t if two_sided else n_t // 2, len(qs)))
    for iq, q in enumerate(qs):
        # the window acts on the WHOLE (periodic) frequency axis, so it is applied
        # before the one-sided slice -- windowing a half-axis would wrap the lag
        # domain onto the wrong period.
        tensor = np.einsum("wa,wb->wab", Sqw[:, iq, :].conj(), Sqw[:, iq, :])
        tensor = _window_correlation(tensor, win)[sl]
        out[:, iq] = np.real(_contract(tensor, q, cross_section)) * norm
    if two_sided:
        order = np.argsort(energies)
        return energies[order], out[order]
    return energies[sl], out


def _contract(tensor, q, cross_section):
    """(n_ω,) neutron contraction of a (n_ω,3,3) correlation tensor at q."""
    cs = (cross_section or "perp").lower()
    if cs == "trace":
        return np.einsum("waa->w", tensor)
    if cs in ("xx", "yy", "zz"):
        a = {"xx": 0, "yy": 1, "zz": 2}[cs]
        return tensor[:, a, a]
    qn = np.linalg.norm(q)
    P = np.eye(3) if qn < 1e-12 else np.eye(3) - np.outer(q / qn, q / qn)
    return np.einsum("ab,wab->w", P, tensor)



def classical_to_quantum_factor(energies, kT):
    """|w/kT| / (1 - exp(-w/kT)) -- the classical-to-quantum correspondence factor.

    Classical spin dynamics equipartitions: every mode carries kT, so the classical
    S(q,w) is NOT on the quantum intensity scale. The factor below maps it onto one,
    and is what Sunny applies in `intensities(sc, ...; kT)`
    (SampledCorrelations/DataRetrieval.jl). It is equivalent to
    `abs(w/kT) * thermal_prefactor(w; kT)`, tends to 1 as w -> 0 (where classical
    statistics are already right) and to |w|/kT in the quantum limit w >> kT, which
    is exactly the Bose suppression a classical calculation is missing.

    `kT` in meV, matching `energies`. Returns 1 everywhere for kT = None.
    """
    w = np.asarray(energies, float)
    if kT is None:
        return np.ones_like(w)
    if kT <= 0:
        raise ValueError(
            f"classical_to_quantum needs a positive kT in meV, got {kT}. Pass "
            f"`classical_to_quantum: false` to leave the classical scale alone.")
    x = w / kT
    out = np.ones_like(x)
    nz = x != 0.0
    # -expm1(-x), not 1 - exp(-x): the latter cancels catastrophically for small x
    # (at x ~ 3e-9 it is already wrong in the 8th digit), which matters because the
    # w -> 0 end of the grid is where the factor is supposed to be exactly 1.
    out[nz] = np.abs(x[nz]) / -np.expm1(-x[nz])
    return out


def check_elastic_leakage(window, subtract_elastic, classical_to_quantum, kT,
                          dt, n_t, on_elastic_leakage="warn", threshold=2.0):
    """Report the `window: cosine` + `subtract_elastic: false` trap (OPEN_WORK #15).

    `window: cosine` convolves the spectrum with [¼, ½, ¼] (see `lag_window`), which
    moves a quarter of the ELASTIC line into the first inelastic bin — where
    `classical_to_quantum_factor` multiplies it by c2q(Δω), 1 at ω = 0 but ~|Δω|/kT
    beyond it. On an ordered magnet at kT ≪ Δω that is catastrophic and looks like
    physics: measured 9.10 in a single bin of a spectrum whose entire LSWT band sum
    is 0.5 (item 12's table). `subtract_elastic: true` removes the delta before the
    window can smear it, and then the two windows agree.

    All three switches are independent booleans defaulting to false, so the dangerous
    combination is one keystroke away and says nothing — hence this check. It REPORTS
    rather than decides (making `cosine` imply `subtract_elastic` would silently
    change what the config asked for), and the condition is a computable number
    rather than a guess: the amplification IS c2q(Δω) with Δω = 2π/T the energy grid
    step, so the warning names it.

    Quiet unless ALL of: `window` is cosine, `subtract_elastic` is off,
    `classical_to_quantum` is on, and the amplification reaches `threshold`
    (default 2, i.e. kT ≲ Δω/1.6). Below that the smear costs no more than the one
    bin of Hann broadening `lag_window` documents, which is the point of the window.

    `on_elastic_leakage`: "warn" (default), "error" or "off". Returns the message
    (or None), so a caller can report it its own way.
    """
    mode = str(on_elastic_leakage or "warn").lower()
    if mode == "off":
        return None
    if lag_window(2, window) is None or subtract_elastic or not classical_to_quantum:
        return None
    if kT is None or float(kT) <= 0 or int(n_t) < 2 or float(dt) <= 0:
        return None
    dw = 2 * np.pi / (int(n_t) * float(dt))
    amp = float(classical_to_quantum_factor(np.array([dw]), float(kT))[0])
    if amp < float(threshold):
        return None
    msg = (f"window: cosine with subtract_elastic: false on a spectrum that is "
           f"classical_to_quantum-corrected at kT={float(kT):g} meV. The window "
           f"smears a quarter of the ELASTIC line into the first inelastic bin "
           f"(dw={dw:.3g} meV), where the c2q factor multiplies it by {amp:.3g}x. "
           f"If this model is ordered, that one bin can outweigh the whole magnon "
           f"band. Set `subtract_elastic: true` (the pair that behaves), or "
           f"`window: rectangular`, or `on_elastic_leakage: off` to silence this.")
    if mode == "error":
        raise RuntimeError(msg)
    logger.warning(msg)
    return msg


def sampled_correlations(model, params, q_cart, kT, supercell=(6, 1, 1),
                         dt=0.02, n_steps=2048, n_traj=8, therm_sweeps=2000,
                         record_every=1, cross_section="perp", seed=0,
                         classical_to_quantum=True, disorder=None,
                         periodic=(True, True, True), window="rectangular",
                         subtract_elastic=False, on_elastic_leakage="warn"):
    """Thermalize by Metropolis then evolve LL dynamics; average S(q,ω) over `n_traj`
    independent thermal starts. Returns a DynamicsResult.

    `classical_to_quantum` (default on) rescales the result onto the quantum
    intensity scale -- see `classical_to_quantum_factor`. Without it the output is
    the raw classical S(q,ω), which equipartitions and therefore carries far too
    much weight at ħω >> kT; set it False only to inspect that raw quantity.

    The ABSOLUTE scale is Sunny's and the LSWT engine's -- per chemical cell, with
    the 1/2π of the time transform -- so the corrected result can be compared with
    `calculate_sqw` and with data directly. See `dynamical_structure_factor`.

    `window: cosine` tapers the time correlation so the truncated trajectory does not
    leak weight across the whole frequency axis, and `subtract_elastic: true` removes
    the elastic delta first. Use them TOGETHER on an ordered magnet -- see
    `lag_window` for the measurement behind that, and for why the default stays
    `rectangular`. Asking for one without the other where it is dangerous is
    reported by `check_elastic_leakage` (`on_elastic_leakage`)."""
    from .thermal_mc import build_supercell, _sweep, n_chemical_cells

    check_elastic_leakage(window, subtract_elastic, classical_to_quantum, kT,
                          dt * record_every,
                          int(np.ceil(int(n_steps) / int(record_every))),
                          on_elastic_leakage)

    H, b, N, S, pos = build_supercell(model, params, supercell, disorder=disorder,
                                      periodic=periodic)
    n_cells = n_chemical_cells(model, supercell)
    rng = np.random.default_rng(seed)
    beta = 1.0 / kT

    energies = None
    acc = None
    for it in range(n_traj):
        m = rng.standard_normal((N, 3))
        m *= S / np.linalg.norm(m, axis=1, keepdims=True)
        g = H @ m.ravel() + b
        for _ in range(therm_sweeps):
            _sweep(m, g, H, b, beta, S, rng)
        traj = evolve(H, b, S, m, dt, n_steps, record_every)
        e, sqw = dynamical_structure_factor(traj, pos, q_cart, dt * record_every,
                                            cross_section, n_cells=n_cells,
                                            window=window,
                                            subtract_elastic=subtract_elastic)
        acc = sqw if acc is None else acc + sqw
        energies = e
    sqw = acc / n_traj
    if classical_to_quantum:
        sqw = sqw * classical_to_quantum_factor(energies, kT)[:, None]
    return DynamicsResult(q_vectors=np.asarray(q_cart, float).reshape(-1, 3),
                          energies=energies, sqw=sqw, temperature=kT,
                          classical_to_quantum=bool(classical_to_quantum))
