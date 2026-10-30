"""An independent ray-tracing Monte Carlo of a direct-geometry chopper spectrometer, the
oracle for `magcalc.resolution` (test-only; not collected by pytest).

It shares nothing with the module under test but the instrument's description. Each
neutron gets an emission time from the pulse and an inverse speed u = 1/v from a flat
band. It passes each chopper by the chopper's own geometry at its arrival time:
- a single disk if its transverse position is inside the moving slot;
- a counter-rotating pair if it is inside both slots, moving in opposite directions;
- a straight-slit Fermi chopper if its straight path clears the rotated channel.

It then scatters at the sample with a fixed true energy transfer and travels to the
detector over the true L2, depth included. Its recorded transfer is computed the way a
reduction does, with exact kinematics and no linearisation: the nominal Ei and the
nominal time at the sample. The linear model in `resolution.py` should reproduce the
distribution of recorded-minus-true energy, mean removed.
"""
import numpy as np

V = 437.393     # m/s per sqrt(meV)


def _passes(ch, t, y, rng):
    """Boolean mask: neutrons at time t (s) and transverse position y (m) through `ch`.
    Only the opening the chopper is phased to counts (ideal frame-overlap choppers): a
    neutron reaching it a turn early or late is stopped."""
    dt = t - ch["phase"]
    kind = ch["kind"]
    if kind == "disk":
        s = 2 * np.pi * ch["r"] * ch["f"] * dt
        return np.abs(y - s) < ch["w"] / 2
    if kind == "pair":
        s = 2 * np.pi * ch["r"] * ch["f"] * dt
        return (np.abs(y - s) < ch["w"] / 2) & (np.abs(y + s) < ch.get("w2", ch["w"]) / 2)
    if kind == "fermi":
        phi = 2 * np.pi * ch["f"] * dt
        a = rng.uniform(-ch["w"] / 2, ch["w"] / 2, len(t))         # offset in its channel
        return (np.abs(phi) < np.pi / 4) & (np.abs(a + ch["L"] * np.tan(phi)) < ch["w"] / 2)
    raise ValueError(kind)


def trace(rng, *, ei, l1, l2, choppers, pulse, omega, depth=0.0, n_want=300_000,
          batch=2_000_000, u_half=None, max_batches=400):
    """Recorded-minus-true energy transfer (meV) of transmitted neutrons, mean removed.

    choppers: dicts with kind ('disk', 'pair', 'fermi'), x (m from the moderator), f (Hz),
        w (slot width at the beam, m; for 'fermi' the channel width), w2 (pair), r (beam
        radius, m), b (beam width at the chopper, m; the neutron's transverse position is
        uniform in it), L (Fermi slit length, m). Each is phased to pass the nominal
        neutron (emission 0, speed sqrt(ei) V) at the centre of its opening.
    pulse: callable (rng, n) -> emission times (s) about the phasing time 0.
    omega: the true energy transfer (meV); depth: full width (m) of a uniform extra L2.
    """
    u0 = 1.0 / (V * np.sqrt(ei))
    chs = [dict(c, phase=c["x"] * u0) for c in choppers]
    if u_half is None:
        # wide enough for every opening: the slowest-closing chopper over the shortest lever
        widths = []
        for c in chs:
            if c["kind"] == "fermi":
                widths.append(c["w"] / (2 * np.pi * c["f"] * c["L"]))
            else:
                widths.append((c["w"] + c.get("w2", c["w"]) + c.get("b", 0.0))
                              / (2 * np.pi * c["r"] * c["f"]))
        # ... and the pulse: a late emission passes a chopper only with a faster speed
        span = np.diff(np.percentile(pulse(rng, 200_000), [0.01, 99.99]))[0]
        xs = sorted(c["x"] for c in chs)
        lever = min([xs[0]] + [b - a for a, b in zip(xs, xs[1:])])
        u_half = 1.5 * (sum(widths) + span) / lever
    out = []
    got = 0
    for _ in range(max_batches):
        t0 = pulse(rng, batch)
        u = u0 + rng.uniform(-u_half, u_half, batch)
        ok = u > 0
        for c in chs:
            y = rng.uniform(-c.get("b", 0.0) / 2, c.get("b", 0.0) / 2, batch)
            ok &= _passes(c, t0 + c["x"] * u, y, rng)
        t0, u = t0[ok], u[ok]
        ei_true = (1.0 / (u * V)) ** 2
        ef = ei_true - omega
        good = ef > 0
        t0, u, ef = t0[good], u[good], ef[good]
        vf = V * np.sqrt(ef)
        dl = rng.uniform(-depth / 2, depth / 2, len(u)) if depth else 0.0
        t_det = t0 + l1 * u + (l2 + dl) / vf
        t_s0 = l1 * u0                                   # the reduction's time at the sample
        ef_m = ((l2 / (t_det - t_s0)) / V) ** 2
        out.append((ei - ef_m) - omega)
        got += len(u)
        if got >= n_want:
            break
    de = np.concatenate(out)
    if len(de) < n_want // 10:
        raise RuntimeError(f"only {len(de)} neutrons transmitted; widen u_half or batch")
    return de - de.mean()


def sampler(dist, n_grid=400_001):
    """(rng, n) -> draws from a magcalc Distribution by inverse CDF on a fine grid; for
    a pulse given only as a pdf, which is all the oracle is allowed to share."""
    t = np.linspace(*dist.support(), n_grid)
    c = np.cumsum(dist.pdf(t))
    c /= c[-1]

    def draw(rng, n):
        return np.interp(rng.uniform(0, 1, n), c, t)
    return draw
