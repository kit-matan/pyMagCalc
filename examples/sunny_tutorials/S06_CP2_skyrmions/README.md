# S06 — CP² skyrmions from a dynamical quench

Port of Sunny tutorial `06_CP2_Skyrmions.jl`. **Ported 2026-08-13**, at Sunny's own
system size.

```bash
python examples/sunny_tutorials/S06_CP2_skyrmions/quench.py          # L = 40, ~4 min
python examples/sunny_tutorials/S06_CP2_skyrmions/quench.py --L 16   # quick look
magcalc run examples/sunny_tutorials/S06_CP2_skyrmions/config.yaml   # the model only
```

`config.yaml` is the Hamiltonian; `quench.py` is the tutorial. The product here is a
non-equilibrium **texture**, not a spectrum, so there is no `tasks:` entry that
expresses it. The figure it writes (`S06_skyrmions.png`, gitignored like every other
generated plot) reproduces the tutorial's three panels.

## The physics

An SU(3) model on a triangular lattice: ferromagnetic J₁ = −1, antiferromagnetic
J₂ = 2/(1+√5) on the second (√3) shell, exchange anisotropy Δ = 2.6, field h = 15.5
and easy-**plane** single-ion D = 19. Randomize, then relax with damped CP²
(Landau–Lifshitz–Gilbert-like) dynamics at kT = 0. Skyrmions are the metastable
texture the quench leaves behind — **not** the ground state, so a minimizer or a
Metropolis equilibration would destroy exactly the object of interest.

Measured on the shipped run (L = 40, dt = 0.01, damping 0.05, seed 0):

| τ | E/site | ⟨S^z⟩ | \|⟨S⟩\| | Q_SU(3) |
|---|---|---|---|---|
| 4 | +0.439119 | 0.4418 | 0.7459 | +12 |
| 16 | +0.035968 | 0.3656 | 0.6439 | −4 |
| 256 | −0.046842 | 0.2859 | 0.5509 | −6 |

The charge is an exact integer at every snapshot — quantization is a property of the
discrete texture, not an approximation — and it is non-zero, which is the difference
between a skyrmion liquid and the uniform state. The whole 25 600-step run takes
**214 s**.

**Why `mode: SUN`, in one number.** The uniform SU(3) ground state of this
Hamiltonian is the *quadrupolar* |m=0⟩ level at E/site = 0 with **zero dipole
moment** — the tutorial's white background, a quantum paramagnet. The DIPOLE ground
state of the same Hamiltonian is a canted in-plane state at −4.644706, because
classical dipoles cannot represent |m=0⟩ at all. The two theories disagree
qualitatively, not marginally.

That is also why the figure plots the **SU(3) Berry curvature** per plaquette
(`sun_topological_charge`) rather than the dipole solid angle: over most of the area
⟨S⟩ ≈ 0, and the Berg–Lüscher formula normalizes every spin, so a dipole texture
there is not merely inaccurate but undefined. `topological_charge` now refuses on
that ground instead of returning quantized-looking noise.

## What unblocked it

Two things, in this order — and the first was a red herring worth recording.

1. **The J₂ bond shell was the leading suspect, and was innocent.** A wrong second
   shell suppresses exactly the frustration that sets the skyrmion length scale and
   would look precisely like the observed symptom. It matches Sunny:
   `Bond(1,1,[1,2,0])` is |a₁ + 2a₂| = √3 with coordination 6, and
   `magcalc symmetry` finds the same shells (1.0 ×6, √3 ×6, 2.0 ×6).
   Pinned in `tests/test_sunny_tutorials.py`.

2. **The real blocker was performance, and it was three orders of magnitude.**
   `SUNModel.local_field(i, ·)` scans the entire bond list per site, so building the
   derivative cost O(sites²): ~16 s/step at 1600 sites, i.e. ~55 hours for this run.
   The quench was therefore only ever run at 64–256 sites, where it relaxes to the
   uniform state — which looks like a physics failure and is a size failure. A
   skyrmion is several lattice constants across.

   `sun/dynamics.py` now sums the bond list once and never forms the matrices h_i at
   all, only h_i|Z_i⟩. Same physics, different loop order:

   | sites | before | after |
   |---|---|---|
   | 64 | 57 ms/step | 0.39 ms |
   | 256 | 270 ms | 1.3 ms |
   | **1600** | **~16 000 ms** | **8.4 ms** |

   Model construction was then the bottleneck (36 s), from a linear scan in
   `SUNModel._replicate`; it is a dict lookup now (3.4 s).

## What is pinned

- **The Hamiltonian, against Sunny 0.8.1 to 5.4e-13** — the energy of an *arbitrary*
  coherent-state configuration, which exercises both exchange shells, the anisotropy
  and the Zeeman at once. A ground-state energy would be far weaker, since the state
  relaxes to fit whatever Hamiltonian it is given. Both codes build the same
  configuration from a closed-form function of the site index, so no RNG is shared.
- **The two sign conventions**, separately: Sunny's `g = -1` becomes
  `H_dir = [0,0,-1]` here, and the on-site term must come out `19(S^z)² − 15.5 S^z`
  = diag(3.5, 0, 34.5). The field was silently *dropped* by `mode: SUN` until
  2026-08-04 — found while porting this tutorial (`tests/test_sun_zeeman.py`), and
  without it the texture decays to |m=0⟩ everywhere.
- **The quench itself**, in `tests/test_sun_quench.py`: dE/dt = −2λ·Var(h) (which
  fixes the damping SIGN — the dipole engine once had it backwards), norm
  conservation, the λ = 0 reduction to the conservative flow, and the vectorized
  derivative against the `local_field` definition including the biquadratic and
  mixed-spin paths. None of that had a test before this port.
- **The skyrmion number is quantized and non-zero** after a quench (slow marker).

## A bug this port found

`method: anneal` — the *recommended* ground-state search — ended with a
`steepest_descent` polish whose result was taken **unconditionally**. That step
ignores the on-site block H_ii, so with this model's D = 19 it walked from the exact
minimum (−4.644706) to a local **maximum** (+0.520665), on every seed, and
`minimize_energy` reported "4 of 4 runs hit the minimum" — the very consensus the
docs prescribe as the acceptance criterion. The polish is now kept only if it
lowers the energy (`magcalc/annealing.py`, `tests/test_annealing.py`).

The pre-existing `test_steepest_descent_is_monotone` could not see it: it runs on an
AFM chain, whose H_ii is zero.
