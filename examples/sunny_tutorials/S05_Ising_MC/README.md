# S05 — 2D Ising model by Monte Carlo

Port of Sunny tutorial `05_MC_Ising.jl`.

> **This file previously said OUT OF SCOPE.** That was correct when written —
> pyMagCalc had ground-state annealing but no thermal sampling. Parallel-tempering
> Monte Carlo (Gap Tier 2 #6) and the Ising flip proposal (this port) have since
> landed.

## How a continuous-spin engine does Ising

Sunny does not have an Ising code path either. It builds one out of continuous spins
with two ingredients:

- `polarize_spins!(sys, (0,0,1))` — start every spin along +z;
- `propose_flip` — the only proposed move is `S → −S`.

Nothing in that pair ever leaves the ±z axis, so the continuous sampler *is* Ising.
pyMagCalc now takes the same two knobs:

```yaml
thermal_mc:
  propose: flip        # Sunny `propose_flip`
  init: [0, 0, 1]      # Sunny `polarize_spins!`
```

`tests/test_sunny_tutorials.py::test_S05_flip_proposal_keeps_the_system_ising` asserts
the mechanism directly — after 50 sweeps every spin must still satisfy
`Sx = Sy = 0, |Sz| = S`. If the proposal ever fell back to the uniform sphere move
this would be a *Heisenberg* model with a different `Tc`, and the Onsager comparison
below would be meaningless while still looking reasonable.

## Validation — Onsager's exact solution

The 2D square-lattice Ising model is exactly solved, so this port needs no oracle
code at all. Spontaneous magnetization `m(T) = [1 − sinh⁻⁴(2J/T)]^(1/8)`:

| T | pyMagCalc | Onsager (exact) | error |
|---|---|---|---|
| 1.5 | 0.9870 | 0.9865 | **0.05%** |
| 2.0 | 0.9100 | 0.9113 | 0.14% |
| 2.269 = Tc | 0.060 | 0 | finite-size rounding |
| 2.6 | 0.002 | 0 | — |
| 3.2 | 0.009 | 0 | — |

and the internal energy at criticality, `E/N = −√2 J` exactly:

| | pyMagCalc | exact |
|---|---|---|
| E/N at Tc | −1.453 | −1.4142 |

3% high, which is the expected finite-size effect on a 24×24 lattice at a critical
point (correlations there are cut off by the box).

`Tc = 2J / ln(1 + √2) = 2.26919`.

## Why `swap_every: 0`

**Replica exchange destroys the very thing this tutorial measures**, and the config
disables it on purpose.

Below `Tc` the Ising model has two degenerate broken-symmetry states. With replica
swaps on, a replica wanders up to high temperature, decorrelates, and comes back with
the *opposite* sign — so ⟨m⟩ averages toward zero. Measured at T = 1.5:

| | ⟨m⟩ |
|---|---|
| swaps on | 0.35 |
| swaps off | 0.9870 |
| Onsager | 0.9865 |

Sunny has no such problem because its `LocalSampler` runs at a single temperature.
`test_S05_replica_swaps_would_destroy_the_broken_symmetry` records this so the
setting is not "tidied up" later.

(Parallel tempering is still the right default for *frustrated* models, where the
whole point is to escape metastable states. It is wrong here precisely because the
degenerate states are physical.)

## Differences from Sunny

- **Lattice 24×24, not 128×128.** Enough for the magnetization to match Onsager to
  0.05%; the full 128² is a fine thing to run but slow as a regression test. Raise
  `supercell` if you want sharper critical behaviour.
- **`g` is irrelevant here.** Sunny uses `Moment(s=1, g=-1)`; with no applied field
  the g-factor drops out of every quantity this tutorial computes.
- `J = -1.0` in the config is **ferromagnetic** in pyMagCalc's sign convention,
  matching Sunny's `set_exchange!(sys, -1.0, ...)`.

## Running it

```bash
magcalc run examples/sunny_tutorials/S05_Ising_MC/config.yaml
```

Reports ⟨E⟩/N, C/N, magnetization and susceptibility across the temperature ladder,
bracketing `Tc`.
