# S07 — Long-range dipole-dipole on the pyrochlore

Port of Sunny tutorial `07_Dipole_Dipole.jl` (Del Maestro & Gingras, *J. Phys.:
Cond. Matter* **16**, 3339 (2004)).

## The physics

Gd³⁺ (s = 7/2) on the pyrochlore lattice, nearest-neighbour AFM `J1 = 0.304 K`,
plus the infinite-range magnetic dipole-dipole interaction. The NN-AFM pyrochlore
is a textbook frustrated magnet: on its own it has a macroscopic manifold of
degenerate classical ground states and **flat zero-energy** spin-wave modes. The
dipole-dipole term lifts this degeneracy, selects a long-range-ordered state, and
raises the flat modes to finite energy. `config.yaml` reproduces this: annealing
finds the dipole-selected state reproducibly (E = −6.9745 meV, 7/12 runs, stable
real magnons, |Im ω| ~ 4e-16), and the lowest bands sit at ~0.08–0.13 meV rather
than at zero — the lifted frustrated modes (compare Sunny's Fig 2 right panel).

## The three panels of the tutorial (Fig 2)

The tutorial compares one spin configuration under three Hamiltonians. In
pyMagCalc, edit the `dipole_dipole` line of `config.yaml`:

| Panel | `interactions.dipole_dipole` |
|---|---|
| Without long-range dipole (frustrated, flat zero modes) | *(delete the line)* |
| With long-range dipole (Ewald) — **as shipped** | `{method: ewald}` |
| Truncated at 5 Å (Sunny's dashed curves) | `{method: truncated, cutoff: 5.0}` |

## Validation

The dipole-dipole **engine** is pinned independently against Sunny 0.8.1 in
`tests/test_ewald.py`:

- Ewald `enable_dipole_dipole!` reproduced to **1.3e-8**;
- the truncated real-space sum **converges to** the Ewald result as the cutoff
  grows (they differ only by the shape-dependent surface/demagnetisation term);
- the classical energy includes the dipolar term, so the minimiser optimises the
  same Hamiltonian LSWT diagonalises (matches Sunny's `energy_per_site` to 1e-9).

So the quantitative accuracy of the dipolar sum is fixed by that test; this folder
exercises it on the pyrochlore to reproduce the tutorial's qualitative result —
the lifting of the frustrated flat band.

**Units:** everything is meV (J1 converted from 0.304 K). pyMagCalc's Ewald
constant is meV-based; the dipole-to-J1 ratio is physical, so the band *shape*
matches Sunny, whose plot axis is in Kelvin (= meV / 0.0862).

**Degeneracy caveat:** the ordered state is one of several symmetry-equivalent
domains; Sunny selects among them with a random `seed`, pyMagCalc via the anneal
`seed`. Different domains give the same bands.
