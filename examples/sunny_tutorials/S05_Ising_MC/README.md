# S05 — 2D Ising Monte Carlo — OUT OF SCOPE

Sunny tutorial `05_MC_Ising.jl` simulates the classical 2D Ising model on a
128×128 square lattice with a **thermal Monte-Carlo** sampler (`LocalSampler` with
`propose_flip`), sweeping near the exact critical temperature
`Tc = 2/ln(1+√2)` to show the ferromagnetic ordering transition.

**pyMagCalc has ground-state Monte-Carlo annealing** (`magcalc/annealing.py`, used
by `minimization: {method: anneal}`) **but not thermal Monte-Carlo sampling** at a
target temperature (equilibrium statistics, critical phenomena, order parameter vs
T). That is Gap Tier 2 #6 in `GAP_STATUS.md` — not done. This tutorial is about
the finite-*T* phase transition, which is a thermodynamic-sampling task, not a
spin-wave calculation, so there is no LSWT analogue to point to.

(The annealing minimiser *does* use a Metropolis + cooling schedule internally —
but to find the T = 0 ground state, not to sample the equilibrium ensemble at
finite *T*.)
