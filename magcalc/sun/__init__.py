"""SU(N) ("generalized") spin-wave theory.

Dipole LSWT expands each spin as a single boson about a classical direction, which
structurally cannot represent single-ion (multipolar) excitations -- transitions between
the local crystal-field levels of a large-S ion. FeI2's single-ion bound state is the
canonical example, and it is simply absent from a dipole-mode calculation.

This is a SECOND engine, not an extension of the dipole one (Sunny is organised the same
way: HamiltonianDipole.jl vs HamiltonianSUN.jl). Each site carries an N-level local
Hilbert space (N = 2S+1); the reference state is a coherent state |Z> in CP^(N-1), and
excitations are the N-1 bosons connecting |Z> to the other local levels. The dynamical
matrix is 2(N-1)L x 2(N-1)L rather than 2L x 2L.

See SUN_PLAN.md for the validation gates. The load-bearing one: for S=1/2 (N=2) SU(N)
is IDENTICAL to dipole LSWT, so any convention error fails loudly there.
"""
from .lswt import SUNModel
from .operators import coherent_from_direction, spin_matrices, stevens_matrices

__all__ = ["spin_matrices", "stevens_matrices", "coherent_from_direction", "SUNModel"]
