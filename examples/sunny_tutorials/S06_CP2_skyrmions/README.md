# S06 — CP² skyrmion liquid — OUT OF SCOPE

Sunny tutorial `06_CP2_Skyrmions.jl` is a **non-equilibrium** study: it quenches a
triangular-lattice spin-1 system (with competing FM/AFM exchange, easy-plane
single-ion anisotropy, and a field) from a high-temperature state using the
**SU(N) generalization of Landau-Lifshitz dynamics with Langevin damping**, and
watches a disordered liquid of CP² skyrmions form. It tracks topological charge
density in real time.

**pyMagCalc implements neither SU(N) real-space Langevin dynamics nor
non-equilibrium quench simulation nor topological-charge diagnostics.** These are
time-dependent, finite-*T*, real-space dynamics (related to Gap Tier 2 #5/#6 in
`GAP_STATUS.md`) — categorically outside a linear-spin-wave engine, which computes
only the harmonic excitation spectrum about a static equilibrium.

There is no LSWT analogue: the object of study (a dynamically-formed topological
defect liquid) does not exist in the harmonic-spectrum framework.
