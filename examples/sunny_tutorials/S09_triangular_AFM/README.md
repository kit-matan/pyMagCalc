# S09 — Disordered triangular antiferromagnet (partial port)

Port of Sunny tutorial `09_Disorder_KPM.jl`.

## What is ported (`config.yaml`)

The **clean** LSWT limit: one s=1/2 site per cell on a triangular lattice with
nearest-neighbour AFM `J = 1`. The classical ground state is the coplanar
120° order (propagation vector `k = [1/3, 1/3, 0]`), handled exactly with the
rotating-frame `single_k` method.

**Validation (analytic):** the 120° triangular AFM has the exact LSWT dispersion
`ω_q = 3JS √[(1−γ_q)(1+2γ_q)]`, whose maximum is `ω_max = 3JS·√(9/8) = 1.591 meV`
for `J=1, S=1/2`. The ported spectrum reproduces this, with gapless Goldstone
modes at Γ and at the K-point ordering wavevector `[1/3, 1/3, 0]`.

## What is NOT ported

The tutorial's actual subject is **disorder broadening** computed with
`SpinWaveTheoryKPM` (kernel polynomial method) on a large inhomogeneous supercell
with stochastic exchange constants and g-factors (modelling Mg/Ga site disorder
in YbMgGaO₄).

**KPM now exists** (`magcalc/sun/kpm.py`, Gap Tier 2 #10 — this README used to say it
did not). What is still missing is the other half: **per-bond disorder in LSWT**, i.e.
Sunny's `to_inhomogeneous` + `set_exchange_at!` on a large supercell. Vacancies and
open boundaries landed for the *classical* samplers (Gap 4 #16a); the LSWT half is
Gap 4 **#16b**, still open, so the disorder-broadened spectrum remains out of reach.

The clean 120° dispersion above is the coherent spectrum that the disorder broadens.
