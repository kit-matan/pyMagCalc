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
in YbMgGaO₄). pyMagCalc has neither KPM (Gap Tier 2 #10) nor per-bond disorder /
`to_inhomogeneous`, so the disorder-broadened spectrum is out of scope. The clean
120° dispersion above is the coherent spectrum that the disorder broadens.
