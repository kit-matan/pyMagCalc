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

**Both ingredients now exist.** KPM is `magcalc/sun/kpm.py` (Gap Tier 2 #10) and
per-bond disorder is `sun.lswt.apply_bond_disorder(model, sigma, seed)` (Gap 4 #16b),
the analogue of Sunny's `to_inhomogeneous` + `set_exchange_at!`. Disorder must be
applied to a model built on a SUPERCELL — on the chemical cell it is not disorder,
just a different clean model repeated.

**The port is still not finished**, and the honest reason is the reference state, not
the machinery. This folder's clean config uses the rotating-frame `single_k` 120°
order, which the SU(N)/KPM path does not consume; driving disorder + KPM needs the
120° state as an explicit real-space supercell. Run on a ferromagnetic placeholder
instead — which is *not* the ground state of a triangular AFM — the spectrum is not
physical (measured: disorder narrowed the KPM width rather than broadening it, which
is what expanding about a non-minimum buys you).

So what remains is building the 120° state on a real-space supercell and then
applying disorder + KPM. The disorder itself is validated independently
(`tests/test_bond_disorder.py`).

The clean 120° dispersion above is the coherent spectrum that the disorder broadens.
