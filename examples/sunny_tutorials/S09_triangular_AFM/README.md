# S09 — Disordered triangular antiferromagnet

Port of Sunny tutorial `09_Disorder_KPM.jl`. **Completed 2026-08-13.**

```bash
magcalc run examples/sunny_tutorials/S09_triangular_AFM/config.yaml            # clean, rotating frame
magcalc run examples/sunny_tutorials/S09_triangular_AFM/config_supercell.yaml  # clean, real-space cell
python examples/sunny_tutorials/S09_triangular_AFM/disorder_kpm.py             # disorder + KPM, ~30 s
python examples/sunny_tutorials/S09_triangular_AFM/disorder_kpm.py --L 30      # Sunny's system size
```

One s=1/2 site per cell on a triangular lattice, nearest-neighbour AFM `J = 1`. The
classical ground state is the coplanar 120° order at `k = [1/3, 1/3, 0]`.

| file | what it is |
|---|---|
| `config.yaml` | the clean spectrum via the rotating-frame `single_k` method |
| `config_supercell.yaml` | the same state as an explicit √3×√3 cell, in `mode: SUN` |
| `disorder_kpm.py` | the tutorial's subject: disorder, re-relaxation, KPM |

## Why the port needed a second config

`config.yaml`'s rotating-frame `single_k` structure is exact and cheap, and the
SU(N)/KPM path cannot consume it — disorder is a per-bond property of a large
**real-space** cell. `config_supercell.yaml` therefore carries the 120° state as a
√3×√3 magnetic cell, which is what `disorder_kpm.py` enlarges.

**Check the energy before any spectrum.** The exact classical value is

    E/site = (1/2) · 6 · J · S² · cos(120°) = −0.375 meV

and the run reports `SU(N) ground state: E/site = -0.37500000`. This is not a
formality: the *transposed* supercell matrix has the same |det| = 3 and cannot host
the order at all (`k·(a₁+a₂) = 2/3 ∉ ℤ`), landing on a frustrated collinear state at
+0.0833 — with a perfectly plausible-looking dispersion. Commensurability is the
criterion; the energy is how you see it.

**Validation (analytic).** The 120° state has the exact LSWT dispersion
`ω_q = 3JS√[(1−γ_q)(1+2γ_q)]`, `ω_max = 3JS·√(9/8) = 1.5910 meV`. The three bands of
the √3×√3 cell reproduce `{ω(q−k), ω(q), ω(q+k)}` to **1e-13** at generic q, and the
9×9 cell's S(q,ω) equals the √3×√3 cell's to **1e-9** — the 78 extra folded bands
carry exactly zero weight. All pinned in `tests/test_s09_disorder_kpm.py`.

## Disorder

`disorder_kpm.py` follows the tutorial: enlarge the cell, scale every NN exchange by
`1 + σ·ξ` (ξ standard normal, one draw per bond), **re-relax**, then compute S(q,ω)
with KPM. Measured at L = 12 (144 sites), 3 seeds, averaged over q away from the
Goldstone points at Γ and K:

| | clean | σ = 0.1 | σ = 1/3 (Sunny's) |
|---|---|---|---|
| width √(⟨E²⟩−⟨E⟩²) | 0.1834 meV | 0.2071 (+12.9 %) | 0.3945 (+115 %) |
| peak intensity | 7.564 | 7.232 (−4.4 %) | 4.893 (−35 %) |
| weight above `ω_max` | 0.0003 | 0.0002 | **0.0213** |

At Sunny's own system size — L = 30, 900 sites — the same run gives **+13.5 %** with
a seed-to-seed spread of 0.4 %, and the clean numbers are identical to the L = 12
ones (they describe the same crystal). So L = 12 is already converged for this
observable; L = 30 costs ~13 min (130 s per KPM map, ~180 s per relaxation) against
~30 s.

The last row is the tutorial's "the discrete bands broaden into a continuum", stated
against the analytic bound rather than against a recorded number: the clean spectrum
has no weight above `ω_max = 1.5910 meV` by construction, so 2 % of the weight up
there is disorder-induced continuum. Seed-to-seed spread of the width is 1–3 %, well
below the effect.

## Two things this port turned up, neither of which was the thing being looked for

### 1. A real bug in `sun/kpm.py` — and the old diagnosis here was wrong

The note this README used to carry read: *"disorder narrowed the KPM width instead of
broadening it, which is what expanding about a non-minimum buys you"*, blaming the
ferromagnetic placeholder reference state. The placeholder was indeed wrong (E/site =
+0.75 on a lattice whose ground state is −0.375) — and fixing it did **not** fix the
narrowing.

The narrowing was `kpm.py` running its Chebyshev recursion on `D̂` where the structure
factor needs `conj(D̂)`. The two agree whenever `D̂` is real — every collinear,
inversion-symmetric model, which is everything the KPM suite tested — so it survived
until a **non-collinear supercell**: on the clean 81-site 120° cell it put ~5 % of the
intensity onto bands that carry none, at LOW energy. A clean spectrum that arrives
pre-broadened makes real disorder look like narrowing.

A second, independent defect was found next to it: the moments are assembled from the
annihilation block of the Nambu basis while `structure_factor` uses the creation
block, and the two carry complex-conjugate weight matrices. Symmetric channels
(`perp`, `trace`, …) cannot see the difference; **`cross_section: chiral` came back
with the sign reversed**, on a model as ordinary as a ferromagnet. Both are fixed and
pinned in `tests/test_kpm.py` against exact diagonalization.

### 2. At Sunny's σ = 1/3 the 120° state is not a classical minimum

`H₂(q) ⪰ 0` at every q is the exact criterion for the reference state to be a
classical minimum. Measured on a 9×9 cell over the Γ–K–M–Γ path:

| σ | min eig H₂ | max \|Im ω\| |
|---|---|---|
| ≤ 0.10 | > −1e-9 | < 1e-7 |
| 0.15 | −1e-4 … −7e-4 (2 seeds of 3) | 0.02 … 0.04 |
| 1/3 | −8e-4 … −1e-2 | 0.04 … 0.16 |

0.16 meV of imaginary energy on a 1.591 meV band is 10 %. **The relaxation is not at
fault**: Metropolis annealing from three temperatures and a damped CP^(N−1) quench
both return the same state to 8 decimals, and relaxing in cells of 2×, 3×, 4× the
disorder period does not lower the energy — the disorder is destabilising the 120°
order itself, which is the physics that makes YbMgGaO₄ interesting in the first
place.

**KPM cannot notice by itself.** It never diagonalizes, so there is no Cholesky to
fail and no imaginary energy to report: about a non-minimum it returns a smooth,
plausible spectrum. The check is now the engine's — `SUNModel.assert_stable(qs)`,
`min eig H₂(q) ≥ 0` at every q, one shifted Cholesky each — and both this script and
the runner's `kpm_sqw` task apply it, at the same tolerance, rather than two
hand-rolled versions. `disorder_kpm.py` refuses by default; `--force` reproduces the
tutorial's setting anyway. The shipped default is σ = 0.1, which is stable and still
broadens measurably.

Sunny's own tutorial makes no such check — its `SpinWaveTheoryKPM` has no
positive-definiteness guard, and the model is the same one.
