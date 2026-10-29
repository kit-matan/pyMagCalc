# Regenerate tests/data_biquadratic_noncollinear_sunny.json: Sunny 0.8.1's dipole-LSWT
# bands and band-summed intensities for a NON-COLLINEAR model with a biquadratic term,
# for tests/test_biquadratic_noncollinear.py.
#
#   JULIA_PKG_OFFLINE=true julia --project=@v1.12 tests/biquadratic_noncollinear_sunny_reference.jl
#
# (Sunny 0.8.1 is installed in the v1.12 environment on the maintainer's machine; any
# environment with Sunny 0.8.1 works.)
#
# Why this model: until 2026-10-29 pyMagCalc's symbolic build dropped the
# (j, i)-ordered boson words of B (S_i.S_j)^2 on every bond leaving the cell, and their
# coefficient vanishes for collinear moments and for equally rotated sites -- so every
# earlier biquadratic check (all collinear) passed with the bug in. Here the 120-degree
# state of the triangular AFM sits in a 3x3 supercell, and every nearest-neighbour bond
# crossing the supercell boundary joins two differently oriented sites.
#
# Triangular lattice a = 3 A (c = 10 A), s = 1, J = 1 meV and B = BIQ on the
# nearest-neighbour bonds, 120-degree order S(r) = (cos phi, sin phi, 0) with
# phi = 2 pi (n1 + n2) / 3. Mode :dipole_uncorrected, which is what pyMagCalc's dipole
# engine is (CLAUDE.md, section 5b1). q in RLU of the CHEMICAL cell. Measure
# `ssf_perp(sys; apply_g=false)`, equal to pyMagCalc's `perp` in absolute units.
using Sunny, LinearAlgebra, Printf

const J = 1.0
const BIQ = 0.06

cryst = Crystal(lattice_vectors(3.0, 3.0, 10.0, 90, 90, 120), [[0, 0, 0]])
sys0 = System(cryst, [1 => Moment(s=1, g=2)], :dipole_uncorrected)
set_exchange!(sys0, J, Bond(1, 1, [1, 0, 0]); biquad=BIQ)
sys = repeat_periodically(sys0, (3, 3, 1))
for n1 in 1:3, n2 in 1:3
    phi = 2pi * ((n1 - 1) + (n2 - 1)) / 3
    set_dipole!(sys, [cos(phi), sin(phi), 0.0], (n1, n2, 1, 1))
end

# The state must be a classical minimum, or both codes would agree about a
# meaningless spectrum: relaxing from it must not lower the energy.
e0 = energy(sys)
sys_r = clone_system(sys)
minimize_energy!(sys_r)
@printf("E = %.12f, relaxed E = %.12f\n", e0, energy(sys_r))
@assert energy(sys_r) > e0 - 1e-9

qs = [[0.13, 0.0, 0.0], [0.21, 0.37, 0.0], [0.5, 0.0, 0.0], [0.3, 0.2, 0.0],
      [0.41, -0.17, 0.0], [0.07, 0.29, 0.0]]
swt = SpinWaveTheory(sys; measure=ssf_perp(sys; apply_g=false))
res = intensities_bands(swt, qs)
@assert all(isfinite, res.disp)

open(joinpath(@__DIR__, "data_biquadratic_noncollinear_sunny.json"), "w") do io
    println(io, "{")
    println(io, "  \"sunny_version\": \"0.8.1\", \"mode\": \"dipole_uncorrected\",")
    @printf(io, "  \"J\": %.6f, \"B\": %.6f, \"s\": 1.0, \"a\": 3.0, \"c\": 10.0,\n", J, BIQ)
    @printf(io, "  \"energy_per_site\": %.15f,\n", e0 / 9)
    println(io, "  \"q_rlu\": [", join(["[" * join([@sprintf("%.15f", x) for x in q], ", ") * "]" for q in qs], ", "), "],")
    nq = length(qs)
    bands = [sort(res.disp[:, i]) for i in 1:nq]
    totals = [sum(res.data[:, i]) for i in 1:nq]
    println(io, "  \"bands\": [", join(["[" * join([@sprintf("%.12f", x) for x in b], ", ") * "]" for b in bands], ",\n            "), "],")
    println(io, "  \"total_intensity\": [", join([@sprintf("%.12f", x) for x in totals], ", "), "]")
    println(io, "}")
end
println("wrote data_biquadratic_noncollinear_sunny.json")
