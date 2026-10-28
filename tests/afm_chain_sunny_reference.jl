# Regenerate tests/data_afm_chain_sunny.json: Sunny 0.8.1's one-magnon intensities for
# the Neel S = 1 Heisenberg chain on two powder shells, for tests/test_gapless_backends.py.
#
#   JULIA_PKG_OFFLINE=true julia --project=@v1.12 tests/afm_chain_sunny_reference.jl
#
# (Sunny 0.8.1 is installed in the v1.12 environment on the maintainer's machine; any
# environment with Sunny 0.8.1 works.)
#
# The model is a collinear antiferromagnet, so its two magnons are DEGENERATE at every
# q. That is what the fixture is for: fMagCalc's -q eigenvector matching used to take a
# single best-overlap column, which a degenerate pair need not have, and on these shells
# it returned zero intensity at about one point in eight. Only the group total of a
# degenerate pair is an observable, so the test compares totals.
#
# Cell: a = 6 A with sites at x = 0 and 1/2 (bond 3 A), b = c = 10 A; J = 1 meV on
# the nearest-neighbour bond, moments +-z. The q points are pyMagCalc's
# `numerical.fibonacci_sphere_points(|Q|, 200)` for |Q| = 1.6 and 2.6 1/A, rebuilt here
# from the same formula (the test checks they agree), in Cartesian 1/A.
# Measure: `ssf_perp(sys; apply_g=false)`, which pyMagCalc's `perp` equals in absolute
# units (tests/test_absolute_normalization.py). Sunny regularizes H by 1e-8, which
# moves energies near the Goldstone points; the test compares above 0.2 meV only.
using Sunny, LinearAlgebra, Printf

function fibonacci(qmag, n)
    pts = Vector{Vector{Float64}}()
    for k in 0:n-1
        idx = k + 0.5
        phi = acos(1 - 2 * idx / n)
        theta = pi * (1 + sqrt(5)) * idx
        push!(pts, [qmag * sin(phi) * cos(theta), qmag * sin(phi) * sin(theta), qmag * cos(phi)])
    end
    return pts
end

latvecs = [6.0 0 0; 0 10.0 0; 0 0 10.0]            # columns = rows here (diagonal)
cryst = Crystal(latvecs, [[0, 0, 0], [0.5, 0, 0]], 1)
sys = System(cryst, [1 => Moment(s=1, g=2), 2 => Moment(s=1, g=2)], :dipole)
set_exchange!(sys, 1.0, Bond(1, 2, [0, 0, 0]))
set_exchange!(sys, 1.0, Bond(2, 1, [1, 0, 0]))
set_dipole!(sys, [0, 0, 1], (1, 1, 1, 1))
set_dipole!(sys, [0, 0, -1], (1, 1, 1, 2))
swt = SpinWaveTheory(sys; measure=ssf_perp(sys; apply_g=false))

shells = [1.6, 2.6]
qc = vcat([fibonacci(q, 200) for q in shells]...)
rlu = [latvecs' * q / (2pi) for q in qc]
res = intensities_bands(swt, rlu)

out = joinpath(@__DIR__, "data_afm_chain_sunny.json")
open(out, "w") do io
    println(io, "{")
    println(io, "  \"source\": \"Sunny 0.8.1 intensities_bands, ssf_perp(apply_g=false), tests/afm_chain_sunny_reference.jl\",")
    println(io, "  \"shells\": [1.6, 2.6], \"num_samples\": 200,")
    fmt(v) = "[" * join([@sprintf("%.17g", x) for x in v], ", ") * "]"
    println(io, "  \"q_cart\": [", join([fmt(q) for q in qc], ", "), "],")
    println(io, "  \"energies\": [", join([fmt(res.disp[:, k]) for k in 1:length(qc)], ", "), "],")
    println(io, "  \"total_intensity\": ", fmt([sum(real.(res.data[:, k])) for k in 1:length(qc)]))
    println(io, "}")
end
println("wrote ", out, " (", length(qc), " points)")
