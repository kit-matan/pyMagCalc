# Regenerate tests/data_g_tensor_sqw_sunny.json: Sunny 0.8.1's one-magnon intensities
# measured on the MOMENT g.S (`ssf_perp(sys; apply_g=true)`), for
# tests/test_sqw_batch.py (MagCalc.sqw_batch(..., apply_g=True)).
#
#   JULIA_PKG_OFFLINE=true julia --project=@v1.12 tests/g_tensor_sqw_sunny_reference.jl
#
# The model is chosen so that nothing about the g tensor can cancel:
#   - a ferrimagnetic chain, S_A = 1/2 (Cu2+) and S_B = 1 (Ni2+), J = 1.3 meV on the
#     bond between them, so the two branches are non-degenerate and each weights the
#     two sites differently;
#   - the moments along a generic axis n = (0.36, -0.48, 0.8), not a lattice axis, so
#     the local-to-lab rotation is not the identity;
#   - a different, NON-SYMMETRIC g tensor with off-diagonal entries on each site, so a
#     transposed or site-swapped g changes the answer;
#   - Sunny's form factors for both ions (g_lande = 2, i.e. <j0>), so the form factor
#     and the g tensor must attach to the same site.
# The g tensors do not enter the Hamiltonian (no field), only the measurement.
using Sunny, LinearAlgebra, Printf

latvecs = [6.0 0 0; 0 9.0 0; 0 0 9.0]
cryst = Crystal(latvecs, [[0, 0, 0], [0.5, 0, 0]], 1; types=["A", "B"])
gA = [2.10 0.00 0.30; 0.05 1.95 -0.12; 0.25 0.10 2.20]
gB = [2.30 -0.20 0.00; 0.15 2.05 0.40; -0.10 0.35 1.90]
sys = System(cryst, [1 => Moment(s=1/2, g=gA), 2 => Moment(s=1, g=gB)], :dipole_uncorrected)
J = 1.3
set_exchange!(sys, J, Bond(1, 2, [0, 0, 0]))
set_exchange!(sys, J, Bond(2, 1, [1, 0, 0]))
n = normalize([0.36, -0.48, 0.8])
set_dipole!(sys, n, (1, 1, 1, 1))
set_dipole!(sys, -n, (1, 1, 1, 2))

formfactors = [1 => FormFactor("Cu2"), 2 => FormFactor("Ni2")]
rlu = [[0.13, 0.0, 0.0], [0.31, 0.2, 0.0], [0.47, 0.0, 0.3], [0.77, 0.1, 0.4],
       [1.21, 0.0, 0.0], [1.5, 0.3, 0.2], [2.35, -0.4, 0.15], [0.9, 0.6, -0.7]]
out_g = intensities_bands(SpinWaveTheory(sys; measure=ssf_perp(sys; apply_g=true, formfactors)), rlu)
out_s = intensities_bands(SpinWaveTheory(sys; measure=ssf_perp(sys; apply_g=false, formfactors)), rlu)

out = joinpath(@__DIR__, "data_g_tensor_sqw_sunny.json")
fmt(v) = "[" * join([@sprintf("%.17g", x) for x in v], ", ") * "]"
mat(m) = "[" * join([fmt(m[i, :]) for i in 1:3], ", ") * "]"
open(out, "w") do io
    println(io, "{")
    println(io, "  \"source\": \"Sunny 0.8.1 intensities_bands, ssf_perp(apply_g=true/false, formfactors), tests/g_tensor_sqw_sunny_reference.jl\",")
    println(io, "  \"lattice\": ", mat(latvecs), ",")
    println(io, "  \"J\": ", J, ", \"spins\": [0.5, 1.0], \"ions\": [\"Cu2+\", \"Ni2+\"],")
    println(io, "  \"axis\": ", fmt(n), ",")
    println(io, "  \"g\": [", mat(gA), ", ", mat(gB), "],")
    println(io, "  \"q_rlu\": [", join([fmt(q) for q in rlu], ", "), "],")
    println(io, "  \"energies\": [", join([fmt(out_g.disp[:, k]) for k in 1:length(rlu)], ", "), "],")
    println(io, "  \"intensity_apply_g\": [", join([fmt(real.(out_g.data[:, k])) for k in 1:length(rlu)], ", "), "],")
    println(io, "  \"intensity_bare\": [", join([fmt(real.(out_s.data[:, k])) for k in 1:length(rlu)], ", "), "]")
    println(io, "}")
end
println("wrote ", out)
