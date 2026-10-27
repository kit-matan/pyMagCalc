# Regenerate tests/data_ewald_sunny.json: the dipolar interaction tensor A_ij(q) as
# Sunny 0.8.1 computes it, for tests/test_ewald_sunny_tensor.py.
#
#   JULIA_PKG_OFFLINE=true julia --project=@v1.12 tests/ewald_sunny_reference.jl
#
# (Sunny 0.8.1 is installed in the v1.12 environment on the maintainer's machine; any
# environment with Sunny 0.8.1 works.)
#
# What is stored is Sunny's own `precompute_dipole_ewald_at_wavevector` for a single
# magnetic cell, dims = (1,1,1), WITHOUT the mu0*muB^2 prefactor and in Sunny's phase
# convention (phases over lattice translations only). The test applies both to compare
# with pyMagCalc's `ewald`, which returns meV/muB^2 in the full-bond gauge:
#
#   A_pymagcalc_ij(q) = mu0*muB^2 * A_sunny_ij(q) * exp(i q . (r_j - r_i))
#
# The three cells exercise every branch of the sum: a non-orthogonal multi-site cell,
# a triclinic cell with a non-trivial demagnetization tensor, and the one-site chain
# of tests/test_ewald.py. The q points include Gamma (the k = 0 surface term), generic
# points, a zone-boundary point, and points outside the first zone (the q folding).
using Sunny, LinearAlgebra, Printf

const MODELS = [
    (name = "monoclinic_3site",
     # rows are lattice vectors (pyMagCalc convention)
     latvecs = [5.1 0.0 0.0; 0.9 6.3 0.0; 0.0 0.0 4.4],
     positions = [[0.0, 0.0, 0.0], [0.31, 0.12, 0.47], [0.62, 0.71, 0.18]],
     demag = Matrix(I, 3, 3) / 3),
    (name = "triclinic_2site",
     latvecs = [4.2 0.0 0.0; 1.1 5.0 0.0; 0.7 -0.6 5.8],
     positions = [[0.05, 0.10, 0.20], [0.55, 0.40, 0.65]],
     demag = [0.2 0.0 0.0; 0.0 0.3 0.0; 0.0 0.0 0.5]),
    (name = "chain_1site",
     latvecs = [3.0 0.0 0.0; 0.0 10.0 0.0; 0.0 0.0 10.0],
     positions = [[0.0, 0.0, 0.0]],
     demag = Matrix(I, 3, 3) / 3),
    # Irregular 8-site cell: exercises the pair bookkeeping of the precomputed sum at
    # a size where an indexing slip between pairs could not hide.
    (name = "orthorhombic_8site",
     latvecs = [6.2 0.0 0.0; 0.0 7.1 0.0; 0.0 0.0 5.3],
     positions = [[0.0, 0.0, 0.0], [0.5, 0.11, 0.07], [0.13, 0.52, 0.21], [0.61, 0.58, 0.33],
                  [0.27, 0.19, 0.55], [0.74, 0.31, 0.62], [0.08, 0.83, 0.71], [0.42, 0.66, 0.91]],
     demag = [0.25 0.05 0.0; 0.05 0.35 0.0; 0.0 0.0 0.4]),
]
const QS = [[0.0, 0.0, 0.0], [0.13, 0.27, 0.41], [0.5, 0.0, 0.25],
            [1.2, -0.3, 0.7], [-0.45, 0.5, 0.5]]

function jarr(x)            # Julia matrix -> nested vector, row-major
    [collect(x[i, :]) for i in axes(x, 1)]
end

open(joinpath(@__DIR__, "data_ewald_sunny.json"), "w") do io
    println(io, "{")
    println(io, "  \"sunny_version\": \"", pkgversion(Sunny), "\",")
    println(io, "  \"generator\": \"tests/ewald_sunny_reference.jl\",")
    println(io, "  \"models\": [")
    for (mi, m) in enumerate(MODELS)
        # Sunny stores lattice vectors as COLUMNS; spacegroup 1 keeps the sites as given.
        cryst = Crystal(Matrix(m.latvecs'), [Sunny.Vec3(p...) for p in m.positions], 1)
        @assert length(cryst.positions) == length(m.positions)
        for (a, b) in zip(cryst.positions, m.positions)
            @assert norm(a - b) < 1e-12 "Sunny reordered or moved a site"
        end
        na = length(m.positions)
        println(io, "    {\"name\": \"", m.name, "\",")
        println(io, "     \"latvecs\": ", jarr(m.latvecs), ",")
        println(io, "     \"positions\": ", m.positions, ",")
        println(io, "     \"demag\": ", jarr(m.demag), ",")
        println(io, "     \"q_rlu\": ", QS, ",")
        println(io, "     \"A_re\": [")
        blocks_re, blocks_im = String[], String[]
        for q in QS
            A = Sunny.precompute_dipole_ewald_at_wavevector(cryst, (1, 1, 1), Sunny.Mat3(m.demag), Sunny.Vec3(q...))
            re = [[[[real(A[1, 1, 1, i, j][a, b]) for b in 1:3] for a in 1:3] for j in 1:na] for i in 1:na]
            im = [[[[imag(A[1, 1, 1, i, j][a, b]) for b in 1:3] for a in 1:3] for j in 1:na] for i in 1:na]
            push!(blocks_re, string(re)); push!(blocks_im, string(im))
        end
        println(io, "       ", join(blocks_re, ",\n       "), "],")
        println(io, "     \"A_im\": [")
        println(io, "       ", join(blocks_im, ",\n       "), "]}", mi < length(MODELS) ? "," : "")
    end
    println(io, "  ]")
    println(io, "}")
end
println("wrote data_ewald_sunny.json")
