"""Physical constants shared by every engine.

The point of this module is that there is exactly ONE of each. `MU_B` used to be
a literal `5.788e-2` written out in six places (`generic_model` x2, `spiral_opt`,
`thermal_mc`, `sun/lswt`, `sun/entangled`, `sun/dimer_series`), four of them as a
function-local. Nothing tied them together, so "fixing" the value in one place
would have desynced the engines silently: the field would then differ between the
dipole and SU(N) paths by a few parts in 10^4 -- far too small to notice in a
spectrum, and far too large to be rounding.

Import these; do not re-type the numbers.

    from .constants import MU_B, GAMMA_ELECTRON

`tests/test_combination_matrix.py::test_every_engine_uses_the_same_bohr_magneton`
enforces that, by checking the engines share this object AND that no module has
reintroduced a literal of its own.
"""
import math as _math

# Bohr magneton, meV / T.
#
# This is the CODATA value 5.7883817982e-2 truncated to four figures, i.e. 6.6e-5
# relative. That truncation is DELIBERATE and load-bearing: it is the value every
# pinned Zeeman number in this repo was measured against (SW29's 0.10622/1.72668
# meV, Sunny's 0.231535 meV FM gap at 2 T, the entangled-dimer splittings). Moving
# it to full CODATA precision would shift every in-field energy by 6.6e-5 relative
# and is not a free "accuracy fix" -- re-derive the pinned references first.
MU_B = 5.788e-2

# Electron g-factor. The engines' Zeeman is H = GAMMA_ELECTRON * MU_B * B . S, so
# a field in Tesla reproduces the g = 2 splitting 2*MU_B*B -- the SpinW / Sunny
# convention. With per-site g-tensors the coupling is instead MU_B * B . g_i . S_i
# (an isotropic g = 2 reduces to this exactly; asserted in the tests).
#
# HISTORY (do not halve this again): gamma was once set to 1.0 to compensate a
# "reported doubling" -- which was really the legacy S^0 parameter filter
# double-counting the H_mag*H_dir bilinear term. When the boson-degree truncation
# removed that double count, every in-field result was silently HALVED for months
# (SW29's verified 0.10622/1.72668 became 0.511/1.322; caught 2026-07 by bisect +
# Sunny cross-check).
GAMMA_ELECTRON = 2.0

# --------------------------------------------------------------------------
# Dipolar prefactor: mu0 * mu_B^2, meV * A^3.
#
# RECONCILED 2026-08-15, deliberately and as an ACCURACY CHANGE -- read this
# before "tidying" it further.
#
# The two dipolar paths (ewald.py's exact lattice sum, generic_model.py's
# truncated real-space sum) need the same constant with and without the 4*pi, and
# for a long time they carried two INDEPENDENTLY typed literals that disagreed:
#
#     MU0_MUB2_MEV_A3 = 0.6745817653   (Sunny's value, truncated to 10 figures)
#     DIPOLE_PREFACTOR_MEV_A3 = 0.05368216   (as typed)
#     mu0 mu_B^2 / 4pi        = 0.05368151123615953
#     -> the second was 1.2e-5 RELATIVE too large; not a truncation of the first
#        at all, but a different number.
#
# Which one is right is not a matter of taste: mu0*mu_B^2 is a physical constant,
# Sunny states it to full double precision as 0.6745817653324668
# (Sunny.jl-main/src/Units.jl, `vacuum_permeability`), and 4*pi is exact. So the
# 4-pi-reduced constant is now DERIVED rather than typed, and cannot drift again.
#
# What this moved, measured rather than assumed: only the TRUNCATED sum, by
# -1.2e-5 relative, and in the direction of the Ewald result it is checked
# against -- `test_truncated_sum_converges_to_ewald` compares the two paths to
# 1e-4 absolute on a ~4 meV band, where this is a ~5e-5 shift, i.e. the same
# order as the tolerance. It was re-run before and after (see the test's own
# note) rather than the tolerance being widened to fit. `MU0_MUB2_MEV_A3` itself
# moved by 4.8e-11 relative, going to Sunny's full precision.
#
# This is NOT the `MU_B` case above, and the difference is the point: MU_B's
# four-figure truncation is what every pinned Zeeman number in the repo was
# MEASURED against, so moving it would invalidate those references. Nothing was
# pinned against 0.05368216 -- the only test of the truncated sum is a comparison
# with the Ewald path, which used the other constant. There was no reference to
# preserve, only an inconsistency.
MU0_MUB2_MEV_A3 = 0.6745817653324668   # Sunny Units.jl; used by ewald.py
DIPOLE_PREFACTOR_MEV_A3 = MU0_MUB2_MEV_A3 / (4.0 * _math.pi)  # generic_model.py

__all__ = [
    "MU_B",
    "GAMMA_ELECTRON",
    "MU0_MUB2_MEV_A3",
    "DIPOLE_PREFACTOR_MEV_A3",
]
