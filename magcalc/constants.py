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
# THEY DO NOT AGREE, AND THE VALUES ARE PRESERVED RATHER THAN RECONCILED.
# These two are nominally the same Sunny constant with and without the 4*pi,
# but they were truncated independently:
#
#     MU0_MUB2_MEV_A3 / (4*pi) = 0.053681511234
#     DIPOLE_PREFACTOR_MEV_A3  = 0.05368216
#     -> 6.5e-7 absolute, 1.2e-5 RELATIVE apart
#
# So deriving either from the other is NOT a no-op refactor; it is an accuracy
# change to whichever path gets the new number. That matters concretely:
# `test_truncated_sum_converges_to_ewald` asserts the truncated real-space sum
# reaches the Ewald answer to 1e-4 absolute on a ~4 meV band, and a 1.2e-5
# relative shift is ~5e-5 there -- the same order as the tolerance.
#
# This is the `MU_B` lesson again (see above): a truncation that every pinned
# number was measured against is load-bearing, and "tidying" it silently moves
# results. Reconciling these two -- picking the CODATA value, deriving one from
# the other, and RE-MEASURING the pinned dipolar numbers -- is a deliberate
# change that deserves its own commit and its own oracle run. Until then they
# live here, side by side, where the discrepancy is visible instead of being
# two unrelated-looking literals in two modules.
MU0_MUB2_MEV_A3 = 0.6745817653      # used by ewald.py (exact lattice sum)
DIPOLE_PREFACTOR_MEV_A3 = 0.05368216  # used by generic_model.py (truncated sum)

__all__ = [
    "MU_B",
    "GAMMA_ELECTRON",
    "MU0_MUB2_MEV_A3",
    "DIPOLE_PREFACTOR_MEV_A3",
]
