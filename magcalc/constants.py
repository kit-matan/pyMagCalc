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

# NOT here (yet): the dipolar prefactor, which is duplicated in the same way --
# `ewald.MU0_MUB2_MEV_A3 = 0.6745817653` and
# `generic_model.DIPOLE_PREFACTOR_MEV_A3 = 0.05368216` are the same Sunny constant
# with and without the 4*pi. Both are module-level and cross-referenced in comments,
# so they are far less liable to drift than the mu_B literals were; folding them in
# here is a separate, easy cleanup.

__all__ = ["MU_B", "GAMMA_ELECTRON"]
