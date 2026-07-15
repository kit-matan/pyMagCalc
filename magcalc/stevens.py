"""Stevens operator equivalents O_k^q as classical spin polynomials.

Convention pinned to Sunny 0.8.1 `stevens_matrices(Inf)`, i.e. the large-s limit
in which the spin operators commute and each O_k^q is a HOMOGENEOUS polynomial of
degree k in (Sx, Sy, Sz). That is exactly the object needed by dipole-mode LSWT:
generic_model substitutes the Holstein-Primakoff expansion for Sx/Sy/Sz and the
engine truncates to the quadratic-boson terms.

The table below was generated mechanically from Sunny (not transcribed by hand):

    julia -e 'using Sunny; O = Sunny.stevens_matrices(Inf);
              for k in [2,4,6], q in k:-1:-k; println("$k|$q|", O[k,q]); end'

Note there is no S(S+1) constant and no classical-to-quantum renormalization
factor -- matching pyMagCalc's existing convention of using physical
(un-renormalized) anisotropy values. Sunny's `:dipole` mode applies an extra
renormalization; compare against its `:dipole_uncorrected` mode.
"""
from typing import Any, Dict, Tuple

import sympy as sp

# (k, q) -> polynomial in the symbols Sx, Sy, Sz
STEVENS_POLY: Dict[Tuple[int, int], str] = {
    (2, 2): "Sx**2 - Sy**2",
    (2, 1): "Sz*Sx",
    (2, 0): "-Sx**2 - Sy**2 + 2*Sz**2",
    (2, -1): "Sz*Sy",
    (2, -2): "2*Sy*Sx",
    (4, 4): "Sx**4 - 6*Sy**2*Sx**2 + Sy**4",
    (4, 3): "Sz*Sx**3 - 3*Sz*Sy**2*Sx",
    (4, 2): "-Sx**4 + Sy**4 + 6*Sz**2*Sx**2 - 6*Sz**2*Sy**2",
    (4, 1): "-3*Sz*Sx**3 - 3*Sz*Sy**2*Sx + 4*Sz**3*Sx",
    (4, 0): "3*Sx**4 + 6*Sy**2*Sx**2 + 3*Sy**4 - 24*Sz**2*Sx**2 - 24*Sz**2*Sy**2 + 8*Sz**4",
    (4, -1): "-3*Sz*Sy*Sx**2 - 3*Sz*Sy**3 + 4*Sz**3*Sy",
    (4, -2): "-2*Sy*Sx**3 - 2*Sy**3*Sx + 12*Sz**2*Sy*Sx",
    (4, -3): "3*Sz*Sy*Sx**2 - Sz*Sy**3",
    (4, -4): "4*Sy*Sx**3 - 4*Sy**3*Sx",
    (6, 6): "Sx**6 - 15*Sy**2*Sx**4 + 15*Sy**4*Sx**2 - Sy**6",
    (6, 5): "Sz*Sx**5 - 10*Sz*Sy**2*Sx**3 + 5*Sz*Sy**4*Sx",
    (6, 4): "-Sx**6 + 5*Sy**2*Sx**4 + 5*Sy**4*Sx**2 - Sy**6 + 10*Sz**2*Sx**4 - 60*Sz**2*Sy**2*Sx**2 + 10*Sz**2*Sy**4",
    (6, 3): "-3*Sz*Sx**5 + 6*Sz*Sy**2*Sx**3 + 9*Sz*Sy**4*Sx + 8*Sz**3*Sx**3 - 24*Sz**3*Sy**2*Sx",
    (6, 2): "Sx**6 + Sy**2*Sx**4 - Sy**4*Sx**2 - Sy**6 - 16*Sz**2*Sx**4 + 16*Sz**2*Sy**4 + 16*Sz**4*Sx**2 - 16*Sz**4*Sy**2",
    (6, 1): "5*Sz*Sx**5 + 10*Sz*Sy**2*Sx**3 + 5*Sz*Sy**4*Sx - 20*Sz**3*Sx**3 - 20*Sz**3*Sy**2*Sx + 8*Sz**5*Sx",
    (6, 0): "-5*Sx**6 - 15*Sy**2*Sx**4 - 15*Sy**4*Sx**2 - 5*Sy**6 + 90*Sz**2*Sx**4 + 180*Sz**2*Sy**2*Sx**2 + 90*Sz**2*Sy**4 - 120*Sz**4*Sx**2 - 120*Sz**4*Sy**2 + 16*Sz**6",
    (6, -1): "5*Sz*Sy*Sx**4 + 10*Sz*Sy**3*Sx**2 + 5*Sz*Sy**5 - 20*Sz**3*Sy*Sx**2 - 20*Sz**3*Sy**3 + 8*Sz**5*Sy",
    (6, -2): "2*Sy*Sx**5 + 4*Sy**3*Sx**3 + 2*Sy**5*Sx - 32*Sz**2*Sy*Sx**3 - 32*Sz**2*Sy**3*Sx + 32*Sz**4*Sy*Sx",
    (6, -3): "-9*Sz*Sy*Sx**4 - 6*Sz*Sy**3*Sx**2 + 3*Sz*Sy**5 + 24*Sz**3*Sy*Sx**2 - 8*Sz**3*Sy**3",
    (6, -4): "-4*Sy*Sx**5 + 4*Sy**5*Sx + 40*Sz**2*Sy*Sx**3 - 40*Sz**2*Sy**3*Sx",
    (6, -5): "5*Sz*Sy*Sx**4 - 10*Sz*Sy**3*Sx**2 + Sz*Sy**5",
    (6, -6): "6*Sy*Sx**5 - 20*Sy**3*Sx**3 + 6*Sy**5*Sx",
}

STEVENS_KS = (2, 4, 6)


def stevens_polynomial(k: int, q: int, Sx: Any, Sy: Any, Sz: Any) -> Any:
    """O_k^q evaluated on the given spin components (symbols or expressions).

    k must be 2, 4 or 6 (the orders Sunny supports) and -k <= q <= k. k is even
    because time-reversal symmetry forbids odd-rank crystal-field terms.
    """
    key = (int(k), int(q))
    if key not in STEVENS_POLY:
        raise ValueError(
            f"Stevens operator O_{k}^{q} is not available. Supported: k in "
            f"{STEVENS_KS} (even orders only, time-reversal), -k <= q <= k."
        )
    sx, sy, sz = sp.symbols("Sx Sy Sz")
    expr = sp.sympify(STEVENS_POLY[key], locals={"Sx": sx, "Sy": sy, "Sz": sz})
    return expr.subs({sx: Sx, sy: Sy, sz: Sz}, simultaneous=True)
