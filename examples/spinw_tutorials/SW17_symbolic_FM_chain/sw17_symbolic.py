"""SW17 - symbolic spin-wave dispersion of the ferromagnetic chain.

Port of SpinW Tutorial 17. pyMagCalc constructs the LSWT (Bogoliubov)
Hamiltonian symbolically in SymPy before lambdifying, so the analytic
magnon dispersion can be read straight out of `MagCalc.HMat_sym` -- the
pyMagCalc analogue of SpinW's `spinwavesym`.

Run from the pyMagCalc repo root:
    python examples/spinw_tutorials/SW17_symbolic_FM_chain/sw17_symbolic.py
"""
import os
import sympy as sp
import yaml

import magcalc as mc
from magcalc.generic_model import GenericSpinModel

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
    model = GenericSpinModel(cfg)
    thetas, phis = model.generate_magnetic_structure()
    model.set_magnetic_structure(thetas, phis)

    calc = mc.MagCalc(spin_model_module=model, spin_magnitude=1.0,
                      hamiltonian_params=[-1.0],
                      cache_file_base=os.path.join(HERE, ".sw17"), cache_mode="none")

    H = calc.HMat_sym                      # 2gH Bogoliubov matrix (symbolic)
    S = calc.S_sym                         # spin magnitude symbol
    J = calc.params_sym_flat[0]            # exchange symbol (p0)
    kx = calc.kx                           # Cartesian momentum symbol

    # The FM chain gives a diagonal 2x2 2gH; the physical magnon energy is
    # the positive eigenvalue. a = 3 A, so q.a = 3*kx.
    omega = sp.simplify(sp.Abs(H[1, 1]))
    omega_readable = 2 * S * sp.Abs(J) * (1 - sp.cos(3 * kx))

    print("Symbolic 2gH Hamiltonian entries:")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            print(f"  H[{i},{j}] = {sp.simplify(H[i, j])}")
    print()
    print(f"Analytic magnon dispersion (a = 3 A):")
    print(f"  E(q) = |H[1,1]| = {omega}")
    print(f"       = {omega_readable}")
    print(f"       = 2*S*|J|*(1 - cos(q_x * a))")
    print()
    print("Zone centre  q=0        : E = 0")
    print("Zone boundary q_x=pi/a  : E = 4*S*|J|   (= 4 meV for S=1, |J|=1)")


if __name__ == "__main__":
    main()
