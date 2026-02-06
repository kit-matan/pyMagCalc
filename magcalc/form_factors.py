import numpy as np
import logging

logger = logging.getLogger(__name__)

# Data from International Tables for Crystallography Vol C, Table 4.4.4.1
# Formula for j0(s): j0(s) = A*exp(-a*s^2) + B*exp(-b*s^2) + C*exp(-c*s^2) + D
# s = sin(theta)/lambda = Q / (4*pi)

FORM_FACTOR_COEFFICIENTS = {
    # 3d ions (j0)
    "Ti3+": {"A": 0.4391, "a": 12.1009, "B": 0.5238, "b": 5.1517, "C": 0.0521, "c": 0.1703, "D": -0.0152},
    "V4+":  {"A": 0.4026, "a": 15.6558, "B": 0.5404, "b": 6.3054, "C": 0.0716, "c": 0.2312, "D": -0.0145},
    "V3+":  {"A": 0.3542, "a": 14.8690, "B": 0.5752, "b": 6.1360, "C": 0.0886, "c": 0.1982, "D": -0.0180},
    "V2+":  {"A": 0.2882, "a": 14.2863, "B": 0.6139, "b": 5.9238, "C": 0.1177, "c": 0.1558, "D": -0.0198},
    "Cr3+": {"A": 0.2974, "a": 19.4678, "B": 0.6094, "b": 7.7348, "C": 0.1147, "c": 0.2505, "D": -0.0215},
    "Cr2+": {"A": 0.2223, "a": 18.2325, "B": 0.6553, "b": 7.3341, "C": 0.1481, "c": 0.1915, "D": -0.0257},
    "Mn4+": {"A": 0.2238, "a": 23.4913, "B": 0.6559, "b": 9.2452, "C": 0.1444, "c": 0.3013, "D": -0.0241},
    "Mn3+": {"A": 0.1524, "a": 21.3653, "B": 0.7067, "b": 8.7188, "C": 0.1764, "c": 0.2238, "D": -0.0355},
    "Mn2+": {"A": 0.1084, "a": 20.3547, "B": 0.7410, "b": 8.3619, "C": 0.1989, "c": 0.1805, "D": -0.0483},
    "Fe3+": {"A": 0.0626, "a": 27.2721, "B": 0.7554, "b": 10.3800, "C": 0.2464, "c": 0.2797, "D": -0.0644},
    "Fe2+": {"A": 0.0142, "a": 24.3639, "B": 0.7853, "b": 9.9407, "C": 0.2936, "c": 0.2111, "D": -0.0931},
    "Co2+": {"A": -0.0556, "a": 34.6983, "B": 0.8118, "b": 11.8315, "C": 0.3571, "c": 0.2829, "D": -0.1133},
    "Ni2+": {"A": -0.1986, "a": 54.4373, "B": 0.8647, "b": 13.5654, "C": 0.4578, "c": 0.3805, "D": -0.1239},
    "Cu2+": {"A": -0.1561, "a": 63.3630, "B": 0.8523, "b": 14.8698, "C": 0.4851, "c": 0.5065, "D": -0.1813},
}

def get_j0(ion, Q_mag):
    """
    Calculate j0(s) for a given ion and Q magnitude.
    s = Q / (4 * pi)
    """
    if ion not in FORM_FACTOR_COEFFICIENTS:
        # Try to strip charge if not found (e.g. "Fe3+" -> "Fe")
        # However, coefficients are usually specific to oxidation state.
        # Fallback to a neutral atom or warn.
        logger.warning(f"Ion '{ion}' not found in form factor database. Returning 1.0.")
        return 1.0
    
    coeffs = FORM_FACTOR_COEFFICIENTS[ion]
    s = Q_mag / (4.0 * np.pi)
    s2 = s**2
    
    j0 = (coeffs["A"] * np.exp(-coeffs["a"] * s2) +
          coeffs["B"] * np.exp(-coeffs["b"] * s2) +
          coeffs["C"] * np.exp(-coeffs["c"] * s2) +
          coeffs["D"])
    return j0

def get_form_factor(ion, Q_mag, g=2.0):
    """
    Calculate the magnetic form factor F(Q).
    Default formula using only j0 (g=2 approximation).
    In the future, j2 could be added for better accuracy if coefficients are provided.
    F(Q) = j0(s) + (g-2)/g * j2(s). For g=2, F(Q) = j0(s).
    """
    # For now, we only implement j0
    return get_j0(ion, Q_mag)
