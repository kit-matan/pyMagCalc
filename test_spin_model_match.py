import sympy as sp
import numpy as np
import yaml
import logging

import spin_model as sm_orig
import generic_spin_model as gsm

gsm_logger = logging.getLogger("generic_spin_model")
gsm_logger.setLevel(logging.INFO)
if not gsm_logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    gsm_logger.addHandler(ch)

J1s, J2s, Dys, Dzs, Hs = sp.symbols("J1 J2 Dy Dz H")
params_list_orig = [J1s, J2s, Dys, Dzs, Hs]
symbolic_params_map_generic = {"J1": J1s, "J2": J2s, "Dy": Dys, "Dz": Dzs, "H": Hs}


def compare_sympy_expr(expr1, expr2, tol=1e-12, context=""):
    # Using expand().simplify() for more robust comparison
    simplified_diff = (expr1 - expr2).expand().simplify()
    if simplified_diff == sp.S.Zero:
        return True
    try:
        numerical_diff = simplified_diff.evalf(n=5, strict=False)
        if abs(numerical_diff) < tol:
            return True
    except (AttributeError, TypeError, NotImplementedError):
        pass
    # For debugging the difference if it's not zero
    # if context == "Hamiltonian":
    #     print(f"DEBUG HAM PRINT: Simplified Diff for Hamiltonian: {simplified_diff}")
    return False


print("Original model computations...")
mpr_orig_list = sm_orig.mpr(params_list_orig)
Jex_orig, DM_orig_mat = sm_orig.spin_interactions(params_list_orig)
apos_orig_ouc_coords = sm_orig.atom_pos_ouc()
nspins_orig_ouc = len(apos_orig_ouc_coords)
S_ops_list_shared = [
    sp.Matrix([sp.Symbol(f"Sx_{i}"), sp.Symbol(f"Sy_{i}"), sp.Symbol(f"Sz_{i}")])
    for i in range(nspins_orig_ouc)
]
Sxyz_orig_for_Ham = S_ops_list_shared
Ham_orig = sm_orig.Hamiltonian(Sxyz_orig_for_Ham, params_list_orig)
print("Original Hamiltonian construction done.")

print("\nGeneric model computations...")
with open("KFe3J_config.yaml", "r") as f:
    config_data = yaml.safe_load(f)

uc_vecs_generic = gsm.unit_cell_from_config(config_data["crystal_structure"])
atom_pos_uc_generic = gsm.atom_pos_from_config(
    config_data["crystal_structure"], uc_vecs_generic
)
atom_pos_ouc_generic_coords = gsm.atom_pos_ouc_from_config(
    atom_pos_uc_generic, uc_vecs_generic, config_data.get("calculation_settings", {})
)
nspins_generic_ouc = len(atom_pos_ouc_generic_coords)
mpr_generic_list = gsm.mpr_from_config(
    config_data["crystal_structure"], symbolic_params_map_generic
)
Jex_generic, DM_generic_mat = gsm.spin_interactions_from_config(
    symbolic_params_map_generic,
    config_data["interactions"],
    atom_pos_uc_generic,
    atom_pos_ouc_generic_coords,
    uc_vecs_generic,
)
Sxyz_generic_for_Ham = S_ops_list_shared
Ham_generic = gsm.Hamiltonian_from_config(
    Sxyz_generic_for_Ham, symbolic_params_map_generic, config_data
)
print("Generic Hamiltonian construction done.")

print("\n--- Comparisons ---")
print(f"Num OUC atoms: Original={nspins_orig_ouc}, Generic={nspins_generic_ouc}")
assert (
    nspins_orig_ouc == nspins_generic_ouc
), f"Mismatch in OUC atoms: Orig {nspins_orig_ouc}, Gen {nspins_generic_ouc}"
assert np.allclose(
    apos_orig_ouc_coords, atom_pos_ouc_generic_coords, atol=gsm.DIST_TOL
), "OUC atom positions do not match."
print("OUC atom positions match: True")

mpr_overall_match = True
assert len(mpr_orig_list) == len(mpr_generic_list), "MPR list length mismatch"
for i in range(len(mpr_orig_list)):
    matrix_match_current = True
    for r_mpr in range(mpr_orig_list[i].rows):
        for c_mpr in range(mpr_orig_list[i].cols):
            if not compare_sympy_expr(
                mpr_orig_list[i][r_mpr, c_mpr], mpr_generic_list[i][r_mpr, c_mpr]
            ):
                matrix_match_current = False
                break
        if not matrix_match_current:
            break
    if not matrix_match_current:
        mpr_overall_match = False
        print(f"Error: MPR matrix {i} mismatch.")
        break
print(f"MPR matrices match: {mpr_overall_match}")
assert mpr_overall_match, "MPR matrices do not match."

jex_mismatches = []
Jex_overall_match = True
for r_idx in range(Jex_orig.rows):
    for c_idx in range(Jex_orig.cols):
        orig_val = Jex_orig[r_idx, c_idx]
        gen_val = Jex_generic[r_idx, c_idx]
        if not compare_sympy_expr(orig_val, gen_val, context=f"Jex[{r_idx},{c_idx}]"):
            Jex_overall_match = False
            jex_mismatches.append(
                {
                    "index": (r_idx, c_idx),
                    "original": orig_val,
                    "generic": gen_val,
                    "diff": (orig_val - gen_val).expand().simplify(),
                }
            )

print(f"Jex matrices overall match: {Jex_overall_match}")
if not Jex_overall_match:
    print("\n--- Jex Mismatches ---")
    for mismatch in jex_mismatches:
        print(
            f"Index: {mismatch['index']}, Original: {mismatch['original']}, Generic: {mismatch['generic']}, Diff: {mismatch['diff']}"
        )
else:
    print("Jex matrices successfully match!")

if Jex_overall_match:
    dm_mismatches = []
    DM_overall_match = True

    print(
        f"\nType of DM_orig_mat: {type(DM_orig_mat)}, Shape: {DM_orig_mat.shape if hasattr(DM_orig_mat, 'shape') else 'No shape'}"
    )
    print(
        f"Type of DM_generic_mat: {type(DM_generic_mat)}, Shape: {DM_generic_mat.shape if hasattr(DM_generic_mat, 'shape') else 'No shape'}"
    )

    if hasattr(DM_orig_mat, "shape") and DM_orig_mat.rows > 0 and DM_orig_mat.cols > 0:
        print(
            f"Type of DM_orig_mat[0,0]: {type(DM_orig_mat[0,0])}, Shape of DM_orig_mat[0,0]: {DM_orig_mat[0,0].shape if hasattr(DM_orig_mat[0,0], 'shape') else 'No shape (element)'}"
        )
    if (
        hasattr(DM_generic_mat, "shape")
        and DM_generic_mat.rows > 0
        and DM_generic_mat.cols > 0
    ):
        print(
            f"Type of DM_generic_mat[0,0]: {type(DM_generic_mat[0,0])}, Shape of DM_generic_mat[0,0]: {DM_generic_mat[0,0].shape if hasattr(DM_generic_mat[0,0], 'shape') else 'No shape (element)'}"
        )

    assert DM_orig_mat.shape == DM_generic_mat.shape, "DM matrix (outer) shape mismatch"

    EXPECTED_DM_VECTOR_SHAPE = (3, 1)

    for r in range(DM_orig_mat.rows):
        for c in range(DM_orig_mat.cols):
            dm_vec_orig = DM_orig_mat[r, c]
            dm_vec_gen = DM_generic_mat[r, c]

            if not (
                hasattr(dm_vec_orig, "shape")
                and dm_vec_orig.shape == EXPECTED_DM_VECTOR_SHAPE
                and hasattr(dm_vec_gen, "shape")
                and dm_vec_gen.shape == EXPECTED_DM_VECTOR_SHAPE
            ):
                DM_overall_match = False
                dm_mismatches.append(
                    {
                        "index": (r, c, "vector_shape"),
                        "original": f"Shape {dm_vec_orig.shape if hasattr(dm_vec_orig,'shape') else type(dm_vec_orig)}",
                        "generic": f"Shape {dm_vec_gen.shape if hasattr(dm_vec_gen,'shape') else type(dm_vec_gen)}",
                        "diff": f"DM vector element shape mismatch (expected {EXPECTED_DM_VECTOR_SHAPE})",
                    }
                )
                continue

            for k_row in range(EXPECTED_DM_VECTOR_SHAPE[0]):
                orig_dm_comp = dm_vec_orig[k_row, 0]
                gen_dm_comp = dm_vec_gen[k_row, 0]
                if not compare_sympy_expr(
                    orig_dm_comp, gen_dm_comp, context=f"DM[{r},{c}][{k_row},0]"
                ):
                    DM_overall_match = False
                    dm_mismatches.append(
                        {
                            "index": (r, c, k_row),
                            "original": orig_dm_comp,
                            "generic": gen_dm_comp,
                            "diff": (orig_dm_comp - gen_dm_comp).expand().simplify(),
                        }
                    )

    print(f"\nDM matrices overall match: {DM_overall_match}")
    if not DM_overall_match:
        print("\n--- DM Mismatches ---")
        for mismatch in dm_mismatches:
            if mismatch["index"][-1] == "vector_shape":
                print(
                    f"Index: DM[{mismatch['index'][0]},{mismatch['index'][1]}], Original: {mismatch['original']}, Generic: {mismatch['generic']}, Diff: {mismatch['diff']}"
                )
            else:
                print(
                    f"Index: DM[{mismatch['index'][0]},{mismatch['index'][1]}][{mismatch['index'][2]},0], Original: {mismatch['original']}, Generic: {mismatch['generic']}, Diff: {mismatch['diff']}"
                )
    else:
        print("DM matrices successfully match!")
else:
    print("\nSkipping DM check due to Jex mismatches.")

if Jex_overall_match and ("DM_overall_match" in locals() and DM_overall_match):
    # Use (expr1 - expr2).expand().simplify() for Hamiltonian comparison
    ham_diff_expanded_simplified = (Ham_orig - Ham_generic).expand().simplify()
    Hamiltonian_match_final = ham_diff_expanded_simplified == sp.S.Zero

    print(f"\nHamiltonians match: {Hamiltonian_match_final}")
    if not Hamiltonian_match_final:
        print("Error: Hamiltonians mismatch.")
        print(
            "Hamiltonian difference (expanded and simplified):",
            ham_diff_expanded_simplified,
        )
        assert False, "Hamiltonians do not match."
    else:
        print("Hamiltonians successfully match!")
        print(
            "\nAll checks passed successfully! The generic model with KFe3J_config.txt matches spin_model.py."
        )
else:
    print("\nSkipping Hamiltonian check due to Jex or DM mismatches.")

if (
    not Jex_overall_match
    or ("DM_overall_match" in locals() and not DM_overall_match)
    or ("Hamiltonian_match_final" in locals() and not Hamiltonian_match_final)
):
    print("\nThere were mismatches. Please review the output.")
    assert False, "One or more components (Jex, DM, Hamiltonian) have mismatches."
