import sympy as sp
from sympy import I, Add
import numpy as np
import timeit
import logging
import os  # for cpu_count
from multiprocessing import Pool
from typing import List, Tuple, Dict, Any, Optional, Union

# Helper for type hinting
import numpy.typing as npt

logger = logging.getLogger(__name__)

def _setup_hp_operators(
    nspins_ouc: int, S_sym: sp.Symbol
) -> Tuple[List[sp.Symbol], List[sp.Symbol], List[sp.Matrix]]:
    """
    Create symbolic Holstein-Primakoff boson operators and local spin operators.

    Args:
        nspins_ouc (int): Number of spins in the original unit cell + neighbors (OUC).
        S_sym (sp.Symbol): Symbolic representation of the spin magnitude S.
    Returns:
        Tuple[List[sp.Symbol], List[sp.Symbol], List[sp.Matrix]]: Boson annihilation operators (c), creation operators (cd), and local spin operators (Sx, Sy, Sz) expressed in terms of bosons.
    """
    c_ops = sp.symbols("c0:%d" % nspins_ouc, commutative=False)
    cd_ops = sp.symbols("cd0:%d" % nspins_ouc, commutative=False)
    spin_ops_local = [
        sp.Matrix(
            (
                sp.sqrt(S_sym / 2) * (c_ops[i] + cd_ops[i]),
                sp.sqrt(S_sym / 2) * (c_ops[i] - cd_ops[i]) / I,
                S_sym - cd_ops[i] * c_ops[i],
            )
        )
        for i in range(nspins_ouc)
    ]
    return c_ops, cd_ops, spin_ops_local


def _rotate_spin_operators(
    spin_ops_local: List[sp.Matrix],
    rotation_matrices: List[Union[npt.NDArray, sp.Matrix]],
    nspins: int,
    nspins_ouc: int,
) -> List[sp.Matrix]:
    """
    Rotate local spin operators (defined along local z-axis) to the global frame.

    Args:
        spin_ops_local (List[sp.Matrix]): List of local spin operators (3x1 matrices) for each spin in OUC.
        rotation_matrices (List[Union[npt.NDArray, sp.Matrix]]): List of rotation matrices (3x3) for each spin in the magnetic unit cell. These are applied cyclically to the OUC spins.
        nspins (int): Number of spins in the magnetic unit cell.
        nspins_ouc (int): Number of spins in the OUC.
    Returns:
        List[sp.Matrix]: List of global spin operators (3x1 matrices) for each spin in OUC.
    """
    spin_ops_global_ouc = [
        rotation_matrices[j] * spin_ops_local[nspins * i + j]
        for i in range(int(nspins_ouc / nspins))
        for j in range(nspins)
    ]
    return spin_ops_global_ouc


def _prepare_hamiltonian(
    spin_model_module: Any,
    spin_ops_global_ouc: List[sp.Matrix],
    params_sym: List[sp.Symbol],
    S_sym: sp.Symbol,
) -> sp.Expr:
    """
    Construct the symbolic Hamiltonian using the user-defined model.

    Retrieves the Hamiltonian expression from the `spin_model_module`, expands it,
    and attempts to filter out terms higher than quadratic in boson operators
    by analyzing powers of the symbolic spin S.

    Args:
        spin_model_module: The user-provided spin model module.
        spin_ops_global_ouc (List[sp.Matrix]): List of global spin operators for OUC spins.
        params_sym (List[sp.Symbol]): List of symbolic Hamiltonian parameters.
        S_sym (sp.Symbol): Symbolic spin magnitude.
    Returns:
        sp.Expr: The expanded and filtered symbolic Hamiltonian expression (up to quadratic boson terms).
    """
    hamiltonian_sym = spin_model_module.Hamiltonian(spin_ops_global_ouc, params_sym)
    hamiltonian_sym = sp.expand(hamiltonian_sym)
    # --- Filter Hamiltonian terms (Keep only up to quadratic in boson ops) ---
    # Check if S_sym is present. If not (substituted), we assume filtering was done by module.
    if not hamiltonian_sym.has(S_sym):
        logger.info("S_sym not found in Hamiltonian (likely substituted). Skipping S-power filtering.")
    else:
        # Legacy filtering logic
        # This logic seems specific to how the Hamiltonian is constructed in the model.
        # It assumes higher powers of S correspond to higher orders in boson operators.
        # A more robust approach might involve explicitly checking boson operator powers.
        hamiltonian_S0 = hamiltonian_sym.coeff(S_sym, 0)
        if params_sym:
            # This part seems potentially problematic or overly specific.
            # It keeps the term linear in the *last* parameter from the S^0 part,
            # plus the S^1 and S^2 terms. Revisit if this causes issues.
            
            # Fix: Ensure params_sym[-1] is a symbol (scalar) before using it in coeff/mul.
            # If it's a list (vector value), this logic is invalid/unnecessary for that param.
            if isinstance(params_sym[-1], sp.Symbol):
                 term_last_param = hamiltonian_S0.coeff(params_sym[-1]) * params_sym[-1]
            else:
                 term_last_param = 0

            hamiltonian_sym = (
                term_last_param
                + hamiltonian_sym.coeff(S_sym, 1) * S_sym
                + hamiltonian_sym.coeff(S_sym, 2) * S_sym**2
            )
        else:
            hamiltonian_sym = (
                hamiltonian_sym.coeff(S_sym, 1) * S_sym
                + hamiltonian_sym.coeff(S_sym, 2) * S_sym**2
            )
    hamiltonian_sym = sp.expand(hamiltonian_sym)
    # --- End Filtering ---
    return hamiltonian_sym


def _is_nonzero_at(matrix, i, j):
    """Robustly check if a matrix/list-of-lists has a non-zero value at (i, j)."""
    if matrix is None:
        return False
    try:
        # Try SymPy/NumPy matrix indexing M[i, j]
        val = matrix[i, j]
    except:
        try:
            # Try list-of-lists indexing M[i][j]
            val = matrix[i][j]
        except:
            return False
            
    if val is None or val == 0:
        return False
    # If it's a vector (3x1 Matrix), check its content
    if hasattr(val, "is_zero_matrix"):
        return not val.is_zero_matrix
    return True

def _generate_fourier_lookup(
    spin_model_module,
    k_sym: List[sp.Symbol],
    nspins: int,
    nspins_ouc: int,
    c_ops: List[sp.Symbol],
    cd_ops: List[sp.Symbol],
    ck_ops: List[sp.Symbol],
    ckd_ops: List[sp.Symbol],
    cmk_ops: List[sp.Symbol],
    cmkd_ops: List[sp.Symbol],
    params_sym: List[sp.Symbol],
) -> Dict[Tuple[str, str], sp.Expr]:
    """
    Generate dictionary for Fourier Transform substitutions.
    
    Returns:
        Dict[(str, str), sp.Expr]: Map from (OpName1, OpName2) to k-space expression.
    """
    atom_positions_uc = spin_model_module.atom_pos()
    atom_positions_ouc = spin_model_module.atom_pos_ouc()
    
    # Flexible unpacking: accommodate models returning 1, 2, 3 or more values.
    # Usually J, DM, K, H_vec, etc.
    res = spin_model_module.spin_interactions(params_sym)
    if not isinstance(res, (list, tuple)):
        res = (res,)
        
    # We only care about interaction matrices (shape nspins x nspins_ouc)
    interaction_matrices = []
    for item in res:
        # Heuristic: interaction matrices have nspins "rows"
        try:
            if hasattr(item, "rows"): # SymPy Matrix
                if item.rows == nspins:
                    interaction_matrices.append(item)
            elif len(item) == nspins: # List of lists or similar
                interaction_matrices.append(item)
        except:
            pass
    
    ft_lookup = {}

    for i in range(nspins):
        for j in range(nspins_ouc):
            # Check if ANY interaction matrix has a non-zero term for this pair
            is_nonzero = any(_is_nonzero_at(m, i, j) for m in interaction_matrices)

            if not is_nonzero:
                continue

            disp_vec = atom_positions_uc[i, :] - atom_positions_ouc[j, :]
            k_dot_dr = sum(k * dr for k, dr in zip(k_sym, disp_vec))
            
            # Using Rewrite(sin) as in original code
            exp_plus_ikdr = sp.exp(I * k_dot_dr).rewrite(sp.sin)
            exp_minus_ikdr = sp.exp(-I * k_dot_dr).rewrite(sp.sin)

            j_uc = j % nspins

            # cd_i * cd_j
            val1 = 1/2 * (
                ckd_ops[i] * cmkd_ops[j_uc] * exp_minus_ikdr
                + cmkd_ops[i] * ckd_ops[j_uc] * exp_plus_ikdr
            )
            ft_lookup[(cd_ops[i].name, cd_ops[j].name)] = val1
            
            # c_i * c_j
            val2 = 1/2 * (
                ck_ops[i] * cmk_ops[j_uc] * exp_plus_ikdr
                + cmk_ops[i] * ck_ops[j_uc] * exp_minus_ikdr
            )
            ft_lookup[(c_ops[i].name, c_ops[j].name)] = val2
            
            # cd_i * c_j
            val3 = 1/2 * (
                ckd_ops[i] * ck_ops[j_uc] * exp_minus_ikdr
                + cmkd_ops[i] * cmk_ops[j_uc] * exp_plus_ikdr
            )
            ft_lookup[(cd_ops[i].name, c_ops[j].name)] = val3
            
            # c_i * cd_j
            val4 = 1/2 * (
                ck_ops[i] * ckd_ops[j_uc] * exp_plus_ikdr
                + cmk_ops[i] * cmkd_ops[j_uc] * exp_minus_ikdr
            )
            ft_lookup[(c_ops[i].name, cd_ops[j].name)] = val4
            
    # Add the diagonal term substitution (present in original code, seems important)
    for j in range(nspins_ouc):
        j_uc = j % nspins  # Map OUC index j to UC index
        # cd_j * c_j -> 0.5 * (ckd_j * ck_j + cmkd_j * cmk_j)
        val_diag = 1 / 2 * (ckd_ops[j_uc] * ck_ops[j_uc] + cmkd_ops[j_uc] * cmk_ops[j_uc])
        ft_lookup[(cd_ops[j].name, c_ops[j].name)] = val_diag

    return ft_lookup


def _fourier_transform_terms(
    args: Tuple[sp.Expr, Dict[Tuple[str, str], sp.Expr]],
) -> sp.Expr:
    """
    Apply Fourier Transform substitutions using dictionary lookup.
    
    Parses quadratic terms of form (coeff * Op1 * Op2) and replaces (Op1 * Op2)
    with the corresponding k-space expression found in `ft_lookup`.
    """
    expr, ft_lookup = args
             
    terms = expr.as_ordered_terms()
    new_terms = []
    
    for term in terms:
        coeff, rest = term.as_coeff_Mul()
        
        # Identify operators (non-commutative)
        if rest.is_Mul:
            args_list = rest.args
        else:
            args_list = (rest,)
            
        nc_args = [a for a in args_list if not a.is_commutative]
        c_args = [a for a in args_list if a.is_commutative]
        
        effective_coeff = coeff * sp.Mul(*c_args)
        
        op1 = None
        op2 = None
        
        if len(nc_args) == 2:
            op1 = nc_args[0]
            op2 = nc_args[1]
        elif len(nc_args) == 1 and nc_args[0].is_Pow:
             base, exp = nc_args[0].as_base_exp()
             if exp == 2:
                 op1 = base
                 op2 = base
        
        if op1 and op2:
            key = (op1.name, op2.name)
            if key in ft_lookup:
                new_expr = ft_lookup[key]
                new_terms.append(effective_coeff * new_expr)
            else:
                new_terms.append(term)
        else:
             new_terms.append(term)
             
    return sp.Add(*new_terms)


def _normal_order_terms(args: Tuple[sp.Expr, List[sp.Symbol], List[sp.Symbol], int]) -> sp.Expr:
    """
    Normal order quadratic boson terms in a Hamiltonian expression.
    
    Transforms terms like c_k * c_k_dagger into c_k_dagger * c_k + 1.
    Keeps terms like c_k_dagger * c_k, c_k * c_minus_k, c_k_dagger * c_minus_k_dagger as is.
    Assumes only quadratic terms are present.
    
    Args:
        args: Tuple containing:
            expr_terms (sp.Expr): A sum of terms to process.
            ck_ops: List of ck operators.
            ckd_ops: List of ckd operators.
            nspins: Number of spins.
            
    Returns:
        sp.Expr: The normal ordered expression.
    """
    expr, ck_ops, ckd_ops, nspins = args
    
    # Map operator names to objects for fast lookup
    ck_map = {op.name: (i, 'c') for i, op in enumerate(ck_ops)}
    ckd_map = {op.name: (i, 'cd') for i, op in enumerate(ckd_ops)}
    cmk_map = {f"cmk{i}": (i, 'cm') for i in range(nspins)} # Assuming naming convention
    cmkd_map = {f"cmkd{i}": (i, 'cmd') for i in range(nspins)}
    
    # Combined map
    op_map = {**ck_map, **ckd_map, **cmk_map, **cmkd_map}
    
    terms = expr.as_ordered_terms()
    new_terms = []
    
    for term in terms:
        coeff, ops = term.as_coeff_Mul()
        
        # Identify operators in the term
        # This part assumes simple structure: coeff * op1 * op2 or coeff * op1
        # Complex terms might need rigorous parsing, but usually it's product of 2 non-commuting symbols
        
        non_commuting_factors = term.atoms(sp.Symbol)
        non_commuting_factors = [s for s in non_commuting_factors if not s.is_commutative]
        
        nc_part = term / coeff
        
        if not non_commuting_factors:
            new_terms.append(term)
            continue
            
        if len(non_commuting_factors) != 2:
            new_terms.append(term)
            continue
            
        args_nc = nc_part.args
        if not args_nc: 
             new_terms.append(term)
             continue
             
        # Extract the two operators in order
        op1 = args_nc[0]
        op2 = args_nc[1]
        
        # Handle powers (e.g. ck**2) 
        if len(args_nc) == 1 and args_nc[0].is_Pow:
             base, exp = args_nc[0].as_base_exp()
             if exp == 2:
                 op1 = base
                 op2 = base
             else:
                 new_terms.append(term)
                 continue
        elif len(args_nc) > 2:
             new_terms.append(term)
             continue

        if op1.name not in op_map or op2.name not in op_map:
             new_terms.append(term)
             continue
             
        idx1, type1 = op_map[op1.name]
        idx2, type2 = op_map[op2.name]
        
        # Commutation Rules:
        # [c_i, cd_j] = delta_ij
        if type1 == 'c' and type2 == 'cd':
            # Swap
            new_term = coeff * op2 * op1
            if idx1 == idx2:
                new_term += coeff # +1 * coeff
            new_terms.append(new_term)
            
        elif type1 == 'cm' and type2 == 'cmd':
             # Swap
            new_term = coeff * op2 * op1
            if idx1 == idx2:
                new_term += coeff
            new_terms.append(new_term)
        else:
            new_terms.append(term)
            
    return Add(*new_terms)


def _process_hamiltonian_terms(
    hamiltonian_sym: sp.Expr,
    fourier_lookup: Dict[Tuple[str, str], sp.Expr],
    nspins: int,
    ck_ops: List[sp.Symbol],
    ckd_ops: List[sp.Symbol],
    cmk_ops: List[sp.Symbol],
    cmkd_ops: List[sp.Symbol],
) -> sp.Expr:
    """
    Apply Fourier substitutions and then Normal Order the terms.
    
    Optimized to avoid generic symbolic substitution for commutation rules and F.T.
    """
    logger.info("Applying Fourier transform substitutions...")
    start_time_ft = timeit.default_timer()
    hamiltonian_terms = hamiltonian_sym.as_ordered_terms()

    # Chunking terms for parallel processing
    num_cpus = os.cpu_count() or 1
    chunk_size = max(1, len(hamiltonian_terms) // (num_cpus * 4))
    chunks = [hamiltonian_terms[i:i + chunk_size] for i in range(0, len(hamiltonian_terms), chunk_size)]
    
    pool_args_ft = [
        (sp.Add(*chunk), fourier_lookup) for chunk in chunks
    ]
    
    with Pool() as pool:
        results_ft = list(pool.imap(_fourier_transform_terms, pool_args_ft))
        
    hamiltonian_k_space = Add(*results_ft).expand()
    end_time_ft = timeit.default_timer()
    logger.info(
        f"Fourier transform substitution took: {np.round(end_time_ft - start_time_ft, 2)} s"
    )

    logger.info("Applying Normal Ordering (Commutation Rules)...")
    start_time_comm = timeit.default_timer()
    
    k_terms = hamiltonian_k_space.as_ordered_terms()
    
    # Chunking terms for parallel processing
    chunk_size = max(1, len(k_terms) // (num_cpus * 4))
    chunks = [k_terms[i:i + chunk_size] for i in range(0, len(k_terms), chunk_size)]
    
    pool_args_no = [
        (Add(*chunk), ck_ops, ckd_ops, nspins) for chunk in chunks
    ]
    
    with Pool() as pool:
        results_no = list(pool.imap(_normal_order_terms, pool_args_no))
        
    hamiltonian_normal_ordered = Add(*results_no)
    
    # Expand one last time to ensure coeff * Op1 * Op2 structure
    hamiltonian_normal_ordered = hamiltonian_normal_ordered.expand()
    
    logger.info(
        f"Normal ordering took: {timeit.default_timer() - start_time_comm:.2f} s"
    )

    return hamiltonian_normal_ordered


def _build_TwogH2_matrix(
    hamiltonian_normal_ordered: sp.Expr, 
    nspins: int,
    ck_ops: List[sp.Symbol],
    ckd_ops: List[sp.Symbol],
    cmk_ops: List[sp.Symbol],
    cmkd_ops: List[sp.Symbol],
) -> sp.Matrix:
    """
    Extract the dynamical matrix H2 directly from normal ordered Hamiltonian.
    """
    
    # Map operator string to row/col index
    row_map = {}
    col_map = {}
    
    for i in range(nspins):
        # Rows: Psi.dag
        row_map[ckd_ops[i]] = i          # c_k^dagger -> 0..N-1
        row_map[cmk_ops[i]] = i + nspins # c_-k -> N..2N-1
        
        # Cols: Psi
        col_map[ck_ops[i]] = i           # c_k -> 0..N-1
        col_map[cmkd_ops[i]] = i + nspins# c_-k^dagger -> N..2N-1
    
    n_dim = 2 * nspins
    H2_matrix = sp.zeros(n_dim, n_dim)
    
    terms_norm = hamiltonian_normal_ordered.as_ordered_terms()
    dropped_log = []
    dropped_count = 0
    
    for term in terms_norm:
        coeff, ops = term.as_coeff_Mul()
        
        if ops.is_Mul:
             args_ops = ops.args
        elif ops.is_Pow: 
             args_ops = (ops,)
        elif ops.is_Symbol: 
             args_ops = (ops,)
        else:
             args_ops = (ops,)
             
        nc_ops = [o for o in args_ops if not o.is_commutative]
        
        if len(nc_ops) == 1 and nc_ops[0].is_Pow:
             base, exp = nc_ops[0].as_base_exp()
             if exp == 2:
                 op1 = base
                 op2 = base
             else:
                 op1 = None; op2 = None
        elif len(nc_ops) == 2:
             op1 = nc_ops[0]
             op2 = nc_ops[1]
        else:
             op1 = None; op2 = None
             
        if op1 is None or op2 is None:
            if len(nc_ops) == 0:
                 continue 
            dropped_log.append(f"Structure mismatch: {term}")
            continue
            
        comm_ops = [o for o in args_ops if o.is_commutative]
        comm_factor = sp.Mul(*comm_ops)
        term_val = coeff * comm_factor

        idx1 = row_map.get(op1)
        idx2 = col_map.get(op2)

        if idx1 is not None and idx2 is not None:
             H2_matrix[idx1, idx2] += term_val
        else:
             idx1_rev = row_map.get(op2)
             idx2_rev = col_map.get(op1)
             
             if idx1_rev is not None and idx2_rev is not None:
                 H2_matrix[idx1_rev, idx2_rev] += term_val
             else:
                 dropped_log.append(f"Unparseable ({op1},{op2}): {term}")
                 dropped_count += 1
                 continue

    if dropped_count > 0:
        logger.warning(f"Construct TwogH2: Dropped {dropped_count} terms. First few: {dropped_log[:5]}")
    
    g_list = [1] * nspins + [-1] * nspins
    TwogH2 = sp.zeros(n_dim, n_dim)
    for i in range(n_dim):
        for j in range(n_dim):
            val = 2 * g_list[i] * H2_matrix[i, j]
            TwogH2[i, j] = val
                
    return TwogH2


def _build_ud_matrix(
    rotation_matrices: List[Union[npt.NDArray, sp.Matrix]], nspins: int
) -> sp.Matrix:
    """
    Construct the block-diagonal rotation matrix Ud.
    """
    Ud_rotation_matrix_blocks = []
    for i in range(nspins):
        rot_mat = rotation_matrices[i]
        if isinstance(rot_mat, np.ndarray):
            rot_mat_sym = sp.Matrix(rot_mat)
        else:
            rot_mat_sym = rot_mat
        Ud_rotation_matrix_blocks.append(rot_mat_sym)
    Ud_rotation_matrix = sp.diag(*Ud_rotation_matrix_blocks)
    return Ud_rotation_matrix


def _define_fourier_substitutions_generic(
    k_sym: List[sp.Symbol],
    nspins_uc: int,
    c_ops_ouc: List[sp.Symbol],  # (N_ouc)
    cd_ops_ouc: List[sp.Symbol],  # (N_ouc)
    ck_ops_uc: List[sp.Symbol],  # (N_uc)
    ckd_ops_uc: List[sp.Symbol],  # (N_uc)
    cmk_ops_uc: List[sp.Symbol],  # (N_uc)
    cmkd_ops_uc: List[sp.Symbol],  # (N_uc)
    atom_pos_uc_cart: np.ndarray,  # (N_uc, 3)
    atom_pos_ouc_cart: np.ndarray,  # (N_ouc, 3)
    Jex_sym_matrix: sp.Matrix,  # (N_uc, N_ouc)
    DM_sym_matrix: sp.Matrix = None, # (N_uc, N_ouc) of 3x1
    Kex_sym_matrix: sp.Matrix = None, # (N_uc, N_ouc) of 3x1
) -> List[List[sp.Expr]]:
    """
    Define Fourier substitutions for the configuration-driven model.
    """
    nspins_ouc = len(atom_pos_ouc_cart)
    # Basic checks skipped for brevity/duplication
    
    fourier_substitutions = []

    for i_uc in range(nspins_uc):
        for j_ouc in range(nspins_ouc):
            # Check if ANY of the interaction matrices have non-zero for this pair
            is_nonzero = (
                _is_nonzero_at(Jex_sym_matrix, i_uc, j_ouc)
                or _is_nonzero_at(DM_sym_matrix, i_uc, j_ouc)
                or _is_nonzero_at(Kex_sym_matrix, i_uc, j_ouc)
            )

            if not is_nonzero:
                continue

            disp_vec = atom_pos_uc_cart[i_uc, :] - atom_pos_ouc_cart[j_ouc, :]
            k_dot_dr = sum(k_comp * dr_comp for k_comp, dr_comp in zip(k_sym, disp_vec))

            exp_plus_ikdr = sp.exp(I * k_dot_dr).rewrite(sp.sin)
            exp_minus_ikdr = sp.exp(-I * k_dot_dr).rewrite(sp.sin)

            idx_op1_real_space = i_uc
            idx_op2_real_space = j_ouc

            idx_op1_k_space = i_uc
            idx_op2_k_space = j_ouc % nspins_uc

            # ...
            # Implementation moved from core.py
            
            idx_op1_real_space = i_uc
            idx_op2_real_space = j_ouc

            idx_op1_k_space = i_uc
            idx_op2_k_space = j_ouc % nspins_uc

            sub_list_for_pair = [
                [
                    cd_ops_ouc[idx_op1_real_space] * cd_ops_ouc[idx_op2_real_space],
                    sp.S(1)
                    / 2
                    * (
                        ckd_ops_uc[idx_op1_k_space]
                        * cmkd_ops_uc[idx_op2_k_space]
                        * exp_minus_ikdr
                        + cmkd_ops_uc[idx_op1_k_space]
                        * ckd_ops_uc[idx_op2_k_space]
                        * exp_plus_ikdr
                    ),
                ],
                [
                    c_ops_ouc[idx_op1_real_space] * c_ops_ouc[idx_op2_real_space],
                    sp.S(1)
                    / 2
                    * (
                        ck_ops_uc[idx_op1_k_space]
                        * cmk_ops_uc[idx_op2_k_space]
                        * exp_plus_ikdr
                        + cmk_ops_uc[idx_op1_k_space]
                        * ck_ops_uc[idx_op2_k_space]
                        * exp_minus_ikdr
                    ),
                ],
                [
                    cd_ops_ouc[idx_op1_real_space] * c_ops_ouc[idx_op2_real_space],
                    sp.S(1)
                    / 2
                    * (
                        ckd_ops_uc[idx_op1_k_space]
                        * ck_ops_uc[idx_op2_k_space]
                        * exp_minus_ikdr
                        + cmkd_ops_uc[idx_op1_k_space]
                        * cmk_ops_uc[idx_op2_k_space]
                        * exp_plus_ikdr
                    ),
                ],
                [
                    c_ops_ouc[idx_op1_real_space] * cd_ops_ouc[idx_op2_real_space],
                    sp.S(1)
                    / 2
                    * (
                        ck_ops_uc[idx_op1_k_space]
                        * ckd_ops_uc[idx_op2_k_space]
                        * exp_plus_ikdr
                        + cmk_ops_uc[idx_op1_k_space]
                        * cmkd_ops_uc[idx_op2_k_space]
                        * exp_minus_ikdr
                    ),
                ],
            ]
            fourier_substitutions.extend(sub_list_for_pair)

    for j_ouc_diag in range(nspins_ouc):
        j_uc_diag = j_ouc_diag % nspins_uc
        fourier_substitutions.append(
            [
                cd_ops_ouc[j_ouc_diag] * c_ops_ouc[j_ouc_diag],
                sp.S(1)
                / 2
                * (
                    ckd_ops_uc[j_uc_diag] * ck_ops_uc[j_uc_diag]
                    + cmkd_ops_uc[j_uc_diag] * cmk_ops_uc[j_uc_diag]
                ),
            ]
        )

    unique_substitutions = []
    seen_keys = set()
    for sub_pair in fourier_substitutions:
        key = sub_pair[0]
        if key not in seen_keys:
            unique_substitutions.append(sub_pair)
            seen_keys.add(key)

    return unique_substitutions

def gen_HM(
    spin_model_module,  
    k_sym: List[sp.Symbol],
    S_sym: sp.Symbol,
    params_sym: List[sp.Symbol],
) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Generate the symbolic dynamical matrix (2gH) and rotation matrix (Ud).
    """
    logger.info("Starting symbolic matrix generation (gen_HM)...")
    start_time_total = timeit.default_timer()

    # --- Get Model Info ---
    try:
        atom_positions_uc = spin_model_module.atom_pos()
        nspins = len(atom_positions_uc)
        atom_positions_ouc = spin_model_module.atom_pos_ouc()
        nspins_ouc = len(atom_positions_ouc)
        rotation_matrices = spin_model_module.mpr(params_sym)
    except Exception as e:
        logger.exception("Error retrieving data from spin_model_module.")
        raise RuntimeError("Failed to get model info for gen_HM.") from e

    logger.info(f"Number of spins in the unit cell: {nspins}")

    # --- Setup Operators ---
    c_ops, cd_ops, spin_ops_local = _setup_hp_operators(nspins_ouc, S_sym)
    spin_ops_global_ouc = _rotate_spin_operators(
        spin_ops_local, rotation_matrices, nspins, nspins_ouc
    )

    logger.info("Constructing symbolic Hamiltonian from spin model...")
    start_time_ham = timeit.default_timer()
    hamiltonian_sym = _prepare_hamiltonian(
        spin_model_module, spin_ops_global_ouc, params_sym, S_sym
    )
    end_time_ham = timeit.default_timer()
    logger.info(f"Symbolic Hamiltonian construction took: {end_time_ham - start_time_ham:.2f} s")

    # --- Define k-space operators ---
    ck_ops = [sp.Symbol("ck%d" % j, commutative=False) for j in range(nspins)]
    ckd_ops = [sp.Symbol("ckd%d" % j, commutative=False) for j in range(nspins)]
    cmk_ops = [sp.Symbol("cmk%d" % j, commutative=False) for j in range(nspins)]
    cmkd_ops = [sp.Symbol("cmkd%d" % j, commutative=False) for j in range(nspins)]

    # --- Define Substitution Rules ---
    fourier_lookup = _generate_fourier_lookup(
        spin_model_module,
        k_sym,
        nspins,
        nspins_ouc,
        c_ops,
        cd_ops,
        ck_ops,
        ckd_ops,
        cmk_ops,
        cmkd_ops,
        params_sym,
    )

    # --- Apply Substitutions & Normal Ordering ---
    try:
        hamiltonian_normal_ordered = _process_hamiltonian_terms(
            hamiltonian_sym,
            fourier_lookup,
            nspins,
            ck_ops,
            ckd_ops,
            cmk_ops,
            cmkd_ops,
        )
    except Exception as e:
        logger.exception("Error during symbolic substitution in gen_HM.")
        raise RuntimeError("Symbolic substitution failed.") from e

    # --- Build TwogH2 Matrix directly ---
    dynamical_matrix_TwogH2 = _build_TwogH2_matrix(
        hamiltonian_normal_ordered, 
        nspins, 
        ck_ops, 
        ckd_ops, 
        cmk_ops, 
        cmkd_ops
    )

    # --- Build Ud Matrix ---
    Ud_rotation_matrix = _build_ud_matrix(rotation_matrices, nspins)

    end_time_total = timeit.default_timer()
    logger.info(
        f"Total run-time for gen_HM: {np.round((end_time_total - start_time_total), 2)} s."
    )

    return dynamical_matrix_TwogH2, Ud_rotation_matrix
