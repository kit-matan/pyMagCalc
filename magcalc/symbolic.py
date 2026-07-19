import sympy as sp
from sympy import I, Add
import numpy as np
import timeit
import logging
import os  # for cpu_count
from multiprocessing import Pool
from typing import List, Tuple, Dict, Any, Union

# Helper for type hinting
import numpy.typing as npt

logger = logging.getLogger(__name__)

def _setup_hp_operators(
    nspins_ouc: int, S_sym: sp.Symbol, nspins: int = None,
    spin_ratios: List[float] = None,
) -> Tuple[List[sp.Symbol], List[sp.Symbol], List[sp.Matrix]]:
    """
    Create symbolic Holstein-Primakoff boson operators and local spin operators.

    Args:
        nspins_ouc (int): Number of spins in the original unit cell + neighbors (OUC).
        S_sym (sp.Symbol): Symbolic representation of the (reference) spin magnitude S.
        nspins (int, optional): Number of spins in the magnetic unit cell. OUC sites
            ``j`` are cyclic copies of unit-cell site ``j % nspins``.
        spin_ratios (List[float], optional): Per-unit-cell-site ratios ``S_i / S_ref``
            (length ``nspins``). Site ``j`` in the OUC then gets magnitude
            ``spin_ratios[j % nspins] * S_sym``. When omitted (or all 1.0) this
            reduces exactly to a single global spin magnitude, so equal-spin models
            are unchanged. The ratios are numeric constants multiplying the single
            symbol ``S_sym``, so the S-power filtering and numerical binding of
            ``S_sym`` are untouched.
    Returns:
        Tuple[List[sp.Symbol], List[sp.Symbol], List[sp.Matrix]]: Boson annihilation operators (c), creation operators (cd), and local spin operators (Sx, Sy, Sz) expressed in terms of bosons.
    """
    c_ops = sp.symbols("c0:%d" % nspins_ouc, commutative=False)
    cd_ops = sp.symbols("cd0:%d" % nspins_ouc, commutative=False)

    def _mag(i):
        # Per-site magnitude S_i = ratio_i * S_sym (ratio numeric, S_sym symbolic).
        if spin_ratios and nspins:
            r = spin_ratios[i % nspins]
            if r != 1.0:
                return sp.nsimplify(r) * S_sym
        return S_sym

    spin_ops_local = [
        sp.Matrix(
            (
                sp.sqrt(_mag(i) / 2) * (c_ops[i] + cd_ops[i]),
                sp.sqrt(_mag(i) / 2) * (c_ops[i] - cd_ops[i]) / I,
                _mag(i) - cd_ops[i] * c_ops[i],
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


def boson_degree(term: sp.Expr) -> int:
    """Total power of the Holstein-Primakoff boson operators (c*/cd*) in one term.

    Bosons are the NON-COMMUTATIVE symbols -- identifying them by a "c" name
    prefix (the old rule) silently miscounted any model parameter whose name
    starts with "c" (chi, c1, ...), which dropped Hamiltonian terms.
    """
    powers = term.as_powers_dict()
    return sum(
        powers.get(s, 0) for s in term.atoms(sp.Symbol) if not s.is_commutative
    )


def _is_purely_quadratic_in_bosons(expr: sp.Expr) -> bool:
    """True when every additive term of `expr` is exactly quadratic in bosons.

    Used to detect that the spin model already performed the LSWT truncation, so
    the legacy S-power filter can (and must) be skipped -- see _prepare_hamiltonian.
    """
    if expr == 0:
        return False
    terms = expr.as_ordered_terms() if hasattr(expr, "as_ordered_terms") else [expr]
    saw_boson = False
    for t in terms:
        deg = boson_degree(t)
        if deg != 2:
            return False
        saw_boson = True
    return saw_boson


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
    elif _is_purely_quadratic_in_bosons(hamiltonian_sym):
        # The model already truncated to the LSWT (boson-degree-2) terms itself --
        # GenericSpinModel._apply_substitution_and_filter does exactly this. The
        # S-power heuristic below is then not just redundant but WRONG for any
        # term of higher order in the spin operators: the quadratic-boson part of
        # a quartic term (biquadratic exchange, Stevens O_4/O_6) carries S^3, and
        # `coeff(S,1)*S + coeff(S,2)*S^2` would silently delete it. Boson degree is
        # the physically correct LSWT criterion, so when the model has already
        # applied it there is nothing left to filter.
        logger.debug(
            "Hamiltonian is already purely quadratic in boson operators; "
            "skipping the S-power filter (it would drop S^3 terms from quartic "
            "spin operators)."
        )
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
            
            term_params_S0 = 0
            for p in params_sym:
                if isinstance(p, sp.Symbol):
                    term_params_S0 += hamiltonian_S0.coeff(p) * p
                elif isinstance(p, (list, tuple, np.ndarray, sp.Matrix)):
                    # Handle flattened lists or symbols in matrices
                    for sub_p in p:
                        if isinstance(sub_p, sp.Symbol):
                            term_params_S0 += hamiltonian_S0.coeff(sub_p) * sub_p
        
            hamiltonian_sym = (
                term_params_S0
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
            
    # On-site (i == j, dr = 0) substitutions. These cover terms that live on a
    # single site and are therefore NOT generated by the pair loop above (which
    # only fires when an inter-site interaction matrix is non-zero). The most
    # important case is single-ion anisotropy K (S.n)^2, which contributes
    # on-site number terms (cd_j c_j, c_j cd_j) AND on-site ANOMALOUS terms
    # (cd_j^2, c_j^2). Without the anomalous substitutions the c^2/cd^2 pieces
    # are dropped when the dynamical matrix is assembled, leaving the diagonal
    # and anomalous parts unbalanced -- which spuriously gaps Goldstone modes
    # (e.g. the in-plane mode of an easy-plane magnet). With dr = 0 the phase
    # factors are 1, so these are the pair-loop forms specialized to i = j.
    for j in range(nspins_ouc):
        j_uc = j % nspins  # Map OUC index j to UC index
        # cd_j * c_j -> 0.5 * (ckd_j ck_j + cmkd_j cmk_j)
        ft_lookup[(cd_ops[j].name, c_ops[j].name)] = (
            1 / 2 * (ckd_ops[j_uc] * ck_ops[j_uc] + cmkd_ops[j_uc] * cmk_ops[j_uc])
        )
        # c_j * cd_j -> 0.5 * (ck_j ckd_j + cmk_j cmkd_j)
        ft_lookup[(c_ops[j].name, cd_ops[j].name)] = (
            1 / 2 * (ck_ops[j_uc] * ckd_ops[j_uc] + cmk_ops[j_uc] * cmkd_ops[j_uc])
        )
        # cd_j^2 -> 0.5 * (ckd_j cmkd_j + cmkd_j ckd_j)   (anomalous)
        ft_lookup[(cd_ops[j].name, cd_ops[j].name)] = (
            1 / 2 * (ckd_ops[j_uc] * cmkd_ops[j_uc] + cmkd_ops[j_uc] * ckd_ops[j_uc])
        )
        # c_j^2 -> 0.5 * (ck_j cmk_j + cmk_j ck_j)         (anomalous)
        ft_lookup[(c_ops[j].name, c_ops[j].name)] = (
            1 / 2 * (ck_ops[j_uc] * cmk_ops[j_uc] + cmk_ops[j_uc] * ck_ops[j_uc])
        )

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


# Fanning these symbolic maps out to a multiprocessing Pool pickles SymPy
# expressions to/from workers and pays a fixed spawn+teardown cost of ~2-3 s
# (macOS 'spawn' re-imports the world in every worker). Profiling gen_HM showed
# that for typical models (hundreds–low-thousands of terms) that overhead
# *dwarfs* the actual symbolic work — the Pool teardown and result unpickling
# alone were ~4 s — making the parallel path 3-5x SLOWER than doing it
# in-process. Only fan out once there are enough terms that the per-chunk
# compute can amortize the spawn cost. Measured: even a ~4700-term model
# (9-spin non-collinear cell) is FASTER serial (20.6 s) than parallel (25.6 s),
# so the crossover is high — set the gate well above realistic model sizes and
# treat the Pool as a safety valve for genuinely huge cells only.
_PARALLEL_TERM_THRESHOLD = 8000


def _map_terms(worker, pool_args, n_terms):
    """Map ``worker`` over ``pool_args``, using a Pool only when the workload
    (``n_terms``) is large enough to outweigh process spawn/pickle overhead;
    otherwise run serially in-process (far faster for small/medium models)."""
    if (
        n_terms < _PARALLEL_TERM_THRESHOLD
        or (os.cpu_count() or 1) < 2
        or len(pool_args) < 2
    ):
        return [worker(a) for a in pool_args]
    with Pool() as pool:
        return list(pool.imap(worker, pool_args))


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

    results_ft = _map_terms(
        _fourier_transform_terms, pool_args_ft, len(hamiltonian_terms)
    )

    hamiltonian_k_space = Add(*results_ft).expand()
    end_time_ft = timeit.default_timer()
    logger.info(
        f"Fourier transform substitution took: {np.round(end_time_ft - start_time_ft, 2)} s"
    )

    # NOTE: no explicit normal-ordering pass. _build_TwogH2_matrix places a
    # quadratic term coeff*op1*op2 into the same H2 element whichever order the
    # two operators appear in (it tries the reversed row/col mapping), and the
    # commutator c-number constants normal ordering would generate are dropped
    # by the matrix construction anyway (LSWT discards the constant). The old
    # pass therefore had no effect on the output matrix; it only re-walked
    # every term.
    return hamiltonian_k_space


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

    # --- Per-site spin magnitudes (mixed-spin support) ---
    # Site i uses magnitude S_i = ratio_i * S_sym, with ratio_i = S_i / S_ref.
    # S_ref is the value S_sym binds to numerically (the first atom's spin_S,
    # matching MagCalc.spin_magnitude). Models without spin_magnitudes(), or
    # with all spins equal, yield ratios == 1 -> identical to the single-S path.
    spin_ratios = None
    try:
        mags = getattr(spin_model_module, 'spin_magnitudes', lambda: [])()
        if mags and len(mags) == nspins:
            S_ref = float(mags[0])
            if S_ref > 0 and any(abs(float(m) - S_ref) > 1e-12 for m in mags):
                spin_ratios = [float(m) / S_ref for m in mags]
                logger.info(f"Mixed-spin model detected; per-site S ratios = {spin_ratios}")
    except Exception:
        logger.debug("spin_magnitudes() unavailable; assuming a single global spin S.")

    # --- Setup Operators ---
    c_ops, cd_ops, spin_ops_local = _setup_hp_operators(
        nspins_ouc, S_sym, nspins=nspins, spin_ratios=spin_ratios
    )
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
