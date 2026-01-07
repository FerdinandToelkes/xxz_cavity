import numpy as np

from scipy.sparse import csr_matrix, diags, identity, kron

from src.basis import Basis
from src.utils import circular_right_shift

def count_pairs(n: int, d: int, width: int, boundary_conditions: str = "periodic") -> int:
    """
    Count neighboring 1-1 pairs that are d places apart in bit representation. Note, that
    periodic boundary conditions are considered.
    Arguments:
        n (int): The integer whose bits are to be analyzed.
        d (int): The distance between neighboring bits to consider.
        width (int): The bit-width to consider.
        boundary_condition (str): The type of boundary condition to consider.
    Returns:
        int: The count of neighboring 1-1 pairs d places apart.
    Example:
        count_pairs(0b111011, 1) -> 4
        count_pairs(0b111011, 2) -> 4 
    """
    if boundary_conditions == "periodic":
        # perform circular shift by d and count 1-1 pairs
        n_shifted = circular_right_shift(n, d, width)
    elif boundary_conditions == "open":
        # perform regular shift by d and count 1-1 pairs
        n_shifted = n >> d
    else:
        raise ValueError("Invalid boundary condition. Use 'periodic' or 'open'.")
    return (n & n_shifted).bit_count()

def flip_bit(n: int, i: int) -> int:
    """Flip bit i in integer n. Note: i starts from 0 (least significant bit)."""
    return n ^ (1 << i)

def check_if_bit_set(n: int, i: int) -> bool:
    """Check if bit i in integer n is set (1). Note: i starts from 0 (least significant bit).
       Returns True if bit is set, False otherwise.
    """
    return (n & (1 << i)) != 0

def count_bits_between(x: int, i: int, j: int, width: int, inclusive: bool = True) -> int:
    """
    Count the number of 1-bits in x between bit positions i and j (inclusive or exclusive).
    Bits are numbered from 0 (least significant bit).
    Arguments:
        x (int): The integer whose bits are to be analyzed.
        i (int): The starting bit position.
        j (int): The ending bit position.
        inclusive (bool): Whether to include bit at positions j in the count.
    Returns:
        int: The count of 1-bits between positions i and j.
    Raises:
        ValueError: If i > j, if i or j are negative or if j >= width.
    Example:
        count_bits_between(0b1011110, 1, 4, 7, inclusive=True) -> 4
        count_bits_between(0b1011110, 1, 4, 7, inclusive=False) -> 3
    """
    if i > j:
        raise ValueError(f"i ({i}) must be less than or equal to j ({j}).")
    if i < 0 or j < 0:
        raise ValueError(f"i ({i}) and j ({j}) must be non-negative integers.")
    if j > width:
        raise ValueError(f"j ({j}) must be less than the bit-width ({width}).")
    
    # Create mask covering bits between i and j
    if inclusive:
        mask = ((1 << (j - i + 1)) - 1) << i
    else:
        mask = ((1 << (j - i)) - 1) << i

    return (x & mask).bit_count()

def build_photon_number_matrix(basis: Basis) -> csr_matrix:
    """
    Construct the photon number operator matrix for a given maximum photon number N_ph.
    Note, that dim(photon basis) = N_ph + 1.
    Arguments:
        N_ph (int): The maximum photon number.
    Returns:
        csr_matrix: The photon number operator matrix of shape (N_f*(N_ph+1),N_f*(N_ph+1)).
    """
    dim_el = len(basis.fermion_states)
    dim_ph = len(basis.photon_states)
    # Build a diagonal array of photon energies since N|i> = i|i>
    diag = np.arange(0, dim_ph)
    op_ph = diags(diag, format='csr')
    op_el = identity(dim_el, format='csr')
    op = kron(op_el, op_ph, format='csr')
    # convert to csr for efficient diagonalization
    return csr_matrix(op)

def build_fermion_number_matrix(basis: Basis, site: int) -> csr_matrix:
    """
    Construct the fermion number operator matrix n_i for a given basis.
    Arguments:
        basis (Basis): The basis object containing fermion states.
        site (int): The site index i where the number operator is applied (0 <= i < L).
    Returns:
        csr_matrix: The fermion number operator matrix of shape (dim(basis), dim(basis)).
    Raises:
        ValueError: If site index is out of bounds.
    """
    if site < 0 or site >= basis.L:
        raise ValueError(f"Site index {site} is out of bounds for basis with L={basis.L}.")
    
    dim_el = len(basis.fermion_states)
    dim_ph = len(basis.photon_states)
    diag = np.zeros(dim_el)
    for idx, state in enumerate(basis.fermion_states):
        if check_if_bit_set(state, site):
            diag[idx] = 1
    op_el = diags(diag, format='csr')
    op_ph = identity(dim_ph, format='csr')
    op = kron(op_el, op_ph, format='csr')
    return csr_matrix(op)

def build_longest_range_fermion_number_matrix(basis: Basis, boundary_conditions: str) -> csr_matrix:
    """
    Construct the "longest-range" fermion number operator matrix n_max = n_0 * n_{l} for a given basis,
    where l = L//2 for periodic boundary conditions and l = L-1 for open boundary conditions.
    Arguments:
        basis (Basis): The basis object containing fermion states.
        boundary_conditions (str): The type of boundary condition to consider ("periodic" or "open").
    Returns:
        csr_matrix: The longest-range fermion number operator matrix of shape (dim(basis), dim(basis)).
    Raises:
        ValueError: If boundary condition is invalid.
    """
    if boundary_conditions == "periodic":
        l = basis.L // 2 # since we start counting from 0
    elif boundary_conditions == "open":
        l = basis.L - 1
    else:
        raise ValueError("Invalid boundary condition. Use 'periodic' or 'open'.")
    
    dim_el = len(basis.fermion_states)
    dim_ph = len(basis.photon_states)
    diag = np.zeros(dim_el)
    for idx, state in enumerate(basis.fermion_states):
        n_0 = 1 if check_if_bit_set(state, 0) else 0
        n_l = 1 if check_if_bit_set(state, l) else 0
        diag[idx] = n_0 * n_l
    op_el = diags(diag, format='csr')
    op_ph = identity(dim_ph, format='csr')
    op = kron(op_el, op_ph, format='csr')
    return csr_matrix(op)


# not used up to now, but here for completeness

# def photon_creator(state: tuple[float, int], max_photons: int) -> tuple[float, int]:
#     """
#     Apply photon creation operator on state n.
#     Arguments:
#         state (tuple[float, int]): The tuple representing the current state (prefactor, photon number).
#         max_photons (int): The maximum allowed photon number.
#     Returns:
#         tuple[float, int]: The coefficient and the new photon number state after applying the creation operator, or (0, 0) if max_photons is reached.
#     """
#     prefactor, n = state
#     if n >= max_photons:
#         return 0, 0
#     else:
#         return prefactor * np.sqrt(n + 1), n + 1
    
# def photon_annihilator(state: tuple[float, int]) -> tuple[float, int]:
#     """
#     Apply photon annihilation operator on state n.
#     Arguments:
#         state (tuple[float, int]): The tuple representing the current state (prefactor, photon number).
#     Returns:
#         tuple[float, int]: The coefficient and the new photon number state after applying the annihilation operator, or (0, 0) if n is zero.
#     """
#     prefactor, n = state
#     if n == 0:
#         return 0, 0
#     else:
#         return prefactor * np.sqrt(n), n - 1
#
# def fermion_creator(state: tuple[float, int], i: int, width: int) -> tuple[float, int]:
#     """
#     Apply creation operator at site i on state n.
#     Arguments:
#         state (tuple[float, int]): The tuple representing the current state (prefactor, basis state).
#         i (int): The site index where the creation operator is applied.
#         width (int): The bit-width to consider.
#     Returns:
#         tuple[float, int]: The prefactor and the new basis state after applying the creation operator, or (0, 0) if the site is already occupied.
#     """
#     prefactor, n = state
#     if n & (1 << i) != 0:
#         # site already occupied
#         return 0, 0
#     else:
#         # prefactor = current_prefactor * (-1) ** sum_{i>j} b_i for |s> = |b_{width-1} ... b_0>
#         prefactor = prefactor * (-1) ** count_bits_between(n, i+1, width, width, inclusive=False)
#         return prefactor, flip_bit(n, i)
    
# def fermion_annihilator(state: tuple[float, int], i: int, width: int) -> tuple[float, int]:
#     """
#     Apply annihilation operator at site i on state n.
#     Arguments:
#         state (tuple[float, int]): The tuple representing the current state (prefactor, basis state).
#         i (int): The site index where the annihilation operator is applied.
#         width (int): The bit-width to consider.
#     Returns:
#         tuple[float, int]: The prefactor and the new basis state after applying the annihilation operator, or (0, 0) if the site is unoccupied.
#     """
#     prefactor, n = state
#     if n & (1 << i) == 0:
#         # site unoccupied
#         return 0, 0
#     else:
#         # prefactor = current_prefactor * (-1) ** sum_{i>j} b_i for |s> = |b_{width-1} ... b_0>
#         prefactor = prefactor * (-1) ** count_bits_between(n, i+1, width, width, inclusive=False)
#         return prefactor, flip_bit(n, i)
    
# def fermion_number_operator(state: tuple[float, int], i: int) -> tuple[float, int]:
#     """
#     Apply number operator at site i on state n. Note, this operator is diagonal in the occupation basis.
#     Arguments:
#         state (tuple[float, int]): The tuple representing the current state (prefactor, basis state).
#         i (int): The site index where the number operator is applied.
#     Returns:
#         tuple[float, int]: The occupation number (0 or 1) and the new basis state after applying the number operator.
#     """
#     prefactor, n = state
#     n_i = (n >> i) & 1
#     return (prefactor * n_i, n_i*n)

# def total_fermion_number_operator(state: tuple[float, int], width: int) -> tuple[float, int]:
#     """
#     Apply total number operator on state n, with N = sum_i n_i. Note, this operator is diagonal in the occupation basis.
#     Arguments:
#         state (tuple[float, int]): The tuple representing the current state (prefactor, basis state).
#         width (int): The bit-width to consider.
#     Returns:
#         tuple[float, int]: The total number of occupied sites and the new basis state after applying the total number operator.
#     """
#     prefactor, n = state
#     # ensure that we only count bits within the specified width
#     mask = (1 << width) - 1
#     N = (n & mask).bit_count()
#     final_state = 0 if N == 0 else n
#     return (prefactor * N, final_state)

