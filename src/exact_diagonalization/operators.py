from src.exact_diagonalization.utils import circular_right_shift

def count_pairs(n: int, d: int, width: int, boundary_conditions: str = "periodic") -> int:
    """
    Count neighboring 1-1 pairs that are d places apart in bit representation. Note, that
    periodic boundary conditions are considered.
    Args:
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

def count_bits_between(x: int, i: int, j: int, inclusive: bool = True) -> int:
    """
    Count the number of 1-bits in x between bit positions i and j (inclusive or exclusive).
    Bits are numbered from 0 (least significant bit).
    Args:
        x (int): The integer whose bits are to be analyzed.
        i (int): The starting bit position.
        j (int): The ending bit position.
        inclusive (bool): Whether to include bit at positions j in the count.
    Returns:
        int: The count of 1-bits between positions i and j.
    Example:
        count_bits_between(0b1011110, 1, 4, inclusive=True) -> 4
        count_bits_between(0b1011110, 1, 4, inclusive=False) -> 3
    """
    if i > j:
        raise ValueError(f"i ({i}) must be less than or equal to j ({j}).")

    # Create mask covering bits between i and j
    if inclusive:
        mask = ((1 << (j - i + 1)) - 1) << i
    else:
        mask = ((1 << (j - i)) - 1) << i

    return (x & mask).bit_count()


def fermion_creator(n: int, i: int, width: int) -> int:
    """
    Apply creation operator at site i on state n.
    Args:
        n (int): The integer representing the current state.
        i (int): The site index where the creation operator is applied.
        width (int): The bit-width to consider.
    Returns:
        int: The new state after applying the creation operator, or 0 if the site is already occupied.
    """
    if n & (1 << i) != 0:
        # site already occupied
        return 0
    else:
        # sign = sum_{i>j} b_i for |s> = |b_{width-1} ... b_0>
        sign = (-1) ** count_bits_between(n, i+1, width, inclusive=False)
        return sign * flip_bit(n, i)
    
def fermion_annihilator(n: int, i: int, width: int) -> int:
    """
    Apply annihilation operator at site i on state n.
    Args:
        n (int): The integer representing the current state.
        i (int): The site index where the annihilation operator is applied.
        width (int): The bit-width to consider.
    Returns:
        int: The new state after applying the annihilation operator, or 0 if the site is unoccupied.
    """
    if n & (1 << i) == 0:
        # site unoccupied
        return 0
    else:
        # sign = sum_{i>j} b_i for |s> = |b_{width-1} ... b_0>
        sign = (-1) ** count_bits_between(n, i+1, width, inclusive=False)
        print(f"sign: {sign}, n: {bin(n)}, i: {i}, width: {width}")
        return sign * flip_bit(n, i)