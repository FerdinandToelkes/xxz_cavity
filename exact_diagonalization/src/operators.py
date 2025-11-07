from src.utils import circular_right_shift

def count_pairs(n: int, d: int, width: int, boundary_condition: str = "periodic") -> int:
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
    if boundary_condition == "periodic":
        # perform circular shift by d and count 1-1 pairs
        n_shifted = circular_right_shift(n, d, width)
    elif boundary_condition == "open":
        # perform regular shift by d and count 1-1 pairs
        n_shifted = n >> d
    else:
        raise ValueError("Invalid boundary condition. Use 'periodic' or 'open'.")
    return (n & n_shifted).bit_count()

def flip_bit(n: int, i: int) -> int:
    """Flip bit i in integer n. Note: i starts from 0 (least significant bit)."""
    return n ^ (1 << i)


# def bit_occupancy(n: int, i: int) -> int:
#     """Return 1 if site i is occupied."""
#     return (n >> i) & 1