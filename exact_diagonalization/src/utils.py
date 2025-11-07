
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


def circular_left_shift(n: int, k: int, width: int) -> int:
    """
    Rotate bits of n to the left by k positions within given width.
    Args:
        n (int): The integer whose bits are to be rotated.
        k (int): Number of positions to rotate.
        width (int): The bit-width to consider for rotation.
    Returns:
        int: The integer resulting from the left rotation.
    Raises:
        ValueError: If width is smaller than the bit-length of n.
    Example:
        rotate_left(0b10100, 2, 5) -> 0b10010
    """
    # Ensure width is not smaller than input bit-length
    if n.bit_length() > width:
        raise ValueError("Width must be at least as large as the bit-length of n.")

    # Ensure k is within the width
    k %= width

    # Glue the rotated parts together, e.g. 10100 << 2 = 10100 00, 10100 >> 3 = 000 10
    # such that 1010000 | 0000010 = 1010010
    n_rotated = ((n << k) | (n >> (width - k)))

    # Mask to keep only the relevant width bits, e.g. 1010010 & 0011111 = 10010
    n_rotated = n_rotated & ((1 << width) - 1)
    return n_rotated

def circular_right_shift(n: int, k: int, width: int) -> int:
    """
    Rotate bits of n to the right by k positions within given width.
    Args:
        n (int): The integer whose bits are to be rotated.
        k (int): Number of positions to rotate.
        width (int): The bit-width to consider for rotation.
    Returns:
        int: The integer resulting from the right rotation.
    Raises:
        ValueError: If width is smaller than the bit-length of n.
    Example:
        rotate_right(0b10100, 2, 5) -> 0b00101
    """
    # Ensure width is not smaller than input bit-length
    if n.bit_length() > width:
        raise ValueError("Width must be at least as large as the bit-length of n.")
    
    # Ensure k is within the width
    k %= width

    # Glue the rotated parts together, e.g. 10100 >> 2 = 00101, 10100 << 3 = 10100000
    # such that 00000101 | 10100000 = 10100101
    n_rotated = ((n >> k) | (n << (width - k)))

    # Mask to keep only the relevant width bits, e.g. 10100101 & 0011111 = 00101
    n_rotated = n_rotated & ((1 << width) - 1)
    return n_rotated


# def flip_bit(n: int, i: int) -> int:
#     """Flip bit i in integer n."""
#     return n ^ (1 << i)

# def bit_occupancy(n: int, i: int) -> int:
#     """Return 1 if site i is occupied."""
#     return (n >> i) & 1