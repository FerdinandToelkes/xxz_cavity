import pytest
# from numpy.testing import assert_allclose

from src.exact_diagonalization.operators import count_pairs, flip_bit, count_bits_between, build_photon_number_matrix
#, fermion_creator, fermion_annihilator, fermion_number_operator, total_fermion_number_operator 
from src.exact_diagonalization.basis import Basis    

@pytest.mark.parametrize("n, d, width, boundary_conditions, expected_count", [
    (0b0, 1, 1, "periodic", 0),
    (0b1, 1, 1, "periodic", 1),
    (0b11, 1, 2, "periodic", 2),
    (0b111, 1, 3, "periodic", 3),
    (0b1011, 1, 4, "periodic", 2),
    (0b111011, 1, 6, "periodic", 4),
    (0b1, 1, 1, "open", 0),
    (0b11, 1, 2, "open", 1),
    (0b111, 1, 3, "open", 2),
    (0b1011, 1, 4, "open", 1),
    (0b111011, 1, 6, "open", 3),
])
def test_count_pairs(n: int, d: int, width: int, boundary_conditions: str, expected_count: int):
    assert count_pairs(n, d, width, boundary_conditions) == expected_count

def test_count_pairs_invalid_boundary_conditions():
    with pytest.raises(ValueError):
        count_pairs(0b111011, 1, 6, "invalid")

@pytest.mark.parametrize("n, i, expected", [
    (0b0000, 0, 0b0001),
    (0b0001, 0, 0b0000),
    (0b0010, 1, 0b0000),
    (0b0000, 1, 0b0010),
    (0b1010, 2, 0b1110),
    (0b1110, 2, 0b1010),
])
def test_flip_bit(n: int, i: int, expected: int):
    assert flip_bit(n, i) == expected

@pytest.mark.parametrize("x, i, j, width, inclusive, expected_count", [
    (0b0, 0, 0, 1, True, 0),
    (0b1, 0, 0, 1, True, 1),
    (0b1101, 1, 3, 4, True, 2),
    (0b1101, 1, 3, 4, False, 1),
    (0b111111, 0, 5, 6, True, 6),
    (0b111111, 0, 5, 6, False, 5),

])
def test_count_bits_between(x: int, i: int, j: int, width: int, inclusive: bool, expected_count: int):
    assert count_bits_between(x, i, j, width, inclusive=inclusive) == expected_count

def test_count_bits_between_invalid_indices():
    with pytest.raises(ValueError):
        count_bits_between(0b1101, 3, 1, 4)
    with pytest.raises(ValueError):
        count_bits_between(0b1101, -1, 3, 4)
    with pytest.raises(ValueError):
        count_bits_between(0b1101, 1, -1, 4)
    with pytest.raises(ValueError):
        count_bits_between(0b1101, 1, 10, 4)

@pytest.mark.parametrize("L, N_f, N_ph, expected_diag", [
    (1, 0, 4, [0, 1, 2, 3, 4]),
    (2, 1, 3, [0, 1, 2, 3, 0, 1, 2, 3]),
    (2, 2, 2, [0, 1, 2, 0, 1, 2]),
    (3, 1, 2, [0, 1, 2, 0, 1, 2, 0, 1, 2]),
    (3, 2, 2, [0, 1, 2, 0, 1, 2, 0, 1, 2]),
])
def test_build_photon_number_matrix(L: int, N_f: int, N_ph: int, expected_diag: list[int]):
    basis = Basis(L, N_f, N_ph)  
    photon_number_matrix = build_photon_number_matrix(basis)
    for i in range(len(basis)):
        assert photon_number_matrix[i, i] == expected_diag[i]

 
    


# tests of operators that are not currently used in the codebase 

# @pytest.mark.parametrize("state, site, L, expected", [
#     ((1, 0b0000), 0, 4, (1, 0b0001)),
#     ((1, 0b0001), 1, 4, (1, 0b0011)),
#     ((-0.5, 0b0011), 2, 4, (-0.5, 0b0111)),
#     ((-0.75, 0b0111), 3, 4, (-0.75, 0b1111)),
#     ((1, 0b1111), 0, 4, (0, 0)),  # already occupied
#     ((1, 0b1111), 2, 4, (0, 0)),  # already occupied
#     ((1, 0b10), 0, 2, (-1, 0b11)),  # fermionic sign
#     ((1, 0b100), 1, 3, (-1, 0b110)),  # fermionic sign
#     ((1, 0b100), 0, 3, (-1, 0b101)),  # fermionic sign
#     ((1, 0b110), 0, 3, (1, 0b111)),  # fermionic sign
#     ((1, 0b1010001110), 5, 10, (1, 0b1010101110)),
#     ((1, 0b1010001110), 0, 10, (-1, 0b1010001111)),
# ])
# def test_fermion_creator(state: tuple[float, int], site: int, L: int, expected: tuple[float, int]):
#     assert fermion_creator(state, site, L) == expected
    
# @pytest.mark.parametrize("state, site, L, expected", [
#     ((1, 0b0001), 0, 4, (1, 0b0000)),
#     ((1, 0b0011), 1, 4, (1, 0b0001)),
#     ((-0.5, 0b0111), 2, 4, (-0.5, 0b0011)),
#     ((-0.75, 0b1111), 3, 4, (-0.75, 0b0111)),
#     ((1, 0b0000), 0, 4, (0, 0)),  # unoccupied
#     ((1, 0b1010), 2, 4, (0, 0)),  # unoccupied
#     ((1, 0b11), 0, 2, (-1, 0b10)),  # fermionic sign
#     ((1, 0b110), 1, 3, (-1, 0b100)),  # fermionic sign
#     ((1, 0b101), 0, 3, (-1, 0b100)),  # fermionic sign
#     ((1, 0b111), 0, 3, (1, 0b110)),  # fermionic sign
#     ((1, 0b1010101110), 5, 10, (1, 0b1010001110)),
#     ((1, 0b1010001111), 0, 10, (-1, 0b1010001110)),
# ])
# def test_fermion_annihilator(state: tuple[float, int], site: int, L: int, expected: tuple[float, int]):
#     assert fermion_annihilator(state, site, L) == expected, f"Failed for state {state}, site {site}, L {L}"


# @pytest.mark.parametrize("state, site, expected", [
#     ((1, 0b0000), 0, (0, 0)),
#     ((1, 0b0011), 1, (1, 0b0011)),
#     ((1, 0b1011), 2, (0, 0)),
#     ((1, 0b101100101001), 3, (1, 0b101100101001)),
#     ((1, 0b101100101001), 4, (0, 0)),
# ])
# def test_fermion_number_operator(state: tuple[float, int], site: int, expected: tuple[float, int]):
#     assert fermion_number_operator(state, site) == expected

# def test_fermion_number_operator_via_creation_annihilation():
#     # check if n_j = c_j^† c_j
#     state = (1, 0b101)
#     annihilated = fermion_annihilator(state, 0, 3)
#     assert annihilated == (-1, 0b100)
#     created = fermion_creator(annihilated, 0, 3)
#     assert created == fermion_number_operator(state, 0)

# def test_creator_annihilator_right_inverse():
#     state = (1, 0b101)
#     created = fermion_creator(state, 1, 3)
#     annihilated = fermion_annihilator(created, 1, 3)
#     assert annihilated == state


# @pytest.mark.parametrize("state, width, expected", [
#     ((1, 0b0000), 4, (0, 0)),
#     ((1, 0b1111), 4, (4, 0b1111)),
#     ((-3, 0b1010), 4, (-6, 0b1010)),
#     ((1, 0b100101), 6, (3, 0b100101)),
#     ((1, 0b111000111), 9, (6, 0b111000111)),
# ])
# def test_total_fermion_number_operator(state: tuple[float, int], width: int, expected: tuple[float, int]):
#     assert total_fermion_number_operator(state, width) == expected