import pytest

from ed.basis import Basis

@pytest.mark.parametrize("L, N_f, N_ph, expected_states", [
    (2, 0, 0, [(0b00, 0)]),
    (10, 10, 0, [(0b1111111111, 0)]),
    (3, 1, 0, [(0b001, 0), (0b010, 0), (0b100, 0)]),
    (4, 2, 0, [(0b0011, 0), (0b0101, 0), (0b0110, 0), (0b1001, 0), (0b1010, 0), (0b1100, 0)]),
    (2, 1, 2, [(0b01, 0), (0b01, 1), (0b01, 2), (0b10, 0), (0b10, 1), (0b10, 2)]),
    (3, 2, 1, [(0b011, 0), (0b011, 1), (0b101, 0), (0b101, 1), (0b110, 0), (0b110, 1)]),
])
def test_generate_basis(L: int, N_f: int, N_ph: int, expected_states: list[tuple[int, int]]):
    basis = Basis(L, N_f, N_ph)
    assert basis.states == expected_states

   
@pytest.mark.parametrize("L, N_f, expected_states", [
    (2, 0, [0b00]),
    (10, 10, [0b1111111111]),
    (3, 1, [0b001, 0b010, 0b100]),
    (4, 2, [0b0011, 0b0101, 0b0110, 0b1001, 0b1010, 0b1100]),
])
def test_generate_fermion_basis(L: int, N_f: int, expected_states: list[int]):
    basis = Basis(L, N_f)
    assert basis.fermion_states == expected_states


@pytest.mark.parametrize("L, N_f, N_ph, expected_states", [
    (4, 2, 0, [0]),
    (4, 2, 3, [0, 1, 2, 3]),
])
def test_generate_photonic_basis(L: int, N_f: int, N_ph: int, expected_states: list[int]):
    basis = Basis(L, N_f, N_ph)
    assert basis.photon_states == expected_states

@pytest.mark.parametrize("L, N_f, N_ph, expected_indices", [
    (10, 10, 0, {(0b1111111111, 0): 0}),
    (3, 1, 0, {(0b001, 0): 0, (0b010, 0): 1, (0b100, 0): 2}),
    (4, 2, 0, {(0b0011, 0): 0, (0b0101, 0): 1, (0b0110, 0): 2, (0b1001, 0): 3, (0b1010, 0): 4, (0b1100, 0): 5}),
    (2, 1, 1, {(0b01, 0): 0, (0b01, 1): 1, (0b10, 0): 2, (0b10, 1): 3}),
])
def test_state_index(L: int, N_f: int, N_ph: int, expected_indices: dict[tuple[int, int], int]):
    basis = Basis(L, N_f, N_ph)
    for state, expected_index in expected_indices.items():
        assert basis.state_index(state) == expected_index

def test_state_index_invalid():
    basis = Basis(4, 2)
    with pytest.raises(KeyError):
        basis.state_index((0b1111, 0))  # Not in basis

@pytest.mark.parametrize("L, N_f, N_ph, expected_indices", [
    (4, 2, 0, {0b0011: 0, 0b0101: 1, 0b0110: 2, 0b1001: 3, 0b1010: 4, 0b1100: 5}),
])
def test_fermion_state_index(L: int, N_f: int, N_ph: int, expected_indices: dict[int, int]):
    basis = Basis(L, N_f, N_ph)
    for state, expected_index in expected_indices.items():
        assert basis.fermion_state_index(state) == expected_index

def test_fermion_state_index_invalid():
    basis = Basis(4, 2)
    with pytest.raises(KeyError):
        basis.fermion_state_index(0b1111)  # Not in basis

@pytest.mark.parametrize("L, N_f, N_ph, expected_indices", [
    (4, 2, 3, {0: 0, 1: 1, 2: 2, 3: 3}),
])
def test_photon_state_index(L: int, N_f: int, N_ph: int, expected_indices: dict[int, int]):
    basis = Basis(L, N_f, N_ph)
    for state, expected_index in expected_indices.items():
        assert basis.photon_state_index(state) == expected_index

def test_photon_state_index_invalid():
    basis = Basis(4, 2, 3)
    with pytest.raises(KeyError):
        basis.photon_state_index(4)  # Not in basis

@pytest.mark.parametrize("L, N_f, N_ph, expected_states", [
    (3, 1, 0, [(0b001, 0), (0b010, 0), (0b100, 0)]),
    (2, 1, 2, [(0b01, 0), (0b01, 1), (0b01, 2), (0b10, 0), (0b10, 1), (0b10, 2)]),
])
def test_basis_len_and_iter(L: int, N_f: int, N_ph: int, expected_states: list[tuple[int, int]]):
    basis = Basis(L, N_f, N_ph)
    assert len(basis) == len(expected_states)

    for idx, state in enumerate(basis):
        assert state == expected_states[idx]