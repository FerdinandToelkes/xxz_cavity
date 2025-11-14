import pytest

from src.exact_diagonalization.basis import Basis

def test_generate_basis():
    basis = Basis(2, 0)
    expected_states = [(0b00, 0)]
    assert basis.states == expected_states

    basis = Basis(10, 10)
    expected_states = [(0b1111111111, 0)]
    assert basis.states == expected_states

    basis = Basis(3, 1)
    expected_states = [(0b001, 0), (0b010, 0), (0b100, 0)]
    assert basis.states == expected_states

    basis = Basis(4, 2)
    expected_states = [(0b0011, 0), (0b0101, 0), (0b0110, 0), (0b1001, 0), (0b1010, 0), (0b1100, 0)]
    assert basis.states == expected_states

def test_state_index():
    basis = Basis(2, 0)
    assert basis.state_index((0b00, 0)) == 0

    basis = Basis(10, 10)
    assert basis.state_index((0b1111111111, 0)) == 0

    basis = Basis(3, 1)
    assert basis.state_index((0b001, 0)) == 0
    assert basis.state_index((0b010, 0)) == 1
    assert basis.state_index((0b100, 0)) == 2

    basis = Basis(4, 2)
    assert basis.state_index((0b0011, 0)) == 0
    assert basis.state_index((0b0101, 0)) == 1
    assert basis.state_index((0b0110, 0)) == 2
    assert basis.state_index((0b1001, 0)) == 3
    assert basis.state_index((0b1010, 0)) == 4
    assert basis.state_index((0b1100, 0)) == 5
    with pytest.raises(KeyError):
        basis.state_index((0b1111, 0))  # Not in basis

def test_basis_len_and_iter():
    basis = Basis(3, 1)
    expected_states = [(0b001, 0), (0b010, 0), (0b100, 0)]
    assert len(basis) == len(expected_states)

    for idx, state in enumerate(basis):
        assert state == expected_states[idx]