import pytest

from src.basis import Basis

def test_generate_basis():
    basis = Basis(2, 0)
    expected_states = [0b00]
    assert basis.states == expected_states

    basis = Basis(10, 10)
    expected_states = [0b1111111111]
    assert basis.states == expected_states

    basis = Basis(3, 1)
    expected_states = [0b001, 0b010, 0b100]
    assert basis.states == expected_states

    basis = Basis(4, 2)
    expected_states = [0b0011, 0b0101, 0b0110, 0b1001, 0b1010, 0b1100]
    assert basis.states == expected_states

def test_state_index():
    basis = Basis(2, 0)
    assert basis.state_index(0b00) == 0

    basis = Basis(10, 10)
    assert basis.state_index(0b1111111111) == 0

    basis = Basis(3, 1)
    assert basis.state_index(0b001) == 0
    assert basis.state_index(0b010) == 1
    assert basis.state_index(0b100) == 2

    basis = Basis(4, 2)
    assert basis.state_index(0b0011) == 0
    assert basis.state_index(0b0101) == 1
    assert basis.state_index(0b0110) == 2
    assert basis.state_index(0b1001) == 3
    assert basis.state_index(0b1010) == 4
    assert basis.state_index(0b1100) == 5
    with pytest.raises(KeyError):
        basis.state_index(0b1111)  # Not in basis

def test_basis_len_and_iter():
    basis = Basis(3, 1)
    expected_states = [0b001, 0b010, 0b100]
    assert len(basis) == len(expected_states)

    for idx, state in enumerate(basis):
        assert state == expected_states[idx]