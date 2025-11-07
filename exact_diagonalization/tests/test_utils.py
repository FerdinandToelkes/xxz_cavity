import pytest

from src.utils import circular_left_shift, circular_right_shift



def test_circular_left_shift():
    assert circular_left_shift(0b00001, 1, 5) == 0b00010
    assert circular_left_shift(0b00001, 2, 5) == 0b00100
    assert circular_left_shift(0b00001, 3, 5) == 0b01000
    assert circular_left_shift(0b00001, 4, 5) == 0b10000
    assert circular_left_shift(0b00001, 5, 5) == 0b00001

    assert circular_left_shift(0b1010011010, 2, 10) == 0b1001101010
    assert circular_left_shift(0b1010011010, 5, 10) == 0b1101010100

    with pytest.raises(ValueError):
        circular_left_shift(0b1010011010, 5, 9)

def test_circular_right_shift():
    assert circular_right_shift(0b00001, 1, 5) == 0b10000
    assert circular_right_shift(0b00001, 2, 5) == 0b01000
    assert circular_right_shift(0b00001, 3, 5) == 0b00100
    assert circular_right_shift(0b00001, 4, 5) == 0b00010
    assert circular_right_shift(0b00001, 5, 5) == 0b00001

    assert circular_right_shift(0b1010011010, 2, 10) == 0b1010100110
    assert circular_right_shift(0b1010011010, 5, 10) == 0b1101010100

    with pytest.raises(ValueError):
        circular_right_shift(0b1010011010, 5, 9)