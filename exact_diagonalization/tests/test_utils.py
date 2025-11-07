import pytest

from src.utils import count_pairs, circular_left_shift, circular_right_shift



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


def test_count_pairs():
    assert count_pairs(0b0, 1, 1) == 0
    assert count_pairs(0b1, 1, 1, "periodic") == 1
    assert count_pairs(0b11, 1, 2, "periodic") == 2
    assert count_pairs(0b111, 1, 3, "periodic") == 3
    assert count_pairs(0b1011, 1, 4, "periodic") == 2
    assert count_pairs(0b111011, 1, 6, "periodic") == 4

    assert count_pairs(0b1, 1, 1, "open") == 0
    assert count_pairs(0b11, 1, 2, "open") == 1
    assert count_pairs(0b111, 1, 3, "open") == 2
    assert count_pairs(0b1011, 1, 4, "open") == 1
    assert count_pairs(0b111011, 1, 6, "open") == 3

    with pytest.raises(ValueError):
        count_pairs(0b111011, 1, 6, "invalid")



