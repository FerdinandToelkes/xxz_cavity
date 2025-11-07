import pytest
from src.operators import count_pairs, flip_bit

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

def test_flip_bit():
    assert flip_bit(0b0000, 0) == 0b0001
    assert flip_bit(0b0001, 0) == 0b0000
    assert flip_bit(0b0010, 1) == 0b0000
    assert flip_bit(0b0000, 1) == 0b0010
    assert flip_bit(0b1010, 2) == 0b1110
    assert flip_bit(0b1110, 2) == 0b1010