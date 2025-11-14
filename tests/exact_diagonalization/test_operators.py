import pytest
from src.exact_diagonalization.operators import count_pairs, flip_bit, count_bits_between, fermion_creator, \
    fermion_annihilator, fermion_number_operator, total_fermion_number_operator 
    

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

def test_count_bits_between():
    assert count_bits_between(0b0, 0, 0) == 0
    assert count_bits_between(0b1, 0, 0) == 1
    assert count_bits_between(0b1101, 1, 3, inclusive=True) == 2
    assert count_bits_between(0b1101, 1, 3, inclusive=False) == 1
    assert count_bits_between(0b111111, 0, 5, inclusive=True) == 6
    assert count_bits_between(0b111111, 0, 5, inclusive=False) == 5

    with pytest.raises(ValueError):
        count_bits_between(0b1101, 3, 1)

def test_fermion_creator():
    # basic tests
    assert fermion_creator((1, 0b0000), 0, 4) == (1, 0b0001)
    assert fermion_creator((1, 0b0001), 1, 4) == (1, 0b0011)
    assert fermion_creator((-0.5, 0b0011), 2, 4) == (-0.5, 0b0111)
    assert fermion_creator((-0.75, 0b0111), 3, 4) == (-0.75, 0b1111)
    assert fermion_creator((1, 0b1111), 0, 4) == (0, 0)  
    assert fermion_creator((1, 0b1111), 2, 4) == (0, 0)  
    # test fermionic sign
    assert fermion_creator((1, 0b10), 0, 2) == (-1, 0b11)
    assert fermion_creator((1, 0b100), 1, 3) == (-1, 0b110)
    assert fermion_creator((1, 0b100), 0, 3) == (-1, 0b101)
    assert fermion_creator((1, 0b110), 0, 3) == (1, 0b111)
    assert fermion_creator((1, 0b1010001110), 5, 10) == (1, 0b1010101110)
    assert fermion_creator((1, 0b1010001110), 0, 10) == (-1, 0b1010001111)

def test_fermion_annihilator():
    # basic tests
    assert fermion_annihilator((1, 0b0001), 0, 4) == (1, 0b0000)
    assert fermion_annihilator((1, 0b0011), 1, 4) == (1, 0b0001)
    assert fermion_annihilator((-0.5, 0b0111), 2, 4) == (-0.5, 0b0011)
    assert fermion_annihilator((-0.75, 0b1111), 3, 4) == (-0.75, 0b0111)
    assert fermion_annihilator((1, 0b0000), 0, 4) == (0, 0)
    assert fermion_annihilator((1, 0b1010), 2, 4) == (0, 0)
    # test fermionic sign
    assert fermion_annihilator((1, 0b11), 0, 2) == (-1, 0b10)
    assert fermion_annihilator((1, 0b110), 1, 3) == (-1, 0b100)
    assert fermion_annihilator((1, 0b101), 0, 3) == (-1, 0b100)
    assert fermion_annihilator((1, 0b111), 0, 3) == (1, 0b110)
    assert fermion_annihilator((1, 0b1010101110), 5, 10) == (1, 0b1010001110)
    assert fermion_annihilator((1, 0b1010001111), 0, 10) == (-1, 0b1010001110)


def test_fermion_number_operator():
    assert fermion_number_operator((1, 0b0000), 0) == (0, 0)
    assert fermion_number_operator((1, 0b0011), 1) == (1, 0b0011)
    assert fermion_number_operator((1, 0b1011), 2) == (0, 0)
    assert fermion_number_operator((1, 0b101100101001), 3) == (1, 0b101100101001)
    assert fermion_number_operator((1, 0b101100101001), 4) == (0, 0)
    # check if n_j = c_j^† c_j
    state = (1, 0b101)
    annihilated = fermion_annihilator(state, 0, 3)
    assert annihilated == (-1, 0b100)
    created = fermion_creator(annihilated, 0, 3)
    assert created == fermion_number_operator(state, 0)

def test_total_fermion_number_operator():
    assert total_fermion_number_operator((1, 0b0000), 4) == (0, 0)
    assert total_fermion_number_operator((1, 0b1111), 4) == (4, 0b1111)
    assert total_fermion_number_operator((-3, 0b1010), 4) == (-6, 0b1010)
    assert total_fermion_number_operator((1, 0b100101), 6) == (3, 0b100101)
    assert total_fermion_number_operator((1, 0b111000111), 9) == (6, 0b111000111)