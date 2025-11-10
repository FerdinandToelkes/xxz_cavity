from src.exact_diagonalization.hamiltonian import Hamiltonian
from src.exact_diagonalization.basis import Basis

def test_hamiltonian_construction():
    basis = Basis(4, 2)  # 4 sites, 2 particles
    hamiltonian = Hamiltonian(basis, t=1.0, U=2.0, boundary_conditions="periodic")

    # see notes on tablet
    expected_matrix = [
        [2.0, -1.0,  0.0,  0.0,  1.0,  0.0],
        [-1.0, 0.0, -1.0, -1.0,  0.0,  1.0],
        [0.0, -1.0,  2.0,  0.0, -1.0,  0.0],
        [0.0, -1.0,  0.0,  2.0, -1.0,  0.0],
        [1.0,  0.0, -1.0, -1.0,  0.0, -1.0],
        [0.0,  1.0,  0.0,  0.0, -1.0,  2.0],
    ]

    # note that hamiltonian.matrix is a sparse matrix
    for i in range(len(expected_matrix)):
        for j in range(len(expected_matrix)):
            assert hamiltonian.matrix[i, j] == expected_matrix[i][j]

    basis = Basis(4, 2)  # 4 sites, 2 particles
    hamiltonian = Hamiltonian(basis, t=-1.0, U=-2.0, boundary_conditions="periodic")

    expected_matrix = [
        [-2.0,  1.0,  0.0,  0.0, -1.0,  0.0],
        [ 1.0,  0.0,  1.0,  1.0,  0.0, -1.0],
        [ 0.0,  1.0, -2.0,  0.0,  1.0,  0.0],
        [ 0.0,  1.0,  0.0, -2.0,  1.0,  0.0],
        [-1.0,  0.0,  1.0,  1.0,  0.0,  1.0],
        [ 0.0, -1.0,  0.0,  0.0,  1.0, -2.0],
    ]

    for i in range(len(expected_matrix)):
        for j in range(len(expected_matrix)):
            assert hamiltonian.matrix[i, j] == expected_matrix[i][j]

    basis = Basis(4, 2)  # 4 sites, 2 particles
    hamiltonian = Hamiltonian(basis, t=-1.0, U=2.0, boundary_conditions="open")

    # see notes on tablet
    expected_matrix = [
        [ 2.0,  1.0,  0.0,  0.0,  0.0,  0.0],
        [ 1.0,  0.0,  1.0,  1.0,  0.0,  0.0],
        [ 0.0,  1.0,  2.0,  0.0,  1.0,  0.0],
        [ 0.0,  1.0,  0.0,  0.0,  1.0,  0.0],
        [ 0.0,  0.0,  1.0,  1.0,  0.0,  1.0],
        [ 0.0,  0.0,  0.0,  0.0,  1.0,  2.0],
    ]

    for i in range(len(expected_matrix)):
        for j in range(len(expected_matrix)):
            assert hamiltonian.matrix[i, j] == expected_matrix[i][j]