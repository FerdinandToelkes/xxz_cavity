from src.exact_diagonalization.hamiltonian import Hamiltonian
from src.exact_diagonalization.basis import Basis

def test_hamiltonian_construction_pbc():
    basis = Basis(4, 2)  # 4 sites, 2 particles
    hamiltonian = Hamiltonian(basis, boundary_conditions="periodic")
    H = hamiltonian.construct_hamiltonian_matrix(t=1.0, U=2.0)

    # see notes on tablet
    hopping_matrix = [
        [0.0, -1.0,  0.0,  0.0,  1.0,  0.0],
        [-1.0, 0.0, -1.0, -1.0,  0.0,  1.0],
        [0.0, -1.0,  0.0,  0.0, -1.0,  0.0],
        [0.0, -1.0,  0.0,  0.0, -1.0,  0.0],
        [1.0,  0.0, -1.0, -1.0,  0.0, -1.0],
        [0.0,  1.0,  0.0,  0.0, -1.0,  0.0],
    ]

    interaction_matrix = [
        [1.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  1.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  1.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  0.0,  0.0,  1.0],
    ]

    full_matrix = [
        [2.0, -1.0,  0.0,  0.0,  1.0,  0.0],
        [-1.0, 0.0, -1.0, -1.0,  0.0,  1.0],
        [0.0, -1.0,  2.0,  0.0, -1.0,  0.0],
        [0.0, -1.0,  0.0,  2.0, -1.0,  0.0],
        [1.0,  0.0, -1.0, -1.0,  0.0, -1.0],
        [0.0,  1.0,  0.0,  0.0, -1.0,  2.0],
    ]

    # note that hamiltonian.matrices are sparse matrices
    for i in range(len(full_matrix)):
        for j in range(len(full_matrix)):
            assert hamiltonian.hopping_matrix[i, j] == hopping_matrix[i][j]
            assert hamiltonian.interaction_matrix[i, j] == interaction_matrix[i][j]
            assert H[i, j] == full_matrix[i][j]

    # Now test with different t and U
    H = hamiltonian.construct_hamiltonian_matrix(t=-3.0, U=-4.0)

    full_matrix = [
        [-4.0,  3.0,  0.0,  0.0, -3.0,  0.0],
        [ 3.0,  0.0,  3.0,  3.0,  0.0, -3.0],
        [ 0.0,  3.0, -4.0,  0.0,  3.0,  0.0],
        [ 0.0,  3.0,  0.0, -4.0,  3.0,  0.0],
        [-3.0,  0.0,  3.0,  3.0,  0.0,  3.0],
        [ 0.0, -3.0,  0.0,  0.0,  3.0, -4.0],
    ]

    for i in range(len(full_matrix)):
        for j in range(len(full_matrix)):
            assert H[i, j] == full_matrix[i][j]

def test_hamiltonian_construction_obc():
    basis = Basis(4, 2)  # 4 sites, 2 particles
    hamiltonian = Hamiltonian(basis, boundary_conditions="open")
    H = hamiltonian.construct_hamiltonian_matrix(t=-1.0, U=2.0)

    # without t and U specified
    hopping_matrix = [
        [ 0.0, -1.0,  0.0,  0.0,  0.0,  0.0],
        [-1.0,  0.0, -1.0, -1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0,  0.0, -1.0,  0.0],
        [ 0.0, -1.0,  0.0,  0.0, -1.0,  0.0],
        [ 0.0,  0.0, -1.0, -1.0,  0.0, -1.0],
        [ 0.0,  0.0,  0.0,  0.0, -1.0,  0.0],
    ]

    interaction_matrix = [
        [1.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  1.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  0.0,  0.0,  1.0],
    ]

    # H = t* H_hopping + U * H_interaction with t=-1.0, U=2.0
    full_matrix = [
        [ 2.0,  1.0,  0.0,  0.0,  0.0,  0.0],
        [ 1.0,  0.0,  1.0,  1.0,  0.0,  0.0],
        [ 0.0,  1.0,  2.0,  0.0,  1.0,  0.0],
        [ 0.0,  1.0,  0.0,  0.0,  1.0,  0.0],
        [ 0.0,  0.0,  1.0,  1.0,  0.0,  1.0],
        [ 0.0,  0.0,  0.0,  0.0,  1.0,  2.0],
    ]

    for i in range(len(full_matrix)):
        for j in range(len(full_matrix)):
            assert hamiltonian.hopping_matrix[i, j] == hopping_matrix[i][j]
            assert hamiltonian.interaction_matrix[i, j] == interaction_matrix[i][j]
            assert H[i, j] == full_matrix[i][j]