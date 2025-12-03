import numpy as np
import pytest

from numpy.testing import assert_allclose

from src.exact_diagonalization.analyzer import Analyzer
from src.exact_diagonalization.hamiltonian_builder import HamiltonianBuilder
from src.exact_diagonalization.basis import Basis
from src.exact_diagonalization.analyzer import Analyzer


@pytest.mark.parametrize("L, N_f, N_ph, t, U, g, omega", [
    (4, 2, 3, 1.0, 1.0, 1.5, 1.0),
    (6, 3, 2, 1.0, 4.0, 1.0, 2.0),
    (8, 4, 4, 1.0, -1.0, 0.0, 0.0),
])
def test_diagonalize(L: int, N_f: int, N_ph: int, t: float, U: float, g: float, omega: float):
    basis = Basis(L, N_f, N_ph)
    builder = HamiltonianBuilder(basis, g=g, boundary_conditions="periodic")
    H = builder.build_hamiltonian_matrix(t=t, U=U, omega=omega)
    analyzer = Analyzer(H, basis)

    k = 3
    evals, evecs = analyzer.diagonalize(k=k)

    # Check the shapes of the returned eigenvalues and eigenvectors
    assert evals.shape == (k,) 
    assert evecs.shape == (H.get_shape()[0], k)


@pytest.mark.parametrize("L, N_f, N_ph, t, U, g, omega", [
    (4, 2, 2, 1.0, 2.0, 0.5, 1.0),
    (6, 3, 3, 1.0, 5.0, 1.0, 2.0),
    (8, 4, 4, 1.0, -1.0, 0.0, 0.0),
])
def test_ground_state(L: int, N_f: int, N_ph: int, t: float, U: float, g: float, omega: float):
    basis = Basis(L, N_f, N_ph)
    builder = HamiltonianBuilder(basis, g=g, boundary_conditions="periodic")
    H = builder.build_hamiltonian_matrix(t=t, U=U, omega=omega)
    analyzer = Analyzer(H, basis)
    gs = analyzer.ground_state()

    # Check the shape of the ground state vector
    assert gs.shape == (H.get_shape()[0],)

@pytest.mark.parametrize("L, N_f, N_ph, t, U, g, omega", [
    (4, 2, 4, 1.0, 2.0, -0.5, 1.0),
    (6, 3, 3, 1.0, 4.0, 1.0, -2.0),
    (8, 4, 2, 1.0, -3.0, 0.0, 0.0),
])
def test_build_psi_matrix(L: int, N_f: int, N_ph: int, t: float, U: float, g: float, omega: float, atol: float = 1e-14):
    """Check whether the assumption of having the basis in lexicographical order is valid in build_psi_matrix."""
    basis = Basis(L, N_f, N_ph)
    builder = HamiltonianBuilder(basis, g=g, boundary_conditions="periodic")
    H = builder.build_hamiltonian_matrix(t=t, U=U, omega=omega)
    analyzer = Analyzer(H, basis)
    psi = analyzer.diagonalize(k=1)[1][:, 0]  # ground state
    
    # carefully reshape psi into a matrix with dimensions (dim_fermions, dim_photons)
    expected_psi_matrix = np.zeros((analyzer.dim_el, analyzer.dim_ph), dtype=complex)
    for state in analyzer.basis.states:
        idx_el = analyzer.basis.fermion_states.index(state[0])
        idx_ph = analyzer.basis.photon_states.index(state[1])
        basis_idx = analyzer.basis.states.index(state)
        expected_psi_matrix[idx_el, idx_ph] = psi[basis_idx]

    psi_matrix = analyzer.build_psi_matrix(psi)
    assert_allclose(psi_matrix, expected_psi_matrix, atol=atol)


def test_entanglement_entropy_fermions_photons():
    L = 4
    N_f = 2
    N_ph = 2
    t = 1.0
    U = 2.0
    g = 0.5
    omega = 1.0
    basis = Basis(L, N_f, N_ph)
    builder = HamiltonianBuilder(basis, g=g, boundary_conditions="periodic")
    H = builder.build_hamiltonian_matrix(t=t, U=U, omega=omega)
    analyzer = Analyzer(H, basis)

    # Create a maximally entangled state between fermions and photons
    dim_el = analyzer.dim_el
    dim_ph = analyzer.dim_ph
    psi = np.zeros(dim_el * dim_ph, dtype=complex)
    for i in range(min(dim_el, dim_ph)):
        psi[i * dim_ph + i] = 1 / np.sqrt(min(dim_el, dim_ph))
    
    entropy = analyzer.entanglement_entropy_fermions_photons(psi)
    expected_entropy = np.log(min(dim_el, dim_ph))
    assert np.isclose(entropy, expected_entropy), f"Expected {expected_entropy}, got {entropy}"


# TODO: Go again over that test
@pytest.mark.parametrize("L, N_f, N_ph, t, U, g, omega", [
    (4, 2, 2, 1.0, 2.0, 0.5, 1.0),
    (6, 3, 3, 1.0, 5.0, 1.0, 2.0),
    (8, 4, 4, 1.0, -1.0, 0.0, 0.0),
])
def test_expectation_value(L: int, N_f: int, N_ph: int, t: float, U: float, g: float, omega: float):
    basis = Basis(L, N_f, N_ph)
    builder = HamiltonianBuilder(basis, g=g, boundary_conditions="periodic")
    H = builder.build_hamiltonian_matrix(t=t, U=U, omega=omega)
    analyzer = Analyzer(H, basis)

    # Create a test state
    dim_el = analyzer.dim_el
    dim_ph = analyzer.dim_ph
    psi = np.ones(dim_el * dim_ph, dtype=complex)
    psi /= np.linalg.norm(psi)  # normalize

    # Define an operator (identity operator in this case)
    operator = np.eye(dim_el * dim_ph)

    expectation = analyzer.expectation_value(psi, operator)
    expected_expectation = 1.0  # since psi is normalized and operator is identity
    assert_allclose(expectation, expected_expectation)

# TODO: Go again over that test
def test_expectation_value_non_hermitian_operator():
    L = 4
    N_f = 2
    N_ph = 2
    t = 1.0
    U = 2.0
    g = 0.5
    omega = 1.0
    basis = Basis(L, N_f, N_ph)
    builder = HamiltonianBuilder(basis, g=g, boundary_conditions="periodic")
    H = builder.build_hamiltonian_matrix(t=t, U=U, omega=omega)
    analyzer = Analyzer(H, basis)

    # Create a test state
    dim_el = analyzer.dim_el
    dim_ph = analyzer.dim_ph
    psi = np.zeros(dim_el * dim_ph, dtype=complex)
    for i in range(min(dim_el, dim_ph)):
        psi[i * dim_ph + i] = 1 / np.sqrt(min(dim_el, dim_ph))

    # Define a non-hermitian operator
    operator = np.array([[0, 1], [0, 0]], dtype=complex)

    with pytest.raises(ValueError):
        analyzer.expectation_value(psi, operator)