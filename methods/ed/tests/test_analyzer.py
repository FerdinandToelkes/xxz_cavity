import numpy as np
import pytest

from scipy.sparse import csr_matrix

from tests.utils import assert_allclose
from ed.basis import Basis
from ed.hamiltonian_builder import HamiltonianBuilder
from ed.analyzer import Analyzer


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

def test_diagonalize_non_hermitian():
    # Create a non-Hermitian matrix
    H_non_hermitian = np.array([[1, 2], [3, 4]], dtype=complex)
    H_non_hermitian_csr = csr_matrix(H_non_hermitian)

    basis = Basis(2, 1)  # Dummy basis
    analyzer = Analyzer(H_non_hermitian_csr, basis)

    with pytest.raises(ValueError):
        analyzer.diagonalize(k=1)


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
def test_build_psi_matrix(L: int, N_f: int, N_ph: int, t: float, U: float, g: float, omega: float):
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
    assert_allclose(psi_matrix, expected_psi_matrix)

@pytest.mark.parametrize("L, N_f, N_ph, t, U, g, omega, boundary_conditions", [
    (2, 1, 3, 1.0, 2.0, 0.5, 1.0, "open"),
    (4, 2, 2, 1.0, 2.0, 0.5, 1.0, "periodic"),
    (6, 3, 3, 1.0, 4.0, 1.0, 2.0, "periodic"),
    (8, 4, 4, 1.0, -1.0, 0.0, 0.0, "periodic"),
])
def test_entanglement_entropy_fermions_photons_minimal(L: int, N_f: int, N_ph: int, t: float, U: float, 
                                                       g: float, omega: float, boundary_conditions: str):
    basis = Basis(L, N_f, N_ph)
    builder = HamiltonianBuilder(basis, g=g, boundary_conditions=boundary_conditions)
    H = builder.build_hamiltonian_matrix(t=t, U=U, omega=omega)
    analyzer = Analyzer(H, basis)

    # Create a minimally entangled state between fermions and photons
    dim_el = analyzer.dim_el
    dim_ph = analyzer.dim_ph
    psi = np.zeros(dim_el * dim_ph, dtype=complex)
    # only set first element
    psi[0] = 1 
    
    entropy = analyzer.entanglement_entropy_fermions_photons(psi)
    expected_entropy = 0.0  # since only one component is non-zero
    assert_allclose(entropy, expected_entropy)

@pytest.mark.parametrize("L, N_f, N_ph, t, U, g, omega, boundary_conditions", [
    (2, 1, 3, 1.0, 2.0, 0.5, 1.0, "open"),
    (4, 2, 2, 1.0, 2.0, 0.5, 1.0, "periodic"),
    (6, 3, 3, 1.0, 4.0, 1.0, 2.0, "periodic"),
    (8, 4, 4, 1.0, -1.0, 0.0, 0.0, "periodic"),
])
def test_entanglement_entropy_fermions_photons_maximal(L: int, N_f: int, N_ph: int, t: float, U: float, 
                                                       g: float, omega: float, boundary_conditions: str):
    basis = Basis(L, N_f, N_ph)
    builder = HamiltonianBuilder(basis, g=g, boundary_conditions=boundary_conditions)
    H = builder.build_hamiltonian_matrix(t=t, U=U, omega=omega)
    analyzer = Analyzer(H, basis)

    # Create a maximally entangled state between fermions and photons
    dim_el = analyzer.dim_el
    dim_ph = analyzer.dim_ph
    psi = np.zeros(dim_el * dim_ph, dtype=complex)
    # ensure that when psi is built, this will be diagonal in the psi_matrix
    for i in range(min(dim_el, dim_ph)):
        basis_idx = i * dim_ph + i  
        psi[basis_idx] = 1.0
    psi /= np.linalg.norm(psi) # normalize
    
    entropy = analyzer.entanglement_entropy_fermions_photons(psi)
    expected_entropy = np.log(min(dim_el, dim_ph))  # see e.g. Wikipedia page on entanglement entropy
    assert_allclose(entropy, expected_entropy)


@pytest.mark.parametrize("L, N_f, N_ph, t, U, g, omega", [
    (4, 2, 2, 1.0, 2.0, 0.5, 1.0),
    (6, 3, 3, 1.0, 5.0, 1.0, 2.0),
    (8, 4, 4, 1.0, -1.0, 0.0, 0.0),
])
def test_entanglement_entropy_fermions_photons_non_normalized(L: int, N_f: int, N_ph: int, t: float, U: float, g: float, omega: float):
    basis = Basis(L, N_f, N_ph)
    builder = HamiltonianBuilder(basis, g=g, boundary_conditions="periodic")
    H = builder.build_hamiltonian_matrix(t=t, U=U, omega=omega)
    analyzer = Analyzer(H, basis)

    # Create a non-normalized state
    dim_el = analyzer.dim_el
    dim_ph = analyzer.dim_ph
    psi = np.ones(dim_el * dim_ph, dtype=complex)  # not normalized

    with pytest.raises(ValueError):
        analyzer.entanglement_entropy_fermions_photons(psi)

@pytest.mark.parametrize("L, N_f, N_ph, t, U, g, omega", [
    (4, 2, 3, 1.0, 1.5, 1.5, 0.75),
    (6, 3, 1, 1.0, 3.3, 1.2, 2.2),
    (8, 4, 2, 1.0, -1.5, 0.0, 0.0),
])
def test_probability_distribution_photon_number_uniform(L: int, N_f: int, N_ph: int, t: float, U: float, g: float, omega: float):
    basis = Basis(L, N_f, N_ph)
    builder = HamiltonianBuilder(basis, g=g, boundary_conditions="periodic")
    H = builder.build_hamiltonian_matrix(t=t, U=U, omega=omega)
    analyzer = Analyzer(H, basis)

    # Create a test state with known photon number distribution
    dim_el = analyzer.dim_el
    dim_ph = analyzer.dim_ph
    psi = np.zeros(dim_el * dim_ph, dtype=complex)
    # Set amplitudes for different photon numbers
    for n_ph in range(dim_ph):
        for n_el in range(dim_el):
            basis_idx = n_el * dim_ph + n_ph
            psi[basis_idx] = 1.0 / np.sqrt(dim_el * dim_ph)  # equal superposition

    prob_distribution = analyzer.probability_distribution_photon_number(psi)
    expected_distribution = np.full(dim_ph, 1.0 / dim_ph)  # uniform distribution over photon numbers
    assert_allclose(prob_distribution, expected_distribution)

@pytest.mark.parametrize("L, N_f, N_ph, t, U, g, omega, target_n_ph", [
    (4, 2, 5, 1.0, 4.0, 0.5, 1.0, 3),
    (6, 3, 4, 1.0, 2.0, 1.0, 2.0, 2),
    (8, 4, 6, 1.0, -1.0, 0.0, 0.0, 4),
])
def test_probability_distribution_photon_number_single_photon_number(L: int, N_f: int, N_ph: int, t: float, U: float, g: float, omega: float, target_n_ph: int):
    basis = Basis(L, N_f, N_ph)
    builder = HamiltonianBuilder(basis, g=g, boundary_conditions="open")
    H = builder.build_hamiltonian_matrix(t=t, U=U, omega=omega)
    analyzer = Analyzer(H, basis)

    # Create a test state with all amplitude in a single photon number sector
    dim_el = analyzer.dim_el
    dim_ph = analyzer.dim_ph
    psi = np.zeros(dim_el * dim_ph, dtype=complex)
    for n_el in range(dim_el):
        basis_idx = n_el * dim_ph + target_n_ph
        psi[basis_idx] = 1.0 / np.sqrt(dim_el)  # equal superposition in this sector

    prob_distribution = analyzer.probability_distribution_photon_number(psi)
    expected_distribution = np.zeros(dim_ph)
    expected_distribution[target_n_ph] = 1.0  # all probability in target_n_ph
    assert_allclose(prob_distribution, expected_distribution)

@pytest.mark.parametrize("L, N_f, N_ph, t, U, g, omega", [
    (4, 2, 5, 1.0, 2.5, 0.75, -1.0),
    (6, 3, 1, 1.0, 4.0, 2.0, 1.0),
    (8, 4, 3, 1.0, 1.0, 0.0, 0.0),
])
def test_probability_distribution_photon_number_non_normalized(L: int, N_f: int, N_ph: int, t: float, U: float, g: float, omega: float):
    basis = Basis(L, N_f, N_ph)
    builder = HamiltonianBuilder(basis, g=g, boundary_conditions="periodic")
    H = builder.build_hamiltonian_matrix(t=t, U=U, omega=omega)
    analyzer = Analyzer(H, basis)

    # Create a non-normalized state
    dim_el = analyzer.dim_el
    dim_ph = analyzer.dim_ph
    psi = np.ones(dim_el * dim_ph, dtype=complex)  # not normalized

    with pytest.raises(ValueError):
        analyzer.probability_distribution_photon_number(psi)

@pytest.mark.parametrize("L, N_f, N_ph, t, U, g, omega", [
    (4, 2, 2, 0, 0, 0, 0), # only dimensions matter since we set psi ourselves
    (6, 3, 3, 0, 0, 0, 0),
    (8, 4, 4, 0, 0, 0, 0),
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

@pytest.mark.parametrize("L, N_f, N_ph, omega, boundary_condition, expected_n_photon", [
    (2, 1, 2, 1, "open", 0.0),
    (4, 2, 4, -1, "periodic", 4.0),
    (6, 3, 6, 1, "periodic", 0.0),
    (8, 4, 8, -1, "open", 8.0),
])
def test_expectation_value_photon_number(L: int, N_f: int, N_ph: int, omega: float, boundary_condition: str, expected_n_photon: float):
    # make simple non-diagonal Hamiltonian -> Lanczos doesn't work for good for already diagonal matrices 
    t, U, g = 1.0, 0.0, 0.0
    basis = Basis(L, N_f, N_ph)
    builder = HamiltonianBuilder(basis, g=g, boundary_conditions=boundary_condition)
    H = builder.build_hamiltonian_matrix(t=t, U=U, omega=omega)
    analyzer = Analyzer(H=H, basis=basis)
    # evals, evecs = analyzer.diagonalize(k=10)
    ground_state = analyzer.ground_state()

    # Define an operator (identity operator in this case)
    operator = analyzer.photon_number_matrix

    expectation = analyzer.expectation_value(ground_state, operator)
    assert_allclose(expectation, expected_n_photon)

@pytest.mark.parametrize("L, N_f, N_ph, t, U, g, omega", [
    (2, 1, 0, 0, 0, 0, 0), # only dimensions matter since we set psi ourselves
])
def test_expectation_value_non_hermitian_operator(L: int, N_f: int, N_ph: int, t: float, U: float, g: float, omega: float):
    basis = Basis(L, N_f, N_ph)
    builder = HamiltonianBuilder(basis, g=g, boundary_conditions="open")
    H = builder.build_hamiltonian_matrix(t=t, U=U, omega=omega)
    analyzer = Analyzer(H, basis)

    # Create a test state
    dim_el = analyzer.dim_el
    dim_ph = analyzer.dim_ph
    psi = np.ones(dim_el * dim_ph, dtype=complex)
    psi /= np.linalg.norm(psi)  # normalize

    # Define a complex-valued, non-hermitian operator -> expectation value can be complex
    operator = np.array([[0, 1j], [0, 0]], dtype=complex)
    with pytest.raises(ValueError):
        analyzer.expectation_value(psi, operator)