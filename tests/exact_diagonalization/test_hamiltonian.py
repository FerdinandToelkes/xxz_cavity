import numpy as np
import pytest

from numpy.testing import assert_allclose

from src.exact_diagonalization.hamiltonian import Hamiltonian
from src.exact_diagonalization.basis import Basis

def test_hamiltonian_invalid_boundary_conditions():
    basis = Basis(2, 2)  # 2 sites, 2 particles
    with pytest.raises(ValueError):
        Hamiltonian(basis, boundary_conditions="invalid_bc")
        
    with pytest.raises(ValueError):
        Hamiltonian(basis, boundary_conditions="periodic") # not valid for two sites

@pytest.mark.parametrize("t", [-3, 1, 2])
@pytest.mark.parametrize("U", [-4, -1, 3])
def test_hamiltonian_construction_pbc_without_light_matter(t: float, U: float, atol: float = 1e-14):
    basis = Basis(4, 2)  # 4 sites, 2 particles
    hamiltonian = Hamiltonian(basis, g=0, boundary_conditions="periodic")
    H = hamiltonian.construct_hamiltonian_matrix(t, U, omega=0)

    # see notes on tablet
    expected_hopping_matrix = np.array([
        [0.0, -1.0,  0.0,  0.0,  1.0,  0.0],
        [-1.0, 0.0, -1.0, -1.0,  0.0,  1.0],
        [0.0, -1.0,  0.0,  0.0, -1.0,  0.0],
        [0.0, -1.0,  0.0,  0.0, -1.0,  0.0],
        [1.0,  0.0, -1.0, -1.0,  0.0, -1.0],
        [0.0,  1.0,  0.0,  0.0, -1.0,  0.0],
    ])

    # count occupied neighboring pairs for interaction
    expected_interaction_matrix = np.array([
        [1.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  1.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  1.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  0.0,  0.0,  1.0],
    ])

    full_expected_matrix = expected_hopping_matrix * t + expected_interaction_matrix * U
    hopping_matrix = hamiltonian.hopping_matrix
    interaction_matrix = hamiltonian.interaction_matrix

    # hermitian check
    assert_allclose(hopping_matrix.toarray(), hopping_matrix.getH().toarray(), rtol=0, atol=atol)
    assert_allclose(interaction_matrix.toarray(), interaction_matrix.getH().toarray(), rtol=0, atol=atol)
    assert_allclose(H.toarray(), H.getH().toarray(), rtol=0, atol=atol)

    # compare constructed matrices with expected ones
    assert_allclose(hopping_matrix.toarray(), expected_hopping_matrix, rtol=0, atol=atol)
    assert_allclose(interaction_matrix.toarray(), expected_interaction_matrix, rtol=0, atol=atol)
    assert_allclose(H.toarray(), full_expected_matrix, rtol=0, atol=atol)

@pytest.mark.parametrize("t", [-3, 1, 2])
@pytest.mark.parametrize("U", [-4, -1, 3])
def test_hamiltonian_construction_obc_without_light_matter(t: float, U: float, atol: float = 1e-14):
    basis = Basis(4, 2)  # 4 sites, 2 particles
    hamiltonian = Hamiltonian(basis, g=0, boundary_conditions="open")
    H = hamiltonian.construct_hamiltonian_matrix(t, U, omega=0)

    # without t and U specified
    expected_hopping_matrix = np.array([
        [ 0.0, -1.0,  0.0,  0.0,  0.0,  0.0],
        [-1.0,  0.0, -1.0, -1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0,  0.0, -1.0,  0.0],
        [ 0.0, -1.0,  0.0,  0.0, -1.0,  0.0],
        [ 0.0,  0.0, -1.0, -1.0,  0.0, -1.0],
        [ 0.0,  0.0,  0.0,  0.0, -1.0,  0.0],
    ])

    # count occupied neighboring pairs for interaction
    expected_interaction_matrix = np.array([
        [1.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  1.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  0.0,  0.0,  1.0],
    ])

    full_expected_matrix = t * expected_hopping_matrix + U * expected_interaction_matrix
    hopping_matrix = hamiltonian.hopping_matrix
    interaction_matrix = hamiltonian.interaction_matrix

    # hermitian check
    assert_allclose(hopping_matrix.toarray(), hopping_matrix.getH().toarray(), rtol=0, atol=atol)
    assert_allclose(interaction_matrix.toarray(), interaction_matrix.getH().toarray(), rtol=0, atol=atol)
    assert_allclose(H.toarray(), H.getH().toarray(), rtol=0, atol=atol)

    # compare constructed matrices with expected ones
    assert_allclose(hopping_matrix.toarray(), expected_hopping_matrix, rtol=0, atol=atol)
    assert_allclose(interaction_matrix.toarray(), expected_interaction_matrix, rtol=0, atol=atol)
    assert_allclose(H.toarray(), full_expected_matrix, rtol=0, atol=atol)



@pytest.mark.parametrize("t", [-3, 2])
@pytest.mark.parametrize("U", [1])
@pytest.mark.parametrize("g", [0.5,  np.pi/4, np.pi/2])
@pytest.mark.parametrize("omega", [1, 2])
def test_hamiltonian_construction_obc_with_light_matter(t: float, U: float, g: float, omega: float, atol: float = 1e-14):
    basis = Basis(2, 1, 1)  # 2 sites, 1 particle, max 1 photon
    hamiltonian = Hamiltonian(basis, g=g, boundary_conditions="open")
    H = hamiltonian.construct_hamiltonian_matrix(t, U, omega)

    # expected matrices from tablet notes
    expected_hopping_matrix = -1 * np.array([
            [0, 0, np.cos(g), 1j*np.sin(g)],
            [0, 0, 1j*np.sin(g), np.cos(g)],
            [np.cos(g), -1j*np.sin(g), 0, 0],
            [-1j*np.sin(g), np.cos(g), 0, 0]
    ])

    # count number of photons in each state
    expected_photon_energy_matrix = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ])
    # no interaction term for 1 particle
    full_expected_matrix = t * expected_hopping_matrix + omega * expected_photon_energy_matrix
    hopping_matrix = hamiltonian.hopping_matrix
    photon_energy_matrix = hamiltonian.photon_energy_matrix

    # hermitian check
    assert_allclose(hopping_matrix.toarray(), hopping_matrix.getH().toarray(), rtol=0, atol=atol)
    assert_allclose(photon_energy_matrix.toarray(), photon_energy_matrix.getH().toarray(), rtol=0, atol=atol)
    assert_allclose(H.toarray(), H.getH().toarray(), rtol=0, atol=atol)

    # compare constructed matrices with expected ones
    assert_allclose(hopping_matrix.toarray(), expected_hopping_matrix, rtol=0, atol=atol)
    assert_allclose(photon_energy_matrix.toarray(), expected_photon_energy_matrix, rtol=0, atol=atol)
    assert_allclose(H.toarray(), full_expected_matrix, rtol=0, atol=atol)

def expected_peierls_matrix(g: float, N_ph: int) -> np.ndarray:
    """ 
    Helper function to compute expected Peierls phase matrix for small N_ph.
    See tablet notes for computation of exp(ig(a + a^+)) matrices.
    Arguments:
        g (float): Coupling strength.
        N_ph (int): Maximum photon number.
    Returns:
        np.ndarray: The expected Peierls phase matrix.
    """
    if N_ph == 1:
        return np.array([
            [np.cos(g), 1j*np.sin(g)],
            [1j*np.sin(g), np.cos(g)]
        ])

    if N_ph == 2:
        s3 = np.sqrt(3)
        return (1/3)*np.array([
            [np.cos(s3*g)+2,  1j*s3*np.sin(s3*g),    np.sqrt(2)*(np.cos(s3*g)-1)],
            [1j*s3*np.sin(s3*g),  3*np.cos(s3*g),    1j*np.sqrt(6)*np.sin(s3*g)],
            [np.sqrt(2)*(np.cos(s3*g)-1), 1j*np.sqrt(6)*np.sin(s3*g), 2*np.cos(s3*g) + 1]
        ])

    raise ValueError("Unsupported photon number")


@pytest.mark.parametrize("N_ph", [1, 2])
@pytest.mark.parametrize("g", [0, 1, np.pi/2, 2, np.pi, 4, 3*np.pi/2])
def test_construct_peierls_phase_matrix(N_ph: int, g: float, atol: float = 1e-14):
    # max  N_ph photons (sites and particles don't matter here)
    basis = Basis(1, 0, N_ph)
    H = Hamiltonian(basis, g=g, boundary_conditions="open")
    P = H.construct_peierls_phase_matrix()

    # compare with expected matrix
    expected = expected_peierls_matrix(g, N_ph)
    assert_allclose(P.toarray(), expected, rtol=0, atol=atol)

    # sanity: unitarity of exp(ig(a + a^+))
    identity = np.eye(N_ph+1)
    product = P @ P.getH()
    assert_allclose(product.toarray(), identity, rtol=0, atol=atol)
