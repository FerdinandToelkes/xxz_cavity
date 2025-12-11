import numpy as np


from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from src.exact_diagonalization.basis import Basis
from src.exact_diagonalization.operators import build_photon_number_matrix, build_longest_range_fermion_number_matrix
from src.exact_diagonalization.utils import is_hermitian

class Analyzer:
    def __init__(self, H: csr_matrix, basis: Basis, boundary_conditions: str="periodic"):            
        self.H = H
        self.basis = basis 
        self.dim_el = len(basis.fermion_states)
        self.dim_ph = len(basis.photon_states)
        self.boundary_conditions = boundary_conditions
        self.evals = None
        self.evecs = None

        # TODO: check if it makes sense to precompute these here
        self.photon_number_matrix = build_photon_number_matrix(self.basis)
        self.longest_range_fermion_number_matrix = build_longest_range_fermion_number_matrix(self.basis, self.boundary_conditions)

    def diagonalize(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the Hamiltonian matrix H and return the lowest k eigenvalues and eigenvectors.
        Please be sure that H is not already diagonalized when calling this method since
        the Lanczos algorithm used here cannot deal nicely with already diagonal matrices.
        Arguments:
            H (csr_matrix): The Hamiltonian matrix to diagonalize.
            k (int): The number of lowest eigenvalues and eigenvectors to compute.
        Returns:
            evals (np.ndarray): The lowest k eigenvalues.
            evecs (np.ndarray): The corresponding eigenvectors.
        Raises:
            ValueError: If H is not Hermitian.
        """
        # ensure that matrix is Hermitian
        if not is_hermitian(self.H):
            raise ValueError("Hamiltonian matrix H must be Hermitian.")
        # get smallest algebraic eigenvalues, i.e. those with lowest real part
        self.evals, self.evecs = eigsh(self.H, k=k, which='SA')
        return self.evals, self.evecs

    
    def ground_state(self) -> np.ndarray:
        if self.evecs is None:
            self.evals, self.evecs = self.diagonalize(k=1)
        return self.evecs[:, 0] # type: ignore

    def build_psi_matrix(self, psi: np.ndarray) -> np.ndarray:
        """
        Reshape the full state vector psi into a matrix with dimensions (dim_fermions, dim_photons).
        We assume that the basis is ordered lexicographically such that the first dim_fermions entries
        correspond to the first photon state, the next dim_fermions to the second photon state, and so on.
        Arguments:
            psi (np.ndarray): The state vector of the full system.
        Returns:
            np.ndarray: The reshaped state matrix.
        """
        psi_matrix = psi.reshape((self.dim_el, self.dim_ph))
        return psi_matrix

    def entanglement_entropy_fermions_photons(self, psi: np.ndarray) -> float:
        """
        Compute the Von Neumann entanglement entropy between fermionic and photonic 
        subsystems given a state vector psi.
        Arguments:
            psi (np.ndarray): The state vector of the full system.
        Returns:
            float: The entanglement entropy S.
        """
        # build psi_matrix and perform SVD
        psi_matrix = self.build_psi_matrix(psi)
        S = np.linalg.svd(psi_matrix, compute_uv=False)
        
        # compute entanglement entropy
        S_squared = S**2
        nonzero = S_squared[S_squared > 0] # avoid log(0)

        # check that the sum of squared singular values is 1 (as for every density matrix)
        norm = np.sum(S_squared)
        if not np.isclose(norm, 1.0):
            raise ValueError(f"Sum of squared singular values is not 1 but {norm}. Ensure that psi is normalized.")

        entropy = -np.sum(nonzero * np.log(nonzero))
        
        return entropy
    
    def probability_distribution_photon_number(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute the probability distribution of photon numbers for a given state vector psi.
        Note: Make use of our ordering of the basis states to efficiently compute this, where
        the psi matrix has dimensions (dim_fermions, dim_photons) and contains all amplitudes 
        for a given photon number in one column.
        Arguments:
            psi (np.ndarray): The state vector of the full system.
        Raises:
            ValueError: If the total probability does not sum to 1.
        Returns:
            np.ndarray: The probability distribution over photon numbers.
        """
        psi_matrix = self.build_psi_matrix(psi)
        prob_distribution = np.sum(np.abs(psi_matrix)**2, axis=0)

        # check that the probability distribution sums to 1
        total_prob = np.sum(prob_distribution)
        if not np.isclose(total_prob, 1.0):
            raise ValueError(f"Total probability is not 1 but {total_prob}. Ensure that psi is normalized.")
        return prob_distribution

    def expectation_value(self, psi: np.typing.NDArray[np.complex128], operator: csr_matrix | np.ndarray) -> float:
        """
        Compute the expectation value of the photon number operator for a given state vector psi.
        Arguments:
            psi (np.ndarray): The state vector of the full system.
            operator (np.ndarray): The operator whose expectation value is to be computed.
        Returns:
            float: The expectation value of the photon number.
        """
        expectation_value = psi.conj() @ (operator @ psi)
        if not np.isclose(expectation_value.imag, 0.0):
            raise ValueError("Expectation value has a significant imaginary part for photon number.")
        return float(expectation_value.real)
    
    
    # def photon_number(self, psi: np.typing.NDArray[np.complex128]) -> float:
    #     """
    #     Compute the expectation value of the photon number operator for a given state vector psi.
    #     Arguments:
    #         psi (np.ndarray): The state vector of the full system.
    #     Returns:
    #         float: The expectation value of the photon number.
    #     """
    #     a_dag_a = self.photon_number_matrix  # precomputed operator
    #     expectation_value = psi.conj() @ (a_dag_a @ psi)
    #     if not np.isclose(expectation_value.imag, 0.0):
    #         raise ValueError("Expectation value has a significant imaginary part for photon number.")
    #     return float(expectation_value.real)


# if __name__ == "__main__":
#     L = 12
#     N_f = L // 2 # half-filling
#     N_ph = 10 
#     t = 1.0
#     U = 3.0 * t
#     omega = 10.0 * t
#     # omega = 0.0 * t
#     # g = 0.0
#     g = 0.5

#     basis = Basis(L, N_f, N_ph)  
#     builder = HamiltonianBuilder(basis, g=g, boundary_conditions="periodic")
#     H = builder.build_hamiltonian_matrix(t=t, U=U, omega=omega)
#     analyzer = Analyzer(H, basis) 
#     evals, evecs = analyzer.diagonalize(k=5)
#     gs = analyzer.ground_state()
#     n_photon = analyzer.photon_number(gs)
    