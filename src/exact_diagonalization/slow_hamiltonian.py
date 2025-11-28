
from scipy.sparse import lil_matrix, csr_matrix

from src.exact_diagonalization.basis import Basis
from src.exact_diagonalization.operators import count_pairs, flip_bit
from src.exact_diagonalization.utils import is_hermitian

# TODO: Try constructing full Hamiltonian via tensor products of photonic and fermionic parts and compare performance
class Hamiltonian:
    """
    Class to construct the Hamiltonian matrix for a given basis, hopping term t, and interaction term U.
    Attributes:
        basis (Basis): The basis object containing the states.
        boundary_conditions (str): The type of boundary conditions ("periodic" or "open").
    """

    def __init__(self, basis: Basis, boundary_conditions: str = "periodic"):
        self.basis = basis
        self.boundary_conditions = boundary_conditions
        self.periodic = boundary_conditions == "periodic"
        
        self.hopping_matrix = self.construct_hopping_matrix()
        self.interaction_matrix = self.construct_interaction_matrix()
        self.photon_energy_matrix = self.construct_photon_energy_matrix()
        assert is_hermitian(self.hopping_matrix), "Hopping matrix is not Hermitian!"
        assert is_hermitian(self.interaction_matrix), "Interaction matrix is not Hermitian!"
        assert is_hermitian(self.photon_energy_matrix), "Photon energy matrix is not Hermitian!"
        

    def construct_interaction_matrix(self) -> csr_matrix:
        """
        Construct the interaction part of the Hamiltonian matrix for the given basis.
        Returns:
            csr_matrix: The interaction matrix as a sparse CSR matrix.
        """
        L = self.basis.L
        dim = len(self.basis)
        H = lil_matrix((dim, dim), dtype=float)

        for k, full_state in enumerate(self.basis):
            f_state = full_state[0] # fermionic part
            # Diagonal terms: interaction energy
            num_pairs = count_pairs(f_state, 1, L, self.boundary_conditions)
            H[k, k] += num_pairs # multiply with U later

        # convert to csr for efficient diagonalization etc.
        return csr_matrix(H)
    
    def construct_photon_energy_matrix(self) -> csr_matrix:
        """
        Construct the interaction part of the Hamiltonian matrix for the given basis.
        Returns:
            csr_matrix: The interaction matrix as a sparse CSR matrix.
        """
        dim = len(self.basis)
        H = lil_matrix((dim, dim), dtype=float)

        for k, full_state in enumerate(self.basis):
            ph_state = full_state[1] # photon part
            # Diagonal terms: photon energy
            H[k, k] += ph_state # multiply with omega later

        # convert to csr for efficient diagonalization etc.
        return csr_matrix(H)
    
    def construct_hopping_matrix(self) -> csr_matrix:
        """
        Construct the hopping part of the Hamiltonian matrix for the given basis.
        Returns:
            csr_matrix: The hopping matrix as a sparse CSR matrix.
        """
        L = self.basis.L
        N_f = self.basis.N_f
        dim = len(self.basis)
        H = lil_matrix((dim, dim), dtype=float)

        # Off-diagonal terms: hopping
        for k, full_state in enumerate(self.basis):
            f_state = full_state[0]  # fermionic part
            p_state = full_state[1]  # photon part
            
            for site in range(L - 1 + int(self.periodic)):
                next_site = (site + 1) % L 

                # Check if we can hop from site to next_site (site occupied, next_site empty)
                if ((f_state >> site) & 1) == 1 and ((f_state >> next_site) & 1) == 0:
                    new_f_state = f_state
                    # Remove particle from current site and add to next site
                    new_f_state = flip_bit(new_f_state, site)       
                    new_f_state = flip_bit(new_f_state, next_site)
                    new_state = (new_f_state, p_state)

                    # Only sign change due to hopping over boundary if periodic
                    # see notes on tablet for proof
                    sign = (-1)**(N_f - 1) if next_site < site else 1
                    k_prime = self.basis.state_index(new_state)
                    H[k, k_prime] += (-1) * sign # multiply with t later
                    # also take Hermitian conjugate into account since we only consider hopping in one direction
                    H[k_prime, k] += (-1) * sign 
        # convert to csr for efficient diagonalization etc.
        return csr_matrix(H)

    def construct_hamiltonian_matrix(self, t: float, U: float, omega: float) -> csr_matrix:
        """
        Construct the Hamiltonian matrix for the given basis, hopping term t, and interaction term U from hopping and interaction matrices.
        Returns:
            csr_matrix: The Hamiltonian matrix as a sparse CSR matrix.
        """
        H_hopping = self.construct_hopping_matrix()
        H_interaction = self.construct_interaction_matrix()
        if self.basis.N_ph == 0 or omega == 0:
            H = t * H_hopping + U * H_interaction
        else:
            H_photon = self.construct_photon_energy_matrix()
            H = t * H_hopping + U * H_interaction + omega * H_photon
        return H
    
