
from scipy.sparse import lil_matrix, csr_array

from src.exact_diagonalization.basis import Basis
from src.exact_diagonalization.operators import count_pairs, flip_bit

class Hamiltonian:
    

    def __init__(self, basis: Basis, t: float, U: float, boundary_conditions: str = "periodic"):
        self.basis = basis
        self.t = t
        self.U = U
        self.boundary_conditions = boundary_conditions
        self.periodic = boundary_conditions == "periodic"
        
        self.matrix = self._construct_hamiltonian()


    def _construct_hamiltonian(self) -> csr_array:
        """
        Construct the Hamiltonian matrix for the given basis, hopping term t, and interaction term U.
        Returns:
            csr_array: The Hamiltonian matrix as a sparse CSR matrix.
        """
        L = self.basis.L
        N = self.basis.N
        dim = len(self.basis)
        H = lil_matrix((dim, dim), dtype=float)

        for k, state in enumerate(self.basis):
            # Diagonal terms: interaction energy
            num_pairs = count_pairs(state, 1, L, self.boundary_conditions)
            H[k, k] += self.U * num_pairs

            # Off-diagonal terms: hopping
            for site in range(L - 1 + int(self.periodic)):
                next_site = (site + 1) % L 

                # Check if we can hop from site to next_site (site occupied, next_site empty)
                if ((state >> site) & 1) == 1 and ((state >> next_site) & 1) == 0:
                    new_state = state
                    # Remove particle from current site and add to next site
                    new_state = flip_bit(new_state, site)       
                    new_state = flip_bit(new_state, next_site)

                    # Only sign change due to hopping over boundary if periodic
                    # see notes on tablet for proof
                    sign = (-1)**(N - 1) if next_site < site else 1
                    k_prime = self.basis.state_index(new_state)
                    H[k, k_prime] += -self.t * sign
                    H[k_prime, k] += -self.t * sign # Hermitian conjugate
        # convert to csr for efficient diagonalization
        return H.tocsr()