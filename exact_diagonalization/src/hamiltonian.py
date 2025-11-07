from src.basis import Basis
from src.operators import count_pairs, flip_bit

class Hamiltonian:
    

    def __init__(self, basis: Basis, t: float, U: float, boundary_condition: str = "periodic"):
        self.basis = basis
        self.t = t
        self.U = U
        self.boundary_condition = boundary_condition
        self.size = len(basis)
        self.matrix = self._construct_hamiltonian()


    def _construct_hamiltonian(self) -> list[list[float]]:
        """
        Construct the Hamiltonian matrix for the given basis, hopping term t, and interaction term U.
        Returns:
            list[list[float]]: The Hamiltonian matrix as a 2D list.
        """
        L = self.basis.L
        H = [[0.0 for _ in range(self.size)] for _ in range(self.size)]

        for i, state in enumerate(self.basis):
            # Diagonal terms: interaction energy
            num_pairs = count_pairs(state, 1, L, self.boundary_condition)
            H[i][i] += self.U * num_pairs

            # Off-diagonal terms: hopping
            for site in range(L):
                next_site = (site + 1) % L if self.boundary_condition == "periodic" else site + 1
                if next_site >= L:
                    continue  # Skip if out of bounds for open boundary

                # Check if we can hop from site to next_site
                if ((state >> site) & 1) == 1 and ((state >> next_site) & 1) == 0:
                    new_state = state
                    new_state = flip_bit(new_state, site)       # Remove particle from current site
                    new_state = flip_bit(new_state, next_site)  # Add particle to next site

                    j = self.basis.state_index(new_state)
                    H[i][j] += -self.t
                    H[j][i] += -self.t  # Hermitian conjugate

        return H