import numpy as np
from scipy.sparse import identity, diags, kron, lil_matrix, csr_matrix
from scipy.linalg import eigh_tridiagonal


from src.exact_diagonalization.basis import Basis
from src.exact_diagonalization.operators import count_pairs, flip_bit
from src.exact_diagonalization.utils import is_hermitian

class Hamiltonian:
    """
    Class to construct the Hamiltonian matrix for a given basis, hopping term t, and interaction term U.
    Attributes:
        basis (Basis): The basis object containing the states.
        g (float): The light-matter coupling strength.
        boundary_conditions (str): The type of boundary conditions ("periodic" or "open").
    """

    def __init__(self, basis: Basis, g: float = 0, boundary_conditions: str = "periodic"):
        if basis.L <= 2 and boundary_conditions == "periodic":
            raise ValueError("Periodic boundary conditions are not well-defined for L <= 2.")
        self.basis = basis
        self.g = g
        self.boundary_conditions = boundary_conditions
        self.periodic = boundary_conditions == "periodic"
        self.dim_el = len(self.basis.fermion_states)
        self.dim_ph = len(self.basis.photon_states)
        
        self.hopping_matrix = self.construct_hopping_matrix()
        self.interaction_matrix = self.construct_interaction_matrix()
        self.photon_energy_matrix = self.construct_photon_energy_matrix()
        assert is_hermitian(self.hopping_matrix), "Hopping matrix is not Hermitian!"
        assert is_hermitian(self.interaction_matrix), "Interaction matrix is not Hermitian!"
        assert is_hermitian(self.photon_energy_matrix), "Photon energy matrix is not Hermitian!"
        

    def construct_interaction_matrix(self) -> csr_matrix:
        """
        Construct the interaction part of the Hamiltonian matrix for the given basis.
        Note that this implementation uses Kronecker products to build the full Hamiltonian
        taking advantage of the separability of the photonic and fermionic parts.
        Returns:
            csr_matrix: The interaction matrix as a sparse CSR matrix.
        """
        L = self.basis.L

        # Build a diagonal array of interaction strengths
        diag_vals = np.zeros(self.dim_el)
        for k, state in enumerate(self.basis.fermion_states):
            diag_vals[k] = count_pairs(state, 1, L, self.boundary_conditions)

        # electronic part diagonal, photonic = identity
        H_el = diags(diag_vals, format='csr')
        H_ph = identity(self.dim_ph, format='csr')
        H = kron(H_el, H_ph, format='csr')
        # convert to csr for efficient diagonalization etc.
        return csr_matrix(H)

      
    def construct_photon_energy_matrix(self) -> csr_matrix:
        """
        Construct the interaction part of the Hamiltonian matrix for the given basis.
        Note that this implementation uses Kronecker products to build the full Hamiltonian
        taking advantage of the separability of the photonic and fermionic parts.
        Returns:
            csr_matrix: The interaction matrix as a sparse CSR matrix.
        """
        # Build a diagonal array of photon energies
        diag_vals = np.zeros(self.dim_ph)
        for state in self.basis.photon_states:
            diag_vals[state] += state # multiply with omega later

        # photonic part diagonal, electronic = identity
        H_ph = diags(diag_vals, format='csr')
        H_el = identity(self.dim_el, format='csr')
        H = kron(H_el, H_ph, format='csr')
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
        dim_el = self.dim_el
        H_el_site_to_next_site = lil_matrix((dim_el, dim_el), dtype=float)

        # note that we only construct hopping from site to next_site, the Hermitian conjugate is added later
        for k, state in enumerate(self.basis.fermion_states):
            for site in range(L - 1 + int(self.periodic)):
                next_site = (site + 1) % L 

                # Check if we can hop from site to next_site (site occupied, next_site empty)
                if ((state >> site) & 1) == 1 and ((state >> next_site) & 1) == 0:
                    # Remove particle from current site and add to next site
                    new_state = state
                    new_state = flip_bit(new_state, site)       
                    new_state = flip_bit(new_state, next_site)
                    # Only sign change due to hopping over boundary if periodic
                    # see notes on tablet for proof
                    sign = (-1)**(N_f - 1) if next_site < site else 1
                    k_prime = self.basis.fermion_state_index(new_state)
                    H_el_site_to_next_site[k, k_prime] += (-1) * sign # multiply with t later
                    

        # construct full hopping matrix via Kronecker product with photonic part
        H_ph = self.construct_peierls_phase_matrix()
        H_site_to_next_site = kron(H_el_site_to_next_site, H_ph, format='csr')
        H_site_to_next_site = csr_matrix(H_site_to_next_site)
        # add Hermitian conjugate (hopping from next_site to site)
        H = H_site_to_next_site + H_site_to_next_site.getH()
        return H
    
    def construct_peierls_phase_matrix(self) -> csr_matrix:
        """
        Construct the Peierls phase part exp(ig(a + a^+)) of the Hamiltonian matrix for the given photonic basis.
        Note that we obtain exp(-ig(a + a^+)) as the Hermitian conjugate.

        Returns:
            csr_matrix: The Peierls phase matrix as a sparse CSR matrix.
        """
        # construct matrix for photonic part only
        diags = np.zeros(self.dim_ph)
        off_diag = np.sqrt(np.arange(1, self.dim_ph)) # i.e. from 1 to N_ph-1
        
        # diagonalize a + a^\dagger =: A which is tridiagonal in the number basis
        eigenvals, eigenvecs = eigh_tridiagonal(diags, off_diag)

        # write A = V D V^\dagger with D diagonal matrix of eigenvals and V matrix of eigenvecs
        phases = np.exp(1j * self.g * eigenvals)
        U = eigenvecs @ np.diag(phases) @ eigenvecs.conj().T

        return csr_matrix(U)


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
    


if __name__ == "__main__":
    from src.exact_diagonalization.slow_hamiltonian import Hamiltonian as BaseHamiltonian

    # simple test
    L = 16
    N_f = L // 2
    N_ph = 10
    g = np.pi / 2
    basis = Basis(L, N_f, N_ph)
    bc = "open"
    hamiltonian = Hamiltonian(basis, g, boundary_conditions=bc)
   
    
    # compare construction times for fancy and normal interaction matrix
    import time
    start = time.time()
    H_fancy = hamiltonian.construct_hopping_matrix()
    end = time.time()

    print(f"Fancy interaction matrix construction time: {end - start:.6f} seconds")
    print(f"Shape of fancy interaction matrix: {H_fancy.shape}")
    # H_fancy = H_fancy.todense()
    # # print with only 2 decimal places
    # np.set_printoptions(precision=2, suppress=True)
    # print(f"H_fancy:\n{H_fancy}")
    
    base_hamiltonian = BaseHamiltonian(basis, boundary_conditions=bc)
    start = time.time()
    H_normal = base_hamiltonian.construct_hopping_matrix()
    end = time.time()
    print(f"Normal interaction matrix construction time: {end - start:.6f} seconds")
    print(f"Shape of normal interaction matrix: {H_normal.shape}")
    exit()
    # convert to dense for printing
    # H_fancy = H_fancy.todense()
    # H_normal = H_normal.todense()
    # print(f"fancy:\n{H_fancy}\nnormal:\n{H_normal}")
   
    # check if sparse matrices are equal
    diff = H_fancy - H_normal
    assert diff.nnz == 0, f"Matrices are not equal! Number of differing elements: {diff.nnz}"



    