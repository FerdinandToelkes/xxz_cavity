class Basis:
    """
    Class to generate and manage the basis states for a system with L sites, N particles, and a maximum photon number n_max.
    """
    def __init__(self, L: int, N_f: int, N_ph: int = 0):
        """
        Initialize the Basis object.
        Argugments:
            L (int): Number of lattice sites.
            N_f (int): Number of fermions.
            N_ph (int): Maximum photon number (default is 0 for fermionic basis).
        """
        self.L = L
        self.N_f = N_f
        self.N_ph = N_ph
        self.fermion_states = self._generate_fermion_basis()
        self.photon_states = self._generate_photonic_basis()
        self.states = self._generate_basis()
        # set up index map for fast lookup (Lin table)
        self.index_map = {state: idx for idx, state in enumerate(self.states)}
        self.fermion_index_map = {state: idx for idx, state in enumerate(self.fermion_states)}
        self.photon_index_map = {state: idx for idx, state in enumerate(self.photon_states)}

    

    def _generate_basis(self) -> list[tuple[int, int]]:
        """
        Generate all possible states for the given L, N_f and N_ph, i.e. all bit strings of length L with N ones combined with photon numbers from 0 to N_ph.
        Returns:
            list[tuple[int, int]]: List of tuples representing the basis states.
        """
        # 1 << L = 2**L but much faster
        fermion_states = [bits for bits in range(1 << self.L) if bits.bit_count() == self.N_f]
        states = [(f, p) for f in fermion_states for p in range(self.N_ph + 1)]
        return states
    
    def _generate_fermion_basis(self) -> list[int]:
        """
        Generate all possible fermionic states for the given L and N_f, i.e. all bit strings of length L with N_f ones.
        Returns:
            list[int]: List of integers representing the fermionic basis states.
        """
        # 1 << L = 2**L but much faster
        states = [bits for bits in range(1 << self.L) if bits.bit_count() == self.N_f]
        return states
    
    def _generate_photonic_basis(self) -> list[int]:
        """
        Generate all possible photonic states for the given N_ph, i.e.photon numbers from 0 to N_ph.
        Returns:
            list[int]: List of integers representing the photonic basis states.
        """
        states = [p for p in range(self.N_ph + 1)]
        return states


    def state_index(self, state: tuple[int, int]) -> int:
        """Return index of a given basis state."""
        return self.index_map[state]
    
    def fermion_state_index(self, state: int) -> int:
        """Return index of a given fermionic basis state."""
        return self.fermion_index_map[state]
    
    def photon_state_index(self, state: int) -> int:
        """Return index of a given photonic basis state."""
        return self.photon_index_map[state]
    
    # such that "len(basis)" returns number of basis states
    def __len__(self):
        return len(self.states)

    # such that "for state in basis" works
    def __iter__(self):
        return iter(self.states)