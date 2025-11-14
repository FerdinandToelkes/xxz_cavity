class Basis:
    """
    Class to generate and manage the basis states for a system with L sites, N particles, and a maximum photon number n_max.
    """
    def __init__(self, L: int, N: int, n_max: int = 0):
        """
        Initialize the Basis object.
        Args:
            L (int): Number of lattice sites.
            N (int): Number of particles.
            n_max (int): Maximum photon number (default is 0 for fermionic basis).
        """
        self.L = L
        self.N = N
        self.n_max = n_max
        self.states = self._generate_basis()
        # set up index map for fast lookup (Lin table)
        self.index_map = {state: idx for idx, state in enumerate(self.states)}

    def _generate_basis(self) -> list[tuple[int, int]]:
        """
        Generate all possible states for the given L, N and n_max, i.e. all bit strings of length L with N ones combined with photon numbers from 0 to n_max.
        Returns:
            list[tuple[int, int]]: List of tuples representing the basis states.
        """
        # 1 << L = 2**L but much faster
        fermion_states = [bits for bits in range(1 << self.L) if bits.bit_count() == self.N]
        states = [(f, p) for f in fermion_states for p in range(self.n_max + 1)]
        return states


    def state_index(self, state: tuple[int, int]) -> int:
        """Return index of a given basis state."""
        return self.index_map[state]

    # such that "len(basis)" returns number of basis states
    def __len__(self):
        return len(self.states)

    # such that "for state in basis" works
    def __iter__(self):
        return iter(self.states)