class Basis:

    def __init__(self, L: int, N: int):
        self.L = L
        self.N = N
        self.states = self._generate_basis()
        # set up index map for fast lookup (Lin table)
        self.index_map = {state: idx for idx, state in enumerate(self.states)}

    def _generate_basis(self) -> list[int]:
        """
        Generate all possible states for the given L and N, i.e. all bit strings of length L with N ones.
        Returns:
            list[int]: List of integers representing the basis states.
        """
        # 1 << L = 2**L but much faster
        states = [bits for bits in range(1 << self.L) if bits.bit_count() == self.N]
        return states


    def state_index(self, state: int) -> int:
        """Return index of a given basis state."""
        return self.index_map[state]

    # such that "len(basis)" returns number of basis states
    def __len__(self):
        return len(self.states)

    # such that "for state in basis" works
    def __iter__(self):
        return iter(self.states)