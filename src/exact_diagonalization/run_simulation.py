import numpy as np

# put this in a config file later
ROOT_DATA_DIR = "data/xxz_cavity/exact_diagonalization/"

def sweep_photon_number_vs_omega(self, omega_list: np.ndarray, t: float, U: float) -> np.ndarray:
    photon_numbers = []
    for omega in omega_list:
        H = self.base_hamiltonian.build_hamiltonian_matrix(t=t, U=U, omega=omega)
        evals, evecs = self.diagonalize(H, k=1) # only interested in ground state
        gs = evecs[:, 0]
        n_photon = self.photon_number(gs)
        photon_numbers.append(n_photon)
    return np.array(photon_numbers)

def sweep_photon_number_vs_U(self, U_list: np.ndarray, t: float, omega: float) -> np.ndarray:
    photon_numbers = []
    for U in U_list:
        H = self.base_hamiltonian.build_hamiltonian_matrix(t=t, U=U, omega=omega)
        evals, evecs = self.diagonalize(H, k=1) # only interested in ground state
        gs = evecs[:, 0]
        n_photon = self.photon_number(gs)
        photon_numbers.append(n_photon)
    return np.array(photon_numbers)

def sweep_entanglement_entropy_vs_U(self, U_list: np.ndarray, t: float, omega: float) -> np.ndarray:
    entropies = []
    for U in U_list:
        H = self.base_hamiltonian.build_hamiltonian_matrix(t=t, U=U, omega=omega)
        evals, evecs = self.diagonalize(H, k=1) # only interested in ground state
        gs = evecs[:, 0]
        S = self.entanglement_entropy_fermions_photons(gs)
        entropies.append(S)
    return np.array(entropies)

def sweep_entanglement_entropy_vs_omega(self, omega_list: np.ndarray, t: float, U: float) -> np.ndarray:
    entropies = []
    for omega in omega_list:
        H = self.base_hamiltonian.build_hamiltonian_matrix(t=t, U=U, omega=omega)
        evals, evecs = self.diagonalize(H, k=1) # only interested in ground state
        gs = evecs[:, 0]
        S = self.entanglement_entropy_fermions_photons(gs)
        entropies.append(S)
    return np.array(entropies)

def sweep_ground_state_vs_U(self, U_list: np.ndarray, t: float, omega: float) -> np.ndarray:
    ground_states = []
    for U in U_list:
        H = self.base_hamiltonian.build_hamiltonian_matrix(t=t, U=U, omega=omega)
        evals, evecs = self.diagonalize(H, k=1) # only interested in ground state
        gs = evecs[:, 0]
        ground_states.append(gs)
    return np.array(ground_states)


if __name__ == "__main__":
    L = 12
    N_f = L // 2 # half-filling
    N_ph = 10 
    t = 1.0
    U = 3.0 * t
    omega = 10.0 * t
    # omega = 0.0 * t
    # g = 0.0
    g = 0.5
    
    omega_list = np.linspace(1, 100 + 1, 100) * t
    #entropies_omega = analyzer.sweep_entanglement_entropy_vs_omega(omega_list, t=t, U=U)
    # plot_entanglement_entropy_vs_omega_log_log(omega_list, entropies_omega)
    # photon_numbers_omega = analyzer.sweep_photon_number_vs_omega(omega_list, t=t, U=U)
    # # plot_photon_number_vs_omega(omega_list, photon_numbers)
    # plot_photon_number_vs_omega_log_log(omega_list, photon_numbers_omega)
    
    U_list = np.linspace(0, 20, 100) * t
    # entropies_U = analyzer.sweep_entanglement_entropy_vs_U(U_list, t=t, omega=omega)
    # plot_entanglement_entropy_vs_U(U_list, entropies_U)
    # photon_numbers_U = analyzer.sweep_photon_number_vs_U(U_list, t=t, omega=omega)
    # plot_photon_number_vs_U(U_list, photon_numbers_U)

    U_list = np.linspace(0, 3, 20) * t
    # ground_states = analyzer.sweep_ground_state_vs_U(U_list, t=t, omega=omega)
    # plot_ground_state_amplitudes_vs_U(U_list, ground_states)