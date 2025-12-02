import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from src.exact_diagonalization.basis import Basis
from src.exact_diagonalization.operators import construct_photon_number_matrix
from src.exact_diagonalization.hamiltonian import Hamiltonian

class Analyzer:
    def __init__(self, basis: Basis, g: float = 0, boundary_conditions: str = "periodic"):
        self.base_hamiltonian = Hamiltonian(basis, g, boundary_conditions)              
        self.basis = basis  

        self.photon_number_matrix = construct_photon_number_matrix(self.basis)

    def diagonalize(self, H: csr_matrix, k: int):
        # do sparse lanczos if large, dense otherwise
        self.evals, self.evecs = eigsh(H, k=k, which='SA')
        return self.evals, self.evecs

    
    def photon_number(self, psi):
        a_dag_a = self.photon_number_matrix  # precomputed operator
        return psi.conj() @ (a_dag_a @ psi)
    
    def sweep_photon_number_vs_omega(self, omega_list: np.ndarray, t: float, U: float) -> np.ndarray:
        photon_numbers = []
        for omega in omega_list:
            H = self.base_hamiltonian.construct_hamiltonian_matrix(t=t, U=U, omega=omega)
            evals, evecs = self.diagonalize(H, k=1) # only interested in ground state
            gs = evecs[:, 0]
            n_photon = self.photon_number(gs)
            photon_numbers.append(n_photon)
        return np.array(photon_numbers)
    
    def sweep_photon_number_vs_U(self, U_list: np.ndarray, t: float, omega: float) -> np.ndarray:
        photon_numbers = []
        for U in U_list:
            H = self.base_hamiltonian.construct_hamiltonian_matrix(t=t, U=U, omega=omega)
            evals, evecs = self.diagonalize(H, k=1) # only interested in ground state
            gs = evecs[:, 0]
            n_photon = self.photon_number(gs)
            photon_numbers.append(n_photon)
        return np.array(photon_numbers)

    def sweep_photon_number_vs_g(self):
        pass

    
def plot_photon_number_vs_omega(omega_list: np.ndarray, photon_numbers: np.ndarray):
    plt.figure()
    plt.plot(omega_list, photon_numbers, label="Photon Number")
    plt.xlabel(r'Photon Frequency $\Omega$')
    plt.ylabel(r'Average Photon Number $\langle N_{ph} \rangle$')
    plt.title("Photon Number vs Photon Frequency")
    plt.legend()
    plt.grid()
    plt.show()

def plot_photon_number_vs_omega_log_log(omega_list: np.ndarray, photon_numbers: np.ndarray):
    plt.figure()
    plt.loglog(omega_list, photon_numbers, label="Photon Number")
    plt.xlabel(r'Photon Frequency $\Omega$')
    plt.ylabel(r'Average Photon Number $\langle N_{ph} \rangle$')
    plt.title("Photon Number vs Photon Frequency (Log-Log Scale)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_photon_number_vs_U(U_list: np.ndarray, photon_numbers: np.ndarray):
    plt.figure()
    plt.plot(U_list, photon_numbers, label="Photon Number")
    plt.xlabel('Interaction Strength U')
    plt.ylabel(r'Average Photon Number $\langle N_{ph} \rangle$')
    plt.title("Photon Number vs Interaction Strength U")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    L = 10
    N_f = L // 2 # half-filling
    N_ph = 10 
    t = 1.0
    U = 3.0 * t
    omega = 10.0 * t
    g = 0.5

    basis = Basis(L, N_f, N_ph)  
    analyzer = Analyzer(basis, g=g, boundary_conditions="periodic") 
    omega_list = np.linspace(1, 100 + 1, 100) * t
    # photon_numbers_omega = analyzer.sweep_photon_number_vs_omega(omega_list, t=t, U=U)
    # # plot_photon_number_vs_omega(omega_list, photon_numbers)
    # plot_photon_number_vs_omega_log_log(omega_list, photon_numbers_omega)
    
    U_list = np.linspace(0, 20, 100) * t
    photon_numbers_U = analyzer.sweep_photon_number_vs_U(U_list, t=t, omega=omega)
    plot_photon_number_vs_U(U_list, photon_numbers_U)