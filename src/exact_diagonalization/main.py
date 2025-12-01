import argparse
import numpy as np
import time
import matplotlib.pyplot as plt 

from scipy.sparse.linalg import eigsh 

from src.exact_diagonalization.hamiltonian import Hamiltonian
from src.exact_diagonalization.basis import Basis



def parse_arguments():
    parser = argparse.ArgumentParser(description="Exact Diagonalization of 1D Hubbard Model")
    parser.add_argument("-L", type=int, default=4, help="Number of lattice sites")
    parser.add_argument("-N_f", type=int, default=2, help="Number of fermions")
    parser.add_argument("-N_ph", type=int, default=1, help="Maximum photon number")
    parser.add_argument("-t", type=float, default=1.0, help="Hopping parameter")
    parser.add_argument("-U", type=float, default=2.0, help="On-site interaction strength")
    parser.add_argument("-g", type=float, default=0.0, help="Coupling strength")
    parser.add_argument("-bc", "--boundary_conditions", type=str, choices=["periodic", "open"], default="periodic", help="Type of boundary conditions")
    return vars(parser.parse_args())



def main(L: int, N_f: int, N_ph: int, t: float, U: float, g: float, boundary_conditions: str):
    basis = Basis(L, N_f, N_ph)  # L sites, N particles
    hamiltonian = Hamiltonian(basis, g=g, boundary_conditions=boundary_conditions)
    
    # print(f"Interaction hamiltonian:\n{hamiltonian.interaction_matrix}")
    # print(f"Photon hamiltonian:\n{hamiltonian.photon_energy_matrix}")
    # print(f"Hopping hamiltonian:\n{hamiltonian.hopping_matrix}")

    # fix t to one and vary U
    U_list = np.linspace(-2, 20, 100)
    # U_list = [U]
    k = 3
    energies = {f"E_{i}": [] for i in range(k)}  # store lowest k eigenvalues
    for i, Ut in enumerate(U_list):
        H = hamiltonian.construct_hamiltonian_matrix(t=t, U=Ut, omega=1)
        
        # Diagonalize the Hamiltonian
        eigenvalues, eigenvectors = eigsh(H, k=k, which='SA') 
        # Store the lowest k eigenvalues
        for j in range(k):
            energies[f"E_{j}"].append(eigenvalues[j])
        if i % 10 == 0:
            print(f"Lowest eigenvalue for U={Ut:.2f}: {eigenvalues[0]:.2f}")
    energies_per_site = {key: np.array(val) / L for key, val in energies.items()}



    # print first 10 energies
    # for key, energy in energies.items():
    #     print(f"{key}: {energy[:10]}")


    # plot results
    for key, energy in energies_per_site.items():
        plt.plot(U_list, energy, label=key)
    plt.xlabel("U")
    plt.ylabel("Energy per site")
    plt.title("Ground state energy per site vs U")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args = parse_arguments()
    start = time.time()
    main(**args)
    end = time.time()
    print(f"Execution time: {end - start} seconds")