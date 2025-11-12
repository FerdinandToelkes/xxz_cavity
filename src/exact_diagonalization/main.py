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
    parser.add_argument("-N", type=int, default=2, help="Number of particles")
    parser.add_argument("-t", type=float, default=1.0, help="Hopping parameter")
    parser.add_argument("-U", type=float, default=2.0, help="On-site interaction strength")
    parser.add_argument("-bc", "--boundary_conditions", type=str, choices=["periodic", "open"], default="periodic", help="Type of boundary conditions")
    return vars(parser.parse_args())



def main(L: int, N: int, t: float, U: float, boundary_conditions: str):
    basis = Basis(L, N)  # L sites, N particles
    hamiltonian = Hamiltonian(basis, boundary_conditions=boundary_conditions)
    
    # fix t to one and vary U
    U_list = np.linspace(-10, 10, 100)
    energies = []
    for i, Ut in enumerate(U_list):
        H = hamiltonian.construct_hamiltonian_matrix(t=1.0, U=Ut)
        # Diagonalize the Hamiltonian
        eigenvalues, eigenvectors = eigsh(H, k=1, which='SA') 
        energies.append(eigenvalues[0])
        if i % 10 == 0:
            print(f"Lowest eigenvalue for U={Ut:.2f}: {eigenvalues[0]:.2f}")
    energies_per_site = np.array(energies) / L

    # plot results
    plt.plot(U_list, energies_per_site)
    plt.xlabel("U")
    plt.ylabel("Energy per site")
    plt.title("Ground state energy per site vs U")
    plt.show()


if __name__ == "__main__":
    args = parse_arguments()
    start = time.time()
    main(**args)
    end = time.time()
    print(f"Execution time: {end - start} seconds")