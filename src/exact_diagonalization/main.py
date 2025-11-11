import argparse
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
    hamiltonian = Hamiltonian(basis, t=t, U=U, boundary_conditions=boundary_conditions)
    H = hamiltonian.matrix
    run_checks = True
    if run_checks is True:
        H_np = H.toarray()
        assert (H_np == H_np.T.conj()).all(), "Hamiltonian is not Hermitian!"
    print("Hamiltonian matrix (in CSR format):")
    print(H)

    # Diagonalize the Hamiltonian
    eigenvalues, eigenvectors = eigsh(H, k=1, which='SA') 
    print(f"Lowest eigenvalue:\n{eigenvalues[0]}")
    print(f"Corresponding eigenvector:\n{eigenvectors[:,0]}")


if __name__ == "__main__":
    args = parse_arguments()
    main(**args)