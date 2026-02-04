import jax
import os
import netket as nk
from netket.operator.spin import sigmax, sigmaz
from netket.operator import LocalOperator


# set jax to use cpu only
os.environ["JAX_PLATFORM_NAME"] = "cpu"



def main():
    """Tutorial 1: Defining a Hamiltonian and getting its sparse matrix representation.
    
    """
    N = 20

    # define Hilbert space
    hi = nk.hilbert.Spin(s=1/2, N=N)

    # rs = hi.random_state(jax.random.key(0), 3)
    # print(rs)

    # define Hamiltonian
    Gamma = -1
    V = -1
    H = sum([Gamma * sigmax(hi, i) for i in range(N)])
    # H = sum([Gamma * sigmax(hi, i) for i in range(N)], start=LocalOperator(hi))
    H += sum([V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N) for i in range(N)])

    # ED
    sp_h = H.to_sparse()
    print(f"Shape of sparse Hamiltonian: {sp_h.shape}")



if __name__ == "__main__":
    main()