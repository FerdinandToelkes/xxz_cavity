# xxz_cavity

[![CI - ED](https://github.com/FerdinandToelkes/xxz_cavity/actions/workflows/CI-ed.yml/badge.svg)](https://github.com/FerdinandToelkes/xxz_cavity/actions/workflows/CI-ed.yml)
[![CI - DMRG](https://github.com/FerdinandToelkes/xxz_cavity/actions/workflows/CI-Dmrg.yml/badge.svg)](https://github.com/FerdinandToelkes/xxz_cavity/actions/workflows/CI-Dmrg.yml)
[![Codecov](https://codecov.io/gh/FerdinandToelkes/xxz_cavity/branch/main/graph/badge.svg)](https://codecov.io/gh/FerdinandToelkes/xxz_cavity)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

---
The goal of this project is to study the extended Hubbard Hamiltonian for a one-dimensional lattice of spinless fermions at half-filling. These fermions are coupled to a single cavity mode within the dipole approximation, i.e. the mode is spatially uniform across the lattice. The Hamiltonian reads:
```math
H = \sum_{j=1}^{L-1} -t \left( e^{i\frac{g}{\sqrt{L}}(a + a^\dagger)} c^\dagger_j c_{j+1} + e^{-i\frac{g}{\sqrt{L}}(a + a^\dagger)} c^\dagger_{j+1} c_{j} \right) + \sum_{j=1}^{L-1} U n_j n_{j+1} + \Omega N_{\text{ph}} \, .
```
Passetti et al. examined this Hamiltonian in the paper [Cavity Light-Matter Entanglement through Quantum Fluctuations](https://link.aps.org/doi/10.1103/PhysRevLett.131.023601) using, among other methods, the density matrix renormalization group (DMRG) algorithm. The aim of this project is to use this system to benchmark various numerical approaches, namely
- Exact diagonalization (ED)
- Density matrix renormalization group (DMRG)
- Neural quantum states (NQS)

## Project Structure

The repository is organized as a monorepo containing multiple method-specific subprojects. Each method is implemented as a self-contained package ([Julia](https://pkgdocs.julialang.org/dev/creating-packages/#Adding-tests-to-the-package) or [Python](https://packaging.python.org/en/latest/tutorials/packaging-projects/)) with its own dependencies, source code, and scripts for data generation. A shared configuration and a common output format enable direct comparison between methods. Each method-specific `scripts` directory contains executable entry points for producing benchmark data, which are written to `numpy`-compatible files. Top-level analysis and comparison scripts (to be added) operate on these outputs to facilitate a direct comparison between different numerical approaches. The benchmarked quantities include the entanglement entropy, the photon number, the photon number distribution, and the longest-range correlation, as these observables were also investigated by Passetti et al.

## Installation guide

## Notes on the theoretical background

I aim to provide a concise overview of the relevant concepts involved, as well as pointers to resources that may be useful when starting to work on this topic. I have attached my personal, handwritten notes here for readers interested in additional details. These notes have not been rigorously proofread, so please excuse any mistakes they may/probably contain. Hopefully it is possible to decode my handwriting.

### The Hamiltonian

In the absence of coupling to electromagnetic modes, the Hamiltonian reduces to the extended Hubbard model, which can be derived by following, for example, the book *Grundkurs Theoretische Physik 7 – Viel-Teilchen-Theorie* by W. Nolting. The introduction of the Peierls phase is discussed, for instance, in the paper [Gauge fixing for strongly correlated electrons coupled to quantum light](https://link.aps.org/doi/10.1103/PhysRevB.103.075131) by Dmytruk and Schiró or in [Electromagnetic coupling in tight-binding models for strongly correlated light and matter](https://link.aps.org/doi/10.1103/PhysRevB.101.205140) by Li et al., both of which are also cited by Passetti et al. Further discussion of the first of these two papers can be found in my notes.

### Exact diagonalization (ED)

Exact diagonalization (ED) is a natural starting point, as it is conceptually straightforward and allows one to become familiar with the Hamiltonian under investigation. The basic idea of ED is to construct a matrix representation of the Hamiltonian by evaluating its action on the occupation-number basis. Once this matrix representation is obtained, standard numerical algorithms can be used to compute the ground-state energy and the corresponding eigenstate.

For Hamiltonians composed of few-site operators, such as on-site interactions or nearest-neighbor hopping terms, the resulting matrix is known to be sparse. This sparsity can be exploited to construct the Hamiltonian efficiently, and to apply algorithms such as the Lanczos method that are well suited for computing a small number of extremal eigenstates. If the system exhibits symmetries, the Hamiltonian acquires a block-diagonal structure, with each block corresponding to a fixed quantum number. In this case, conserved quantities, such as the total number of fermions, can be used to restrict the construction to a single symmetry sector, thereby significantly reducing the effective dimension of the problem. The main limitation of ED is its restriction to small system sizes. The dimension of the Hilbert space, and hence of the Hamiltonian matrix, grows exponentially with system size, which quickly renders both memory requirements and runtimes prohibitive.

I implemented my own ED code, loosely following the paper [Introduction to Hubbard Model and Exact Diagonalization](https://arxiv.org/pdf/0807.4878) by Jafari and the tutorial [Writing an Exact Diagonalization (ED) Routine for the Hubbard
Model Hamiltonian](https://stanford.edu/~xunger08/Exact%20Diagonalization%20Tutorial.pdf) by Ding. Since the system consists of fermionic degrees of freedom coupled to a bosonic mode, the Hilbert space is constructed as a tensor product of fermionic and bosonic basis states. For the fermionic sector, I use a bitwise integer representation to implement the action of the Hamiltonian. The fermionic and photonic parts of the Hamiltonian are constructed separately and subsequently combined using `numpy`’s Kronecker product, which provides a concrete realization of the tensor product. I found this approach to be more efficient than constructing the Hamiltonian directly on the full combined Hilbert space. 

As the bosonic sector can, in principle, host an infinite number of photons, a photon-number cutoff is introduced. All bosonic operators, including the matrix representation of the Peierls phase, are therefore evaluated within this truncated Hilbert space. To compute the Peierls phase, I first construct the matrix $A$ representing  $a^\dagger + a$ in the truncated photon basis. I then diagonalize this matrix as $A = V D V^\dagger$ where $D$ is the diagonal matrix of eigenvalues and $V$ contains the corresponding eigenvectors. The matrix exponential is subsequently obtained as
```math
\exp\left(i\frac{g}{\sqrt{L}} (a^\dagger + a) \right) = V \exp \left( i\frac{g}{\sqrt{L}} D \right) V^\dagger.
```
I carried out analytical calculations for specific limiting cases to validate both the construction of the Hamiltonian and the implementation of the Peierls phase. These calculations are documented in my handwritten notes, and their results are used in a hard-coded manner within the unit tests.

### Density matrix renormalization group (DMRG)
The density matrix renormalization group (DMRG) algorithm is a variational method based on matrix product states (MPS). Within the tensor network formalism, one can show that DMRG provides an efficient approach for computing ground states of one-dimensional gapped Hamiltonians. To familiarize myself with the basic ideas of tensor networks and MPS, I relied primarily on the papers [A Practical Introduction to Tensor Networks: Matrix Product States and Projected Entangled Pair States](http://arxiv.org/abs/1306.2164) by Orús and 
[Tensor Networks in a Nutshell](http://arxiv.org/abs/1708.00006) by Biamonte and Bergholm to get familiarized with the basic ideas of tensor networks and MPS. Given the popularity of DMRG, many additional resources are available. In this context, the review [The density-matrix renormalization group in the age of matrix product states](http://arxiv.org/abs/1008.3477) by Schollwöck is often referred to as the “bible” of tensor network methods.

A very quick and dirty summary of DMRG could go as follows. An MPS representation of a quantum many-body state can be written as
```math
|\psi\rangle = \sum_{\sigma_1 \dots \sigma_L} M^{\sigma_1} \, M^{\sigma_2} \, \dots \, M^{\sigma_L} |\sigma_1 \dots \sigma_L\rangle
```
where the $M^{\sigma_i}$ are matrices and the indices $\sigma_i$ label the local physical degrees of freedom (for example, $\sigma_i=0,1$ for spin-1/2 particles). Starting from an initial guess for $|\psi\rangle$ , the algorithm proceeds by sweeping through the tensors and updating them successively through a local minimization of 
```math
\langle \psi | \hat{H} | \psi \rangle - \lambda \langle \psi | \psi \rangle \, .
```
To carry out this procedure, the Hamiltonian must be represented in a form compatible with an MPS. This is achieved using matrix product operators (MPOs), which can be written as
```math
\hat{O} = \sum_{\boldsymbol{\sigma}, \boldsymbol{\sigma}'} W^{\sigma_1, \sigma'_1} \, W^{\sigma_2, \sigma'_2} \, \dots \, W^{\sigma_L, \sigma'_L} | \boldsymbol{\sigma} \rangle \langle \boldsymbol{\sigma}' | \, .
```
Each local minimization step reduces to an effective eigenvalue problem, which can be solved using iterative algorithms such as the Lanczos method. Multiple sweeps over the MPS are performed until convergence is reached. Convergence can be monitored, for example, by evaluating $\langle \psi | \hat{H}^2 | \psi \rangle - \left( \psi | \hat{H} | \psi \right)$ which vanishes if $\psi\rangle$ is an exact eigenstate of $\hat{H}$. A concise and well-balanced introduction to DMRG and its underlying concepts can be found in [The density-matrix renormalization group: a short introduction](https://royalsocietypublishing.org/doi/10.1098/rsta.2010.0382) by Schollwöck.

I implemented DMRG using the Julia version of [The ITensor Software Library for Tensor Network Calculations](https://scipost.org/10.21468/SciPostPhysCodeb.4). In order to rigorously test my implementation, I studied different approaches to constructing MPOs. One particularly pedagogical method is based on finite state machines (FSMs), as described for example in the first part of [Finite automata for caching in matrix product algorithms](https://link.aps.org/doi/10.1103/PhysRevA.78.012356) by Crosswhite and Bacon. A more direct construction scheme is presented by Hubig et al. in [Generic Construction of Efficient Matrix Product Operators](http://arxiv.org/abs/1611.02498),which also discusses the treatment of Abelian symmetries (such as particle number conservation) and fermionic operators via the Jordan–Wigner transformation. The `ITensor` in [Matrix product operators, matrix product states, and ab initio density matrix renormalization group algorithms](https://pubs.aip.org/jcp/article/145/1/014102/899058/Matrix-product-operators-matrix-product-states-and) by Chan et al. for the implementation of the `OpSum` interface used to construct MPOs from sums of local operators (at least according to [this blog post](https://itensor.discourse.group/t/manual-construction-of-nearest-neighbor-hopping-of-spinless-fermions-on-a-1d-chain/2567/5)).

TODO: Explain how to deal with photonic mode


Outline of the DMRG approach and its numerical structure.

### Neural quantum states (NQS)

---

## Contributing

Contributions are always welcome. Please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

# Notes

## Exact Diagonalization

### To Do
- open vs periodic boundary conditions


- run_simulation also for no photons and no coupling case
    - use FFT to transform ground states into momentum space (only for periodic boundary conditions) and compare to dispersion relation
    - show ground state for the extreme cases of strong interaction and no interaction 
    - locate phase transitions?
    
- use iTensors to implement DMRG
- implement periodic boundary conditions in DMRG code (see for example https://github.com/ITensor/ITensorMPOConstruction.jl) 

- write numpy style docstrings for functions -> autodoc with sphinx

### Open Questions
- why is the 1/omega**2 line not matching with entanglement entropy? -> is there a factor missing -> yes probably because paper just show proportionality and only the functional dependence is relevant
- why are there some values for fermion and photon numbers where the entropy exhibits non-continuous behavior?
- behaviour of longest range correlator looks BAD -> maybe too small systems -> check with larger systems with DMRG?

## DMRG

- switch to periodic boundary conditions -> how has passetti done it?
- see https://github.com/GiacomoPassetti/dmrg_cav/blob/main/Peier_v1(trivial).py


## NQS

- got general overview of different architectures
- ground state by optimizing variational state (NN) within VMC
- dynamics via t-VMC -> time dependence in parameters -> Fisher matrix
- exited states?


## Monte Carlo

- 




