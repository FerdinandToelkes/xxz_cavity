# xxz_cavity

[![CI - ED](https://github.com/FerdinandToelkes/xxz_cavity/actions/workflows/CI-ed.yml/badge.svg)](https://github.com/FerdinandToelkes/xxz_cavity/actions/workflows/CI-ed.yml)
[![CI - DMRG](https://github.com/FerdinandToelkes/xxz_cavity/actions/workflows/CI-Dmrg.yml/badge.svg)](https://github.com/FerdinandToelkes/xxz_cavity/actions/workflows/CI-Dmrg.yml)
[![Codecov](https://codecov.io/gh/FerdinandToelkes/xxz_cavity/branch/main/graph/badge.svg)](https://codecov.io/gh/FerdinandToelkes/xxz_cavity)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

---
The goal of this project is to study the extended Hubbard Hamiltonian for a one-dimensional lattice of spinless fermions. These fermions are coupled to a single cavity mode within the dipole approximation, i.e. the mode is spatially uniform across the lattice. The Hamiltonian reads:
```math
H = \sum_{j=1}^{L-1} -t \left( e^{i\frac{g}{\sqrt{L}}(a + a^\dagger)} c^\dagger_j c_{j+1} + e^{-i\frac{g}{\sqrt{L}}(a + a^\dagger)} c^\dagger_{j+1} c_{j} \right) + \sum_{j=1}^{L-1} U n_j n_{j+1} + \Omega N_{\text{ph}} \, .
```
Passetti et al. examined this Hamiltonian in the paper [Cavity Light-Matter Entanglement through Quantum Fluctuations](https://link.aps.org/doi/10.1103/PhysRevLett.131.023601) using, among other methods, the density matrix renormalization group (DMRG) algorithm. The aim of this project is to use this system to benchmark various numerical approaches, namely
- Exact diagonalization (ED)
- Density matrix renormalization group (DMRG)
- Neural quantum states (NQS)

## Project Structure

The repository is organized as a monorepo containing multiple method-specific subprojects. Each method is implemented as a self-contained package ([Julia](https://pkgdocs.julialang.org/dev/creating-packages/#Adding-tests-to-the-package) or [Python](https://packaging.python.org/en/latest/tutorials/packaging-projects/)) with its own dependencies, source code, and scripts for data generation. A shared configuration and a common output format enable direct comparison between methods. Each method-specific `scripts` directory contains executable entry points for producing benchmark data, which are written to `numpy`-compatible files. Top-level analysis and comparison scripts (to be added) operate on these outputs to facilitate a direct comparison between different numerical approaches. The benchmarked quantities include the entanglement entropy, the photon number, the photon number distribution, and the longest-range correlation, as these observables were also investigated by Passetti et al.

## Notes on the theoretical background

I aim to provide a concise overview of the relevant concepts involved, as well as pointers to resources that may be useful when starting to work on this topic. I have attached my personal, handwritten notes here for readers interested in additional details. These notes have not been rigorously proofread, so please excuse any mistakes they may/probably contain. Hopefully it is possible to decode my handwriting.

### The Hamiltonian

In the absence of coupling to electromagnetic modes, the Hamiltonian reduces to the extended Hubbard model, which can be derived by following, for example, the book *Grundkurs Theoretische Physik 7 – Viel-Teilchen-Theorie* by W. Nolting. The introduction of the Peierls phase is discussed, for instance, in the paper [Gauge fixing for strongly correlated electrons coupled to quantum light](https://link.aps.org/doi/10.1103/PhysRevB.103.075131) by Dmytruk and Schiró or in [Electromagnetic coupling in tight-binding models for strongly correlated light and matter](https://link.aps.org/doi/10.1103/PhysRevB.101.205140) by Li et al., both of which are also cited by Passetti et al. Further discussion of the first of these two papers can be found in my notes.

### Exact diagonalization
Explanation of the ED implementation and design philosophy.

### Density matrix renormalization group (DMRG)
Outline of the DMRG approach and its numerical structure.

### Neural quantum states NQS

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

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




