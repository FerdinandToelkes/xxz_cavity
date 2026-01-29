# xxz_cavity

[![CI - ED](https://github.com/FerdinandToelkes/xxz_cavity/actions/workflows/CI-ed.yml/badge.svg)](https://github.com/FerdinandToelkes/xxz_cavity/actions/workflows/CI-ed.yml)
[![CI - DMRG](https://github.com/FerdinandToelkes/xxz_cavity/actions/workflows/CI-Dmrg.yml/badge.svg)](https://github.com/FerdinandToelkes/xxz_cavity/actions/workflows/CI-Dmrg.yml)
[![Codecov](https://codecov.io/gh/FerdinandToelkes/xxz_cavity/branch/main/graph/badge.svg)](https://codecov.io/gh/FerdinandToelkes/xxz_cavity)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

---

## Notes on the theoretical background

### The Hamiltonian
Brief overview of the model and its physical origin  
(e.g. Hubbard/XXZ limit, see Nolting).

### Exact Diagonalization
Explanation of the ED implementation and design philosophy.

### Density Matrix Renormalization Group (DMRG)
Outline of the DMRG approach and its numerical structure.

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




