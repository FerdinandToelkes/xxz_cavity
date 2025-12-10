import hydra
import logging
import os
import h5py
import numpy as np

from omegaconf import DictConfig, OmegaConf

from src.exact_diagonalization.basis import Basis
from src.exact_diagonalization.hamiltonian_builder import HamiltonianBuilder
from src.exact_diagonalization.analyzer import Analyzer
 

logger = logging.getLogger(__name__)


def get_name_from_parameters(system_params: dict, params: dict, param_name: str) -> str:
    """
    Generate a string representation of the system parameters excluding the sweep parameter.
    Arguments:
        system_params (dict): The system parameters from the config.
        params (dict): The sweep parameters from the config.
        param_name (str): The name of the parameter being swept.
    Returns:
        str: A string representation of the parameters for naming.
    """
    parameters_as_name = ""
    for key, value in system_params.items():
        if not key.startswith(param_name):
            parameters_as_name += f"{key}={value}_"
    for key, value in params.items():
        if not key.startswith(param_name):
            parameters_as_name += f"{key}={value}_"
    # remove trailing underscore
    parameters_as_name = parameters_as_name.rstrip("_")
    return parameters_as_name

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(OmegaConf.to_container(cfg, resolve=True)))
    
    system_params = cfg.system
    params = cfg.convergence_checks.parameters
    observables = cfg.convergence_checks.observables
    photon_number_cutoffs = cfg.convergence_checks.photon_number_cutoffs

    if system_params.N_f is None:
        system_params.N_f = system_params.L // 2

    # TODO: make this nicer!
    g = params.g / np.sqrt(system_params.L)

    results = {}
    for observable in observables:
        exp_values = []
        for N_ph_cutoff in photon_number_cutoffs:
            logger.debug(f"Running convergence check for observable: {observable} with photon cutoff: {N_ph_cutoff}")
            basis = Basis(system_params.L, system_params.N_f, N_ph_cutoff)
            builder = HamiltonianBuilder(basis, g=g, boundary_conditions=system_params.boundary_conditions)
            H = builder.build_hamiltonian_matrix(t=params.t, U=params.U, omega=params.omega)
            analyzer = Analyzer(H, basis)
            gs = analyzer.ground_state()
            if observable == "photon_number":
                exp_value = analyzer.expectation_value(gs, analyzer.photon_number_matrix)
            elif observable == "longest_range_correlation":
                exp_value = analyzer.expectation_value(gs, analyzer.longest_range_fermion_number_matrix)
            elif observable == "entanglement_entropy":
                exp_value = analyzer.entanglement_entropy_fermions_photons(gs)
            else:
                raise ValueError(f"Unknown observable: {observable}")
            
            exp_values.append(exp_value)
            logger.debug(f"Result for {observable} with photon cutoff {N_ph_cutoff}: {exp_value}")

        results[observable] = (photon_number_cutoffs, exp_values)

    # save to hdf5 file
    root_data_dir = cfg.root_data_dir
    filename = os.path.join(root_data_dir, f"ed_results_N_ph_convergence.h5")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    parameters_as_name = get_name_from_parameters(system_params, params, "N_ph")
    logger.info(f"Saving results to {filename} under group {parameters_as_name}")
    
    with h5py.File(filename, "a") as f:
        # create or open the group
        grp = f.require_group(parameters_as_name)

        # Create datasets for each observable
        for observable, data in results.items():
            if observable not in grp:
                grp.create_dataset(observable, data=data)
            else:
                overwrite = input(f"Dataset {observable} already exists in group {parameters_as_name}. Overwrite? (y/n): ")
                if overwrite.lower() == 'y':
                    del grp[observable]
                    grp.create_dataset(observable, data=data)
                else:
                    logger.info(f"Skipping dataset {observable} in group {parameters_as_name}.")

if __name__ == "__main__":
    main()
