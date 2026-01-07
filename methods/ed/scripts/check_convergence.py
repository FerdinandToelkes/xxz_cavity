import hydra
import logging
import os
import numpy as np

from omegaconf import DictConfig, OmegaConf

from src.basis import Basis
from src.hamiltonian_builder import HamiltonianBuilder
from src.analyzer import Analyzer
from scripts.utils import log_config, register_hydra_resolvers, get_name_from_parameters, save_relevant_config
 

logger = logging.getLogger(__name__)
register_hydra_resolvers()


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log_config(logger, cfg)
    
    root_data_dir = cfg.root_data_dir
    system_params = cfg.system
    params = cfg.convergence_checks.parameters
    observables = cfg.convergence_checks.observables
    photon_number_cutoffs = cfg.convergence_checks.photon_number_cutoffs

    for observable in observables:
        logger.info(f"Running convergence check for observable: {observable}")
        parameters_as_name = get_name_from_parameters(system_params, params, "N_ph")
        save_dir = os.path.join(root_data_dir, observable, "convergence_check", parameters_as_name)

        # if save dir already exists, ask whether to overwrite
        if os.path.exists(save_dir) and not cfg.overwrite_existing:
            logger.warning(f"Save directory {save_dir} already exists. Skipping sweep. To overwrite, set 'overwrite_existing' to true in the config.")
            continue
        os.makedirs(save_dir, exist_ok=True)
        
        vals = []
        for N_ph_cutoff in photon_number_cutoffs:

            basis = Basis(system_params.L, system_params.N_f, N_ph_cutoff)
            builder = HamiltonianBuilder(basis, g=params.g, boundary_conditions=system_params.boundary_conditions)
            H = builder.build_hamiltonian_matrix(t=params.t, U=params.U, omega=params.omega)
            analyzer = Analyzer(H, basis)
            gs = analyzer.ground_state()
            if observable == "photon_number":
                value = analyzer.expectation_value(gs, analyzer.photon_number_matrix)
            elif observable == "longest_range_correlation":
                value = analyzer.expectation_value(gs, analyzer.longest_range_fermion_number_matrix)
            elif observable == "entanglement_entropy":
                value = analyzer.entanglement_entropy_fermions_photons(gs)
            else:
                raise ValueError(f"Unknown observable: {observable}")
            
            vals.append(value)

        # save results and used config
        results_filename = os.path.join(save_dir, "results.npy")
        np.save(results_filename, (photon_number_cutoffs, np.array(vals)))
        config_filename = os.path.join(save_dir, "used_config.yaml")
        save_relevant_config(config_filename, cfg, observable, system_params, params)
        logger.info(f"Saved convergence check results to {results_filename} and config to {config_filename}")

    
if __name__ == "__main__":
    main()
