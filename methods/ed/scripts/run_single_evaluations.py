import hydra
import logging
import os
import numpy as np

from omegaconf import DictConfig

from ed.basis import Basis
from ed.hamiltonian_builder import HamiltonianBuilder
from ed.analyzer import Analyzer
from scripts.utils import log_config, register_hydra_resolvers, get_name_from_parameters, save_relevant_config
 

logger = logging.getLogger(__name__)
register_hydra_resolvers()

# this script can be run from the methods/ed/ directory with:
# python3 -m scripts.run_single_evaluations system.L=4 system.boundary_conditions=open

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log_config(logger, cfg)
    
    root_data_dir = cfg.root_data_dir
    system_params = cfg.system
    observable = "photon_number_distribution"
    params = cfg.single_evaluations[observable].parameters
    
    # non existing param to exclude no parameters from name
    param_name = "I_DO_NOT_THINK"
    parameters_as_name = get_name_from_parameters(system_params, params, param_name) 
    save_dir = os.path.join(root_data_dir, observable, "single_evaluations", parameters_as_name)
    
    # if save dir already exists, ask whether to overwrite
    if os.path.exists(save_dir) and not cfg.overwrite_existing:
        logger.warning(f"Save directory {save_dir} already exists. To overwrite, set 'overwrite_existing' to true in the config.")
        return
    os.makedirs(save_dir, exist_ok=True)

    # setup basis and hamiltonian
    basis = Basis(system_params.L, system_params.N_f, system_params.N_ph)
    builder = HamiltonianBuilder(basis, g=params.g, boundary_conditions=system_params.boundary_conditions)
    H = builder.build_hamiltonian_matrix(t=params.t, U=params.U, omega=params.omega)
    analyzer = Analyzer(H, basis)
    gs = analyzer.ground_state()
    photon_number_distribution = analyzer.probability_distribution_photon_number(gs)
    # Save the photon number distribution
    np.save(os.path.join(save_dir, "results.npy"), photon_number_distribution)
    save_relevant_config(os.path.join(save_dir, "used_config.yaml"), cfg, observable, system_params, params)
    logger.info(f"Saved photon number distribution to {save_dir}")


if __name__ == "__main__":
    main()