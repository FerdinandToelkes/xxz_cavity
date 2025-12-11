import numpy as np
import hydra
import logging
import os

from omegaconf import DictConfig, OmegaConf
from scipy.sparse import csr_matrix



from src.exact_diagonalization.basis import Basis
from src.exact_diagonalization.hamiltonian_builder import HamiltonianBuilder
from src.exact_diagonalization.analyzer import Analyzer
from scripts.exact_diagonalization.utils import log_config, register_hydra_resolvers, get_name_from_parameters, save_relevant_config

logger = logging.getLogger(__name__)
register_hydra_resolvers()


def get_sweep_range(params: dict, name: str) -> tuple[float, float, int]:
    """
    Extract sweep range parameters from the config.
    Arguments:
        params (dict): The parameters dictionary.
        name (str): The name of the parameter to sweep.
    Raises:
        KeyError: If any of the required parameters are missing.
    Returns:
        tuple[float, float, int]: The minimum value, maximum value, and number of points for the sweep.
    """
    try:
        return params[f"{name}_min"], params[f"{name}_max"], params[f"{name}_points"]
    except KeyError as e:
        raise KeyError(f"Missing parameter for sweep '{name}': {e}")


def sweep_parameter(observable: str, builder: HamiltonianBuilder, param_name: str, param_values: np.ndarray,
                    fixed_params: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Sweep a given parameter for a specified observable.
    Arguments:
        observable (str): The observable to compute ("photon_number", "entanglement_entropy", "longest_range_correlation").
        builder (HamiltonianBuilder): The Hamiltonian builder instance.
        param_name (str): The name of the parameter to sweep.
        param_values (np.ndarray): The values of the parameter to sweep.
        fixed_params (DictConfig): The fixed parameters for the Hamiltonian.
    Raises:
        ValueError: If an unknown observable is provided.
    Returns:
        tuple[np.ndarray, np.ndarray]: The parameter values and the corresponding computed observable values.
    """
    vals = []
    # extract parameters from fixed_params to a dictionary
    params = dict(fixed_params)
    for val in param_values:
        params[param_name] = val
        H = builder.build_hamiltonian_matrix(t=params["t"], U=params["U"], omega=params["omega"])
        analyzer = Analyzer(H, builder.basis)
        gs = analyzer.ground_state()
        if observable == "photon_number":
            value = analyzer.expectation_value(gs, analyzer.photon_number_matrix)
        elif observable == "entanglement_entropy":
            value = analyzer.entanglement_entropy_fermions_photons(gs)
        elif observable == "longest_range_correlation":
            value = analyzer.expectation_value(gs, analyzer.longest_range_fermion_number_matrix)
        else:
            raise ValueError(f"Unknown observable: {observable}")
        vals.append(value)
    return (param_values, np.array(vals))



@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    log_config(logger, cfg)
    
    # setup that is independent of sweeps
    root_data_dir = cfg.root_data_dir
    system_params = cfg.system
    basis = Basis(system_params.L, system_params.N_f, system_params.N_ph)

    for observable, observable_cfg in cfg.sweeps.items():
        for sweep in list(observable_cfg.keys()):
            logger.info(f"Running sweep: {sweep} for observable: {observable}")
            
            params = observable_cfg[sweep].parameters
            builder = HamiltonianBuilder(basis, g=params.g, boundary_conditions=system_params.boundary_conditions)

            # create save directory using parameter names
            param_name = sweep.removesuffix("_sweep")
            parameters_as_name = get_name_from_parameters(system_params, params, param_name)
            save_dir = os.path.join(root_data_dir, observable, sweep, parameters_as_name)
            
            # if save dir already exists, ask whether to overwrite
            if os.path.exists(save_dir) and not cfg.overwrite_existing:
                logger.warning(f"Save directory {save_dir} already exists. Skipping sweep. To overwrite, set 'overwrite_existing' to true in the config.")
                continue
            os.makedirs(save_dir, exist_ok=True)

            p_min, p_max, p_pts = get_sweep_range(params, param_name)
            param_values = np.linspace(p_min, p_max, p_pts)
            results = sweep_parameter(observable, builder, param_name, param_values, params)

            # save results and used config
            results_filename = os.path.join(save_dir, "results.npy")
            config_filename = os.path.join(save_dir, "used_config.yaml")
            np.save(results_filename, results)
            save_relevant_config(config_filename, cfg, observable, system_params, params)
            
            logger.info(f"Saved results to {results_filename} and config to {config_filename}")
            
    return
    
    

if __name__ == "__main__":
    main()




