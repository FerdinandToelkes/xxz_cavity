import numpy as np
import hydra
import logging
import h5py
import os

from omegaconf import DictConfig, OmegaConf
from scipy.sparse import csr_matrix


from src.exact_diagonalization.basis import Basis
from src.exact_diagonalization.hamiltonian_builder import HamiltonianBuilder
from src.exact_diagonalization.analyzer import Analyzer

logger = logging.getLogger(__name__)



def sweep_parameter_for_photon_number(builder: HamiltonianBuilder, param_name: str, param_values: np.ndarray, fixed_params: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    exp_vals = []
    # extract parameters from fixed_params to a dictionary
    params = dict(fixed_params)
    for val in param_values:
        params[param_name] = val
        H = builder.build_hamiltonian_matrix(t=params["t"], U=params["U"], omega=params["omega"])
        analyzer = Analyzer(H, builder.basis)
        gs = analyzer.ground_state()
        exp_value = analyzer.expectation_value(gs, analyzer.photon_number_matrix)
        exp_vals.append(exp_value)
    return (param_values, np.array(exp_vals))

def sweep_parameter_for_longest_range_correlator(builder: HamiltonianBuilder, param_name: str, param_values: np.ndarray, fixed_params: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    exp_vals = []
    # extract parameters from fixed_params to a dictionary
    params = dict(fixed_params)
    for val in param_values:
        params[param_name] = val
        H = builder.build_hamiltonian_matrix(t=params["t"], U=params["U"], omega=params["omega"])
        analyzer = Analyzer(H, builder.basis)
        gs = analyzer.ground_state()
        exp_value = analyzer.expectation_value(gs, analyzer.longest_range_fermion_number_matrix)
        exp_vals.append(exp_value)
    return (param_values, np.array(exp_vals))

def sweep_parameter_for_entanglement_entropy(builder: HamiltonianBuilder, param_name: str, param_values: np.ndarray, fixed_params: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    exp_vals = []
    # extract parameters from fixed_params to a dictionary
    params = dict(fixed_params)
    for val in param_values:
        params[param_name] = val
        H = builder.build_hamiltonian_matrix(t=params["t"], U=params["U"], omega=params["omega"])
        analyzer = Analyzer(H, builder.basis)
        gs = analyzer.ground_state()
        S = analyzer.entanglement_entropy_fermions_photons(gs)
        exp_vals.append(S)
    return (param_values, np.array(exp_vals))

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
        parameters_as_name += f"{key}={value}_"
    for key, value in params.items():
        if not key.startswith(param_name):
            parameters_as_name += f"{key}={value}_"
    # remove trailing underscore
    parameters_as_name = parameters_as_name.rstrip("_")
    return parameters_as_name

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(OmegaConf.to_container(cfg, resolve=True)))
    
    root_data_dir = cfg.root_data_dir
    # the system parameters are independent of the sweep parameters
    system_params = cfg.system
    if system_params.N_f is None:
        system_params.N_f = system_params.L // 2
    
    basis = Basis(system_params.L, system_params.N_f, system_params.N_ph)

    
    sweeps_per_observable = {}
    for observable, observable_cfg in cfg.sweeps.items():
        sweeps_per_observable[observable] = list(observable_cfg.keys())
    logger.info(f"Observables per sweep: {sweeps_per_observable}")
    

    for observable, observable_cfg in cfg.sweeps.items():
        for sweep in sweeps_per_observable[observable]:
            logger.info(f"Running sweep: {sweep}, observable: {observable}")
            
            # TODO: make this nicer!
            params = observable_cfg[sweep].parameters
            g = params.g / np.sqrt(system_params.L)
            builder = HamiltonianBuilder(basis, g=g, boundary_conditions=system_params.boundary_conditions)

            param_name = sweep.removesuffix("_sweep")
            try:
                p_min = params[f"{param_name}_min"]
                p_max = params[f"{param_name}_max"]
                p_pts = params[f"{param_name}_points"]
            except KeyError as e:
                raise KeyError(f"Missing parameter for sweep '{sweep}': {e}")

            param_values = np.linspace(p_min, p_max, p_pts)
            if observable == "photon_number":
                sweep_parameter = sweep_parameter_for_photon_number
            elif observable == "longest_range_correlation":
                print(f"Running longest range correlation sweep for {sweep}")
                sweep_parameter = sweep_parameter_for_longest_range_correlator
            elif observable == "entanglement_entropy":
                sweep_parameter = sweep_parameter_for_entanglement_entropy
            else:
                raise ValueError(f"Unknown observable: {observable}")
            results = sweep_parameter(builder, param_name, param_values, params)

            # create a file if it does not exist yet otherwise append to it
            filename = os.path.join(root_data_dir, f"ed_results_{sweep}.h5")
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # group by parameters
            parameters_as_name = get_name_from_parameters(system_params, params, param_name)
            
            with h5py.File(filename, "a") as f:
                # Create or open the group
                grp = f.require_group(parameters_as_name)

                # Create dataset in that group
                if observable not in grp:
                    grp.create_dataset(observable, data=results)
                else:
                    logger.warning(f"Dataset '{observable}' already exists in group '{parameters_as_name}'.")

                # Add config as attribute to the group (not the whole file)
                grp.attrs['config'] = OmegaConf.to_yaml(
                    OmegaConf.to_container(cfg, resolve=True)
                )

            logger.info(f"Saved results to {filename}")
            
    return
    
    

if __name__ == "__main__":
    main()


# def sweep_photon_number_vs_omega(self, omega_list: np.ndarray, t: float, U: float) -> np.ndarray:
#     photon_numbers = []
#     for omega in omega_list:
#         H = self.base_hamiltonian.build_hamiltonian_matrix(t=t, U=U, omega=omega)
#         evals, evecs = self.diagonalize(H, k=1) # only interested in ground state
#         gs = evecs[:, 0]
#         n_photon = self.photon_number(gs)
#         photon_numbers.append(n_photon)
#     return np.array(photon_numbers)

# def sweep_photon_number_vs_U(self, U_list: np.ndarray, t: float, omega: float) -> np.ndarray:
#     photon_numbers = []
#     for U in U_list:
#         H = self.base_hamiltonian.build_hamiltonian_matrix(t=t, U=U, omega=omega)
#         evals, evecs = self.diagonalize(H, k=1) # only interested in ground state
#         gs = evecs[:, 0]
#         n_photon = self.photon_number(gs)
#         photon_numbers.append(n_photon)
#     return np.array(photon_numbers)

# def sweep_entanglement_entropy_vs_U(self, U_list: np.ndarray, t: float, omega: float) -> np.ndarray:
#     entropies = []
#     for U in U_list:
#         H = self.base_hamiltonian.build_hamiltonian_matrix(t=t, U=U, omega=omega)
#         evals, evecs = self.diagonalize(H, k=1) # only interested in ground state
#         gs = evecs[:, 0]
#         S = self.entanglement_entropy_fermions_photons(gs)
#         entropies.append(S)
#     return np.array(entropies)

# def sweep_entanglement_entropy_vs_omega(self, omega_list: np.ndarray, t: float, U: float) -> np.ndarray:
#     entropies = []
#     for omega in omega_list:
#         H = self.base_hamiltonian.build_hamiltonian_matrix(t=t, U=U, omega=omega)
#         evals, evecs = self.diagonalize(H, k=1) # only interested in ground state
#         gs = evecs[:, 0]
#         S = self.entanglement_entropy_fermions_photons(gs)
#         entropies.append(S)
#     return np.array(entropies)

# def sweep_ground_state_vs_U(self, U_list: np.ndarray, t: float, omega: float) -> np.ndarray:
#     ground_states = []
#     for U in U_list:
#         H = self.base_hamiltonian.build_hamiltonian_matrix(t=t, U=U, omega=omega)
#         evals, evecs = self.diagonalize(H, k=1) # only interested in ground state
#         gs = evecs[:, 0]
#         ground_states.append(gs)
#     return np.array(ground_states)

# if __name__ == "__main__":
#     L = 12
#     N_f = L // 2 # half-filling
#     N_ph = 10 
#     t = 1.0
#     U = 3.0 * t
#     omega = 10.0 * t
#     # omega = 0.0 * t
#     # g = 0.0
#     g = 0.5
    
#     omega_list = np.linspace(1, 100 + 1, 100) * t
#     #entropies_omega = analyzer.sweep_entanglement_entropy_vs_omega(omega_list, t=t, U=U)
#     # plot_entanglement_entropy_vs_omega_log_log(omega_list, entropies_omega)
#     # photon_numbers_omega = analyzer.sweep_photon_number_vs_omega(omega_list, t=t, U=U)
#     # # plot_photon_number_vs_omega(omega_list, photon_numbers)
#     # plot_photon_number_vs_omega_log_log(omega_list, photon_numbers_omega)
    
#     U_list = np.linspace(0, 20, 100) * t
#     # entropies_U = analyzer.sweep_entanglement_entropy_vs_U(U_list, t=t, omega=omega)
#     # plot_entanglement_entropy_vs_U(U_list, entropies_U)
#     # photon_numbers_U = analyzer.sweep_photon_number_vs_U(U_list, t=t, omega=omega)
#     # plot_photon_number_vs_U(U_list, photon_numbers_U)

#     U_list = np.linspace(0, 3, 20) * t
#     # ground_states = analyzer.sweep_ground_state_vs_U(U_list, t=t, omega=omega)
#     # plot_ground_state_amplitudes_vs_U(U_list, ground_states)


