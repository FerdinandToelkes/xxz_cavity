import numpy as np
import hydra
import logging
import h5py
import os

from omegaconf import DictConfig, OmegaConf


from src.exact_diagonalization.basis import Basis
from src.exact_diagonalization.hamiltonian_builder import HamiltonianBuilder
from src.exact_diagonalization.analyzer import Analyzer

logger = logging.getLogger(__name__)



def sweep_parameter(builder: HamiltonianBuilder, param_name: str, param_values: np.ndarray, fixed_params: DictConfig) -> np.ndarray:
    results = []
    # extract parameters from fixed_params to a dictionary
    params = dict(fixed_params)
    for val in param_values:
        params[param_name] = val
        H = builder.build_hamiltonian_matrix(t=params["t"], U=params["U"], omega=params["omega"])
        analyzer = Analyzer(H, builder.basis)
        gs = analyzer.ground_state()
        result = analyzer.expectation_value(gs, analyzer.photon_number_matrix)
        results.append(result)
    return np.array(results)

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(OmegaConf.to_container(cfg, resolve=True)))
    
    # root_data_dir = os.path.expanduser(cfg.root_data_dir)
    root_data_dir = cfg.root_data_dir
    # the system parameters are independent of the sweep parameters
    system_params = cfg.system
    if system_params.N_f is None:
        system_params.N_f = system_params.L // 2
    basis = Basis(system_params.L, system_params.N_f, system_params.N_ph)


    # check for all sweeps defined in the config
    # list all fields in cfg
    sweeps = list(cfg.keys())
    sweeps = [str(s) for s in sweeps if str(s).endswith("_sweep")]
    logger.info(f"Detected sweeps: {sweeps}")
    
    observables_per_sweep = {}
    for sweep in sweeps:
        sweep_cfg = cfg[sweep]
        observables_per_sweep[sweep] = list(sweep_cfg.keys())
    # params = cfg.method.system.parameters
    logger.info(f"Observables per sweep: {observables_per_sweep}")

    for sweep in sweeps:
        for observable in observables_per_sweep[sweep]:
            logger.info(f"Running sweep: {sweep}, observable: {observable}")
            sweep_cfg = cfg[sweep]
            params = sweep_cfg[observable].parameters
            builder = HamiltonianBuilder(basis, g=params.g, boundary_conditions=system_params.boundary_conditions)

            param_name = sweep.split("_sweep")[0]
            param_values = np.linspace(params[f"{param_name}_min"], params[f"{param_name}_max"], params[f"{param_name}_points"])
            results = sweep_parameter(builder, param_name, param_values, params)

            # 
            
            # save results to hdf5 with description attributes
            filename = os.path.join(root_data_dir, f"ed_{sweep}_{observable}.h5")
            if not os.path.exists(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with h5py.File(filename, "w") as f:
                f.create_dataset("param_values", data=param_values)
                f.create_dataset("results", data=results)
                f.attrs["sweep"] = sweep
                f.attrs["observable"] = observable
                f.attrs["param_name"] = param_name
                f.attrs["description"] = f"Sweep of {param_name} for observable {observable} using exact diagonalization."
            logger.info(f"Saved results to {filename}")
            
    return
    
    
    builder = HamiltonianBuilder(basis, g=params.g, boundary_conditions=params.boundary_conditions)
    param_name = 'U'
    param_values = np.linspace(0, 20, 100) * params.t
    results = sweep_parameter(builder, param_name, param_values, params)

    # plot results
    import matplotlib.pyplot as plt
    plt.plot(param_values / params.t, results)
    plt.xlabel(f"{param_name} / t")
    plt.ylabel("Photon Number Expectation Value")
    # plot y axis as 10^-4 multiples
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))
    plt.title("Photon Number vs " + param_name)
    plt.grid()
    plt.show()
    
    

if __name__ == "__main__":
    main()
    exit()


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


