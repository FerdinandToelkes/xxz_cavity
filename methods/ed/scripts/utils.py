import logging
import math

from omegaconf import DictConfig, OmegaConf

def log_config(logger: logging.Logger, cfg: DictConfig) -> None:
    """Log the configuration in YAML format."""
    logger.info(OmegaConf.to_yaml(OmegaConf.to_container(cfg, resolve=True)))

def register_hydra_resolvers() -> None:
    """Register custom Hydra resolvers."""
    OmegaConf.register_new_resolver("half_filling", lambda L: L // 2)
    OmegaConf.register_new_resolver("gscale", lambda g0, L: g0 / math.sqrt(L))

def get_name_from_parameters(system_params: dict, params: dict, param_name: str) -> str:
    """
    Generate a string representation of the system parameters excluding the sweep parameter.
    Exclude 'g' from the name as it can have many decimal places making the name unwieldy and 
    it can be inferred from other parameters anyways.
    Arguments:
        system_params (dict): The system parameters from the config.
        params (dict): The sweep parameters from the config.
        param_name (str): The name of the parameter being swept.
    Returns:
        str: A string representation of the parameters for naming.
    """
    parameters_as_name = ""
    for key, value in system_params.items():
        if not key.startswith(param_name) and key != "g":
            parameters_as_name += f"{key}={value}_"
    for key, value in params.items():
        if not key.startswith(param_name) and key != "g":
            parameters_as_name += f"{key}={value}_"
    # remove trailing underscore
    parameters_as_name = parameters_as_name.rstrip("_")
    return parameters_as_name

def save_relevant_config(config_filename: str, cfg: DictConfig, observable: str, 
                         system_params: DictConfig, params: DictConfig) -> None:
    """
    Save relevant parts of the configuration to a YAML file such that the sweep can be reproduced.
    Arguments:
        config_filename (str): The path to save the configuration file.
        cfg (DictConfig): The full configuration.
        observable (str): The observable being swept.
        system_params (DictConfig): The system parameters.
        params (DictConfig): The sweep parameters.
    """
    # ensure that the DictConfig is converted to a regular dict for saving
    resolved_system_params = OmegaConf.to_container(system_params, resolve=True)
    resolved_params = OmegaConf.to_container(params, resolve=True)
    relevant = {
        "observable": observable,
        "system": resolved_system_params,
        "params": resolved_params,
        "seed": cfg.seed
    }
    OmegaConf.save(
        OmegaConf.create(relevant),
        config_filename
    )
    