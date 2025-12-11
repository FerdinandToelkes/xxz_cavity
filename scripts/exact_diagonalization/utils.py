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

def save_relevant_config(config_filename: str, cfg: DictConfig, observable: str, system_params: dict, params: dict) -> None:
    """
    Save relevant parts of the configuration to a YAML file such that the sweep can be reproduced.
    Arguments:
        config_filename (str): The path to save the configuration file.
        cfg (DictConfig): The full configuration.
        observable (str): The observable being swept.
        system_params (dict): The system parameters.
        params (dict): The sweep parameters.
    """
    relevant = {
        "observable": observable,
        "system": system_params,
        "params": params,
        "seed": cfg.seed
    }
    OmegaConf.save(
        OmegaConf.create(relevant),
        config_filename
    )
    