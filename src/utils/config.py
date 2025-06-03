"""Configuration utilities for loading and merging YAML configurations."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        DictConfig: Loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return OmegaConf.create(config_dict)

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")


def merge_configs(*configs: DictConfig) -> DictConfig:
    """
    Merge multiple configurations with later configs overriding earlier ones.

    Args:
        *configs: Variable number of DictConfig objects to merge

    Returns:
        DictConfig: Merged configuration
    """
    if not configs:
        return OmegaConf.create({})

    merged = configs[0]
    for config in configs[1:]:
        merged = OmegaConf.merge(merged, config)

    return merged


def load_and_merge_configs(
    base_config_path: Union[str, Path],
    override_configs: Optional[Dict[str, Union[str, Path]]] = None,
) -> DictConfig:
    """
    Load a base configuration and merge with override configurations.

    Args:
        base_config_path: Path to the base configuration file
        override_configs: Dictionary of config names to paths for overrides

    Returns:
        DictConfig: Merged configuration
    """
    base_config = load_config(base_config_path)

    if not override_configs:
        return base_config

    configs_to_merge = [base_config]

    for config_name, config_path in override_configs.items():
        try:
            override_config = load_config(config_path)
            configs_to_merge.append(override_config)
        except FileNotFoundError:
            print(f"Warning: Override config {config_name} not found at {config_path}")

    return merge_configs(*configs_to_merge)


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """
    Save a configuration to a YAML file.

    Args:
        config: Configuration to save
        output_path: Path where to save the configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(OmegaConf.to_yaml(config), f, default_flow_style=False)


def get_config_from_env(
    env_var: str, default_path: Optional[str] = None
) -> Optional[str]:
    """
    Get configuration path from environment variable.

    Args:
        env_var: Environment variable name
        default_path: Default path if environment variable is not set

    Returns:
        Configuration path or None
    """
    return os.getenv(env_var, default_path)


def validate_config(config: DictConfig, required_keys: list[str]) -> None:
    """
    Validate that required keys exist in the configuration.

    Args:
        config: Configuration to validate
        required_keys: List of required keys (supports nested keys with dot notation)

    Raises:
        KeyError: If required key is missing
    """
    for key in required_keys:
        value = OmegaConf.select(config, key)
        if value is None:
            raise KeyError(f"Required configuration key missing: {key}")
