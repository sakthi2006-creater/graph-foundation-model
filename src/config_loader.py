"""
Configuration loading and management from YAML files.

Handles loading, parsing, and validating config.yaml with support for
environment variable overrides and CLI argument overrides.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger


class ConfigLoader:
    """Load and manage YAML configuration with override support."""

    def __init__(self, config_path: str | Path = "config.yaml"):
        """
        Initialize config loader.

        Args:
            config_path: Path to config.yaml file

        Raises:
            FileNotFoundError: If config file does not exist
            yaml.YAMLError: If config file is invalid YAML
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.config = self._load_yaml()
        logger.info(f"Loaded config from {self.config_path}")

    def _load_yaml(self) -> Dict[str, Any]:
        """
        Load YAML configuration file.

        Returns:
            dict: Parsed YAML configuration

        Raises:
            yaml.YAMLError: If YAML parsing fails
        """
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
                if config is None:
                    raise yaml.YAMLError("Config file is empty")
                return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by dot-separated key.

        Args:
            key: Dot-separated key (e.g., 'model.foundation.hidden_dim')
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config.get('model.foundation.hidden_dim', 128)
            128
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def set(self, key: str, value: Any) -> None:
        """
        Set config value by dot-separated key.

        Args:
            key: Dot-separated key (e.g., 'model.foundation.hidden_dim')
            value: Value to set

        Example:
            >>> config.set('model.foundation.hidden_dim', 256)
        """
        keys = key.split(".")
        current = self.config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def override_from_env(self, prefix: str = "GRAPH_") -> None:
        """
        Override config values from environment variables.

        Environment variables must start with prefix and use underscores
        to represent nested keys (e.g., GRAPH_MODEL_FOUNDATION_HIDDEN_DIM=256).

        Args:
            prefix: Environment variable prefix
        """
        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                # Remove prefix and convert to config key
                config_key = env_key[len(prefix) :].lower()
                config_key = config_key.replace("_", ".")

                # Try to parse as number
                try:
                    if "." in env_value:
                        value = float(env_value)
                    else:
                        value = int(env_value)
                except ValueError:
                    # Keep as string
                    if env_value.lower() in ("true", "false"):
                        value = env_value.lower() == "true"
                    else:
                        value = env_value

                self.set(config_key, value)
                logger.info(f"Override from env: {config_key} = {value}")

    def override_from_args(self, args: Optional[argparse.Namespace] = None) -> None:
        """
        Override config values from argparse arguments.

        Args:
            args: argparse.Namespace with CLI arguments
        """
        if args is None:
            return

        for key, value in vars(args).items():
            if value is not None and key not in ("config",):
                # Convert key: learning_rate -> model.learning_rate or training.learning_rate
                # For now, just set it directly with dots
                self.set(key, value)
                logger.info(f"Override from args: {key} = {value}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def __getitem__(self, key: str) -> Any:
        """Enable dict-like access: config['model.foundation.hidden_dim']"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Enable dict-like setting: config['model.foundation.hidden_dim'] = 256"""
        self.set(key, value)


def load_config(
    config_path: str = "config.yaml",
    args: Optional[argparse.Namespace] = None,
    env_prefix: str = "GRAPH_",
) -> ConfigLoader:
    """
    Load configuration with all overrides applied.

    Priority (lowest to highest):
    1. config.yaml defaults
    2. Environment variables (GRAPH_* prefix)
    3. CLI arguments

    Args:
        config_path: Path to config.yaml
        args: argparse.Namespace with CLI arguments
        env_prefix: Environment variable prefix

    Returns:
        ConfigLoader instance

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If config file invalid
    """
    config = ConfigLoader(config_path)
    config.override_from_env(env_prefix)
    config.override_from_args(args)
    return config


def create_config_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for config overrides.

    Returns:
        argparse.ArgumentParser configured with common training arguments
    """
    parser = argparse.ArgumentParser(
        description="Graph Foundation Model - Config Overrides",
        add_help=True,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )

    # Device & reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed (default: from config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device: auto, cuda, cpu (default: from config)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning_rate",
        "--lr",
        type=float,
        dest="learning_rate",
        help="Learning rate (default: from config)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size (default: from config)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs (default: from config)",
    )

    # Model
    parser.add_argument(
        "--hidden_dim",
        type=int,
        help="Hidden dimension (default: from config)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        help="Number of layers (default: from config)",
    )

    # Logging
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )

    return parser
