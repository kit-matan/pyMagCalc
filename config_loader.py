#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Loading Utility for MagCalc.

This module provides functions to load and validate the spin model
configuration from a YAML file.
"""
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_spin_model_config(filepath: str) -> Dict[str, Any]:
    """
    Loads and performs basic validation of the spin model configuration
    from a YAML file.

    Args:
        filepath (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration as a Python dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If there's an error parsing the YAML or if essential
                    sections are missing from the configuration.
    """
    logger.info(f"Loading spin model configuration from: {filepath}")
    try:
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {filepath}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {filepath}: {e}")
        raise ValueError(f"Invalid YAML format in {filepath}") from e

    # Basic validation for essential sections
    required_sections = ["crystal_structure", "interactions", "parameters"]
    for section in required_sections:
        if section not in config:
            msg = f"Missing required section '{section}' in configuration file: {filepath}"
            logger.error(msg)
            raise ValueError(msg)

    logger.info("Spin model configuration loaded and basic validation passed.")
    return config
